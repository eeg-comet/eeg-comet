#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Functions

"""

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import spatial
from sklearn.model_selection import KFold
from pyclustering.cluster import kmeans, xmeans, agglomerative, elbow, silhouette
from pyclustering.utils.metric import distance_metric, type_metric
from functions.data_utils.extract_peaks_maps import initialize_cluster_centers, generate_maps_and_peaks

from tensorflow.keras.layers import Conv1D, Flatten, Dense, Reshape, Input
from keras.models import Model
from sklearn.decomposition import PCA


class MicrostateClusterer:

    def __init__(self, n_inits=10, max_iter=500, tolerance=1e-6):
        self.n_inits = n_inits
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.best_maps = None
        self.best_gev = 0
        self.best_residual = None

    def corr_vectors(self, A, B, axis=0):
        """Computes the Pearson correlation between two matrices A and B along a specified axis."""
        if A.shape != B.shape:
            raise ValueError("Both matrices A and B must have the same shape.")

        # Center and normalize matrices
        An = A - np.mean(A, axis=axis, keepdims=True)
        An /= np.linalg.norm(An, axis=axis, keepdims=True)
        Bn = B - np.mean(B, axis=axis, keepdims=True)
        Bn /= np.linalg.norm(Bn, axis=axis, keepdims=True)

        return np.sum(An * Bn, axis=axis)

    def compute_gev(self, data, maps):
        """Calculates the global explained variance (GEV) of microstate maps based on input data."""

        gfp = np.std(data, axis=0)  # Global Field Power

        # Normalize maps
        if maps.ndim == 1:
            maps = maps / np.linalg.norm(maps)
            maps = np.reshape(maps, (1, -1))
        else:
            maps = maps / np.linalg.norm(maps, axis=1, keepdims=True)

        # Compute activation and segmentation
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        selected_maps = maps[segmentation, :]
        map_corr = self.corr_vectors(data, selected_maps.T)

        return np.sum((gfp * map_corr) ** 2) / np.sum(gfp ** 2)

    @staticmethod
    def calculate_spatial_similarity(self, metric, point1, point2):
        """Computes similarity between two points using cosine similarity or spatial correlation."""

        if metric == 'Cosine Similarity':
            # Calculates the cosine similarity
            dist = spatial.distance.cosine(point1, point2)
        elif metric == 'Spatial Correlation':
            # Calculates the spatial correlation
            dist = spatial.distance.correlation(point1, point2)
        else:
            raise ValueError("Failed to match metric")
        return 1 - dist

    def modified_kmeans(self, data, initial_maps, n_states, max_iter=500, thresh=1e-6, verbose=True):
        """Performs modified K-Means clustering on data with a specified number of microstate maps."""
        # Initial setup
        n_channels, n_samples = data.shape
        maps = initial_maps.copy()
        data_sum_sq = np.sum(data ** 2)
        prev_residual = np.inf

        # Clustering iterations
        for iteration in range(max_iter):
            # Assign each sample to the best matching microstate
            activation = maps.dot(data)
            segmentation = np.argmax(np.abs(activation), axis=0)

            for state in range(n_states):
                idx = (segmentation == state)
                maps[state] = np.dot(data[:, idx], activation[state, idx])
                maps[state] /= np.linalg.norm(maps[state])

            # Estimate residual noise
            act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
            residual = abs(data_sum_sq - act_sum_sq) / float(n_samples * (n_channels - 1))

            # Check for convergence
            if (prev_residual - residual) < (thresh * residual):
                if verbose:
                    print('Converged at', iteration, 'iterations.')
                break

            prev_residual = residual

        return maps, prev_residual

    def run_modified_kmeans(self, preprocessed_data_path, extension, datatype,
                            maps2use, n_states, n_inits, initializer='Random', max_iter=500, thresh=1e-6, verbose=True):
        """Runs modified K-Means clustering with multiple initializations to find the best microstate maps."""

        all_data, _ = generate_maps_and_peaks(preprocessed_data_path, extension, datatype, use_percentages=100)
        best_residual, best_gev, best_maps = None, 0, None

        for init in range(n_inits):
            if verbose:
                print(f'\nClustering #{init + 1} of {n_inits}')

            initial_maps = initialize_cluster_centers(maps2use, n_states, initializer)
            maps, residual = self.modified_kmeans(maps2use, initial_maps, n_states, max_iter, thresh, verbose=verbose)
            gev = self.compute_gev(all_data, maps)

            if verbose:
                print(f'Found {n_states} Microstate Maps')
                print(f'GEV: {gev}')

            # Update the best results if current gev is higher
            if gev > best_gev:
                best_residual, best_gev, best_maps = residual, gev, maps

        if verbose:
            print(f'\nBest GEV: {best_gev}')
        return best_maps, best_gev, best_residual

    def create_eeg_autoencoder(self, input_shape, encoding_dim):
        """Creates an autoencoder model for EEG data compression and reconstruction."""
        # Encoder
        input_layer = Input(shape=input_shape, name='input')
        x = Flatten()(input_layer)
        encoded = Dense(encoding_dim, activation='relu', name='embedding')(x)

        # Decoder
        x = Dense(np.prod(input_shape), activation='relu')(encoded)
        decoded = Reshape(input_shape)(x)

        autoencoder = Model(input_layer, decoded, name='autoencoder')
        encoder = Model(input_layer, encoded, name='encoder')

        # Compile the autoencoder (you can change the optimizer and loss function if needed)
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder, encoder

    def find_original_centroids(self, eeg_data, cluster_labels, n_clusters):
        """Determines the original centroids of clustered data points."""

        original_centroids = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_data = eeg_data[cluster_indices]
            cluster_mean = np.mean(cluster_data, axis=0)
            original_centroids.append(cluster_mean)
        original_centroids = np.array(original_centroids)

        return original_centroids

    def calculate_residuals(self, eeg_data, autoencoder):
        # Encode and then decode the data to get the reconstructed data
        reconstructed_data = autoencoder.predict(eeg_data)

        # Calculate residuals (reconstruction errors) for each data point
        residuals = np.mean(np.abs(eeg_data - reconstructed_data), axis=1)

        return residuals

    def extract_features_with_autoencoder(self, eeg_data, encoding_dim=10):
        """Extracts features from EEG data using an autoencoder."""
        # Create the autoencoder
        input_shape = eeg_data.shape[1:]
        autoencoder, encoder = self.create_eeg_autoencoder(input_shape, encoding_dim)

        # Train the autoencoder on EEG data
        autoencoder.fit(eeg_data, eeg_data, epochs=10, batch_size=64, shuffle=True)

        # Extract features using the encoder
        encoded_features = encoder.predict(eeg_data)

        return encoded_features, autoencoder

    def extract_features_with_pca(self, eeg_data, pca_components=10):
        """Reduces dimensionality of EEG data using Principal Component Analysis (PCA)."""
        # Perform PCA to reduce dimensionality
        pca = PCA(n_components=pca_components)
        reduced_data = pca.fit_transform(eeg_data)

        return reduced_data

    def clustering_func(self, preprocessed_data_path, extension, datatype,
                        n_channels, method, n_states, initializer, use_percentages,
                        min_dist, clustering_option, optimizer_mode='gev', parameter_value=5, kmin=2, kmax=10):
        """
        Performs clustering on preprocessed data to find microstate maps.

        Parameters:
            preprocessed_data_path: Path to the preprocessed data.
            hdf_concatenated_data_path: Path to the HDF concatenated data
            n_channels: Number of channels in the data
            method: Clustering method to use
            n_states: Number of microstate maps
            initializer: Initialization method
            min_dist: Minimum distance
            n_inits: Number of initializations
            tolerance: Convergence threshold
            clustering_option: Distance metric to use for clustering

        Returns:
            best_maps: Best cluster centers (microstate maps)
            best_gev: Best global explained variance
            best_residual: Best residual error
            n_states: Optimal number of clusters (microstate maps)
        """

        def metric_function(point1, point2):
            return self.calculate_spatial_similarity(clustering_option, point1, point2)

        maps2use, peaks2use = generate_maps_and_peaks(preprocessed_data_path, extension, datatype,
                                                      use_percentages, min_dist)

        elbow_optimizer = ClusterOptimizer(maps2use, min_dist, self.n_inits, kmin, kmax, preprocessed_data_path, extension, datatype, self.tolerance, self.max_iter)
        if n_states == 'auto':
            n_states = elbow_optimizer.find_optimal_k(optimizer_mode, parameter_value)
            print(f'result: n_states = {n_states}')

        if method == 'Modified K-Means Clustering':
            best_maps, best_gev, best_residual = self.run_modified_kmeans(
                preprocessed_data_path, extension, datatype,
                maps2use=maps2use,
                n_states=n_states,
                n_inits=self.n_inits,
                initializer=initializer,
                max_iter=self.max_iter,
                thresh=self.tolerance
            )

        else:

            initial_centers = initialize_cluster_centers(maps2use, n_states, initializer)
            maps2use = np.transpose(maps2use)

            if method == 'K-Means Clustering':
                metric = distance_metric(type_metric.USER_DEFINED, func=metric_function)
                clustering_instance = kmeans.kmeans(maps2use, initial_centers,
                                             tolerance=self.tolerance, itermax=self.max_iter,
                                             metric=metric)

            elif method == 'PCA + K-Means Clustering':
                encoded_features = self.extract_features_with_pca(maps2use, pca_components=10)
                initial_centers = initialize_cluster_centers(np.transpose(encoded_features), n_states, initializer)
                metric = distance_metric(type_metric.USER_DEFINED, func=metric_function)
                #metric = distance_metric(type_metric.EUCLIDEAN)
                clustering_instance = kmeans.kmeans(encoded_features, initial_centers,
                                                    tolerance=self.tolerance, itermax=self.max_iter,
                                                    metric=metric)

            elif method == 'Autoencoder + K-Means Clustering':
                encoded_features, autoencoder = self.extract_features_with_autoencoder(maps2use, encoding_dim=10)
                initial_centers = initialize_cluster_centers(np.transpose(encoded_features), n_states, initializer)
                metric = distance_metric(type_metric.USER_DEFINED, func=metric_function)
                clustering_instance = kmeans.kmeans(encoded_features, initial_centers,
                                                    tolerance=self.tolerance, itermax=self.max_iter,
                                                    metric=metric)

            elif method == 'X-Means Clustering':
                if clustering_option == 'Bayesian Information Criterion':
                    CRITERION = xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION
                elif clustering_option == 'Minimum Noiseless Description Length':
                    CRITERION = xmeans.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH
                else:
                    raise ValueError("Failed to match metric")
                clustering_instance = xmeans.xmeans(maps2use, initial_centers, n_states,
                                     tolerance=self.tolerance, criterion=CRITERION)

            elif method == 'Agglomerative Hierarchical Clustering':
                from sklearn.cluster import AgglomerativeClustering
                from sklearn.metrics import pairwise_distances
                def cosine_distance(X, Y=None):
                    return pairwise_distances(X, Y, metric='cosine')

                clustering_instance = AgglomerativeClustering(n_clusters=n_states,
                                                              affinity=metric_function,
                                                              linkage='average')
            else:
                raise ValueError("Failed to match method")

            best_gev = 0
            for init in range(self.n_inits):
                print('\nClustering #', str(init + 1), 'of', str(self.n_inits))

                if method == 'K-Means Clustering':
                    clustering_instance.process()
                    residual = clustering_instance.get_total_wce()
                    centroids = clustering_instance.get_centers()
                elif method in ['PCA + K-Means Clustering', 'Autoencoder + K-Means Clustering']:
                    clustering_instance.process()
                    cluster_labels = clustering_instance.get_clusters()
                    # Flatten the cluster labels
                    cluster_labels_flat = np.zeros(len(maps2use))
                    for cluster_id, cluster in enumerate(cluster_labels):
                        cluster_labels_flat[cluster] = cluster_id
                    # Find original centroids
                    centroids = self.find_original_centroids(maps2use, cluster_labels_flat, n_states)
                    # Calculate residuals
                    residual = 0#self.calculate_residuals(maps2use, autoencoder)
                elif method == 'Agglomerative Hierarchical Clustering':
                    clusters = clustering_instance.fit_predict(maps2use)
                    residual = 0#clustering_instance.get_total_wce()
                    centroids = np.empty((n_states,n_channels))
                    for cl in range(len(clusters)):
                        centroids[cl,:] = np.mean(maps2use[clusters[cl],:],axis=0)


                GEV_R = self.compute_gev(np.transpose(maps2use), np.array(centroids))

                print('Found', str(int(n_states)), 'Microstate Maps')
                print('GEV:', str(GEV_R))
                if GEV_R > best_gev:
                    best_gev = GEV_R
                    best_maps = centroids
                    best_residual = residual

            print('\nBest GEV:', str(best_gev))

        return best_maps, best_gev, best_residual, n_states



class ClusterOptimizer:
    """Finds the optimal number of clusters (K) for clustering EEG data.

        Attributes:
            maps2use (numpy array): The EEG data for clustering.
            min_dist (float): The minimum distance between clusters.
            n_inits (int): Number of initializations for K-Means.
            kmin (int): Minimum number of clusters to evaluate.
            kmax (int): Maximum number of clusters to evaluate.
            tolerance (float, optional): The tolerance for convergence.
            max_iter (int, optional): Maximum iterations for K-Means.
        """
    def __init__(self, maps2use, min_dist, n_inits, kmin, kmax, preprocessed_data_path, extension, datatype, tolerance=None, max_iter=None):
        """Initialize the ClusterOptimizer with given parameters."""
        self.maps2use = maps2use
        self.min_dist = min_dist
        self.tolerance = tolerance
        self.n_inits = n_inits
        self.kmin = kmin
        self.kmax = kmax
        self.max_iter = max_iter
        self.microstate_clusterer = MicrostateClusterer(n_inits=1)
        self.preprocessed_data_path = preprocessed_data_path
        self.extension = extension
        self.datatype = datatype
        self.N, self.RES, self.GEV, self.SIL = [], [], [], []

    def _cluster_and_evaluate(self, k):
        """Cluster the EEG data for a given K and evaluate the quality using various metrics."""
        gev_i, residual_i = 0, 0
        for init in range(self.n_inits):
            maps, gev, residual = self.microstate_clusterer.run_modified_kmeans(
                preprocessed_data_path=self.preprocessed_data_path,
                extension=self.extension,
                datatype=self.datatype,
                maps2use=self.maps2use,
                n_states=k,
                n_inits=1,
                initializer="Random",
                max_iter=self.max_iter,
                thresh=self.tolerance,
                verbose=False
            )
            gev_i += gev
            residual_i += residual
            # Compute the Silhouette score
            activation = np.array(maps).dot(self.maps2use)
            classes = np.argmax(np.abs(activation), axis=0)
            sil_score_i = np.mean(abs(self.microstate_clusterer.corr_vectors(self.maps2use, maps[classes].T)))

        return gev_i / self.n_inits, residual_i / self.n_inits, sil_score_i / self.n_inits

    def find_elbow_with_plot(self, ax1, ax2, ax3):
        """Find the elbow point with plots to visualize the clustering metrics."""
        for k in range(self.kmin, self.kmax + 1):
            print(f'\nClustering data with {k} microstates')
            gev_mean, residual_mean, sil_mean = self._cluster_and_evaluate(k)
            self.N.append(k)
            self.RES.append(residual_mean)
            self.GEV.append(gev_mean)
            self.SIL.append(sil_mean)

        df = pd.DataFrame({
            'K': self.N,
            'Residual': self.RES,
            'Global Explained Variance': self.GEV,
            'Silhouette Score': self.SIL
        })
        self._plot(df, ax1, ax2, ax3)


    def find_optimal_k(self, optimizer_mode='cv', parameter_value=5):
        """Find the elbow point without generating plots."""
        if optimizer_mode == 'gs':
            return self.find_optimal_k_gap_statistic(int(parameter_value))
        elif optimizer_mode == 'cv':
            return self.find_optimal_k_cross_validation(int(parameter_value))
        else:
            parameter_value = parameter_value / 100
            k_range = range(self.kmin, self.kmax + 1)
            progress_bar = tqdm(k_range, desc="Progress", ncols=100, position=0, leave=True)

            for k in progress_bar:
                progress_bar.set_postfix({"k": k})
                print(f'\nClustering data with {k} microstates')
                gev_mean, residual_mean, sil_mean = self._cluster_and_evaluate(k)
                self.N.append(k)
                self.RES.append(residual_mean)
                self.GEV.append(gev_mean)
                self.SIL.append(sil_mean)
                if len(self.N) == 1:
                    continue
                if self._should_stop(optimizer_mode, gev_mean, residual_mean, sil_mean, parameter_value):
                    return k
            return int((self.kmin + self.kmax) / 2)

    def _calculate_wcss(self, data, n_clusters):
        # Initialize variables to store the WCSS
        wcss = 0
        # Run your modified K-means clustering algorithm
        maps, _, _ = self.microstate_clusterer.run_modified_kmeans(
            preprocessed_data_path=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            maps2use=self.maps2use,
            n_states=n_clusters,
            n_inits=1,
            initializer="Random",
            max_iter=self.max_iter,
            thresh=self.tolerance,
            verbose=False
        )
        segmentation = np.argmax(np.abs(maps.dot(data)), axis=0)
        # Calculate WCSS for each cluster
        for cluster_idx in range(n_clusters):
            cluster_points = data[:, segmentation == cluster_idx]
            cluster_center = maps[cluster_idx]

            # Calculate the sum of squares of distances within the cluster
            cluster_sse = np.sum(np.sum((cluster_points - cluster_center.reshape(-1, 1)) ** 2, axis=0))
            wcss += cluster_sse
        return wcss

    def find_optimal_k_gap_statistic(self, n_random_datasets=5):
        print(
            f"\nIdentifying the optimal number of clusters using the gap statistic method with {n_random_datasets}-random datasets")

        # Load or generate your data
        data, _ = generate_maps_and_peaks(self.preprocessed_data_path,
                                          self.extension,
                                          self.datatype,
                                          use_percentages=100
                                          )

        # Number of clusters to consider
        k_values = range(self.kmin, self.kmax + 1)

        # Initialize arrays to store WCSS values
        wcss_real = np.zeros(len(k_values))
        wcss_random = np.zeros((len(k_values), n_random_datasets))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(k_values) * (1 + n_random_datasets), ncols=100, position=0, leave=True)

        for i, k in enumerate(k_values):
            wcss_real[i] = self._calculate_wcss(data, k)
            progress_bar.update(1)

        for j in range(n_random_datasets):
            random_data = np.random.rand(*data.shape)  # Generate random data with the same shape as your data
            for i, k in enumerate(k_values):
                wcss_random[i, j] = self._calculate_wcss(random_data, k)
                progress_bar.update(1)

        # Calculate the expected WCSS for random data
        wcss_random_mean = wcss_random.mean(axis=1)

        # Calculate the gap statistic
        gap = np.log(wcss_random_mean) - np.log(wcss_real)

        # Find the optimal number of clusters (the maximum point of the gap statistic)
        optimal_clusters = np.argmax(gap) + self.kmin

        # Close the progress bar
        progress_bar.close()

        print(f"\nOptimal clusters: {optimal_clusters}")
        return optimal_clusters

    def find_optimal_k_cross_validation(self, n_splits=5):
        print(f"\nIdentifying the optimal number of microstates using {n_splits}-fold cross-validation method")
        data, _ = generate_maps_and_peaks(self.preprocessed_data_path,
                                          self.extension,
                                          self.datatype,
                                          use_percentages=100
                                          )
        # Number of clusters to consider
        k_values = range(self.kmin, self.kmax + 1)
        # Initialize arrays to store cross-validation scores
        cv_scores = []

        # Calculate the total number of updates
        total_updates = len(k_values) * n_splits

        # Create a tqdm progress bar for k_values * n_splits
        combined_progress = tqdm(total=total_updates, desc="Progress", ncols=100, position=0, leave=True)

        for k in k_values:
            wcss = 0
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(data.T):
                train_data, test_data = data[:, train_index], data[:, test_index]
                # Run the modified K-means clustering algorithm
                maps, _, _ = self.microstate_clusterer.run_modified_kmeans(
                    preprocessed_data_path=self.preprocessed_data_path,
                    extension=self.extension,
                    datatype=self.datatype,
                    maps2use=self.maps2use,
                    n_states=k,
                    n_inits=1,
                    initializer="Random",
                    max_iter=self.max_iter,
                    thresh=self.tolerance,
                    verbose=False
                )
                segmentation = np.argmax(np.abs(maps.dot(test_data)), axis=0)
                # Calculate WCSS for each cluster
                for cluster_idx in range(k):
                    cluster_points = test_data[:, segmentation == cluster_idx]
                    cluster_center = maps[cluster_idx]
                    # Calculate the sum of squares of distances within the cluster
                    cluster_sse = np.sum(np.sum((cluster_points - cluster_center.reshape(-1, 1)) ** 2, axis=0))
                    wcss += cluster_sse
                # Update the combined progress bar
                combined_progress.update(1)
            # Calculate the average WCSS over all folds
            avg_wcss = wcss / n_splits
            cv_scores.append(avg_wcss)

        combined_progress.close()

        # Find the optimal number of clusters (k) with the minimum average WCSS
        optimal_clusters = k_values[np.argmin(cv_scores)]
        print(f"\nOptimal clusters: {optimal_clusters}")
        return optimal_clusters

    def _should_stop(self, optimizer_mode, gev_mean, residual_mean, sil_mean, threshold):
        """Decide if the clustering should stop based on the given stopping mode and threshold."""
        if optimizer_mode == 'gev':
            return abs(gev_mean - self.GEV[-2]) / self.GEV[-2] < threshold
        elif optimizer_mode == 'res':
            return abs(residual_mean - self.RES[-2]) / self.RES[-2] < threshold
        elif optimizer_mode == 'sil':
            return abs(sil_mean - self.SIL[-2]) / self.SIL[-2] < threshold
        else:
            raise ValueError("Invalid optimizer_mode. Choose 'gev', 'res', or 'sil'.")

    def _plot(self, df, ax1, ax2, ax3):
        """Generate line plots for the calculated metrics to visualize the elbow point."""
        # Visualize the result
        res_fig = sns.lineplot(data=df, x="K", y="Residual",
                               linewidth=5, marker="o", markersize=16, dashes=False, ax=ax1)
        res_fig.set_xlabel("Number of Microstate Maps", size=16)
        res_fig.set_ylabel("Residual", size=16)

        xlabel_format = '{:,.0f}'
        ticks_loc = ax1.get_xticks().tolist()
        ax1.set_xticks(ax1.get_xticks().tolist())
        ax1.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], size=14)
        ylabel_format = '{:,.2f}'
        ticks_loc = ax1.get_yticks().tolist()
        ax1.set_yticks(ax1.get_yticks().tolist())
        ax1.set_yticklabels([ylabel_format.format(x) for x in ticks_loc], size=14)

        gev_fig = sns.lineplot(data=df, x="K", y="Global Explained Variance",
                               linewidth=5, style=None, marker="o", markersize=16, dashes=False, ax=ax2)
        gev_fig.set_xlabel("Number of Microstate Maps", size=16)
        gev_fig.set_ylabel("Global Explained Variance", size=16)

        xlabel_format = '{:,.0f}'
        ticks_loc = ax2.get_xticks().tolist()
        ax2.set_xticks(ax2.get_xticks().tolist())
        ax2.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], size=14)
        ylabel_format = '{:,.2f}'
        ticks_loc = ax2.get_yticks().tolist()
        ax2.set_yticks(ax2.get_yticks().tolist())
        ax2.set_yticklabels([ylabel_format.format(x) for x in ticks_loc], size=14)

        silhouette_fig = sns.lineplot(data=df, x="K", y="Silhouette Score",
                                      linewidth=5, style=None, marker="o", markersize=16, dashes=False, ax=ax3)
        silhouette_fig.set_xlabel("Number of Microstate Maps", size=16)
        silhouette_fig.set_ylabel("Silhouette Score", size=16)

        xlabel_format = '{:,.0f}'
        ticks_loc = ax3.get_xticks().tolist()
        ax3.set_xticks(ax3.get_xticks().tolist())
        ax3.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], size=14)
        ylabel_format = '{:,.2f}'
        ticks_loc = ax3.get_yticks().tolist()
        ax3.set_yticks(ax3.get_yticks().tolist())
        ax3.set_yticklabels([ylabel_format.format(x) for x in ticks_loc], size=14)

