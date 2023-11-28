#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Functions

"""

import numpy as np
from tqdm import tqdm
from scipy import spatial
from sklearn.model_selection import KFold
from pyclustering.cluster import kmeans, xmeans, agglomerative, elbow, silhouette
from pyclustering.cluster.agglomerative import type_link
from pyclustering.utils.metric import distance_metric, type_metric
from functions.data_utils.extract_peaks_maps import initialize_cluster_centers, generate_maps_and_peaks
from keras.layers import Conv1D, Flatten, Dense, Reshape, Input
from keras.models import Model
from sklearn.decomposition import PCA
from gui.progress_dialog import ProgressDialog


class MicrostateClusterer:

    def __init__(self, n_inits=10, max_iter=500, tolerance=1e-6):
        self.number_of_repeats = n_inits
        self.max_iterations = max_iter
        self.clustering_tolerance = tolerance
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
    def calculate_spatial_similarity(metric, point1, point2):
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
        modified_kmeans_results = {}

        for init in range(n_inits):
            if verbose:
                print(f'\nClustering #{init + 1} of {n_inits}')

            initial_maps = initialize_cluster_centers(maps2use, n_states, initializer)
            maps, residual = self.modified_kmeans(maps2use, initial_maps, n_states, max_iter, thresh, verbose=verbose)
            gev = self.compute_gev(all_data, maps)

            # Store the results for this initialization in the dictionary
            modified_kmeans_results[init] = {
                'maps': maps,
                'gev': gev,
                'residual': residual
            }

            if verbose:
                print(f'Found {n_states} Microstate Maps')
                print(f'GEV: {gev}')

            # Update the best results if current gev is higher
            if gev > best_gev:
                best_residual, best_gev, best_maps = residual, gev, maps

        modified_kmeans_results['best'] = {
            'maps': best_maps,
            'gev': best_gev,
            'residual': best_residual
        }

        if verbose:
            print(f'\nBest GEV: {best_gev}')
        return modified_kmeans_results

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
                        n_pca, method, n_states, initializer, use_percentages,
                        min_dist, clustering_option, optimizer_mode='gev', parameter_value=5, kmin=2, kmax=10):
        """
        Performs clustering on preprocessed data to find microstate maps.

        Parameters:
            preprocessed_data_path: Path to the preprocessed data.
            hdf_concatenated_data_path: Path to the HDF concatenated data
            n_pca: Number of principal components for the PCA method
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

        # Create an instance of the progress dialog
        progress_dialog = ProgressDialog()
        progress_dialog.set_window_title("Clustering ...")
        progress_dialog.show()

        def metric_function(point1, point2):
            return self.calculate_spatial_similarity(clustering_option, point1, point2)

        maps2use, peaks2use = generate_maps_and_peaks(preprocessed_data_path,
                                                      extension,
                                                      datatype,
                                                      use_percentages,
                                                      min_dist
                                                      )
        """
        if n_states == 'auto':
            from functions.clustering_utils.clusterer_optimizer import ClustererOptimizer
            self.clusterer_optimizer = ClustererOptimizer(maps2use,
                                                          min_dist,
                                                          self.number_of_repeats,
                                                          kmin,
                                                          kmax,
                                                          preprocessed_data_path,
                                                          extension,
                                                          datatype,
                                                          self.clustering_tolerance,
                                                          self.max_iterations
                                                          )
            self.optimal_k, self.k_values, self.target_values = self.clusterer_optimizer.find_optimal_k(optimizer_mode,
                                                                                                        parameter_value)
            n_states = self.optimal_k
            print(f'result: n_states = {n_states}')
        """

        if method == 'Modified K-Means Clustering':
            modified_kmeans_results = self.run_modified_kmeans(
                preprocessed_data_path, extension, datatype,
                maps2use=maps2use,
                n_states=n_states,
                n_inits=self.number_of_repeats,
                initializer=initializer,
                max_iter=self.max_iterations,
                thresh=self.clustering_tolerance
            )
            best_maps = modified_kmeans_results['best']['maps']
            best_gev = modified_kmeans_results['best']['gev']
            best_residual = modified_kmeans_results['best']['residual']

        else:

            initial_centers = initialize_cluster_centers(maps2use, n_states, initializer)
            maps2use = np.transpose(maps2use)

            if method == 'K-Means Clustering':
                metric = distance_metric(type_metric.USER_DEFINED, func=metric_function)
                clustering_instance = kmeans.kmeans(maps2use, initial_centers,
                                             tolerance=self.clustering_tolerance, itermax=self.max_iterations,
                                             metric=metric)

            elif method == 'PCA + K-Means Clustering':
                encoded_features = self.extract_features_with_pca(maps2use, pca_components=n_pca)
                initial_centers = initialize_cluster_centers(np.transpose(encoded_features), n_states, initializer)
                metric = distance_metric(type_metric.USER_DEFINED, func=metric_function)
                #metric = distance_metric(type_metric.EUCLIDEAN)
                clustering_instance = kmeans.kmeans(encoded_features, initial_centers,
                                                    tolerance=self.clustering_tolerance, itermax=self.max_iterations,
                                                    metric=metric)

            elif method == 'Autoencoder + K-Means Clustering':
                encoded_features, autoencoder = self.extract_features_with_autoencoder(maps2use, encoding_dim=10)
                initial_centers = initialize_cluster_centers(np.transpose(encoded_features), n_states, initializer)
                metric = distance_metric(type_metric.USER_DEFINED, func=metric_function)
                clustering_instance = kmeans.kmeans(encoded_features, initial_centers,
                                                    tolerance=self.clustering_tolerance, itermax=self.max_iterations,
                                                    metric=metric)

            elif method == 'X-Means Clustering':
                if clustering_option == 'Bayesian Information Criterion':
                    CRITERION = xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION
                elif clustering_option == 'Minimum Noiseless Description Length':
                    CRITERION = xmeans.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH
                else:
                    raise ValueError("Failed to match metric")
                clustering_instance = xmeans.xmeans(maps2use, initial_centers, n_states,
                                     tolerance=self.clustering_tolerance, criterion=CRITERION)

            elif method == 'Agglomerative Hierarchical Clustering':
                encoded_features = self.extract_features_with_pca(maps2use, pca_components=n_pca)
                clustering_instance = agglomerative.agglomerative(encoded_features, n_states,
                                                                  agglomerative.type_link.SINGLE_LINK, ccore=True)

            else:
                raise ValueError("Failed to match method")


            best_gev = 0
            for init in range(self.number_of_repeats):
                progress_dialog.set_label_text(f"Clustering {int(n_states)} Microstate Maps\nInitialization #{init + 1} of {self.number_of_repeats}")
                print('\nClustering #', str(init + 1), 'of', str(self.number_of_repeats))

                if method in ['K-Means Clustering', 'X-Means Clustering']:
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
                    clustering_instance.process()
                    cluster_labels = clustering_instance.get_clusters(maps2use)
                    residual = 0#clustering_instance.get_total_wce()
                    # Flatten the cluster labels
                    cluster_labels_flat = np.zeros(len(maps2use))
                    for cluster_id, cluster in enumerate(cluster_labels):
                        cluster_labels_flat[cluster] = cluster_id
                    # Find original centroids
                    centroids = self.find_original_centroids(maps2use, cluster_labels_flat, n_states)


                GEV_R = self.compute_gev(np.transpose(maps2use), np.array(centroids))

                progress_dialog.update_progress(init + 1, self.number_of_repeats)
                progress_dialog.set_line_edit_text(f"Global Explained Variance: {GEV_R}")
                print('Found', str(int(n_states)), 'Microstate Maps')
                print('GEV:', str(GEV_R))
                if GEV_R > best_gev:
                    best_gev = GEV_R
                    best_maps = centroids
                    best_residual = residual

            print('\nBest GEV:', str(best_gev))
            progress_dialog.close()

        return best_maps, best_gev, best_residual

