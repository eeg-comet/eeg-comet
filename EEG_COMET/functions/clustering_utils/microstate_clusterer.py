#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Functions

"""

import numpy as np
import pandas as pd
from scipy import spatial
from pyclustering.cluster import kmeans, xmeans
from pyclustering.utils.metric import distance_metric, type_metric
from keras.layers import Flatten, Dense, Reshape, Input
from keras.models import Model
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from functions.data_utils.data_initializer import DataInitializer


class MicrostateClusterer:

    def __init__(self, n_inits=10, max_iter=500, tolerance=1e-6):
        self.number_of_repeats = n_inits
        self.max_iterations = max_iter
        self.clustering_tolerance = tolerance
        self.best_maps = None

    @staticmethod
    def corr_vectors(array1, array2, axis=0):
        """
        Computes the Pearson correlation between two matrices A and B along a specified axis.
        """
        # Center and normalize matrices
        array1n = array1 - np.mean(array1, axis=axis, keepdims=True)
        array1n /= np.linalg.norm(array1n, axis=axis, ord=2, keepdims=True)
        array2n = array2 - np.mean(array2, axis=axis, keepdims=True)
        array2n /= np.linalg.norm(array2n, axis=axis, ord=2, keepdims=True)
        return np.sum(array1n * array2n, axis=axis)

    def compute_gev(self, data, maps):
        """
        Calculates the global explained variance (GEV) of microstate maps based on input data.
        """
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
    def microstates2csv(microstates, eeg_info, microstate_maps_path, headers=None):
        """
        Export microstate maps to a CSV file.
        """
        # Transpose the microstates array if needed
        if microstates.shape[1] == len(eeg_info['ch_names']):
            microstates = microstates.T

        # Save Best Maps
        microstates = np.array(microstates)
        maps_df = pd.DataFrame(microstates, index=eeg_info['ch_names'])

        if headers is not None:
            maps_df.columns = headers
        else:
            headers = [f'{i + 1}' for i in range(microstates.shape[1])]
            maps_df.columns = headers

        maps_df.to_csv(microstate_maps_path)

    @staticmethod
    def calculate_spatial_similarity(metric, point1, point2):
        """
        Computes similarity between two points using cosine similarity or spatial correlation.
        """
        if metric == 'Cosine Similarity':
            # Calculates the cosine similarity
            dist = spatial.distance.cosine(point1, point2)
        else: # metric == 'Spatial Correlation':
            # Calculates the spatial correlation
            dist = spatial.distance.correlation(point1, point2)
        return 1 - abs(dist)

    @staticmethod
    def modified_kmeans(data, initial_maps, n_states, max_iter=500, thresh=1e-6, verbose=True):
        """
        Performs modified K-Means clustering on data with a specified number of microstate maps.
        """
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
        """
        Runs modified K-Means clustering with multiple initializations to find the best microstate maps.
        """

        data_initializer = DataInitializer()
        all_data, _ = data_initializer.generate_maps_and_peaks(
            preprocessed_data_path, extension, datatype, use_percentages=100
        )
        best_residual, best_gev, best_maps = None, 0, None
        modified_kmeans_results = {}

        for init in range(n_inits):
            if verbose:
                print(f'\nClustering #{init + 1} of {n_inits}')
            initial_maps = data_initializer.initialize_cluster_centers(maps2use, n_states, initializer)
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

    def run_aahc(self, preprocessed_data_path, extension, datatype, maps2use, n_states, n_maps2use=1000, verbose=True):
        """
        Performs AAHC clustering on data with a specified number of microstate maps.
        """
        def select_random_maps_subset(maps2use, n_maps2use):
            # Generate random indices to select maps
            random_indices = np.random.choice(maps2use.shape[1], size=n_maps2use, replace=False)
            # Use the random indices to select maps
            selected_maps = maps2use[:, random_indices]
            return selected_maps

        def process_reassignment(cluster_index_to_reassign, c_i, cluster_data, maps):
            cluster_data_subset = cluster_data[cluster_index_to_reassign, :]
            mapsn = maps - np.mean(maps, axis=1, keepdims=True)
            mapsn /= np.linalg.norm(mapsn, axis=1, ord=2, keepdims=True)
            cluster_data_subsetn = cluster_data_subset - np.mean(cluster_data_subset, axis=0, keepdims=True)
            cluster_data_subsetn /= np.linalg.norm(cluster_data_subsetn, axis=0, ord=2, keepdims=True)
            map_corr = np.sum(mapsn * cluster_data_subset, axis=1)
            new_assignment = np.argmax(np.abs(map_corr), axis=0)
            c_i[new_assignment].append(cluster_index_to_reassign)

        def process_cluster(cluster_index, c_i, cluster_data, maps):
            data_indices = c_i[cluster_index]
            cluster_data_subset = cluster_data[data_indices, :]
            covariance_matrix = np.dot(cluster_data_subset.T, cluster_data_subset)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            principal_component = eigenvectors[:, np.argmax(np.abs(eigenvalues))]
            principal_component = np.real(principal_component)
            reconstructed_data = principal_component / np.sqrt(np.sum(principal_component ** 2))
            # Calculate residual (difference between original and reconstructed data)
            residual = cluster_data_subset - np.dot(reconstructed_data, cluster_data_subset.T).T
            maps[cluster_index, :] = principal_component / np.sqrt(np.sum(principal_component ** 2))
            return residual

        # Initial setup
        # Number of parallel workers (adjust as needed)
        n_jobs = -1  # Use all available cores
        all_data, _ = DataInitializer().generate_maps_and_peaks(
            preprocessed_data_path, extension, datatype, use_percentages=100
        )
        n_channels, n_samples = all_data.shape
        # Get GFP peaks
        gfp = all_data.std(axis=0)
        gfp2 = np.sum(gfp ** 2)
        # Initial number of clusters and Store original GFP peaks and indices
        maps2use = select_random_maps_subset(maps2use, n_maps2use)
        maps = np.transpose(maps2use)
        n_maps = maps.shape[0]
        batch_size = int(n_maps/10)
        cluster_data = maps
        print(f"Initial number of clusters: {n_maps:d}\n")
        # Cluster indices w.r.t. original size, normalized GFP peak data
        c_i = [[k] for k in range(n_maps)]
        # Main loop: atomize + agglomerate
        while n_maps > n_states:
            if verbose:
                print(f"\r\r\t\tAAHC > n: {n_maps:d} => {n_maps - 1:d}", end="")
            # Correlations of the data sequence with each cluster
            # Assuming you initialize 'segmentation' somewhere before the loop
            segmentation = np.zeros(n_samples, dtype=int)
            # Memory-efficient normalization
            maps_norm = np.linalg.norm(maps, axis=1, ord=2, keepdims=True)
            maps /= maps_norm
            # Process data in batches for activation
            activation = np.zeros((n_maps, n_samples))
            for i in range(0, n_samples, batch_size):
                data_batch = all_data[:, i:i + batch_size]
                activation[:, i:i + batch_size] = maps.dot(data_batch)
            # GEV (global explained variance) of cluster k
            gev = np.zeros(n_maps)
            for state in range(n_maps):
                idx = (segmentation == state)
                map_corr = self.corr_vectors(all_data[:, idx], maps[segmentation[idx]].T)
                gev[state] = np.sum((gfp[idx] * map_corr) ** 2) / gfp2
            # Merge cluster with the minimum GEV
            imin = np.argmin(gev)
            # N => N-1
            maps = np.vstack((maps[:imin, :], maps[imin + 1:, :]))
            c_i, re_c = c_i[:imin] + c_i[imin + 1:], c_i[imin]
            re_cluster = []  # indices of updated clusters
            # Parallelize the loop
            Parallel(n_jobs=n_jobs)(delayed(process_reassignment)(cluster_index_to_reassign, c_i, cluster_data, maps) for
                                    cluster_index_to_reassign in re_c)
            n_maps = len(c_i)
            # Update clusters
            re_cluster = list(set(re_cluster))  # unique list of updated clusters
            # Parallelize the loop
            residuals = Parallel(n_jobs=n_jobs)(
                delayed(process_cluster)(cluster_index, c_i, cluster_data, maps) for cluster_index in re_cluster)
        best_gev = self.compute_gev(all_data, maps)
        print(f'\nBest GEV: {best_gev}')
        return maps, np.sum(residuals), best_gev

    @staticmethod
    def create_eeg_autoencoder(input_shape, encoding_dim):
        """
        Creates an autoencoder model for EEG data compression and reconstruction.
        """
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

    @staticmethod
    def find_original_centroids(eeg_data, cluster_labels, n_clusters):
        """Determines the original centroids of clustered data points."""
        original_centroids = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_data = eeg_data[cluster_indices]
            cluster_mean = np.mean(cluster_data, axis=0)
            original_centroids.append(cluster_mean)
        original_centroids = np.array(original_centroids)
        return original_centroids

    @staticmethod
    def calculate_residuals(eeg_data, autoencoder):
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

    @staticmethod
    def extract_features_with_pca(eeg_data, pca_components=10):
        """
        Reduces dimensionality of EEG data using Principal Component Analysis (PCA).
        """
        # Perform PCA to reduce dimensionality
        pca = PCA(n_components=pca_components)
        reduced_data = pca.fit_transform(eeg_data)
        return reduced_data

        # if method == 'Agglomerative Hierarchical Clustering':
        #     best_maps, best_residual, best_gev = self.run_aahc(
        #         preprocessed_data_path, extension, datatype,
        #         maps2use=maps2use,
        #         n_states=n_states,
        #         n_maps2use=50  # TODO: edit
        #     )
        #
        # else:

    def get_clustering_instance(self, maps2use, initial_maps, method, n_states, clustering_option):
        """
        Performs clustering on preprocessed data to find microstate maps.
        """

        def metric_function(point1, point2):
            return self.calculate_spatial_similarity(clustering_option, point1, point2)

        # Transpose maps2use
        maps2use = np.transpose(maps2use)

        # Define metric for clustering
        metric = distance_metric(type_metric.USER_DEFINED, func=metric_function)

        if method == 'K-Means Clustering':
            clustering_instance = kmeans.kmeans(
                data=maps2use,
                initial_centers=initial_maps,
                tolerance=self.clustering_tolerance,
                itermax=self.max_iterations,
                metric=metric
            )
        elif method == 'PCA + K-Means Clustering':
            clustering_instance = kmeans.kmeans(
                data=maps2use,
                initial_centers=initial_maps,
                tolerance=self.clustering_tolerance,
                itermax=self.max_iterations,
                metric=metric
            )
        elif method == 'Autoencoder + K-Means Clustering':
            clustering_instance = kmeans.kmeans(
                data=maps2use,
                initial_centers=initial_maps,
                tolerance=self.clustering_tolerance,
                itermax=self.max_iterations,
                metric=metric
            )
        elif method == 'X-Means Clustering':
            # Determine the splitting criterion for X-Means
            if clustering_option == 'Bayesian Information Criterion':
                criterion = xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION
            elif clustering_option == 'Minimum Noiseless Description Length':
                criterion = xmeans.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH
            else:
                raise ValueError("Failed to match criterion")

            clustering_instance = xmeans.xmeans(
                data=maps2use,
                initial_centers=initial_maps,
                kmax=n_states,
                tolerance=self.clustering_tolerance,
                criterion=criterion
            )
        else:
            raise ValueError("Failed to match method")

        return clustering_instance
