#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Functions

"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import spatial
from pyclustering.cluster import kmeans, xmeans, agglomerative, elbow, silhouette
from pyclustering.utils.metric import distance_metric, type_metric
from functions.data_utils.extract_peaks_maps import initialize_cluster_centers, generate_maps_and_peaks

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
        """Computes the global explained variance (GEV) of microstate maps."""

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

    def modified_kmeans(self, data, initial_maps, n_states, max_iter=500, thresh=1e-6):
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
                print('Converged at', iteration, 'iterations.')
                break

            prev_residual = residual

        return maps, prev_residual

    def run_modified_kmeans(self, preprocessed_data_path, extension, datatype,
                            maps2use, n_states, n_inits, initializer='Random', max_iter=500, thresh=1e-6):

        all_data, _ = generate_maps_and_peaks(preprocessed_data_path, extension, datatype, use_percentages=100)

        best_residual, best_gev, best_maps = None, 0, None

        for init in range(n_inits):
            print(f'\nClustering #{init + 1} of {n_inits}')

            initial_maps = initialize_cluster_centers(maps2use, n_states, initializer)
            maps, residual = self.modified_kmeans(maps2use, initial_maps, n_states, max_iter, thresh)
            gev = self.compute_gev(all_data, maps)

            print(f'Found {n_states} Microstate Maps')
            print(f'GEV: {gev}')

            # Update the best results if current gev is higher
            if gev > best_gev:
                best_residual, best_gev, best_maps = residual, gev, maps

        print(f'\nBest GEV: {best_gev}')
        return best_maps, best_gev, best_residual

    def clustering_func(self, preprocessed_data_path, extension, datatype,
                        n_channels, method, n_states, initializer, use_percentages,
                        min_dist, metric, stopping_mode='gev', stopping_parameter=10.0, kmin=2, kmax=10):
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
            metric: Distance metric to use for clustering

        Returns:
            best_maps: Best cluster centers (microstate maps)
            best_gev: Best global explained variance
            best_residual: Best residual error
            n_states: Optimal number of clusters (microstate maps)
        """

        maps2use, peaks2use = generate_maps_and_peaks(preprocessed_data_path, extension, datatype,
                                                      use_percentages, min_dist)

        elbow_optimizer = ElbowOptimizer(maps2use, min_dist, self.n_inits, kmin, kmax, self.tolerance, self.max_iter)
        if n_states == 'auto':
            n_states = elbow_optimizer.find_elbow_without_plot(stopping_mode='gev', threshold=stopping_parameter)
            print(f'result: n_states = {n_states}')

        if method == 'Modified K-means':
            best_maps, best_gev, best_residual = self.run_modified_kmeans(
                preprocessed_data_path, extension, datatype,
                maps2use=maps2use,
                n_states=n_states,
                n_inits=self.n_inits,
                initializer=initializer,
                max_iter=self.max_iter,
                thresh=self.tolerance)
        else:

            initial_centers = initialize_cluster_centers(maps2use, n_states, initializer)
            maps2use = np.transpose(maps2use)

            if method == 'K-means':
                def metric_function(point1, point2):
                    if metric == 'Cosine Similarity':
                        # Calculates the cosine similarity
                        dist = spatial.distance.cosine(point1, point2)
                    elif metric == 'Spatial Correlation':
                        # Calculates the spatial correlation
                        dist = spatial.distance.correlation(point1, point2)
                    else:
                        raise ValueError("Failed to match metric")
                    return 1 - dist

                metric = distance_metric(type_metric.USER_DEFINED, func=metric_function)
                clustering_instance = kmeans.kmeans(maps2use, initial_centers,
                                             tolerance=self.tolerance, itermax=self.max_iter,
                                             metric=metric)

            elif method == 'X-means':
                if metric == 'Bayesian Information Criterion':
                    CRITERION = xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION
                elif metric == 'Minimum Noiseless Description Length':
                    CRITERION = xmeans.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH
                else:
                    raise ValueError("Failed to match metric")
                clustering_instance = xmeans.xmeans(maps2use, initial_centers, n_states,
                                     tolerance=self.tolerance, criterion=CRITERION)
            elif method == 'Agglomerative hierarchical clustering':
                from sklearn.cluster import AgglomerativeClustering
                from sklearn.metrics import pairwise_distances
                def cosine_distance(X, Y=None):
                    return pairwise_distances(X, Y, metric='cosine')

                clustering_instance = AgglomerativeClustering(n_clusters=n_states,
                                                              affinity=cosine_distance,
                                                              linkage='average')
            else:
                raise ValueError("Failed to match method")

            best_gev = 0
            for init in range(self.n_inits):
                print('\nClustering #', str(init + 1), 'of', str(self.n_inits))

                if method == 'K-means':
                    clustering_instance.process()
                    residual = clustering_instance.get_total_wce()
                    centroids = clustering_instance.get_centers()
                elif method == 'Agglomerative hierarchical clustering':
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



class ElbowOptimizer:
    """Finds the optimal number of clusters (K) for clustering EEG data using the Elbow method.

        Attributes:
            maps2use (numpy array): The EEG data for clustering.
            min_dist (float): The minimum distance between clusters.
            n_inits (int): Number of initializations for K-Means.
            kmin (int): Minimum number of clusters to evaluate.
            kmax (int): Maximum number of clusters to evaluate.
            tolerance (float, optional): The tolerance for convergence.
            max_iter (int, optional): Maximum iterations for K-Means.
        """
    def __init__(self, maps2use, min_dist, n_inits, kmin, kmax, tolerance=None, max_iter=None):
        """Initialize the ElbowOptimizer with given parameters."""
        self.maps2use = maps2use
        self.min_dist = min_dist
        self.tolerance = tolerance
        self.n_inits = n_inits
        self.kmin = kmin
        self.kmax = kmax
        self.max_iter = max_iter
        self.microstate_clusterer = MicrostateClusterer(n_inits=1)
        self.N, self.RES, self.GEV, self.SIL = [], [], [], []

    def _cluster_and_evaluate(self, k):
        """Cluster the EEG data for a given K and evaluate the quality using various metrics."""
        gev_i, residual_i = 0, 0
        for init in range(self.n_inits):
            maps, gev, residual = self.microstate_clusterer.run_modified_kmeans(
                maps2use=self.maps2use,
                n_states=k,
                n_inits=1,
                initializer="Random",
                max_iter=self.max_iter,
                thresh=self.tolerance
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

    def find_elbow_without_plot(self, stopping_mode='gev', threshold=0.1):
        """Find the elbow point without generating plots."""
        for k in range(self.kmin, self.kmax + 1):
            print(f'\nClustering data with {k} microstates')
            gev_mean, residual_mean, sil_mean = self._cluster_and_evaluate(k)
            self.N.append(k)
            self.RES.append(residual_mean)
            self.GEV.append(gev_mean)
            self.SIL.append(sil_mean)

            if len(self.N) == 1:
                continue

            if self._should_stop(stopping_mode, gev_mean, residual_mean, sil_mean, threshold):
                return k

        return int((self.kmin + self.kmax) / 2)

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


    def _should_stop(self, stopping_mode, gev_mean, residual_mean, sil_mean, threshold):
        """Decide if the clustering should stop based on the given stopping mode and threshold."""
        if stopping_mode == 'gev':
            return abs(gev_mean - self.GEV[-2]) / self.GEV[-2] < threshold
        elif stopping_mode == 'res':
            return abs(residual_mean - self.RES[-2]) / self.RES[-2] < threshold
        elif stopping_mode == 'sil':
            return abs(sil_mean - self.SIL[-2]) / self.SIL[-2] < threshold
        else:
            raise ValueError("Invalid stopping_mode. Choose 'gev', 'res', or 'sil'.")