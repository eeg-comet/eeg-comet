#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Functions

"""

import numpy as np

from pyclustering.cluster import kmeans, xmeans, agglomerative, elbow, silhouette
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from functions.utils.compute_gev import compute_gev
from functions.utils.extract_peaks_maps import initialize_cluster_centers, generate_maps_and_peaks
#from functions.utils.elbow_function import get_elbow_without_plt


def modified_kmeans(data, initial_maps, n_states, max_iter=500, thresh=1e-6):

    # Get the dimensions of the data
    n_channels, n_samples = data.shape

    # Make a copy of initial_maps to avoid modifying the original array
    maps = initial_maps.copy()

    # Compute the sum of squares of the data
    data_sum_sq = np.sum(data ** 2)

    # Initialize iteration count and residuals
    iteration = 1
    prev_residual = np.inf
    residual = prev_residual

    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        for state in range(n_states):
            idx = (segmentation == state)

            # Update the microstate map for the current state
            maps[state] = np.dot(data[:, idx], activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        # V-maps, T-like symbol-segmentation
        act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq) / float(n_samples * (n_channels - 1))

        # Check for convergence
        if (prev_residual - residual) < (thresh * residual):
            print('Converged at', iteration, 'iterations.')
            break

        # Update previous residual and iteration count
        prev_residual = residual
        iteration += 1

    return maps, prev_residual


def run_modified_kmeans(maps2use, n_states, n_inits, initializer='Random', max_iter=500, thresh=1e-6):

    # Initialize variables to store the best results
    best_residual = None
    best_gev = 0
    best_maps = None

    # Perform clustering for multiple initializations
    for init in range(n_inits):
        print('\nClustering #', init + 1, 'of', n_inits)

        # Initialize microstate maps using the specified initializer
        initial_maps = initialize_cluster_centers(maps2use, n_states, initializer)

        # Run modified k-means algorithm
        maps, residual = modified_kmeans(maps2use, initial_maps, n_states, max_iter, thresh)

        # Compute GEV (Global Explained Variance)
        gev = compute_gev(maps2use, maps)

        print('Found', n_states, 'Microstate Maps')
        print('GEV:', gev)

        # Update the best results if the current GEV is higher
        if gev > best_gev:
            best_residual, best_gev, best_maps = residual, gev, maps

    print('\nBest GEV:', best_gev)
    return best_maps, best_gev, best_residual


def clustering_func(preprocessed_data_path, extension, datatype,
                    n_channels, method, n_states, initializer, use_percentages,
                    min_dist, max_iter, tolerance, n_inits, metric,
                    stopping_mode='gev', stopping_parameter=10.0, kmin=2, kmax=10):
    '''
    preprocessed_data_path: Path to the preprocessed data.
    hdf_concatenated_data_path: Path to the HDF concatenated data.
    n_channels: Number of channels in the data.
    method: Clustering method to use.
    n_states: Number of microstate maps
    initializer: Initialization method
    min_dist: Minimum distance
    n_inits: Number of initializations
    tolerance: Convergence threshold
    metric: Distance metric to use for clustering.

    '''

    maps2use, peaks2use = generate_maps_and_peaks(preprocessed_data_path, extension, datatype,
                                                  int(use_percentages), min_dist)

    if n_states == 'auto':

        #n_states = get_elbow_without_plt(maps2use, min_dist, tolerance, n_inits,
        #                                 kmin=kmin, kmax=kmax, stopping_mode=stopping_mode, threshold=stopping_parameter)
        print(f'result: n_states = {n_states}')

    if method == 'Modified K-means':
        best_maps, best_gev, best_residual = run_modified_kmeans(
            maps2use=maps2use,
            n_states=n_states,
            n_inits=n_inits,
            initializer=initializer,
            max_iter=max_iter,
            thresh=tolerance)
    else:

        initial_centers = initialize_cluster_centers(maps2use, n_states, initializer)
        maps2use = np.transpose(maps2use)

        if method == 'K-means':
            from scipy.spatial import distance
            if metric == 'Cosine Similarity':
                def cosine_sim(point1, point2):
                    return 1 - abs(distance.cosine(point1, point2))

                METRIC = distance_metric(type_metric.USER_DEFINED, func=cosine_sim)
            elif metric == 'Spatial Correlation':
                def spatial_corr(point1, point2):
                    return 1 - abs(distance.correlation(point1, point2))

                METRIC = distance_metric(type_metric.USER_DEFINED, func=spatial_corr)
            else:
                raise ValueError("Failed to match metric")

            clustering_instance = kmeans.kmeans(maps2use, initial_centers,
                                         tolerance=tolerance, itermax=max_iter,
                                         metric=METRIC)

        elif method == 'X-means':
            if metric == 'Bayesian Information Criterion':
                CRITERION = xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION
            elif metric == 'Minimum Noiseless Description Length':
                CRITERION = xmeans.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH 
            else:
                raise ValueError("Failed to match metric")
            clustering_instance = xmeans.xmeans(maps2use, initial_centers, n_states,
                                 tolerance=tolerance, criterion=CRITERION)
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
        for init in range(n_inits):
            print('\nClustering #', str(init + 1), 'of', str(n_inits))

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


            GEV_R = compute_gev(np.transpose(maps2use), np.array(centroids))

            print('Found', str(int(n_states)), 'Microstate Maps')
            print('GEV:', str(GEV_R))
            if GEV_R > best_gev:
                best_gev = GEV_R
                best_maps = centroids
                best_residual = residual

        print('\nBest GEV:', str(best_gev))

    return best_maps, best_gev, best_residual, n_states
