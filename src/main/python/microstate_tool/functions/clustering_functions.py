#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Functions

"""

import os
import mne
import numpy as np
import h5py
import pickle
# Don't remove these imports
# import sklearn.utils._cython_blas
# import sklearn.neighbors._typedefs
# import sklearn.neighbors._quad_tree
# import sklearn.tree
# import sklearn.utils._weight_vector
# import sklearn.tree._utils
# End
from fnmatch import fnmatch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from scipy.spatial import distance

from pyclustering.cluster import kmeans, xmeans, bsas, clarans, mbsas, optics, rock, agglomerative, ttsas, dbscan, elbow, silhouette
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import MiniBatchKMeans

from functions.utils.data_io import import_hdf_data
from functions.modified_kmeans import run_modified_kmeans, run_minibatch_modified_kmeans
from functions.utils.compute_gev import compute_gev
from functions.utils.extract_peaks_maps import extract_peaks_maps
from functions.utils.map_initializer import map_initializer
from functions.utils.elbow_function import get_elbow_without_plt


def number_of_clusters(maps, cmin=2, cmax=10):
    '''
    maps: Microstate maps 
    cmin: Minimum number of clusters to consider 
    cmax: Maximum number of clusters to consider 

    '''

    # create instance of Elbow method using C value from 2 to 10.
    # TODO: DEBUG: set ccore to False just for test on arm64 macOS
    elbow_instance = elbow.elbow(maps, cmin, cmax, ccore=True)
    # process input data and obtain results of analysis
    elbow_instance.process()
    amount_clusters = elbow_instance.get_amount()   # most probable amount of clusters
    #wce = elbow_instance.get_wce()                  # total within-cluster errors for each K
    return amount_clusters


def remove_similar_maps(data, centers, clusters):
    '''
    data: Input data matrix.
    centers: Cluster centers.
    clusters: Cluster assignments for each data point.
    '''
    sim = abs(cosine_similarity(centers))
    sim = np.triu(sim)
    np.fill_diagonal(sim, 0)
    #similar_maps = np.array(top_n_indexes(sim, N_STATES))
    similar_maps = np.array(np.where(sim>0.90)).transpose()
    final_clusters = np.empty((len(similar_maps),2),dtype=object)
    remove_maps = []
    i = 0
    for s in range(len(similar_maps)):
        to_rm = np.argmin([compute_gev(data, centers[similar_maps[s,0]]),
                          compute_gev(data, centers[similar_maps[s,1]])])
        #centers = np.delete(centers, similar_maps[s, to_rm], axis=0)
        
        final_clusters[s,0] = similar_maps[s, 1-to_rm]
        final_clusters[s,1] = np.sort(np.append(clusters[similar_maps[s, 1-to_rm]],
                                         clusters[similar_maps[s, to_rm]]))
        
        remove_maps = np.append(remove_maps, similar_maps[s, to_rm])
        i += 1
        
    remove_maps = np.unique(remove_maps)
    remove_maps = remove_maps.astype(int)
    #print(remove_maps)
    #max_arg = np.unravel_index(sim.argmax(), sim.shape)
    final_centers = np.delete(centers, remove_maps, axis=0)
    return final_centers, final_clusters


def clustering_func(preprocessed_data_path, hdf_concatenated_data_path,
                    n_channels, method, n_states, initializer, min_dist, n_inits, tolerance, metric, elbow_version='traditional', stopping_mode='gev', stopping_parameter=0.04):
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
    if n_states == 'auto':
        print("Loading the concatenated data ...")
        concatenated_data = import_hdf_data(hdf_concatenated_data_path)
        if elbow_version == 'traditional':            
            maps, _ = extract_peaks_maps(concatenated_data, min_dist)
            print('Using Elbow method to find the optimal number of microstate maps')
            n_states = number_of_clusters(maps)
        elif elbow_version == 'modified':
            n_states = get_elbow_without_plt(concatenated_data, min_dist, tolerance, n_inits, kmin=2, kmax=10, stopping_mode=stopping_mode, gev_threshold=stopping_parameter, res_threshold=stopping_parameter, sil_threshold=stopping_parameter)
        print(f'result: n_states = {n_states}')

    if method == 'Modified K-means':
        print("Loading the concatenated data ...")
        concatenated_data = import_hdf_data(hdf_concatenated_data_path)
        best_maps, best_gev, best_residual = run_modified_kmeans(
            data=concatenated_data,
            min_dist=min_dist,
            n_states=n_states,
            thresh=tolerance,
            n_inits=n_inits,
            initializer=initializer)
    elif method == 'Mini Batch Modified K-means':
        best_maps, best_gev, best_residual = run_minibatch_modified_kmeans(
            preprocessed_data_path=preprocessed_data_path,
            min_dist=min_dist,
            n_states=n_states,
            thresh=tolerance,
            n_inits=n_inits,
            initializer=initializer)
        concatenated_data = import_hdf_data(hdf_concatenated_data_path)
        best_gev = compute_gev(concatenated_data, np.array(best_maps))
        print('GEV:', str(best_gev))
    else:
        print("Loading the concatenated data ...")
        concatenated_data = import_hdf_data(hdf_concatenated_data_path)
        maps, peaks = extract_peaks_maps(concatenated_data, min_dist)

        initial_centers = map_initializer(
            concatenated_data,
            maps,
            peaks,
            n_states,
            initializer)

        if method == 'K-means':
            if metric == 'Euclidean':
                #METRIC = distance_metric(type_metric.EUCLIDEAN)
                def euclidean(point1, point2):
                    return 1 - abs(distance.euclidean(point1, point2))
                METRIC = distance_metric(type_metric.USER_DEFINED, func=euclidean)
            elif metric == 'Euclidean Square':
                #METRIC = distance_metric(type_metric.EUCLIDEAN_SQUARE)
                def euclidean_sqr(point1, point2):
                    return 1 - abs(distance.sqeuclidean(point1, point2))
                METRIC = distance_metric(type_metric.USER_DEFINED, func=euclidean_sqr)
            elif metric == 'Cosine Similarity':
                def cosine_sim(point1, point2):
                    return 1 - abs(distance.cosine(point1, point2))
                METRIC = distance_metric(type_metric.USER_DEFINED, func=cosine_sim)
            elif metric == 'Spatial Correlation':
                def spatial_corr(point1, point2):
                    return 1 - abs(pearsonr(point1, point2)[0])
                    # return 1 - abs(distance.correlation(point1, point2))
                METRIC = distance_metric(type_metric.USER_DEFINED, func=spatial_corr)
            elif metric == 'Dynamic Time Warping':
                from scipy.spatial.distance import euclidean
                def dtw(point1, point2):
                    dtw_distance, _ = fastdtw(point1, point2, dist=euclidean)
                    return dtw_distance
                    #return 1 - abs(dtw_distance)
                METRIC = distance_metric(type_metric.USER_DEFINED, func=dtw)
            else:
                raise ValueError("Failed to match metric")


            clustering_instance = kmeans.kmeans(maps, initial_centers,
                                         tolerance=tolerance, itermax=1000,
                                         metric=METRIC)

        elif method == 'X-means':
            if metric == 'Bayesian Information Criterion':
                CRITERION = xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION
            elif metric == 'Minimum Noiseless Description Length':
                CRITERION = xmeans.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH 
            else:
                raise ValueError("Failed to match metric")
            clustering_instance = xmeans.xmeans(maps, initial_centers, n_states,
                                 tolerance=tolerance, criterion=CRITERION)
        elif method == 'Agglomerative hierarchical clustering':
            if metric == 'Single Link':
                aahc_link = agglomerative.type_link.SINGLE_LINK
            elif metric == 'Complete Link':
                aahc_link = agglomerative.type_link.COMPLETE_LINK
            elif metric == 'Average Link':
                aahc_link = agglomerative.type_link.AVERAGE_LINK
            elif metric == 'Centroid Link':
                aahc_link = agglomerative.type_link.CENTROID_LINK
            else:
                raise ValueError("Failed to match metric")
            clustering_instance = agglomerative.agglomerative(maps, n_states, aahc_link, True);

        elif method == 'BSAS':
            clustering_instance = bsas.bsas(maps, n_states, tolerance);
        elif method == 'CLARANS':
            clustering_instance = clarans.clarans(maps, n_states, 100, 10);
        elif method == 'MBSAS':
            clustering_instance = mbsas.mbsas(maps, n_states, tolerance);
        elif method == 'OPTICS':
            clustering_instance = optics.optics(maps, 2.0, 3,
                                         amount_of_clusters=n_states);
        elif method == 'ROCK':
            clustering_instance = rock.rock(maps, 1.0, n_states, tolerance);
        elif method == 'DBSCAN':
            clustering_instance = dbscan.dbscan(maps, tolerance, min_dist, True);
        elif method == 'TTSAS':
            clustering_instance = ttsas.ttsas(maps, n_states, tolerance);
        else:
            raise ValueError("Failed to match method")


        best_gev = 0
        for init in range(n_inits):
            print('\nClustering #', str(init + 1), 'of', str(n_inits))

            centers = np.zeros((n_states, n_channels))
            #n = 0
            #while centers.shape[0] != int(n_states/2):
                ###
            # Run Cluster Analysis
            clustering_instance.process()
            clusters = clustering_instance.get_clusters()
            #score = silhouette.silhouette(maps, clusters).process().get_score()
            best_residual = 0#clustering_instance.get_total_wce()
            centers = np.empty((n_states,n_channels))
            for cl in range(len(clusters)):
                centers[cl,:] = np.mean(maps[clusters[cl],:],axis=0)
           
            # Filter Maps
            #centers, clusters = remove_similar_maps(data, centers, clusters)
            '''
            if n == 5:
                not_converged = True
                print('not converged')
                break
            else:
                not_converged = False
                n += 1
                '''
            ###

            #if not not_converged:
                ###
            GEV_R = compute_gev(concatenated_data, np.array(centers))

            print('Found', str(int(n_states)), 'Microstate Maps')
            print('GEV:', str(GEV_R))
            if GEV_R > best_gev:
                best_gev = GEV_R
                best_maps = centers

        print('\nBest GEV:', str(best_gev))

    return best_maps, best_gev, best_residual, n_states
