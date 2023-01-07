#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:52:42 2021

@author: amin
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
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from scipy.spatial import distance

from pyclustering.cluster import kmeans, xmeans, bsas, clarans, mbsas, optics, rock, agglomerative, elbow, silhouette
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import MiniBatchKMeans


from functions import modified_kmeans
from functions.utils.compute_gev import compute_gev


def eegInfo(folder):
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in [f for f in filenames if f.startswith("EEG_INFO")]:
            eegInfo_path = os.path.join(dirpath, filename)
    if os.path.exists(eegInfo_path):
        with open(eegInfo_path, 'rb') as f:
            info = pickle.load(f)
    return info

def smooth_data(gfp, kernel_size):
    kernel = np.ones(kernel_size)/kernel_size
    smoothed_data = np.convolve(gfp, kernel, mode='same')
    return smoothed_data

def pre_clustering(data, min_dist):
    # Global Field Potential (GFP)
    #if smoothing:
    #    for i in range(len(data)):
    #        data[i] = smooth_data(data[i], smoothing)
    gfp = np.std(data, axis=0)
    #test_gfp = gfp[0:500]
    #plt.plot(test_gfp)
    if min_dist:
        # Find GFP Peaks
        #smoothing = smoothing / (1000/fs)
        #min_dist = int(smoothing / fs)
        #gfp = smooth_data(gfp, min_dist)
        #test_gfp = gfp[0:500]
        #plt.plot(test_gfp)

        peaks, _ = find_peaks(gfp, distance=min_dist)
    else:
        peaks, _ = find_peaks(gfp)
    #peaks = find_peaks_cwt(gfp, widths=10)
    #test_peaks_in = peaks <= 500
    #test_peaks = peaks[test_peaks_in]
    #plt.plot(test_peaks, test_gfp[test_peaks], "x")
    #plt.show()
    # Create Maps
    maps = data[:, peaks].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    return maps, peaks


def number_of_clusters(maps, cmin=2, cmax=10):
    # create instance of Elbow method using C value from 2 to 10.
    elbow_instance = elbow.elbow(maps, cmin, cmax)
    # process input data and obtain results of analysis
    elbow_instance.process()
    amount_clusters = elbow_instance.get_amount()   # most probable amount of clusters
    #wce = elbow_instance.get_wce()                  # total within-cluster errors for each K
    return amount_clusters

def initialize_centers(data, maps, peaks, n_states, initializer):
    # create instance of K-Means algorithm with prepared centers
    if initializer == 'Random':
        random_state = np.random.RandomState(None)
        chosen_peaks = random_state.choice(len(peaks),size=n_states,replace=False)
        initial_peaks = peaks[chosen_peaks].tolist()
        initial_centers = data[:, initial_peaks].T
    elif initializer == 'K-Means++':
        # Calculate initial centers using K-Means++ method.
        initial_centers = kmeans_plusplus_initializer(maps,n_states).initialize()
    initial_centers /= np.linalg.norm(initial_centers, axis=1, keepdims=True)
    return initial_centers

def remove_similar_maps(data, centers, clusters):
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

def clustering_func(data, n_channels, method, n_states, initializer,
                    min_dist, repeat, tolerance, metric):
    
    #n_channels = data.shape[0]

    maps, peaks = pre_clustering(data, min_dist)

    if n_states=='auto':
        print('Using Elbow method to find the optimal number of microstate maps')
        n_states = number_of_clusters(maps)

    if method == 'Modified K-means':
        best_maps, best_gev, best_residual = modified_kmeans.segment(data=data,
                                                         peaks=peaks,
                                                         n_states=n_states,
                                                         n_inits=repeat,
                                                         thresh=tolerance,
                                                         min_peak_dist=min_dist,
                                                         max_n_peaks=None)
    else:

        initial_centers = initialize_centers(
            data,
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
                    return 1 - abs(distance.correlation(point1, point2))
                METRIC = distance_metric(type_metric.USER_DEFINED, func=spatial_corr)

            clustering_instance = kmeans.kmeans(maps, initial_centers,
                                         tolerance=tolerance, itermax=1000,
                                         metric=METRIC)




        elif method == 'X-means':
            if metric == 'Bayesian Information Criterion':
                CRITERION = xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION
            elif metric == 'Minimum Noiseless Description Length':
                CRITERION = xmeans.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH 
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
            clustering_instance = rock.rock(maps, 1.0, n_states);

        best_gev = 0
        for r in range(repeat):
            print('\nClustering: ', r+1)
            print('Number of Microstate Maps: ', int(n_states))
            
            
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
            GEV_R = compute_gev(data, np.array(centers))
            print('GEV = ', GEV_R)
            if GEV_R > best_gev:
                best_gev = GEV_R
                best_maps = centers
    
        print('\nBest GEV = ', best_gev)

    return best_maps, best_gev, best_residual


def clustering_minibatch(outputfolder, method, n_states, initializer,
                    min_dist, repeat, tolerance, metric):
    eeglist = []
    extension = "*.h5"
    for path, _, files in os.walk(os.path.join(outputfolder, 'preprocessed_data')):
        for name in files:
            if fnmatch(name, extension):
                eeglist.append(os.path.join(path, name))

    all_maps = []
    for file in eeglist:
        print(file)
        with h5py.File(file, "r") as f:
            a_group_key = list(f.keys())[0]
            data = list(f[a_group_key])
        data = np.asarray(data)
        n_channels = data.shape[0]

        maps, peaks = pre_clustering(data, min_dist)
        all_maps = np.append(all_maps, maps)

        maps, _ = clustering_func(data, n_channels, method, n_states, initializer,
                    min_dist, repeat, tolerance, metric)
        best_gev, avg_gev = 0, 0
        for file in eeglist:
            with h5py.File(file, "r") as f:
                # List all groups
                a_group_key = list(f.keys())[0]
                # Get the data
                data = list(f[a_group_key])
            gev = compute_gev(data, np.array(maps))
            avg_gev = avg_gev + gev
        avg_gev = avg_gev/len(eeglist)
        print(avg_gev)
        if avg_gev > best_gev:
            best_maps, best_gev = maps, gev

    return best_maps, best_gev