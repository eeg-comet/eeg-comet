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
from itertools import groupby

from pyclustering.cluster import kmeans, xmeans, bsas, clarans, mbsas, optics, rock, elbow
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import MiniBatchKMeans


from functions import modified_kmeans


def eegInfo(folder):
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in [f for f in filenames if f.startswith("EEG_INFO")]:
            eegInfo_path = os.path.join(dirpath, filename)
    if os.path.exists(eegInfo_path):
        with open(eegInfo_path, 'rb') as f:
            info = pickle.load(f)
    return info
    
def _corr_vectors(A, B, axis=0):
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.sum(An * Bn, axis=axis)

def smooth_data(gfp, kernel_size):
    kernel = np.ones(kernel_size)/kernel_size
    smoothed_data = np.convolve(gfp, kernel, mode='same')
    return smoothed_data

def pre_clustering(data, fs, smoothing):
    # Global Field Potential (GFP)
    if smoothing:
        for i in range(len(data)):
            data[i] = smooth_data(data[i], smoothing)
    gfp = np.std(data, axis=0)
    #test_gfp = gfp[0:500]
    #plt.plot(test_gfp)
    if smoothing:
        gfp = smooth_data(gfp, smoothing)
        #test_gfp = gfp[0:500]
        #plt.plot(test_gfp)
    # Find GFP Peaks

    min_dist = smoothing
    peaks, _ = find_peaks(gfp, distance=min_dist)
    #test_peaks_in = peaks <= 500
    #test_peaks = peaks[test_peaks_in]
    #plt.plot(test_peaks, test_gfp[test_peaks], "x")
    #plt.show()
    # Create Maps
    maps = data[:, peaks].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    return maps, peaks


def compute_gev(data, maps):
    gfp = np.std(data, axis=0)
    gfp_sum_sq = np.sum(gfp ** 2)
    if maps.ndim == 1:
        maps /= np.linalg.norm(maps, keepdims=True)
        maps = np.reshape(maps, (1,-1))
    else:
        maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    activation = np.array(maps).dot(data)
    segmentation = np.argmax(np.abs(activation), axis=0)
    map_corr = _corr_vectors(data, maps[segmentation].T)
    gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
    return gev

    
def plot_maps(maps, info):
    maps = np.array(maps)
    plt.figure(figsize=(2 * len(maps), 2))
    for i, map in enumerate(maps):
        plt.subplot(1, len(maps), i + 1)
        mne.viz.plot_topomap(map, info)
        plt.title('%d' % i)


def number_of_clusters(maps, cmin=2, cmax=10):
    # create instance of Elbow method using C value from 2 to 10.
    elbow_instance = elbow.elbow(maps, cmin, cmax)
    # process input data and obtain results of analysis
    elbow_instance.process()
    amount_clusters = elbow_instance.get_amount()   # most probable amount of clusters
    #wce = elbow_instance.get_wce()                  # total within-cluster errors for each K
    return amount_clusters

def plot_gev_maps(data, cmin, cmax, repeat, tolerance, smoothing):
    # plot gev versus number of maps
    N, GEV = [], []
    for n_maps in range(cmin, cmax):
        best_gev = modified_kmeans.segment(data=data,
                                           n_states=n_maps,
                                           n_inits=repeat,
                                           thresh=tolerance,
                                           min_peak_dist=smoothing)[2]
        N = np.append(N, n_maps)
        GEV = np.append(GEV, best_gev)
    plt.plot(N, GEV)

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

def clustering_func(data, n_channels, maps, method, n_states, initial_centers,
                    smoothing, repeat, tolerance, metric):
    
    #n_channels = data.shape[0]
    
    if method == 'MODIFIED K-MEANS':
        #
        #plot_gev_maps(data, 2, 10, repeat, tolerance, smoothing)
        #
        best_maps, final_segmentation, best_gev = modified_kmeans.segment(data=data,
                                                             n_states=n_states,
                                                             n_inits=repeat,
                                                             thresh=tolerance,
                                                             min_peak_dist=smoothing,
                                                             max_n_peaks=None)
            
    else:
        if method == 'K-MEANS':
            if metric == 'Euclidean':
                METRIC = type_metric.EUCLIDEAN
            elif metric == 'Euclidean Square':
                METRIC = type_metric.EUCLIDEAN_SQUARE
            elif metric == 'Manhattan':
                METRIC = type_metric.MANHATTAN
            elif metric == 'Chebyshev':
                METRIC = type_metric.CHEBYSHEV
            elif metric == 'Minkowski':
                METRIC = type_metric.MINKOWSKI
            clustering_instance = kmeans.kmeans(maps, initial_centers,
                                         tolerance=tolerance, itermax=100,
                                         metric=distance_metric(METRIC))
        
        elif method == 'X-MEANS':
            if metric == 'Bayesian Information Criterion':
                CRITERION = xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION
            elif metric == 'Minimum Noiseless Description Length':
                CRITERION = xmeans.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH 
            clustering_instance = xmeans.xmeans(maps, initial_centers, n_states,
                                 tolerance=tolerance, criterion=CRITERION)
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
            best_gev = 0
            #if not not_converged:
                ###
            GEV_R = compute_gev(data, np.array(centers))
            print('GEV = ', GEV_R)
            if GEV_R > best_gev:
                best_gev = GEV_R
                best_maps = centers
    
        print('\nBest GEV = ', best_gev)
        
        activation = np.array(best_maps).dot(data)
        final_segmentation = np.argmax(np.abs(activation), axis=0)

        # remove isolated segments
        count_dups = [sum(1 for _ in group) for _, group in groupby(final_segmentation)]
        for C in range(len(count_dups)):
            if count_dups[C] < 2:
                index = int(np.sum(count_dups[0:C]))
                if C == 0:
                    final_segmentation[index] = final_segmentation[index + 1]
                else:
                    final_segmentation[index] = final_segmentation[index - 1]
    return best_maps, final_segmentation, best_gev


def clustering_minibatch(folder, fs, smoothing, n_states, initializer, tolerance):
    
    minibatchk = MiniBatchKMeans(n_clusters=n_states,
                                 init=initializer.lower(),
                                 max_iter=100,
                                 batch_size=10,
                                 verbose=1,
                                 tol=tolerance)
    
    eeglist = []
    extension="*.h5"
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if fnmatch(name, extension):
                eeglist.append(os.path.join(path, name))
    
    for filename in eeglist:
        with h5py.File(filename, "r") as f:
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            data = list(f[a_group_key])
        data = np.asarray(data)
        maps, peaks = pre_clustering(data, fs, smoothing)
        
        minibatchk = minibatchk.partial_fit(np.abs(maps))
    
    best_maps = minibatchk.cluster_centers_
    final_segmentation = minibatchk.labels_
    
    best_gev = compute_gev(data, np.array(best_maps))
    print('\nBest GEV = ', best_gev)
    
    return best_maps, final_segmentation, best_gev