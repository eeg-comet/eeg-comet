#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:52:42 2021

@author: amin
"""

import mne
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks

from pyclustering.cluster import kmeans, xmeans, bsas, clarans, mbsas, optics, rock, elbow
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


###### temp
### change
import os
from fnmatch import fnmatch
from scipy.signal import butter, lfilter
import collections
from matplotlib import pyplot as plt
import glob

eeglist = []
input_folder="/home/amin/Encfs/TMSEEG_DATA/microstate_toolbox/data/"
pattern="*"
extension=".set"
for path, subdirs, files in os.walk(input_folder):
    for name in files:
        if fnmatch(name, pattern+extension):
            eeglist.append(os.path.join(path, name))
print(eeglist)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def concatenate_files(folder):
    os.chdir(folder)
    extension = 'set'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    
    for file in range(len(all_filenames)):
        print(100*file/len(all_filenames))
        filename = all_filenames[file]
        # Load the example MNE data
        EEG = mne.io.read_raw_eeglab(folder+filename, preload=True, verbose='CRITICAL')
        # Select EEG channels from the dataset
        EEG = EEG.pick_types(meg=False, eeg=True, eog=False, verbose='CRITICAL')
        if file == 0:
            channels = EEG.info['ch_names']
        else:
            channels = np.append(channels,EEG.info['ch_names'])
    
    counter = collections.Counter(channels)
    counter = np.array(list(counter.items()))
    channels2remove = counter[np.where(counter[:,1].astype(float) < len(all_filenames)),0].tolist()

    for file in range(len(all_filenames)):
        print(100*file/len(all_filenames))
        filename = all_filenames[file]
        print('\nLoading EEG Files ... ', filename)
        # Load the example MNE data
        EEG = mne.io.read_raw_eeglab(folder+filename, preload=True, verbose='CRITICAL')
        # Select EEG channels from the dataset    
        EEG = EEG.pick_types(meg=False, eeg=True, eog=False,
                             exclude=channels2remove[0], verbose='CRITICAL')
        EEG = EEG.set_eeg_reference('average')
        
        data_tmp = EEG[:,:][0]
        data_len = data_tmp.shape[1]
        
        #n_channels = EEG.info['nchan']
        eeg_info = EEG.info
        # Sampling Rate
        global Fs
        Fs = 250
        
        if EEG.info['sfreq'] != Fs:
            EEG = EEG.resample(sfreq=Fs)
        if file == 0:
            filenames = filename
            data = data_tmp
            data_length = data_len
        else:
            filenames = np.append(filenames, filename)
            data = np.append(data, data_tmp, axis=1)
            data_length = np.append(data_length, data_len)
    return data, filenames, eeg_info, data_length

DATA, FILENAMES, EEG_INFO, LENGTH_DATA = concatenate_files(input_folder)
print(FILENAMES)
n_channels = EEG_INFO['nchan']
# Filter Data
INPUT_DATA = butter_bandpass_filter(DATA,2,20,Fs,order=5)
###### temp
### change



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

def _pre_clustering(data, fs, smoothing):
    # Global Field Potential (GFP)
    gfp = np.std(data, axis=0)
    if smoothing:
        gfp = smooth_data(gfp, smoothing)
    # Find GFP Peaks
    min_dist = 50
    peaks, _ = find_peaks(gfp, distance=fs*min_dist/1000)
    # Create Maps
    maps = data[:, peaks].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    return maps, peaks


def _gev(data, maps):
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



def concatenate_files(data):
    count = 0
    if count == 0:
        DATA = data
    else:
        DATA = np.append(DATA, data, axis=1)
        count = count+1
    return DATA

def number_of_clusters(maps, cmin=4, cmax=20):
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
        to_rm = np.argmin([_gev(data, centers[similar_maps[s,0]]),
                          _gev(data, centers[similar_maps[s,1]])])
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

def clustering_func(data, maps, method, n_states, initial_centers, repeat, tolerance, metric):
    
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
        print('Number of Microstate Maps: ', int(n_states/2))
        
        
        centers = np.zeros((n_states, n_channels))
        n = 0
        while centers.shape[0] != int(n_states/2):
            # Run Cluster Analysis
            clustering_instance.process()
            clusters = clustering_instance.get_clusters()
            centers = np.empty((n_states,n_channels))
            for cl in range(len(clusters)):
                centers[cl,:] = np.mean(maps[clusters[cl],:],axis=0)
            # Filter Maps
            centers, clusters = remove_similar_maps(data, centers, clusters)
            
            if n == 5:
                not_converged = True
                print('not converged')
                break
            else:
                not_converged = False
                n += 1
        
        GEV = 0
        if not not_converged:
            GEV_R = _gev(INPUT_DATA, np.array(centers))
            print('GEV = ', GEV_R)
            if GEV_R > GEV:
                GEV = GEV_R
                best_maps = centers
    
    print('\nBest GEV = ', GEV)
    
    activation = np.array(best_maps).dot(DATA)
    final_segmentation = np.argmax(np.abs(activation), axis=0)  
    
    return clustering_instance, best_maps, GEV, final_segmentation