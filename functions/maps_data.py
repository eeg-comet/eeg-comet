#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 08:54:07 2021

@author: amin
"""

import mne
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
from itertools import groupby


#from pyclustering import cluster
from pyclustering.cluster import kmeans, xmeans, bsas, clarans, mbsas, optics, rock, elbow
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import os
from fnmatch import fnmatch

### change
from scipy.signal import butter, lfilter
import collections
from matplotlib import pyplot as plt
from matplotlib.pyplot import ion, show
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt5')

import random

#input_folder="/home/amin/Encfs/TMSEEG_DATA/microstate_toolbox/data/"
input_folder="/media/amin/Seagate Expansion Drive/AMIN/RS_EEG/EEG_Preprocessed/"

save_folder = '/media/amin/Seagate Expansion Drive/AMIN/RS_EEG/RSEEG/img_clustering/5class/'

counter = 970


iterations = 10000
for it in range(iterations):
    print("iteration ", it)
    converged = 0
    
    eeglist = []

    R_P = np.random.randint(3)
    if R_P == 0:
        pattern="*EC"
    elif R_P == 1:
        pattern="*EO"
    else:
        pattern="*"
    
    extension=".set"
    for path, subdirs, files in os.walk(input_folder):
        for name in files:
            if fnmatch(name, pattern+extension):
                eeglist.append(os.path.join(path, name))
    
    # number of eeg files
    n_eeg = np.random.randint(3,20)
    #n_eeg = 6
    eeglist = random.sample(eeglist,n_eeg)
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
    
    def concatenate_files(all_filenames):
        
        for file in range(len(all_filenames)):
            print(100*file/len(all_filenames))
            filename = all_filenames[file]
            # Load the example MNE data
            EEG = mne.io.read_raw_eeglab(filename, preload=True, verbose='CRITICAL')
            # Select EEG channels from the dataset
            EEG = EEG.pick_types(meg=False, eeg=True, eog=False, verbose='CRITICAL')
            if file == 0:
                channels = EEG.info['ch_names']
            else:
                channels = np.append(channels,EEG.info['ch_names'])
        
        counter = collections.Counter(channels)
        counter = np.array(list(counter.items()))
        channels2remove = counter[np.where(counter[:,1].astype(float) < len(all_filenames)),0].tolist()
        print('Channels to remove: ', channels2remove)
    
        for file in range(len(all_filenames)):
            print(100*file/len(all_filenames))
            filename = all_filenames[file]
            print('\nLoading EEG Files ... ', filename)
            # Load the example MNE data
            EEG = mne.io.read_raw_eeglab(filename, preload=True, verbose='CRITICAL')
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
    
    
    
    
    '''
    # Global Field Potential
    gfp = np.std(DATA, axis=0)
    # Smooth Data
    gfp = utils_microstate.smooth_data(gfp, kernel_size)    
    peaks, _ = find_peaks(gfp, distance=Fs*min_dist/1000)
    maps = DATA[:, peaks].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    '''
    ### change
    
    
    ###
    DATA, FILENAMES, EEG_INFO, LENGTH_DATA = concatenate_files(eeglist)
    print(FILENAMES)
    n_channels = EEG_INFO['nchan']
    # Filter Data
    INPUT_DATA = butter_bandpass_filter(DATA,2,20,Fs,order=5)
    #INPUT_DATA = DATA
    data_sum_sq = np.sum(DATA ** 2)
    
    
    ## Functions
    
    
    
    
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
            #plt.title('%d' % i)
    
    
    
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
        similar_maps = np.array(np.where(sim>0.8)).transpose()
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
    
    def clustering_func(data, maps, method, n_states, initial_centers, tolerance, metric):
        
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
            
            
            if n == 10:
                not_converged = True
                print('not converged')
                break
            else:
                not_converged = False
                n += 1
        
        if not not_converged:
            GEV = _gev(INPUT_DATA, np.array(centers))
            print('GEV = ', GEV)
            best_maps = centers
            
            activation = np.array(best_maps).dot(DATA)
            final_segmentation = np.argmax(np.abs(activation), axis=0)
        else:
            clustering_instance, best_maps, GEV, final_segmentation = [], [], 0, []
        
        return clustering_instance, best_maps, GEV, final_segmentation
    
    def extract_features(segmentation, fs, features):
        if segmentation is not None:
            extracted_features_df = pd.DataFrame()
            for i in range(len(LENGTH_DATA)):
                if i==0:
                    start = 0
                    stop = LENGTH_DATA[i]
                    stop_pre = stop
                else:
                    start = stop_pre
                    stop = stop_pre+LENGTH_DATA[i]
                    stop_pre = stop
                segment_each = segmentation[start:stop]
                
                extracted_features, headers = [], []
                headers = np.append(headers, "Filename")
                extracted_features = np.append(extracted_features, FILENAMES[i])
                for c in np.unique(segmentation).tolist():
                    if "FOC" in features:
                        # Frequency of Occurence for each map per second
                        FOC = 100*np.count_nonzero(segment_each == c)/len(segment_each)
                        headers = np.append(headers, "FOC_"+c)
                        extracted_features = np.append(extracted_features, FOC)
                    if "MMD" in features:
                        # Mean Microstates Duration
                        D = [sum(1 for i in g) for k,g in groupby(segment_each) if k==c]
                        MMD = (1000/fs) * (c if not D else sum(D) / len(D))
                        headers = np.append(headers, "MMD_"+c)
                        extracted_features = np.append(extracted_features, MMD)
                extracted_features_df = extracted_features_df.append(pd.DataFrame(extracted_features.reshape(1,len(extracted_features)),
                                                     columns=headers.tolist()))
        return extracted_features_df
    
    def save_features(extracted_features_df, save_path):
        save_name = os.path.join(save_path, 'extracted_features.csv')
        extracted_features_df.to_csv(save_name, index=False, header=True)
    
    
    
    
    #####
    
    def do_clustering(METHOD, N_STATES):
            
            Fs = 250 #from previous window
            DATA = INPUT_DATA #from previous window
            RAND = np.random.rand()
            if RAND < 0.5:
                SMOOTHING = 5
            else:
                SMOOTHING = 9
            MAPS, PEAKS = _pre_clustering(DATA, Fs, SMOOTHING)
                    
            TOLERANCE = 1E-5
            
            GEV = 0
            BEST_MAPS = None
            
            REPEAT = 3
            for r in range(REPEAT):
                print('\nClustering: ', r+1)
                print('Number of Microstate Maps: ', int(N_STATES/2))
            
                RAND = np.random.rand()
                if RAND < 0.5:
                    INITIALIZER = "Random"
                else:
                    INITIALIZER = "K-Means++"
                INITIAL_CENTERS = initialize_centers(DATA, MAPS, PEAKS, N_STATES, INITIALIZER)
                
                RAND1 = np.random.randint(4)
                RAND2 = np.random.rand()
                if METHOD == 'K-MEANS':
                    if RAND1 == 0:
                        METRIC = 'Euclidean Square'
                    elif RAND1 == 1:
                        METRIC = 'Euclidean'
                    elif RAND1 == 2:
                        METRIC = 'Chebyshev'
                    elif RAND1 == 3:
                        METRIC = 'Manhattan'
                elif METHOD == 'X-MEANS':
                    if RAND2 < 0.5:
                        METRIC = 'Minimum Noiseless Description Length'
                    else:
                        METRIC = 'Bayesian Information Criterion' 
                print("Initializer: ", INITIALIZER)
                print("Method: ", METHOD, "Metric: ", METRIC)
                
                clustering_instance, best_maps, gev, final_segmentation = clustering_func(DATA, MAPS, METHOD,
                                                                                          N_STATES, INITIAL_CENTERS,
                                                                                          TOLERANCE, METRIC)
                if gev > GEV:
                    GEV = gev
                    BEST_MAPS = best_maps
                    break
            
            global converged
            if GEV==0:
                converged = 0
                print("not converged")
            else:
                converged = 1
                print("Best GEV: ", GEV)
                plot_maps(BEST_MAPS, EEG_INFO)
            
            return BEST_MAPS
        
    
    N_STATES = 5
    RAND = np.random.rand()
    if RAND < 0.5:
        METHOD = 'K-MEANS'
    else:
        METHOD = 'X-MEANS'
    best_maps = do_clustering(METHOD, int(N_STATES*2))
    
    print(" *** CONVERGED *** ")
    if converged:
        counter = counter + 1
        micro_labels = ["1", "2", "3", "4", "5"]
        ion()
        for i in range(N_STATES):
            fig = plt.figure()
            mne.viz.plot_topomap(best_maps[i,:], EEG_INFO, sensors=False)
            fig.savefig(save_folder+micro_labels[i]+'_'+str(counter)+'.png', bbox_inches='tight')
            fig.show()


#%%


'''
print("\n")
micro_labels = list(str(label) for label in input("enter list of labels ").strip().split())[:N_STATES]
print("Labels: ", micro_labels)

ion()
for i in range(N_STATES):
    fig = plt.figure()
    mne.viz.plot_topomap(best_maps[i,:], EEG_INFO, sensors=False)
    fig.savefig(save_folder+micro_labels[i]+'/'+micro_labels[i]+'_'+str(counter)+'.png', bbox_inches='tight')
    fig.show()
'''
        
