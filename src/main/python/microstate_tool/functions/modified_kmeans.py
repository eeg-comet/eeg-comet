

import numpy as np
import h5py
import os.path
from collections import Counter
from functions.utils.data_io import find_data
from functions.utils.compute_gev import compute_gev
from functions.utils.map_initializer import map_initializer
from functions.utils.extract_peaks_maps import extract_peaks_maps


def modified_kmeans(data, initial_maps, n_states, thresh):


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

    while residual > thresh:
        # Assign each sample to the best matching microstate
        activation = np.dot(maps, data) ** 2
        segmentation = np.argmax(activation, axis=0)

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


def segmentation_smooth(data, maps, n_states, epsilon=1e-6, b=3, lamb=5):
    '''
    data: V
    maps: Gamma (T-like symbol)
    n_states: N_mu in paper
    epsilon: convergence criterion parameter
    b: window size parameter
    lamb: non-smoothness penalty parameter (lambda)
    
    segmentation: L
    n_channels: N_s in paper
    n_samples: N_T

    '''

    n_channels, n_samples = data.shape 
    data_sum_sq = np.sum(data ** 2)
    iteration = 0
    prev_residual = 0
    residual = np.inf
    thresh = epsilon

    # STEP 2 in TABLE 2
    # V dot Gamma
    activation = maps.dot(data)
    # L
    segmentation = np.argmax(np.abs(activation), axis=0) 
    print(f'SEG BEFORE: {segmentation[400:500]}')

    # STEP 3 in TABLE 2
    raw_segmentation = segmentation
    
    # STEP 4 in TABLE 2
    act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
    e1 = abs(data_sum_sq - act_sum_sq)
    e2 = e1 / float(n_samples * (n_channels - 1)) 

    while residual > thresh:
        print(f'iteration: {iteration+1} residual: {residual}, thresh: {thresh}')
        
        # STEP 5 in TABLE 2
        windows = np.lib.stride_tricks.sliding_window_view(raw_segmentation, 2*b+1)
        N_bkt = np.zeros((windows.shape[0], n_states))
        for i, window in enumerate(windows):
            cnt = Counter(window)
            N_bkt[i] = [cnt[x] for x in range(n_states)]
        raw_segmentation[b:n_samples-b] = np.argmin((np.sum(data**2, axis=0) - (np.sum(maps[segmentation].T * data, axis=0) ** 2))[b:n_samples-b] / (2*e2*(n_channels - 1)) - (lamb*N_bkt).T, axis=0)

        # STEP 6 in TABLE 2
        segmentation = raw_segmentation# .copy()
        
        # STEP 7 in TABLE 2
        act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        e1 = abs(data_sum_sq - act_sum_sq)
        sigma_mu = (e1 / float(n_samples * (n_channels - 1)))
        residual = abs(prev_residual - sigma_mu)

        # STEP 8 in TABLE 2
        prev_residual = sigma_mu
        thresh = epsilon * sigma_mu
        iteration += 1

    print(f'SEG AFTER : {segmentation[400:500]}')
    print('Finishes after', str(iteration), 'Iterations.')
    
    # STEP 9 in TABLE 2
    # WARN: seems like we didn't use the result of this step
    
    # STEP 10 in TABLE 2
    # sigma_d_squared = data_sum_sq / float(n_samples * (n_channels - 1))
    # R_squared = 1 - sigma_mu/sigma_d_squared
    
    # return segmentation

def run_modified_kmeans(data, min_dist, n_states, thresh, n_inits, initializer):
    # Extract peaks and maps from the data
    all_maps, peaks = extract_peaks_maps(data, min_dist)
    # print(all_maps.shape)
    # print(peaks.shape)
    # print(data.shape)
    # Initialize variables to store the best results
    best_residual = None
    best_gev = 0
    best_maps = None

    # Perform clustering for multiple initializations
    for init in range(n_inits):
        print('\nClustering #', init + 1, 'of', n_inits)

        # Initialize microstate maps using the specified initializer
        initial_maps = map_initializer(data, all_maps, peaks, n_states, initializer)

        # Run modified k-means algorithm
        maps, residual = modified_kmeans(data, initial_maps, n_states, thresh)

        # Compute GEV (Global Explained Variance)
        gev = compute_gev(data, maps)

        print('Found', n_states, 'Microstate Maps')
        print('GEV:', gev)

        # Update the best results if the current GEV is higher
        if gev > best_gev:
            best_residual, best_gev, best_maps = residual, gev, maps

    print('\nBest GEV:', best_gev)
    return best_maps, best_gev, best_residual


def run_minibatch_modified_kmeans(preprocessed_data_path, min_dist, n_states, thresh, n_inits, initializer):
    file_names = find_data(preprocessed_data_path, ".hdf", "*")
    counter = 1
    best_residual, best_gev, best_maps = None, 0, None
    for filename in file_names:
        hf = h5py.File(filename, "r")
        dataset_k = hf[list(hf.keys())[0]]
        dataset_k = np.asarray(dataset_k)
        filename = os.path.split(filename)[1].split('.')[0]
        print('\n', filename)
        if counter == 1:
            all_maps, peaks = extract_peaks_maps(dataset_k, min_dist)
            initial_maps = map_initializer(dataset_k, all_maps, peaks, n_states, initializer)
        else:
            initial_maps = best_maps
        for init in range(n_inits):
            print('Mini Batch Clustering #', str(init + 1), 'of', str(n_inits))
            maps, residual = modified_kmeans(dataset_k, initial_maps, n_states, thresh)
            gev = compute_gev(dataset_k, maps)
            print('Found', str(n_states), 'Microstate Maps')
            print('GEV:', str(gev))
            if gev > best_gev:
                best_residual, best_gev, best_maps = residual, gev, maps
            counter += 1
    return best_maps, best_gev, best_residual
