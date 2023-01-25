

import numpy as np
import h5py
import os.path
from functions.utils.compute_gev import compute_gev
from functions.utils.map_initializer import map_initializer
from functions.utils.extract_peaks_maps import extract_peaks_maps


def modified_kmeans(data, initial_maps, n_states, thresh):
    n_channels, n_samples = data.shape
    maps = initial_maps
    data_sum_sq = np.sum(data ** 2)
    iteration = 1
    prev_residual = np.inf
    residual = prev_residual
    while residual > thresh:
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        for state in range(n_states):
            idx = (segmentation == state)
            maps[state] = data[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])
        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))
        if (prev_residual - residual) < (thresh * residual):
            print('Converged at', str(iteration), 'Iterations.')
            break
        prev_residual = residual
        iteration += iteration
    return maps, prev_residual


def run_modified_kmeans(data, min_dist, n_states, thresh, n_inits, initializer):
    all_maps, peaks = extract_peaks_maps(data, min_dist)
    best_residual = None
    best_gev = 0
    best_maps = None
    for init in range(n_inits):
        print('\nClustering #', str(init+1), 'of', str(n_inits))
        initial_maps = map_initializer(data, all_maps, peaks, n_states, initializer)
        maps, residual = modified_kmeans(data, initial_maps, n_states, thresh)
        gev = compute_gev(data, maps)
        print('Found', str(n_states), 'Microstate Maps')
        print('GEV:', str(gev))
        if gev > best_gev:
            best_residual, best_gev, best_maps = residual, gev, maps
    print('\nBest GEV:', str(best_gev))
    return best_maps, best_gev, best_residual


def run_minibatch_modified_kmeans(study_name, outputfolder, min_dist, n_states, thresh, n_inits, initializer):
    preprocessed_data_path = os.path.join(outputfolder, study_name + '_data.hdf')
    hf = h5py.File(preprocessed_data_path, 'r')
    dset = hf[study_name]
    counter = 1
    best_residual = None
    best_gev = 0
    best_maps = None
    for k in list(dset.keys()):
        print(k)
        print(counter)
        dataset_k = list(dset[k])
        dataset_k = np.asarray(dataset_k)
        if counter == 1:
            all_maps, peaks = extract_peaks_maps(dataset_k, min_dist)
            initial_maps = map_initializer(dataset_k, all_maps, peaks, n_states, initializer)
        else:
            initial_maps = best_maps
        for init in range(n_inits):
            print('\nClustering #', str(init + 1), 'of', str(n_inits))
            maps, residual = modified_kmeans(dataset_k, initial_maps, n_states, thresh)
            gev = compute_gev(dataset_k, maps)
            print('Found', str(n_states), 'Microstate Maps')
            print('GEV:', str(gev))
            if gev > best_gev:
                best_residual, best_gev, best_maps = residual, gev, maps
            counter += 1

    return best_maps, best_gev, best_residual
