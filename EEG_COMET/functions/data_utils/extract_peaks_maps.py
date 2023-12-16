"""
Description: This script provides functions for extracting GFP (Global Field Power) peaks,
and generating microstate maps for clustering_utils.

"""

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import correlate
from functions.data_utils.data_io import DataIO


def initialize_cluster_centers(maps2use, n_states, initializer):
    """
    Initialize cluster centers for k-means clustering algorithm.
    """

    # Initialize with random states
    if initializer == 'Random':
        random_state = np.random.RandomState(None)
        initial_peaks = random_state.choice(np.size(maps2use, 1), size=n_states, replace=False)
        initial_centers = maps2use[:, initial_peaks].T

    # Initialize with K-Means++
    elif initializer == 'K-Means++':
        initial_centers = []
        initial_idx = np.random.choice(np.size(maps2use, 1))
        initial_centers.append(maps2use[:, initial_idx])
        for _ in range(1, n_states):
            dists = np.array([
                min([abs(correlate(d, c, mode='valid')[0]) for c in initial_centers])
                for d in maps2use.T  # Transposed to iterate over columns
            ])
            probs = dists / dists.sum()
            next_idx = np.random.choice(np.size(maps2use, 1), p=probs)
            next_centroid = maps2use[:, next_idx]
            initial_centers.append(next_centroid)
        initial_centers = np.array(initial_centers)

    # Normalize the centroids
    initial_centers /= np.linalg.norm(initial_centers, axis=1, keepdims=True)

    return initial_centers


def extract_gfp_peaks_and_maps(data, use_percentages=None, min_dist=None):
    """
    Extract GFP peaks and maps at peaks from EEG data.
    """
    # Global Field Potential (GFP)
    gfp = np.std(data, axis=0)

    if use_percentages is not None:
        num_samples = int(data.shape[1] * (int(use_percentages) / 100))
        peaks = np.random.choice(data.shape[1], size=num_samples, replace=False)
        maps = data[:, peaks]
    else:
        if min_dist == 0:
            min_dist = None
        peaks, _ = find_peaks(gfp, distance=min_dist)
        maps = data[:, peaks]

    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    return maps, peaks


def generate_maps_and_peaks(preprocessed_folder, extension, datatype, use_percentages=None, min_dist=None):
    """
    Generate GFP maps and peak indices from preprocessed EEG data.
    """

    data_io = DataIO()
    all_preprocessed_paths, _ = data_io.find_data(preprocessed_folder, extension)
    maps2use, peaks2use = [], []
    counter = 0
    for eeg_path in all_preprocessed_paths:
        eeg = data_io.load_eegs(eeg_path, extension, datatype)
        eeg_data = data_io.get_eeg_data(eeg, datatype)
        maps, peaks = extract_gfp_peaks_and_maps(eeg_data, use_percentages, min_dist)

        if counter == 0:
            maps2use = maps
            peaks2use = peaks
        else:
            maps2use = np.hstack((maps2use, maps))
            peaks2use = np.hstack((peaks2use, peaks))
        counter = counter + 1

    return maps2use, peaks2use
