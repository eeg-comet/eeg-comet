"""
Last Modified: April 18th, 2023
Description: This script provides functions for extracting GFP (Global Field Power) peaks,
and generating microstate maps for clustering_utils.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import correlate
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from functions.data_utils.data_io import DataIO


def initialize_cluster_centers(maps2use, n_states, initializer):
    """
    Initialize cluster centers for k-means clustering algorithm.

    Args:
        maps2use (2D numpy array): Microstate maps for clustering.
        n_states (int): Number of states for clustering.
        initializer (str): Cluster initialization method ('Random' or 'K-Means++').

    Returns:
        initial_centers (2D numpy array): Initial cluster centers.
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

    Args:
        data (numpy array): 2D array of EEG data (channels x samples).
        use_percentages (int or None, optional): If provided, randomly select the specified percentage of time points. Default is None.
        min_dist (int or None, optional): Minimum distance between peaks.

    Returns:
        maps (numpy array): 2D array of GFP maps at peaks (peaks x channels).
        peaks (numpy array or None): 1D array of GFP peak indices in the EEG data, or None if `use_percentages` is provided.
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

    Args:
        preprocessed_folder (str): Path to the folder containing preprocessed EEG data files.
        extension (str): File extension of the EEG data files.
        datatype (str): Data type of the EEG data (e.g., 'float32').
        use_percentages (int or None): If provided, specify the percentage of time points to use for clustering_utils.
        min_dist (int or None): Minimum distance between peaks.

    Returns:
        maps2use (numpy array): Concatenated GFP maps at peaks (peaks x channels).
        peaks2use (numpy array): Concatenated indices of GFP peaks in the EEG data.
    """
    #print('Generating Maps and Peaks...')

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

