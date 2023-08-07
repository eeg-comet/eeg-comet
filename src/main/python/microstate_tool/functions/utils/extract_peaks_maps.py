"""
Last Modified: April 18th, 2023
Description: This script provides functions for extracting GFP (Global Field Power) peaks,
and generating microstate maps for clustering.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import numpy as np
from scipy.signal import find_peaks
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from functions.utils.gfp_func import gfp_func
from functions.utils.data_io import find_data, load_eegs, get_eeg_data


def initialize_cluster_centers(maps, n_states, initializer):
    """
    Initialize cluster centers for k-means clustering algorithm.

    Args:
        maps (2D numpy array): Microstate maps for clustering.
        n_states (int): Number of states for clustering.
        initializer (str): Cluster initialization method ('Random' or 'K-Means++').

    Returns:
        initial_centers (2D numpy array): Initial cluster centers.
    """
    if initializer == 'Random':
        random_state = np.random.RandomState(None)
        initial_peaks = random_state.choice(np.size(maps, 1), size=n_states, replace=False)
        initial_centers = maps[:, initial_peaks].T
    elif initializer == 'K-Means++':
        initial_centers = kmeans_plusplus_initializer(maps, n_states).initialize()
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
    gfp = gfp_func(data)

    if use_percentages is not None:
        num_samples = int(data.shape[1] * (use_percentages / 100))
        peaks = np.random.choice(data.shape[1], size=num_samples, replace=False)
        maps = data[:, peaks]
    else:
        peaks, _ = find_peaks(gfp, distance=min_dist)
        maps = data[:, peaks]

    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    return maps, peaks


def generate_maps_and_peaks(preprocessed_folder, extension, datatype, use_percentages, min_dist):
    """
    Generate GFP maps and peak indices from preprocessed EEG data.

    Args:
        preprocessed_folder (str): Path to the folder containing preprocessed EEG data files.
        extension (str): File extension of the EEG data files.
        datatype (str): Data type of the EEG data (e.g., 'float32').
        use_percentages (int or None): If provided, specify the percentage of time points to use for clustering.
        min_dist (int or None): Minimum distance between peaks.

    Returns:
        maps2use (numpy array): Concatenated GFP maps at peaks (peaks x channels).
        peaks2use (numpy array): Concatenated indices of GFP peaks in the EEG data.
    """
    print('Generating Maps and Peaks...')

    all_preprocessed_paths = find_data(preprocessed_folder, extension)
    maps2use, peaks2use = [], []
    counter = 0
    for eeg_path in all_preprocessed_paths:
        eeg = load_eegs(eeg_path, extension, datatype)
        eeg_data = get_eeg_data(eeg, datatype)
        maps, peaks = extract_gfp_peaks_and_maps(eeg_data, use_percentages, min_dist)

        if counter == 0:
            maps2use = maps
            peaks2use = peaks
        else:
            maps2use = np.hstack((maps2use, maps))
            peaks2use = np.hstack((peaks2use, peaks))
        counter = counter + 1

    return maps2use, peaks2use

