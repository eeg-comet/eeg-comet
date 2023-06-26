"""
Last Modified: April 18th, 2023

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


def map_initializer(data, maps, peaks, n_states, initializer):
    """
    Description: Initializes cluster centers for k-means clustering algorithm.

    Inputs:
        data (2D array of floats): EEG data
            The data to cluster. Each row corresponds to a channel, and each column corresponds to a time sample.
        maps (2D array of floats): Microstate maps
            The microstate maps used for clustering. Each row corresponds to a microstate, and each column corresponds to a time sample.
        peaks (1D array of ints): Peak indices
            The peak indices of the EEG data.
        n_states (int): Number of states
            The number of states to cluster the data into.
        initializer (string): Cluster initialization method
            The cluster initialization method. Supported methods include 'Random' and 'K-Means++'.

    Outputs:
        initial_centers (2D array of floats): Initial cluster centers
            The initial cluster centers.

    """
    # create instance of K-Means algorithm with prepared centers
    if initializer == 'Random':
        random_state = np.random.RandomState(None)
        chosen_peaks = random_state.choice(len(peaks), size=n_states, replace=False)
        initial_peaks = peaks[chosen_peaks].tolist()
        initial_centers = data[:, initial_peaks].T
    elif initializer == 'K-Means++':
        # Calculate initial centers using K-Means++ method.
        initial_centers = kmeans_plusplus_initializer(maps, n_states).initialize()
    initial_centers /= np.linalg.norm(initial_centers, axis=1, keepdims=True)
    return initial_centers
