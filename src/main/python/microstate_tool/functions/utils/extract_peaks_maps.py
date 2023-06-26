"""
Last Modified: April 18th, 2023
Description: This file contains a function that extracts GFP peaks and maps from EEG data.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import numpy as np
from scipy.signal import find_peaks
from functions.utils.gfp_func import gfp_func


def extract_peaks_maps(data, min_dist=None):
    """
    Extracts GFP peaks and maps at peaks from EEG data.

    Inputs:
        data (numpy array): 2D array of EEG data in the format (channels x samples).
        min_dist (int or None, optional): The minimum distance between peaks. If None, the default value is used.

    Outputs:
        maps (numpy array): 2D array of GFP maps at peaks in the format (peaks x channels).
        peaks (numpy array): 1D array of indices of the GFP peaks in the EEG data.
    """
    # Extract GFP Peaks and Maps at Peaks
    gfp = gfp_func(data)
    if min_dist:
        peaks, _ = find_peaks(gfp, distance=min_dist)
    else:
        peaks, _ = find_peaks(gfp)
    # Create Maps
    maps = data[:, peaks].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    return maps, peaks
