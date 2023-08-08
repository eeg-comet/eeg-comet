"""
Last Modified: April 18th, 2023
Description: This file provides a function for computing the global explained variance (GEV) of microstate maps.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import numpy as np
from functions.utils.corr_vectors import corr_vectors


def compute_gev(data, maps):
    """
    Computes the global explained variance (GEV) of microstate maps.

    Inputs:
        data (numpy array): The EEG data.
        maps (numpy array): The microstate maps.

    Outputs:
        gev (float): The GEV.
    """
    # Global Field Potential (GFP)
    gfp = np.std(data, axis=0)

    # Normalize the maps
    if maps.ndim == 1:
        maps /= np.linalg.norm(maps, keepdims=True)
        maps = np.reshape(maps, (1, -1))
    else:
        maps /= np.linalg.norm(maps, axis=1, keepdims=True)

    # Compute activation and segmentation
    activation = np.array(maps).dot(data)
    segmentation = np.argmax(np.abs(activation), axis=0)

    # Compute map correlations
    map_corr = corr_vectors(data, maps[segmentation].T)

    # Compute GEV
    gev = sum((gfp * map_corr) ** 2) / np.sum(gfp ** 2)

    return gev
