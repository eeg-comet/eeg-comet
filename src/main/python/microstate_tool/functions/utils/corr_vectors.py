"""
Last Modified: April 18th, 2023
Description: This file contains a function that computes the correlation between two matrices along a specified axis.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import numpy as np


def corr_vectors(A, B, axis=0):
    """
    Computes the correlation between two matrices along a specified axis.

    Inputs:
        A (numpy array): The first matrix.
        B (numpy array): The second matrix.
        axis (int): The axis along which to compute the correlation. Defaults to 0.

    Outputs:
        corr (numpy array): The correlation between the two matrices along the specified axis.
    """
    An = A - np.mean(A, axis=axis)  # Subtract the mean along the specified axis
    Bn = B - np.mean(B, axis=axis)  # Subtract the mean along the specified axis
    An /= np.linalg.norm(An, axis=axis)  # Normalize the matrix along the specified axis
    Bn /= np.linalg.norm(Bn, axis=axis)  # Normalize the matrix along the specified axis
    corr = np.sum(An * Bn, axis=axis)  # Compute the correlation between the two matrices along the specified axis
    return corr
