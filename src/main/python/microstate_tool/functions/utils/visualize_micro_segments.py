"""
Last Modified: April 18th, 2023
Description: This file defines a function for estimating the functional connectivity matrix from EEG data using various methods.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

from dyconnmap.fc.plv import plv
from dyconnmap.fc.coherence import coherence
from dyconnmap.fc.corr import corr
from dyconnmap.fc.cos import cos
from dyconnmap.fc.icoherence import icoherence
from dyconnmap.fc.iplv import iplv
from dyconnmap.fc.pli import pli


def estimate_fc_mat(eeg_data, method, channel_names):
    """
    Estimates the functional connectivity matrix from EEG data using various methods.

    Inputs:
        eeg_data (numpy array): The EEG data to estimate functional connectivity from.
        method (string): The method to use for estimating functional connectivity. Can be 'COH', 'CORR', 'COS', 'ICOH', 'PLV', 'IPLV', or 'PLI'.
        channel_names (list of strings): The names of the EEG channels.

    Outputs:
        CONN_MAT (numpy array): The estimated functional connectivity matrix.
    """
    fs = 250  # The sampling rate of the EEG data
    fb = [2, 20]  # The frequency band of interest

    if method == 'COH':
        ### Coherence
        CONN_MAT = coherence(eeg_data, fb, fs)
    elif method == 'CORR':
        ### Correlation
        CONN_MAT = corr(eeg_data, fb, fs)
    elif method == 'COS':
        ### Cosine
        CONN_MAT = cos(eeg_data, fb, fs)
    elif method == 'ICOH':
        ### Imaginary Coherence
        CONN_MAT = icoherence(eeg_data, fb, fs)
    elif method == 'PLV':
        ### Phase Locking Value
        CONN_MAT = plv(eeg_data, fb, fs)[1]
    elif method == 'IPLV':
        ### Imaginary part of Phase Locking Value
        CONN_MAT = iplv(eeg_data, fb, fs)[1]
    elif method == 'PLI':
        ### Phase Lag Index
        CONN_MAT = pli(eeg_data, fb, fs)[1]
    else:
        raise ValueError("Failed to match method")

    CONN_MAT = CONN_MAT + CONN_MAT.T  # Add the transpose to make the matrix symmetric
    return CONN_MAT
