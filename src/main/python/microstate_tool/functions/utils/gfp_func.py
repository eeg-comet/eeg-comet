"""
Last Modified: April 18th, 2023
Description: Calculates the Global Field Potential (GFP) of a given EEG data.

Inputs:
    data (2D array of floats): EEG data
        The EEG data to calculate the GFP from. Each row corresponds to a channel, and each column corresponds to a time sample.

Outputs:
    gfp (1D array of floats): Global Field Potential (GFP)
        The GFP values for each time sample.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import numpy as np

def gfp_func(data):
    # Global Field Potential (GFP)
    mse = ((data - data.mean(axis=0)) ** 2).mean(axis=0)
    # Return square root of mean square error
    return np.sqrt(mse)
