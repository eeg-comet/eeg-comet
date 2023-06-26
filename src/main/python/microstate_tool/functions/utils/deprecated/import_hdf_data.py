"""
Last Modified: April 18th, 2023
Description: Imports and preprocesses HDF5 data.

Inputs:
    hdf_data_path (string): Path to the HDF5 data file.

Outputs:
    dset (2D array of floats): Preprocessed data
        The preprocessed data from the HDF5 file. Each row corresponds to a channel, and each column corresponds to a time sample.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import h5py
import numpy as np
from scipy.stats import zscore


def import_hdf_data(hdf_data_path):
    # Open HDF5 file in read-only mode
    hf = h5py.File(hdf_data_path, "r")

    # Extract the study name from the HDF5 file
    study_name = list(hf.keys())[0]

    # Get the dataset from the HDF5 file and convert it to a NumPy array
    dset = hf[study_name]
    dset = np.asarray(dset)

    # Apply z-score normalization to the data along the time axis
    dset = zscore(dset, axis=1)

    # Close the HDF5 file
    hf.close()

    # Return the preprocessed data
    return dset
