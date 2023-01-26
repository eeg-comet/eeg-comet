
import h5py
import numpy as np
from scipy.stats import zscore


def import_hdf_data(hdf_data_path):
    hf = h5py.File(hdf_data_path, "r")
    study_name = list(hf.keys())[0]
    dset = hf[study_name]
    dset = np.asarray(dset)
    dset = zscore(dset, axis=1)
    return dset
