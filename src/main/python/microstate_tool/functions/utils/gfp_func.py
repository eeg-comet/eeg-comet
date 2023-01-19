
import numpy as np


def gfp_func(data):
    # Global Field Potential (GFP)
    mse = ((data - data.mean(axis=0)) ** 2).mean(axis=0)
    return np.sqrt(mse)
