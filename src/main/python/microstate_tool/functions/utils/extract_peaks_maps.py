
import numpy as np
from scipy.signal import find_peaks
from functions.utils.gfp_func import gfp_func


def extract_peaks_maps(data, min_dist):
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
