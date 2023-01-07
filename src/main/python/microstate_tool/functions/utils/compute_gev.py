
import numpy as np
from functions.utils.corr_vectors import corr_vectors

def compute_gev(data, maps):
    gfp = np.std(data, axis=0)
    gfp_sum_sq = np.sum(gfp ** 2)
    if maps.ndim == 1:
        maps /= np.linalg.norm(maps, keepdims=True)
        maps = np.reshape(maps, (1,-1))
    else:
        maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    activation = np.array(maps).dot(data)
    segmentation = np.argmax(np.abs(activation), axis=0)
    map_corr = corr_vectors(data, maps[segmentation].T)
    gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
    return gev