"""
Last Modified: April 18th, 2023
Description: This file defines functions for EEG feature extraction and clustering using microstate analysis.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
###
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
import numpy as np
from scipy.stats import zscore
from mne.utils import logger, verbose


@verbose
def segment(data, peaks, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
            normalize=True, min_peak_dist=4, max_n_peaks=10000,
            return_polarity=False, random_state=None, verbose=None):
    logger.info('Finding %d microstates, using %d random intitializations' %
                (n_states, n_inits))

    # if normalize:
    # for i in range(data.shape[0]):
    #    data[i,:] = np.convolve(data[i,:], min_peak_dist, mode='same')
    # from scipy.ndimage.filters import gaussian_filter1d
    # data = gaussian_filter1d(data, sigma=3)
    # data = zscore(data, axis=1)

    # Find peaks in the global field power (GFP)
    gfp = np.std(data, axis=0)
    n_peaks = len(peaks)

    # Limit the number of peaks by randomly selecting them
    if max_n_peaks is not None:
        max_n_peaks = min(n_peaks, max_n_peaks)
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        chosen_peaks = random_state.choice(n_peaks, size=max_n_peaks,
                                           replace=False)
        peaks = peaks[chosen_peaks]

    # Cache this value for later
    gfp_sum_sq = np.sum(gfp ** 2)

    # Do several runs of the k-means algorithm, keep track of the best
    # segmentation.
    best_gev = 0
    best_maps = None
    best_segmentation = None
    best_polarity = None

    for _ in range(n_inits):
        maps, residual = _mod_kmeans(data[:, peaks], n_states, n_inits, max_iter, thresh,
                                     random_state, verbose)
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        map_corr = _corr_vectors(data, maps[segmentation].T)

        # Compare across iterations using global explained variance (GEV) of
        # the found microstates.
        gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
        logger.info('GEV of found microstates: %f' % gev)
        if gev > best_gev:
            best_residual, best_gev, best_maps = residual, gev, maps
            # best_polarity = np.sign(np.choose(segmentation, activation))

    if return_polarity:
        return best_maps, best_polarity, best_gev
    else:
        return best_maps, best_gev, best_residual


@verbose
def _mod_kmeans(data, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
                random_state=None, verbose=None):
    """The modified K-means clustering algorithm.

    See :func:`segment` for the meaning of the parameters and return
    values.
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    init_times = random_state.choice(n_samples, size=n_states, replace=False)
    maps = data[:, init_times].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)  # Normalize the maps

    prev_residual = np.inf
    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = (segmentation == state)
            if np.sum(idx) == 0:
                warnings.warn('Some microstates are never activated')
                maps[state] = 0
                continue

            # Find largest eigenvector
            # cov = data[:, idx].dot(data[:, idx].T)
            # _, vec = eigh(cov, eigvals=(n_channels - 1, n_channels - 1))
            # maps[state] = vec.ravel()
            maps[state] = data[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))

        # Have we converged?
        if (prev_residual - residual) < (thresh * residual):
            logger.info('Converged at %d iterations.' % iteration)
            break

        prev_residual = residual
    else:
        warnings.warn('Modified K-means algorithm failed to converge.')

    return maps, prev_residual


def _corr_vectors(A, B, axis=0):
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.sum(An * Bn, axis=axis)
###
###
import h5py
filename = "C://Users//amin_//Documents//GitHub//output_test//test_study//preprocessed_data//sub-010005_EC.h5"
f = h5py.File(filename, 'r')
dataset = f["sub-010005_EC"]
data = np.asarray(dataset[:])
###

def optimal_clusters(data, Cmin, Cmax):
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp)
    maps = data[:, peaks].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)

    N, RES, GEV = [], [], []
    for C in range(Cmin, Cmax + 1):
        print(C)
        maps, gev, residual = segment(data=data,
                                                      peaks=peaks,
                                                      n_states=C,
                                                      n_inits=1,
                                                      max_n_peaks=None)

        N = np.append(N, C)
        RES = np.append(RES, residual)
        GEV = np.append(GEV, gev)

        activation = np.array(maps).dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        #intra_cluster_distance =
        #nearest_cluster_distance =

    # Silhouette

optimal_clusters(data, 2, 12)