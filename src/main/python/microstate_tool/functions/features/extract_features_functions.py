#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:07:42 2021

@author: amin
"""

import os.path
import numpy as np
import pandas as pd
import h5py
import pickle
from itertools import groupby

from functions.utils.data_io import find_data
from functions.utils.compute_gev import compute_gev
from functions.utils.save_features import save_features
from functions.utils.remove_consecutive_duplicates import remove_consecutive_duplicates

from functions.features.frequency_occurrence import frequency_occurrence
from functions.features.mean_durations import mean_durations
from functions.features.transition_probability import transition_matrix
from functions.features.lempel_ziv_complexity import lempel_ziv_complexity


def save_segmentation_results(filenames, len_data, fs, segmentation, file_format, save_path):
    for i in range(len(len_data)):
        if i == 0:
            start = 0
            stop = int(float(len_data[i]))
            stop_pre = stop
        else:
            start = stop_pre
            stop = stop_pre + int(float(len_data[i]))
            stop_pre = stop
        segment_each = segmentation[start:stop]
        time = np.arange(0, (1000 / fs) * len(segment_each), (1000 / fs))
        segmentation_df = pd.DataFrame({'time': time, 'segmentation': segment_each})
        filename = os.path.splitext(os.path.basename(filenames[i]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_features(segmentation_df, 'raw_segmentation_' + filename[0], file_format,
                      save_path)


def extract_segments(list_h5, segmentation_dir, micro_labels):

    for file in range(len(list_h5)):
        print(100 * file / (len(list_h5)))

        eeg_file_name = os.path.splitext(os.path.basename(list_h5[file]))[0]
        #eeg_file_name = os.path.split(filenames[seg])[1].split(".")[0]
        print(eeg_file_name)

        segment_file = os.path.join(segmentation_dir, 'raw_segmentation_' + eeg_file_name + '.csv')
        segment_data = pd.read_csv(segment_file, header=0)['segmentation']
        segment_data = segment_data.to_numpy()

        data_dir = os.path.join(list_h5[file])
        hf = h5py.File(data_dir, 'r+')
        data = hf[eeg_file_name]
        print(data.shape)

        for micro_map in micro_labels:
            for m in range(len(micro_labels)):
                segment_data = np.char.replace(segment_data, str(m), micro_labels[m])

            #seg_map = list(np.where(segment_data == micro_map)[0])

            name = 'map_' + micro_map
            if name in hf.keys():
                del hf[name]
            hf.create_dataset(name, data=segment_data, compression="gzip", compression_opts=9)

        hf.close()


def save_raw_results(segmentation_folder, fs, maps, micro_labels,
                     output_path, file_format, save_path):
    list_segmentations = find_data(segmentation_folder, file_format, '')
    for s in list_segmentations:
        filename = os.path.splitext(os.path.basename(s))[0]
        filename = filename.split("raw_segmentation_")[1]
        segment_each = pd.read_csv(s)
        segment_each = segment_each.iloc[:, 1:]
        segment_each = segment_each['segmentation'].to_list()
        # Add labels
        for i in range(len(micro_labels)):
            segment_each = np.char.replace(segment_each, str(i), micro_labels[i])

    # Save microstate maps
    for dirpath, dirnames, filenames in os.walk(output_path):
        for filename in [f for f in filenames if f.startswith("EEG_INFO")]:
            eegInfo_path = os.path.join(dirpath, filename)
    with open(eegInfo_path, 'rb') as f:
        eeg_info = pickle.load(f)
    if micro_labels != []:
        maps_df = pd.DataFrame(maps.T, columns=micro_labels, index=eeg_info.ch_names)
    else:
        maps_df = pd.DataFrame(maps.T, index=eeg_info.ch_names)
    save_features(maps_df, 'microstate_maps', file_format, save_path)


def save_transitions(segmentation_folder, micro_labels, file_format, save_path):
    list_segmentations = find_data(segmentation_folder, file_format, '')
    for s in list_segmentations:
        filename = os.path.splitext(os.path.basename(s))[0]
        filename = filename.split("raw_segmentation_")[1]
        segment_each = pd.read_csv(s)
        segment_each = segment_each.iloc[:, 1:]
        segment_each = segment_each['segmentation'].to_list()
        tm = transition_matrix(segment_each)
        headers = []
        for c in micro_labels:
            headers = np.append(headers, c)
        tm_df = pd.DataFrame(tm, columns=headers, index=headers)
        save_features(tm_df, 'transition_matrix_' + filename, file_format, save_path)


def extract_features(preprocessed_data_path, hf_segmentation_path, maps, micro_labels, fs, features, min_length, duration_of_window):
    file_names = find_data(preprocessed_data_path, ".hdf", "*")
    hf_segmentation = h5py.File(hf_segmentation_path, 'r')
    study_name = list(hf_segmentation.keys())[0]
    extracted_features_df = pd.DataFrame()
    fs = int(fs)
    for filename in file_names:
        extracted_features, headers = [], []
        lst_dict = []
        headers = np.append(headers, "Filename")
        hf = h5py.File(filename, "r")
        data = hf[list(hf.keys())[0]]
        data = np.asarray(data)

        filename = os.path.split(filename)[1].split('.')[0]
        print("\nExtracting Features", filename)
        extracted_features = np.append(extracted_features, filename)
        segment = np.asarray(hf_segmentation[study_name][filename][:])
        segment = [" ".join(item) for item in segment.astype('U10')]
        segment = np.asarray(segment)

        for c in micro_labels:
            # Global Explained Variance
            if "GEV" in features:
                GEV = 100 * compute_gev(data, maps[micro_labels.index(c), :])
                headers = np.append(headers, "GEV_" + c)
                extracted_features = np.append(extracted_features, GEV)
            # Coverage of each map per data
            if "COV" in features:
                COV = 100 * np.count_nonzero(segment == c) / len(segment)
                headers = np.append(headers, "COV_" + c)
                extracted_features = np.append(extracted_features, COV)
            # Mean Microstate Duration
            if "MMD" in features:
                MMD_ALL = mean_durations(segment, fs)
                MMD_MAP = MMD_ALL[c]
                headers = np.append(headers, "MMD_" + c)
                extracted_features = np.append(extracted_features, MMD_MAP)
            # Frequency of Occurrence for each map per second
            if "OCC" in features:
                OCC_ALL = frequency_occurrence(segment, fs, duration_of_window=duration_of_window)
                OCC_MAP = OCC_ALL[c]
                headers = np.append(headers, "OCC_" + c)
                extracted_features = np.append(extracted_features, OCC_MAP)
        # Transition Probabilities
        if "TP" in features:
            TP_MATRIX = transition_matrix(segment)
            for row in range(len(micro_labels)):
                for col in range(len(micro_labels)):
                    if not row == col:
                        TP = TP_MATRIX[row, col]
                        headers = np.append(headers, "TP_" + micro_labels[row] + micro_labels[col])
                        extracted_features = np.append(extracted_features, TP)
        # Lempel-Ziv Complexity
        if "LZC" in features:
            #segment_trimmed = segment[:min_length]
            #transitioning_sequence = remove_consecutive_duplicates(segment_trimmed)
            LZC = lempel_ziv_complexity(segment) # / min_length
            headers = np.append(headers, "LZC")
            extracted_features = np.append(extracted_features, LZC)

        lst_dict.append(dict(zip(headers.tolist(), extracted_features.tolist())))
        extracted_features_df = extracted_features_df.append(lst_dict)

    return extracted_features_df


def extract_dynamic_features(preprocessed_data_path, hf_segmentation_path, fs, window_second, overlap, micro_labels):
    """
    Slide a window of size window_size over the data with overlap between windows
    and compute the mean of each window
    Parameters:
    data (numpy.ndarray): Input data vector
    window_size (int): Size of the sliding window
    overlap (float): Overlap between windows as a fraction of window_size
    Returns:
    numpy.ndarray: Array containing the means of each window
    """

    # Convert window_second to window_size
    window_size = int(window_second * fs)

    # Find data
    file_names = find_data(preprocessed_data_path, ".hdf", "*")
    hf_segmentation = h5py.File(hf_segmentation_path, 'r')
    study_name = list(hf_segmentation.keys())[0]
    for filename in file_names:
        filename = os.path.split(filename)[1].split('.')[0]
        print("\nExtracting Features", filename)

        # Load segmented data
        segment = np.asarray(hf_segmentation[study_name][filename][:])
        segment = [" ".join(item) for item in segment.astype('U10')]
        segment = np.asarray(segment)

        # Calculate the number of windows
        num_windows = int(np.ceil((len(segment) - window_size) / (window_size * (1 - overlap))) + 1)

        # Initialize the output array
        window_occ = np.zeros((num_windows, len(micro_labels)))
        window_dur = np.zeros((num_windows, len(micro_labels)))

        # Slide the window over the data and compute the mean of each window
        for c in range(len(micro_labels)):
            for i in range(num_windows):
                start_index = int(i * window_size * (1 - overlap))
                end_index = start_index + window_size
                window_segment = segment[start_index:end_index]
                window_segment_no_duplicates = remove_consecutive_duplicates(np.asarray(window_segment))
                window_segment_no_duplicates = np.asarray(list(window_segment_no_duplicates))

                D = [sum(1 for i in g) for k, g in groupby(window_segment) if k == micro_labels[c]]
                window_dur[i, c] = (1000 / fs) * (micro_labels[c] if not D else sum(D) / len(D))
                window_occ[i, c] = np.count_nonzero(window_segment_no_duplicates == micro_labels[c])
                
    return window_occ, window_dur

