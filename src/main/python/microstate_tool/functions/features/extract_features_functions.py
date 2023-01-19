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

from functions.utils.find_data import find_data
from functions.utils.compute_gev import compute_gev
from functions.utils.save_features import save_features
from functions.utils.remove_consecutive_duplicates import remove_consecutive_duplicates
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


def extract_features(hf_data_path, hf_segmentation_path, maps, micro_labels, fs, features, min_length):
    # Load data and segment files
    hf_data = h5py.File(hf_data_path, 'r')
    hf_segmentation = h5py.File(hf_segmentation_path, 'r')
    study_name = list(hf_data.keys())[0]
    file_names = list(hf_data[study_name].keys())

    extracted_features_df = pd.DataFrame()

    for filename in file_names:
        extracted_features, headers = [], []
        lst_dict = []
        headers = np.append(headers, "Filename")
        print(study_name, filename)
        extracted_features = np.append(extracted_features, filename)
        data = np.asarray(hf_data[study_name][filename][:])
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
            # Frequency of Occurrence for each map per second
            if "FOC" in features:
                FOC = np.count_nonzero(segment == c) / fs
                headers = np.append(headers, "FOC_" + c)
                extracted_features = np.append(extracted_features, FOC)
            # Mean Microstate Duration
            if "MMD" in features:
                D = [sum(1 for i in g) for k, g in groupby(segment) if k == c]
                MMD = (1000 / fs) * (c if not D else sum(D) / len(D))
                headers = np.append(headers, "MMD_" + c)
                extracted_features = np.append(extracted_features, MMD)
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
            segment_trimmed = segment[:min_length]
            transitioning_sequence = remove_consecutive_duplicates(segment_trimmed)
            LZC = lempel_ziv_complexity(transitioning_sequence) / min_length
            headers = np.append(headers, "LZC")
            extracted_features = np.append(extracted_features, LZC)

        lst_dict.append(dict(zip(headers.tolist(), extracted_features.tolist())))
        extracted_features_df = extracted_features_df.append(lst_dict)

    return extracted_features_df
