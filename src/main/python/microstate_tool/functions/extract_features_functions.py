#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:07:42 2021

@author: amin
"""

import os
import numpy as np
import pandas as pd
import h5py
import pickle
from itertools import groupby
import matplotlib.pyplot as plt
from functions.clustering_functions import compute_gev
from functions.lempel_ziv_complexity import lempel_ziv_complexity


def save_features(df, filename, file_format, path):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, filename + file_format)
    if file_format == '.csv':
        df.to_csv(save_path, header=True)
    elif file_format == '.pkl':
        df.to_pickle(save_path)
    elif file_format == '.hdf':
        df.to_hdf(save_path)
    elif file_format == '.json':
        df.to_json(save_path)


def transition_matrix(segmentation, visualize=False, colormap='Blues'):
    list_segment_each_tmp = listToString(segmentation)
    list_unique_segment_each = remove_consec_duplicates(list_segment_each_tmp)
    list_unique_segment_each = [(list_unique_segment_each[i:i + 1]) for i in range(0, len(list_unique_segment_each), 1)]
    segmentation = np.asarray(list_unique_segment_each)
    df = pd.DataFrame(segmentation)
    df['shift'] = df[0].shift(-1)
    df['count'] = 1
    trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
    labels = list(trans_mat.columns.levels[1])
    trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values
    np.fill_diagonal(trans_mat, 0)
    if visualize == True:
        plt.matshow(trans_mat, cmap=colormap)
        x_pos = np.arange(len(labels))
        plt.xticks(x_pos, labels)
        y_pos = np.arange(len(labels))
        plt.yticks(y_pos, labels)
        plt.colorbar()
        plt.show()
    return trans_mat

# remove consecutive duplicates from string
def remove_consec_duplicates(s):
    new_s = ""
    prev = ""
    for c in s:
        if len(new_s) == 0:
            new_s += c
            prev = c
        if c == prev:
            continue
        else:
            new_s += c
            prev = c
    return new_s

def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1

def save_raw_results(filenames, len_data, fs, save_segmentation, segmentation,
                     save_maps, maps, micro_labels,
                     save_transitions, output_path, file_format, save_path):
    micro_maps = np.unique(segmentation).tolist()
    if segmentation is not None:
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
            if save_segmentation:
                save_path_raw_segmentation = os.path.join(save_path, 'raw_segmentation')
                if not os.path.exists(save_path_raw_segmentation):
                    os.makedirs(save_path_raw_segmentation)
                save_features(segmentation_df, 'raw_segmentation_' + filename[0], file_format,
                              save_path_raw_segmentation)

            if save_transitions:
                save_path_raw_transitions = os.path.join(save_path, 'raw_transitions')
                if not os.path.exists(save_path_raw_transitions):
                    os.makedirs(save_path_raw_transitions)
                tm = transition_matrix(segment_each)
                headers = []
                for c in micro_maps:
                    headers = np.append(headers, c)
                tm_df = pd.DataFrame(tm, columns=headers, index=headers)
                save_features(tm_df, 'transition_matrix_' + filename[0], file_format,
                              save_path_raw_transitions)

    if save_maps:
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


def extract_features(filenames, len_data, h5files, segmentation, maps, fs, features):
    if segmentation is not None:

        len_data = [int(i) for i in len_data]
        extracted_features_df = pd.DataFrame()
        if "LZC" in features:
            for i in range(len(len_data)):
                if i == 0:
                    start = 0
                    stop = len_data[i]
                    stop_pre = stop
                else:
                    start = stop_pre
                    stop = stop_pre + len_data[i]
                    stop_pre = stop
                segment_each_tmp = segmentation[start:stop]

                list_segment_each_tmp = listToString(segment_each_tmp)
                list_unique_segment_each = remove_consec_duplicates(list_segment_each_tmp)
                # number of transitioning sequence
                transitioning_sequence = len(list_unique_segment_each)
                if i == 0:
                    min_transitioning_sequence = transitioning_sequence
                else:
                    if transitioning_sequence < min_transitioning_sequence:
                        min_transitioning_sequence = transitioning_sequence

        for i in range(len(len_data)):
            if i == 0:
                start = 0
                stop = len_data[i]
                stop_pre = stop
            else:
                start = stop_pre
                stop = stop_pre + len_data[i]
                stop_pre = stop
            segment_each = segmentation[start:stop]

            extracted_features, headers = [], []
            headers = np.append(headers, "Filename")
            filename = os.path.splitext(os.path.basename(filenames[i]))
            extracted_features = np.append(extracted_features, filename[0])

            micro_maps = np.unique(segmentation).tolist()
            for c in micro_maps:
                if "COV" in features:
                    # Coverage of each map per data
                    COV = 100 * np.count_nonzero(segment_each == c) / len(segment_each)
                    headers = np.append(headers, "COV_" + c)
                    extracted_features = np.append(extracted_features, COV)
                if "FOC" in features:
                    # Frequency of Occurrence for each map per second
                    FOC = np.count_nonzero(segment_each == c) / fs
                    headers = np.append(headers, "FOC_" + c)
                    extracted_features = np.append(extracted_features, FOC)
                if "MMD" in features:
                    # Mean Microstates Duration
                    D = [sum(1 for i in g) for k, g in groupby(segment_each) if k == c]
                    MMD = (1000 / fs) * (c if not D else sum(D) / len(D))
                    headers = np.append(headers, "MMD_" + c)
                    extracted_features = np.append(extracted_features, MMD)

            if "TP" in features:
                # Transition Probabilities
                TP_MATRIX = transition_matrix(segment_each)
                for row in range(len(micro_maps)):
                    for col in range(len(micro_maps)):
                        if not row == col:
                            TP = 100 * TP_MATRIX[row, col]
                            headers = np.append(headers, "TP_" + micro_maps[row] + micro_maps[col])
                            extracted_features = np.append(extracted_features, TP)

            if "GEV" in features:
                with h5py.File(h5files[i], "r") as f:
                    a_group_key = list(f.keys())[0]
                    data = list(f[a_group_key])
                data = np.asarray(data)
                for c in micro_maps:
                    # Global Explained Variance
                    GEV = 100 * compute_gev(data, maps[micro_maps.index(c), :])
                    headers = np.append(headers, "GEV_" + c)
                    extracted_features = np.append(extracted_features, GEV)

            if "LZC" in features:
                list_segment_each = listToString(segment_each)
                transitioning_sequence = remove_consec_duplicates(list_segment_each)
                LZC = lempel_ziv_complexity(transitioning_sequence[:min(len_data)])
                headers = np.append(headers, "LZC")
                extracted_features = np.append(extracted_features, LZC)

            extracted_features_df = extracted_features_df.append(
                pd.DataFrame(extracted_features.reshape(1, len(extracted_features)),
                             columns=headers.tolist()))

    return extracted_features_df
