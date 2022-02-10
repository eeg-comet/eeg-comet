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


def transition_matrix(segmentation, visualize=False, colormap='Blues'):
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


def substitude_maps_with_duration(segmentation, min_duration):
    segmentation = np.array(segmentation)
    count_dups = [sum(1 for _ in group) for _, group in groupby(segmentation)]
    for C in range(len(count_dups)):
        if count_dups[C] <= min_duration:
            start = int(np.sum(count_dups[0:C]))
            stop = int(start + count_dups[C])
            if C == 0:
                segmentation[start:stop] = segmentation[stop + 1]
            else:
                segmentation[start:stop] = segmentation[start - 1]
    return segmentation

def save_raw_results(filenames, len_data, fs, save_segmentation, segmentation,
                     save_maps, maps, micro_labels,
                     save_transitions, output_path, saveformat, save_path):
    micro_maps = np.unique(segmentation).tolist()
    if segmentation is not None:
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
            time = np.arange(0, (1000 / fs) * len(segment_each), (1000 / fs))
            segmentation_df = pd.DataFrame(segment_each,
                                           columns=['segmentation'],
                                           index=time)
            filename = os.path.splitext(os.path.basename(filenames[i]))
            if save_segmentation:
                save_name = os.path.join(save_path, 'raw_segmentation_' + filename[0])
                if saveformat == 'csv':
                    segmentation_df.to_csv(save_name + '.csv')
                elif saveformat == 'pkl':
                    segmentation_df.to_pickle(save_name + '.pkl')
                elif saveformat == 'hdf':
                    segmentation_df.to_hdf(save_name + '.hdf')
                elif saveformat == 'json':
                    segmentation_df.to_json(save_name + '.json')
            if save_transitions:
                save_name = os.path.join(save_path, 'transition_matrix_' + filename[0])
                tm = transition_matrix(segment_each)
                headers = []
                for c in micro_maps:
                    headers = np.append(headers, c)
                tm_df = pd.DataFrame(tm, columns=headers, index=headers)
                if saveformat == 'csv':
                    tm_df.to_csv(save_name + '.csv')
                elif saveformat == 'pkl':
                    tm_df.to_pickle(save_name + '.pkl')
                elif saveformat == 'hdf':
                    tm_df.to_hdf(save_name + '.hdf')
                elif saveformat == 'json':
                    tm_df.to_json(save_name + '.json')

    if save_maps:
        for dirpath, dirnames, filenames in os.walk(output_path):
            for filename in [f for f in filenames if f.startswith("EEG_INFO")]:
                eegInfo_path = os.path.join(dirpath, filename)
        with open(eegInfo_path, 'rb') as f:
            eeg_info = pickle.load(f)
        save_name = os.path.join(save_path, 'microstate_maps')
        if micro_labels != []:
            maps_df = pd.DataFrame(maps.T, columns=micro_labels, index=eeg_info.ch_names)
        else:
            maps_df = pd.DataFrame(maps.T, index=eeg_info.ch_names)
        if saveformat == 'csv':
            maps_df.to_csv(save_name + '.csv')
        elif saveformat == 'pkl':
            maps_df.to_pickle(save_name + '.pkl')
        elif saveformat == 'hdf':
            maps_df.to_hdf(save_name + '.hdf')
        elif saveformat == 'json':
            maps_df.to_json(save_name + '.json')


def extract_features(filenames, len_data, h5files, segmentation, maps, fs, features):
    if segmentation is not None:


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
                LZC = lempel_ziv_complexity(transitioning_sequence[:min_transitioning_sequence])
                headers = np.append(headers, "LZC")
                extracted_features = np.append(extracted_features, LZC)

            extracted_features_df = extracted_features_df.append(
                pd.DataFrame(extracted_features.reshape(1, len(extracted_features)),
                             columns=headers.tolist()))

    return extracted_features_df


def save_features(extracted_features_df, saveformat, save_path):
    if saveformat == 'csv':
        save_name = os.path.join(save_path, 'extracted_features.csv')
        extracted_features_df.to_csv(save_name, index=False, header=True)
    elif saveformat == 'pkl':
        save_name = os.path.join(save_path, 'extracted_features.pkl')
        extracted_features_df.to_pickle(save_name)
    elif saveformat == 'hdf':
        save_name = os.path.join(save_path, 'extracted_features.h5')
        extracted_features_df.to_hdf(save_name)
    elif saveformat == 'json':
        save_name = os.path.join(save_path, 'extracted_features.json')
        extracted_features_df.to_json(save_name)