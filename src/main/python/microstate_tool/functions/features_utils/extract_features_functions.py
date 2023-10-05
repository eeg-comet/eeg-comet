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

from functions.data_utils.data_io import find_data, load_eegs, get_eeg_data, save_features
from functions.features_utils.feature_extractor import FeatureExtractor

def save_segmentation_results(filenames, len_data, sampling_rate, segmentation, file_format, save_path):
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
        time = np.arange(0, (1000 / sampling_rate) * len(segment_each), (1000 / sampling_rate))
        segmentation_df = pd.DataFrame({'time': time, 'segmentation': segment_each})
        filename = os.path.splitext(os.path.basename(filenames[i]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_features(segmentation_df, 'raw_segmentation_' + filename[0], file_format,
                      save_path)


def save_raw_results(segmentation_folder, sampling_rate, maps, micro_labels,
                     output_path, file_format, save_path):
    list_segmentations, _ = find_data(segmentation_folder, file_format)
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


def extract_features(preprocessed_data_path, hf_segmentation_path, maps, micro_labels, sampling_rate, features, min_length,
                     window_size, extension, datatype, mode='static'):

    list_eeg_path, list_eeg_names = find_data(preprocessed_data_path, extension)
    hf_segmentation = h5py.File(hf_segmentation_path, 'r')
    study_name = list(hf_segmentation.keys())[0]
    extracted_features_df = pd.DataFrame()
    sampling_rate = int(sampling_rate)
    for eeg_path in list_eeg_path:
        extracted_features, headers = [], []
        lst_dict = []
        headers = np.append(headers, "Filename")
        eeg = load_eegs(eeg_path, extension, datatype)
        data = get_eeg_data(eeg, datatype)

        filename = os.path.split(filename)[1].split('.')[0]
        print("\nExtracting Features", filename)
        extracted_features = np.append(extracted_features, filename)
        segment = np.asarray(hf_segmentation[study_name][filename][:])
        segment = [" ".join(item) for item in segment.astype('U10')]
        #segment = np.asarray(segment)

        output_format = '.csv'
        feature_extractor = FeatureExtractor(segment, sampling_rate, window_size, mode)

        if "COV" in features:
            extracted_coverages = feature_extractor.microstate_coverage()
            feature_extractor.export_results(extracted_coverages, format=output_format, filename=filename+'_coverage_static')
        if "OCC" in features:
            extracted_occurrences = feature_extractor.microstate_occurrence()
            feature_extractor.export_results(extracted_occurrences, format=output_format, filename=filename+'_occurrence_static')
        if "DUR" in features:
            extracted_durations = feature_extractor.microstate_duration()
            feature_extractor.export_results(extracted_durations, format=output_format, filename=filename+'_duration_static')
        if "TP" in features:
            extracted_probabilities = feature_extractor.transition_probability()
            feature_extractor.export_results(extracted_probabilities, format=output_format, filename=filename+'_transition_probability_static')
        if "LZC" in features:
            extracted_complexity = feature_extractor.lempel_ziv_complexity()
            feature_extractor.export_results(extracted_complexity, format=output_format, filename=filename+'_complexity_static')