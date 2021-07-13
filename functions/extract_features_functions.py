#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:07:42 2021

@author: amin
"""

import os
import numpy as np
import pandas as pd
from itertools import groupby


def extract_features(filenames, len_data, segmentation, fs, features):
    if segmentation is not None:
        extracted_features_df = pd.DataFrame()
        for i in range(len(len_data)):
            if i==0:
                start = 0
                stop = len_data[i]
                stop_pre = stop
            else:
                start = stop_pre
                stop = stop_pre+len_data[i]
                stop_pre = stop
            segment_each = segmentation[start:stop]
            
            extracted_features, headers = [], []
            headers = np.append(headers, "Filename")
            extracted_features = np.append(extracted_features, filenames[i])
            for c in np.unique(segmentation).tolist():
                if "FOC" in features:
                    # Frequency of Occurence for each map per second
                    FOC = 100*np.count_nonzero(segment_each == c)/len(segment_each)
                    headers = np.append(headers, "FOC_"+c)
                    extracted_features = np.append(extracted_features, FOC)
                if "MMD" in features:
                    # Mean Microstates Duration
                    D = [sum(1 for i in g) for k,g in groupby(segment_each) if k==c]
                    MMD = (1000/fs) * (c if not D else sum(D) / len(D))
                    headers = np.append(headers, "MMD_"+c)
                    extracted_features = np.append(extracted_features, MMD)
            extracted_features_df = extracted_features_df.append(pd.DataFrame(extracted_features.reshape(1,len(extracted_features)),
                                                 columns=headers.tolist()))
    return extracted_features_df

def save_features(extracted_features_df, save_path):
    save_name = os.path.join(save_path, 'extracted_features.csv')
    extracted_features_df.to_csv(save_name, index=False, header=True)