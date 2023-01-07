#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:54:12 2021

@author: amin
"""

import os
import h5py
import numpy as np
import collections
import pickle
from functions.utils.load_data import load_eegs
from scipy.signal import butter, filtfilt


def filter_eeg(data, fs, lowcut, highcut, order):
    nyq = 0.5 * fs
    l_freq = lowcut / nyq
    h_freq = highcut / nyq
    b, a = butter(order, [l_freq, h_freq], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def preprocess_eegs(eegfile, list_eegs, eeg_format, datatype, filter_true,
                    filtermethod, lowcut, highcut, downsample_true,
                    fs, chan2rm):
    channels2remove = ['']
    # temp # remove     if datatype == "epoched": channels2remove[0] = ['CB1', 'CB2']
    if chan2rm == 'missing':
        # Missing channels to remove
        print("\nchecking channel names ...\n")
        for file in range(len(list_eegs)):
            filename = list_eegs[file]
            # Load the eeg data
            eeg = load_eegs(filename, eeg_format, datatype, channels2remove)
            if file == 0:
                channels = eeg.info['ch_names']
            else:
                channels = np.append(channels, eeg.info['ch_names'])
        counter = collections.Counter(channels)
        counter = np.array(list(counter.items()))
        channels2remove = counter[np.where(counter[:, 1].astype(float) < len(list_eegs)), 0].tolist()
    else:
        channels2remove[0] = chan2rm.split(",")
    #print("Channels to remove: ", channels2remove[0])

    # for file in range(len(list_eegs)):
    # print(100*(file+1)/len(list_eegs))
    # eegfile = list_eegs[file]
    # print('\nLoading eeg Files ... ', filename)
    print(eegfile)
    # Load the eeg data
    eeg = load_eegs(eegfile, eeg_format, datatype, channels2remove[0])

    # Filter
    if filter_true:
        eeg = eeg.filter(l_freq=lowcut, h_freq=highcut,
                         method=filtermethod)
    # Downsample
    if downsample_true:
        if eeg.info['sfreq'] != fs:
            eeg = eeg.resample(fs)

    if datatype == "epoched":
        for index in range(eeg.__len__()):
            if index == 0:
                eeg_data = np.squeeze(eeg[0].get_data())
            else:
                epoch = np.squeeze(eeg[index].get_data())
                eeg_data = np.append(eeg_data, epoch, axis=1)
    else:
        eeg_data = eeg.get_data()
    length_data = eeg_data.shape[1]

    eeg_info = eeg.info
    progress = 100 * (list_eegs.index(eegfile) + 1) / len(list_eegs)
    return progress, eeg_data, length_data, eeg_info, channels2remove

