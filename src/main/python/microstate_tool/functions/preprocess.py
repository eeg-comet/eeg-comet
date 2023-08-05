#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing EEG data

"""

import numpy as np
import collections
from functions.utils.data_io import load_eegs


def preprocess_eegs(eegfile,
                    list_eegs,
                    eeg_format,
                    datatype,
                    channel_location_dir,
                    filter_true,
                    filtermethod,
                    lowcut,
                    highcut,
                    downsample_true,
                    fs,
                    chan2rm):

    channels2remove = ['']
    if chan2rm == 'missing':
        # Missing channels to remove
        # print("\nchecking channel names ...\n")
        for file in range(len(list_eegs)):
            filename = list_eegs[file]
            # Load the eeg data
            eeg = load_eegs(filename, eeg_format, datatype, channel_location_dir, channels2remove)
            if file == 0:
                channels = eeg.info['ch_names']
            else:
                channels = np.append(channels, eeg.info['ch_names'])
        counter = collections.Counter(channels)
        counter = np.array(list(counter.items()))
        channels2remove = counter[np.where(counter[:, 1].astype(float) < len(list_eegs)), 0].tolist()
    else:
        channels2remove[0] = chan2rm.split(",")
    # print("Channels to remove: ", channels2remove[0])

    # print(eegfile)
    # Load the eeg data
    eeg = load_eegs(eegfile, eeg_format, datatype, channel_location_dir, channels2remove[0])

    # Filter
    if filter_true:
        eeg = eeg.filter(l_freq=lowcut, h_freq=highcut,
                         method=filtermethod, n_jobs=-1)
    # Downsample
    sfreq = eeg.info['sfreq']
    if downsample_true:
        if sfreq != fs:
            eeg = eeg.resample(fs)
    else:
        fs = sfreq

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
    return progress, eeg, eeg_data, length_data, eeg_info, channels2remove
