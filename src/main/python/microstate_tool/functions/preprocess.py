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
from functions.load_data import load_eegs
from scipy.signal import butter, lfilter


def filter_eeg(data, fs, lowcut, highcut, order):
    nyq = 0.5 * fs
    l_freq = lowcut / nyq
    h_freq = highcut / nyq
    b, a = butter(order, [l_freq, h_freq], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data


def preprocess_eegs(eegfile, list_eegs, eeg_format, datatype, filter_true,
                    filtermethod, lowcut, highcut, downsample_true,
                    fs, save_folder):
    channels2remove = ['']
    # temp # remove
    if datatype == "epoched":
        channels2remove[0] = ['CB1', 'CB2']
    else:
        # Missing channels to remove
        print("\nchecking channel names ...\n")
        for file in range(len(list_eegs)):
            filename = list_eegs[file]
            # Load the EEG data
            EEG = load_eegs(filename, eeg_format, datatype)
            # Select EEG channels from the dataset
            EEG = EEG.pick_types(meg=False, eeg=True, eog=False, verbose='CRITICAL')
            if file == 0:
                channels = EEG.info['ch_names']
            else:
                channels = np.append(channels, EEG.info['ch_names'])
        counter = collections.Counter(channels)
        counter = np.array(list(counter.items()))
        channels2remove = counter[np.where(counter[:, 1].astype(float) < len(list_eegs)), 0].tolist()
    print("Channels to remove: ", channels2remove[0])

    # for file in range(len(list_eegs)):
    # print(100*(file+1)/len(list_eegs))
    # eegfile = list_eegs[file]
    # print('\nLoading EEG Files ... ', filename)
    print(eegfile)
    # Load the EEG data
    EEG = load_eegs(eegfile, eeg_format, datatype)
    # Select EEG channels from the dataset
    EEG = EEG.pick_types(meg=False, eeg=True, eog=False,
                         exclude=channels2remove[0], verbose='CRITICAL')
    EEG = EEG.set_eeg_reference('average')

    #print(INFO)

    # Filter
    if filter_true:
        EEG = EEG.filter(l_freq=lowcut, h_freq=highcut,
                         method=filtermethod)
    # else:
    # EEG = mne.filter.filter_data(eeg_data, Fs,
    #                             l_freq=lowcut, h_freq=highcut,
    #                             method=filtermethod)
    # DATA = filter_eeg(DATA, fs, lowcut, highcut, order)
    # Downsample
    if downsample_true:
        if EEG.info['sfreq'] != fs:
            EEG = EEG.resample(fs)
            sample_rate = fs
            # if datatype == "epoched":
            #    EEG = EEG.resample(fs)
            # else:
            #    downsamp_factor = round(float(Fs)/fs)
            #    EEG = mne.filter.resample(EEG, down=downsamp_factor ,npad='auto')
            # n_samples = round(len(DATA)*float(fs)/EEG.info['sfreq'])
            # DATA = resample(DATA, n_samples)

    INFO = EEG.info
    #Fs = INFO['sfreq']
    #INFO['highpass'] = lowcut
    #INFO['lowpass'] = highcut

    if datatype == "epoched":
        for index in range(EEG.__len__()):
            if index == 0:
                eeg_data = np.squeeze(EEG[0].get_data())
            else:
                epoch = np.squeeze(EEG[index].get_data())
                eeg_data = np.append(eeg_data, epoch, axis=1)
        # eeg_data = eeg_data.reshape(eeg_data.shape[1],
        #                            eeg_data.shape[0]*eeg_data.shape[2])
    else:
        eeg_data = EEG.get_data()

    length_data = eeg_data.shape[1]

    # DATA = EEG[:,:][0]
    # Save data
    name = os.path.basename(eegfile)
    name = os.path.splitext(name)[0]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with h5py.File(os.path.join(save_folder, name + '.h5'), 'w') as f:
        f.create_dataset(name, data=eeg_data)

    # Save EEG info
    with open(os.path.join(save_folder, 'EEG_INFO.pickle'), 'wb') as p:
        pickle.dump(INFO, p)
    progress = 100 * (list_eegs.index(eegfile) + 1) / len(list_eegs)
    return progress, length_data


