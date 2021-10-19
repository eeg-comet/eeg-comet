#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:54:12 2021

@author: amin
"""

import os
import numpy as np
import mne
import collections
import h5py
from scipy.signal import butter, lfilter, resample


def filter_eeg(data, fs, lowcut, highcut, order):
    nyq = 0.5 * fs
    l_freq = lowcut / nyq
    h_freq = highcut / nyq
    b, a = butter(order, [l_freq, h_freq], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data

def chan2rm_eeg(list_eegs):
    # Find EEG channels that all files have
    for file in range(len(list_eegs)):
        print(100*file/len(list_eegs))
        filename = list_eegs[file]
        # Load the example MNE data
        EEG = mne.io.read_raw_eeglab(filename, preload=True, verbose='CRITICAL')
        # Select EEG channels from the dataset
        EEG = EEG.pick_types(meg=False, eeg=True, eog=False, verbose='CRITICAL')
        if file == 0:
            channels = EEG.info['ch_names']
        else:
            channels = np.append(channels,EEG.info['ch_names'])
    counter = collections.Counter(channels)
    counter = np.array(list(counter.items()))
    channels2remove = counter[np.where(counter[:,1].astype(float) < len(list_eegs)),0].tolist()
    return channels2remove

def preprocess_eeg(filename, channels2remove, filter_bool, lowcut, highcut, order,
                        downsample_bool, fs):
    #for file in range(len(list_eegs)):
    #print(100*file/len(list_eegs))
    #filename = list_eegs[file]
    print('\nLoading EEG Files ... ', filename)
    # Load the example MNE data
    EEG = mne.io.read_raw_eeglab(filename, preload=True, verbose='CRITICAL')
    # Select EEG channels from the dataset    
    EEG = EEG.pick_types(meg=False, eeg=True, eog=False,
                         exclude=channels2remove[0], verbose='CRITICAL')
    EEG = EEG.set_eeg_reference('average')
    
    #n_channels = EEG.info['nchan']
    #INFO = EEG.info
    
    DATA = EEG[:,:][0]        
    if filter_bool:
        DATA = filter_eeg(DATA, fs, lowcut, highcut, order)
        
    if downsample_bool:
        if EEG.info['sfreq'] != fs:
            n_samples = round(len(DATA)*float(fs)/EEG.info['sfreq'])
            DATA = resample(DATA, n_samples)
    return DATA

def save_preprocessed_eeg(data, filename, save_folder):
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]
    with h5py.File(save_folder+'/'+name+'.h5','w') as f:
        f.create_dataset(name, data=data)

