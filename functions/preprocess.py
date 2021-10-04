#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:54:12 2021

@author: amin
"""

import mne
import os
import h5py
import numpy as np
import collections
import pickle
from functions.load_data import load_eegs
from scipy.signal import butter, lfilter, resample


def filter_eeg(data, fs, lowcut, highcut, order):
    nyq = 0.5 * fs
    l_freq = lowcut / nyq
    h_freq = highcut / nyq
    b, a = butter(order, [l_freq, h_freq], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data

def preprocess_eegs(eegfile, list_eegs, eeg_format, filter_bool, filtmethod,
                    lowcut, highcut, downsample_bool, fs, save_folder):

    # Missing channels to remove
    print("\nchecking channel names ...\n")
    for file in range(len(list_eegs)):
        filename = list_eegs[file]
        # Load the EEG data
        EEG = load_eegs(filename, eeg_format)
        # Select EEG channels from the dataset
        EEG = EEG.pick_types(meg=False, eeg=True, eog=False, verbose='CRITICAL')
        if file == 0:
            channels = EEG.info['ch_names']
        else:
            channels = np.append(channels, EEG.info['ch_names'])
    counter = collections.Counter(channels)
    counter = np.array(list(counter.items()))
    channels2remove = counter[np.where(counter[:,1].astype(float) < len(list_eegs)),0].tolist()
    print("Channels to remove: ", channels2remove[0])
    
    if filtmethod=='FIR':
        filtermethod = 'fir'
    elif filtmethod=='IIR':
        filtermethod = 'iir'
    
    #for file in range(len(list_eegs)):
    #print(100*(file+1)/len(list_eegs))
    #filename = list_eegs[file]
    #print('\nLoading EEG Files ... ', filename)
    # Load the EEG data
    EEG = load_eegs(eegfile, eeg_format)
    # Select EEG channels from the dataset
    EEG = EEG.pick_types(meg=False, eeg=True, eog=False,
                     exclude=channels2remove[0], verbose='CRITICAL')
    EEG = EEG.set_eeg_reference('average')
    INFO = EEG.info
    INFO['highpass'] = lowcut
    INFO['lowpass'] = highcut
    Fs = INFO['sfreq']
    
    # Filter
    if filter_bool:
        EEG = mne.filter.filter_data(EEG.get_data(), Fs,
                                     l_freq=lowcut, h_freq=highcut,
                                     method=filtermethod)
        #DATA = filter_eeg(DATA, fs, lowcut, highcut, order)
    # Downsample
    if downsample_bool:
        if Fs != fs:
            EEG = mne.filter.resample(EEG, down=4 ,npad='auto')
            #n_samples = round(len(DATA)*float(fs)/EEG.info['sfreq'])
            #DATA = resample(DATA, n_samples)
    
    DATA = EEG
    #DATA = EEG[:,:][0] 
    # Save data
    name = os.path.basename(eegfile)
    name = os.path.splitext(name)[0]
    with h5py.File(os.path.join(save_folder, name+'.h5'),'w') as f:
        f.create_dataset(name, data=DATA)
    
    # Save EEG info
    with open(os.path.join(save_folder,'EEG_INFO.pickle'), 'wb') as p:
        pickle.dump(INFO, p)
    progress = 100*(list_eegs.index(eegfile)+1)/len(list_eegs)
    return progress
#def save_preprocessed_eeg(data, filename, save_folder):
#    name = os.path.basename(filename)
#    name = os.path.splitext(name)[0]
#    with h5py.File(save_folder+'/'+name+'.h5','w') as f:
#        f.create_dataset(name, data=data)

