#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:30:24 2021

@author: amin
"""

import mne

def load_eegs(filename, eeg_format):
    
    # Load the EEG file
    if eeg_format==".vhdr":
        EEG = mne.io.read_raw_brainvision(filename, preload=True, verbose='CRITICAL')
    elif eeg_format==".edf":
        EEG = mne.io.read_raw_edf(filename, preload=True, verbose='CRITICAL')
    elif eeg_format==".bdf":
        EEG = mne.io.read_raw_bdf(filename, preload=True, verbose='CRITICAL')
    elif eeg_format==".gdf":
        EEG = mne.io.read_raw_gdf(filename, preload=True, verbose='CRITICAL')
    elif eeg_format==".cnt":
        EEG = mne.io.read_raw_cnt(filename, preload=True, verbose='CRITICAL')
    elif eeg_format==".egi" or eeg_format==".mff":
        EEG = mne.io.read_raw_egi(filename, preload=True, verbose='CRITICAL')            
    elif eeg_format==".set":
        EEG = mne.io.read_raw_eeglab(filename, preload=True, verbose='CRITICAL')
    elif eeg_format==".data":
        EEG = mne.io.read_raw_nicolet(filename, preload=True, verbose='CRITICAL')
    elif eeg_format==".nxe":
        EEG = mne.io.read_raw_eximia(filename, preload=True, verbose='CRITICAL')
    elif eeg_format==".lay":
        EEG = mne.io.read_raw_persyst(filename, preload=True, verbose='CRITICAL')
    elif eeg_format==".eeg":
        EEG = mne.io.read_raw_nihon(filename, preload=True, verbose='CRITICAL')
        
    return EEG