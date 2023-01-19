#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:30:24 2021

@author: amin
"""

import mne


def load_eegs(filename, eeg_format, datatype, chan2rm):
    
    if datatype == 'continuous':
        # Load the eeg file
        if eeg_format == ".vhdr":
            eeg = mne.io.read_raw_brainvision(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".edf":
            eeg = mne.io.read_raw_edf(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".bdf":
            eeg = mne.io.read_raw_bdf(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".gdf":
            eeg = mne.io.read_raw_gdf(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".cnt":
            eeg = mne.io.read_raw_cnt(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".egi" or eeg_format == ".mff":
            eeg = mne.io.read_raw_egi(filename, preload=True, verbose='CRITICAL')            
        elif eeg_format == ".set":
            eeg = mne.io.read_raw_eeglab(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".data":
            eeg = mne.io.read_raw_nicolet(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".nxe":
            eeg = mne.io.read_raw_eximia(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".lay":
            eeg = mne.io.read_raw_persyst(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".eeg":
            eeg = mne.io.read_raw_nihon(filename, preload=True, verbose='CRITICAL')
    elif datatype == 'epoched':
        if eeg_format == ".set":
            eeg = mne.io.read_epochs_eeglab(filename, verbose='CRITICAL')

    eeg = eeg.pick_types(meg=False, eeg=True, eog=False,
                         exclude=chan2rm, verbose='CRITICAL')
    eeg = eeg.set_eeg_reference('average')
    return eeg
