#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Modified: April 18th, 2023
Description: Loads and preprocesses EEG data from a file.

Inputs:
    filename (string): Name of the file containing the EEG data.
    eeg_format (string): File format of the EEG data.
    datatype (string): Type of EEG data, either 'continuous' or 'epoched'.
    channel_location_dir (string): Directory containing the channel location file for the EEG data.
    chan2rm (list of strings): Channels to be removed from the EEG data.

Outputs:
    eeg (MNE Raw or Epochs object): Preprocessed EEG data
        The preprocessed EEG data loaded from the file.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import mne


def load_eegs(filename, eeg_format, datatype, channel_location_dir, chan2rm):
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

    # Load the eeg channel location
    if channel_location_dir:
        montage = mne.channels.read_custom_montage(channel_location_dir)
        eeg.set_montage(montage)

    # Pick channels
    eeg = eeg.pick_types(meg=False, eeg=True, eog=False,
                         exclude=chan2rm, verbose='CRITICAL')

    # Apply an average reference
    eeg = eeg.set_eeg_reference('average')
    return eeg
