#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Preprocess EEG data by applying filtering, downsampling, and channel removal.


"""

import numpy as np
import collections
from functions.utils.data_io import load_eegs


def preprocess_eegs(eegfile, list_eegs, eeg_format, datatype, channel_location_dir,
                    filter_true, filtermethod, lowcut, highcut, downsample_true, fs, chan2rm):

    """
    Preprocess EEG data according to specified parameters.

    Args:
        eegfile (str): Path to the EEG file to preprocess.
        list_eegs (list): List of EEG filenames.
        eeg_format (str): Format of EEG data (e.g., 'edf', 'fif').
        datatype (str): Data type, ('raw' or 'epoched').
        channel_location_dir (str): Directory containing channel location information.
        filter_true (bool): Whether to apply filtering.
        filtermethod (str): Filtering method ('iir' or 'fir').
        lowcut (float): Lowcut frequency for filtering.
        highcut (float): Highcut frequency for filtering.
        downsample_true (bool): Whether to apply downsampling.
        fs (float): Target sampling frequency after downsampling.
        chan2rm (str): List of channels to remove or 'missing' for automatic removal.

    Returns:
        tuple: A tuple containing progress percentage, preprocessed EEG object, EEG data, data length,
               EEG info, and channels to remove.
    """

    verbose = 'WARNING'
    channels2remove = ['']

    if chan2rm == 'missing':
        # Automatic detection of missing channels to remove
        for file in range(len(list_eegs)):
            filename = list_eegs[file]
            # Load the EEG data
            eeg = load_eegs(filename, eeg_format, datatype, channel_location_dir, channels2remove)
            if file == 0:
                channels = eeg.info['ch_names']
            else:
                channels = np.append(channels, eeg.info['ch_names'])
        # Count channel occurrences
        counter = collections.Counter(channels)
        counter = np.array(list(counter.items()))
        channels2remove = counter[np.where(counter[:, 1].astype(float) < len(list_eegs)), 0].tolist()
    else:
        channels2remove[0] = chan2rm

    # Load the EEG data
    eeg = load_eegs(eegfile, eeg_format, datatype, channel_location_dir, channels2remove[0])

    # Apply filtering
    if filter_true:
        eeg = eeg.filter(l_freq=lowcut, h_freq=highcut, method=filtermethod, n_jobs=-1, verbose=verbose)

    # Apply downsampling
    sfreq = eeg.info['sfreq']
    if downsample_true:
        if sfreq != fs:
            eeg = eeg.resample(fs, verbose=verbose)

    # Combine epoched data
    if datatype == "epoched":
        for index in range(eeg.__len__()):
            if index == 0:
                eeg_data = np.squeeze(eeg[0].get_data())
            else:
                epoch = np.squeeze(eeg[index].get_data())
                eeg_data = np.append(eeg_data, epoch, axis=1)
    else:
        eeg_data = eeg.get_data()

    # Get data length and EEG info
    length_data = eeg_data.shape[1]
    eeg_info = eeg.info

    # Calculate progress
    progress = 100 * (list_eegs.index(eegfile) + 1) / len(list_eegs)

    return progress, eeg, eeg_data, length_data, eeg_info, channels2remove
