
"""
This script defines a class called DataPreprocessor for preprocessing EEG data.
The class provides a method preprocess_eegs that applies filtering, downsampling, and channel removal
to EEG data according to specified parameters.

"""

import numpy as np
import collections
from functions.data_utils.data_io import DataIO


class DataPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def preprocess_eegs(eegfile, list_eegs, eeg_format, datatype, channel_location_dir,
                        filter_true, filtermethod, lowcut, highcut, downsample_true, fs, chan2rm):

        verbose = 'ERROR'
        channels2remove = ['']

        if chan2rm == 'missing':
            # Automatic detection of missing channels to remove
            for file in range(len(list_eegs)):
                filename = list_eegs[file]
                # Load the EEG data
                eeg = DataIO().load_eegs(filename, eeg_format, datatype, channel_location_dir, channels2remove)
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
        eeg = DataIO().load_eegs(eegfile, eeg_format, datatype, channel_location_dir, channels2remove[0])

        # Apply filtering
        if filter_true:
            eeg = eeg.filter(
                l_freq=lowcut, h_freq=highcut, method=filtermethod, phase='zero', n_jobs=-1, verbose=verbose)

        # Apply downsampling
        sfreq = eeg.info['sfreq']
        if downsample_true:
            if sfreq != fs:
                eeg = eeg.resample(fs, verbose=verbose)

        # Add average reference projection
        eeg.set_eeg_reference('average', projection=True, verbose=verbose)
        # Apply the added projection
        eeg.apply_proj(verbose=verbose)

        # Get the raw data or combine epoched data
        eeg_data = DataIO.get_eeg_data(eeg, datatype)

        # Get data length and EEG info
        length_data = eeg_data.shape[1]
        eeg_info = eeg.info

        return eeg, eeg_data, length_data, eeg_info, channels2remove
