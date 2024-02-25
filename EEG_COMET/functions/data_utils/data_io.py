"""
Description: EEG Data Loading and Preprocessing Script

This script provides functions for loading and preprocessing EEG data from different formats using the MNE library.
It supports loading raw or epoched data, applying channel location information, removing channels, and applying average
reference projection. The script is designed to enhance flexibility and efficiency when working with EEG data.
"""

import os.path
import pickle
from fnmatch import fnmatch
import numpy as np
import mne


class DataIO:
    def __init__(self):
        pass

    @staticmethod
    def save_eeg_info(eeg_info_path, eeg_info):
        """
        Save EEG information to a binary file using pickle.
        """
    
        with open(eeg_info_path, 'wb') as f:
            pickle.dump(eeg_info, f)  # Serialize and save the eeg_info object to the specified file path

    @staticmethod
    def load_eeg_info(eeg_info_path):
        """
        Load EEG information from a binary file using pickle.
        """
    
        with open(eeg_info_path, 'rb') as f:
            eeg_info = pickle.load(f)  # Deserialize and load the eeg_info object from the specified file path
        return eeg_info

    @staticmethod
    def find_data(input_folder, extension, pattern='*'):
        """
        Recursively search for data files within a folder based on extension and pattern.
        """
    
        list_path = []  # Initialize an empty list to store matching file paths
        list_filename = []
    
        # Walk through the directory tree rooted at input_folder
        for path, subdirs, files in os.walk(input_folder):
            for name in files:
                # Check if the file name matches the pattern with the specified extension
                if fnmatch(name, pattern + extension):
                    # Add the matching file's full path to the list
                    list_path.append(os.path.join(path, name))
                    list_filename.append(name.split('.')[0])
        return list_path, list_filename

    @staticmethod
    def load_eegs(eeg_path, eeg_format, datatype, channel_location_dir='', chan2rm=[]):
        """
        Load EEG data from different formats and preprocess if needed.
        """
        verbose = 'ERROR'

        if datatype == 'raw':
    
            # Load EEG data based on the specified format
            if eeg_format == ".vhdr":
                eeg = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".edf":
                eeg = mne.io.read_raw_edf(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".bdf":
                eeg = mne.io.read_raw_bdf(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".gdf":
                eeg = mne.io.read_raw_gdf(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".cnt":
                eeg = mne.io.read_raw_cnt(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".egi" or eeg_format == ".mff":
                eeg = mne.io.read_raw_egi(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".set":
                eeg = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".data":
                eeg = mne.io.read_raw_nicolet(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".nxe":
                eeg = mne.io.read_raw_eximia(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".lay":
                eeg = mne.io.read_raw_persyst(eeg_path, preload=True, verbose=verbose)
            elif eeg_format == ".eeg":
                eeg = mne.io.read_raw_nihon(eeg_path, preload=True, verbose=verbose)
        elif datatype == 'epoched':
            if eeg_format == ".set":
                eeg = mne.io.read_epochs_eeglab(eeg_path, verbose=verbose)
    
        # Optional: Load channel locations if provided
        try:
            if os.path.isfile(channel_location_dir):
                montage = mne.channels.read_custom_montage(channel_location_dir)
                eeg.set_montage(montage, match_case=False, on_missing='warn', verbose=verbose)
            elif channel_location_dir in mne.channels.get_builtin_montages():
                montage = mne.channels.make_standard_montage(channel_location_dir)
                eeg.set_montage(montage, match_case=False, on_missing='warn', verbose=verbose)
        except Exception as e:
            print('Invalid File, Try Loading Again. Details:' + e)
    
        # Drop channels
        channel_names = eeg.info['ch_names']
        if any(chan2rm) and not all(elem == '' for elem in chan2rm) and chan2rm in channel_names:
            eeg = eeg.drop_channels(chan2rm)
        if 'TRIGGER' in channel_names:
            eeg = eeg.drop_channels('TRIGGER')

        # Add average reference projection
        eeg.set_eeg_reference('average', projection=True, verbose=verbose)
        # Apply the added projection
        eeg.apply_proj(verbose=verbose)
    
        return eeg

    @staticmethod
    def export_eegs(eeg, save_path, extension, datatype):
        """
        Export EEG data to a specified file format.
        """

        # List of available file extensions for export
        available_extensions = ['.vhdr', '.set', '.edf']
    
        # Check if the provided extension is valid, otherwise default to '.set'
        if extension not in available_extensions:
            extension = '.set'
    
        # Export EEG data based on the specified datatype
        if datatype == 'raw':
            mne.export.export_raw(save_path + extension, eeg, fmt='auto', overwrite=True)
        elif datatype == 'epoched':
            mne.export.export_epochs(save_path + extension, eeg, fmt='auto', overwrite=True)

    @staticmethod
    def get_eeg_data(eeg, datatype):
        """
        Extract EEG data from MNE-Python Epochs or Raw object.
        """
    
        if datatype == "epoched":
            # Concatenate the epoch data along the specified axis
            for index in range(eeg.__len__()):
                if index == 0:
                    eeg_data = np.squeeze(eeg[0].get_data())
                else:
                    epoch = np.squeeze(eeg[index].get_data())
                    eeg_data = np.append(eeg_data, epoch, axis=1)
        else:
            eeg_data = eeg.get_data()  # Get continuous raw data
    
        return eeg_data
    