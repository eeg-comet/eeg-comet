"""
Last Modified: April 18th, 2023
Description: This file provides functions for loading and saving EEG information using pickle.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import pickle


def save_eeg_info(eeg_info_path, eeg_info):
    """
    Saves EEG information using pickle.

    Inputs:
        eeg_info_path (string): Path to the file where the EEG information should be saved.
        eeg_info (object): The EEG information to save.

    Outputs:
        None
    """
    with open(eeg_info_path, 'wb') as f:
        pickle.dump(eeg_info, f)


def load_eeg_info(eeg_info_path):
    """
    Loads EEG information using pickle.

    Inputs:
        eeg_info_path (string): Path to the file containing the EEG information.

    Outputs:
        eeg_info (object): The EEG information loaded from the file.
    """
    with open(eeg_info_path, 'rb') as f:
        eeg_info = pickle.load(f)
    return eeg_info
