"""
Description: EEG Data Loading and Preprocessing Script

This script provides functions for loading and preprocessing EEG data from different formats using the MNE-Python library.
It supports loading raw or epoched data, applying channel location information, removing channels, and applying average
reference projection. The script is designed to enhance flexibility and efficiency when working with EEG data.
"""

import os.path
import pickle
from fnmatch import fnmatch
import numpy as np
import mne


def save_eeg_info(eeg_info_path, eeg_info):
    """
    Save EEG information to a binary file using pickle.

    Parameters:
    eeg_info_path (str): The file path where the EEG information will be saved.
    eeg_info (mne.Info): The EEG information obtained from raw.info or epoched.info.

    Returns:
    None

    Explanation:
    This function saves EEG-related information (stored in the mne.Info object) to a binary file using the pickle module.
    Pickle is a Python module used for serializing and deserializing objects, making it suitable for saving complex
    data structures.

    Example usage:
    >>> raw = mne.io.read_raw_fif('raw_data.fif')
    >>> eeg_info = raw.info  # Get EEG information from raw object
    >>> save_eeg_info('/path/to/eeg_info.pkl', eeg_info)
    """

    with open(eeg_info_path, 'wb') as f:
        pickle.dump(eeg_info, f)  # Serialize and save the eeg_info object to the specified file path


def load_eeg_info(eeg_info_path):
    """
    Load EEG information from a binary file using pickle.

    Parameters:
    eeg_info_path (str): The file path from which EEG information will be loaded.

    Returns:
    eeg_info (mne.Info): The loaded EEG information obtained from the saved pickle file.

    Explanation:
    This function loads EEG-related information from a binary file that was previously saved using the
    save_eeg_info function. It uses the pickle module to deserialize and retrieve the mne.Info object.

    Example usage:
    >>> loaded_info = load_eeg_info('/path/to/eeg_info.pkl')
    >>> print(loaded_info['subject_info'])
    # Access various EEG-related properties from the loaded mne.Info object
    """

    with open(eeg_info_path, 'rb') as f:
        eeg_info = pickle.load(f)  # Deserialize and load the eeg_info object from the specified file path
    return eeg_info


def find_data(input_folder, extension, pattern='*'):
    """
    Recursively search for data files within a folder based on extension and pattern.

    Parameters:
    input_folder (str): Path to the folder where data files are located.
    extension (str): File extension to search for (e.g., ".hdf").
    pattern (str): Pattern to match in file names (e.g., "*").

    Returns:
    list_data (list): A list of file paths matching the search criteria.

    Explanation:
    This function searches for files within the specified folder and its subdirectories
    that have the specified file extension and match the provided pattern in their names.
    It returns a list of file paths that match the search criteria.

    Example usage:
    >>> files = find_data('/path/to/data_folder', '.hdf', 'subject_*')
    >>> for file in files:
    >>>     print(file)
    '/path/to/data_folder/subject_001.hdf'
    '/path/to/data_folder/subject_002.hdf'
    ...
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
                list_filename.append(name[0].split('.')[0])
    return list_path, list_filename


def load_eegs(filename, eeg_format, datatype, channel_location_dir='', chan2rm=[]):
    """
    Load EEG data from different formats and preprocess if needed.

    Parameters:
    filename (str): The path to the EEG data file.
    eeg_format (str): The format of the EEG data file (e.g., '.edf', '.fif', '.set').
    datatype (str): The type of EEG data ('raw' for continuous data or 'epoched' for segmented data).
    channel_location_dir (str): The directory path for channel location information (optional).
    chan2rm (list): List of channel names to be removed (optional).

    Returns:
    eeg_data (mne.Raw or mne.Epochs): The loaded and optionally preprocessed EEG data.

    Explanation:
    This function loads EEG data from different file formats using the MNE-Python library. It provides flexibility
    in choosing the EEG data format and handling different data types (raw or epoched). Additionally, it supports
    optional preprocessing steps such as removing specific channels.

    Example usage:
    >>> eeg_data = load_eegs('eeg_data.edf', 'edf', 'raw')
    >>> eeg_data = load_eegs('eeg_data.fif', 'fif', 'epoched', channel_location_dir='channel_locs/')
    """

    verbose = 'WARNING'
    if datatype == 'raw':

        # Load EEG data based on the specified format
        if eeg_format == ".vhdr":
            eeg = mne.io.read_raw_brainvision(filename, preload=True, verbose=verbose)
        elif eeg_format == ".edf":
            eeg = mne.io.read_raw_edf(filename, preload=True, verbose=verbose)
        elif eeg_format == ".bdf":
            eeg = mne.io.read_raw_bdf(filename, preload=True, verbose=verbose)
        elif eeg_format == ".gdf":
            eeg = mne.io.read_raw_gdf(filename, preload=True, verbose=verbose)
        elif eeg_format == ".cnt":
            eeg = mne.io.read_raw_cnt(filename, preload=True, verbose=verbose)
        elif eeg_format == ".egi" or eeg_format == ".mff":
            eeg = mne.io.read_raw_egi(filename, preload=True, verbose=verbose)
        elif eeg_format == ".set":
            eeg = mne.io.read_raw_eeglab(filename, preload=True, verbose=verbose)
        elif eeg_format == ".data":
            eeg = mne.io.read_raw_nicolet(filename, preload=True, verbose=verbose)
        elif eeg_format == ".nxe":
            eeg = mne.io.read_raw_eximia(filename, preload=True, verbose=verbose)
        elif eeg_format == ".lay":
            eeg = mne.io.read_raw_persyst(filename, preload=True, verbose=verbose)
        elif eeg_format == ".eeg":
            eeg = mne.io.read_raw_nihon(filename, preload=True, verbose=verbose)
    elif datatype == 'epoched':
        if eeg_format == ".set":
            eeg = mne.io.read_epochs_eeglab(filename, verbose=verbose)

    # Optional: Load channel locations if provided
    if os.path.isfile(channel_location_dir):
        montage = mne.channels.read_custom_montage(channel_location_dir)
        eeg.set_montage(montage, match_case=False, on_missing='warn', verbose=verbose)
    elif channel_location_dir in mne.channels.get_builtin_montages():
        montage = mne.channels.make_standard_montage(channel_location_dir)
        eeg.set_montage(montage, match_case=False, on_missing='warn', verbose=verbose)

    # Pick channels
    eeg = eeg.pick_types(meg=False, eeg=True, eog=False,
                         exclude=chan2rm, verbose=verbose)

    # Add average reference projection
    eeg.set_eeg_reference('average', projection=True, verbose=verbose)

    # Apply the added projection
    eeg.apply_proj(verbose=verbose)

    return eeg


def export_eegs(eeg, save_path, extension, datatype):
    """
    Export EEG data to a specified file format.

    Parameters:
    eeg (mne.Raw or mne.Epochs): The EEG data to be exported.
    save_path (str): The directory path where the EEG data will be saved.
    extension (str): The desired file extension for the exported file (e.g., '.vhdr', '.set', '.edf').
    datatype (str): The type of EEG data ('raw' for continuous data or 'epoched' for segmented data).

    Returns:
    None

    Explanation:
    This function exports EEG data to a specified file format using the MNE-Python library. It provides flexibility
    in choosing the file extension and supports both continuous (raw) and segmented (epoched) EEG data.

    Example usage:
    >>> export_eegs(raw_data, '/path/to/save/', '.set', 'raw')
    >>> export_eegs(epoched_data, '/path/to/save/', '.vhdr', 'epoched')
    """

    print('*'*20, 'save')  # Print a line of asterisks for visual separation

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


def get_eeg_data(eeg, datatype):
    """
    Extract EEG data from an MNE-Python Epochs or Raw object.

    Parameters:
    eeg (mne.Epochs or mne.Raw): The EEG data source, either epochs or continuous raw data.
    datatype (str): The type of EEG data ('epoched' for segmented data or any other value for continuous data).

    Returns:
    eeg_data (numpy.ndarray): Extracted EEG data as a NumPy array.

    Explanation:
    This function extracts EEG data from an MNE-Python Epochs or Raw object and returns it as a NumPy array.
    For epoched data, it concatenates the individual epoch data along the specified axis (default: axis=1).
    For continuous raw data, it simply returns the data.

    Example usage:
    >>> epochs = mne.Epochs(...)  # An MNE-Python Epochs object
    >>> raw = mne.Raw(...)  # An MNE-Python Raw object
    >>> extracted_data = get_eeg_data(epochs, "epoched")
    >>> continuous_data = get_eeg_data(raw, "raw")
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


