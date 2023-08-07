"""
Description: EEG Data Loading and Preprocessing Script

This script provides functions for loading and preprocessing EEG data from different formats using the MNE-Python library.
It supports loading raw or epoched data, applying channel location information, removing channels, and applying average
reference projection. The script is designed to enhance flexibility and efficiency when working with EEG data.
"""

import os.path
import pickle
from configparser import ConfigParser
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

    list_data = []  # Initialize an empty list to store matching file paths

    # Walk through the directory tree rooted at input_folder
    for path, subdirs, files in os.walk(input_folder):
        for name in files:
            # Check if the file name matches the pattern with the specified extension
            if fnmatch(name, pattern + extension):
                # Add the matching file's full path to the list
                list_data.append(os.path.join(path, name))

    return list_data


def load_eegs(filename, eeg_format, datatype, channel_location_dir='', chan2rm=[]):
    """
    Load EEG data from different formats and preprocess if needed.

    Parameters:
    filename (str): The path to the EEG data file.
    eeg_format (str): The format of the EEG data file (e.g., 'edf', 'fif', 'set').
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
    if channel_location_dir:
        montage = mne.channels.read_custom_montage(channel_location_dir)
        eeg.set_montage(montage)

    # Pick channels
    eeg = eeg.pick_types(meg=False, eeg=True, eog=False,
                         exclude=chan2rm, verbose=verbose)

    # Add average reference projection
    eeg.set_eeg_reference('average', projection=True)

    # Apply the added projection
    eeg.apply_proj()

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
    data (numpy.ndarray): Extracted EEG data as a NumPy array.

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
                data = np.squeeze(eeg[0].get_data())
            else:
                epoch = np.squeeze(eeg[index].get_data())
                data = np.append(data, epoch, axis=1)
    else:
        data = eeg.get_data()  # Get continuous raw data

    return data


def initialize_config(config_path, config):
    """
    Initialize and populate a configuration dictionary with default values.

    Parameters:
    config_path (str): The file path where the configuration will be saved.
    config (dict): An empty dictionary where configuration settings will be stored.

    Returns:
    None

    Explanation:
    This function initializes a configuration dictionary with predefined sections and default values.
    It sets up various sections for different aspects of a data processing pipeline and assigns initial values.
    The purpose is to provide a structured way to manage and store settings for data processing steps.

    Example usage:
    >>> config = {}  # An empty dictionary to hold configuration settings
    >>> initialize_config('/path/to/save/config.ini', config)
    """

    # Create sections for different processing aspects
    config['progress'] = {}  # Progress tracking
    config['study info'] = {}  # Study information
    config['preprocessing settings'] = {}  # Preprocessing settings
    config['preprocessing results'] = {}  # Preprocessing results
    config['clustering settings'] = {}  # Clustering settings
    config['clustering results'] = {}  # Clustering results
    config['backfitting settings'] = {}  # Backfitting settings
    config['feature extraction settings'] = {}  # Feature extraction settings
    config['feature visualization groups'] = {}  # Feature visualization groups
    config['source localization settings'] = {}  # Source localization settings

    # Initialize progress indicators with 'False'
    str_false = "False"
    config['progress']['done_preprocessing'] = str_false
    config['progress']['done_clustering'] = str_false
    config['progress']['done_labeling_microstates'] = str_false
    config['progress']['done_backfitting'] = str_false
    config['progress']['done_extracting_features'] = str_false
    config['progress']['done_extracting_microsegments'] = str_false
    config['progress']['done_source_localization'] = str_false

    # Save the initialized configuration to the specified file path
    save_config(config_path, config)



def load_config(config_path):
    """
    Load and parse configuration settings from a file.

    Parameters:
    config_path (str): The path to the configuration file.

    Returns:
    config (ConfigParser): A ConfigParser object containing the loaded configuration settings.

    Explanation:
    This function reads and parses configuration settings from a specified file using the ConfigParser module.
    The ConfigParser object can be used to access and manipulate the loaded configuration values.

    Example usage:
    >>> loaded_config = load_config('/path/to/config.ini')
    >>> value = loaded_config.get('section', 'option')
    """

    config = ConfigParser()  # Create a ConfigParser object
    config.read(config_path)  # Read and parse the configuration file
    return config


def save_config(config_path, config):
    """
    Save configuration settings to a file.

    Parameters:
    config_path (str): The path where the configuration file will be saved.
    config (ConfigParser): A ConfigParser object containing the configuration settings.

    Returns:
    config (ConfigParser): The same ConfigParser object provided as input.

    Explanation:
    This function saves configuration settings from a ConfigParser object to a specified file.
    The ConfigParser object should contain the desired configuration values.

    Example usage:
    >>> config = ConfigParser()
    >>> config['section']['option'] = 'value'
    >>> save_config('/path/to/save/config.ini', config)
    """

    with open(config_path, 'w+') as configfile:
        config.write(configfile)  # Write the configuration settings to the specified file
    return config
    

def save_features(df, filename, file_format, path):
    """
    Save a DataFrame containing features to a file with the specified format.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing features to be saved.
    filename (str): The base name of the file (without extension).
    file_format (str): The desired file format ('.csv', '.pkl', '.hdf', '.json').
    path (str): The directory path where the file will be saved.

    Returns:
    None

    Explanation:
    This function allows you to save a DataFrame containing features to a file with the specified format.
    It first checks if the specified path exists, and if not, creates the necessary directories.
    Depending on the chosen file format, the DataFrame is saved using different methods available in pandas.

    Example usage:
    >>> features = pd.DataFrame(...)  # A pandas DataFrame containing features
    >>> save_features(features, 'feature_data', '.csv', '/path/to/save/')
    """

    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory path if it doesn't exist

    save_path = os.path.join(path, filename + file_format)  # Complete file path

    if file_format == '.csv':
        df.to_csv(save_path, header=True, index=False)  # Save DataFrame to CSV format
    elif file_format == '.pkl':
        df.to_pickle(save_path)  # Save DataFrame to pickle format
    elif file_format == '.hdf':
        df.to_hdf(save_path, key='df', mode='w')  # Save DataFrame to HDF5 format
    elif file_format == '.json':
        df.to_json(save_path)  # Save DataFrame to JSON format
        
