
import os.path
import pickle
from fnmatch import fnmatch
import numpy as np
import mne
import warnings


class DataIO:
    """
    The DataIO class provides methods for saving, loading, and manipulating EEG data.
    """
    def __init__(self):
        pass

    @staticmethod
    def save_eeg_info(eeg_info_path, eeg_info):
        """
        Save EEG information to a binary file using pickle.

        Args:
            eeg_info_path (str): The path to save the EEG information.
            eeg_info (object): The EEG information object.
        """
    
        with open(eeg_info_path, 'wb') as f:
            pickle.dump(eeg_info, f)

    @staticmethod
    def load_eeg_info(eeg_info_path):
        """
        Load EEG information from a binary file using pickle.

        Args:
            eeg_info_path (str): The path to load the EEG information from.

        Returns:
            object: The loaded EEG information object.
        """
    
        with open(eeg_info_path, 'rb') as f:
            eeg_info = pickle.load(f)
        return eeg_info

    @staticmethod
    def find_data(input_folder, extension, pattern='*'):
        """
        Recursively search for data files within a folder based on extension and pattern.

        Args:
            input_folder (str): The folder to search for data files.
            extension (str): The file extension to match. If 'auto', load all eeg files with valid formats.
            pattern (str, optional): The pattern to match against the file name. Defaults to '*'.

        Returns:
            tuple: A tuple containing the list of matching file paths and the list of matching file names.
        """
        
        list_path = []
        list_filename = []
        if extension == ".auto":
            valid_eeg_formats = [
                ".vhdr", ".edf", ".bdf", ".gdf",
                ".cnt", ".egi", ".mff", ".set",
                ".data", ".nxe", ".lay", ".eeg"
            ]
            for path, subdirs, files in os.walk(input_folder):
                for name in files:
                    if os.path.splitext(name)[1] in valid_eeg_formats:
                        list_path.append(os.path.join(path, name))
                        list_filename.append(os.path.splitext(name)[0])
        else:
            for path, subdirs, files in os.walk(input_folder):
                for name in files:
                    if fnmatch(name, pattern + extension):
                        list_path.append(os.path.join(path, name))
                        list_filename.append(name.split('.')[0])
        return list_path, list_filename

    @staticmethod
    def load_eegs(eeg_path, datatype, channel_location_dir='', chan2rm=None, verbose='CRITICAL'):
        """
        Load EEG data from different formats and preprocess if needed.

        Args:
            eeg_path (str): The path to the EEG data file.
            datatype (str): The type of the EEG data ('raw' or 'epoched').
            channel_location_dir (str, optional): The path to the channel location file. Defaults to ''.
            chan2rm (list, optional): The list of channels to remove. Defaults to [].
            verbose (str, optional): The verbosity level. Defaults to 'CRITICAL'.

        Returns:
            object: The loaded EEG data object.
        """

        if chan2rm is None:
            chan2rm = []
        with mne.use_log_level(verbose):
            warnings.filterwarnings('ignore')
            if datatype == 'raw':
                eeg = mne.io.read_raw(eeg_path, preload=True, verbose=False)
            elif datatype == 'epoched':
                eeg = mne.io.read_epochs(eeg_path, verbose=False)
            try:
                if os.path.isfile(channel_location_dir):
                    montage = mne.channels.read_custom_montage(channel_location_dir)
                    eeg.set_montage(montage, match_case=False, on_missing='warn')
                elif channel_location_dir in mne.channels.get_builtin_montages():
                    montage = mne.channels.make_standard_montage(channel_location_dir)
                    eeg.set_montage(montage, match_case=False, on_missing='warn')
            except Exception as e:
                print(f'Invalid File, Try Loading Again. Details:{e}')
            channel_names = eeg.info['ch_names']
            if (
                any(chan2rm)
                and any(elem != '' for elem in chan2rm)
                and chan2rm in channel_names
            ):
                eeg = eeg.drop_channels(chan2rm)
            if 'TRIGGER' in channel_names:
                eeg = eeg.drop_channels('TRIGGER')
            eeg.set_eeg_reference('average', projection=True)
            eeg.apply_proj()
        return eeg

    @staticmethod
    def export_eegs(eeg, save_path, extension, datatype):
        """
        Export EEG data to a specified file format.

        Args:
            eeg (object): The EEG data object.
            save_path (str): The path to save the exported EEG data.
            extension (str): The file extension of the exported EEG data.
            datatype (str): The type of the EEG data ('raw' or 'epoched').
        """

        available_extensions = ['.vhdr', '.set', '.edf']
        if extension not in available_extensions:
            extension = '.set'
        if datatype == 'raw':
            mne.export.export_raw(save_path + extension, eeg, fmt='auto', overwrite=True)
        elif datatype == 'epoched':
            mne.export.export_epochs(save_path + extension, eeg, fmt='auto', overwrite=True)

    @staticmethod
    def get_eeg_data(eeg, datatype):
        """
        Extract EEG data from MNE-Python Epochs or Raw object.

        Args:
            eeg (object): The EEG data object.
            datatype (str): The type of the EEG data ('raw' or 'epoched').

        Returns:
            numpy.ndarray: The extracted EEG data.
        """
    
        if datatype == "epoched":
            for index in range(eeg.__len__()):
                if index == 0:
                    eeg_data = np.squeeze(eeg[0].get_data())
                else:
                    epoch = np.squeeze(eeg[index].get_data())
                    eeg_data = np.append(eeg_data, epoch, axis=1)
        else:
            eeg_data = eeg.get_data()
        return eeg_data
    