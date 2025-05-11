import os.path
import pickle
import collections
from fnmatch import fnmatch
import numpy as np
import mne
from scipy.io import loadmat
import warnings


class DataIO:
    """Provides methods for loading, saving, and manipulating EEG data files.

    This class handles various EEG file formats and provides utilities for
    working with channel locations, montages, and data extraction. It supports
    reading and writing different EEG formats and provides helper methods for
    file discovery and data transformation.
    """

    def __init__(self):
        """Initialize the DataIO class."""
        pass

    @staticmethod
    def save_eeg_info(eeg_info_path, eeg_info):
        """Save EEG information to a binary file using pickle.

        Args:
            eeg_info_path (str): Path where the EEG information will be saved
            eeg_info (object): EEG information object to be serialized

        Raises:
            IOError: If the file cannot be written to the specified path
        """
        with open(eeg_info_path, 'wb') as f:
            pickle.dump(eeg_info, f)

    @staticmethod
    def load_eeg_info(eeg_info_path):
        """Load EEG information from a binary file using pickle.

        Args:
            eeg_info_path (str): Path to the pickle file containing EEG information

        Returns:
            object: Deserialized EEG information object

        Raises:
            FileNotFoundError: If the specified file does not exist
            pickle.UnpicklingError: If the file is not a valid pickle file
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
                ".data", ".nxe", ".lay"
            ]
            for path, subdirs, files in os.walk(input_folder):
                for name in files:
                    if os.path.splitext(name)[1] in valid_eeg_formats:
                        if fnmatch(name, pattern):
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
    def _read_mat_locations(fname):
        """Read channel locations from a Brainstorm .mat file.

        Extracts 3D electrode positions from Brainstorm's channel structure,
        converting to the MNE coordinate system (y, x, z) in meters.

        Args:
            fname (str): Path to the Brainstorm .mat file

        Returns:
            dict: Dictionary mapping channel names to 3D positions in meters
                Format: {'ChannelName': [y, x, z], ...}

        Raises:
            ValueError: If the file does not contain the expected 'Channel' key
            IOError: If the file cannot be read
        """
        mat = loadmat(fname)
        if 'Channel' not in mat:
            raise ValueError('MAT file does not contain "Channel" key.')
        channel_data = mat['Channel'][0]
        ch_pos = {}
        for ch in channel_data:
            name = ch['Name'][0]
            loc = ch['Loc'].flatten() if ch['Loc'].shape == (3, 1) else ch['Loc']
            if abs(loc[0]) > 0.5 or abs(loc[1]) > 0.5 or abs(loc[2]) > 0.5:
                loc[0] = loc[0] / 1000.0
                loc[1] = loc[1] / 1000.0
                loc[2] = loc[2] / 1000.0
            ch_pos[name] = [loc[1], loc[0], loc[2]]
        return ch_pos

    @staticmethod
    def _read_ced_locations(fname):
        """Read channel locations from an EEGLAB .ced file.

        Parses a .ced file to extract electrode positions, detecting column layout
        and converting coordinates to meters if necessary.

        Args:
            fname (str): Path to the .ced file

        Returns:
            dict: Dictionary mapping channel names to 3D positions in meters
                Format: {'ChannelName': np.array([y, x, z]), ...}

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file format is invalid or missing required columns
        """
        ch_pos = {}
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"The file {fname} does not exist.")
        with open(fname, 'r') as f:
            lines = f.readlines()
        if not lines:
            raise ValueError("The .ced file is empty.")
        header_line = lines[0].strip()
        if not header_line:
            raise ValueError("The .ced file does not contain a header line.")
        parts = header_line.split('\t')
        if len(parts) < 4:
            parts = header_line.split()
        col_map = {col.strip().lower(): idx for idx, col in enumerate(parts)}
        required_columns = ['labels', 'x', 'y', 'z']
        missing_cols = [col for col in required_columns if col not in col_map]
        if missing_cols:
            raise ValueError(f"Missing required columns in header: {missing_cols}")
        label_idx = col_map['labels']
        x_idx = col_map['x']
        y_idx = col_map['y']
        z_idx = col_map['z']
        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < len(parts):
                parts = line.split()
            if len(parts) <= max(label_idx, x_idx, y_idx, z_idx):
                raise ValueError(f"Invalid line in CED file at line {line_num}: {line}")
            label = parts[label_idx].strip()
            try:
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
            except ValueError:
                raise ValueError(f"Invalid numerical values in line {line_num}: {line}")
            if abs(x) > 0.5 or abs(y) > 0.5 or abs(z) > 0.5:
                x = x / 1000.0
                y = y / 1000.0
                z = z / 1000.0
            ch_pos[label] = np.array([y, x, z])
        return ch_pos

    def load_montage(self, channel_location_dir):
        """Load electrode positions from a file or built-in montage name.

        Supports multiple file formats (.mat, .ced, standard MNE formats) or
        built-in standard montages from MNE.

        Args:
            channel_location_dir (str): Path to channel location file or name of
                standard montage (e.g., 'standard_1020'). If empty, defaults to 'standard_1020'

        Returns:
            mne.channels.DigMontage: Montage object containing electrode positions

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the montage cannot be created from the provided input
        """
        if not channel_location_dir:
            channel_location_dir = "standard_1020"
        try:
            if os.path.isfile(channel_location_dir):
                file_ext = os.path.splitext(channel_location_dir)[1].lower()
                if file_ext == '.mat':
                    ch_pos = self._read_mat_locations(channel_location_dir)
                    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
                elif file_ext == '.ced':
                    ch_pos = self._read_ced_locations(channel_location_dir)
                    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
                else:
                    montage = mne.channels.read_custom_montage(channel_location_dir)
            elif channel_location_dir in mne.channels.get_builtin_montages():
                montage = mne.channels.make_standard_montage(channel_location_dir)
            else:
                raise ValueError(f'Unknown montage: {channel_location_dir}')
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found: {channel_location_dir}')
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise ValueError(f'An error occurred while loading the montage: {e}')
        if montage is None:
            raise ValueError(f'Could not create a montage from the provided directory or name: {channel_location_dir}')
        return montage

    def check_chan2rm(self, list_eegs_path, datatype, channel_location_dir):
        """Identify channels missing in any file across a dataset collection.

        Checks for channel consistency across multiple EEG files and identifies
        channels that should be removed to ensure all datasets have the same channels.

        Args:
            list_eegs_path (list of str): Paths to EEG files to check
            datatype (str): Type of EEG data ('raw' or 'epoched')
            channel_location_dir (str): Path to channel locations or name of standard montage

        Returns:
            list of str: Names of channels that are missing in at least one file

        Notes:
            This is useful for preprocessing multiple files to ensure channel consistency
            before combining or comparing data across files.
        """
        channels = ['']
        for i, filename in enumerate(list_eegs_path):
            eeg = self.load_eeg(filename, datatype, channel_location_dir, preload=False)
            if i == 0:
                channels = eeg.info['ch_names']
            else:
                channels.extend(eeg.info['ch_names'])
        counter = collections.Counter(channels)
        total_files = len(list_eegs_path)
        missing_channels = [chan for chan, count in counter.items() if count < total_files]
        return missing_channels

    def load_eeg(self, eeg_path, datatype, channel_location_dir='', chan2rm=None, preload=True, verbose='CRITICAL'):
        """Load EEG data from various file formats with optional preprocessing.

        Loads raw or epoched EEG data, applies channel locations, removes specified
        channels, and sets up average reference.

        Args:
            eeg_path (str): Path to the EEG data file
            datatype (str): Type of EEG data to load ('raw' or 'epoched')
            channel_location_dir (str, optional): Path to channel locations or name
                of standard montage. Defaults to '' (uses 'standard_1020')
            chan2rm (list of str, optional): List of channel names to remove.
                Defaults to None
            preload (bool, optional): Whether to load data into memory immediately.
                Defaults to True
            verbose (str, optional): MNE verbosity level ('CRITICAL', 'ERROR', etc.).
                Defaults to 'CRITICAL'

        Returns:
            mne.io.Raw or mne.Epochs: Loaded EEG data object

        Raises:
            FileNotFoundError: If the EEG file does not exist
            ValueError: If the datatype is not supported or montage cannot be loaded
        """
        with mne.use_log_level(verbose):
            warnings.filterwarnings('ignore')
            if datatype == 'raw':
                eeg = mne.io.read_raw(eeg_path, preload=preload, verbose=verbose)
            elif datatype == 'epoched':
                eeg = mne.io.read_epochs_eeglab(eeg_path, verbose=verbose)
                # eeg = mne.io.read_epochs(eeg_path, verbose=False)
            montage = self.load_montage(channel_location_dir)
            eeg.set_montage(montage, match_case=False, on_missing='warn')
            ch_names = eeg.info['ch_names']
            if chan2rm is None:
                chan2rm = []
            if (
                    any(chan2rm)
                    and any(elem != '' for elem in chan2rm)
                    and chan2rm in ch_names
            ):
                eeg = eeg.drop_channels(chan2rm)
            if 'TRIGGER' in ch_names:
                eeg = eeg.drop_channels('TRIGGER')
            eeg.set_eeg_reference('average', projection=True)
            eeg.apply_proj()
        return eeg

    @staticmethod
    def export_eegs(eeg, save_path, extension, datatype):
        """Export EEG data to a specified file format.

        Saves EEG data to common file formats supported by MNE, automatically
        handling the appropriate export method based on data type.

        Args:
            eeg (mne.io.Raw or mne.Epochs): EEG data object to export
            save_path (str): Output path without extension (extension will be added)
            extension (str): Desired file extension ('.vhdr', '.set', '.edf')
                If not one of these, defaults to '.set'
            datatype (str): Type of EEG data ('raw' or 'epoched')

        Notes:
            Automatically overwrites existing files with the same name.
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
        """Extract raw NumPy arrays from MNE Raw or Epochs objects.

        Converts MNE objects to NumPy arrays for further processing or analysis.
        For epoched data, concatenates all epochs along the time dimension.

        Args:
            eeg (mne.io.Raw or mne.Epochs): EEG data object
            datatype (str): Type of EEG data ('raw' or 'epoched')

        Returns:
            numpy.ndarray: EEG data as a NumPy array
                For raw data: shape (n_channels, n_times)
                For epoched data: shape (n_channels, n_times_total)
                where n_times_total = n_epochs * n_times_per_epoch

        Raises:
            ValueError: If no data is available (empty epochs object or invalid datatype)
        """
        eeg_data = None
        if datatype == "epoched":
            for index in range(eeg.__len__()):
                if index == 0:
                    eeg_data = np.squeeze(eeg[0].get_data())
                else:
                    epoch = np.squeeze(eeg[index].get_data())
                    eeg_data = np.append(eeg_data, epoch, axis=1)
        else:
            eeg_data = eeg.get_data()
        if eeg_data is None:
            raise ValueError("No data available: empty epochs object or invalid datatype")
        return eeg_data
