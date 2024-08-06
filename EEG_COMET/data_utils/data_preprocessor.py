
import numpy as np
import collections
from mne import use_log_level
from pyprep.find_noisy_channels import NoisyChannels
from mne.preprocessing import ICA
from mne_icalabel import label_components
import warnings
from data_utils.data_io import DataIO


class DataPreprocessor:
    """
    The DataPreprocessor class provides methods for preprocessing EEG data.
    """
    def __init__(self):
        pass

    @staticmethod
    def preprocess_eegs(
            eeg_path, list_eegs, datatype, channel_location_dir, filter_bool, filtermethod, lowcut, highcut,
            downsample_bool, sampling_rate, chan2rm, prep_data_bool, iclabel_bool
    ):
        """
        Preprocess EEG data.

        Args:
            eeg_path (str): The path to the EEG data file.
            list_eegs (list): The list of EEG data files.
            datatype (str): The type of the EEG data ('raw' or 'epoched').
            channel_location_dir (str): The path to the channel location file.
            filter_bool (bool): Whether to apply filtering.
            filtermethod (str): The filtering method.
            lowcut (float): The lowcut frequency for filtering.
            highcut (float): The highcut frequency for filtering.
            downsample_bool (bool): Whether to apply downsampling.
            sampling_rate (float): The target sampling rate.
            chan2rm (str or list): The channels to remove.
            prep_data_bool (bool): Whether to detect noisy channels using pyprep.
            iclabel_bool (bool): Whether to apply ica and remove artifacts using iclabel.

        Returns:
            tuple: A tuple containing the preprocessed EEG data, data length, EEG info, and channels to remove.
        """
        
        verbose = 'ERROR'
        channels2remove = ['']

        # Determine channels to remove and standardize channels across all files
        if chan2rm == 'missing':
            for file in range(len(list_eegs)):
                filename = list_eegs[file]
                # Load the EEG data
                eeg = DataIO().load_eegs(filename, datatype, channel_location_dir, channels2remove)
                if file == 0:
                    channels = eeg.info['ch_names']
                else:
                    channels = np.append(channels, eeg.info['ch_names'])
            counter = collections.Counter(channels)
            counter = np.array(list(counter.items()))
            channels2remove = counter[np.where(counter[:, 1].astype(float) < len(list_eegs)), 0].tolist()
        else:
            channels2remove[0] = chan2rm

        # Load the EEG data
        eeg = DataIO().load_eegs(eeg_path, datatype, channel_location_dir, channels2remove[0])

        # Find and interpolate bad channels
        if prep_data_bool:
            with use_log_level(verbose):
                warnings.filterwarnings('ignore')
                nd = NoisyChannels(eeg, random_state=1337).find_all_bads()
                if nd:
                    bad_channels = nd.get_bads()
                    eeg.info['bads'] = bad_channels
                    eeg.interpolate_bads(reset_bads=False, verbose=verbose)

        # Apply filtering if specified
        if filter_bool:
            eeg = eeg.filter(
                l_freq=lowcut, h_freq=highcut, method=filtermethod, phase='zero', n_jobs=-1, verbose=verbose)

        # Downsample if specified
        if downsample_bool:
            sfreq = eeg.info['sfreq']
            if sfreq != sampling_rate:
                eeg = eeg.resample(sampling_rate, verbose=verbose)

        # Add average reference projection
        eeg.set_eeg_reference('average', projection=True, verbose=verbose)
        eeg.apply_proj(verbose=verbose)

        # # ICA and artifact removal using ICLabel
        if iclabel_bool:
            ica = ICA(n_components=None, random_state=97, method='fastica', verbose=verbose)
            ica.fit(eeg)
            ica_labels = label_components(eeg, ica, method='iclabel')
            artifact_labels = {'eye blink', 'muscle artifact'}
            artifact_indices = [i for i, label in enumerate(ica_labels['labels']) if label in artifact_labels]
            ica.exclude = artifact_indices
            eeg = ica.apply(eeg)

        # Get the raw data or combine epoched data
        eeg_data = DataIO.get_eeg_data(eeg, datatype)

        # Get data length and EEG info
        length_data = eeg_data.shape[1]
        eeg_info = eeg.info

        return eeg, eeg_data, length_data, eeg_info, channels2remove
