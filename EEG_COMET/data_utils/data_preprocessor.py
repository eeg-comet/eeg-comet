
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mne import use_log_level, pick_types, pick_info
from mne.io import RawArray
from mne.epochs import EpochsArray
from mne.time_frequency import psd_array_welch
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels
from meegkit import dss


class DataPreprocessor:
    """
    The DataPreprocessor class provides methods for preprocessing EEG data.
    """
    def __init__(self):
        pass

    @staticmethod
    def identify_bad_channels(eeg, verbose='ERROR'):
        """
        Identify and mark bad channels in the EEG data using NoisyChannels.

        Args:
            eeg (instance of Raw or Epochs): Raw EEG data.
            verbose (str): Logging verbosity level.

        Returns:
            eeg (instance of Raw or Epochs): EEG data with bad channels marked in 'bads'.
        """
        with use_log_level(verbose):
            warnings.filterwarnings('ignore')
            nd = NoisyChannels(eeg, random_state=1337).find_all_bads()
            if nd:
                bad_channels = nd.get_bads()
                eeg.info['bads'] = bad_channels
        return eeg

    @staticmethod
    def spatial_smooth_eeg(eeg, k=6, verbose='ERROR'):
        """
        Perform spatial smoothing on EEG data using K-nearest neighbors approach.
        (works with both Raw and Epochs objects)

        Args:
            eeg (instance of Raw or Epochs): Raw EEG data.
            k (int): Number of neighbors to use for smoothing.
            verbose (str): Verbosity level for logging.

        Returns:
            smoothed_raw (instance of Raw or Epochs): The spatially smoothed EEG data.
        """
        picks_eeg = pick_types(info=eeg.info, meg=False, eeg=True, exclude=[])
        pos = np.array([eeg.info['chs'][i]['loc'][:3] for i in picks_eeg])
        distances = squareform(pdist(pos))
        n_channels = distances.shape[0]
        neighbors = {}
        for i in range(n_channels):
            neighbor_idx = np.argsort(distances[i, :])
            neighbors[i] = neighbor_idx[1:k + 1].tolist()
        is_epochs = hasattr(eeg, 'events')
        if is_epochs:
            eeg_data = eeg.get_data(picks=picks_eeg)
            eeg_data = np.transpose(eeg_data, (1, 2, 0))
            n_channels, n_times, n_epochs = eeg_data.shape
            smoothed_data = np.zeros_like(eeg_data)
            for i in range(n_channels):
                neighborhood = neighbors[i] + [i]
                smoothed_data[i] = np.mean(eeg_data[neighborhood, :, :], axis=0)
            smoothed_data = np.transpose(smoothed_data, (2, 0, 1))

            info_eeg = pick_info(info=eeg.info, sel=picks_eeg)
            smoothed_eeg = EpochsArray(data=smoothed_data, info=info_eeg,
                                       events=eeg.events, event_id=eeg.event_id,
                                       tmin=eeg.tmin, verbose=verbose)
        else:
            eeg_data = eeg.get_data(picks=picks_eeg)
            n_channels, n_times = eeg_data.shape
            smoothed_data = np.zeros_like(eeg_data)
            for i in range(n_channels):
                neighborhood = neighbors[i] + [i]
                smoothed_data[i] = np.mean(eeg_data[neighborhood, :], axis=0)
            info_eeg = pick_info(info=eeg.info, sel=picks_eeg)
            smoothed_eeg = RawArray(data=smoothed_data, info=info_eeg, verbose=verbose)
        return smoothed_eeg

    def preprocess_eeg(self, eeg, filter_bool, filtermethod, lowcut, highcut, downsample_bool, sampling_rate,
                       spatial_smooth_bool, spatial_smooth_k, verbose='ERROR'):
        """
        Preprocess EEG data with optional filtering, downsampling, and re-referencing.

        Args:
            eeg (instance of Raw or Epochs): Raw EEG data.
            filter_bool (bool): Whether to apply temporal filtering.
            filtermethod (str): Filtering method (e.g., FIR, IIR).
            lowcut (int): Low cutoff frequency for filtering.
            highcut (int): High cutoff frequency for filtering.
            downsample_bool (bool): Whether to downsample the data.
            sampling_rate (int): Target sampling rate.
            spatial_smooth_bool (bool): Whether to apply spatial filtering.
            spatial_smooth_k (int): Number of neighbors to use for spatial smoothing.
            verbose (str): Verbosity level for logging.

        Returns:
            eeg (instance of Raw or Epochs): Preprocessed EEG data.
        """
        if filter_bool:
            eeg = eeg.filter(
                l_freq=lowcut, h_freq=highcut, method=filtermethod, phase='zero', n_jobs=-1, verbose=verbose)
        if downsample_bool:
            sfreq = eeg.info['sfreq']
            if sfreq != sampling_rate:
                eeg = eeg.resample(sampling_rate, verbose=verbose)
        if spatial_smooth_bool:
            eeg = self.spatial_smooth_eeg(eeg=eeg, k=spatial_smooth_k, verbose=verbose)
        eeg.set_eeg_reference('average', projection=True, verbose=verbose)
        eeg.apply_proj(verbose=verbose)
        return eeg

    @staticmethod
    def remove_line_noise(eeg, verbose):
        """
        Remove line noise from EEG data using DSS line noise removal.

        Args:
            eeg (instance of Raw or Epochs): Raw EEG data.
            verbose (str): Logging verbosity level.

        Returns:
            eeg (instance of Raw or Epochs): EEG data with line noise removed.
        """
        data = eeg.get_data()
        sfreq = eeg.info['sfreq']
        psds, freqs = psd_array_welch(data, sfreq, fmin=45, fmax=65, verbose=verbose)
        line_noise_frequency = freqs[np.argmax(psds.mean(axis=0))]
        processed_data, _ = dss.dss_line_iter(data.T, line_noise_frequency, eeg.info['sfreq'], show=True)
        eeg._data = processed_data.T
        return eeg

    def auto_clean_raw_eeg(self, eeg, verbose='ERROR'):
        """
        Automatically clean raw EEG data.

        Steps:
            0. Downsample the EEG data:
                Reduce the sampling rate to 1000 Hz for faster processing.
            1. Remove line noise:
                Apply DSS line noise removal to eliminate electrical interference.
            2. High-pass filtering and bad channel detection:
                Filter out low-frequency noise and identify bad channels for removal.
            3. Re-reference to average:
                Set the EEG data to use the average reference for better signal quality.
            4. Artifact removal using ICA:
                Perform Independent Component Analysis (ICA) to detect and exclude artifacts such as
                eye blinks and muscle movements.
            5. Interpolate bad channels:
                Replace bad channels with interpolated data from surrounding channels.

        Args:
            eeg (instance of Raw or Epochs): Raw EEG data.
            verbose (str): Logging verbosity level.

        Returns:
            eeg (instance of Raw or Epochs): Cleaned EEG data.
        """

        with use_log_level(verbose):
            warnings.filterwarnings('ignore')
            eeg = eeg.resample(1000, verbose=verbose)
            eeg = self.remove_line_noise(eeg, verbose)
            eeg.filter(l_freq=1, h_freq=None, method='fir', phase='zero')
            eeg = self.identify_bad_channels(eeg)
            eeg.pick_types(eeg=True, exclude='bads')
            eeg.set_eeg_reference('average', projection=True)
            eeg.apply_proj()
            ica_eeg = ICA(n_components=None, random_state=97, method='fastica')
            ica_eeg.fit(eeg)
            ica_labels_eeg = label_components(eeg, ica_eeg, method='iclabel')
            artifact_labels = {'eye blink', 'muscle artifact'}
            artifact_indices_eeg = [i for i, label in enumerate(ica_labels_eeg['labels']) if
                                          label in artifact_labels]
            ica_eeg.exclude = artifact_indices_eeg
            eeg = ica_eeg.apply(eeg)
            eeg.interpolate_bads(reset_bads=False, verbose=verbose)
        return eeg
