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
    """Provides comprehensive methods for preprocessing electroencephalography (EEG) data.

    This class contains various static and instance methods to clean, filter, and
    prepare EEG data for analysis. It handles bad channel detection, spatial smoothing,
    line noise removal, and automatic cleaning pipelines.
    """

    def __init__(self):
        """Initialize the DataPreprocessor class."""
        pass

    @staticmethod
    def identify_bad_channels(eeg, verbose='ERROR'):
        """Identify and mark bad (noisy) channels in the EEG data.

        Uses the pyprep NoisyChannels algorithm to automatically detect channels
        that have poor signal quality or excessive noise.

        Args:
            eeg (Raw or Epochs): MNE Raw or Epochs object containing EEG data
            verbose (str): Logging verbosity level ('ERROR', 'WARNING', 'INFO', etc.)

        Returns:
            Raw or Epochs: EEG data with identified bad channels marked in info['bads']
        """
        with use_log_level(verbose):
            warnings.filterwarnings('ignore')
            nd = NoisyChannels(eeg, random_state=1337).find_all_bads()
            if nd:
                bad_channels = nd.get_bads()
                eeg.info['bads'] = bad_channels
        return eeg

    @staticmethod
    def spatial_smooth_eeg(eeg, min_neighbors=3, max_neighbors=8, verbose='ERROR'):
        """Apply spatial smoothing to EEG data by averaging signals with neighboring electrodes.

        For each electrode, finds neighbors based on:
        1. Natural "jumps" in the distance distribution
        2. Ensuring at least min_neighbors and at most max_neighbors per channel
        3. Adaptive selection based on electrode distances

        Args:
            eeg (Raw or Epochs): MNE Raw or Epochs object containing EEG data
            min_neighbors (int): Minimum number of neighbors each electrode should have
            max_neighbors (int): Maximum number of neighbors each electrode should have
            verbose (str): Logging verbosity level ('ERROR', 'WARNING', 'INFO', etc.)

        Returns:
            Raw or Epochs: New MNE Raw or Epochs object with spatially smoothed data
        """
        picks_eeg = pick_types(info=eeg.info, meg=False, eeg=True, exclude=[])
        pos = np.array([eeg.info['chs'][i]['loc'][:3] for i in picks_eeg])
        distances = squareform(pdist(pos))
        n_channels = distances.shape[0]
        neighbors = {}
        min_neighbors = max(1, min(min_neighbors, n_channels - 1))
        max_neighbors = max(min_neighbors, min(max_neighbors, n_channels - 1))
        for i in range(n_channels):
            dist_to_others = np.copy(distances[i, :])
            dist_to_others[i] = np.inf
            sorted_indices = np.argsort(dist_to_others)
            sorted_distances = dist_to_others[sorted_indices]
            distance_diffs = np.diff(sorted_distances[:max_neighbors + 1])
            if len(distance_diffs) > 1 and np.max(distance_diffs) > 0:
                norm_diffs = distance_diffs / np.mean(distance_diffs[:3])
                jump_indices = np.where(norm_diffs > 2.0)[0]
                if len(jump_indices) > 0 and jump_indices[0] >= min_neighbors:
                    n_neighbors = jump_indices[0] + 1
                    n_neighbors = min(n_neighbors, max_neighbors)
                else:
                    n_neighbors = max_neighbors
            else:
                n_neighbors = min(min_neighbors + 2, max_neighbors)
            n_neighbors = max(n_neighbors, min_neighbors)
            neighbors[i] = sorted_indices[:n_neighbors].tolist()

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
                       spatial_smooth_bool, verbose='ERROR'):
        """Preprocess EEG data with configurable pipeline options.

        Applies a combination of temporal filtering, downsampling, spatial smoothing,
        and re-referencing based on the provided parameters.

        Args:
            eeg (Raw or Epochs): MNE Raw or Epochs object containing EEG data
            filter_bool (bool): Whether to apply temporal filtering
            filtermethod (str): Filtering method ('FIR', 'IIR', etc.)
            lowcut (float): Low cutoff frequency for filtering in Hz
            highcut (float): High cutoff frequency for filtering in Hz
            downsample_bool (bool): Whether to downsample the data
            sampling_rate (int): Target sampling rate in Hz
            spatial_smooth_bool (bool): Whether to apply spatial smoothing
            verbose (str): Logging verbosity level ('ERROR', 'WARNING', 'INFO', etc.)

        Returns:
            Raw or Epochs: Preprocessed EEG data
        """
        if filter_bool:
            eeg = eeg.filter(
                l_freq=lowcut, h_freq=highcut, method=filtermethod, phase='zero', verbose=verbose)
        if downsample_bool:
            sfreq = eeg.info['sfreq']
            if sfreq != sampling_rate:
                eeg = eeg.resample(sampling_rate, verbose=verbose)
        if spatial_smooth_bool:
            eeg = self.spatial_smooth_eeg(eeg=eeg, verbose=verbose)
        eeg.set_eeg_reference('average', projection=True, verbose=verbose)
        eeg.apply_proj(verbose=verbose)
        return eeg

    @staticmethod
    def remove_line_noise(eeg, verbose):
        """Remove electrical line noise from EEG data using DSS (Denoising Source Separation).

        Automatically detects the peak frequency in the 45-65 Hz range (typically 50 or 60 Hz)
        and applies iterative DSS to remove this noise without distorting the underlying signal.

        Args:
            eeg (Raw or Epochs): MNE Raw or Epochs object containing EEG data
            verbose (str): Logging verbosity level ('ERROR', 'WARNING', 'INFO', etc.)

        Returns:
            Raw or Epochs: EEG data with line noise removed
        """
        data = eeg.get_data()
        sfreq = eeg.info['sfreq']
        psds, freqs = psd_array_welch(data, sfreq, fmin=45, fmax=65, verbose=verbose)
        line_noise_frequency = freqs[np.argmax(psds.mean(axis=0))]
        processed_data, _ = dss.dss_line_iter(data.T, line_noise_frequency, eeg.info['sfreq'], show=True)
        eeg._data = processed_data.T
        return eeg

    def auto_clean_raw_eeg(self, eeg, verbose='ERROR'):
        """Apply a complete automatic cleaning pipeline to raw EEG data.

        Performs a comprehensive cleaning procedure with the following steps:
        1. Downsampling to 1000 Hz for computational efficiency
        2. Line noise removal using DSS
        3. High-pass filtering at 1 Hz to remove slow drifts
        4. Bad channel detection using NoisyChannels algorithm
        5. Re-referencing to average reference
        6. Artifact removal using ICA with automatic component classification via ICLabel
        7. Interpolation of bad channels

        Args:
            eeg (Raw): MNE Raw object containing EEG data
            verbose (str): Logging verbosity level ('ERROR', 'WARNING', 'INFO', etc.)

        Returns:
            Raw: Cleaned EEG data ready for analysis

        Notes:
            This method is designed for raw continuous data, not epoched data.
            The method uses ICA and automatic component classification to remove
            eye blink and muscle artifacts.
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
            artifact_indices_eeg = [i for i, label in enumerate(ica_labels_eeg['labels']) if label in artifact_labels]
            ica_eeg.exclude = artifact_indices_eeg
            eeg = ica_eeg.apply(eeg)
            eeg.interpolate_bads(reset_bads=False, verbose=verbose)
        return eeg
