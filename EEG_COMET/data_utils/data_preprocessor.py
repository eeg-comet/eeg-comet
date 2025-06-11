import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mne import use_log_level, pick_types, pick_info
from mne.io import RawArray
from mne.epochs import EpochsArray
from mne.time_frequency import psd_array_welch
from pyprep.find_noisy_channels import NoisyChannels


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
