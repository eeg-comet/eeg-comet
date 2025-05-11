import numpy as np
from scipy.signal import find_peaks
from scipy.signal import correlate
from data_utils.data_io import DataIO


class DataInitializer:
    """Provides methods for initializing and preparing data for microstate analysis.

    This class contains utilities for extracting Global Field Power (GFP) peaks from EEG data,
    generating microstate maps, and initializing cluster centers for microstate clustering
    algorithms. It works with both continuous and epoched EEG data formats.
    """

    def __init__(self):
        """Initialize the DataInitializer class."""
        pass

    @staticmethod
    def initialize_cluster_centers(maps2use, n_states, initializer):
        """Initialize cluster centers for k-means clustering in microstate analysis.

        Implements two initialization strategies for microstate clustering:
        1. 'K-Means++': A smart initialization that chooses initial centers with
           probability proportional to distance from existing centers
        2. 'Random': Randomly selects maps as initial centers

        Both methods normalize the resulting centers to unit length.

        Args:
            maps2use (numpy.ndarray): Array of potential microstate maps with shape (n_channels, n_samples)
            n_states (int): Number of microstate clusters to identify
            initializer (str): Initialization method, either 'K-Means++' or 'Random'

        Returns:
            numpy.ndarray: Initialized cluster centers with shape (n_states, n_channels),
                           normalized to unit length

        Notes:
            For 'K-Means++', distance between maps is calculated using correlation.
            This implementation follows the standard k-means++ algorithm but adapted
            for EEG topographies.
        """
        if initializer == 'K-Means++':
            initial_idx = np.random.choice(np.size(maps2use, 1))
            initial_centers = [maps2use[:, initial_idx]]
            for _ in range(1, n_states):
                dists = np.array(
                    [min(float(abs(correlate(d, c, mode='valid')[0])) for c in initial_centers) for d in maps2use.T]
                )
                probs = dists / dists.sum()
                next_idx = np.random.choice(np.size(maps2use, 1), p=probs)
                next_centroid = maps2use[:, next_idx]
                initial_centers.append(next_centroid)
            initial_centers = np.array(initial_centers)

        else:
            # initializer == 'Random'
            random_state = np.random.RandomState(None)
            initial_peaks = random_state.choice(np.size(maps2use, 1), size=n_states, replace=False)
            initial_centers = maps2use[:, initial_peaks].T

        initial_centers /= np.linalg.norm(initial_centers, axis=1, keepdims=True)
        return initial_centers

    @staticmethod
    def extract_gfp_peaks_and_maps(data, use_percentages=None, min_dist=None):
        """Extract Global Field Power (GFP) peaks and corresponding topographical maps.

        GFP is calculated as the standard deviation across channels at each time point.
        Peaks in the GFP signal represent moments of highest spatial variability and
        are commonly used as the basis for microstate analysis.

        Args:
            data (numpy.ndarray): EEG data array with shape (n_channels, n_timepoints)
            use_percentages (float, optional): If provided, randomly selects this percentage
                of timepoints instead of detecting peaks. Value should be between 0-100.
                Defaults to None (use peak detection)
            min_dist (int, optional): Minimum distance between peaks in samples.
                If 0, no minimum distance is enforced. Defaults to None

        Returns:
            tuple: Contains:
                - maps (numpy.ndarray): Topographical maps at GFP peaks,
                  shape (n_channels, n_peaks), normalized to unit length
                - peaks (numpy.ndarray): Indices of GFP peaks in the original data

        Notes:
            When use_percentages is provided, peak detection is bypassed and random
            timepoints are selected instead, which can be useful for large datasets.
        """
        gfp = np.std(data, axis=0)

        if use_percentages is not None:
            num_samples = int(data.shape[1] * (int(use_percentages) / 100))
            peaks = np.random.choice(data.shape[1], size=num_samples, replace=False)
        else:
            if min_dist == 0:
                min_dist = None
            peaks, _ = find_peaks(gfp, distance=min_dist)
        maps = data[:, peaks]
        maps /= np.linalg.norm(maps, axis=1, keepdims=True)
        return maps, peaks

    @staticmethod
    def generate_maps_and_peaks(preprocessed_folder, extension, datatype, use_percentages=None, min_dist=None):
        """Generate GFP maps and peak indices from multiple preprocessed EEG files.

        Loads all EEG files from the specified folder that match the extension,
        extracts GFP peaks and maps from each file, and concatenates the results.

        Args:
            preprocessed_folder (str): Directory containing preprocessed EEG files
            extension (str): File extension to match (e.g., '.set', '.vhdr')
            datatype (str): Type of EEG data - 'raw' or 'epoched'
            use_percentages (float, optional): Percentage of data points to randomly
                select instead of using peak detection (0-100). Defaults to None
            min_dist (int, optional): Minimum distance between peaks in samples.
                Defaults to None

        Returns:
            tuple: Contains:
                - maps2use (numpy.ndarray): Concatenated topographical maps from all files,
                  shape (n_channels, total_peaks)
                - peaks2use (numpy.ndarray): Concatenated indices of peaks from all files

        Notes:
            This method is particularly useful for group-level microstate analysis
            where maps from multiple subjects or recordings need to be combined.
        """
        data_io = DataIO()
        all_preprocessed_paths, _ = data_io.find_data(preprocessed_folder, extension)
        maps2use, peaks2use = [], []
        counter = 0
        for eeg_path in all_preprocessed_paths:
            eeg = data_io.load_eeg(eeg_path, datatype)
            eeg_data = data_io.get_eeg_data(eeg, datatype)
            maps, peaks = DataInitializer.extract_gfp_peaks_and_maps(eeg_data, use_percentages, min_dist)

            if counter == 0:
                maps2use = maps
                peaks2use = peaks
            else:
                maps2use = np.hstack((maps2use, maps))
                peaks2use = np.hstack((peaks2use, peaks))
            counter = counter + 1

        return maps2use, peaks2use
