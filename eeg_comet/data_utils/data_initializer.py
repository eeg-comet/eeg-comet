"""Data initialization helpers for microstate analysis (EEG-COMET)."""

import numpy as np
from scipy.signal import correlate, find_peaks

from eeg_comet.data_utils.data_io import DataIO


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
        if initializer == "K-Means++":
            initial_idx = np.random.choice(np.size(maps2use, 1))
            initial_centers = [maps2use[:, initial_idx]]
            for _ in range(1, n_states):
                dists = np.array(
                    [
                        min(float(abs(correlate(d, c, mode="valid")[0])) for c in initial_centers)
                        for d in maps2use.T
                    ]
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
    def extract_gfp_peaks_and_maps(data, use_percentages=None, min_dist=None, random_seed=None):
        """Extract Global Field Power (GFP) peaks and corresponding topographical maps.

        GFP is calculated as the standard deviation across channels at each time point.
        Peaks in the GFP signal represent moments of highest spatial variability and
        are commonly used as the basis for microstate analysis.

        Args:
            data (numpy.ndarray): EEG data array with shape (n_channels, n_timepoints)
            use_percentages (float, optional): If provided, selects timepoints based on percentage:
                - None: Use GFP peak detection with min_dist
                - 0-99: Randomly selects this percentage of timepoints 
                - 100: Uses ALL timepoints (entire data)
                Value should be between 0-100. Defaults to None (use peak detection)
            min_dist (int, optional): Minimum distance between peaks in samples.
                If 0, no minimum distance is enforced. Only used when use_percentages is None
            random_seed (int, optional): Random seed for reproducible random sampling.
                Only used when use_percentages is provided and < 100. Defaults to None

        Returns:
            tuple: Contains:
                - maps (numpy.ndarray): Topographical maps at selected timepoints,
                  shape (n_channels, n_selected_points), normalized to unit length
                - peaks (numpy.ndarray): Indices of selected timepoints in the original data

        Notes:
            Three data selection modes:
            1. use_percentages=None: GFP peak detection (with min_dist constraint)
            2. use_percentages<100: Random subset of data points 
            3. use_percentages=100: All data points (entire dataset)
        """
        gfp = np.std(data, axis=0)

        if use_percentages is not None:
            if use_percentages == 100:
                # Use entire data - all time points for clustering
                peaks = np.arange(data.shape[1])
            else:
                # Use random subset of data based on percentage
                num_samples = int(data.shape[1] * (int(use_percentages) / 100))
                if random_seed is not None:
                    # Set random seed for reproducible sampling
                    rng = np.random.RandomState(random_seed)
                    peaks = rng.choice(data.shape[1], size=num_samples, replace=False)
                else:
                    peaks = np.random.choice(data.shape[1], size=num_samples, replace=False)
        else:
            if min_dist == 0:
                min_dist = None
            # Optional masking around boundaries
            boundary_mask = None
            try:
                # data may be derived from a Raw with annotations; if present in caller, they should pass masked data
                # Here we support a simple convention: if the first channel encodes a mask as NaN at boundary samples
                # we drop those. Otherwise, use distance-based detection only.
                if np.isnan(data[0]).any():
                    boundary_mask = ~np.isnan(data[0])
            except Exception:
                boundary_mask = None

            if boundary_mask is not None and boundary_mask.any():
                valid_idx = np.where(boundary_mask)[0]
                gfp_valid = gfp[valid_idx]
                local_peaks, _ = find_peaks(gfp_valid, distance=min_dist)
                peaks = valid_idx[local_peaks]
            else:
                peaks, _ = find_peaks(gfp, distance=min_dist)
        maps = data[:, peaks]
        maps /= np.linalg.norm(maps, axis=1, keepdims=True)
        return maps, peaks

    @staticmethod
    def generate_maps_and_peaks(
        preprocessed_folder,
        extension,
        datatype,
        use_percentages=None,
        min_dist=None,
        random_seed=None,
    ):
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
            random_seed (int, optional): Random seed for reproducible random sampling.
                Only used when use_percentages is provided. Defaults to None

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

        # When use_percentages is set together with random_seed, reproducibility
        # is achieved by deriving a distinct per-file seed
        # (``file_seed = random_seed + counter``) below and passing it to
        # ``extract_gfp_peaks_and_maps``. This yields a stable but different
        # random sample within each file across runs.
        for eeg_path in all_preprocessed_paths:
            eeg = data_io.load_eeg(eeg_path, datatype)
            eeg_data = data_io.get_eeg_data(eeg, datatype)

            # For random sampling with seed, we need to pass a different seed for each file
            # to ensure different samples but reproducible results
            if use_percentages is not None and random_seed is not None:
                file_seed = random_seed + counter  # Different seed for each file
                maps, peaks = DataInitializer.extract_gfp_peaks_and_maps(
                    eeg_data, use_percentages, min_dist, file_seed
                )
            else:
                maps, peaks = DataInitializer.extract_gfp_peaks_and_maps(
                    eeg_data, use_percentages, min_dist
                )

            if counter == 0:
                maps2use = maps
                peaks2use = peaks
            else:
                maps2use = np.hstack((maps2use, maps))
                peaks2use = np.hstack((peaks2use, peaks))
            counter = counter + 1

        return maps2use, peaks2use
