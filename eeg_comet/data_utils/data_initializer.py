"""Data initialization helpers for microstate analysis (EEG-COMET)."""

import numpy as np
from scipy.signal import find_peaks

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
    def initialize_cluster_centers(maps_to_use, n_states, initializer):
        """Initialize cluster centers for k-means clustering in microstate analysis.

        Implements two initialization strategies for microstate clustering:
        1. 'K-Means++': A smart initialization that chooses initial centers with
           probability proportional to distance from existing centers
        2. 'Random': Randomly selects maps as initial centers

        Both methods normalize the resulting centers to unit length.

        Args:
            maps_to_use (numpy.ndarray): Array of potential microstate maps with shape (n_channels, n_samples)
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
            n_candidates = np.size(maps_to_use, 1)
            # Unit-normalise once so that a dot product is the cosine similarity.
            candidates = maps_to_use / (
                np.linalg.norm(maps_to_use, axis=0, keepdims=True) + 1e-12
            )
            initial_idx = np.random.choice(n_candidates)
            initial_centers = [maps_to_use[:, initial_idx]]
            chosen = [candidates[:, initial_idx]]
            for _ in range(1, n_states):
                # Polarity-invariant distance to the nearest chosen centre:
                # closest centre has the largest |correlation|, so the distance
                # is 1 - max|corr|. Sampling weight is D^2 per Arthur & Vassilvitskii
                # (2007). Weighting by |corr| instead would preferentially seed
                # duplicates of centres already chosen.
                sims = np.abs(np.column_stack([candidates.T @ c for c in chosen]))
                dists = 1.0 - np.max(sims, axis=1)
                weights = np.clip(dists, 0.0, None) ** 2
                total = weights.sum()
                if not np.isfinite(total) or total <= 0:
                    # Every candidate coincides with an existing centre; fall
                    # back to a uniform draw rather than dividing by zero.
                    next_idx = np.random.choice(n_candidates)
                else:
                    next_idx = np.random.choice(n_candidates, p=weights / total)
                initial_centers.append(maps_to_use[:, next_idx])
                chosen.append(candidates[:, next_idx])
            initial_centers = np.array(initial_centers)

        else:
            # initializer == 'Random'. Draw from the global RNG, as the
            # K-Means++ branch does, so that seeding via ``np.random.seed`` makes
            # the initialisation reproducible. A fresh ``RandomState(None)`` is
            # seeded from OS entropy and would ignore the configured seed.
            initial_peaks = np.random.choice(
                np.size(maps_to_use, 1), size=n_states, replace=False
            )
            initial_centers = maps_to_use[:, initial_peaks].T

        initial_centers /= np.linalg.norm(initial_centers, axis=1, keepdims=True)
        return initial_centers

    @staticmethod
    def extract_gfp_peaks_and_maps(
        data, data_percentage=None, min_dist=None, random_seed=None, normalize=True
    ):
        """Extract Global Field Power (GFP) peaks and corresponding topographical maps.

        GFP is calculated as the standard deviation across channels at each time point.
        Peaks in the GFP signal represent moments of highest spatial variability and
        are commonly used as the basis for microstate analysis.

        Args:
            data (numpy.ndarray): EEG data array with shape (n_channels, n_timepoints)
            data_percentage (float, optional): If provided, selects timepoints based on percentage:
                - None: Use GFP peak detection with min_dist
                - 0-99: Randomly selects this percentage of timepoints 
                - 100: Uses ALL timepoints (entire data)
                Value should be between 0-100. Defaults to None (use peak detection)
            min_dist (int, optional): Minimum distance between peaks in samples.
                If 0, no minimum distance is enforced. Only used when data_percentage is None
            random_seed (int, optional): Random seed for reproducible random sampling.
                Only used when data_percentage is provided and < 100. Defaults to None
            normalize (bool): If True (default), normalize each topography to unit
                L2 norm (required for clustering). Set False to keep the original
                amplitudes, which GEV needs so its GFP² weighting is meaningful.

        Returns:
            tuple: Contains:
                - maps (numpy.ndarray): Topographical maps at selected timepoints,
                  shape (n_channels, n_selected_points), unit-normalized when
                  ``normalize`` is True
                - peaks (numpy.ndarray): Indices of selected timepoints in the original data

        Notes:
            Three data selection modes:
            1. data_percentage=None: GFP peak detection (with min_dist constraint)
            2. data_percentage<100: Random subset of data points 
            3. data_percentage=100: All data points (entire dataset)
        """
        gfp = np.std(data, axis=0)

        if data_percentage is not None:
            if data_percentage == 100:
                # Use entire data - all time points for clustering
                peaks = np.arange(data.shape[1])
            else:
                # Use random subset of data based on percentage
                n_samples = int(data.shape[1] * (int(data_percentage) / 100))
                if random_seed is not None:
                    # Set random seed for reproducible sampling
                    rng = np.random.RandomState(random_seed)
                    peaks = rng.choice(data.shape[1], size=n_samples, replace=False)
                else:
                    peaks = np.random.choice(data.shape[1], size=n_samples, replace=False)
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
        # Normalise each topography (column) to unit L2 norm across channels for
        # clustering. GEV instead needs the raw amplitudes so its GFP² weighting
        # emphasises high-power moments (normalize=False preserves them).
        if normalize:
            maps = maps / (np.linalg.norm(maps, axis=0, keepdims=True) + 1e-12)
        return maps, peaks

    @staticmethod
    def generate_maps_and_peaks(
        preprocessed_folder,
        extension,
        data_type,
        data_percentage=None,
        min_dist=None,
        random_seed=None,
        normalize=True,
    ):
        """Generate GFP maps and peak indices from multiple preprocessed EEG files.

        Loads all EEG files from the specified folder that match the extension,
        extracts GFP peaks and maps from each file, and concatenates the results.

        Args:
            preprocessed_folder (str): Directory containing preprocessed EEG files
            extension (str): File extension to match (e.g., '.set', '.vhdr')
            data_type (str): Type of EEG data - 'raw' or 'epoched'
            data_percentage (float, optional): Percentage of data points to randomly
                select instead of using peak detection (0-100). Defaults to None
            min_dist (int, optional): Minimum distance between peaks in samples.
                Defaults to None
            random_seed (int, optional): Random seed for reproducible random sampling.
                Only used when data_percentage is provided. Defaults to None
            normalize (bool): If True (default), unit-normalize each topography
                (for clustering). Set False to preserve raw amplitudes for GEV.

        Returns:
            tuple: Contains:
                - maps_to_use (numpy.ndarray): Concatenated topographical maps from all files,
                  shape (n_channels, total_peaks)
                - peaks_to_use (numpy.ndarray): Concatenated indices of peaks from all files

        Notes:
            This method is particularly useful for group-level microstate analysis
            where maps from multiple subjects or recordings need to be combined.
        """
        data_io = DataIO()
        all_preprocessed_paths, _ = data_io.find_data(preprocessed_folder, extension)
        maps_to_use, peaks_to_use = [], []
        counter = 0

        # When data_percentage is set together with random_seed, reproducibility
        # is achieved by deriving a distinct per-file seed
        # (``file_seed = random_seed + counter``) below and passing it to
        # ``extract_gfp_peaks_and_maps``. This yields a stable but different
        # random sample within each file across runs.
        for eeg_path in all_preprocessed_paths:
            eeg = data_io.load_eeg(eeg_path, data_type)
            eeg_data = data_io.get_eeg_data(eeg, data_type)

            # For random sampling with seed, we need to pass a different seed for each file
            # to ensure different samples but reproducible results
            if data_percentage is not None and random_seed is not None:
                file_seed = random_seed + counter  # Different seed for each file
                maps, peaks = DataInitializer.extract_gfp_peaks_and_maps(
                    eeg_data, data_percentage, min_dist, file_seed, normalize=normalize
                )
            else:
                maps, peaks = DataInitializer.extract_gfp_peaks_and_maps(
                    eeg_data, data_percentage, min_dist, normalize=normalize
                )

            if counter == 0:
                maps_to_use = maps
                peaks_to_use = peaks
            else:
                maps_to_use = np.hstack((maps_to_use, maps))
                peaks_to_use = np.hstack((peaks_to_use, peaks))
            counter = counter + 1

        return maps_to_use, peaks_to_use
