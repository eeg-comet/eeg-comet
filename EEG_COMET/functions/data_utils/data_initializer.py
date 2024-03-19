
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import correlate
from functions.data_utils.data_io import DataIO


class DataInitializer:
    """
    The DataInitializer class provides methods for initializing data for microstate analysis.
    """
    def __init__(self):
        pass

    @staticmethod
    def initialize_cluster_centers(maps2use, n_states, initializer):
        """
        Initialize cluster centers for k-means clustering algorithm.

        Args:
            maps2use (numpy.ndarray): Array containing the microstate maps.
            n_states (int): Number of microstate states.
            initializer (str): Initialization method ('K-Means++' or 'Random').

        Returns:
            numpy.ndarray: Initialized cluster centers.
        """

        if initializer == 'K-Means++':
            initial_idx = np.random.choice(np.size(maps2use, 1))
            initial_centers = [maps2use[:, initial_idx]]
            for _ in range(1, n_states):
                dists = np.array(
                    [
                        min(
                            abs(correlate(d, c, mode='valid')[0])
                            for c in initial_centers
                        )
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
    def extract_gfp_peaks_and_maps(data, use_percentages=None, min_dist=None):
        """
        Extract GFP peaks and maps at peaks from EEG data.

        Args:
            data (numpy.ndarray): Array containing the EEG data.
            use_percentages (float, optional): Percentage of data to use for peak extraction. Defaults to None.
            min_dist (int, optional): Minimum distance between peaks. Defaults to None.

        Returns:
            tuple: A tuple containing the extracted maps and peak indices.
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
        """
        Generate GFP maps and peak indices from preprocessed EEG data.

        Args:
            preprocessed_folder (str): Path to the preprocessed EEG data folder.
            extension (str): File extension of the preprocessed EEG data files.
            datatype (str): Data type ('continuous' or 'epoched').
            use_percentages (float, optional): Percentage of data to use for peak extraction. Defaults to None.
            min_dist (int, optional): Minimum distance between peaks. Defaults to None.

        Returns:
            tuple: A tuple containing the generated maps and peak indices.
        """
        data_io = DataIO()
        all_preprocessed_paths, _ = data_io.find_data(preprocessed_folder, extension)
        maps2use, peaks2use = [], []
        counter = 0
        for eeg_path in all_preprocessed_paths:
            eeg = data_io.load_eegs(eeg_path, extension, datatype)
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
