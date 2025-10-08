"""Feature extraction utilities for EEG-COMET microstate analyses."""

import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from backfitting_utils.segmentation_io import SegmentationIO
from clustering_utils.microstate_clusterer import MicrostateClusterer
from features_utils.feature_helper import FeatureHelper


class FeatureExtractor:
    """The FeatureExtractor class provides methods for extracting microstate features from EEG data."""

    def __init__(
        self, input_sequence, sampling_rate, sliding_window_size=1, feature_mode="averaged"
    ):
        """Initialize the FeatureExtractor class.

        Args:
            input_sequence (list): The input sequence of EEG data.
            sampling_rate (int): The sampling rate of the EEG data.
            sliding_window_size (int): The sliding window size in seconds. Defaults to 1 second.
            feature_mode (str, optional): The feature extraction mode ('averaged' or 'sliding'). Defaults to 'averaged'.
        """
        # Ensure input_sequence contains only hashable types (strings)
        if isinstance(input_sequence, np.ndarray):
            if input_sequence.ndim == 1:
                # 1D array - convert each element to string
                self.input_sequence = [str(item) for item in input_sequence]
            elif input_sequence.ndim == 2:
                # 2D array - keep as numpy array but ensure string dtype
                self.input_sequence = input_sequence.astype(str)
            else:
                # Higher dimensions - flatten and convert to strings
                self.input_sequence = [str(item) for item in input_sequence.flatten()]
        elif isinstance(input_sequence, list):
            # List - ensure all elements are strings
            self.input_sequence = [str(item) for item in input_sequence]
        else:
            # Other types - try to convert to list of strings
            try:
                self.input_sequence = [str(item) for item in input_sequence]
            except Exception:
                self.input_sequence = [str(input_sequence)]

        self.sampling_rate = sampling_rate
        self.sliding_window_size = sliding_window_size
        self.feature_mode = feature_mode

        # Initialize optional export caches to avoid defining attributes outside __init__
        self._rof_data_for_export = {}
        self._rtf_data_for_export = {}

        # Calculate num_windows based on the processed input_sequence
        if hasattr(self.input_sequence, "__len__"):
            # Ensure integer division for window calculation
            window_size_samples = int(sampling_rate * sliding_window_size)
            self.num_windows = len(self.input_sequence) // window_size_samples
        else:
            self.num_windows = 1

    @property
    def rof_data_for_export(self) -> dict:
        """Public read-only access to ROF export data cache."""
        return self._rof_data_for_export

    @property
    def rtf_data_for_export(self) -> dict:
        """Public read-only access to RTF export data cache."""
        return self._rtf_data_for_export

    def _get_flat_sequence(self):
        """Return the input sequence flattened to a 1-D list of hashable elements (strings)."""
        if isinstance(self.input_sequence, np.ndarray):
            return [str(elem) for elem in self.input_sequence.flatten()]
        return [str(elem) for elem in self.input_sequence]

    def global_explained_variance(self, eeg_data, microstate_maps, microstate_labels=None):
        """Compute Global Explained Variance (GEV) for a given EEG data and microstate maps.
        GEV measures how much of the total variance in the EEG data is explained by each microstate.
        The sum of GEV across all microstates should equal 100%.

        Args:
            eeg_data (array-like): The EEG data.
            microstate_maps (array-like): The microstate maps.
            microstate_labels (list, optional): The labels for the microstate maps. Defaults to None.

        Returns:
            list or dict: The GEV values for each window and microstate label.
                         For 'sliding' mode: list of dicts with GEV per window
                         For 'averaged' mode: dict with overall GEV per microstate
        """
        # Check if required data is available
        if eeg_data is None or microstate_maps is None:
            if self.feature_mode == "sliding":
                return [
                    FeatureHelper().initialize_empty_window_data(self.input_sequence)
                    for _ in range(self.num_windows)
                ]
            return {}

        # Ensure microstate_labels are available
        if microstate_labels is None:
            microstate_labels = [f"MS{i+1}" for i in range(microstate_maps.shape[0])]

        if len(microstate_labels) != microstate_maps.shape[0]:
            raise ValueError(
                "Length of microstate_labels must match the number of microstate maps."
            )

        # Create MicrostateClusterer instance with number of states from maps
        clusterer = MicrostateClusterer(n_states=microstate_maps.shape[0])

        # Initialize GEV storage based on mode
        if self.feature_mode == "sliding":
            window_element_gev = [
                FeatureHelper().initialize_empty_window_data(self.input_sequence)
                for _ in range(self.num_windows)
            ]
            window_size_samples = int(self.sliding_window_size * self.sampling_rate)

            # Compute GEV for each window
            for window_index in range(self.num_windows):
                window_start = window_index * window_size_samples
                window_end = (window_index + 1) * window_size_samples
                window_eeg_data = eeg_data[:, window_start:window_end]
                window_labels = self.input_sequence[window_start:window_end]

                if window_eeg_data.size == 0:
                    continue

                # Calculate GFP for the window
                gfp = np.std(window_eeg_data, axis=0)
                gfp_squared_sum = np.sum(gfp**2)

                # Initialize GEV for each microstate in this window
                window_gevs = {label: 0.0 for label in microstate_labels}

                # Group time points by microstate label
                unique_labels = np.unique(window_labels)
                for label in unique_labels:
                    if label in microstate_labels:  # Only process valid labels
                        # Get time points for this microstate
                        label_mask = np.array(window_labels) == label
                        if not np.any(label_mask):
                            continue

                        # Get data and GFP for these time points
                        label_data = window_eeg_data[:, label_mask]
                        label_gfp = gfp[label_mask]

                        # Get corresponding map
                        map_idx = microstate_labels.index(label)
                        current_map = microstate_maps[map_idx : map_idx + 1, :]

                        # Calculate correlation at each time point
                        map_corr = clusterer.corr_vectors(label_data, current_map.T)

                        # Calculate GEV for this microstate
                        window_gevs[label] = (
                            np.sum((label_gfp * map_corr) ** 2) / gfp_squared_sum * 100
                        )

                # Store computed GEVs
                window_element_gev[window_index] = window_gevs

            return window_element_gev

        if self.feature_mode == "averaged":
            # Calculate overall GFP
            gfp = np.std(eeg_data, axis=0)
            gfp_squared_sum = np.sum(gfp**2)

            # Initialize GEV for each microstate
            gevs = {label: 0.0 for label in microstate_labels}

            # Group time points by microstate label
            unique_labels = np.unique(self.input_sequence)
            for label in unique_labels:
                if label in microstate_labels:  # Only process valid labels
                    # Get time points for this microstate
                    label_mask = np.array(self.input_sequence) == label
                    if not np.any(label_mask):
                        continue

                    # Get data and GFP for these time points
                    label_data = eeg_data[:, label_mask]
                    label_gfp = gfp[label_mask]

                    # Get corresponding map
                    map_idx = microstate_labels.index(label)
                    current_map = microstate_maps[map_idx : map_idx + 1, :]

                    # Calculate correlation at each time point
                    map_corr = clusterer.corr_vectors(label_data, current_map.T)

                    # Calculate GEV for this microstate
                    gevs[label] = np.sum((label_gfp * map_corr) ** 2) / gfp_squared_sum * 100

            return gevs
        raise ValueError("Invalid mode. Supported modes are 'averaged' and 'sliding'.")

    def microstate_coverage(self):
        """Calculate the coverage percentage of each element.

        Returns:
            dict: If feature_mode is 'averaged', returns a dictionary with the overall coverage percentage for each element.
                  If feature_mode is 'sliding', returns a list of dictionaries where each dictionary represents the
                  coverage percentages for each element in a window.
        """
        if self.feature_mode == "averaged":
            try:
                element_counts = Counter(self._get_flat_sequence())
                total_elements = len(self._get_flat_sequence())
                return {
                    element: (count / total_elements) * 100
                    for element, count in element_counts.items()
                }
            except Exception:
                raise

        elif self.feature_mode == "sliding":
            try:
                window_element_coverage = [
                    FeatureHelper().initialize_empty_window_data(self.input_sequence)
                    for _ in range(self.num_windows)
                ]

                # Calculate window size in samples as integer
                window_size_samples = int(self.sliding_window_size * self.sampling_rate)

                for window_index in range(self.num_windows):
                    window_start = window_index * window_size_samples
                    window_end = (window_index + 1) * window_size_samples
                    window_input_sequence = self.input_sequence[window_start:window_end]

                    element_counts = Counter(window_input_sequence)
                    total_elements = sum(element_counts.values())
                    if total_elements > 0:
                        for element, count in element_counts.items():
                            coverage = (count / total_elements) * 100
                            window_element_coverage[window_index][element] = coverage

                return window_element_coverage
            except Exception:
                raise

        else:
            raise ValueError("Invalid mode. Supported modes are 'averaged' and 'sliding'.")

    def microstate_occurrence(self):
        """Compute the frequency of occurrence (in Hz) of each element per second.

        Returns:
            dict or list:
                - If feature_mode is 'averaged', returns a dictionary with the frequency (Hz) of each element
                  across the entire input_sequence.
                - If feature_mode is 'sliding', returns a list of dictionaries, where each dictionary represents
                  the frequency (Hz) of each element in a 1-second window.
        """
        samples_per_second = self.sampling_rate
        if self.feature_mode == "averaged":
            sequence_without_repeats = FeatureHelper().remove_repetition_sequence(
                self._get_flat_sequence()
            )
            total_element_counts = Counter(sequence_without_repeats)
            total_duration_seconds = len(self._get_flat_sequence()) / samples_per_second
            return {
                element: count / total_duration_seconds
                for element, count in total_element_counts.items()
            }
        if self.feature_mode == "sliding":
            # Use custom sliding window size
            window_size_samples = int(self.sliding_window_size * samples_per_second)
            num_windows = len(self._get_flat_sequence()) // window_size_samples
            window_change_counts = []
            for window_idx in range(num_windows):
                window_start = window_idx * window_size_samples
                window_end = (window_idx + 1) * window_size_samples
                window_input_sequence = self._get_flat_sequence()[window_start:window_end]
                window_input_sequence = FeatureHelper().remove_repetition_sequence(
                    window_input_sequence
                )
                element_counts = Counter(window_input_sequence)
                window_change_counts.append(
                    {
                        element: count
                        / self.sliding_window_size  # Divide by window size in seconds
                        for element, count in element_counts.items()
                    }
                )
            return window_change_counts
        raise ValueError("Invalid mode. Supported modes are 'averaged' and 'sliding'.")

    def microstate_duration(self):
        """Compute the average duration of each element uninterrupted in the data.

        Returns:
            dict or list: If feature_mode is 'averaged',
                            returns a dictionary with the average duration for each element.
                          If feature_mode is 'sliding' and window_size is not None,
                            returns a list of dictionaries, where each dictionary represents the average duration
                            for each element in a window.
        """

        def calculate_average_durations(input_sequence):
            """Helper function to calculate average durations for a given sequence."""
            durations = {}
            current_element = None
            current_duration = 0

            for item in input_sequence:
                if item != current_element:
                    if current_element is not None:
                        if current_element not in durations:
                            durations[current_element] = []
                        durations[current_element].append(current_duration)
                    current_element = item
                    current_duration = 1
                else:
                    current_duration += 1

            # Handle the last element in the sequence
            if current_element not in durations:
                durations[current_element] = []
            durations[current_element].append(current_duration)

            # Calculate the average duration for each element and convert to milliseconds
            return {
                key: (sum(value) / len(value)) * 1000 / self.sampling_rate
                for key, value in durations.items()
            }

        if self.feature_mode == "averaged":
            # Static mode: calculate average duration for the whole sequence
            average_durations = calculate_average_durations(self._get_flat_sequence())

        elif self.feature_mode == "sliding" and self.sliding_window_size is not None:
            # Dynamic mode: calculate average duration for each window
            windows = []
            window_start = 0
            # Calculate window size in samples as integer
            window_size_samples = int(self.sliding_window_size * self.sampling_rate)
            window_end = window_size_samples

            # Slide through the input sequence in window-sized chunks
            while window_start < len(self.input_sequence):
                window_sequence = self.input_sequence[window_start:window_end]
                if window_sequence:
                    # Calculate average duration for this window and store the result
                    window_avg_duration = calculate_average_durations(window_sequence)
                    windows.append(window_avg_duration)

                # Move to the next window
                window_start = window_end
                window_end += window_size_samples

            average_durations = windows

        else:
            raise ValueError("Invalid mode or missing window size for 'sliding' mode")

        return average_durations

    def compute_transition_probabilities(self):
        """Compute the transition probabilities for a given input_sequence.

        Returns:
            dict: A dictionary containing the transition probabilities for each pair of elements in the input_sequence.
        """
        transitions = defaultdict(int)
        total_transitions = 0
        flat_seq = self._get_flat_sequence()
        for i in range(len(flat_seq) - 1):
            current_element = flat_seq[i]
            next_element = flat_seq[i + 1]
            # Skip self-transitions
            if current_element == next_element:
                continue
            transition_label = f"{current_element}_{next_element}"
            transitions[transition_label] += 1
            total_transitions += 1
        return {pair: count / total_transitions for pair, count in transitions.items()}

    def entropy_rate(self, min_samples=None, kmax=6):
        """Calculate entropy rate using k-history method.
        For sliding windows, calculates entropy rate for each window.
        For averaged mode, calculates entropy rate for the entire sequence.

        Args:
            min_samples (int, optional): Minimum number of samples to use for consistent comparison.
                                       If None, uses the full sequence length.
            kmax (int, optional): Maximum history length to consider. Defaults to 6.

        Returns:
            float or list: If feature_mode is 'averaged', returns the entropy rate of the entire input_sequence.
                          If feature_mode is 'sliding', returns a list of entropy rates for each window.
        """
        # Get number of unique symbols
        ns = len(set(self.input_sequence))

        # Ensure consistent sample size for averaged mode
        if self.feature_mode == "averaged":
            consistent_sequence = FeatureHelper().ensure_consistent_samples(
                self._get_flat_sequence(), min_samples
            )
            h_rate, _ = FeatureHelper().compute_entropy_rate(consistent_sequence, ns, kmax)
            return h_rate

        # For sliding mode, ensure each window has consistent samples
        window_entropies, window_size_samples = FeatureHelper().initialize_dynamic_windows(
            self.input_sequence, self.sampling_rate, self.sliding_window_size
        )

        for window_index in range(len(window_entropies)):
            window_start = window_index * window_size_samples
            window_end = window_start + window_size_samples
            window_input_sequence = self.input_sequence[window_start:window_end]
            # Ensure consistent samples for each window
            window_input_sequence = FeatureHelper().ensure_consistent_samples(
                window_input_sequence, min_samples
            )
            # Calculate entropy rate for this window
            h_rate, _ = FeatureHelper().compute_entropy_rate(window_input_sequence, ns, kmax)
            window_entropies[window_index] = h_rate

        if self.feature_mode == "sliding":
            return window_entropies
        raise ValueError("Invalid mode. Supported modes are 'averaged' and 'sliding'.")

    def lempel_ziv_complexity(self, min_samples=None):
        """Calculate Lempel-Ziv complexity using the LZ76 algorithm.

        Args:
            min_samples (int, optional): Minimum number of samples to use for consistent comparison.
                                       If None, uses the full sequence length.

        Returns:
            float: The Lempel-Ziv complexity of the input sequence.
        """
        # First remove repetitions, then ensure consistent sample size
        sequence_without_repeats = FeatureHelper().remove_repetition_sequence(
            self._get_flat_sequence()
        )
        consistent_sequence = FeatureHelper().ensure_consistent_samples(
            sequence_without_repeats, min_samples
        )
        return FeatureHelper().compute_lempel_ziv_complexity(consistent_sequence)

    def entropy_representation(self, word_size):
        """Calculate the entropy representation of different classes of entropies and their ratio compared to
        theoretical dictionary based on the MicroSynt pipeline.

        Args:
            word_size (int): The size of the word for entropy calculation.

        Returns:
            dict or list: If feature_mode is 'averaged',
                            returns a dictionary with the entropy representation for each entropy class.
                          If feature_mode is 'sliding',
                            returns a list of dictionaries, where each dictionary represents the entropy representation
                            for each entropy class in a window.
        """
        # TODO: not completed
        # word_size = 5
        (
            window_entropy_representations,
            window_size_samples,
        ) = FeatureHelper().initialize_dynamic_windows(
            self.input_sequence, self.sampling_rate, self.sliding_window_size
        )
        for window_index in range(len(window_entropy_representations)):
            window_start = window_index * window_size_samples
            window_end = window_start + window_size_samples
            window_input_sequence = self.input_sequence[window_start:window_end]
            window_entropy_representations[window_index] = FeatureHelper().calculate_entropy(
                window_input_sequence
            )
        if self.feature_mode == "averaged":
            overall_entropy_representations, _ = MicroSynt().sequence_analysis(
                self._get_flat_sequence(), word_size
            )
            return overall_entropy_representations
        if self.feature_mode == "sliding":
            return window_entropy_representations

        raise ValueError("Invalid mode. Supported modes are 'averaged' and 'sliding'.")

    def relative_occurrence_frequency(self, time_array, input_sequence=None):
        """Compute baseline-corrected relative occurrence frequency (ROF).

        Designed for epoched TMS-EEG data. Steps:
        1) Count occurrences for each microstate at each time point across trials
        2) Average across trials to create temporal profiles
        3) Apply centered log-ratio (CLR) transform for compositional data
        4) Baseline-correct using pre-TMS period (-1000ms to -10ms)

        Args:
            time_array (numpy.ndarray): Time points in milliseconds for the epoch
            input_sequence (numpy.ndarray | list | None): Optional labels array; if None, uses self.input_sequence

        Returns:
            dict: Baseline-corrected ROF values and related metrics
        """
        seq = input_sequence if input_sequence is not None else self.input_sequence

        # Ensure numpy 2D array of string type
        seq_array = np.asarray(seq)
        if seq_array.dtype == object:
            seq_array = seq_array.astype(str)

        if seq_array.ndim != 2:
            raise ValueError(
                f"ROF calculation requires epoched data with shape (trials, timepoints). Current shape: {seq_array.shape}"
            )

        return FeatureHelper().compute_relative_occurrence_frequency(
            seq_array, time_array, microstates=None
        )

    def relative_transition_frequency(self, time_array=None, input_sequence=None):
        """Extract baseline-corrected relative transition frequencies (RTF).

        Steps:
            1) Build transition series by counting transitions at each time point
            2) Average over trials
            3) Calculate averages per time window
            4) Apply baseline correction

        Args:
            time_array (array-like, optional): Time points in ms. If None, uses default.
            input_sequence (array-like, optional): Sequence to use. If None, uses self.input_sequence.

        Returns:
            dict: Baseline-corrected RTF values and related metrics
        """
        seq = input_sequence if input_sequence is not None else self.input_sequence

        # Ensure numpy 2D array of string type
        seq_array = np.asarray(seq)
        if seq_array.dtype == object:
            seq_array = seq_array.astype(str)

        if seq_array.ndim != 2:
            raise ValueError(
                f"RTF calculation requires epoched data with shape (trials, timepoints). Current shape: {seq_array.shape}"
            )

        return FeatureHelper().compute_relative_transition_frequency(
            seq_array, time_array, microstates=None, time_window_ranges=None
        )

    def hurst_exponent(self, min_samples=50, max_samples=2500, num_scales=50):
        """Calculate Hurst exponent using Detrended Fluctuation Analysis (DFA).
        For sliding windows, calculates Hurst exponent for each window.
        For averaged mode, calculates Hurst exponent for the entire sequence.

        The calculation follows Van de Ville et al. (2010) and von Wegner et al. (2016):
        - Uses 50 logarithmically spaced time scales over 50-2500 samples
        - Creates random walks by partitioning states into two subsets with ±1 values
        - Averages Hurst exponents across all possible partitions
        - For 4 states: uses (2,2) partitions
        - For 5 states: uses (2,3) partitions

        Args:
            min_samples (int): Minimum window size (default: 50 samples = 200ms at 250Hz)
            max_samples (int): Maximum window size (default: 2500 samples = 10s at 250Hz)
            num_scales (int): Number of logarithmically spaced scales (default: 50)

        Returns:
            float or list: If feature_mode is 'averaged', returns the Hurst exponent of the entire input_sequence.
                          If feature_mode is 'sliding', returns a list of Hurst exponents for each window.
        """
        if self.feature_mode == "averaged":
            return FeatureHelper().calculate_hurst_exponent(
                self._get_flat_sequence(),
                min_samples=min_samples,
                max_samples=max_samples,
                num_scales=num_scales,
            )

        # For sliding mode
        window_hurst_exponents, window_size_samples = FeatureHelper().initialize_dynamic_windows(
            self.input_sequence, self.sampling_rate, self.sliding_window_size
        )

        for window_index in range(len(window_hurst_exponents)):
            window_start = window_index * window_size_samples
            window_end = window_start + window_size_samples
            window_input_sequence = self.input_sequence[window_start:window_end]

            # Calculate Hurst exponent for this window
            window_hurst_exponents[window_index] = FeatureHelper().calculate_hurst_exponent(
                window_input_sequence,
                min_samples=min_samples,
                max_samples=max_samples,
                num_scales=num_scales,
            )

        if self.feature_mode == "sliding":
            return window_hurst_exponents
        raise ValueError("Invalid mode. Supported modes are 'averaged' and 'sliding'.")

    def extract_microstate_features(
        self,
        filename,
        feature_list,
        eeg_data=None,
        microstate_maps=None,
        microstate_labels=None,
        word_size=2,
        min_samples=None,
        time_array=None,
        epoched_labels=None,
    ):
        """Extracts a set of microstate features from EEG data input_sequences, given a list of feature identifiers.
        The function operates in two modes: 'averaged' and 'sliding'.

        In 'averaged' mode, the function returns aggregated feature values over the entire input_sequence for each feature.
        In 'sliding' mode, it returns features calculated over a set of windows within the input_sequence.

        Args:
            filename (str): The filename of the EEG data.
            feature_list (list): The list of feature identifiers to extract.
            eeg_data (array-like, optional): The EEG data. Defaults to None.
            microstate_maps (array-like, optional): The microstate maps. Defaults to None.
            microstate_labels (list, optional): The labels for the microstate maps. Defaults to None.
            word_size (int, optional): The size of the word for entropy calculation. Defaults to 2.
            min_samples (int, optional): Minimum number of samples to use for consistent comparison in SE and LZC.
                                       If None, uses the full sequence length.
            time_array (numpy.ndarray, optional): Time points in milliseconds for epoched data. Required for ROF calculation.
            epoched_labels (numpy.ndarray, optional): Epoched labels for ROF calculation.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted microstate features.
        """
        features_dict = []

        if "COV" in feature_list:
            extracted_microstate_coverage = self.microstate_coverage()
            features_dict.append(("COV", extracted_microstate_coverage))
        if "OCC" in feature_list:
            extracted_microstate_occurrence = self.microstate_occurrence()
            features_dict.append(("OCC", extracted_microstate_occurrence))
        if "DUR" in feature_list:
            extracted_microstate_duration = self.microstate_duration()
            features_dict.append(("DUR", extracted_microstate_duration))
        if "GEV" in feature_list:
            extracted_explained_variance = self.global_explained_variance(
                eeg_data, microstate_maps, microstate_labels
            )
            features_dict.append(("GEV", extracted_explained_variance))
        if "ER" in feature_list:
            extracted_entropy = self.entropy_rate(min_samples=min_samples)
            features_dict.append(("ER", extracted_entropy))
        if "ERR" in feature_list:
            extracted_entropy_representation = self.entropy_representation(word_size)
            features_dict.append(("ERR", extracted_entropy_representation))
        if "HE" in feature_list:
            extracted_hurst = self.hurst_exponent()
            # Replace None with NaN for consistency
            if extracted_hurst is None:
                extracted_hurst = np.nan
            features_dict.append(("HE", extracted_hurst))
        if "ROF" in feature_list:
            try:
                if time_array is None or epoched_labels is None:
                    print(
                        f"Warning: ROF feature skipped for {filename} - requires time_array and epoched_labels for epoched TMS-EEG data"
                    )
                else:
                    extracted_rof = self.relative_occurrence_frequency(
                        time_array, input_sequence=epoched_labels
                    )
                    features_dict.append(("ROF", extracted_rof))
            except ValueError as e:
                print(f"Warning: ROF feature skipped for {filename} - {str(e)}")

        if "RTF" in feature_list:
            try:
                if time_array is None or epoched_labels is None:
                    print(
                        f"Warning: RTF feature skipped for {filename} - requires time_array and epoched_labels for epoched TMS-EEG data"
                    )
                else:
                    extracted_rtf = self.relative_transition_frequency(
                        time_array, input_sequence=epoched_labels
                    )
                    features_dict.append(("RTF", extracted_rtf))
            except ValueError as e:
                print(f"Warning: RTF feature skipped for {filename} - {str(e)}")
            except Exception as e:
                print(f"Warning: RTF feature skipped for {filename} - {str(e)}")

        # Only add TP and LZC if the mode is not sliding
        if "LZC" in feature_list and self.feature_mode != "sliding":
            extracted_microstate_complexity = self.lempel_ziv_complexity(min_samples=min_samples)
            features_dict.append(("LZC", extracted_microstate_complexity))
        if "TP" in feature_list and self.feature_mode != "sliding":
            extracted_microstate_transition_probability = self.compute_transition_probabilities()
            features_dict.append(("TP", extracted_microstate_transition_probability))

        # Create a list to hold the data
        output_features_data = []

        # Handle ROF and RTF features specially due to their complex return structures
        special_features_dict = []
        non_special_features_dict = []

        for feature, feature_data in features_dict:
            if feature in ["ROF", "RTF"]:
                # ROF and RTF return complex dictionaries, handle them separately
                special_features_dict.append((feature, feature_data))
            else:
                non_special_features_dict.append((feature, feature_data))

        # Process non-special features first
        for feature, feature_data in non_special_features_dict:
            if self.feature_mode == "sliding":
                if not isinstance(feature_data, (list, tuple)):
                    feature_data = [feature_data]
                for window_index, window_data in enumerate(feature_data):
                    if isinstance(window_data, dict):
                        output_features_data.extend(
                            [filename, window_index, f"{feature}_{element}", value]
                            for element, value in window_data.items()
                        )
                    else:
                        output_features_data.append([filename, window_index, feature, window_data])
            else:  # self.feature_mode == 'averaged'
                if isinstance(feature_data, dict):
                    output_features_data.extend(
                        [filename, f"{feature}_{element}", value]
                        for element, value in feature_data.items()
                    )
                else:
                    output_features_data.append([filename, feature, feature_data])

        # Process special features separately
        for feature, feature_data in special_features_dict:
            if self.feature_mode == "averaged":
                if feature == "ROF":
                    # For averaged mode, add summary statistics from ROF
                    microstates = feature_data.get("microstates", [])
                    baseline_corrected_rof = feature_data.get("occurrences_clr_bc", {})

                    if baseline_corrected_rof:
                        # Add mean baseline-corrected ROF for each microstate
                        for microstate in microstates:
                            if microstate in baseline_corrected_rof:
                                mean_rof = np.mean(baseline_corrected_rof[microstate])
                                output_features_data.append(
                                    [filename, f"ROF_mean_{microstate}", mean_rof]
                                )

                                # Add std baseline-corrected ROF for each microstate
                                std_rof = np.std(baseline_corrected_rof[microstate])
                                output_features_data.append(
                                    [filename, f"ROF_std_{microstate}", std_rof]
                                )

                elif feature == "RTF":
                    # For averaged mode, add summary statistics from RTF
                    microstates = feature_data.get("microstates", [])
                    transition_averages_bc = feature_data.get("transition_averages_bc", {})

                    if transition_averages_bc:
                        # Add transition averages for each time window
                        for window_name, transition_matrix in transition_averages_bc.items():
                            if isinstance(transition_matrix, np.ndarray):
                                # Extract individual transitions
                                for i, from_state in enumerate(microstates):
                                    for j, to_state in enumerate(microstates):
                                        if i != j:  # Skip self-transitions
                                            transition_value = transition_matrix[i, j]
                                            output_features_data.append(
                                                [
                                                    filename,
                                                    f"RTF_{window_name}_{from_state}_{to_state}",
                                                    transition_value,
                                                ]
                                            )

            elif self.feature_mode == "sliding":
                # For sliding mode, this would typically not be used since ROF/RTF are for epoched data
                # But if needed, could add time-resolved values here
                pass

        # Store full ROF data for separate export if ROF was computed
        if special_features_dict:
            # Store the full ROF data in the DataFrame metadata for later export
            if not hasattr(self, "_rof_data_for_export"):
                self._rof_data_for_export = {}
            # Store the full RTF data in the DataFrame metadata for later export
            if not hasattr(self, "_rtf_data_for_export"):
                self._rtf_data_for_export = {}

            # Process ROF data
            for feature, feature_data in special_features_dict:
                if feature == "ROF":
                    self._rof_data_for_export[filename] = feature_data
                elif feature == "RTF":
                    self._rtf_data_for_export[filename] = feature_data

        if self.feature_mode == "sliding":
            columns = ["Filename", "Window_index", "Feature", "Value"]
            output_features_df = pd.DataFrame(output_features_data, columns=columns)
            output_features_df = output_features_df.pivot_table(
                index=["Filename", "Window_index"], columns="Feature", values="Value"
            ).reset_index()
        else:  # self.feature_mode == 'averaged'
            columns = ["Filename", "Feature", "Value"]
            output_features_df = pd.DataFrame(output_features_data, columns=columns)
            output_features_df = output_features_df.pivot_table(
                index="Filename", columns="Feature", values="Value"
            ).reset_index()

        return output_features_df


class MicroSynt:
    """The MicroSynt class provides methods for analyzing entropy distribution in an input sequence and
    generating surrogate statistics based on input sequences.
    """

    def __init__(self) -> None:
        """Initialize a MicroSynt instance. The class is stateless; no setup required."""
        # Intentionally empty initializer for clarity and linter compliance
        return

    @staticmethod
    def sequence_analysis(input_sequence, word_size):
        """Analyze the entropy distribution in an input sequence.

        Args:
            input_sequence (str or list): The input sequence of symbols.
            word_size (int): The size of the word for entropy calculation.

        Returns:
            tuple: A tuple containing the entropy representation percentages for the real and theoretical dictionaries.
        """
        # Check if the input_sequence is a list, if so, join it into a string
        if isinstance(input_sequence, list):
            input_sequence = "".join(input_sequence)

        theoretical_dictionary = FeatureHelper().generate_theoretical_dictionary(
            input_sequence, word_size
        )
        real_dictionary, sequence_representation_real = FeatureHelper().generate_real_dictionary(
            input_sequence, word_size
        )

        # Compute entropy for each word in real_dictionary
        word_entropy = {}
        for word in real_dictionary:
            entropy = FeatureHelper().calculate_entropy(word)
            word_entropy[word] = entropy

        # Get unique entropy values and sort them (lowest first)
        unique_entropies_sorted = sorted(set(word_entropy.values()))

        # Convert sequence_representation to entropy classes
        entropy_representation_real = {
            f"EntropyClass{i + 1}": 0 for i in range(len(unique_entropies_sorted))
        }

        for word, count in sequence_representation_real.items():
            entropy = word_entropy[word]
            entropy_class = f"EntropyClass{unique_entropies_sorted.index(entropy) + 1}"
            entropy_representation_real[entropy_class] += count

        # Calculate percentages for each entropy class
        total_words_real = sum(entropy_representation_real.values())
        entropy_representation_percentage_real = {
            ec: count / total_words_real for ec, count in entropy_representation_real.items()
        }

        # Convert sequence_representation for theoretical dictionary to entropy classes
        entropy_representation_theoretical = {
            f"EntropyClass{i + 1}": 0 for i in range(len(unique_entropies_sorted))
        }

        for word in theoretical_dictionary:
            entropy = FeatureHelper().calculate_entropy(word)
            entropy_class = f"EntropyClass{unique_entropies_sorted.index(entropy) + 1}"
            entropy_representation_theoretical[entropy_class] += 1

        # Calculate percentages for each entropy class
        total_words_theoretical = sum(entropy_representation_theoretical.values())
        entropy_representation_percentage_theoretical = {
            ec: count / total_words_theoretical
            for ec, count in entropy_representation_theoretical.items()
        }

        return entropy_representation_percentage_real, entropy_representation_percentage_theoretical

    def surrogate_statistics(self, input_sequence, word_size, repeats=1000):
        """Generate surrogate statistics based on input sequences.

        Args:
            input_sequence (str or list): The input sequence of symbols.
            word_size (int): The size of the word for entropy calculation.
            repeats (int, optional): The number of surrogate sequences to generate. Defaults to 1000.

        Returns:
            dict: A dictionary containing the distributions of representation ratios for each entropy class.
        """
        distributions = {}

        for _ in range(repeats):
            # Generate surrogate sequence
            surrogate_sequence = FeatureHelper().generate_synthetic_sequence(
                input_sequence, "surrogate"
            )
            surrogate_no_permanence_sequence = FeatureHelper().remove_repetition_sequence(
                surrogate_sequence
            )

            # Generate entropy representations
            surrogate_entropy_representation, surrogate_entropy_representation_theoretical = (
                self.sequence_analysis(surrogate_no_permanence_sequence, word_size)
            )

            # Calculate representation ratios
            surrogate_entropy_representation_ratio = (
                FeatureHelper().calculate_representation_ratios(
                    surrogate_entropy_representation, surrogate_entropy_representation_theoretical
                )
            )

            # Append the ratio to the distribution for each class separately
            for key, value in surrogate_entropy_representation_ratio.items():
                # Rename the key
                new_key = key.replace("RepresentationRatio", "Distribution")
                if new_key not in distributions:
                    distributions[new_key] = []
                distributions[new_key].append(value)

        return distributions


class FeatureExtractionCoordinator:
    """Coordinator class for feature extraction that organizes results by mode and type."""

    def __init__(self):
        """Initialize the coordinator."""

    def extract_features(
        self,
        segmentation,
        feature_list,
        feature_mode,
        feature_types,
        sliding_window_size=1,
        pre_window_size=1,
        post_window_size=1,
        min_samples=None,
    ):
        """Extract features from segmentation data and organize by mode and type.

        Parameters:
        -----------
        segmentation : dict
            Segmentation data containing 'labels', 'time', etc.
        feature_list : list
            List of features to extract
        feature_mode : list
            List of feature modes ('averaged', 'sliding')
        feature_types : list
            List of feature types ('real', 'surrogate', 'random')
        sliding_window_size : int
            Window size for sliding window analysis
        pre_window_size : int
            Pre-window size for epoched data
        post_window_size : int
            Post-window size for epoched data
        min_samples : int, optional
            Minimum number of samples to use for consistent comparison in SE and LZC.
            If None, uses the full sequence length.

        Returns:
        --------
        dict
            Dictionary organized by feature_mode and feature_type
        """
        results = {}

        # Get basic info from segmentation
        labels = segmentation.get("labels", [])
        time = segmentation.get("time", [])

        if not labels:
            return results

        # Calculate sampling rate
        sampling_rate = 1000 / (time[1] - time[0]) if len(time) > 1 else 250

        # Check if this is epoched data with original 2D structure
        eeg_data = segmentation.get("eeg_data", None)
        is_epoched_data = (
            eeg_data is not None and len(eeg_data.shape) == 3
        )  # (trials, channels, timepoints)

        # Process each feature mode
        for mode in feature_mode:
            if mode not in results:
                results[mode] = {}

            # Special handling for epoched data with sliding mode - extract TMS pre/post features
            if is_epoched_data and mode == "sliding":
                results[mode] = self._extract_epoched_sliding_features(
                    segmentation, feature_list, feature_types, sampling_rate, min_samples
                )
                continue

            # Process each feature type (standard processing for non-epoched or averaged mode)
            for feature_type in feature_types:
                if feature_type not in results[mode]:
                    results[mode][feature_type] = []

                # Get the input sequence based on feature type
                if feature_type == "real":
                    input_sequence = labels
                elif feature_type == "surrogate":
                    # Create surrogate data by shuffling
                    input_sequence = labels.copy()
                    np.random.shuffle(input_sequence)
                elif feature_type == "random":
                    # Create random data with same length and unique values
                    unique_labels = list(set(labels))
                    input_sequence = np.random.choice(unique_labels, size=len(labels))
                else:
                    input_sequence = labels

                # Create feature extractor for this combination
                feature_extractor = FeatureExtractor(
                    input_sequence=input_sequence,
                    sampling_rate=sampling_rate,
                    sliding_window_size=sliding_window_size,
                    feature_mode=mode,
                )

                # Extract features for this file
                filename = segmentation.get("filename", "unknown")

                # Get additional data if needed
                microstate_maps = segmentation.get("microstate_maps", None)
                microstate_labels = segmentation.get("microstate_labels", None)

                # Pass time_array if available for ROF calculation
                # For ROF/RTF with epoched data, use single-epoch time array if available
                time_array = segmentation.get("time_single_epoch", segmentation.get("time", None))

                # Determine epoched labels if present
                epoched_labels_param = segmentation.get("epoched_labels", None)
                if (
                    isinstance(epoched_labels_param, np.ndarray)
                    and epoched_labels_param.dtype == object
                ):
                    epoched_labels_param = epoched_labels_param.astype(str)

                extracted_df = feature_extractor.extract_microstate_features(
                    filename=filename,
                    feature_list=feature_list,
                    eeg_data=eeg_data,
                    microstate_maps=microstate_maps,
                    microstate_labels=microstate_labels,
                    min_samples=min_samples,
                    time_array=time_array,
                    epoched_labels=epoched_labels_param,
                )

                results[mode][feature_type].append(extracted_df)

                # Store ROF data if it was computed
                if feature_extractor.rof_data_for_export:
                    if "rof_data" not in results[mode]:
                        results[mode]["rof_data"] = {}
                    results[mode]["rof_data"].update(feature_extractor.rof_data_for_export)

                # Store RTF data if it was computed
                if feature_extractor.rtf_data_for_export:
                    if "rtf_data" not in results[mode]:
                        results[mode]["rtf_data"] = {}
                    results[mode]["rtf_data"].update(feature_extractor.rtf_data_for_export)

        return results

    def _extract_epoched_sliding_features(
        self, segmentation, feature_list, feature_types, sampling_rate, min_samples
    ):
        """Extract features for epoched TMS data with sliding windows.

        This method implements specialized feature extraction for TMS-EEG epoched data when
        the sliding features checkbox is enabled. Instead of standard sliding windows, it
        extracts features from specific time periods around the TMS pulse:

        - Pre-TMS: -1000ms to -10ms (avoiding TMS artifact period)
        - Post-TMS: +20ms to +1000ms (avoiding immediate TMS artifact)

        Features are extracted separately for each trial, allowing analysis of:
        - Coverage percentage of microstate A 1 second before TMS to -10ms before TMS
        - Post-TMS coverage from 20ms to 1 second after TMS
        - All other selected microstate features for both periods

        The output includes real_sliding_features with trial-by-trial pre/post TMS data.

        Parameters:
        -----------
        segmentation : dict
            Segmentation data containing labels, time, EEG data, etc.
        feature_list : list
            List of features to extract (COV, OCC, DUR, GEV, etc.)
        feature_types : list
            List of feature types ('real', 'surrogate', 'random')
        sampling_rate : float
            Sampling rate in Hz
        min_samples : int, optional
            Minimum number of samples for consistent comparison

        Returns:
        --------
        dict
            Results organized by feature_type with pre/post TMS features
            Each result includes Window_Type, Trial, Time_Start_ms, Time_End_ms columns
        """
        results = {}

        # Get data
        eeg_data = segmentation.get("eeg_data", None)  # Shape: (trials, channels, timepoints)
        time_array = np.array(segmentation.get("time", []))
        filename = segmentation.get("filename", "unknown")
        microstate_maps = segmentation.get("microstate_maps", None)
        microstate_labels = segmentation.get("microstate_labels", None)

        if eeg_data is None or len(time_array) == 0:
            return {feature_type: [] for feature_type in feature_types}

        # Load original segmentation data from file to get trial structure
        segmentation_data = self._load_original_segmentation_data(segmentation)

        if segmentation_data is None:
            return {feature_type: [] for feature_type in feature_types}

        n_trials = segmentation_data.shape[0]

        # Define TMS windows in milliseconds
        pre_tms_start = -1000  # -1000ms
        pre_tms_end = -10  # -10ms
        post_tms_start = 20  # +20ms
        post_tms_end = 1000  # +1000ms

        # Find time indices for windows
        pre_start_idx = np.searchsorted(time_array, pre_tms_start)
        pre_end_idx = np.searchsorted(time_array, pre_tms_end)
        post_start_idx = np.searchsorted(time_array, post_tms_start)
        post_end_idx = np.searchsorted(time_array, post_tms_end)

        # Ensure valid indices
        pre_start_idx = max(0, pre_start_idx)
        pre_end_idx = min(len(time_array), pre_end_idx)
        post_start_idx = max(0, post_start_idx)
        post_end_idx = min(len(time_array), post_end_idx)

        # Process each feature type
        for feature_type in feature_types:
            results[feature_type] = []

            # Extract features for each trial
            for trial_idx in range(n_trials):
                # Get trial segmentation labels
                trial_labels = segmentation_data[trial_idx, :].tolist()
                trial_eeg = eeg_data[trial_idx, :, :] if eeg_data.shape[0] > trial_idx else None

                # Process pre-TMS window
                if pre_end_idx > pre_start_idx:
                    pre_labels = trial_labels[pre_start_idx:pre_end_idx]
                    pre_eeg = (
                        trial_eeg[:, pre_start_idx:pre_end_idx] if trial_eeg is not None else None
                    )

                    # Apply feature type transformation
                    if feature_type == "real":
                        pre_input_sequence = pre_labels
                    elif feature_type == "surrogate":
                        pre_input_sequence = pre_labels.copy()
                        np.random.shuffle(pre_input_sequence)
                    elif feature_type == "random":
                        unique_labels = list(set(trial_labels))
                        pre_input_sequence = np.random.choice(unique_labels, size=len(pre_labels))
                    else:
                        pre_input_sequence = pre_labels

                    # Extract features for pre-TMS window
                    if len(pre_input_sequence) > 0:
                        pre_extractor = FeatureExtractor(
                            input_sequence=pre_input_sequence,
                            sampling_rate=sampling_rate,
                            feature_mode="averaged",  # Use averaged mode for each window
                        )

                        pre_df = pre_extractor.extract_microstate_features(
                            filename=f"{filename}_trial{trial_idx+1}_preTMS",
                            feature_list=feature_list,
                            eeg_data=pre_eeg,
                            microstate_maps=microstate_maps,
                            microstate_labels=microstate_labels,
                            min_samples=min_samples,
                            time_array=(
                                time_array[pre_start_idx:pre_end_idx]
                                if "ROF" in feature_list
                                else None
                            ),
                            epoched_labels=getattr(
                                pre_extractor, "_rof_epoched_labels", None
                            ),  # Pass epoched_labels to extractor
                        )

                        # Add window type and trial info
                        pre_df["Window_Type"] = "PreTMS"
                        pre_df["Trial"] = trial_idx + 1
                        pre_df["Time_Start_ms"] = pre_tms_start
                        pre_df["Time_End_ms"] = pre_tms_end

                        results[feature_type].append(pre_df)

                # Process post-TMS window
                if post_end_idx > post_start_idx:
                    post_labels = trial_labels[post_start_idx:post_end_idx]
                    post_eeg = (
                        trial_eeg[:, post_start_idx:post_end_idx] if trial_eeg is not None else None
                    )

                    # Apply feature type transformation
                    if feature_type == "real":
                        post_input_sequence = post_labels
                    elif feature_type == "surrogate":
                        post_input_sequence = post_labels.copy()
                        np.random.shuffle(post_input_sequence)
                    elif feature_type == "random":
                        unique_labels = list(set(trial_labels))
                        post_input_sequence = np.random.choice(unique_labels, size=len(post_labels))
                    else:
                        post_input_sequence = post_labels

                    # Extract features for post-TMS window
                    if len(post_input_sequence) > 0:
                        post_extractor = FeatureExtractor(
                            input_sequence=post_input_sequence,
                            sampling_rate=sampling_rate,
                            feature_mode="averaged",  # Use averaged mode for each window
                        )

                        post_df = post_extractor.extract_microstate_features(
                            filename=f"{filename}_trial{trial_idx+1}_postTMS",
                            feature_list=feature_list,
                            eeg_data=post_eeg,
                            microstate_maps=microstate_maps,
                            microstate_labels=microstate_labels,
                            min_samples=min_samples,
                            time_array=(
                                time_array[post_start_idx:post_end_idx]
                                if "ROF" in feature_list
                                else None
                            ),
                            epoched_labels=getattr(
                                post_extractor, "_rof_epoched_labels", None
                            ),  # Pass epoched_labels to extractor
                        )

                        # Add window type and trial info
                        post_df["Window_Type"] = "PostTMS"
                        post_df["Trial"] = trial_idx + 1
                        post_df["Time_Start_ms"] = post_tms_start
                        post_df["Time_End_ms"] = post_tms_end

                        results[feature_type].append(post_df)

        return results

    @staticmethod
    def _load_original_segmentation_data(segmentation):
        """Load the original segmentation data to get trial structure.

        Parameters:
        -----------
        segmentation : dict
            Segmentation data containing filename

        Returns:
        --------
        numpy.ndarray or None
            Original segmentation array with shape (trials, timepoints)
        """
        try:
            # Check if original segmentation array is already provided
            if "original_segmentation_array" in segmentation:
                return segmentation["original_segmentation_array"]

            filename = segmentation.get("filename", "")
            if not filename:
                return None

            # Try to find and load the original segmentation file
            # This is a simplified approach - you may need to adjust paths
            segmentation_io = SegmentationIO()

            # Try different possible paths/formats
            possible_extensions = [".csv", ".pkl", ".hdf", ".json"]

            for ext in possible_extensions:
                try:
                    # Try loading with different extensions
                    seg_path = filename.replace(os.path.splitext(filename)[1], ext)
                    if os.path.exists(seg_path):
                        return segmentation_io.load_segmentation(seg_path, import_format=ext)
                except Exception:
                    continue

            # If file loading fails, try to reconstruct from available data
            # This is a fallback - assumes single trial flattened into labels
            labels = segmentation.get("labels", [])
            if labels:
                return np.array([labels])  # Single trial

        except Exception as e:
            print(f"Warning: Could not load original segmentation data: {e}")

        return None
