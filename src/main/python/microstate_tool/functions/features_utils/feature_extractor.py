
"""
This class provides a set of methods for performing feature extraction on a given segment of data.
The features_utils include microstate coverage, microstate occurrence, microstate duration, transition probability,
and Lempel-Ziv complexity. The class supports both 'static' and 'dynamic' modes for calculating these features_utils.
"""

import pandas as pd
from collections import Counter, defaultdict
from functions.clustering_utils.microstate_clusterer import MicrostateClusterer


class FeatureExtractor:
    def __init__(self, segment, sampling_rate, window_size, mode='static'):
        """
        Initialize the FeatureExtractor class.

        Args:
            segment (list): A list containing elements to analyze.
            sampling_rate (int): The sampling rate of the segment in Hz.
            window_size (int): The size of the non-overlapping windows in seconds.
            mode (str): The mode for calculation. 'static' returns average values over all windows,
                        'dynamic' returns values for each window.
        """
        self.segment = segment
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.mode = mode
        self.num_windows = len(segment) // (sampling_rate * window_size)

    def _calculate_window_index(self, i):
        """
        Calculate the window index for a given position.

        Args:
            i (int): Position in the segment.

        Returns:
            int: Window index.
        """
        return i // (self.sampling_rate * self.window_size)

    def _remove_consecutive_duplicates(self, str_array):
        """
        Remove consecutive duplicate characters from a string.

        Args:
            str_array (str or list): The string or list of strings to remove duplicates from.

        Returns:
            new_str_array: The new string with consecutive duplicates removed.
        """
        if isinstance(str_array, list):
            str_array = "".join(str_array)

        new_str_array = ""
        prev_char = None

        for char in str_array:
            if char != prev_char:
                new_str_array += str(char)
                prev_char = char

        return new_str_array

    def _initialize_empty_window_data(self):
        """
        Create a dictionary with zero values for all elements in the segment.

        Returns:
            dict: Dictionary with elements as keys and zero values.
        """
        return {element: 0 for element in set(self.segment)}

    def global_explained_variance(self, eeg_data, microstate_maps, microstate_labels=None):
        """
        Compute Global Explained Variance (GEV) for a given EEG data and microstate maps.

        Parameters:
        eeg_data: np.array, shape (n_channels, n_timepoints)
            The EEG data matrix.
        microstate_maps: np.array, shape (n_maps, n_channels)
            The microstate topography used for computation.
        microstate_labels: list of strings, optional
            Labels for each microstate map.

        Returns:
        gev: float or list of float or dict
            The Global Explained Variance.
        """
        if microstate_labels and len(microstate_labels) != microstate_maps.shape[0]:
            raise ValueError("Length of microstate_labels must match the number of microstate maps.")

        window_element_gev = [self._initialize_empty_window_data() for _ in range(self.num_windows)]
        window_size_samples = self.window_size * self.sampling_rate

        if self.mode == 'dynamic':
            gevs = {}
            for i, label in enumerate(microstate_labels):
                gevs[label] = []
                for window_index in range(self.num_windows):
                    window_start = window_index * window_size_samples
                    window_end = (window_index + 1) * window_size_samples
                    window_eeg_data = eeg_data[:, window_start:window_end]

                    # Check if window is empty
                    if window_eeg_data.size == 0:
                        gevs[label].append(0)
                        continue

                    # Compute GEV using the imported function
                    gev = MicrostateClusterer().compute_gev(window_eeg_data, microstate_maps[i, :])
                    window_element_gev[window_index][label] = gev
            return window_element_gev

        elif self.mode == 'static':
            gevs = {}
            for i, label in enumerate(microstate_labels):
                # Compute GEV
                gev = MicrostateClusterer().compute_gev(eeg_data, microstate_maps[i, :])
                gevs[label] = gev
            return gevs

        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def microstate_coverage(self):
        """
        Calculate the coverage percentage of each element within windows.

        Returns:
            dict or list of dict: Depending on the mode, returns either the average coverage
                                  per element over all windows (mode='static'), or the dynamic
                                  coverage of each element per window (mode='dynamic').
        """
        window_element_coverage = [self._initialize_empty_window_data() for _ in range(self.num_windows)]

        for window_index in range(self.num_windows):
            window_start = window_index * self.window_size * self.sampling_rate
            window_end = (window_index + 1) * self.window_size * self.sampling_rate
            window_segment = self.segment[window_start:window_end]
            element_counts = Counter(window_segment)

            total_elements = sum(element_counts.values())
            if total_elements > 0:
                for element, count in element_counts.items():
                    coverage = count / total_elements * 100
                    window_element_coverage[window_index][element] = coverage

        if self.mode == 'static':
            total_coverage = Counter()
            for window_coverage in window_element_coverage:
                total_coverage.update(window_coverage)
            num_windows = len(window_element_coverage)
            average_coverage = {element: min(coverage / num_windows, 100.0) for element, coverage in
                                total_coverage.items()}
            return average_coverage
        elif self.mode == 'dynamic':
            return window_element_coverage
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def microstate_occurrence(self):
        """
        Compute the number of times an element changes from another element within non-overlapping windows.

        Returns:
            dict or list of dict: Depending on the mode, returns either the average count
                                  per symbol over all windows (mode='static'), or the dynamic
                                  count of changes per window (mode='dynamic').
        """
        window_change_counts = [self._initialize_empty_window_data() for _ in range(self.num_windows)]
        total_element_counts = Counter()

        for window_index in range(self.num_windows):
            window_start = window_index * (self.sampling_rate * self.window_size)
            window_end = window_start + (self.sampling_rate * self.window_size)
            window_segment = self.segment[window_start:window_end]
            window_segment = self._remove_consecutive_duplicates(window_segment)

            element_counts = Counter(window_segment)
            window_change_counts[window_index] = element_counts
            total_element_counts.update(element_counts)

        if self.mode == 'static':
            num_windows = len(window_change_counts)
            average_counts = {element: count / num_windows for element, count in total_element_counts.items()}
            return average_counts
        elif self.mode == 'dynamic':
            return window_change_counts
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def microstate_duration(self):
        """
        Compute the duration of each element within non-overlapping windows.

        Returns:
            dict or list of dict: Depending on the mode, returns either the average duration
                                  per symbol over all windows (mode='static'), or the dynamic
                                  duration per window (mode='dynamic').
        """
        window_element_durations = [Counter() for _ in range(self.num_windows)]

        for i, s in enumerate(self.segment):
            window_index = i // (self.sampling_rate * self.window_size)
            if window_index < self.num_windows:
                window_element_durations[window_index][s] += 1

        duration_per_window = [{element: count * self.window_size for element, count in window_counts.items()}
                               for window_counts in window_element_durations]

        if self.mode == 'static':
            total_element_durations = Counter()
            for window in duration_per_window:
                total_element_durations.update(window)
            num_windows = len(duration_per_window)
            average_duration = {element: duration / num_windows for element, duration in
                                total_element_durations.items()}
            return average_duration
        elif self.mode == 'dynamic':
            return duration_per_window
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")


    def compute_transition_probabilities(self):
        """
        Compute the transition probabilities for a given segment.

        Returns:
        transition_prob: dict
            The transition probabilities.
        """

        transitions = defaultdict(int)
        total_transitions = 0

        for i in range(len(self.segment) - 1):
            transition_label = f"{self.segment[i]}_{self.segment[i + 1]}"
            transitions[transition_label] += 1
            total_transitions += 1

        probabilities = {pair: count / total_transitions for pair, count in transitions.items()}
        return probabilities

    def lempel_ziv_complexity(self):
        """
        Calculate Lempel-Ziv complexity using the LZ76 algorithm and a sliding window implementation.

        Returns:
        dict: Dictionary containing either the average complexity over all windows (mode='static'),
              or a list of complexities for each window (mode='dynamic').
        """

        window_size_samples = int(self.sampling_rate * self.window_size)
        num_windows = len(self.segment) // window_size_samples
        window_complexities = [0.0 for _ in range(self.num_windows)]  # Initialize list with zeros

        for window_index in range(num_windows):
            window_start = window_index * window_size_samples
            window_end = window_start + window_size_samples
            window_segment = self.segment[window_start:window_end]
            window_complexities[window_index] = self._calculate_single_window_complexity(window_segment)

        if self.mode == 'static':
            avg_complexity = sum(window_complexities) / num_windows if num_windows > 0 else 0.0
            return avg_complexity

        elif self.mode == 'dynamic':
            return window_complexities

        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def _calculate_single_window_complexity(self, segment):
        """
        Calculate Lempel-Ziv complexity for a single window using the LZ76 algorithm.

        Args:
            segment (list): Input sequence of symbols.

        Returns:
            float: Lempel-Ziv complexity of the input sequence.
        """
        i, k, l = 0, 1, 1
        c, k_max = 1, 1

        while True:
            if segment[i + k - 1] == segment[l + k - 1]:
                k = k + 1
                if l + k > len(segment):
                    c = c + 1
                    break
            else:
                if k > k_max:
                    k_max = k
                i = i + 1
                if i == l:
                    c = c + 1
                    l = l + k_max
                    if l + 1 > len(segment):
                        break
                    else:
                        i = 0
                        k = 1
                        k_max = 1
                else:
                    k = 1
        return c / len(segment)

    def extract_microstate_features(self, filename, feature_list, eeg_data=None, microstate_maps=None, microstate_labels=None):
        """
        Extracts a set of microstate features from EEG data segments, given a list of feature identifiers.
        The function operates in two modes: 'static' and 'dynamic'.

        In 'static' mode, the function returns aggregated feature values over the entire segment for each feature.
        In 'dynamic' mode, it returns features calculated over a set of windows within the segment.

        Note: The 'TP' (Transition Probability) feature is only supported in 'static' mode.
              If 'TP' is requested in 'dynamic' mode, it will be ignored.

        Args:
            filename (str): Name of the file being analyzed.
            feature_list (list): List of feature identifiers to extract.
                                 Possible values include 'COV', 'OCC', 'DUR', 'GEV', and 'TP'.
            eeg_data (array, optional): Raw EEG data. Required if 'GEV' is among the features to be extracted.
            microstate_maps (array, optional): Microstate maps. Required if 'GEV' is among the features to be extracted.
            microstate_labels (array, optional): Microstate labels. Required if 'GEV' is among the features to be extracted.

        Returns:
            pd.DataFrame: Extracted features organized based on the chosen mode ('static' or 'dynamic').
                          The DataFrame will have different columns depending on the mode.
        """

        features_dict = []

        if 'COV' in feature_list:
            extracted_microstate_coverage = self.microstate_coverage()
            features_dict.append(('COV', extracted_microstate_coverage))
        if 'OCC' in feature_list:
            extracted_microstate_occurrence = self.microstate_occurrence()
            features_dict.append(('OCC', extracted_microstate_occurrence))
        if 'DUR' in feature_list:
            extracted_microstate_duration = self.microstate_duration()
            features_dict.append(('DUR', extracted_microstate_duration))
        if 'GEV' in feature_list:
            extracted_explained_variance = self.global_explained_variance(eeg_data, microstate_maps, microstate_labels)
            features_dict.append(('GEV', extracted_explained_variance))

        # Only add TP if the mode is not dynamic
        if 'TP' in feature_list and self.mode != 'dynamic':
            extracted_microstate_transition_probability = self.compute_transition_probabilities()
            features_dict.append(('TP', extracted_microstate_transition_probability))

        if 'LZC' in feature_list:
            extracted_microstate_complexity = self.lempel_ziv_complexity()
            features_dict.append(('LZC', extracted_microstate_complexity))

        # Create a list to hold the data
        output_features_data = []

        for feature, feature_data in features_dict:
            if self.mode == 'static':
                if isinstance(feature_data, dict):
                    for element, value in feature_data.items():
                        output_features_data.append([filename, f"{feature}_{element}", value])
                else:
                    output_features_data.append([filename, feature, feature_data])
            elif self.mode == 'dynamic':
                for window_index, window_data in enumerate(feature_data):
                    if isinstance(window_data, dict):
                        for element, value in window_data.items():
                            output_features_data.append([filename, window_index, f"{feature}_{element}", value])
                    else:
                        output_features_data.append([filename, window_index, feature, window_data])

        if self.mode == 'static':
            columns = ['Filename', 'Feature', 'Value']
        elif self.mode == 'dynamic':
            columns = ["Filename", "Window_index", "Feature", "Value"]

        output_features_df = pd.DataFrame(output_features_data, columns=columns)

        if self.mode == 'static':
            output_features_df = output_features_df.pivot_table(index='Filename', columns='Feature', values='Value').reset_index()
        elif self.mode == 'dynamic':
            output_features_df = output_features_df.pivot_table(index=["Filename", "Window_index"], columns="Feature", values="Value").reset_index()

        return output_features_df

