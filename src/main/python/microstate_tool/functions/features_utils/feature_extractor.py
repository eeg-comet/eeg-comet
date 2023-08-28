
"""
This class provides a set of methods for performing feature extraction on a given segment of data.
The features_utils include microstate coverage, microstate occurrence, microstate duration, transition probability,
and Lempel-Ziv complexity. The class supports both 'static' and 'dynamic' modes for calculating these features_utils.
"""

import pandas as pd
from collections import Counter, defaultdict


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
                new_str_array += char
                prev_char = char

        return new_str_array

    def _initialize_empty_window_data(self):
        """
        Create a dictionary with zero values for all elements in the segment.

        Returns:
            dict: Dictionary with elements as keys and zero values.
        """
        return {element: 0 for element in set(self.segment)}

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

    def transition_probability(self):
        """
        Calculate transition probability using a sliding window implementation.

        Returns:
            dict or list of dict: Depending on the mode, returns either the average transition probabilities
                                  over all windows (mode='static'), or the transition probabilities of each
                                  window (mode='dynamic').
        """
        n = len(self.segment)
        if n == 0:
            return {} if self.mode == 'static' else []

        window_size_samples = int(self.sampling_rate * self.window_size)

        if self.mode == 'static':
            num_windows = n // window_size_samples
            total_probabilities = defaultdict(float)

            for i in range(num_windows):
                window_segment = self.segment[i * window_size_samples : (i + 1) * window_size_samples]
                window_probabilities = self._calculate_single_window_probabilities(window_segment)
                for pair, prob in window_probabilities.items():
                    total_probabilities[pair] += prob

            for pair in total_probabilities:
                total_probabilities[pair] /= num_windows

            return total_probabilities

        elif self.mode == 'dynamic':
            num_windows = n // window_size_samples
            window_probabilities = []

            for i in range(num_windows):
                window_segment = self.segment[i * window_size_samples : (i + 1) * window_size_samples]
                window_probabilities.append(self._calculate_single_window_probabilities(window_segment))

            return window_probabilities
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def _calculate_single_window_probabilities(self, segment):
        """
        Calculate transition probabilities for a single window.

        Args:
            segment (list): Input sequence of symbols.

        Returns:
            dict: Transition probabilities of the input sequence.
        """
        transitions = defaultdict(int)
        total_transitions = 0

        for i in range(len(segment) - 1):
            transitions[(segment[i], segment[i + 1])] += 1
            total_transitions += 1

        probabilities = {pair: count / total_transitions for pair, count in transitions.items()}
        return probabilities

    def lempel_ziv_complexity(self):
        """
        Calculate Lempel-Ziv complexity using the LZ76 algorithm and a sliding window implementation.

        Returns:
            float or list of float: Depending on the mode, returns either the average complexity
                                    over all windows (mode='static'), or the complexity of each
                                    window (mode='dynamic').
        """
        n = len(self.segment)
        if n == 0:
            return 0.0 if self.mode == 'static' else []

        window_size_samples = int(self.sampling_rate * self.window_size)

        if self.mode == 'static':
            num_windows = n // window_size_samples
            total_complexity = 0.0

            for i in range(num_windows):
                window_segment = self.segment[i * window_size_samples : (i + 1) * window_size_samples]
                window_complexity = self._calculate_single_window_complexity(window_segment)
                total_complexity += window_complexity

            return total_complexity / num_windows if num_windows > 0 else 0.0

        elif self.mode == 'dynamic':
            num_windows = n // window_size_samples
            window_complexities = []

            for i in range(num_windows):
                window_segment = self.segment[i * window_size_samples : (i + 1) * window_size_samples]
                window_complexities.append(self._calculate_single_window_complexity(window_segment))

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

    def extract_features(self, filename, feature_list):
        """
        Extract features from the provided segment data using the specified feature extraction methods.

        Args:
            filename (str): Name of the file being analyzed.
            feature_list (list): List of feature identifiers to extract (e.g., ['COV', 'OCC', 'DUR']).

        Returns:
            DataFrame: Extracted features organized based on the chosen mode ('static' or 'dynamic').
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
        if 'TP' in feature_list:
            extracted_microstate_transition_probability = self.transition_probability()
            features_dict.append(extracted_microstate_transition_probability)
        if 'LZC' in feature_list:
            extracted_microstate_complexity = self.lempel_ziv_complexity()
            features_dict.append(extracted_microstate_complexity)

        # Create a list to hold the data
        output_features_data = []

        for feature, feature_data in features_dict:
            if self.mode == 'static':
                for element, value in feature_data.items():
                    output_features_data.append([filename, f"{feature}_{element}", value])
            elif self.mode == 'dynamic':
                for window_index, window_data in enumerate(feature_data):
                    for element, value in window_data.items():
                        output_features_data.append([filename, window_index, f"{feature}_{element}", value])

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

