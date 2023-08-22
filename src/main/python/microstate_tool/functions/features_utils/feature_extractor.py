
"""
This class provides a set of methods for performing feature extraction on a given segment of data.
The features_utils include microstate coverage, microstate occurrence, microstate duration, transition probability,
and Lempel-Ziv complexity. The class supports both 'static' and 'dynamic' modes for calculating these features_utils.
"""

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

    def _calculate_window_index(self, i):
        """
        Calculate the window index for a given position.

        Args:
            i (int): Position in the segment.

        Returns:
            int: Window index.
        """
        return i // (self.sampling_rate * self.window_size)

    def microstate_coverage(self):
        """
        Calculate the coverage percentage of each element within windows.

        Returns:
            dict or list of dict: Depending on the mode, returns either the average coverage
                                  per element over all windows (mode='static'), or the dynamic
                                  coverage of each element per window (mode='dynamic').
        """
        num_windows = len(self.segment) // (self.sampling_rate * self.window_size)

        window_element_coverage = [{} for _ in range(num_windows)]

        element_counts = Counter(self.segment)

        for i, s in enumerate(self.segment):
            window_index = self._calculate_window_index(i)
            if window_index < num_windows:
                coverage = element_counts[s] / len(self.segment) * 100  # Convert to percentage
                window_element_coverage[window_index][s] = coverage

        if self.mode == 'static':
            # Calculate the average coverage over all windows
            total_coverage = Counter()
            for window_coverage in window_element_coverage:
                total_coverage.update(window_coverage)
            num_windows = len(window_element_coverage)
            average_coverage = {element: min(coverage / num_windows, 100.0) for element, coverage in total_coverage.items()}
            return average_coverage
        elif self.mode == 'dynamic':
            return window_element_coverage
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def microstate_occurrence(self):
        """
        Compute the frequency of occurrence of elements within non-overlapping windows.

        Returns:
            dict or list of dict: Depending on the mode, returns either the average frequency
                                  per symbol over all windows (mode='static'), or the dynamic
                                  frequency of occurrence per window (mode='dynamic').
        """
        num_windows = len(self.segment) // (self.sampling_rate * self.window_size)

        window_symbol_counts = [Counter() for _ in range(num_windows)]

        for i, s in enumerate(self.segment):
            window_index = self._calculate_window_index(i)
            if window_index < num_windows:
                window_symbol_counts[window_index][s] += 1

        freq_per_window = [{symbol: count for symbol, count in window_counts.items()} for window_counts in window_symbol_counts]

        if self.mode == 'static':
            # Calculate the average frequency over all windows
            total_symbol_counts = Counter()
            for window in freq_per_window:
                total_symbol_counts.update(window)
            num_windows = len(freq_per_window)
            average_frequency = {symbol: count / num_windows for symbol, count in total_symbol_counts.items()}
            return average_frequency
        elif self.mode == 'dynamic':
            return freq_per_window
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def microstate_duration(self):
        """
        Compute the duration of each element within non-overlapping windows.

        Returns:
            dict or list of dict: Depending on the mode, returns either the average frequency
                                  per symbol over all windows (mode='static'), or the dynamic
                                  frequency of occurrence per window (mode='dynamic').
        """
        num_windows = len(self.segment) // (self.sampling_rate * self.window_size)

        window_element_durations = [Counter() for _ in range(num_windows)]

        for i, s in enumerate(self.segment):
            window_index = i // (self.sampling_rate * self.window_size)
            if window_index < num_windows:
                window_element_durations[window_index][s] += 1

        duration_per_window = [{element: count * 1000 / (self.sampling_rate * self.window_size) for element, count in window_counts.items()} for window_counts in window_element_durations]

        if self.mode == 'static':
            total_element_durations = Counter()
            for window in duration_per_window:
                total_element_durations.update(window)
            num_windows = len(duration_per_window)
            average_duration = {element: duration / num_windows for element, duration in total_element_durations.items()}
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

