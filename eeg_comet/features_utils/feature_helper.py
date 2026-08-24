"""Helper utilities for computing microstate features (EEG-COMET)."""

import itertools
import math
import random
from collections import Counter

import numpy as np
from scipy.stats import gmean


class FeatureHelper:
    """The FeatureHelper class provides utility methods for microstate feature extraction."""

    def __init__(self) -> None:
        """Initialize a FeatureHelper instance. No state is maintained."""
        return

    @staticmethod
    def ensure_consistent_samples(input_sequence, min_samples=None):
        """Ensure consistent number of samples for entropy and complexity calculations.
        If min_samples is provided, uses that as the target length, otherwise uses the input length.
        Truncates longer sequences to match the shortest sequence length.

        Args:
            input_sequence (list or str): The input sequence of elements.
            min_samples (int, optional): Minimum number of samples to use. If None, uses full sequence.

        Returns:
            list or str: The input sequence truncated to the consistent length.
        """
        if min_samples is not None and min_samples > 0:
            # Truncate to specified minimum samples
            if len(input_sequence) > min_samples:
                return input_sequence[:min_samples]
            if len(input_sequence) < min_samples:
                raise ValueError(
                    f"Input sequence length {len(input_sequence)} is less than required minimum {min_samples}"
                )
            return input_sequence
        return input_sequence

    @staticmethod
    def _joint_entropy(input_sequence, n_symbols, history_length):
        """Shannon's joint entropy over a sliding history window.

        Args:
            input_sequence: symbolic time series
            n_symbols: number of symbols
            history_length: length of the k-history

        Returns:
            hk: joint entropy for the k-history
        """
        N = len(input_sequence)
        # history_length-dimensional array for k-history frequencies
        f = np.zeros(tuple(history_length * [n_symbols]))

        # Convert sequence to integers if they're not already
        sequence_int = np.zeros(len(input_sequence), dtype=int)
        unique_symbols = list(set(input_sequence))
        symbol_to_int = {symbol: i for i, symbol in enumerate(unique_symbols)}
        for i, symbol in enumerate(input_sequence):
            sequence_int[i] = symbol_to_int[symbol]

        for t in range(N - history_length):
            # Get consecutive symbols and convert to tuple of indices
            idx = tuple(sequence_int[t : t + history_length])
            f[idx] += 1.0

        f /= N - history_length  # normalize distribution
        return -np.sum(f[f > 0] * np.log(f[f > 0]))

    def compute_entropy_rate(self, input_sequence, n_symbols, k_max=6):
        """Calculate entropy rate using the k-history method.
        Fits a line to joint entropy values for different k, slope = entropy rate.

        Args:
            input_sequence: symbolic time series
            n_symbols: number of symbols
            k_max: maximum history length to consider

        Returns:
            h_rate: entropy rate (slope of joint entropy vs k)
            excess_entropy: excess entropy (y-intercept)
        """
        h_ = np.zeros(k_max)
        for k in range(k_max):
            h_[k] = self._joint_entropy(input_sequence, n_symbols, k + 1)
        ks = np.arange(1, k_max + 1)
        # Fit line to get entropy rate (slope)
        h_rate, excess_entropy = np.polyfit(ks, h_, 1)
        return h_rate, excess_entropy

    @staticmethod
    def initialize_empty_window_data(input_sequence):
        """Create a dictionary with zero values for all elements in the input_sequence.

        Args:
            input_sequence (list or str): The input sequence of elements.

        Returns:
            dict: A dictionary with zero values for each element in the input_sequence.
        """
        try:
            unique_elements = set(input_sequence)
            return {element: 0 for element in unique_elements}
        except Exception:
            # Fallback: convert all elements to strings
            input_sequence_str = [str(item) for item in input_sequence]
            unique_elements = set(input_sequence_str)
            return {element: 0 for element in unique_elements}

    @staticmethod
    def initialize_dynamic_windows(input_sequence, sampling_rate, window_size):
        """Divide the input sequence into fixed-size windows and initialize a list to store feature values for each window.

        Args:
            input_sequence (list or str): The input sequence of elements.
            sampling_rate (int): The sampling rate of the input sequence.
            window_size (int): The size of the window in seconds.

        Returns:
            tuple: A tuple containing a list to store feature values for each window and the sample size of each window.
        """
        # Calculate the number of samples in each window
        window_size_samples = int(sampling_rate * window_size)

        # Calculate the total number of windows
        n_windows = len(input_sequence) // window_size_samples

        # Initialize a list with zeros for each window
        windows = [0.0] * n_windows
        return windows, window_size_samples

    @staticmethod
    def remove_repetition_sequence(input_sequence):
        """Collapse each run of identical labels to a single entry.

        Elements are compared whole rather than character by character, so
        multi-character labels such as ``"NaN"`` or ``"MS1"`` survive intact.

        Args:
            input_sequence (list or str): The input sequence of elements.

        Returns:
            list: The input sequence with consecutive repetitions collapsed.
        """
        collapsed = []
        prev_element = _SENTINEL = object()

        for element in input_sequence:
            if element != prev_element:
                collapsed.append(element)
                prev_element = element

        return collapsed

    @staticmethod
    def compute_lempel_ziv_complexity(input_no_permanence_sequence):
        """Calculate Lempel-Ziv complexity for a single window using the LZ76 algorithm.

        Args:
            input_no_permanence_sequence (str): The input sequence with consecutive repetitions removed.

        Returns:
            float: The Lempel-Ziv complexity of the input sequence.
        """
        # Initialize variables
        n = len(input_no_permanence_sequence)
        # LZ76 needs at least three symbols to index i+k and label+k, and the
        # n/log2(n) normaliser is undefined at n == 1. Short sequences arise
        # whenever a recording collapses to one or two runs.
        if n < 3:
            return 0.0
        i, k, label, c, k_max = 0, 1, 1, 1, 1

        # Iterate until a break condition is met
        while True:
            # Check if the current characters at indices i+k and l+k are different
            if input_no_permanence_sequence[i + k] != input_no_permanence_sequence[label + k]:
                # Update k_max if necessary
                if k > k_max:
                    k_max = k
                # Move to the next position i
                i += 1
                # Check if i reaches l (a new block)
                if i == label:
                    # Increment the complexity
                    c += 1
                    # Move l to the end of the current block
                    label += k_max
                    if label + 1 > n - 1:
                        break
                    # Reset indices and k_max for the next block
                    i = 0
                    k_max = 1
                k = 1
            else:
                # Increment k if the characters are the same
                k += 1
                # Check if the end of the sequence is reached
                if label + k > n - 1:
                    # Increment the complexity and exit the loop
                    c += 1
                    break

        # Calculate b (maximum complexity)
        b = 1.0 * n / math.log(n, 2)
        # Return the normalized complexity
        return c / b

    @staticmethod
    def centered_log_ratio(x, delta=0.5):
        """Apply centered log-ratio transformation to compositional data.

        This function applies the centered log-ratio (CLR) transformation to a
        compositional vector, efficiently handling zeros using the multiplicative
        replacement method.

        Args:
            x (numpy.ndarray): Compositional vector containing proportions of microstate occurrences
            delta (float): Scaling factor that determines the proportion of the smallest
                         non-zero component used for replacing zeros (default: 0.5)

        Returns:
            numpy.ndarray: CLR-transformed vector
        """
        # Ensure input is numpy array
        x = np.asarray(x)

        # Get number of components
        D = len(x)

        # Identify zero elements
        zero_mask = x == 0
        n_zeros = np.sum(zero_mask)

        # Handle zeros using multiplicative replacement
        if n_zeros > 0:
            # Total sum of non-zero components
            x_nonzero_sum = np.sum(x[~zero_mask])

            # Initialize replacement vector
            x_replaced = x.copy()

            # Calculate replacement value
            epsilon = delta * np.min(x[x > 0]) / D

            # Replace zeros with epsilon
            x_replaced[zero_mask] = epsilon

            # Adjust non-zero components to maintain compositional constraint
            if x_nonzero_sum > 0:
                x_replaced[~zero_mask] = x[~zero_mask] - (epsilon * n_zeros) * (
                    x[~zero_mask] / x_nonzero_sum
                )

            # Ensure data remains compositional (sums to original total)
            x_replaced = x_replaced / np.sum(x_replaced) * np.sum(x)
        else:
            # No zeros to replace
            x_replaced = x

        # Apply CLR transformation
        # Calculate geometric mean of replaced vector
        geo_mean = gmean(x_replaced)

        # Perform CLR transformation
        return np.log(x_replaced / geo_mean)

    @staticmethod
    def compute_relative_occurrence_frequency(input_sequence, time_array, microstates=None, baseline_window=None):
        """Extract relative occurrence frequencies (ROF) - baseline-corrected CLR values.

        This function calculates ROF for different EEG microstates using the following approach:
        1. Calculates occurrence proportion for each microstate at each time point
        2. Applies CLR transformation with multiplicative replacement for zeros
        3. Applies baseline correction by subtracting baseline median from all time points

        Args:
            input_sequence (numpy.ndarray): Epoched microstate data with shape (trials, timepoints)
            time_array (numpy.ndarray): Time points in milliseconds
            microstates (list, optional): List of microstate labels. If None, inferred from data.
            baseline_window (list, optional): Baseline time window [start, end] in milliseconds. 
                                             Defaults to [-1000, -10].

        Returns:
            dict: Dictionary containing:
                - 'occurrences': Raw occurrence proportions for each microstate
                - 'occurrences_clr': CLR-transformed occurrences
                - 'occurrences_clr_bc': Baseline-corrected CLR occurrences
                - 'baseline_indices': Indices corresponding to baseline period
                - 'microstates': List of microstate labels
        """
        # Ensure time_array is a numpy array
        time_array = np.asarray(time_array)

        if input_sequence.ndim != 2:
            raise ValueError("Input sequence must be 2D array with shape (trials, timepoints)")

        # Use provided baseline window or default to [-1000, -10] ms
        if baseline_window is None:
            baseline_window = [-1000, -10]
        
        baseline_start = baseline_window[0]
        baseline_end = baseline_window[1]

        # Find baseline indices
        baseline_indices = np.where((time_array >= baseline_start) & (time_array <= baseline_end))[
            0
        ]

        if len(baseline_indices) == 0:
            raise ValueError("No baseline period found in time array")

        # Get microstates from data if not provided
        if microstates is None:
            unique_vals = np.unique(input_sequence.flatten())

            # Convert to hashable types (strings) - should already be strings from SegmentationIO
            microstates = [str(val) for val in unique_vals]
        else:
            # Ensure provided microstates are hashable
            microstates = [str(ms) for ms in microstates]

        n_trials = input_sequence.shape[0]

        # Calculate occurrence proportion for each microstate
        occurrences = {}
        for _i, microstate in enumerate(microstates):
            try:
                # Input sequence should now be string array, so direct comparison should work
                index = input_sequence == microstate
                # Calculate proportion at each time point
                occurrence = np.sum(index, axis=0) / n_trials
                occurrences[microstate] = occurrence
            except Exception:
                raise

        # Combine all microstate occurrences into matrix
        all_occ = np.column_stack([occurrences[ms] for ms in microstates])

        # Apply CLR transformation
        clr_occ = np.zeros_like(all_occ)
        for jj in range(all_occ.shape[0]):
            try:
                clr_occ[jj, :] = FeatureHelper.centered_log_ratio(all_occ[jj, :], delta=0.5)
            except Exception as e:
                print(f"Error in CLR transformation at timepoint {jj}: {e}")
                print(f"Input values: {all_occ[jj, :]}")
                raise

        # Baseline correction: subtract median of PRE-event from ALL timepoints
        # baseline_indices contains indices of pre-event timepoints
        baseline_shift_clr = np.median(clr_occ[baseline_indices, :], axis=0)
        baseline_corr_clr = clr_occ - baseline_shift_clr

        # Store results
        occurrences_clr = {}
        occurrences_clr_bc = {}
        for i, microstate in enumerate(microstates):
            occurrences_clr[microstate] = clr_occ[:, i]
            occurrences_clr_bc[microstate] = baseline_corr_clr[:, i]

        return {
            "occurrences": occurrences,
            "occurrences_clr": occurrences_clr,
            "occurrences_clr_bc": occurrences_clr_bc,
            "baseline_indices": baseline_indices,
            "microstates": microstates,
            "time_ms": time_array,  # store time array for export
        }

    @staticmethod
    def compute_relative_transition_frequency(
        input_sequence, time_array, microstates=None, time_window_ranges=None
    ):
        """Extract relative transition frequencies (RTF) - transition probabilities with baseline correction.

        This function calculates RTF by:
        1. Creating transition time series by counting transitions at each time point
        2. Normalizing by number of trials (for this subject)
        3. Calculating average transitions for each time window
        4. Computing percentage change from baseline: ((post - baseline) / baseline) × 100

        Normalization: Each subject's transitions are divided by that subject's trial count,
        making RTF values comparable across subjects with different numbers of trials.

        Args:
            input_sequence (numpy.ndarray): Epoched microstate data with shape (trials, timepoints)
            time_array (numpy.ndarray): Time points in milliseconds
            microstates (list, optional): List of microstate labels. If None, inferred from data.
            time_window_ranges (dict, optional): Dictionary with time window definitions.
                                               Default: {'baseline': [-1000, -10], 'post_tms': [20, 1000]}

        Returns:
            dict: Dictionary containing:
                - 'transitions_time_series': 3D array of transition counts per trial over time
                - 'transition_averages': Average transitions for each time window (per trial)
                - 'transition_averages_bc': Percentage change from baseline (can exceed ±100%)
                - 'time_window_indices': Indices for each time window
                - 'microstates': List of microstate labels
        """
        # Ensure time_array is a numpy array BEFORE calling .min() and .max()
        time_array = np.asarray(time_array)

        if input_sequence.ndim != 2:
            raise ValueError("Input sequence must be 2D array with shape (trials, timepoints)")

        # Default time windows
        if time_window_ranges is None:
            time_window_ranges = {"baseline": [-1000, -10], "post_tms": [20, 1000]}

        # Get microstates from data if not provided
        if microstates is None:
            unique_vals = np.unique(input_sequence.flatten())

            # Convert to hashable types (strings) - should already be strings from SegmentationIO
            microstates = [str(val) for val in unique_vals]
        else:
            # Ensure provided microstates are hashable
            microstates = [str(ms) for ms in microstates]

        # Initialize state mapping
        n_states = len(microstates)
        state_label_to_code = {label: i for i, label in enumerate(microstates)}

        # Ensure time_array is a numpy array
        time_array = np.asarray(time_array)

        # Define time vectors
        dt = time_array[1] - time_array[0] if len(time_array) > 1 else 1
        transition_times = time_array[:-1] + dt / 2  # Transition midpoints

        # Compute time window indices
        time_window_indices = {}
        for window_name, time_range in time_window_ranges.items():
            start_time, end_time = time_range
            time_indices = (transition_times >= start_time) & (transition_times <= end_time)
            time_window_indices[window_name] = time_indices

        # Initialize transition time series
        transitions_time_series = np.zeros((n_states, n_states, len(transition_times)))

        n_trials = input_sequence.shape[0]

        # Process each trial
        for trial_idx in range(n_trials):
            try:
                # Get state sequence for this trial - should already be strings
                state_labels_trial = input_sequence[trial_idx, :]

                # Convert to numeric codes
                state_numbers_trial = np.array(
                    [state_label_to_code[label] for label in state_labels_trial]
                )

                # Find transitions (where consecutive states differ)
                s1 = state_numbers_trial[:-1]
                s2 = state_numbers_trial[1:]
                transition_indices = np.where(s1 != s2)[0]

                # Update transition counts
                for k in transition_indices:
                    from_state = s1[k]
                    to_state = s2[k]
                    transitions_time_series[from_state, to_state, k] += 1

            except Exception as e:
                print(f"Error processing trial {trial_idx}: {e}")
                print(f"Trial data sample: {state_labels_trial[:5]}")
                raise

        # Normalize by this subject's number of trials
        # This makes each subject's RTF values standardized (per-trial rate)
        # allowing fair comparison across subjects with different trial counts
        transitions_time_series = transitions_time_series / n_trials
        
        # Calculate average transitions for each time window
        transition_averages = {}
        for window_name, time_indices in time_window_indices.items():
            transition_avg = np.zeros((n_states, n_states))

            # Average transitions within window (excluding self-transitions)
            # These values are already per-trial averages from normalization above
            for s1 in range(n_states):
                for s2 in range(n_states):
                    if s1 != s2:
                        data = transitions_time_series[s1, s2, time_indices]
                        # Mean across time points within window
                        transition_avg[s1, s2] = np.mean(data)

            transition_averages[window_name] = transition_avg

        # Baseline correction - compute percentage change from baseline
        transition_averages_bc = {}
        if "baseline" in transition_averages:
            baseline_averages = transition_averages["baseline"]
            for window_name, window_avg in transition_averages.items():
                if window_name != "baseline":
                    # Compute percentage change: ((post - baseline) / baseline) * 100
                    # Handle division by zero by using small epsilon
                    epsilon = 1e-10
                    percentage_change = np.divide(
                        window_avg - baseline_averages,
                        baseline_averages + epsilon
                    ) * 100
                    transition_averages_bc[window_name] = percentage_change
        else:
            print("Warning: No baseline window found for baseline correction")

        return {
            "transitions_time_series": transitions_time_series,
            "transition_averages": transition_averages,
            "transition_averages_bc": transition_averages_bc,
            "time_window_indices": time_window_indices,
            "microstates": microstates,
        }

    @staticmethod
    def generate_synthetic_sequence(input_sequence, method="random"):
        """Generate a synthetic sequence based on the input sequence.

        Args:
            input_sequence (list or str): The input sequence of elements.
            method (str, optional): The method to generate the synthetic sequence. Defaults to 'random'.

        Returns:
            str: The generated synthetic sequence.
        """
        if isinstance(input_sequence, list):
            input_sequence = "".join(input_sequence)

        synthetic_sequence = ""

        if method == "surrogate":
            # Shuffle the entire sequence of characters
            shuffled_chars = list(input_sequence)
            random.shuffle(shuffled_chars)
            synthetic_sequence = "".join(shuffled_chars)

        elif method == "random":
            # Get unique letters from the input sequence
            unique_letters = list(set(input_sequence))

            # Generate synthetic sequence by randomly selecting from unique letters
            for _ in input_sequence:
                synthetic_sequence += random.choice(unique_letters)

        return synthetic_sequence

    @staticmethod
    def generate_theoretical_dictionary(input_sequence, word_size):
        """Generate a theoretical dictionary of non-repeating words.

        Args:
            input_sequence (list or str): The input sequence of elements.
            word_size (int): The size of the word for the theoretical dictionary.

        Returns:
            list: The generated theoretical dictionary of non-repeating words.
        """
        # Count the number of unique characters in the input sequence
        unique_characters = sorted(set(input_sequence))

        # Generate all possible non-repeating words of the given size
        possible_words = itertools.product(unique_characters, repeat=word_size)

        return [
            "".join(word)
            for word in possible_words
            if all(word[i] != word[i + 1] for i in range(len(word) - 1))
        ]

    @staticmethod
    def generate_real_dictionary(input_sequence, word_size):
        """Generate a real dictionary from an input sequence.

        Args:
            input_sequence (list or str): The input sequence of elements.
            word_size (int): The size of the word for the real dictionary.

        Returns:
            tuple: A tuple containing the sorted real dictionary and the sequence representation.
        """
        # Check if the input_sequence is a list, if so, join it into a string
        if isinstance(input_sequence, list):
            input_sequence = "".join(input_sequence)

        # Initialize an empty dictionary to store the combinations and their counts
        real_dictionary, sequence_representation = {}, {}

        # Iterate through the input sequence
        for i in range(len(input_sequence) - word_size + 1):
            # Extract consecutive substring of given length
            combination = input_sequence[i : i + word_size]
            # Convert combination to a hashable type (e.g., tuple) before using it as a key
            combination_key = tuple(combination)
            # Add combination to the dictionary and update its count
            if combination_key not in real_dictionary:
                real_dictionary[combination_key] = 1
            else:
                real_dictionary[combination_key] += 1

        # Generate sequence representation
        for word, count in real_dictionary.items():
            sequence_representation[word] = count

        # Sort the sequence representation dictionary
        sequence_representation = dict(sorted(sequence_representation.items()))

        return sorted(real_dictionary), sequence_representation

    @staticmethod
    def calculate_entropy(input_sequence):
        """Calculate the Shannon entropy of a sequence.

        Args:
            input_sequence (list or str): The input sequence of elements.

        Returns:
            float: The Shannon entropy of the input sequence.
        """
        # Count the frequency of each microstate in the input sequence
        microstate_counts = Counter(input_sequence)

        # Total number of microstates
        total_microstates = len(input_sequence)

        # Initialize entropy
        entropy = 0

        # Calculate entropy
        for count in microstate_counts.values():
            probability = count / total_microstates
            entropy -= probability * math.log(probability)

        return entropy

    @staticmethod
    def calculate_representation_ratios(*entropy_dicts):
        """Calculate representation ratios between entropy distributions.

        Args:
            *entropy_dicts (dict): Variable number of entropy dictionaries.

        Returns:
            dict: A dictionary containing the representation ratios between entropy distributions.
        """
        representation_ratios = {}
        first_entropy_dict = entropy_dicts[0]

        for class_name, entropy_value in first_entropy_dict.items():
            ratio_key = f"RepresentationRatio_{class_name}"
            second_entropy_value = entropy_dicts[1].get(class_name)
            if second_entropy_value is not None:
                representation_ratios[ratio_key] = entropy_value / second_entropy_value

        return dict(sorted(representation_ratios.items(), key=lambda item: item[0]))

    @staticmethod
    def generate_partitions(n_states):
        """Generate all possible (2,n-2) partitions for n states.
        For 4 states: generates (2,2) partitions
        For 5 states: generates (2,3) partitions.

        Args:
            n_states (int): Total number of states

        Returns:
            list: List of tuples, each containing two lists representing the partition
        """
        states = list(range(n_states))
        partitions = []

        # Generate all possible combinations of size 2
        for subset1 in itertools.combinations(states, 2):
            subset2 = [x for x in states if x not in subset1]
            partitions.append((list(subset1), subset2))

        return partitions

    @staticmethod
    def create_random_walk(sequence, partition):
        """Convert categorical sequence into random walk using ±1 based on partition.

        Args:
            sequence (list): Original sequence of states
            partition (tuple): Tuple of (subset1, subset2) defining the partition

        Returns:
            numpy.ndarray: Random walk sequence of ±1
        """
        subset1, _ = partition
        return np.array([1 if state in subset1 else -1 for state in sequence])

    @staticmethod
    def dfa(sequence, scales):
        """Perform Detrended Fluctuation Analysis.

        Args:
            sequence (numpy.ndarray): Input sequence (random walk)
            scales (numpy.ndarray): Array of window sizes to use

        Returns:
            numpy.ndarray: Fluctuation function F(s) for each scale
        """
        # Cumulative sum/random walk
        walk = np.cumsum(sequence - np.mean(sequence))
        fluctuations = np.zeros(len(scales))

        for i, scale in enumerate(scales):
            # Number of windows
            n_windows = int(len(walk) // scale)
            if n_windows < 2:
                fluctuations[i] = np.nan
                continue

            # Reshape data into windows
            windows = walk[: n_windows * scale].reshape((n_windows, scale))
            x = np.arange(scale)

            # Calculate local trends for each window separately and compute variance
            var = 0.0
            for window in windows:
                coef = np.polyfit(x, window, 1)
                trend = np.polyval(coef, x)
                var += np.mean((window - trend) ** 2)
            var /= n_windows
            fluctuations[i] = np.sqrt(var)

        return fluctuations  # Can contain NaNs for invalid scales

    def calculate_hurst_exponent(self, sequence, min_samples=50, max_samples=2500, n_scales=50):
        """Calculate Hurst exponent using DFA for a given sequence."""
        if len(sequence) < min_samples * 2:
            return None

        # Map unique symbols to integer indices
        unique_symbols = list(set(sequence))
        symbol_to_idx = {s: i for i, s in enumerate(unique_symbols)}
        seq_int = np.array([symbol_to_idx[s] for s in sequence])
        n_states = len(unique_symbols)

        # Generate partitions
        partitions = self.generate_partitions(n_states)
        if not partitions:
            return None

        # Generate logarithmically spaced scales
        max_samples = min(max_samples, len(seq_int) // 2)
        scales = np.logspace(np.log10(min_samples), np.log10(max_samples), n_scales, dtype=int)
        scales = np.unique(scales)

        hurst_vals = []
        for partition in partitions:
            subset1, _ = partition
            walk = np.where(np.isin(seq_int, subset1), 1, -1)
            fluctuations = self.dfa(walk, scales)
            valid_mask = ~np.isnan(fluctuations) & (fluctuations > 0)
            if np.sum(valid_mask) > 1:
                log_s = np.log10(scales[valid_mask])
                log_f = np.log10(fluctuations[valid_mask])
                slope, _ = np.polyfit(log_s, log_f, 1)
                hurst_vals.append(slope)

        return float(np.mean(hurst_vals)) if hurst_vals else None
