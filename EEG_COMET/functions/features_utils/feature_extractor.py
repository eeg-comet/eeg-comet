
import pandas as pd
from collections import Counter, defaultdict
from functions.clustering_utils.microstate_clusterer import MicrostateClusterer
from functions.features_utils.feature_helper import FeatureHelper


class FeatureExtractor:
    """
    The FeatureExtractor class provides methods for extracting microstate features from EEG data.
    """
    def __init__(self, input_sequence, sampling_rate, window_size, feature_mode='static'):
        """
        Initialize the FeatureExtractor class.

        Args:
            input_sequence (list): The input sequence of EEG data.
            sampling_rate (int): The sampling rate of the EEG data.
            window_size (int): The window size in seconds.
            feature_mode (str, optional): The feature extraction mode ('static' or 'dynamic'). Defaults to 'static'.
        """

        self.input_sequence = input_sequence
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.feature_mode = feature_mode
        self.num_windows = len(input_sequence) // (sampling_rate * window_size)

    def global_explained_variance(self, eeg_data, microstate_maps, microstate_labels=None):
        """
        Compute Global Explained Variance (GEV) for a given EEG data and microstate maps.

        Args:
            eeg_data (array-like): The EEG data.
            microstate_maps (array-like): The microstate maps.
            microstate_labels (list, optional): The labels for the microstate maps. Defaults to None.

        Returns:
            list: The GEV values for each window and microstate label.
        """

        if microstate_labels and len(microstate_labels) != microstate_maps.shape[0]:
            raise ValueError("Length of microstate_labels must match the number of microstate maps.")
        window_element_gev = [FeatureHelper().initialize_empty_window_data(self.input_sequence)
                              for _ in range(self.num_windows)]
        if self.feature_mode == 'dynamic':
            gevs = {}
            window_size_samples = self.window_size * self.sampling_rate
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
        elif self.feature_mode == 'static':
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
            dict or list: If feature_mode is 'static',
                            returns a dictionary with the average coverage percentage for each element.
                          If feature_mode is 'dynamic',
                            returns a list of dictionaries, where each dictionary represents the coverage percentages
                            for each element in a window.
        """

        window_element_coverage = [FeatureHelper().initialize_empty_window_data(self.input_sequence)
                                   for _ in range(self.num_windows)]

        for window_index in range(self.num_windows):
            window_start = window_index * self.window_size * self.sampling_rate
            window_end = (window_index + 1) * self.window_size * self.sampling_rate
            window_input_sequence = self.input_sequence[window_start:window_end]
            element_counts = Counter(window_input_sequence)

            total_elements = sum(element_counts.values())
            if total_elements > 0:
                for element, count in element_counts.items():
                    coverage = count / total_elements * 100
                    window_element_coverage[window_index][element] = coverage

        if self.feature_mode == 'static':
            total_coverage = Counter()
            for window_coverage in window_element_coverage:
                total_coverage |= window_coverage
            num_windows = len(window_element_coverage)
            return {
                element: min(coverage / num_windows, 100.0)
                for element, coverage in total_coverage.items()
            }
        elif self.feature_mode == 'dynamic':
            return window_element_coverage
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def microstate_occurrence(self):
        """
        Compute the number of times an element changes from another element within non-overlapping windows.
        Returns:
        dict or list: If feature_mode is 'static',
                        returns a dictionary with the average occurrence count for each element.
                      If feature_mode is 'dynamic',
                        returns a list of dictionaries, where each dictionary represents the occurrence counts
                        for each element in a window.
        """

        window_change_counts = [FeatureHelper().initialize_empty_window_data(self.input_sequence)
                                for _ in range(self.num_windows)]
        total_element_counts = Counter()
        for window_index in range(self.num_windows):
            window_start = window_index * (self.sampling_rate * self.window_size)
            window_end = window_start + (self.sampling_rate * self.window_size)
            window_input_sequence = self.input_sequence[window_start:window_end]
            window_input_sequence = FeatureHelper().remove_repetition_sequence(window_input_sequence)
            element_counts = Counter(window_input_sequence)
            window_change_counts[window_index] = element_counts
            total_element_counts |= element_counts
        if self.feature_mode == 'static':
            num_windows = len(window_change_counts)
            return {
                element: count / num_windows
                for element, count in total_element_counts.items()
            }
        elif self.feature_mode == 'dynamic':
            return window_change_counts
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def microstate_duration(self):
        """
        Compute the average duration of each element uninterrupted in the data.

        Returns:
            dict or list: If feature_mode is 'static',
                            returns a dictionary with the average duration for each element.
                          If feature_mode is 'dynamic' and window_size is not None,
                            returns a list of dictionaries, where each dictionary represents the average duration
                            for each element in a window.
        """
        if self.feature_mode == 'static':
            durations = {}
            current_input_sequence = None
            current_duration = 0
            for item in self.input_sequence:
                if item != current_input_sequence:
                    if current_input_sequence is not None:
                        if current_input_sequence not in durations:
                            durations[current_input_sequence] = []
                        durations[current_input_sequence].append(current_duration * 1000 / self.sampling_rate)
                    current_input_sequence = item
                    current_duration = 1
                else:
                    current_duration += 1
            if current_input_sequence not in durations:
                durations[current_input_sequence] = []
            durations[current_input_sequence].append(current_duration * 1000 / self.sampling_rate)
            average_durations = {key: sum(value) / len(value) for key, value in durations.items()}
        elif self.feature_mode == 'dynamic' and self.window_size is not None:
            windows = []
            current_window = {}
            current_window_duration = 0
            for item in self.input_sequence:
                if current_window_duration >= self.window_size * self.sampling_rate:
                    windows.append(current_window.copy())
                    current_window = {}
                    current_window_duration = 0
                if item in current_window:
                    current_window[item] += 1
                else:
                    current_window[item] = 1
                current_window_duration += 1
            if current_window:
                windows.append(current_window)
            for window in windows:
                for key in window:
                    window[key] = window[key] * 1000 / self.sampling_rate
            average_durations = windows
        else:
            raise ValueError("Invalid mode or missing window size for 'dynamic' mode")
        return average_durations

    def compute_transition_probabilities(self):
        """
        Compute the transition probabilities for a given input_sequence.

        Returns:
            dict: A dictionary containing the transition probabilities for each pair of elements in the input_sequence.
        """

        transitions = defaultdict(int)
        total_transitions = 0
        for i in range(len(self.input_sequence) - 1):
            current_element = self.input_sequence[i]
            next_element = self.input_sequence[i + 1]
            # Skip self-transitions
            if current_element == next_element:
                continue
            transition_label = f"{current_element}_{next_element}"
            transitions[transition_label] += 1
            total_transitions += 1
        return {pair: count / total_transitions for pair, count in transitions.items()}

    def shannon_entropy(self):
        """
        Calculate Shannon entropy.

        Returns:
            float or list: If feature_mode is 'static', returns the Shannon entropy of the entire input_sequence.
                           If feature_mode is 'dynamic', returns a list of Shannon entropies for each window.
        """

        window_entropies, window_size_samples = FeatureHelper().initialize_dynamic_windows(
            self.input_sequence, self.sampling_rate, self.window_size
        )

        for window_index in range(len(window_entropies)):
            window_start = window_index * window_size_samples
            window_end = window_start + window_size_samples
            window_input_sequence = self.input_sequence[window_start:window_end]
            window_entropies[window_index] = FeatureHelper().calculate_entropy(window_input_sequence)

        if self.feature_mode == 'static':
            return FeatureHelper().calculate_entropy(self.input_sequence)
        elif self.feature_mode == 'dynamic':
            return window_entropies
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def lempel_ziv_complexity(self):
        """
        Calculate Lempel-Ziv complexity using the LZ76 algorithm and a sliding window implementation.
        """

        return FeatureHelper().compute_lempel_ziv_complexity(
            FeatureHelper().remove_repetition_sequence(self.input_sequence)
        )

    def entropy_representation(self, word_size):
        """
        Calculate the entropy representation of different classes of entropies and their ratio compared to
        theoretical dictionary based on the MicroSynt pipeline.

        Args:
            word_size (int): The size of the word for entropy calculation.

        Returns:
            dict or list: If feature_mode is 'static',
                            returns a dictionary with the entropy representation for each entropy class.
                          If feature_mode is 'dynamic',
                            returns a list of dictionaries, where each dictionary represents the entropy representation
                            for each entropy class in a window.
        """
        # TODO: not completed
        # word_size = 5
        window_entropy_representations, window_size_samples = FeatureHelper().initialize_dynamic_windows(
            self.input_sequence, self.sampling_rate, self.window_size
        )
        for window_index in range(len(window_entropy_representations)):
            window_start = window_index * window_size_samples
            window_end = window_start + window_size_samples
            window_input_sequence = self.input_sequence[window_start:window_end]
            window_entropy_representations[window_index] = FeatureHelper().calculate_entropy(window_input_sequence)
        if self.feature_mode == 'static':
            overall_entropy_representations, _ = MicroSynt().sequence_analysis(self.input_sequence, word_size)
            return overall_entropy_representations
        elif self.feature_mode == 'dynamic':
            return window_entropy_representations

        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    def extract_microstate_features(
            self, filename, feature_list, eeg_data=None, microstate_maps=None, microstate_labels=None, word_size=2):
        """
        Extracts a set of microstate features from EEG data input_sequences, given a list of feature identifiers.
        The function operates in two modes: 'static' and 'dynamic'.

        In 'static' mode, the function returns aggregated feature values over the entire input_sequence for each feature.
        In 'dynamic' mode, it returns features calculated over a set of windows within the input_sequence.

        Args:
            filename (str): The filename of the EEG data.
            feature_list (list): The list of feature identifiers to extract.
            eeg_data (array-like, optional): The EEG data. Defaults to None.
            microstate_maps (array-like, optional): The microstate maps. Defaults to None.
            microstate_labels (list, optional): The labels for the microstate maps. Defaults to None.
            word_size (int, optional): The size of the word for entropy calculation. Defaults to 2.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted microstate features.
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
        if 'SE' in feature_list:
            extracted_entropy = self.shannon_entropy()
            features_dict.append(('SE', extracted_entropy))
        if 'ER' in feature_list:
            extracted_entropy_representation = self.entropy_representation(word_size)
            features_dict.append(('ER', extracted_entropy_representation))

        # Only add TP and LZC if the mode is not dynamic
        if 'LZC' in feature_list and self.feature_mode != 'dynamic':
            extracted_microstate_complexity = self.lempel_ziv_complexity()
            features_dict.append(('LZC', extracted_microstate_complexity))
        if 'TP' in feature_list and self.feature_mode != 'dynamic':
            extracted_microstate_transition_probability = self.compute_transition_probabilities()
            features_dict.append(('TP', extracted_microstate_transition_probability))

        # Create a list to hold the data
        output_features_data = []

        for feature, feature_data in features_dict:
            if self.feature_mode == 'static':
                if isinstance(feature_data, dict):
                    output_features_data.extend(
                        [filename, f"{feature}_{element}", value]
                        for element, value in feature_data.items()
                    )
                else:
                    output_features_data.append([filename, feature, feature_data])
            elif self.feature_mode == 'dynamic':
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

        if self.feature_mode == 'dynamic':
            columns = ["Filename", "Window_index", "Feature", "Value"]

        elif self.feature_mode == 'static':
            columns = ['Filename', 'Feature', 'Value']
        output_features_df = pd.DataFrame(output_features_data, columns=columns)

        if self.feature_mode == 'static':
            output_features_df = output_features_df.pivot_table(
                index='Filename', columns='Feature', values='Value').reset_index()
        elif self.feature_mode == 'dynamic':
            output_features_df = output_features_df.pivot_table(
                index=["Filename", "Window_index"], columns="Feature", values="Value").reset_index()

        return output_features_df


class MicroSynt:
    """
    The MicroSynt class provides methods for analyzing entropy distribution in an input sequence and
    generating surrogate statistics based on input sequences.
    """

    @staticmethod
    def sequence_analysis(input_sequence, word_size):
        """
        Analyze the entropy distribution in an input sequence.

        Args:
            input_sequence (str or list): The input sequence of symbols.
            word_size (int): The size of the word for entropy calculation.

        Returns:
            tuple: A tuple containing the entropy representation percentages for the real and theoretical dictionaries.
        """

        # Check if the input_sequence is a list, if so, join it into a string
        if isinstance(input_sequence, list):
            input_sequence = ''.join(input_sequence)

        theoretical_dictionary = FeatureHelper().generate_theoretical_dictionary(input_sequence, word_size)
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
        entropy_representation_real = {f"EntropyClass{i + 1}": 0 for i in range(len(unique_entropies_sorted))}

        for word, count in sequence_representation_real.items():
            entropy = word_entropy[word]
            entropy_class = f"EntropyClass{unique_entropies_sorted.index(entropy) + 1}"
            entropy_representation_real[entropy_class] += count

        # Calculate percentages for each entropy class
        total_words_real = sum(entropy_representation_real.values())
        entropy_representation_percentage_real = {ec: count / total_words_real for ec, count in
                                                  entropy_representation_real.items()}

        # Convert sequence_representation for theoretical dictionary to entropy classes
        entropy_representation_theoretical = {f"EntropyClass{i + 1}": 0 for i in range(len(unique_entropies_sorted))}

        for word in theoretical_dictionary:
            entropy = FeatureHelper().calculate_entropy(word)
            entropy_class = f"EntropyClass{unique_entropies_sorted.index(entropy) + 1}"
            entropy_representation_theoretical[entropy_class] += 1

        # Calculate percentages for each entropy class
        total_words_theoretical = sum(entropy_representation_theoretical.values())
        entropy_representation_percentage_theoretical = {ec: count / total_words_theoretical for ec, count in
                                                         entropy_representation_theoretical.items()}

        return entropy_representation_percentage_real, entropy_representation_percentage_theoretical

    def surrogate_statistics(self, input_sequence, word_size, repeats=1000):
        """
        Generate surrogate statistics based on input sequences.

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
            surrogate_sequence = FeatureHelper().generate_synthetic_sequence(input_sequence, 'surrogate')
            surrogate_no_permanence_sequence = FeatureHelper().remove_repetition_sequence(surrogate_sequence)

            # Generate entropy representations
            surrogate_entropy_representation, surrogate_entropy_representation_theoretical = self.sequence_analysis(
                surrogate_no_permanence_sequence, word_size
            )

            # Calculate representation ratios
            surrogate_entropy_representation_ratio = FeatureHelper().calculate_representation_ratios(
                surrogate_entropy_representation, surrogate_entropy_representation_theoretical
            )

            # Append the ratio to the distribution for each class separately
            for key, value in surrogate_entropy_representation_ratio.items():
                # Rename the key
                new_key = key.replace('RepresentationRatio', 'Distribution')
                if new_key not in distributions:
                    distributions[new_key] = []
                distributions[new_key].append(value)

        return distributions
