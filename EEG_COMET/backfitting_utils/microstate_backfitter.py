
import numpy as np
from scipy.signal import find_peaks
from collections import Counter
from itertools import groupby


class MicrostateBackfitter:
    """
    The MicrostateBackfitter class provides methods for backfitting microstates to EEG data.
    It includes functions for filling segments, smoothing segmentations, labeling segments,
    and calculating similarity scores.

    Args:
        study_name (str): Name of the study.
        preprocessed_data_path (str): Path to the preprocessed data.
        microstate_maps (np.ndarray): Array of microstate maps.
        backfit_to (str): Backfitting method ('all' or 'peaks').
        filter_segments (bool): Flag indicating whether to filter segments.
        filter_segments_option (str): Option for filtering segments ('remove', 'replace_high', 'replace_half', 'smooth').
        identify_short_window (bool): Flag indicating whether to identify short windows.
        micro_labels (list): List of microstate labels.
        segmentation_path (str): Path to the segmentation.
        extension (str): File extension.
        datatype (str): Data type ('continuous' or 'epoched').
        sample_rate (int): Sampling rate.
        smoothing_parameters (list, optional): Smoothing parameters. Defaults to None.
        export_format (str): Export format.
    """
    def __init__(self, study_name, preprocessed_data_path, microstate_maps, backfit_to, filter_segments,
                 filter_segments_option, identify_short_window, micro_labels,
                 segmentation_path, extension, datatype, sample_rate, smoothing_parameters, export_format):
        """
        Initializes a new instance of the MicrostateBackfitter class.
        """

        self.study_name = study_name
        self.preprocessed_data_path = preprocessed_data_path
        self.microstate_maps = microstate_maps
        self.backfit_to = backfit_to
        self.filter_segments = filter_segments
        self.filter_segments_option = filter_segments_option
        self.identify_short_window = identify_short_window
        self.microstate_labels = micro_labels
        self.segmentation_path = segmentation_path
        self.extension = extension
        self.datatype = datatype
        self.sample_rate = sample_rate
        self.smoothing_parameters = smoothing_parameters
        self.export_format = export_format

    @staticmethod
    def fill_with_neighbors_with_higher_count(segmentation):
        """
        Fill the groups of -1 values in the array with the neighbor that has a higher count.
        """
        filled_segmentation = segmentation.copy()
        n = len(segmentation)
        i = 0

        while i < n:
            if segmentation[i] == -1:
                # Find the start and end indices of the group of -1 values
                start = i
                while i < n and segmentation[i] == -1:
                    i += 1
                end = i - 1

                # Count the occurrences of the previous and next values
                prev = segmentation[start - 1] if start > 0 else 0
                next_val = segmentation[end + 1] if end < n - 1 else 0

                # Count the occurrences of the previous and next values in the windows
                prev_count = 0
                if start > 0:
                    window_before = segmentation[max(start - 2, 0):start]
                    prev_count = Counter(window_before).most_common(1)[0][1]

                next_count = 0
                if end < n - 1:
                    window_after = segmentation[end + 2:min(end + 3, n)]
                    next_count = Counter(window_after).most_common(1)[0][1]

                # Fill the group of -1 values with the neighbor with the higher count
                if prev_count > next_count:
                    fill_value = prev
                elif prev_count < next_count:
                    fill_value = next_val
                else:
                    fill_value = prev or next_val

                # Fill the group with the determined fill value
                for j in range(start, end + 1):
                    filled_segmentation[j] = fill_value
            else:
                i += 1

        return filled_segmentation

    @staticmethod
    def fill_with_neighbors_half(segmentation):
        """
        Fill the groups of -1 values in the array by evenly distributing the neighboring values.
        """
        filled_segmentation = segmentation.copy()
        n = len(segmentation)
        i = 0
        count = 0

        while i < n:
            if segmentation[i] == -1:
                # Find the start and end indices of the group of -1 values
                start = i
                while i < n and segmentation[i] == -1:
                    i += 1
                    count += 1
                end = i - 1

                # Determine the fill values based on the previous and next elements
                fill_value_prev = segmentation[start - 1] if start > 0 else 0
                fill_value_next = segmentation[end + 1] if end < n - 1 else 0

                # Determine the number of elements for each fill value
                half_count = count // 2
                half_count_prev = half_count if count % 2 == 0 else half_count + 1

                # Fill the group with the previous and next values
                for j in range(start, start + half_count_prev):
                    filled_segmentation[j] = fill_value_prev
                for j in range(start + half_count_prev, end + 1):
                    filled_segmentation[j] = fill_value_next

                count = 0
            else:
                i += 1

        return filled_segmentation

    @staticmethod
    def segmentation_smooth(data, microstate_maps, n_states, epsilon=1e-6, b=3, lamb=5):
        """
        Smooth the segmentation based on the given parameters.

        Args:
            data (numpy.ndarray): Array containing the data.
            microstate_maps (numpy.ndarray): Array containing the microstate maps.
            n_states (int): Number of microstate states.
            epsilon (float, optional): Convergence criterion parameter. Defaults to 1e-6.
            b (int, optional): Window size parameter. Defaults to 3.
            lamb (int, optional): Non-smoothness penalty parameter. Defaults to 5.

        Returns:
            numpy.ndarray: Smoothed segmentation array.
        """

        n_channels, n_samples = data.shape
        data_sum_sq = np.sum(data ** 2)
        iteration = 0
        prev_residual = 0
        residual = np.inf
        thresh = epsilon

        # STEP 2 in TABLE 2
        # V dot Gamma
        activation = microstate_maps.dot(data)
        # L
        segmentation = np.argmax(np.abs(activation), axis=0)

        # STEP 3 in TABLE 2
        raw_segmentation = segmentation

        # STEP 4 in TABLE 2
        act_sum_sq = np.sum(np.sum(microstate_maps[segmentation].T * data, axis=0) ** 2)
        e1 = abs(data_sum_sq - act_sum_sq)
        e2 = e1 / float(n_samples * (n_channels - 1))

        while residual > thresh:
            # STEP 5 in TABLE 2
            windows = np.lib.stride_tricks.sliding_window_view(raw_segmentation, 2 * b + 1)
            n_bkt = np.zeros((windows.shape[0], n_states))
            for i, window in enumerate(windows):
                cnt = Counter(window)
                n_bkt[i] = [cnt[x] for x in range(n_states)]
            raw_segmentation[b:n_samples - b] = np.argmin(
                (np.sum(data ** 2, axis=0) -
                 (np.sum(microstate_maps[segmentation].T * data, axis=0) ** 2))[b:n_samples - b] / (
                            2 * e2 * (n_channels - 1)) - (lamb * n_bkt).T, axis=0)

            # STEP 6 in TABLE 2
            segmentation = raw_segmentation  # .copy()

            # STEP 7 in TABLE 2
            act_sum_sq = np.sum(np.sum(microstate_maps[segmentation].T * data, axis=0) ** 2)
            e1 = abs(data_sum_sq - act_sum_sq)
            sigma_mu = (e1 / float(n_samples * (n_channels - 1)))
            residual = abs(prev_residual - sigma_mu)

            # STEP 8 in TABLE 2
            prev_residual = sigma_mu
            thresh = epsilon * sigma_mu
            iteration += 1

        return segmentation

    def substitude_maps_with_duration(self, segmentation, segments_less_than, option, data, microstate_maps, n_states,
                                      smoothing_parameters=None):
        """
        Substitute short segments in the segmentation array with neighboring elements based on the chosen option.

        Args:
            self: The MicrostateBackfitter instance.
            segmentation (numpy.ndarray): Array containing the segmentation.
            segments_less_than (int): Threshold for removing segments.
            option (str): Option for substitution ('remove', 'replace_high', 'replace_half', 'smooth').
            data (numpy.ndarray): Array containing the data.
            microstate_maps (numpy.ndarray): Array containing the microstate maps.
            n_states (int): Number of microstate states.
            smoothing_parameters (list, optional): Smoothing parameters. Defaults to None.

        Returns:
            numpy.ndarray: Substituted segmentation array.
        """

        if smoothing_parameters is None:
            smoothing_parameters = [1e-6, 3, 5]

        if option == 'smooth':
            filled_segmentation = self.segmentation_smooth(data, microstate_maps, n_states, *smoothing_parameters)
            filled_segmentation = self.mark_short_segments(filled_segmentation, segments_less_than)
            filled_segmentation = self.fill_with_neighbors_with_higher_count(filled_segmentation)
        else:
            filled_segmentation = self.mark_short_segments(segmentation, segments_less_than)

        if option == 'replace_high':
            filled_segmentation = self.mark_short_segments(segmentation, segments_less_than)
            filled_segmentation = self.fill_with_neighbors_with_higher_count(filled_segmentation)

        if option == 'replace_half':
            filled_segmentation = self.mark_short_segments(segmentation, segments_less_than)
            filled_segmentation = self.fill_with_neighbors_half(filled_segmentation)

        return filled_segmentation

    def label_segments(self, segmentation):
        """
        Label the segments in the segmentation array.

        Args:
            segmentation (list or numpy.ndarray): Array containing the segmentation.

        Returns:
            numpy.ndarray: Array containing the labeled segmentation.
        """

        segmentation = segmentation + 1
        segmentation = list(map(int, segmentation))
        labeled_segmentation = list(map(str, segmentation))
        labeled_segmentation = np.char.replace(labeled_segmentation, str(0), "NaN")
        for m in range(1, len(self.microstate_labels) + 1):
            labeled_segmentation = np.char.replace(labeled_segmentation, str(m), self.microstate_labels[m - 1])
        return labeled_segmentation

    def goodness_fit_segmentation(self, eeg_data, labeled_segmentation):
        """
        Calculate the goodness of fit for the labeled segmentation compared to the EEG data.

        Args:
            self: The MicrostateBackfitter instance.
            eeg_data (numpy.ndarray): Array containing the EEG data.
            labeled_segmentation (numpy.ndarray): Array containing the labeled segmentation.

        Returns:
            float: The mean similarity score between the microstate maps and the EEG data.
        """

        similarity_mean = 0
        for m in range(len(self.microstate_labels)):
            microstate_map = self.microstate_maps[m, :]
            microstate_map_norm = np.linalg.norm(microstate_map)
            data_indices = [i for i, x in enumerate(labeled_segmentation) if x == self.microstate_labels[m]]
            eeg_data_segment = eeg_data[:, data_indices]
            eeg_data_segment_norm = np.linalg.norm(eeg_data_segment, axis=0)
            # Compute the dot product
            dot_product = np.dot(microstate_map, eeg_data_segment)
            # Compute the cosine similarity
            similarity = abs(dot_product / (microstate_map_norm * eeg_data_segment_norm))
            similarity_mean = similarity_mean + similarity.mean()
        return similarity_mean / len(self.microstate_labels)

    @staticmethod
    def mark_short_segments(segmentation, min_occurrence):
        """
        Mark short segments in the segmentation array with -1.

        Args:
            segmentation (list or numpy.ndarray): Array containing the segmentation.
            min_occurrence (int): Minimum number of occurrences for a segment to be considered not short.

        Returns:
            numpy.ndarray: Array with short segments marked as -1.
        """

        if len(segmentation) == 0:
            return segmentation

        new_segmentation = []
        current_element = segmentation[0]
        current_count = 1

        for next_element in segmentation[1:]:
            if next_element == current_element:
                current_count += 1
            else:
                if current_count <= min_occurrence:
                    new_segmentation.extend([-1] * current_count)
                else:
                    new_segmentation.extend([current_element] * current_count)

                current_element = next_element
                current_count = 1

        # Process the last element(s)
        if current_count < min_occurrence:
            new_segmentation.extend([-1] * current_count)
        else:
            new_segmentation.extend([current_element] * current_count)

        return np.asarray(new_segmentation)

    @staticmethod
    def find_optimal_index(values, threshold=0.001):
        """
        Find the index of the last value in the list where the rate of change is below the given threshold.

        Args:
            values (list or numpy.ndarray): List of values.
            threshold (float, optional): Threshold for the rate of change. Defaults to 0.001.

        Returns:
            int: Index of the last value where the rate of change is below the threshold.
        """

        for i in range(1, len(values)):
            rate_of_change = values[i] - values[i - 1]
            if rate_of_change < threshold:
                return i - 1
        return len(values) - 1

    def get_similarity_score(self, eeg, rm_max_len=50):
        """
        Calculate the similarity score between the microstate maps and the EEG data.

        Args:
            eeg (numpy.ndarray): Array containing the EEG data.
            rm_max_len (int, optional): Maximum length of segments to remove. Defaults to 50.

        Returns:
            float: The similarity score between the microstate maps and the EEG data.
        """

        # Generate a list of window lengths to test for segment removal
        len_win2rm_list = list(range(0, rm_max_len, int(1000 / self.sample_rate)))

        # Load EEG data
        eeg_data = eeg.get_data()

        # Iterate over each trial in the EEG data (if epoched)
        for idx in range(len(eeg)) if self.datatype == 'epoched' else [None]:
            # Extract trial data
            trial_data = eeg[idx].get_data() if self.datatype == 'epoched' else eeg_data
            # Calculate correlation matrix between microstate maps and trial data
            correlation_matrix = np.dot(self.microstate_maps, trial_data) / np.sqrt(
                np.sum(self.microstate_maps ** 2, axis=1)[:, np.newaxis] * np.sum(trial_data ** 2, axis=0))

            # Identify the segmentation based on the maximum correlation coefficient
            segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)

        # Calculate similarity scores for different segment removal lengths
        for len_win2rm in len_win2rm_list:
            idx_win2rm = len_win2rm_list.index(len_win2rm)

            # Mark short segments for removal
            segmentation = self.mark_short_segments(segmentation, idx_win2rm)

            # Label segments
            labeled_segmentation = self.label_segments(np.array(segmentation))

            # Calculate similarity score
            similarity_score = self.goodness_fit_segmentation(eeg_data, labeled_segmentation)

        return similarity_score

    @staticmethod
    def identify_optimal_length_filter(similarity_scores):
        """
        Identify the optimal length filter based on the similarity scores.

        Args:
            similarity_scores (list or numpy.ndarray): List of similarity scores.

        Returns:
            int: The optimal length filter.
        """

        # Convert similarity scores to numpy array
        similarity_scores = np.array(similarity_scores)

        # Initialize list to store optimal indices
        optimal_indices = []

        # Loop through each row in the 2D array of similarity scores
        for row in similarity_scores:
            # Compute derivative of the row
            derivative = np.diff(row)

            # Find peaks in the derivative
            peaks, _ = find_peaks(derivative)

            # If peaks exist, select the first peak, otherwise use the length of the row
            inflection_point = peaks[0] + 1 if len(peaks) > 0 else len(row)
            # Append inflection point to optimal indices
            optimal_indices.append(inflection_point)

        return int(np.median(optimal_indices))

    def backfit2all(self, data, filter_segments_less_than):
        """
            Perform segmentation by backfitting microstate maps to all time points in the data.

            Args:
                data (ndarray): The EEG data (can be from a single trial or continuous data).
                filter_segments_less_than (int): The threshold for filtering segments shorter than this value.

            Returns:
                ndarray: The segmentation array where each time point is labeled with a microstate index.
        """
        correlation_matrix = np.dot(self.microstate_maps, data) / np.sqrt(
            np.sum(self.microstate_maps ** 2, axis=1)[:, np.newaxis] * np.sum(data ** 2, axis=0))
        segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)
        if self.filter_segments:
            segmentation = self.substitude_maps_with_duration(
                segmentation=segmentation,
                segments_less_than=filter_segments_less_than,
                option=self.filter_segments_option,
                data=data,
                microstate_maps=self.microstate_maps,
                n_states=len(self.microstate_labels),
                smoothing_parameters=[
                    self.smoothing_parameters[0], filter_segments_less_than, self.smoothing_parameters[2]
                ]
            )
        return segmentation

    def backfit2peaks(self, data):
        """
            Perform segmentation by backfitting microstate maps only to the peaks in global field power (GFP).

            Args:
                data (ndarray): The EEG data (can be from a single trial or continuous data).

            Returns:
                ndarray: The segmentation array where each time point is labeled with a microstate index.
        """
        gfp = np.std(data, axis=0)
        peaks, _ = find_peaks(gfp)
        troughs = [0]
        for p in range(len(peaks) - 1):
            min_arg = np.argmin(gfp[peaks[p]:peaks[p + 1]])
            troughs.append(peaks[p] + min_arg)
        troughs.append(len(gfp))
        diff_troughs = np.diff(troughs)
        activation = np.dot(self.microstate_maps, data[:, peaks])
        segmentation_peaks = np.argmax(np.abs(activation), axis=0)
        segmentation = np.repeat(segmentation_peaks.astype(int), diff_troughs.astype(int))

        return segmentation

    def perform_segmentation(self, eeg, filter_segments_less_than):
        """
        Perform segmentation on the EEG data based on the specified backfitting method.

        Args:
            eeg (mne.io.Raw | mne.Epochs): An instance of an MNE object, either Raw or Epochs, containing the EEG data.
            filter_segments_less_than (int): The threshold for filtering segments shorter than this value.

        Returns:
            tuple: A tuple containing the labeled segmentation, trial filename, trial times, and segmentation
        """
        segmentation_fit = 0
        eeg_data = eeg.get_data()
        if self.datatype == 'epoched':
            segmentation_list = []
            for trial in range(eeg_data.shape[0]):
                trial_data = eeg_data[trial, :, :]
                segmentation = self.backfit2peaks(trial_data) if self.backfit_to == 'peaks' else \
                    self.backfit2all(trial_data, filter_segments_less_than)
                labeled_segmentation = self.label_segments(segmentation)
                segmentation_list.append(labeled_segmentation)
            labeled_segmentation_array = np.vstack(segmentation_list)
        else:
            segmentation = self.backfit2peaks(eeg_data) if self.backfit_to == 'peaks' else \
                self.backfit2all(eeg_data, filter_segments_less_than)
            labeled_segmentation_array = self.label_segments(segmentation)

        # TODO update this
        # similarity_metric = self.goodness_fit_segmentation(trial_data, labeled_segmentation_array)
        # segmentation_fit += similarity_metric

        return labeled_segmentation_array, segmentation_fit
