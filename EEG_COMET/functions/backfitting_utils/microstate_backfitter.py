
import numpy as np
import os.path
from scipy.signal import find_peaks
from collections import Counter
from itertools import groupby


class MicrostateBackfitter:
    def __init__(self, study_name, preprocessed_data_path, microstate_maps, backfit_to,
                 filter_segments, filter_segments_option, identify_short_window, remove_segments_less_than,
                 micro_labels, segmentation_path, extension, datatype, sample_rate, smoothing_parameters, export_format):

        self.study_name = study_name
        self.preprocessed_data_path = preprocessed_data_path
        self.microstate_maps = microstate_maps
        self.backfit_to = backfit_to
        self.filter_segments = filter_segments
        self.filter_segments_option = filter_segments_option
        self.identify_short_window = identify_short_window
        self.remove_segments_less_than = remove_segments_less_than
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
        data: V
        microstate_maps: Gamma (T-like symbol)
        n_states: N_mu in paper
        epsilon: convergence criterion parameter
        b: window size parameter
        lamb: non-smoothness penalty parameter (lambda)

        segmentation: L
        n_channels: N_s in paper
        n_samples: N_T

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
                                      smoothing_parameters=[1e-6, 3, 5]):
        """
        Substitute short segments in the segmentation array with neighboring elements based on the chosen option.
        """

        filled_segmentation = np.copy(segmentation)
        count_dups = [sum(1 for _ in group) for _, group in groupby(filled_segmentation)]

        # Replace short segments if left and right elements are the same
        for i, count in enumerate(count_dups):
            if count <= int(segments_less_than):
                start = int(np.sum(count_dups[0:i]))
                stop = int(start + count)
                # Check if there is a left and right neighbor to the segment
                if start > 0 and stop < len(filled_segmentation):
                    # Check if the left and right neighbors are the same as the segment
                    if filled_segmentation[start - 1] == filled_segmentation[stop]:
                        filled_segmentation[start:stop] = filled_segmentation[start - 1]

        # Recalculate count_dups after replacing segments with identical neighbors
        count_dups = [sum(1 for _ in group) for _, group in groupby(filled_segmentation)]

        # Find short segments
        for C in range(len(count_dups)):
            if count_dups[C] <= int(segments_less_than):
                start = int(np.sum(count_dups[0:C]))
                stop = int(start + count_dups[C])
                filled_segmentation[start:stop] = -1

        filled_segmentation = np.array(filled_segmentation)

        if option == 'remove':
            return filled_segmentation
        elif option == 'replace_high':
            filled_segmentation = self.fill_with_neighbors_with_higher_count(filled_segmentation)
        elif option == 'replace_half':
            filled_segmentation = self.fill_with_neighbors_half(filled_segmentation)
        elif option == 'smooth':
            filled_segmentation = self.segmentation_smooth(data, microstate_maps, n_states, *smoothing_parameters)

        return filled_segmentation

    def label_segments(self, segmentation):
        segmentation = segmentation + 1
        segmentation = list(map(int, segmentation))
        labeled_segmentation = list(map(str, segmentation))
        labeled_segmentation = np.char.replace(labeled_segmentation, str(0), "NaN")
        for m in range(1, len(self.microstate_labels) + 1):
            labeled_segmentation = np.char.replace(labeled_segmentation, str(m), self.microstate_labels[m - 1])
        return labeled_segmentation

    # TODO: create a function to compute the goodness of fit for filtered segmentation
    def goodness_fit_segmentation(self, eeg_data, labeled_segmentation):
        similarity_mean = 0
        for m in range(len(self.microstate_labels)):
            microstate_map = self.microstate_maps[m, :]
            microstate_map_norm = np.linalg.norm(microstate_map)
            data_indices = [i for i, x in enumerate(labeled_segmentation) if x == self.microstate_labels[m]]
            eeg_data_segement = eeg_data[:, data_indices]
            eeg_data_segement_norm = np.linalg.norm(eeg_data_segement, axis=0)
            # Compute the dot product
            dot_product = np.dot(microstate_map, eeg_data_segement)
            # Compute the cosine similarity
            similarity = abs(dot_product / (microstate_map_norm * eeg_data_segement_norm))
            similarity_mean = similarity_mean + similarity.mean()
        return similarity_mean / len(self.microstate_labels)

    @staticmethod
    def mark_short_segments(segmentation, min_occurrence):
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

        return new_segmentation

    @staticmethod
    def find_optimal_index(values, threshold=0.001):
        for i in range(1, len(values)):
            rate_of_change = values[i] - values[i - 1]
            if rate_of_change < threshold:
                return i - 1
        return len(values) - 1

    def get_similarity_score(self, eeg, rm_max_len=50):
        # Generate a list of window lengths to test for segment removal
        len_win2rm_list = list(range(0, rm_max_len, int(1000 / self.sample_rate)))

        # Load EEG data
        eeg_data = eeg.get_data()

        # Iterate over each trial in the EEG data (if epoched)
        for idx in range(len(eeg)) if self.datatype == 'epoched' else [None]:
            # Extract trial data
            if self.datatype == 'epoched':
                trial_data = eeg[idx].get_data()
            else:
                trial_data = eeg_data

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

    def identify_optimal_length_filter(self, similarity_scores):
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
            if len(peaks) > 0:
                inflection_point = peaks[0] + 1
            else:
                inflection_point = len(row)

            # Append inflection point to optimal indices
            optimal_indices.append(inflection_point)

        # Calculate the median of optimal indices as the optimal length to remove segments
        remove_segments_less_than = int(np.median(optimal_indices))

        return remove_segments_less_than

    def perform_segmentation(self, eeg, eeg_name, remove_segments_less_than):
        segmentation_fit = 0
        eeg_data = eeg.get_data()
        eeg_times = eeg.times * 1000

        for idx in range(len(eeg)) if self.datatype == 'epoched' else [None]:
            if self.datatype == 'epoched':
                trial_data = np.squeeze(eeg[idx].get_data())
                trial_times = eeg[idx].times * 1000
            else:
                trial_data = eeg_data
                window_size = 5
                # Smooth Data
                weights = np.repeat(1.0, window_size) / window_size
                trial_data = np.apply_along_axis(
                    lambda x: np.convolve(x, weights, mode='same'), axis=1, arr=trial_data)

                trial_times = eeg_times

        if self.backfit_to == 'all':
            # Calculate the correlation coefficient between each topography and each time point
            correlation_matrix = np.dot(self.microstate_maps, trial_data) / np.sqrt(
                np.sum(self.microstate_maps ** 2, axis=1)[:, np.newaxis] * np.sum(trial_data ** 2, axis=0))

            # Find the time point with the highest correlation coefficient for each topography
            segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)

            # Filter short segments
            if self.filter_segments:
                segmentation = self.substitude_maps_with_duration(segmentation,
                                                                  remove_segments_less_than,
                                                                  self.filter_segments_option,
                                                                  trial_data,
                                                                  self.microstate_maps,
                                                                  len(self.microstate_labels),
                                                                  [self.smoothing_parameters[0],
                                                                   remove_segments_less_than,
                                                                   self.smoothing_parameters[2]])

        elif self.backfit_to == 'peaks':
            gfp = np.std(eeg_data, axis=0)
            peaks, _ = find_peaks(gfp)

            # Define troughs between consecutive peaks
            troughs = [0]
            for p in range(len(peaks) - 1):
                min_arg = np.argmin(gfp[peaks[p]:peaks[p + 1]])
                troughs.append(peaks[p] + min_arg)
            troughs.append(len(gfp))

            # Compute the differences between consecutive troughs
            diff_troughs = np.diff(troughs)

            # Compute the activation of microstate maps with EEG data at identified peaks
            activation = np.dot(self.microstate_maps, eeg_data[:, peaks])

            # Identify the segmentation based on the highest activation
            segmentation_peaks = np.argmax(np.abs(activation), axis=0)

            # Repeat the segmentation based on the identified troughs
            segmentation = np.repeat(segmentation_peaks.astype(int), diff_troughs.astype(int))

        labeled_segmentation = self.label_segments(segmentation)

        similarity_metric = self.goodness_fit_segmentation(trial_data, labeled_segmentation)
        segmentation_fit += similarity_metric

        # Call the export_segmentation method
        if idx is not None:
            trial_filename = f"{eeg_name}_{idx}"
        else:
            trial_filename = eeg_name

        return labeled_segmentation, trial_filename, trial_times, segmentation_fit
