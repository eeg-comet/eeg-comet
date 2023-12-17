
import numpy as np
import os.path
from tqdm import tqdm
from scipy.signal import find_peaks
from collections import Counter
from itertools import groupby
from functions.data_utils.data_io import DataIO
from functions.backfitting_utils.segmentation_io import SegmentationIO
from gui.progress_dialog import ProgressDialog


class MicrostateBackfitter:
    def __init__(self, study_name, preprocessed_data_path, microstate_maps, backfit_to,
                 filter_segments, filter_segments_option, identify_short_window, remove_segments_less_than,
                 micro_labels, save_path, extension, datatype, sample_rate, smooth_param, export_format):

        self.study_name = study_name
        self.preprocessed_data_path = preprocessed_data_path
        self.microstate_maps = microstate_maps
        self.backfit_to = backfit_to
        self.filter_segments = filter_segments
        self.filter_segments_option = filter_segments_option
        self.identify_short_window = identify_short_window
        self.remove_segments_less_than = remove_segments_less_than
        self.microstate_labels = micro_labels
        self.save_path = save_path
        self.extension = extension
        self.datatype = datatype
        self.sample_rate = sample_rate
        self.smooth_param = smooth_param
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
                half_count_next = half_count

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
        '''
        data: V
        microstate_maps: Gamma (T-like symbol)
        n_states: N_mu in paper
        epsilon: convergence criterion parameter
        b: window size parameter
        lamb: non-smoothness penalty parameter (lambda)

        segmentation: L
        n_channels: N_s in paper
        n_samples: N_T

        '''

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
            #print(f'iteration: {iteration + 1} residual: {residual}, thresh: {thresh}')

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
        #print('Finishes after', str(iteration), 'Iterations.')

        return segmentation

    def substitude_maps_with_duration(self, segmentation, segments_less_than, option, data, microstate_maps, n_states,
                                      smooth_param=[1e-6, 3, 5]):
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
            filled_segmentation = self.segmentation_smooth(data, microstate_maps, n_states, *smooth_param)

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

    def perform_segmentation(self):

        # Create an instance of the SegmentationIO class
        segmentation_io = SegmentationIO()

        data_io = DataIO()
        list_eeg_path, list_eeg_names = data_io.find_data(self.preprocessed_data_path, self.extension, "*")

        if self.identify_short_window:
            # Create an instance of the progress dialog
            progress_dialog = ProgressDialog()
            progress_dialog.set_window_title("Filtering Segments ...")
            progress_dialog.set_label_text("Identifying the optimal window length for removal")
            progress_dialog.show()

            rm_max_len = 50
            len_win2rm_list = list(range(0, rm_max_len, int(1000 / self.sample_rate)))
            similarity_scores = np.empty((len(list_eeg_path), len(len_win2rm_list)))
            progress_bar = tqdm(total=len(list_eeg_path), ncols=100, position=0, leave=True,
                                desc="Identifying the optimal window length for removal")
            for eeg_idx, (eeg_path, eeg_name) in enumerate(zip(list_eeg_path, list_eeg_names)):
                progress_dialog.set_line_edit_text(f"{eeg_name}")
                eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
                eeg_data = eeg.get_data()
                idx_eeg = list_eeg_path.index(eeg_path)
                for idx in range(len(eeg)) if self.datatype == 'epoched' else [None]:
                    if self.datatype == 'epoched':
                        trial_data = eeg[idx].get_data()
                    else:
                        trial_data = eeg_data
                    correlation_matrix = np.dot(self.microstate_maps, trial_data) / np.sqrt(
                        np.sum(self.microstate_maps ** 2, axis=1)[:, np.newaxis] * np.sum(trial_data ** 2, axis=0))
                    segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)
                for len_win2rm in len_win2rm_list:
                    idx_win2rm = len_win2rm_list.index(len_win2rm)
                    segmentation = self.mark_short_segments(segmentation, idx_win2rm)
                    labeled_segmentation = self.label_segments(np.array(segmentation))
                    similarity_scores[idx_eeg, idx_win2rm] = self.goodness_fit_segmentation(eeg_data,
                                                                                            labeled_segmentation)

                progress_dialog.update_progress(eeg_idx + 1, len(list_eeg_path))
                progress_bar.update(1)

            progress_dialog.close()
            progress_bar.close()
            similarity_scores = np.array(similarity_scores)

            optimal_indices = []

            # Loop through each row in the 2D array
            for row in similarity_scores:
                derivative = np.diff(row)
                peaks, _ = find_peaks(derivative)
                if len(peaks) > 0:
                    inflection_point = peaks[0] + 1
                else:
                    inflection_point = len(row)
                optimal_indices.append(inflection_point)

            remove_segments_less_than = int(np.median(optimal_indices))

        else:
            remove_segments_less_than = self.remove_segments_less_than

        remove_segments_less_than_ms = remove_segments_less_than * (1000 / self.sample_rate)
        print("Optimal length to remove:", remove_segments_less_than_ms)

        if self.filter_segments_option == 'remove':
            print(f"\nRemoving segments with less than {remove_segments_less_than_ms}ms in duration.")
        elif self.filter_segments_option == 'replace_high':
            print(
                f"\nReplacing segments with less than {remove_segments_less_than_ms}ms"
                f"by the nearby microstate with higher occurrence.")
        elif self.filter_segments_option == 'replace_half':
            print(
                f"\nReplacing segments with less than {remove_segments_less_than_ms}ms"
                f"by half by the previous and half by the next dominant microstate.")
        elif self.filter_segments_option == 'smooth':
            print(
                f"\nSmoothing segments.")

        # Create an instance of the progress dialog
        progress_dialog = ProgressDialog()
        progress_dialog.set_window_title("Backfitting ...")
        progress_dialog.set_label_text("Backfitting microstates to data: ")
        progress_dialog.show()
        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(list_eeg_path), ncols=100, position=0, leave=True)

        segmentation_fit = 0
        for eeg_idx, eeg_path in enumerate(list_eeg_path):
            eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
            eeg_data = eeg.get_data()
            eeg_times = eeg.times * 1000
            filename = os.path.split(eeg_path)[1].split('.')[0]

            progress_dialog.set_line_edit_text(f"{filename}")
            progress_bar.set_description(f"Backfitting microstates to data: {filename}")

            for idx in range(len(eeg)) if self.datatype == 'epoched' else [None]:
                if self.datatype == 'epoched':
                    trial_data = np.squeeze(eeg[idx].get_data())
                    trial_times = eeg[idx].times * 1000
                else:
                    trial_data = eeg_data
                    window_size = 5
                    # Smooth Data
                    weights = np.repeat(1.0, window_size) / window_size
                    trial_data = np.apply_along_axis(lambda x: np.convolve(x, weights, mode='same'), axis=1,
                                                        arr=trial_data)

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
                                                                          [self.smooth_param[0],
                                                                           remove_segments_less_than,
                                                                           self.smooth_param[2]])

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
                segmentation_fit = segmentation_fit + similarity_metric

                # Call the export_segmentation method
                if idx is not None:
                    trial_filename = f"{filename}_{idx}"
                else:
                    trial_filename = filename

                segmentation_io.export_segmentation(
                    self.save_path, trial_filename, labeled_segmentation, trial_times, self.export_format
                )

            progress_dialog.update_progress(eeg_idx + 1, len(list_eeg_path))
            progress_bar.update(1)

        progress_dialog.close()
        progress_bar.close()
        #print(segmentation_fit / len(list_eeg_path))
