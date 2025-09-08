"""Backfitting algorithms for EEG microstates used by EEG-COMET."""

from collections import Counter
import numpy as np
from scipy.signal import find_peaks
from scipy import stats


class MicrostateBackfitter:
    """The MicrostateBackfitter class provides methods for backfitting microstates to EEG data.
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

    def __init__(
        self,
        study_name,
        preprocessed_data_path,
        microstate_maps,
        backfit_to,
        filter_segments,
        filter_segments_option,
        identify_short_window,
        micro_labels,
        segmentation_path,
        extension,
        datatype,
        sample_rate,
        smoothing_parameters,
        export_format,
    ):
        """Initializes a new instance of the MicrostateBackfitter class."""
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
        """Fill the groups of -1 values in the array with the neighbor that has a higher count."""
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
                    window_before = segmentation[max(start - 2, 0) : start]
                    prev_count = Counter(window_before).most_common(1)[0][1]

                next_count = 0
                if end < n - 1:
                    window_after = segmentation[end + 2 : min(end + 3, n)]
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
        """Fill the groups of -1 values in the array by evenly distributing the neighboring values."""
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
        """Smooth the segmentation based on the given parameters.

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
        data_sum_sq = np.sum(data**2)
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
            raw_segmentation[b : n_samples - b] = np.argmin(
                (
                    np.sum(data**2, axis=0)
                    - (np.sum(microstate_maps[segmentation].T * data, axis=0) ** 2)
                )[b : n_samples - b]
                / (2 * e2 * (n_channels - 1))
                - (lamb * n_bkt).T,
                axis=0,
            )

            # STEP 6 in TABLE 2
            segmentation = raw_segmentation  # .copy()

            # STEP 7 in TABLE 2
            act_sum_sq = np.sum(np.sum(microstate_maps[segmentation].T * data, axis=0) ** 2)
            e1 = abs(data_sum_sq - act_sum_sq)
            sigma_mu = e1 / float(n_samples * (n_channels - 1))
            residual = abs(prev_residual - sigma_mu)

            # STEP 8 in TABLE 2
            prev_residual = sigma_mu
            thresh = epsilon * sigma_mu
            iteration += 1

        return segmentation

    def substitude_maps_with_duration(
        self,
        segmentation,
        segments_less_than,
        option,
        data,
        microstate_maps,
        n_states,
        smoothing_parameters=None,
    ):
        """Substitute short segments in the segmentation array with neighboring elements based on the chosen option.

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

        if option == "smooth":
            filled_segmentation = self.segmentation_smooth(
                data, microstate_maps, n_states, *smoothing_parameters
            )
            filled_segmentation = self.mark_short_segments(filled_segmentation, segments_less_than)
            filled_segmentation = self.fill_with_neighbors_with_higher_count(filled_segmentation)
        else:
            filled_segmentation = self.mark_short_segments(segmentation, segments_less_than)

        if option == "replace_high":
            filled_segmentation = self.mark_short_segments(segmentation, segments_less_than)
            filled_segmentation = self.fill_with_neighbors_with_higher_count(filled_segmentation)

        if option == "replace_half":
            filled_segmentation = self.mark_short_segments(segmentation, segments_less_than)
            filled_segmentation = self.fill_with_neighbors_half(filled_segmentation)

        return filled_segmentation

    def label_segments(self, segmentation):
        """Label the segments in the segmentation array.

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
            labeled_segmentation = np.char.replace(
                labeled_segmentation, str(m), self.microstate_labels[m - 1]
            )
        return labeled_segmentation

    def compute_correlation_matrix(self, data_2d):
        """Compute correlation matrix between `self.microstate_maps` and data.

        Shapes:
        - microstate_maps: (n_states, n_channels)
        - data_2d: (n_channels, n_samples)
        Returns:
        - correlation_matrix: (n_states, n_samples)
        """
        denominator = np.sqrt(
            np.sum(self.microstate_maps**2, axis=1)[:, np.newaxis]
            * np.sum(data_2d**2, axis=0)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation_matrix = np.dot(self.microstate_maps, data_2d) / denominator
        return np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)


    @staticmethod
    def mark_short_segments(segmentation, min_occurrence):
        """Mark short segments in the segmentation array with -1.

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



    def identify_optimal_length_filter(self, eeg_files_data, progress_callback=None):
        """Identify the optimal length filter using simplified and efficient approach.

        Args:
            eeg_files_data (list): List of EEG data arrays or file paths to analyze.
            progress_callback (callable, optional): Callback for progress updates.

        Returns:
            int: The optimal length filter in milliseconds.
        """
        optimal_thresholds = []
        total_files = len(eeg_files_data)
        
        for file_idx, eeg_data in enumerate(eeg_files_data):
            if progress_callback:
                progress_callback(file_idx, total_files, f"Analyzing data file {file_idx + 1}/{total_files}")
            
            try:
                # Handle different input types
                if isinstance(eeg_data, str):
                    continue  # Skip file paths for now
                elif hasattr(eeg_data, 'get_data'):
                    data_array = eeg_data.get_data()
                elif isinstance(eeg_data, np.ndarray):
                    data_array = eeg_data
                else:
                    continue
                
                # Ensure 2D format
                if data_array.ndim == 3:
                    data_2d = np.concatenate([data_array[i] for i in range(data_array.shape[0])], axis=1)
                else:
                    data_2d = data_array
                
                if data_2d.size == 0:
                    continue
                
                # Find optimal threshold for this file
                optimal_threshold = self.find_optimal_threshold(data_2d)
                optimal_thresholds.append(optimal_threshold)
                
            except Exception:
                continue
        
        if optimal_thresholds:
            median_threshold = np.median(optimal_thresholds)
            
            # Convert to valid sample count and back to ensure sampling-rate alignment
            optimal_samples = int(round(median_threshold * self.sample_rate / 1000))
            optimal_samples = max(1, optimal_samples)  # At least 1 sample
            final_threshold_ms = optimal_samples * 1000 / self.sample_rate
            
            if progress_callback:
                progress_callback(total_files, total_files, f"Optimal threshold: {final_threshold_ms:.1f}ms ({optimal_samples} samples)")
            return final_threshold_ms
        else:
            # Default 20ms threshold, also adjusted for sampling rate
            default_samples = max(1, int(round(20 * self.sample_rate / 1000)))
            default_threshold_ms = default_samples * 1000 / self.sample_rate
            return default_threshold_ms
    
    def find_optimal_lambda_for_files(self, eeg_files_data, threshold_ms, progress_callback=None):
        """Find optimal lambda (non-smoothness penalty) across multiple files.
        
        Args:
            eeg_files_data (list): List of EEG data arrays to analyze
            threshold_ms (float): Threshold duration in milliseconds to use
            progress_callback (callable, optional): Callback for progress updates
            
        Returns:
            float: Optimal lambda value
        """
        optimal_lambdas = []
        total_files = len(eeg_files_data)
        
        for file_idx, eeg_data in enumerate(eeg_files_data):
            if progress_callback:
                progress_callback(total_files + file_idx + 1, total_files * 2, f"Optimizing smoothing parameter for file {file_idx + 1}/{total_files}")
            
            try:
                # Handle different input types
                if isinstance(eeg_data, str):
                    continue  # Skip file paths for now
                elif hasattr(eeg_data, 'get_data'):
                    data_array = eeg_data.get_data()
                elif isinstance(eeg_data, np.ndarray):
                    data_array = eeg_data
                else:
                    continue
                
                # Ensure 2D format
                if data_array.ndim == 3:
                    data_2d = np.concatenate([data_array[i] for i in range(data_array.shape[0])], axis=1)
                else:
                    data_2d = data_array
                
                if data_2d.size == 0:
                    continue
                
                # Find optimal lambda for this file
                optimal_lambda = self.find_optimal_lambda(data_2d, threshold_ms)
                optimal_lambdas.append(optimal_lambda)
                
            except Exception:
                continue
        
        if optimal_lambdas:
            median_lambda = np.median(optimal_lambdas)
            final_lambda = round(median_lambda, 1)
            
            if progress_callback:
                progress_callback(total_files * 2, total_files * 2, f"Optimal smoothing parameter: λ={final_lambda}")
            return final_lambda
        else:
            return 5.0  # Default lambda value

    def find_optimal_threshold(self, eeg_data):
        """Find optimal threshold for filtering short microstate segments.
        
        This method focuses on the most important criteria:
        1. Segment duration distribution analysis
        2. Template correlation quality
        3. Stability of segmentation

        Args:
            eeg_data (np.ndarray): EEG data (n_channels, n_timepoints)

        Returns:
            float: Optimal threshold in milliseconds
        """
        # Get initial segmentation
        correlation_matrix = self.compute_correlation_matrix(eeg_data)
        initial_segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)
        
        # Test fewer thresholds for speed (10 instead of 20)
        test_thresholds_ms = np.linspace(5, 60, 10)  # Focus on realistic range
        
        quality_scores = []
        
        for thresh_ms in test_thresholds_ms:
            thresh_samples = int(thresh_ms * self.sample_rate / 1000)
            
            # Simple approach: mark short segments and compute quality
            filtered_segmentation = self._filter_short_segments(initial_segmentation, thresh_samples)
            
            # Compute quality metrics
            quality = self._compute_quality(eeg_data, filtered_segmentation)
            quality_scores.append(quality)
        
        # Find the threshold with best quality score
        best_idx = np.argmax(quality_scores)
        optimal_threshold = test_thresholds_ms[best_idx]
        
        # Ensure reasonable bounds (between 10-50ms typically)
        optimal_threshold = max(10.0, min(50.0, optimal_threshold))
        
        # Convert to valid sample count and back to ensure it's sampling-rate appropriate
        optimal_samples = int(round(optimal_threshold * self.sample_rate / 1000))
        optimal_samples = max(1, optimal_samples)  # At least 1 sample
        
        # Convert back to milliseconds based on actual sample count
        optimal_threshold_ms = optimal_samples * 1000 / self.sample_rate
        
        return optimal_threshold_ms
    
    def find_optimal_lambda(self, eeg_data, threshold_ms, test_range=(1, 10), n_tests=5):
        """Find optimal non-smoothness penalty (lambda) for smoothing.
        
        Args:
            eeg_data (np.ndarray): EEG data (n_channels, n_timepoints)
            threshold_ms (float): Threshold duration in milliseconds to use for smoothing
            test_range (tuple): Range of lambda values to test (min, max)
            n_tests (int): Number of lambda values to test
            
        Returns:
            float: Optimal lambda value
        """
        # Get initial segmentation
        correlation_matrix = self.compute_correlation_matrix(eeg_data)
        initial_segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)
        
        # Convert threshold to samples
        threshold_samples = int(threshold_ms * self.sample_rate / 1000)
        
        # Test different lambda values
        test_lambdas = np.linspace(test_range[0], test_range[1], n_tests)
        quality_scores = []
        
        for lamb in test_lambdas:
            try:
                # Apply smoothing with this lambda
                smoothed_segmentation = self.segmentation_smooth(
                    eeg_data, self.microstate_maps, len(self.microstate_maps),
                    epsilon=1e-6, b=threshold_samples, lamb=lamb
                )
                
                # Compute quality of smoothed segmentation
                quality = self._compute_quality(eeg_data, smoothed_segmentation)
                quality_scores.append(quality)
                
            except Exception:
                # If smoothing fails, assign low quality
                quality_scores.append(0.0)
        
        # Find lambda with best quality
        if any(score > 0 for score in quality_scores):
            best_idx = np.argmax(quality_scores)
            optimal_lambda = test_lambdas[best_idx]
            
            # Round to reasonable precision
            return round(optimal_lambda, 1)
        else:
            # Fallback to default if all failed
            return 5.0
    
    def _compute_quality(self, eeg_data, segmentation):
        """Compute quality metric focusing on key aspects.

        Args:
            eeg_data (np.ndarray): EEG data (n_channels, n_timepoints)
            segmentation (np.ndarray): Segmentation labels

        Returns:
            float: Quality score (higher is better)
        """
        valid_indices = segmentation != -1
        if not np.any(valid_indices):
            return 0.0
        
        valid_data = eeg_data[:, valid_indices]
        valid_labels = segmentation[valid_indices]
        
        # 1. Template correlation quality (most important)
        template_correlations = []
        unique_labels = np.unique(valid_labels)
        
        for label in unique_labels:
            if 0 <= label < len(self.microstate_maps):
                label_mask = valid_labels == label
                if np.any(label_mask):
                    label_data = valid_data[:, label_mask]
                    template = self.microstate_maps[label]
                    
                    # Compute correlations for this microstate
                    corrs = []
                    for t in range(label_data.shape[1]):
                        corr = np.corrcoef(template, label_data[:, t])[0, 1]
                        if not np.isnan(corr):
                            corrs.append(abs(corr))
                    
                    if corrs:
                        template_correlations.append(np.mean(corrs))
        
        template_quality = np.mean(template_correlations) if template_correlations else 0.0
        
        # 2. Segment stability (penalize too many transitions)
        n_transitions = np.sum(np.diff(valid_labels) != 0)
        stability_score = 1.0 / (1.0 + n_transitions / len(valid_labels))
        
        # 3. Data coverage (penalize removing too much data)
        coverage_score = len(valid_labels) / len(segmentation)
        
        # Combine with appropriate weights
        quality_score = (0.6 * template_quality + 
                        0.3 * stability_score + 
                        0.1 * coverage_score)
        
        return quality_score

    @staticmethod
    def _filter_short_segments(segmentation, min_samples):
        """Filter segments shorter than min_samples."""
        if len(segmentation) == 0:
            return segmentation
            
        filtered_seg = segmentation.copy()
        current_label = filtered_seg[0]
        start_idx = 0
        
        for i in range(1, len(filtered_seg)):
            if filtered_seg[i] != current_label:
                # Check if previous segment was too short
                if i - start_idx < min_samples:
                    # Mark as invalid (can be handled different ways)
                    filtered_seg[start_idx:i] = -1
                
                start_idx = i
                current_label = filtered_seg[i]
        
        # Check final segment
        if len(filtered_seg) - start_idx < min_samples:
            filtered_seg[start_idx:] = -1
        
        return filtered_seg

    def backfit2all(self, data, filter_segments_less_than):
        """Perform segmentation by backfitting microstate maps to all time points in the data.

        Args:
            data (ndarray): The EEG data (can be from a single trial or continuous data).
            filter_segments_less_than (int): The threshold for filtering segments shorter than this value.

        Returns:
            ndarray: The segmentation array where each time point is labeled with a microstate index.
        """
        correlation_matrix = self.compute_correlation_matrix(data)
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
                    self.smoothing_parameters[0],
                    filter_segments_less_than,
                    self.smoothing_parameters[2],
                ],
            )
        return segmentation

    def backfit2peaks(self, data):
        """Perform segmentation by backfitting microstate maps only to the peaks in global field power (GFP).

        Args:
            data (ndarray): The EEG data (can be from a single trial or continuous data).

        Returns:
            ndarray: The segmentation array where each time point is labeled with a microstate index.
        """
        gfp = np.std(data, axis=0)
        peaks, _ = find_peaks(gfp)
        troughs = [0]
        for p in range(len(peaks) - 1):
            min_arg = np.argmin(gfp[peaks[p] : peaks[p + 1]])
            troughs.append(peaks[p] + min_arg)
        troughs.append(len(gfp))
        diff_troughs = np.diff(troughs)
        activation = np.dot(self.microstate_maps, data[:, peaks])
        segmentation_peaks = np.argmax(np.abs(activation), axis=0)
        return np.repeat(segmentation_peaks.astype(int), diff_troughs.astype(int))

    def perform_segmentation(self, eeg, filter_segments_less_than):
        """Perform segmentation on the EEG data based on the specified backfitting method.

        Args:
            eeg (mne.io.Raw | mne.Epochs): An instance of an MNE object, either Raw or Epochs, containing the EEG data.
            filter_segments_less_than (int): The threshold for filtering segments shorter than this value.

        Returns:
            tuple: A tuple containing the labeled segmentation, trial filename, trial times, and segmentation
        """
        segmentation_fit = 0
        eeg_data = eeg.get_data()
        if self.datatype == "epoched":
            segmentation_list = []
            for trial in range(eeg_data.shape[0]):
                trial_data = eeg_data[trial, :, :]
                segmentation = (
                    self.backfit2peaks(trial_data)
                    if self.backfit_to == "peaks"
                    else self.backfit2all(trial_data, filter_segments_less_than)
                )
                labeled_segmentation = self.label_segments(segmentation)
                segmentation_list.append(labeled_segmentation)
            labeled_segmentation_array = np.vstack(segmentation_list)
        else:
            segmentation = (
                self.backfit2peaks(eeg_data)
                if self.backfit_to == "peaks"
                else self.backfit2all(eeg_data, filter_segments_less_than)
            )
            labeled_segmentation_array = self.label_segments(segmentation)

        return labeled_segmentation_array, segmentation_fit
