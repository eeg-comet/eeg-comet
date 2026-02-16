"""Backfitting algorithms for EEG microstates used by EEG-COMET."""

from collections import Counter
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os


class MicrostateBackfitter:
    """Backfitting class for assigning microstate labels to EEG data.
    
    This class implements algorithms for assigning microstate templates to continuous
    or epoched EEG data, with options for temporal smoothing, quality control, and
    segment filtering.
    
    Args:
        study_name (str): Name of the study.
        preprocessed_data_path (str): Path to preprocessed data files.
        microstate_maps (np.ndarray): Microstate template maps (n_states, n_channels).
        backfit_to (str): Assignment method - 'all' for all timepoints, 'peaks' for GFP peaks only.
        filter_segments (bool): Whether to apply segment filtering.
        filter_segments_option (str): Filtering method - 'remove', 'replace_high', 
            'replace_half', or 'smooth'.
        identify_short_window (bool): Whether to identify optimal filtering threshold.
        micro_labels (list): Labels for each microstate class (e.g., ['A', 'B', 'C', 'D', 'E']).
        segmentation_path (str): Path for saving segmentation results.
        extension (str): File extension for outputs.
        datatype (str): Data format - 'continuous' or 'epoched'.
        sample_rate (int): Sampling rate in Hz.
        smoothing_parameters (list, optional): [epsilon, half_window_size, lambda] for smoothing.
            half_window_size (b) is the half-window as defined in Pascual-Marqui et al. (1995).
            Defaults to [1e-6, 3, 10].
        export_format (str): Format for exporting results.
        min_correlation_threshold (float or False, optional): Minimum absolute correlation 
            threshold for quality control. If False, no correlation filtering is applied.
            Values typically range from 0.5 (liberal) to 0.7 (conservative). Default: False.
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
        min_correlation_threshold=False,
    ):
        """Initialize the MicrostateBackfitter."""
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
        self.smoothing_parameters = smoothing_parameters if smoothing_parameters else [1e-6, 3, 10]
        self.export_format = export_format
        self.min_correlation_threshold = min_correlation_threshold
        if self.microstate_maps.shape[0] != len(self.microstate_labels):
            raise ValueError(
                "microstate_maps.shape[0] must equal len(micro_labels); "
                f"got {self.microstate_maps.shape[0]} and {len(self.microstate_labels)}."
            )

    @staticmethod
    def fill_with_neighbors_with_higher_count(segmentation):
        """Fill rejected segments (-1) with the more frequent neighboring label.
        
        Args:
            segmentation (np.ndarray): Segmentation with -1 values marking rejected segments.
            
        Returns:
            np.ndarray: Segmentation with -1 values replaced.
        """
        filled_segmentation = segmentation.copy()
        n = len(segmentation)
        i = 0

        while i < n:
            if segmentation[i] == -1:
                start = i
                while i < n and segmentation[i] == -1:
                    i += 1
                end = i - 1

                prev = segmentation[start - 1] if start > 0 else -1
                next_val = segmentation[end + 1] if end < n - 1 else -1

                if prev < 0 and next_val < 0:
                    pass  # no neighbor: leave as -1
                elif prev < 0:
                    fill_value = next_val
                    for j in range(start, end + 1):
                        filled_segmentation[j] = fill_value
                elif next_val < 0:
                    fill_value = prev
                    for j in range(start, end + 1):
                        filled_segmentation[j] = fill_value
                else:
                    prev_count = 0
                    if start > 0:
                        window_before = segmentation[max(start - 2, 0) : start]
                        prev_count = Counter(window_before).most_common(1)[0][1]

                    next_count = 0
                    if end < n - 1:
                        window_after = segmentation[end + 2 : min(end + 3, n)]
                        next_count = Counter(window_after).most_common(1)[0][1]

                    if prev_count > next_count:
                        fill_value = prev
                    elif prev_count < next_count:
                        fill_value = next_val
                    else:
                        fill_value = prev if prev >= 0 else next_val
                    for j in range(start, end + 1):
                        filled_segmentation[j] = fill_value
            else:
                i += 1

        return filled_segmentation

    @staticmethod
    def fill_with_neighbors_half(segmentation):
        """Fill rejected segments (-1) by splitting evenly between neighbors.
        
        Args:
            segmentation (np.ndarray): Segmentation with -1 values marking rejected segments.
            
        Returns:
            np.ndarray: Segmentation with -1 values replaced.
        """
        filled_segmentation = segmentation.copy()
        n = len(segmentation)
        i = 0
        count = 0

        while i < n:
            if segmentation[i] == -1:
                start = i
                while i < n and segmentation[i] == -1:
                    i += 1
                    count += 1
                end = i - 1

                fill_value_prev = segmentation[start - 1] if start > 0 else -1
                fill_value_next = segmentation[end + 1] if end < n - 1 else -1

                if fill_value_prev < 0 and fill_value_next < 0:
                    pass  # no neighbor: leave chunk as -1
                elif fill_value_prev < 0:
                    for j in range(start, end + 1):
                        filled_segmentation[j] = fill_value_next
                elif fill_value_next < 0:
                    for j in range(start, end + 1):
                        filled_segmentation[j] = fill_value_prev
                else:
                    half_count = count // 2
                    half_count_prev = half_count if count % 2 == 0 else half_count + 1
                    for j in range(start, start + half_count_prev):
                        filled_segmentation[j] = fill_value_prev
                    for j in range(start + half_count_prev, end + 1):
                        filled_segmentation[j] = fill_value_next

                count = 0
            else:
                i += 1

        return filled_segmentation

    @staticmethod
    def segmentation_smooth(data, microstate_maps, n_states, initial_segmentation=None,
                           epsilon=1e-6, half_window_size=3, lamb=10):
        """Apply Pascual-Marqui et al. (1995) temporal smoothing algorithm.
        
        Balances spatial correspondence with microstate templates against temporal
        smoothness using iterative optimization. Preserves rejected timepoints (-1)
        throughout the smoothing process.
        
        Args:
            data (numpy.ndarray): EEG data (n_channels, n_samples).
            microstate_maps (numpy.ndarray): Template maps (n_states, n_channels).
            n_states (int): Number of microstate classes.
            initial_segmentation (numpy.ndarray, optional): Pre-computed segmentation
                that may contain -1 for rejected timepoints. If None, performs initial
                assignment.
            epsilon (float): Convergence threshold for relative GEV change. Default: 1e-6.
            half_window_size (int): Half temporal window size (b) in samples, as defined
                in Pascual-Marqui et al. (1995). The full window spans
                [t - b, t + b]. At 250 Hz, b=3 equals a 28 ms window. Default: 3.
            lamb (float): Besag factor (non-smoothness penalty). Higher values increase
                temporal continuity. Standard value: 10. Default: 10.
        
        Returns:
            numpy.ndarray: Smoothed segmentation with -1 preserved for rejected timepoints.
        
        References:
            Pascual-Marqui, R.D., Michel, C.M., & Lehmann, D. (1995). Segmentation of 
            brain electrical activity into microstates: model estimation and validation.
            IEEE Trans Biomed Eng, 42(7), 658-665.
        """
        n_channels, n_samples = data.shape
        half_window = half_window_size
        
        # Pre-center and normalize templates once (consistent formulation)
        maps_centered = microstate_maps - np.mean(microstate_maps, axis=1, keepdims=True)
        maps_norms = np.linalg.norm(maps_centered, axis=1, keepdims=True)
        maps_normalized = maps_centered / (maps_norms + 1e-10)
        
        # Pre-center data and compute centered norms for all timepoints
        data_centered = data - np.mean(data, axis=0, keepdims=True)
        data_norms = np.linalg.norm(data_centered, axis=0)
        data_normalized = data_centered / (data_norms[np.newaxis, :] + 1e-10)
        
        if initial_segmentation is None:
            activation = maps_normalized.dot(data_normalized)
            segmentation = np.argmax(np.abs(activation), axis=0)
            valid_mask = np.ones(n_samples, dtype=bool)
        else:
            segmentation = initial_segmentation.copy()
            valid_mask = segmentation >= 0
        
        n_valid = np.sum(valid_mask)
        if n_valid == 0:
            return segmentation
        
        # Compute sigma² using centered quantities for consistency:
        # sigma² = Σ(||x_c||² * (1 - corr²)) / (N * (Ne - 1))
        valid_indices = np.where(valid_mask)[0]
        data_sum_sq = np.sum(data_norms[valid_mask] ** 2)
        
        act_sum_sq = 0.0
        for tf in valid_indices:
            corr = np.dot(maps_normalized[segmentation[tf]], data_normalized[:, tf])
            act_sum_sq += (data_norms[tf] * corr) ** 2
        
        origsigma2 = (data_sum_sq - act_sum_sq) / (n_valid * (n_channels - 1))
        
        if origsigma2 <= 0:
            return segmentation
        
        gev = act_sum_sq / data_sum_sq
        max_iter = 20
        convergence_threshold = epsilon
        
        for smoothi in range(max_iter):
            temp_segmentation = segmentation.copy()
            
            # Only smooth valid (not rejected) timepoints
            for tf in valid_indices:
                histo = np.zeros(n_states, dtype=int)
                
                win_start = max(0, tf - half_window)
                win_end = min(n_samples - 1, tf + half_window)
                
                # Count only valid neighbors
                for tf2 in range(win_start, win_end + 1):
                    if tf2 != tf and segmentation[tf2] >= 0:
                        histo[segmentation[tf2]] += 1
                
                diffmin = np.inf
                best_nc = segmentation[tf]
                data_norm_c = data_norms[tf]
                
                for nc in range(n_states):
                    # Pearson correlation via pre-centered+normalized vectors
                    corr = np.dot(maps_normalized[nc], data_normalized[:, tf])
                    abs_corr_sq = corr ** 2
                    
                    # Cost function with centered norm (consistent with Pearson correlation)
                    diff = (data_norm_c ** 2) * (1 - abs_corr_sq) / (2 * origsigma2 * (n_channels - 1)) - lamb * histo[nc]
                    
                    if diff < diffmin:
                        diffmin = diff
                        best_nc = nc
                
                temp_segmentation[tf] = best_nc
            
            # Stop when no assignment changed (stable segmentation)
            if np.array_equal(segmentation, temp_segmentation):
                break
            # Update only valid timepoints, preserve -1 for rejected
            segmentation[valid_mask] = temp_segmentation[valid_mask]
            
            gevbefore = gev
            
            # Recompute GEV using centered quantities
            act_sum_sq = 0.0
            for tf in valid_indices:
                corr = np.dot(maps_normalized[segmentation[tf]], data_normalized[:, tf])
                act_sum_sq += (data_norms[tf] * corr) ** 2
            
            gev = act_sum_sq / data_sum_sq
            
            rel_diff = abs(gev - gevbefore) / (gevbefore + 1e-10)
            if rel_diff <= convergence_threshold:
                break
        
        return segmentation

    def substitute_maps_with_duration(
        self,
        segmentation,
        segments_less_than,
        option,
        data,
        microstate_maps,
        n_states,
        smoothing_parameters=None,
    ):
        """Apply segment filtering and smoothing based on specified option.

        Args:
            segmentation (numpy.ndarray): Segmentation array (may contain -1 for rejected points).
            segments_less_than (int): Minimum segment duration threshold in samples.
            option (str): Filtering method - 'remove', 'replace_high', 'replace_half', or 'smooth'.
            data (numpy.ndarray): EEG data (n_channels, n_samples).
            microstate_maps (numpy.ndarray): Template maps (n_states, n_channels).
            n_states (int): Number of microstate classes.
            smoothing_parameters (list, optional): [epsilon, half_window_size, lambda].
                Defaults to [1e-6, 3, 10].

        Returns:
            numpy.ndarray: Filtered/smoothed segmentation array.
        """
        if smoothing_parameters is None:
            smoothing_parameters = [1e-6, 3, 10]

        if option == "smooth":
            # Step 1: Apply Pascual-Marqui temporal smoothing
            filled_segmentation = self.segmentation_smooth(
                data,
                microstate_maps,
                n_states,
                initial_segmentation=segmentation,
                epsilon=smoothing_parameters[0],
                half_window_size=smoothing_parameters[1],
                lamb=smoothing_parameters[2]
            )
            # Precompute normalized maps and data once for redistribute loop
            maps_centered = microstate_maps - np.mean(microstate_maps, axis=1, keepdims=True)
            maps_norms = np.linalg.norm(maps_centered, axis=1, keepdims=True)
            maps_normalized = maps_centered / (maps_norms + 1e-10)
            data_centered = data - np.mean(data, axis=0, keepdims=True)
            data_norms = np.linalg.norm(data_centered, axis=0)
            data_normalized = data_centered / (data_norms[np.newaxis, :] + 1e-10)
            # Step 2: Reject short segments and distribute to neighbors (iterative).
            # Redistribution can create new short segments at split boundaries, so
            # repeat mark + distribute until no segment <= segments_less_than remains,
            # matching CARTOOL behavior where the final segmentation has no short segments.
            max_reject_iterations = 15
            for _ in range(max_reject_iterations):
                marked = self.mark_short_segments(filled_segmentation, segments_less_than)
                if not np.any(marked == -1):
                    break
                filled_segmentation = self._distribute_rejected_to_neighbors(
                    marked, data, microstate_maps,
                    maps_normalized=maps_normalized, data_normalized=data_normalized
                )
            
        elif option == "replace_high":
            filled_segmentation = self.mark_short_segments(segmentation, segments_less_than)
            filled_segmentation = self.fill_with_neighbors_with_higher_count(filled_segmentation)
            
        elif option == "replace_half":
            filled_segmentation = self.mark_short_segments(segmentation, segments_less_than)
            filled_segmentation = self.fill_with_neighbors_half(filled_segmentation)
            
        else:  # "remove"
            filled_segmentation = self.mark_short_segments(segmentation, segments_less_than)

        return filled_segmentation

    @staticmethod
    def _distribute_rejected_to_neighbors(segmentation, data, microstate_maps,
                                          maps_normalized=None, data_normalized=None):
        """Distribute rejected segments to neighboring segments using spatial correlation.

        For each contiguous chunk of rejected timepoints (-1), determines the
        optimal label assignment by comparing spatial correlation with the left
        and right neighbor templates. Scans from both ends of the gap inward
        and splits at the crossover point where the opposite neighbor's template
        becomes a better spatial fit.

        Args:
            segmentation (numpy.ndarray): Segmentation with -1 for rejected timepoints.
            data (numpy.ndarray): EEG data (n_channels, n_samples).
            microstate_maps (numpy.ndarray): Template maps (n_states, n_channels).
            maps_normalized (numpy.ndarray, optional): Pre-centered and L2-normalized
                template maps. If None, computed from microstate_maps.
            data_normalized (numpy.ndarray, optional): Pre-centered and L2-normalized
                data. If None, computed from data.

        Returns:
            numpy.ndarray: Segmentation with rejected timepoints re-assigned.
        """
        if maps_normalized is None:
            maps_centered = microstate_maps - np.mean(microstate_maps, axis=1, keepdims=True)
            maps_norms = np.linalg.norm(maps_centered, axis=1, keepdims=True)
            maps_normalized = maps_centered / (maps_norms + 1e-10)
        if data_normalized is None:
            data_centered = data - np.mean(data, axis=0, keepdims=True)
            data_norms = np.linalg.norm(data_centered, axis=0)
            data_normalized = data_centered / (data_norms[np.newaxis, :] + 1e-10)

        return MicrostateBackfitter._distribute_rejected_to_neighbors_impl(
            segmentation, maps_normalized, data_normalized
        )

    @staticmethod
    def _distribute_rejected_to_neighbors_impl(segmentation, maps_normalized, data_normalized):
        """Distribute rejected segments using precomputed normalized maps and data."""
        result = segmentation.copy()
        n_samples = len(segmentation)

        if not np.any(segmentation == -1):
            return result

        # Scan for contiguous rejected chunks and distribute to neighbors
        i = 0
        while i < n_samples:
            if result[i] == -1:
                seg_from = i
                while i < n_samples and result[i] == -1:
                    i += 1
                seg_to = i - 1

                # Identify left and right neighbor labels
                left_label = result[seg_from - 1] if seg_from > 0 else -1
                right_label = result[seg_to + 1] if seg_to < n_samples - 1 else -1

                # No valid neighbors: skip
                if left_label < 0 and right_label < 0:
                    continue

                # Only one neighbor: assign entire chunk to that neighbor
                if left_label < 0:
                    result[seg_from:seg_to + 1] = right_label
                    continue
                if right_label < 0:
                    result[seg_from:seg_to + 1] = left_label
                    continue

                # Same label on both sides: fill directly
                if left_label == right_label:
                    result[seg_from:seg_to + 1] = left_label
                    continue

                # Both neighbors are different valid labels: split based on
                # spatial correlation, scanning inward from both sides
                left_map = maps_normalized[left_label]
                right_map = maps_normalized[right_label]

                # Scan from left: assign to left_label while left corr >= right corr
                tfl = seg_from
                for tf in range(seg_from, seg_to + 1):
                    col = abs(np.dot(left_map, data_normalized[:, tf]))
                    cor = abs(np.dot(right_map, data_normalized[:, tf]))

                    if col < cor:
                        break
                    result[tf] = left_label
                    tfl = tf + 1

                # Left side took everything
                if tfl > seg_to:
                    continue

                # Scan from right: assign to right_label while right corr >= left corr
                tfr = seg_to
                for tf in range(seg_to, tfl - 1, -1):
                    col = abs(np.dot(left_map, data_normalized[:, tf]))
                    cor = abs(np.dot(right_map, data_normalized[:, tf]))

                    if cor < col:
                        break
                    result[tf] = right_label
                    tfr = tf - 1

                # Central remaining part: split evenly
                if tfl <= tfr:
                    tfm = (tfl + tfr + 1) // 2
                    result[tfl:tfm] = left_label
                    result[tfm:tfr + 1] = right_label
            else:
                i += 1

        return result

    def label_segments(self, segmentation):
        """Convert numeric segmentation to string labels.

        Args:
            segmentation (numpy.ndarray): Numeric segmentation array.

        Returns:
            numpy.ndarray: String-labeled segmentation array.
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
        """Compute spatial correlation between microstate templates and data.
        
        Uses Pearson correlation (mean-centered across channels), which is the
        standard approach in microstate analysis for computing spatial similarity.

        Args:
            data_2d (numpy.ndarray): EEG data (n_channels, n_samples).

        Returns:
            numpy.ndarray: Correlation matrix (n_states, n_samples).
        """
        # Mean-center templates across channels
        maps_centered = self.microstate_maps - np.mean(self.microstate_maps, axis=1, keepdims=True)
        maps_norm = np.linalg.norm(maps_centered, axis=1, keepdims=True)
        maps_normalized = maps_centered / (maps_norm + 1e-10)
        
        # Mean-center data across channels (for each timepoint)
        data_centered = data_2d - np.mean(data_2d, axis=0, keepdims=True)
        data_norm = np.linalg.norm(data_centered, axis=0, keepdims=True)
        data_normalized = data_centered / (data_norm + 1e-10)
        
        # Compute Pearson correlation
        correlation_matrix = np.dot(maps_normalized, data_normalized)
        
        return np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def reject_low_correlation_labels(correlation_matrix, segmentation, threshold=False):
        """Reject timepoints with poor template matching based on spatial correlation.
        
        This quality control step marks timepoints as -1 (rejected) if their absolute
        correlation with the assigned microstate template falls below the threshold.
        Only valid labels (>= 0) are threshold-tested; any -1 already present in
        segmentation are left unchanged.
        
        Args:
            correlation_matrix (np.ndarray): Spatial correlations (n_states, n_timepoints).
            segmentation (np.ndarray): Assigned labels (n_timepoints,). May contain -1.
            threshold (float or False): Minimum absolute correlation threshold. If False,
                no filtering is applied. Values typically range from 0.5 (liberal) to 
                0.7 (conservative). Default: False.
        
        Returns:
            np.ndarray: Segmentation with low-correlation timepoints marked as -1 (if threshold is not False).
        """
        # If threshold is False, skip filtering
        if threshold is False:
            return segmentation
        
        segmentation = segmentation.copy()
        n_timepoints = segmentation.shape[0]
        valid_mask = segmentation >= 0
        if not np.any(valid_mask):
            return segmentation
        
        timepoint_indices = np.arange(n_timepoints)
        valid_indices = timepoint_indices[valid_mask]
        valid_labels = segmentation[valid_mask].astype(int)
        
        # Extract correlation only for valid timepoints (avoid indexing with -1)
        assigned_correlations = np.abs(correlation_matrix[valid_labels, valid_indices])
        low_correlation = assigned_correlations < threshold
        segmentation[valid_indices[low_correlation]] = -1
        
        return segmentation

    @staticmethod
    def mark_short_segments(segmentation, min_occurrence):
        """Mark segments with duration ≤ min_occurrence as rejected (-1).

        Segments shorter than or equal to the threshold (in samples) are marked -1.
        Longer segments are left unchanged.

        Args:
            segmentation (numpy.ndarray): Segmentation array.
            min_occurrence (int): Maximum duration (samples) for a segment to be
                considered short; segments with duration ≤ this are rejected.

        Returns:
            numpy.ndarray: Segmentation with short segments marked as -1.
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

        # Process final segment (use <= consistently with intermediate segments)
        if current_count <= min_occurrence:
            new_segmentation.extend([-1] * current_count)
        else:
            new_segmentation.extend([current_element] * current_count)

        return np.asarray(new_segmentation)

    def identify_optimal_length_filter(self, eeg_files_data, progress_callback=None):
        """Identify optimal segment duration threshold across multiple files.

        Args:
            eeg_files_data (list): List of EEG data arrays to analyze.
            progress_callback (callable, optional): Progress update callback.

        Returns:
            float: Optimal threshold duration in milliseconds.
        """
        optimal_thresholds = []
        total_files = len(eeg_files_data)
        
        for file_idx, eeg_data in enumerate(eeg_files_data):
            if progress_callback:
                progress_callback(file_idx, total_files, 
                                f"Analyzing file {file_idx + 1}/{total_files}")
            
            try:
                if isinstance(eeg_data, str):
                    continue
                elif hasattr(eeg_data, 'get_data'):
                    data_array = eeg_data.get_data()
                elif isinstance(eeg_data, np.ndarray):
                    data_array = eeg_data
                else:
                    continue
                
                if data_array.ndim == 3:
                    data_2d = np.concatenate([data_array[i] for i in range(data_array.shape[0])], axis=1)
                else:
                    data_2d = data_array
                
                if data_2d.size == 0:
                    continue
                
                optimal_threshold = self.find_optimal_threshold(data_2d)
                optimal_thresholds.append(optimal_threshold)
                
            except Exception:
                continue
        
        if optimal_thresholds:
            median_threshold = np.median(optimal_thresholds)
            optimal_samples = int(round(median_threshold * self.sample_rate / 1000))
            optimal_samples = max(1, optimal_samples)
            final_threshold_ms = optimal_samples * 1000 / self.sample_rate
            
            if progress_callback:
                progress_callback(total_files, total_files, 
                                f"Optimal threshold: {final_threshold_ms:.1f}ms")
            return final_threshold_ms
        else:
            default_samples = max(1, int(round(20 * self.sample_rate / 1000)))
            return default_samples * 1000 / self.sample_rate
    
    def find_optimal_lambda_for_files(self, eeg_files_data, threshold_ms, progress_callback=None):
        """Find optimal Besag factor (lambda) across multiple files.
        
        Args:
            eeg_files_data (list): List of EEG data arrays to analyze.
            threshold_ms (float): Segment duration threshold in milliseconds.
            progress_callback (callable, optional): Progress update callback.
            
        Returns:
            float: Optimal lambda (Besag factor) value.
        """
        optimal_lambdas = []
        total_files = len(eeg_files_data)
        
        for file_idx, eeg_data in enumerate(eeg_files_data):
            if progress_callback:
                progress_callback(total_files + file_idx + 1, total_files * 2, 
                                f"Optimizing lambda for file {file_idx + 1}/{total_files}")
            
            try:
                if isinstance(eeg_data, str):
                    continue
                elif hasattr(eeg_data, 'get_data'):
                    data_array = eeg_data.get_data()
                elif isinstance(eeg_data, np.ndarray):
                    data_array = eeg_data
                else:
                    continue
                
                if data_array.ndim == 3:
                    data_2d = np.concatenate([data_array[i] for i in range(data_array.shape[0])], axis=1)
                else:
                    data_2d = data_array
                
                if data_2d.size == 0:
                    continue
                
                optimal_lambda = self.find_optimal_lambda(data_2d, threshold_ms)
                optimal_lambdas.append(optimal_lambda)
                
            except Exception:
                continue
        
        if optimal_lambdas:
            median_lambda = np.median(optimal_lambdas)
            final_lambda = round(median_lambda, 1)
            
            if progress_callback:
                progress_callback(total_files * 2, total_files * 2, 
                                f"Optimal lambda: {final_lambda}")
            return final_lambda
        else:
            return 10.0  # Standard literature value

    def find_optimal_threshold(self, eeg_data, return_plot_data=False):
        """Find optimal segment duration threshold via quality optimization.

        Args:
            eeg_data (np.ndarray): EEG data (n_channels, n_timepoints).
            return_plot_data (bool): Whether to return plotting data.

        Returns:
            float or tuple: Optimal threshold in ms, or tuple with plotting data.
        """
        correlation_matrix = self.compute_correlation_matrix(eeg_data)
        initial_segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)

        min_samples = max(1, int(2 * self.sample_rate / 1000))
        max_samples = int(60 * self.sample_rate / 1000)
        
        if max_samples - min_samples <= 20:
            sample_counts = np.arange(min_samples, max_samples + 1)
        else:
            sample_counts = np.linspace(min_samples, max_samples, 20, dtype=int)
        
        test_thresholds_ms = sample_counts * 1000 / self.sample_rate
        quality_scores = []

        for thresh_ms in test_thresholds_ms:
            thresh_samples = int(thresh_ms * self.sample_rate / 1000)
            filtered_segmentation = self._filter_short_segments(initial_segmentation, thresh_samples)
            quality = self._compute_quality(eeg_data, filtered_segmentation)
            quality_scores.append(quality)

        best_idx = np.argmax(quality_scores)
        raw_optimal_threshold = test_thresholds_ms[best_idx]

        # Apply constraints (5-50 ms range)
        min_samples = max(1, int(5 * self.sample_rate / 1000))
        max_samples = int(50 * self.sample_rate / 1000)
        raw_optimal_samples = int(round(raw_optimal_threshold * self.sample_rate / 1000))
        
        optimal_samples = np.clip(raw_optimal_samples, min_samples, max_samples)
        optimal_samples = max(1, optimal_samples)
        optimal_threshold_ms = optimal_samples * 1000 / self.sample_rate

        if return_plot_data:
            return optimal_threshold_ms, test_thresholds_ms, quality_scores, {
                'raw_optimal': raw_optimal_threshold,
                'constraint_applied': optimal_samples != raw_optimal_samples,
                'best_idx': best_idx
            }
        return optimal_threshold_ms

    def plot_threshold_optimization(self, eeg_data, save_path=None):
        """Generate visualization of threshold optimization process.

        Args:
            eeg_data (np.ndarray): EEG data (n_channels, n_timepoints).
            save_path (str, optional): Path to save plot.

        Returns:
            str: Path where plot was saved, or None if failed.
        """
        try:
            result = self.find_optimal_threshold(eeg_data, return_plot_data=True)

            if len(result) == 4:
                optimal_threshold, test_thresholds, quality_scores, info = result
            else:
                return None

            if len(test_thresholds) == 0 or len(quality_scores) == 0:
                return None

            plt.figure(figsize=(12, 8))

            # Quality score plot
            plt.subplot(2, 1, 1)
            plt.plot(test_thresholds, quality_scores, 'b-o', linewidth=2, markersize=6)
            plt.plot(test_thresholds[info['best_idx']], quality_scores[info['best_idx']],
                     'go', markersize=12, label=f"Maximum: {info['raw_optimal']:.1f}ms")

            if info['constraint_applied']:
                optimal_idx = np.argmin(np.abs(test_thresholds - optimal_threshold))
                plt.plot(test_thresholds[optimal_idx], quality_scores[optimal_idx],
                         'ro', markersize=10, label=f"Selected: {optimal_threshold:.1f}ms")

            plt.xlabel('Threshold Duration (ms)', fontsize=12)
            plt.ylabel('Quality Score', fontsize=12)
            plt.title('Threshold Optimization', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)

            # Data retention plot
            plt.subplot(2, 1, 2)
            correlation_matrix = self.compute_correlation_matrix(eeg_data)
            initial_segmentation = np.argmax(np.abs(correlation_matrix), axis=0)

            retention_rates = []
            for thresh_ms in test_thresholds:
                thresh_samples = int(thresh_ms * self.sample_rate / 1000)
                filtered_seg = self._filter_short_segments(initial_segmentation, thresh_samples)
                retention_rate = np.sum(filtered_seg != -1) / len(filtered_seg) * 100
                retention_rates.append(retention_rate)

            plt.plot(test_thresholds, retention_rates, 'purple', linewidth=2)
            plt.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7,
                        label=f'Selected: {optimal_threshold:.1f}ms')
            plt.xlabel('Threshold Duration (ms)', fontsize=12)
            plt.ylabel('Data Retention (%)', fontsize=12)
            plt.title('Data Retention vs Threshold', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)

            plt.tight_layout()

            if save_path is None:
                save_dir = self.segmentation_path if self.segmentation_path else os.getcwd()
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'threshold_optimization.png')

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
                
        except Exception:
            try:
                plt.close()
            except:
                pass
            return None
    
    def find_optimal_lambda(self, eeg_data, threshold_ms, test_range=(5, 15), n_tests=5):
        """Find optimal Besag factor through quality optimization.
        
        Args:
            eeg_data (np.ndarray): EEG data (n_channels, n_timepoints).
            threshold_ms (float): Segment duration threshold in milliseconds.
            test_range (tuple): Range of lambda values to test. Default: (5, 15).
            n_tests (int): Number of lambda values to test. Default: 5.
            
        Returns:
            float: Optimal lambda value.
        """
        correlation_matrix = self.compute_correlation_matrix(eeg_data)
        initial_segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)
        
        # Apply correlation rejection before smoothing (if threshold is set)
        initial_segmentation = self.reject_low_correlation_labels(
            correlation_matrix, initial_segmentation, threshold=self.min_correlation_threshold
        )
        
        half_window_size = 3  # Standard half-window (b) per Pascual-Marqui (1995)
        test_lambdas = np.linspace(test_range[0], test_range[1], n_tests)
        quality_scores = []
        
        for lamb in test_lambdas:
            try:
                smoothed_segmentation = self.segmentation_smooth(
                    eeg_data, 
                    self.microstate_maps, 
                    len(self.microstate_maps),
                    initial_segmentation=initial_segmentation,
                    epsilon=1e-6, 
                    half_window_size=half_window_size,
                    lamb=lamb
                )
                
                quality = self._compute_quality(eeg_data, smoothed_segmentation)
                quality_scores.append(quality)
                
            except Exception:
                quality_scores.append(0.0)
        
        if any(score > 0 for score in quality_scores):
            best_idx = np.argmax(quality_scores)
            return round(test_lambdas[best_idx], 1)
        else:
            return 10.0
    
    def _compute_quality(self, eeg_data, segmentation):
        """Compute quality metric combining correlation, stability, and coverage.

        Uses the same Pearson spatial correlation (center + L2-normalize) as
        compute_correlation_matrix and backfitting for consistency.

        Args:
            eeg_data (np.ndarray): EEG data (n_channels, n_timepoints).
            segmentation (np.ndarray): Segmentation labels.

        Returns:
            float: Quality score (0-1, higher is better).
        """
        valid_indices = segmentation != -1
        if not np.any(valid_indices):
            return 0.0

        valid_data = eeg_data[:, valid_indices]
        valid_labels = segmentation[valid_indices]
        n_states = self.microstate_maps.shape[0]

        # Same correlation definition as compute_correlation_matrix (center + L2 normalize)
        maps_centered = self.microstate_maps - np.mean(self.microstate_maps, axis=1, keepdims=True)
        maps_norm = np.linalg.norm(maps_centered, axis=1, keepdims=True)
        maps_normalized = maps_centered / (maps_norm + 1e-10)
        data_centered = valid_data - np.mean(valid_data, axis=0, keepdims=True)
        data_norm = np.linalg.norm(data_centered, axis=0, keepdims=True)
        data_normalized = data_centered / (data_norm + 1e-10)
        correlation_matrix = np.dot(maps_normalized, data_normalized)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-label mean absolute correlation (vectorized)
        template_correlations = []
        for label in range(n_states):
            label_mask = valid_labels == label
            if np.any(label_mask):
                corrs = np.abs(correlation_matrix[label, label_mask])
                template_correlations.append(np.mean(corrs))
        template_quality = np.mean(template_correlations) if template_correlations else 0.0

        # Temporal stability (fewer transitions = more stable)
        n_transitions = np.sum(np.diff(valid_labels) != 0)
        stability_score = 1.0 / (1.0 + n_transitions / len(valid_labels))

        # Data coverage (proportion of data retained)
        coverage_score = len(valid_labels) / len(segmentation)

        # Weighted combination
        quality_score = (0.6 * template_quality +
                        0.3 * stability_score +
                        0.1 * coverage_score)

        return quality_score

    @staticmethod
    def _filter_short_segments(segmentation, min_samples):
        """Mark segments shorter than threshold as rejected."""
        if len(segmentation) == 0:
            return segmentation
            
        filtered_seg = segmentation.copy()
        current_label = filtered_seg[0]
        start_idx = 0
        
        for i in range(1, len(filtered_seg)):
            if filtered_seg[i] != current_label:
                if i - start_idx < min_samples:
                    filtered_seg[start_idx:i] = -1
                start_idx = i
                current_label = filtered_seg[i]
        
        if len(filtered_seg) - start_idx < min_samples:
            filtered_seg[start_idx:] = -1
        
        return filtered_seg

    def backfit2all(self, data, filter_segments_less_than):
        """Assign microstate labels to all timepoints via template matching.

        Uses Pearson spatial correlation to assign each timepoint to the best-matching
        template. Optionally applies quality control via correlation thresholding to
        reject poorly-fitting timepoints (if min_correlation_threshold is set).

        Args:
            data (ndarray): EEG data (n_channels, n_timepoints).
            filter_segments_less_than (int): Minimum segment duration in samples.

        Returns:
            ndarray: Segmentation with microstate labels (or -1 for rejected points).
        """
        correlation_matrix = self.compute_correlation_matrix(data)
        segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)
        
        # Quality control: reject timepoints below correlation threshold (if enabled)
        segmentation = self.reject_low_correlation_labels(
            correlation_matrix, segmentation, threshold=self.min_correlation_threshold
        )
        
        # Apply smoothing/filtering if enabled
        if self.filter_segments:
            segmentation = self.substitute_maps_with_duration(
                segmentation=segmentation,
                segments_less_than=filter_segments_less_than,
                option=self.filter_segments_option,
                data=data,
                microstate_maps=self.microstate_maps,
                n_states=self.microstate_maps.shape[0],
                smoothing_parameters=[
                    self.smoothing_parameters[0],  # epsilon
                    self.smoothing_parameters[1],  # half_window_size (b)
                    self.smoothing_parameters[2],  # lambda
                ],
            )
        return segmentation

    def backfit2peaks(self, data):
        """Assign microstate labels only at GFP peaks, interpolating between peaks.

        Args:
            data (ndarray): EEG data (n_channels, n_timepoints).

        Returns:
            ndarray: Segmentation with labels interpolated across peak intervals.
                Length always equals data.shape[1].
        """
        n_timepoints = data.shape[1]
        gfp = np.std(data, axis=0)
        peaks, _ = find_peaks(gfp)

        if len(peaks) == 0:
            return self.backfit2all(data, 0)

        # Find troughs between peaks for segment boundaries
        troughs = [0]
        for p in range(len(peaks) - 1):
            min_arg = np.argmin(gfp[peaks[p] : peaks[p + 1]])
            troughs.append(peaks[p] + min_arg)
        troughs.append(len(gfp))
        diff_troughs = np.diff(troughs).astype(int)

        # Assign labels at peaks using Pearson correlation
        data_peaks = data[:, peaks]
        maps_centered = self.microstate_maps - np.mean(self.microstate_maps, axis=1, keepdims=True)
        maps_norm = np.linalg.norm(maps_centered, axis=1, keepdims=True)
        maps_normalized = maps_centered / (maps_norm + 1e-10)

        data_centered = data_peaks - np.mean(data_peaks, axis=0, keepdims=True)
        data_norm = np.linalg.norm(data_centered, axis=0, keepdims=True)
        data_normalized = data_centered / (data_norm + 1e-10)

        activation = np.dot(maps_normalized, data_normalized)
        segmentation_peaks = np.argmax(np.abs(activation), axis=0).astype(int)

        segmentation = np.repeat(segmentation_peaks, diff_troughs)
        if len(segmentation) > n_timepoints:
            segmentation = segmentation[:n_timepoints]
        elif len(segmentation) < n_timepoints:
            pad_len = n_timepoints - len(segmentation)
            segmentation = np.concatenate([segmentation, np.full(pad_len, segmentation[-1], dtype=int)])
        return segmentation

    def perform_segmentation(self, eeg, filter_segments_less_than):
        """Perform microstate segmentation on EEG data.

        Args:
            eeg (mne.io.Raw | mne.Epochs): MNE object with EEG data.
            filter_segments_less_than (int): Minimum segment duration in samples.

        Returns:
            tuple: (labeled_segmentation_array, segmentation_fit).
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