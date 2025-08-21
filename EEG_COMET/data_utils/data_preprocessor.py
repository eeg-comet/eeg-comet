"""Data preprocessing utilities for EEG-COMET.

This module implements the EEG preprocessing pipeline (temporal filtering,
downsampling, spatial smoothing, re-referencing) and, when requested, performs
event-based data selection BEFORE preprocessing so that only the selected data
are filtered/resampled/re-referenced. Event selection supports:

- Raw data: concatenates continuous time windows from each occurrence of the
  selected event label onset to the next event onset (the last window extends
  to the end). The concatenated result is materialized as a new Raw instance.
- Epoched data: keeps only epochs whose event label matches the selected label.

By selecting first and preprocessing afterwards, processing parameters are
consistent across the retained windows/epochs and temporal ordering is preserved.
"""

import warnings
import mne

import numpy as np
from mne import pick_info, pick_types, use_log_level
from mne.epochs import EpochsArray
from mne.io import RawArray
from pyprep.find_noisy_channels import NoisyChannels
from scipy.spatial.distance import pdist, squareform


class DataPreprocessor:
    """Provides comprehensive methods for preprocessing EEG data.

    Responsibilities:
    - Temporal filtering (FIR/IIR)
    - Downsampling
    - Spatial smoothing (neighbor averaging)
    - Average re-referencing and projection application
    - Optional automatic bad channel identification (pyprep)
    - Pre-preprocessing event-based selection (raw/epoched)

    Event-based selection occurs after all preprocessing steps to ensure that
    the concatenated segments/epochs share identical preprocessing parameters.
    For raw data, selection concatenates full label blocks; for epoched data,
    selection reduces to the chosen event label.
    """

    def __init__(self):
        """Initialize the DataPreprocessor class."""
        pass

    @staticmethod
    def identify_bad_channels(eeg, verbose="ERROR"):
        """Identify and mark bad (noisy) channels in the EEG data.

        Uses the pyprep NoisyChannels algorithm to automatically detect channels
        that have poor signal quality or excessive noise.

        Args:
            eeg (Raw or Epochs): MNE Raw or Epochs object containing EEG data
            verbose (str): Logging verbosity level ('ERROR', 'WARNING', 'INFO', etc.)

        Returns:
            Raw or Epochs: EEG data with identified bad channels marked in info['bads']
        """
        with use_log_level(verbose):
            warnings.filterwarnings("ignore")
            try:
                # Restrict to EEG channels that have finite 3D positions (required by RANSAC)
                eeg_picks = pick_types(info=eeg.info, meg=False, eeg=True, exclude=[])
                valid_picks = []
                for pick_idx in eeg_picks:
                    ch = eeg.info["chs"][pick_idx]
                    loc = ch.get("loc", None)
                    if loc is None:
                        continue
                    xyz = loc[:3]
                    if xyz is None:
                        continue
                    if np.all(np.isfinite(xyz)):
                        valid_picks.append(pick_idx)

                # If too few valid channels, skip automatic detection gracefully
                # RANSAC needs multiple channels with valid positions
                if len(valid_picks) < 3:
                    return eeg

                valid_names = [eeg.info["ch_names"][i] for i in valid_picks]

                # Run PyPREP on a copy restricted to valid channels only
                eeg_valid = eeg.copy().pick(valid_picks)
                nd = NoisyChannels(eeg_valid, random_state=1337).find_all_bads()
                if nd:
                    bad_channels_subset = nd.get_bads()
                    # Map back to original channel list (names are preserved)
                    merged_bads = set(eeg.info.get("bads", [])) | set(bad_channels_subset)
                    eeg.info["bads"] = sorted(merged_bads)
            except Exception:
                # If anything goes wrong, leave EEG unchanged rather than failing the pipeline
                return eeg
        return eeg

    @staticmethod
    def spatial_smooth_eeg(eeg, min_neighbors=3, max_neighbors=8, verbose="ERROR"):
        """Apply spatial smoothing to EEG data by averaging signals with neighboring electrodes.

        For each electrode, finds neighbors based on:
        1. Natural "jumps" in the distance distribution
        2. Ensuring at least min_neighbors and at most max_neighbors per channel
        3. Adaptive selection based on electrode distances

        Args:
            eeg (Raw or Epochs): MNE Raw or Epochs object containing EEG data
            min_neighbors (int): Minimum number of neighbors each electrode should have
            max_neighbors (int): Maximum number of neighbors each electrode should have
            verbose (str): Logging verbosity level ('ERROR', 'WARNING', 'INFO', etc.)

        Returns:
            Raw or Epochs: New MNE Raw or Epochs object with spatially smoothed data
        """
        picks_eeg = pick_types(info=eeg.info, meg=False, eeg=True, exclude=[])
        pos = np.array([eeg.info["chs"][i]["loc"][:3] for i in picks_eeg])
        distances = squareform(pdist(pos))
        n_channels = distances.shape[0]
        neighbors = {}
        min_neighbors = max(1, min(min_neighbors, n_channels - 1))
        max_neighbors = max(min_neighbors, min(max_neighbors, n_channels - 1))
        for i in range(n_channels):
            dist_to_others = np.copy(distances[i, :])
            dist_to_others[i] = np.inf
            sorted_indices = np.argsort(dist_to_others)
            sorted_distances = dist_to_others[sorted_indices]
            distance_diffs = np.diff(sorted_distances[: max_neighbors + 1])
            if len(distance_diffs) > 1 and np.max(distance_diffs) > 0:
                norm_diffs = distance_diffs / np.mean(distance_diffs[:3])
                jump_indices = np.where(norm_diffs > 2.0)[0]
                if len(jump_indices) > 0 and jump_indices[0] >= min_neighbors:
                    n_neighbors = jump_indices[0] + 1
                    n_neighbors = min(n_neighbors, max_neighbors)
                else:
                    n_neighbors = max_neighbors
            else:
                n_neighbors = min(min_neighbors + 2, max_neighbors)
            n_neighbors = max(n_neighbors, min_neighbors)
            neighbors[i] = sorted_indices[:n_neighbors].tolist()

        is_epochs = hasattr(eeg, "events")
        if is_epochs:
            eeg_data = eeg.get_data(picks=picks_eeg)
            eeg_data = np.transpose(eeg_data, (1, 2, 0))
            n_channels, n_times, n_epochs = eeg_data.shape
            smoothed_data = np.zeros_like(eeg_data)
            for i in range(n_channels):
                neighborhood = neighbors[i] + [i]
                smoothed_data[i] = np.mean(eeg_data[neighborhood, :, :], axis=0)
            smoothed_data = np.transpose(smoothed_data, (2, 0, 1))

            info_eeg = pick_info(info=eeg.info, sel=picks_eeg)
            smoothed_eeg = EpochsArray(
                data=smoothed_data,
                info=info_eeg,
                events=eeg.events,
                event_id=eeg.event_id,
                tmin=eeg.tmin,
                verbose=verbose,
            )
        else:
            eeg_data = eeg.get_data(picks=picks_eeg)
            n_channels, n_times = eeg_data.shape
            smoothed_data = np.zeros_like(eeg_data)
            for i in range(n_channels):
                neighborhood = neighbors[i] + [i]
                smoothed_data[i] = np.mean(eeg_data[neighborhood, :], axis=0)
            info_eeg = pick_info(info=eeg.info, sel=picks_eeg)
            smoothed_eeg = RawArray(data=smoothed_data, info=info_eeg, verbose=verbose)

        return smoothed_eeg

    def preprocess_eeg(
        self,
        eeg,
        filter_bool,
        filtermethod,
        lowcut,
        highcut,
        downsample_bool,
        sampling_rate,
        spatial_smooth_bool,
        select_events_only=False,
        selected_event_label=None,
        datatype="raw",
        verbose="ERROR",
    ):
        """Preprocess EEG data and optionally select event-specific segments.

        Pipeline (in order):
        0) Event-based selection (if requested)
        1) Temporal filtering (if enabled)
        2) Resampling (if enabled and current sfreq != target)
        3) Spatial smoothing (if enabled)
        4) Average reference and projection apply

        Event-based selection behavior:
        - Raw: concatenates continuous time windows for ``selected_event_label``.
          Windows are defined from each selected event onset to the next event
          onset (last window to end), preserving strict temporal order.
        - Epoched: returns only epochs matching ``selected_event_label`` if present.

        Args:
            eeg (Raw | Epochs): Input MNE object.
            filter_bool (bool): Apply temporal filtering.
            filtermethod (str): Filtering method ('fir', 'iir').
            lowcut (float): Lower cutoff frequency (Hz).
            highcut (float): Upper cutoff frequency (Hz).
            downsample_bool (bool): Apply resampling.
            sampling_rate (int): Target sampling rate (Hz).
            spatial_smooth_bool (bool): Apply spatial smoothing.
            select_events_only (bool): Whether to perform event-based selection.
            selected_event_label (str | None): Label to select (e.g., 'Eyes Closed').
            datatype (str): 'raw' or 'epoched'.
            verbose (str): MNE verbosity level.

        Returns:
            Raw | Epochs: Preprocessed (and possibly event-selected) EEG.
        """
        # 0) Event-based selection FIRST (before any filtering/resampling/reference)
        if select_events_only and selected_event_label:
            label = selected_event_label
            if datatype == "raw":
                # Build windows from selected event onsets to next event onset and create a new Raw
                try:
                    events, event_id = mne.events_from_annotations(eeg)
                except Exception:
                    events, event_id = None, {}
                if events is not None and event_id and label in event_id:
                    # Sort by onset to strictly preserve temporal order
                    order_idx = np.argsort(events[:, 0], kind="stable")
                    events = events[order_idx]

                    sfreq = float(eeg.info["sfreq"])  # Hz
                    code_sel = int(event_id[label])
                    total_samples = int(eeg.n_times)

                    # Collect sample windows in order
                    sample_windows = []  # list of (start_samp, end_samp)
                    for i in range(len(events)):
                        if int(events[i, 2]) == code_sel:
                            start_samp = int(events[i, 0])
                            end_samp = int(events[i + 1, 0]) if i + 1 < len(events) else total_samples
                            if end_samp > start_samp:
                                sample_windows.append((start_samp, end_samp))

                    if sample_windows:
                        # Extract numpy segments then concatenate along time
                        data_segments = []
                        seg_lengths = []
                        for start_samp, end_samp in sample_windows:
                            try:
                                seg_data = eeg.get_data(start=start_samp, stop=end_samp)  # (n_chan, n_time)
                                if seg_data.size > 0:
                                    data_segments.append(seg_data)
                                    seg_lengths.append(seg_data.shape[1])
                            except Exception:
                                continue

                        if data_segments:
                            concat_data = np.concatenate(data_segments, axis=1)
                            # Create new Raw from concatenated data
                            info_copy = eeg.info.copy()
                            try:
                                new_raw = mne.io.RawArray(concat_data, info_copy, verbose=verbose)
                            except Exception:
                                # Fallback: create a minimal info
                                ch_names = eeg.ch_names
                                ch_types = ["eeg"] * len(ch_names)
                                minfo = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                                new_raw = mne.io.RawArray(concat_data, minfo, verbose=verbose)

                            eeg = new_raw
            else:
                # Epoched selection: keep only epochs of the chosen event before preprocessing
                if hasattr(eeg, "event_id") and isinstance(eeg.event_id, dict) and selected_event_label in eeg.event_id:
                    eeg = eeg[selected_event_label]

        # 1) Temporal filtering
        if filter_bool:
            eeg = eeg.filter(
                l_freq=lowcut, h_freq=highcut, method=filtermethod, phase="zero", verbose=verbose
            )
        # 2) Resampling
        if downsample_bool:
            sfreq = eeg.info["sfreq"]
            if sfreq != sampling_rate:
                eeg = eeg.resample(sampling_rate, verbose=verbose)
        # 3) Spatial smoothing
        if spatial_smooth_bool:
            eeg = self.spatial_smooth_eeg(eeg=eeg, verbose=verbose)
        # 4) Average reference
        eeg.set_eeg_reference("average", projection=True, verbose=verbose)
        eeg.apply_proj(verbose=verbose)

        return eeg
