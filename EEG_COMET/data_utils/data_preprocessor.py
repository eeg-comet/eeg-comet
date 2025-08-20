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
from typing import List, Optional, Dict, Set

import numpy as np
from mne import pick_info, pick_types, use_log_level
from mne.epochs import EpochsArray
from mne.io import RawArray
from pyprep.find_noisy_channels import NoisyChannels
from scipy.spatial.distance import pdist, squareform


class DataPreprocessor:
    """Provides comprehensive methods for preprocessing EEG data.

    Responsibilities:
    - Channel consistency checking and interpolation
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
        self._channel_consistency_cache: Dict[str, Set[str]] = {}

    @staticmethod
    def interpolate_missing_channels(
            raw: mne.io.Raw,
            missing_channels: List[str],
            montage_name: str = 'standard_1020',
            custom_montage: Optional[mne.channels.DigMontage] = None,
            montage_file: Optional[str] = None,
            verbose: bool = False
    ) -> mne.io.Raw:
        """
        Interpolate missing EEG channels and add them in the correct positions.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw EEG data object with some channels missing.
        missing_channels : List[str]
            List of channel names that need to be interpolated.
        montage_name : str, optional
            Name of the standard montage to use. Default is 'standard_1020'.
        custom_montage : mne.channels.DigMontage, optional
            Custom montage object with ALL channel positions (not just current channels).
        montage_file : str, optional
            Path to a montage file with ALL channel positions.
        verbose : bool, optional
            If True, print detailed information about the process.

        Returns
        -------
        mne.io.Raw
            Raw object with interpolated channels added in correct positions.
        """

        if verbose:
            print("=" * 60)
            print("STARTING INTERPOLATION PROCESS")
            print("=" * 60)

        # Make a copy to avoid modifying the original data
        raw_copy = raw.copy()

        # Load the FULL montage (with all possible channel positions)
        if custom_montage is not None:
            full_montage = custom_montage
            if verbose:
                print(f"Using custom montage with {len(full_montage.ch_names)} total positions")
        elif montage_file is not None:
            try:
                full_montage = mne.channels.read_custom_montage(montage_file)
                if verbose:
                    print(f"Loaded montage from file: {montage_file}")
                    print(f"Full montage contains {len(full_montage.ch_names)} total positions")
            except Exception as e:
                raise ValueError(f"Failed to load montage from '{montage_file}'. Error: {e}")
        else:
            try:
                full_montage = mne.channels.make_standard_montage(montage_name)
                if verbose:
                    print(f"Using standard montage: {montage_name}")
                    print(f"Full montage contains {len(full_montage.ch_names)} total positions")
            except ValueError as e:
                raise ValueError(f"Invalid montage name '{montage_name}'. Error: {e}")

        # Get channel names
        current_channels = raw_copy.ch_names
        full_montage_channels = full_montage.ch_names

        if verbose:
            print(f"\nCurrent channels ({len(current_channels)}): {current_channels[:10]}...")
            print(f"Full montage has positions for {len(full_montage_channels)} channels")
            print(f"Channels to interpolate: {missing_channels}")

        # Check if channels are actually missing
        already_present = [ch for ch in missing_channels if ch in current_channels]
        if already_present:
            raise ValueError(f"Channels {already_present} are already present in the data")

        # Validate that missing channels exist in the FULL montage
        missing_not_in_montage = []
        for ch in missing_channels:
            if ch not in full_montage_channels:
                # Try case-insensitive match
                found = False
                for mont_ch in full_montage_channels:
                    if ch.lower() == mont_ch.lower():
                        found = True
                        break
                if not found:
                    missing_not_in_montage.append(ch)

        if missing_not_in_montage:
            raise ValueError(f"Channels {missing_not_in_montage} not found in montage. "
                             f"Available channels in full montage: {full_montage_channels[:30]}...")

        # STEP 1: Apply the FULL montage to the raw data
        if verbose:
            print(f"\n" + "=" * 60)
            print("STEP 1: APPLYING FULL MONTAGE TO DATA")
            print("=" * 60)

        raw_copy.set_montage(full_montage, on_missing='ignore', verbose=verbose)

        # STEP 2: Determine channel order from the FULL montage
        # Include only channels that are either current or to be interpolated
        channels_to_include = []
        for ch in full_montage_channels:
            if ch in current_channels or ch in missing_channels:
                channels_to_include.append(ch)

        if verbose:
            print(f"\n" + "=" * 60)
            print("STEP 2: CREATING COMPLETE CHANNEL LAYOUT")
            print("=" * 60)
            print(f"Total channels after interpolation: {len(channels_to_include)}")
            print(f"Using channel order from full montage")

        # Create new info with all channels
        sfreq = raw_copy.info['sfreq']
        info_complete = mne.create_info(
            ch_names=channels_to_include,
            sfreq=sfreq,
            ch_types='eeg'
        )

        # Apply the FULL montage to the complete info
        info_complete.set_montage(full_montage, on_missing='ignore')

        # Create zero-filled data for all channels
        n_samples = raw_copy.n_times
        n_channels_complete = len(channels_to_include)
        data_complete = np.zeros((n_channels_complete, n_samples))

        # Fill in the existing channel data
        for i, ch_name in enumerate(channels_to_include):
            if ch_name in current_channels:
                ch_idx = current_channels.index(ch_name)
                data_complete[i, :] = raw_copy.get_data(picks=[ch_idx])[0]

        # Create a new Raw object with complete channel set
        raw_complete = mne.io.RawArray(data_complete, info_complete, verbose=False)

        # Copy over important metadata
        if raw_copy.annotations is not None:
            raw_complete.set_annotations(raw_copy.annotations)

        # Copy other info fields
        for field in ['description', 'experimenter', 'subject_info', 'device_info']:
            if field in raw_copy.info:
                raw_complete.info[field] = raw_copy.info[field]

        # STEP 3: Mark missing channels as bad and interpolate
        raw_complete.info['bads'] = missing_channels

        if verbose:
            print(f"\n" + "=" * 60)
            print("STEP 3: PERFORMING INTERPOLATION")
            print("=" * 60)
            print(f"Interpolating channels: {missing_channels}")
            print(f"Using full montage with {len(full_montage_channels)} positions")

        # Perform the interpolation
        raw_interpolated = raw_complete.copy().interpolate_bads(
            reset_bads=True,
            verbose=verbose
        )

        # Verify the result
        final_channels = raw_interpolated.ch_names
        if verbose:
            print("\n" + "=" * 60)
            print("INTERPOLATION COMPLETE")
            print("=" * 60)
            print(f"Final number of channels: {len(final_channels)}")

            # Check that all missing channels are now present
            successfully_added = [ch for ch in missing_channels if ch in final_channels]
            failed_to_add = [ch for ch in missing_channels if ch not in final_channels]

            print(f"✓ Successfully interpolated: {successfully_added}")
            if failed_to_add:
                print(f"✗ WARNING: Failed to interpolate: {failed_to_add}")

            # Show channel order
            print(f"\nFirst 15 channels in final data:")
            for i, ch in enumerate(final_channels[:15], 1):
                marker = " 🆕" if ch in successfully_added else ""
                print(f"  {i:2d}. {ch}{marker}")
            if len(final_channels) > 15:
                print(f"  ... ({len(final_channels) - 15} more channels)")

        return raw_interpolated

    def check_channel_consistency(self, eeg_files: List[mne.io.Raw], verbose: bool = False) -> Dict[str, List[str]]:
        """
        Check channel consistency across all EEG files and identify missing channels.

        Parameters
        ----------
        eeg_files : List[mne.io.Raw]
            List of MNE Raw objects to check for channel consistency.
        verbose : bool, optional
            If True, print detailed information about the process.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary with file identifiers as keys and lists of missing channels as values.
        """
        if not eeg_files:
            return {}

        if verbose:
            print("=" * 60)
            print("CHECKING CHANNEL CONSISTENCY ACROSS FILES")
            print("=" * 60)

        # Get all unique channel names across all files
        all_channels = set()
        file_channels = {}
        
        for i, eeg in enumerate(eeg_files):
            file_id = f"file_{i}"
            channels = set(eeg.ch_names)
            file_channels[file_id] = channels
            all_channels.update(channels)
            
            if verbose:
                print(f"{file_id}: {len(channels)} channels")

        if verbose:
            print(f"\nTotal unique channels across all files: {len(all_channels)}")
            print(f"Channel names: {sorted(all_channels)}")

        # Find missing channels for each file
        missing_channels = {}
        for file_id, channels in file_channels.items():
            missing = sorted(all_channels - channels)
            if missing:
                missing_channels[file_id] = missing
                if verbose:
                    print(f"{file_id} missing channels: {missing}")

        return missing_channels

    def ensure_channel_consistency(
        self, 
        eeg_files: List[mne.io.Raw], 
        montage_name: str = 'standard_1020',
        custom_montage: Optional[mne.channels.DigMontage] = None,
        montage_file: Optional[str] = None,
        verbose: bool = False
    ) -> List[mne.io.Raw]:
        """
        Ensure all EEG files have the same channel set by interpolating missing channels.

        Parameters
        ----------
        eeg_files : List[mne.io.Raw]
            List of MNE Raw objects to standardize.
        montage_name : str, optional
            Name of the standard montage to use for interpolation.
        custom_montage : mne.channels.DigMontage, optional
            Custom montage object with ALL channel positions.
        montage_file : str, optional
            Path to a montage file with ALL channel positions.
        verbose : bool, optional
            If True, print detailed information about the process.

        Returns
        -------
        List[mne.io.Raw]
            List of MNE Raw objects with consistent channel sets.
        """
        if not eeg_files:
            return []

        # Check for missing channels
        missing_channels = self.check_channel_consistency(eeg_files, verbose)
        
        if not missing_channels:
            if verbose:
                print("All files already have consistent channel sets.")
            return eeg_files

        # Interpolate missing channels for each file
        standardized_files = []
        for i, eeg in enumerate(eeg_files):
            file_id = f"file_{i}"
            
            if file_id in missing_channels:
                if verbose:
                    print(f"\nInterpolating missing channels for {file_id}...")
                
                try:
                    standardized_eeg = self.interpolate_missing_channels(
                        raw=eeg,
                        missing_channels=missing_channels[file_id],
                        montage_name=montage_name,
                        custom_montage=custom_montage,
                        montage_file=montage_file,
                        verbose=verbose
                    )
                    standardized_files.append(standardized_eeg)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to interpolate channels for {file_id}: {e}")
                    # Keep original file if interpolation fails
                    standardized_files.append(eeg)
            else:
                standardized_files.append(eeg)

        if verbose:
            print(f"\nChannel standardization complete. All files now have {len(standardized_files[0].ch_names)} channels.")

        return standardized_files

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

    def preprocess_multiple_eeg_files(
        self,
        eeg_files: List[mne.io.Raw],
        filter_bool: bool,
        filtermethod: str,
        lowcut: float,
        highcut: float,
        downsample_bool: bool,
        sampling_rate: int,
        spatial_smooth_bool: bool,
        select_events_only: bool = False,
        selected_event_label: Optional[str] = None,
        datatype: str = "raw",
        ensure_channel_consistency: bool = True,
        montage_name: str = 'standard_1020',
        custom_montage: Optional[mne.channels.DigMontage] = None,
        montage_file: Optional[str] = None,
        verbose: str = "ERROR"
    ) -> List[mne.io.Raw]:
        """
        Preprocess multiple EEG files with optional channel consistency checking.

        This method first ensures all files have consistent channel sets by interpolating
        missing channels, then applies the standard preprocessing pipeline to each file.

        Parameters
        ----------
        eeg_files : List[mne.io.Raw]
            List of MNE Raw objects to preprocess.
        filter_bool : bool
            Apply temporal filtering.
        filtermethod : str
            Filtering method ('fir', 'iir').
        lowcut : float
            Lower cutoff frequency (Hz).
        highcut : float
            Upper cutoff frequency (Hz).
        downsample_bool : bool
            Apply resampling.
        sampling_rate : int
            Target sampling rate (Hz).
        spatial_smooth_bool : bool
            Apply spatial smoothing.
        select_events_only : bool, optional
            Whether to perform event-based selection.
        selected_event_label : str | None, optional
            Label to select (e.g., 'Eyes Closed').
        datatype : str, optional
            'raw' or 'epoched'.
        ensure_channel_consistency : bool, optional
            Whether to ensure all files have consistent channel sets before preprocessing.
        montage_name : str, optional
            Name of the standard montage to use for interpolation.
        custom_montage : mne.channels.DigMontage, optional
            Custom montage object with ALL channel positions.
        montage_file : str, optional
            Path to a montage file with ALL channel positions.
        verbose : str, optional
            MNE verbosity level.

        Returns
        -------
        List[mne.io.Raw]
            List of preprocessed MNE Raw objects.
        """
        if not eeg_files:
            return []

        # Step 0: Ensure channel consistency across all files
        if ensure_channel_consistency and len(eeg_files) > 1:
            if verbose != "ERROR":
                print("Ensuring channel consistency across all files...")
            
            eeg_files = self.ensure_channel_consistency(
                eeg_files=eeg_files,
                montage_name=montage_name,
                custom_montage=custom_montage,
                montage_file=montage_file,
                verbose=(verbose != "ERROR")
            )

        # Step 1: Preprocess each file individually
        preprocessed_files = []
        for i, eeg in enumerate(eeg_files):
            if verbose != "ERROR":
                print(f"Preprocessing file {i+1}/{len(eeg_files)}...")
            
            try:
                preprocessed_eeg = self.preprocess_eeg(
                    eeg=eeg,
                    filter_bool=filter_bool,
                    filtermethod=filtermethod,
                    lowcut=lowcut,
                    highcut=highcut,
                    downsample_bool=downsample_bool,
                    sampling_rate=sampling_rate,
                    spatial_smooth_bool=spatial_smooth_bool,
                    select_events_only=select_events_only,
                    selected_event_label=selected_event_label,
                    datatype=datatype,
                    verbose=verbose
                )
                preprocessed_files.append(preprocessed_eeg)
            except Exception as e:
                if verbose != "ERROR":
                    print(f"Warning: Failed to preprocess file {i+1}: {e}")
                # Keep original file if preprocessing fails
                preprocessed_files.append(eeg)

        return preprocessed_files

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
