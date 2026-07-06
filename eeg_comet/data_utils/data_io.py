"""Data I/O helpers for EEG files, montages, and related utilities."""

import collections
import os.path
import warnings
from fnmatch import fnmatch

import mne
import numpy as np
from scipy.io import loadmat
from eeg_comet.data_utils.data_preprocessor import DataPreprocessor


class DataIO:
    """Data I/O utilities for EEG files, montages, and helper transforms."""

    def __init__(self):
        """Initialize the DataIO class."""
        pass

    @staticmethod
    def _umeyama_similarity_transform(source_points: np.ndarray, target_points: np.ndarray, allow_reflection: bool = True):
        """Compute similarity transform (scale, rotation, translation) using Umeyama.

        Args:
            source_points: Nx3 array of source coordinates
            target_points: Nx3 array of target coordinates
            allow_reflection: If True, allow a reflection in rotation

        Returns:
            scale (float), rotation (3x3 np.ndarray), translation (3x1 np.ndarray)
        """
        if source_points.shape != target_points.shape:
            raise ValueError("Source and target must have the same shape")
        if source_points.ndim != 2 or source_points.shape[1] != 3:
            raise ValueError("Points must be of shape (N, 3)")

        num_points = source_points.shape[0]
        if num_points < 3:
            raise ValueError("At least 3 points are required for a stable fit")

        mu_source = source_points.mean(axis=0)
        mu_target = target_points.mean(axis=0)
        X = source_points - mu_source
        Y = target_points - mu_target

        cov = (Y.T @ X) / num_points
        U, S, Vt = np.linalg.svd(cov)
        R = U @ Vt
        if not allow_reflection and np.linalg.det(R) < 0:
            # Enforce proper rotation
            Vt[-1, :] *= -1
            R = U @ Vt
        elif allow_reflection and np.linalg.det(R) < 0:
            # Allow reflection: flip the sign of the last singular value
            Vt[-1, :] *= -1
            S[-1] *= -1
            R = U @ Vt

        var_X = (X**2).sum() / num_points
        if var_X <= 0:
            scale = 1.0
        else:
            scale = S.sum() / var_X

        t = mu_target - scale * (R @ mu_source)
        return scale, R, t

    @staticmethod
    def _fit_ch_pos_to_template(ch_pos: dict, template_names: list[str] | None = None, allow_reflection: bool = True) -> dict:
        """Fit arbitrary channel coordinates to a standard head shape via similarity transform.

        Matches channels to a standard montage by name, computes a similarity transform
        (scale, rotation, translation) to best-align the input to the template, and
        applies it to all input channels.

        Args:
            ch_pos: Mapping of channel name -> np.ndarray([x, y, z]) in arbitrary units
            template_names: Ordered list of template montage names to attempt
            allow_reflection: Whether to allow reflections during alignment

        Returns:
            dict: Fitted channel positions in meters (head coordinate frame)
        """
        if template_names is None:
            template_names = ["standard_1020", "standard_1005"]

        # Build case-insensitive name mapping for input
        input_names_lower_to_orig = {name.lower(): name for name in ch_pos.keys()}

        matched = False
        fitted = None
        for tmpl in template_names:
            try:
                template = mne.channels.make_standard_montage(tmpl)
                template_positions = template.get_positions()
                template_ch_pos = template_positions.get("ch_pos", {})
                # Build arrays of matched points
                common_lower = [
                    nm for nm in input_names_lower_to_orig.keys() if nm in {k.lower(): v for k, v in template_ch_pos.items()}.keys()
                ]
                if len(common_lower) < 3:
                    continue
                # Prepare matched arrays
                source_pts = []
                target_pts = []
                for nm_lower in common_lower:
                    src_name = input_names_lower_to_orig[nm_lower]
                    src_pt = np.asarray(ch_pos[src_name], dtype=float)
                    # Map nm_lower back to actual template key (case-insensitive)
                    # Find first template key with same lowercase
                    for tmpl_key, tmpl_val in template_ch_pos.items():
                        if tmpl_key.lower() == nm_lower:
                            tgt_pt = np.asarray(tmpl_val, dtype=float)
                            break
                    else:
                        continue
                    source_pts.append(src_pt)
                    target_pts.append(tgt_pt)

                source_pts = np.asarray(source_pts)
                target_pts = np.asarray(target_pts)
                if source_pts.shape[0] < 3:
                    continue

                scale, R, t = DataIO._umeyama_similarity_transform(source_pts, target_pts, allow_reflection=allow_reflection)
                # Apply to all points
                fitted = {}
                for name, pt in ch_pos.items():
                    pt = np.asarray(pt, dtype=float)
                    new_pt = scale * (R @ pt) + t
                    fitted[name] = new_pt
                matched = True
                break
            except Exception:
                continue

        if matched and fitted is not None:
            return fitted
        # Fallback: return original positions unchanged
        return {k: np.asarray(v, dtype=float) for k, v in ch_pos.items()}
    @staticmethod
    def find_data(input_folder, extension, pattern="*", exclude_derivatives=False):
        """Recursively search for data files within a folder based on extension and pattern.

        Args:
            input_folder (str): The folder to search for data files.
            extension (str): The file extension to match. If 'auto', load all eeg files with valid formats.
            pattern (str, optional): The pattern to match against the file name. Defaults to '*'.
            exclude_derivatives (bool, optional): If True, exclude any folders or paths containing 
                'derivatives' (case-insensitive) from search. Useful for BIDS datasets when only 
                raw data is desired. Defaults to False.

        Returns:
            tuple: A tuple containing the list of matching file paths and the list of matching file names.
        """
        list_path = []
        list_filename = []
        if extension == ".auto":
            valid_eeg_formats = [
                ".vhdr",
                ".edf",
                ".bdf",
                ".gdf",
                ".cnt",
                ".egi",
                ".mff",
                ".set",
                ".data",
                ".nxe",
                ".lay",
            ]
            for path, subdirs, files in os.walk(input_folder):
                # Skip derivatives folder if exclude_derivatives is True
                if exclude_derivatives:
                    # Skip if current path contains derivatives
                    if 'derivatives' in path.lower():
                        continue
                    # Remove any derivatives folders from subdirectories to explore
                    subdirs[:] = [d for d in subdirs if 'derivatives' not in d.lower()]
                
                for name in files:
                    if os.path.splitext(name)[1] in valid_eeg_formats and fnmatch(name, pattern):
                            list_path.append(os.path.join(path, name))
                            list_filename.append(os.path.splitext(name)[0])
        else:
            for path, subdirs, files in os.walk(input_folder):
                # Skip derivatives folder if exclude_derivatives is True
                if exclude_derivatives:
                    # Skip if current path contains derivatives
                    if 'derivatives' in path.lower():
                        continue
                    # Remove any derivatives folders from subdirectories to explore
                    subdirs[:] = [d for d in subdirs if 'derivatives' not in d.lower()]
                    
                for name in files:
                    if fnmatch(name, pattern + extension):
                        list_path.append(os.path.join(path, name))
                        list_filename.append(name.split(".")[0])
        return list_path, list_filename

    @staticmethod
    def _read_mat_locations(file_path):
        """Read channel locations from a Brainstorm .mat file.

        Extracts 3D electrode positions from Brainstorm's channel structure,
        converting to the MNE coordinate system (y, x, z) in meters.

        Args:
            file_path (str): Path to the Brainstorm .mat file

        Returns:
            dict: Dictionary mapping channel names to 3D positions in meters
                Format: {'ChannelName': [y, x, z], ...}

        Raises:
            ValueError: If the file does not contain the expected 'Channel' key
            IOError: If the file cannot be read
        """
        mat = loadmat(file_path)
        if "Channel" not in mat:
            raise ValueError('MAT file does not contain "Channel" key.')
        channel_data = mat["Channel"][0]
        ch_pos = {}
        for ch in channel_data:
            name = ch["Name"][0]
            loc = ch["Loc"].flatten() if ch["Loc"].shape == (3, 1) else ch["Loc"]
            if abs(loc[0]) > 0.5 or abs(loc[1]) > 0.5 or abs(loc[2]) > 0.5:
                loc[0] = loc[0] / 1000.0
                loc[1] = loc[1] / 1000.0
                loc[2] = loc[2] / 1000.0
            ch_pos[name] = [loc[1], loc[0], loc[2]]
        return ch_pos

    @staticmethod
    def _read_ced_locations(file_path):
        """Read channel locations from an EEGLAB .ced file.

        Parses a .ced file to extract electrode positions, converting from EEGLAB's
        coordinate convention to MNE's RAS (Right-Anterior-Superior) coordinate system.
        Coordinates are directly transformed and scaled to meters (auto-detecting normalized/mm/cm/m units).
        
        EEGLAB CED format: X=anterior-posterior (+front), Y=left-right (+LEFT), Z=inferior-superior
        MNE RAS system: X=left-right (+RIGHT), Y=anterior-posterior (+front), Z=inferior-superior
        Transformation: MNE_X = -CED_Y, MNE_Y = CED_X, MNE_Z = CED_Z
        
        CED files typically contain pre-standardized electrode positions, so no template
        fitting is applied - only coordinate system transformation and unit scaling.

        Args:
            file_path (str): Path to the .ced file

        Returns:
            dict: Dictionary mapping channel names to 3D positions in meters
                Format: {'ChannelName': np.array([x, y, z]), ...} in MNE RAS head coordinates

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file format is invalid or missing required columns
        """
        ch_pos = {}
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path) as f:
            lines = f.readlines()
        if not lines:
            raise ValueError("The .ced file is empty.")
        header_line = lines[0].strip()
        if not header_line:
            raise ValueError("The .ced file does not contain a header line.")
        
        # Parse header - keep empty columns to maintain alignment
        if "\t" in header_line:
            header_parts = header_line.split("\t")
        else:
            header_parts = header_line.split()
        
        # Build column map with original indices (including empty columns)
        col_map = {col.strip().lower(): idx for idx, col in enumerate(header_parts) if col.strip()}
        
        required_columns = ["labels", "x", "y", "z"]
        missing_cols = [col for col in required_columns if col not in col_map]
        if missing_cols:
            raise ValueError(f"Missing required columns in header: {missing_cols}")
        
        label_idx = col_map["labels"]
        x_idx = col_map["x"]
        y_idx = col_map["y"]
        z_idx = col_map["z"]
        
        # Parse data rows - keep empty columns for alignment
        labels = []
        coords = []  # list of (x, y, z) in original units
        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            
            # Split but DON'T filter empty strings - we need to maintain column alignment
            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split()
            
            if len(parts) <= max(label_idx, x_idx, y_idx, z_idx):
                raise ValueError(f"Invalid line in CED file at line {line_num}: {line}")
            
            label = parts[label_idx].strip()
            try:
                x = float(parts[x_idx].strip())
                y = float(parts[y_idx].strip())
                z = float(parts[z_idx].strip())
            except (ValueError, AttributeError) as err:
                raise ValueError(
                    f"Invalid numerical values in line {line_num}: {line}"
                ) from err
            labels.append(label)
            coords.append((x, y, z))

        if not coords:
            return ch_pos

        # Build positions dict with coordinate transformation to match MNE's RAS system
        # EEGLAB CED: X=front, Y=left, Z=up (positive Y is LEFT side)
        # MNE RAS: X=right, Y=front, Z=up (positive X is RIGHT side)
        # Transform: MNE_X = -CED_Y (invert left/right), MNE_Y = CED_X, MNE_Z = CED_Z
        for label, (x, y, z) in zip(labels, coords):
            # Apply coordinate transformation
            mne_x = -y  # CED's left/right becomes MNE's right/left (negated)
            mne_y = x   # CED's front/back becomes MNE's front/back
            mne_z = z   # CED's up/down stays the same
            
            ch_pos[label] = np.array([mne_x, mne_y, mne_z], dtype=float)

        # Determine appropriate scaling to meters
        # CED files typically use normalized coordinates (radius ~1) or mm/cm
        abs_max = max(np.linalg.norm(v) for v in ch_pos.values())
        
        if abs_max > 20.0:
            # Likely in mm (e.g., 85mm)
            scale = 1.0 / 1000.0
        elif abs_max > 1.5:
            # Likely in cm (e.g., 8.5cm)
            scale = 1.0 / 100.0
        elif 0.5 <= abs_max <= 1.5:
            # Likely normalized to unit sphere (radius ~1), scale to typical head size
            # Standard head radius is approximately 8.5-9.5cm = 0.085-0.095m
            scale = 0.095
        elif 0.08 <= abs_max < 0.5:
            # Already in meters (typical head radius 0.085-0.095m)
            scale = 1.0
        else:
            # Very small values, assume normalized and scale
            scale = 0.095

        # Apply scaling to all positions
        for k in ch_pos:
            ch_pos[k] = ch_pos[k] * scale

        return ch_pos

    def load_montage(self, montage):
        """Load electrode positions from a file or built-in montage name.

        Supports multiple file formats (.mat, .ced, standard MNE formats) or
        built-in standard montages from MNE.

        Args:
            montage (str): Path to channel location file or name of
                standard montage (e.g., 'standard_1020'). If empty, defaults to 'standard_1020'

        Returns:
            mne.channels.DigMontage: Montage object containing electrode positions

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the montage cannot be created from the provided input
        """
        if not montage:
            montage = "standard_1020"
        try:
            if os.path.isfile(montage):
                file_ext = os.path.splitext(montage)[1].lower()
                if file_ext == ".mat":
                    ch_pos = self._read_mat_locations(montage)
                    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
                elif file_ext == ".ced":
                    ch_pos = self._read_ced_locations(montage)
                    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
                else:
                    montage = mne.channels.read_custom_montage(montage)
            elif montage in mne.channels.get_builtin_montages():
                montage = mne.channels.make_standard_montage(montage)
            else:
                raise ValueError(f"Unknown montage: {montage}")
        except FileNotFoundError as err:
            raise FileNotFoundError(f"File not found: {montage}") from err
        except ValueError as ve:
            raise ve
        except Exception as err:
            raise ValueError(f"An error occurred while loading the montage: {err}") from err
        if montage is None:
            raise ValueError(
                f"Could not create a montage from the provided directory or name: {montage}"
            )
        return montage

    def check_channels_to_remove(self, list_eegs_path, data_type, montage):
        """Identify consistent and missing channels across a dataset collection.

        Checks for channel consistency across multiple EEG files and identifies
        channels that are consistent across all files and those missing in some files.
        Channel names are compared case-insensitively (e.g., 'Fpz', 'FPz', and 'fpz' are treated as the same).

        Args:
            list_eegs_path (list of str): Paths to EEG files to check
            data_type (str): Type of EEG data ('raw' or 'epoched')
            montage (str): Path to channel locations or name of standard montage

        Returns:
            tuple: (consistent_channels, missing_channels)
                - consistent_channels (list of str): Names of channels present in all files
                - missing_channels (list of str): Names of channels missing in at least one file
        """
        # Dictionary to map canonical lowercase names to original name formats
        channel_name_map = {}

        # Store case-normalized channels from each file
        all_file_channels = []

        # Get channel names from each file and normalize cases
        for filename in list_eegs_path:
            eeg = self.load_eeg(filename, data_type, montage, preload=False)

            # Create normalized version of channel names for this file
            normalized_channels = []
            for chan in eeg.info["ch_names"]:
                chan_lower = chan.lower()

                # Store the first encountered version of each channel as canonical
                if chan_lower not in channel_name_map:
                    channel_name_map[chan_lower] = chan

                normalized_channels.append(chan_lower)

            all_file_channels.append(normalized_channels)

        # Count occurrences of each normalized channel
        counter = collections.Counter()
        for channels in all_file_channels:
            counter.update(channels)

        total_files = len(list_eegs_path)

        # Get normalized channel names with their counts
        consistent_norm_channels = [chan for chan, count in counter.items() if count == total_files]
        missing_norm_channels = [chan for chan, count in counter.items() if count < total_files]

        # Convert back to original casing using the mapping
        consistent_channels = [channel_name_map[chan] for chan in consistent_norm_channels]
        missing_channels = [channel_name_map[chan] for chan in missing_norm_channels]

        return consistent_channels, missing_channels

    def load_eeg(
        self, eeg_path, data_type, montage="", channels_to_remove=None, preload=True, verbose="CRITICAL"
    ):
        """Load EEG data from various file formats with optional preprocessing.

        Loads raw or epoched EEG data, applies channel locations, removes specified
        channels, and sets up average reference.

        Args:
            eeg_path (str): Path to the EEG data file
            data_type (str): Type of EEG data to load ('raw' or 'epoched')
            montage (str, optional): Path to channel locations or name
                of standard montage. Defaults to '' (uses 'standard_1020')
            channels_to_remove (list of str, optional): List of channel names to remove.
                Defaults to None
            preload (bool, optional): Whether to load data into memory immediately.
                Defaults to True
            verbose (str, optional): MNE verbosity level ('CRITICAL', 'ERROR', etc.).
                Defaults to 'CRITICAL'

        Returns:
            mne.io.Raw or mne.Epochs: Loaded EEG data object

        Raises:
            FileNotFoundError: If the EEG file does not exist
            ValueError: If the data_type is not supported or montage cannot be loaded
        """
        with mne.use_log_level(verbose):
            warnings.filterwarnings("ignore")
            if data_type == "raw":
                try:
                    eeg = mne.io.read_raw(eeg_path, preload=preload, verbose=verbose)
                except TypeError as e:
                    # Check if error is about trying to read epoched data as raw
                    if "trials" in str(e) and "raw files" in str(e):
                        raise ValueError(
                            "Data format mismatch: The selected file contains epoched data, but you chose 'Import Raw Data'. "
                            "Please select 'Import Epoched Data' instead, or choose a raw EEG file."
                        ) from e
                    else:
                        # Re-raise other TypeErrors as-is
                        raise
            elif data_type == "epoched":
                eeg = mne.io.read_epochs_eeglab(eeg_path, verbose=verbose)
                # eeg = mne.io.read_epochs(eeg_path, verbose=False)
            
            # Remove auxiliary channels immediately after loading, before any other processing
            eeg, removed_aux_channels = DataPreprocessor.remove_auxiliary_channels(eeg, verbose=verbose)
            
            # Apply montage
            montage = self.load_montage(montage)
            eeg.set_montage(montage, match_case=False, on_missing="warn")
            ch_names = eeg.info["ch_names"]
            if channels_to_remove is None:
                channels_to_remove = []
            if any(channels_to_remove) and any(elem != "" for elem in channels_to_remove) and channels_to_remove in ch_names:
                eeg = eeg.drop_channels(channels_to_remove)
            if "TRIGGER" in ch_names:
                eeg = eeg.drop_channels("TRIGGER")
            eeg.set_eeg_reference("average", projection=True)
            eeg.apply_proj()
        return eeg

    @staticmethod
    def export_eegs(eeg, save_path, extension, data_type):
        """Export EEG data to a specified file format.

        Saves EEG data to common file formats supported by MNE, automatically
        handling the appropriate export method based on data type.

        Args:
            eeg (mne.io.Raw or mne.Epochs): EEG data object to export
            save_path (str): Output path without extension (extension will be added)
            extension (str): Desired file extension ('.vhdr', '.set', '.edf')
                If not one of these, defaults to '.set'
            data_type (str): Type of EEG data ('raw' or 'epoched')

        Notes:
            Automatically overwrites existing files with the same name.
        """
        available_extensions = [".vhdr", ".set", ".edf"]
        if extension not in available_extensions:
            extension = ".set"
        if data_type == "raw":
            mne.export.export_raw(save_path + extension, eeg, fmt="auto", overwrite=True)
        elif data_type == "epoched":
            mne.export.export_epochs(save_path + extension, eeg, fmt="auto", overwrite=True)

    @staticmethod
    def get_eeg_data(eeg, data_type):
        """Extract raw NumPy arrays from MNE Raw or Epochs objects.

        Converts MNE objects to NumPy arrays for further processing or analysis.
        For epoched data, concatenates all epochs along the time dimension.

        Args:
            eeg (mne.io.Raw or mne.Epochs): EEG data object
            data_type (str): Type of EEG data ('raw' or 'epoched')

        Returns:
            numpy.ndarray: EEG data as a NumPy array
                For raw data: shape (n_channels, n_times)
                For epoched data: shape (n_channels, n_times_total)
                where n_times_total = n_epochs * n_times_per_epoch

        Raises:
            ValueError: If no data is available (empty epochs object or invalid data_type)
        """
        eeg_data = None
        if data_type == "epoched":
            for index in range(eeg.__len__()):
                if index == 0:
                    eeg_data = np.squeeze(eeg[0].get_data())
                else:
                    epoch = np.squeeze(eeg[index].get_data())
                    eeg_data = np.append(eeg_data, epoch, axis=1)
        else:
            eeg_data = eeg.get_data()
        if eeg_data is None:
            raise ValueError("No data available: empty epochs object or invalid data_type")
        return eeg_data
