"""Source I/O convenience functions for EEG-COMET."""

import os

import mne
import numpy as np

from eeg_comet.data_utils.data_io import DataIO


class SourceIO:
    """Handle source-related data input/output operations."""

    def __init__(self):
        """Initialize the SourceIO class."""
        self.data_io = DataIO()

    @staticmethod
    def write_stc(stc_data, output_path, filename_prefix="stc", data_type="raw"):
        """Write source time series to disk.

        Args:
            stc_data: The source time series data (single STC or list of STCs for epoched data)
            output_path: The directory path where the source time series will be saved
            filename_prefix: Prefix for the filename (default: "stc")
            data_type: Type of data - "raw" or "epoched"
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if data_type == "epoched" and isinstance(stc_data, list):
            for idx, stc in enumerate(stc_data):
                filename = f"{filename_prefix}_{idx}"
                filepath = os.path.join(output_path, filename)
                # Suppress verbose output during STC save
                with mne.utils.use_log_level("ERROR"):
                    stc.save(filepath, ftype="h5", overwrite=True)
        else:
            filepath = os.path.join(output_path, f"{filename_prefix}_file")
            # Suppress verbose output during STC save
            with mne.utils.use_log_level("ERROR"):
                stc_data.save(filepath, ftype="h5", overwrite=True)

    def read_stc(self, stc_path, pattern="*"):
        """Read source time series from disk.

        Args:
            stc_path: The directory path where the source time series is stored
            pattern: File pattern to match (default: '*')

        Returns:
            List of loaded source time series data
        """
        stc_list, _ = self.data_io.find_data(stc_path, ".h5", pattern=pattern)
        if not stc_list:
            return []

        return [mne.read_source_estimate(stc_path) for stc_path in stc_list]

    @staticmethod
    def read_single_stc(stc_file_path):
        """Read a single STC file.

        Args:
            stc_file_path: Path to the STC file

        Returns:
            STC object or None if file doesn't exist
        """
        if os.path.exists(stc_file_path):
            return mne.read_source_estimate(stc_file_path)
        return None

    def find_stc_files(self, base_path, subject_name=None):
        """Find STC files for a specific subject or all subjects.

        Args:
            base_path: Base directory containing STC data
            subject_name: Specific subject name (optional)

        Returns:
            Dictionary with subject names as keys and file paths as values
        """
        stc_files = {}

        if subject_name:
            subject_path = os.path.join(base_path, "stc", subject_name)
            if os.path.exists(subject_path):
                files, _ = self.data_io.find_data(subject_path, ".h5", pattern="*")
                stc_files[subject_name] = files
        else:
            stc_base_path = os.path.join(base_path, "stc")
            if os.path.exists(stc_base_path):
                subjects = [
                    d
                    for d in os.listdir(stc_base_path)
                    if os.path.isdir(os.path.join(stc_base_path, d))
                ]

                for subject in subjects:
                    subject_path = os.path.join(stc_base_path, subject)
                    files, _ = self.data_io.find_data(subject_path, ".h5", pattern="*")
                    if files:
                        stc_files[subject] = files

        return stc_files

    def load_tess_results(self, tess_path, subject_name, result_type="filtered"):
        """Load TESS analysis results.

        Args:
            tess_path: Path to TESS results directory
            subject_name: Subject name
            result_type: Type of results ('filtered', 'raw', 'pvalues')

        Returns:
            Loaded numpy array or None
        """
        subject_path = os.path.join(tess_path, subject_name)

        if result_type == "filtered":
            pattern = "*filtered_zscore*"
        elif result_type == "raw":
            pattern = "*zscore*"
        elif result_type == "pvalues":
            pattern = "*p_values*"
        else:
            return None

        files, _ = self.data_io.find_data(subject_path, ".npy", pattern=pattern)

        if files:
            try:
                return np.load(files[0])
            except Exception:
                return None
        else:
            return None

    @staticmethod
    def save_tess_results(
        output_path, subject_name, z_scores=None, filtered_z_scores=None, p_values=None
    ):
        """Save TESS analysis results.

        Args:
            output_path: Output directory path
            subject_name: Subject name
            z_scores: Raw z-scores array
            filtered_z_scores: Filtered z-scores array
            p_values: P-values array
        """
        subject_path = os.path.join(output_path, subject_name)
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)

        if z_scores is not None:
            np.save(os.path.join(subject_path, f"{subject_name}-zscore.npy"), z_scores)

        if filtered_z_scores is not None:
            np.save(
                os.path.join(subject_path, f"{subject_name}-filtered_zscore.npy"), filtered_z_scores
            )

        if p_values is not None:
            np.save(os.path.join(subject_path, f"{subject_name}-p_values.npy"), p_values)

    def load_averaged_sources(self, avg_path, subject_name, microstate_label=None):
        """Load averaged source data.

        Args:
            avg_path: Path to averaged sources directory
            subject_name: Subject name
            microstate_label: Specific microstate label (optional)

        Returns:
            Dictionary with microstate labels as keys and data arrays as values,
            or single array if microstate_label specified
        """
        subject_path = os.path.join(avg_path, subject_name)

        if microstate_label:
            file_path = os.path.join(subject_path, f"{subject_name}_{microstate_label}.npy")
            if os.path.exists(file_path):
                try:
                    return np.load(file_path)
                except Exception:
                    return None
            else:
                return None
        else:
            if not os.path.exists(subject_path):
                return {}

            files, _ = self.data_io.find_data(subject_path, ".npy", pattern=f"{subject_name}_*")

            averaged_data = {}
            for file_path in files:
                try:
                    filename = os.path.basename(file_path)
                    microstate = filename.replace(f"{subject_name}_", "").replace(".npy", "")
                    averaged_data[microstate] = np.load(file_path)
                except Exception:
                    continue

            return averaged_data

    @staticmethod
    def save_averaged_sources(output_path, subject_name, averaged_data_dict):
        """Save averaged source data.

        Args:
            output_path: Output directory path
            subject_name: Subject name
            averaged_data_dict: Dictionary with microstate labels as keys and data arrays as values
        """
        subject_path = os.path.join(output_path, subject_name)
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)

        for microstate_label, data_array in averaged_data_dict.items():
            filename = os.path.join(subject_path, f"{subject_name}_{microstate_label}.npy")
            np.save(filename, data_array)

    @staticmethod
    def get_stc_info(stc_file_path):
        """Get information about an STC file without fully loading the data.

        Args:
            stc_file_path: Path to the STC file

        Returns:
            Dictionary with STC information
        """
        try:
            stc = mne.read_source_estimate(stc_file_path)

            return {
                "tmin": stc.tmin,
                "tmax": stc.times[-1],
                "tstep": stc.tstep,
                "n_times": len(stc.times),
                "n_vertices": stc.data.shape[0],
                "data_shape": stc.data.shape,
                "subject": stc.subject,
                "times": stc.times,
            }

        except Exception:
            return None

    @staticmethod
    def validate_stc_file(stc_file_path):
        """Validate if an STC file can be loaded properly.

        Args:
            stc_file_path: Path to the STC file

        Returns:
            Boolean indicating if file is valid
        """
        try:
            mne.read_source_estimate(stc_file_path)
            return True
        except Exception:
            return False

    def find_source_data_by_type(self, base_path, data_type="tess"):
        """Find source data files by type across all subjects.

        Args:
            base_path: Base directory containing source data
            data_type: Type of source data ('tess', 'avg', 'stc')

        Returns:
            Dictionary with subjects and their available files
        """
        if data_type == "tess":
            search_path = os.path.join(base_path, "tess_sources")
        elif data_type == "avg":
            search_path = os.path.join(base_path, "avg_sources")
        elif data_type == "stc":
            search_path = os.path.join(base_path, "stc")
        else:
            return {}

        if not os.path.exists(search_path):
            return {}

        subjects_data = {}
        subjects = [
            d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))
        ]

        for subject in subjects:
            subject_path = os.path.join(search_path, subject)

            if data_type in ["tess", "avg"]:
                files, _ = self.data_io.find_data(subject_path, ".npy", pattern="*")
            else:  # stc
                files, _ = self.data_io.find_data(subject_path, ".h5", pattern="*")

            if files:
                subjects_data[subject] = files

        return subjects_data

    @staticmethod
    def export_src_bem_trans(subjects_dir, subject, src, bem, trans, spacing):
        """Export source space, BEM, and coregistration transformations.

        Args:
            subjects_dir: Directory containing subject data
            subject: The subject identifier
            src: The source space data
            bem: The BEM data
            trans: The coregistration transformation data
            spacing: The spacing parameter
        """
        subject_dir = os.path.join(subjects_dir, subject)
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)

        try:
            # Suppress verbose output during file writing
            with mne.utils.use_log_level("ERROR"):
                mne.write_source_spaces(
                    os.path.join(subject_dir, f"{subject}-{spacing}-src.fif"), src, overwrite=True
                )

                mne.write_bem_solution(
                    os.path.join(subject_dir, f"{subject}-bem.fif"), bem, overwrite=True
                )

                # For fsaverage, don't write transform file (use string "fsaverage" directly)
                if trans != "fsaverage":
                    mne.write_trans(
                        os.path.join(subject_dir, f"{subject}-trans.fif"), trans, overwrite=True
                    )
        except Exception:
            pass

    @staticmethod
    def load_src_bem_trans(subjects_dir, subject, spacing):
        """Load source space, BEM, and coregistration transformations.

        Args:
            subjects_dir: Directory containing subject data
            subject: The subject identifier
            spacing: The spacing parameter

        Returns:
            Tuple of (src, bem, trans) or (None, None, None) if files don't exist
        """
        subject_dir = os.path.join(subjects_dir, subject)

        src_file = os.path.join(subject_dir, f"{subject}-{spacing}-src.fif")
        bem_file = os.path.join(subject_dir, f"{subject}-bem.fif")
        trans_file = os.path.join(subject_dir, f"{subject}-trans.fif")

        try:
            # Handle fsaverage template specially
            if subject == "fsaverage":
                # For fsaverage, use built-in transform
                if os.path.exists(src_file) and os.path.exists(bem_file):
                    src = mne.read_source_spaces(src_file)
                    bem = mne.read_bem_solution(bem_file)
                    trans = "fsaverage"
                    return src, bem, trans
                return None, None, None
            else:
                # For other subjects, load all files including transform
                if all(os.path.exists(f) for f in [src_file, bem_file, trans_file]):
                    src = mne.read_source_spaces(src_file)
                    bem = mne.read_bem_solution(bem_file)
                    trans = mne.read_trans(trans_file)
                    return src, bem, trans
                return None, None, None
        except Exception:
            return None, None, None

    def load_all_averaged_sources_for_microstate(self, avg_path, subjects, microstate_label):
        """Load averaged source data for a specific microstate across multiple subjects.

        Args:
            avg_path: Path to averaged sources directory
            subjects: List of subject names
            microstate_label: Microstate label to load

        Returns:
            List of data arrays for the specified microstate across subjects
        """
        all_data = []

        for subject_name in subjects:
            data = self.load_averaged_sources(avg_path, subject_name, microstate_label)
            if data is not None:
                # Handle 2D data by taking mean across first dimension
                if data.ndim == 2:
                    data = np.mean(data, axis=0)
                all_data.append(data)

        return all_data

    @staticmethod
    def get_available_microstates_for_subject(avg_path, subject_name):
        """Get list of available microstate labels for a specific subject.

        Args:
            avg_path: Path to averaged sources directory
            subject_name: Subject name

        Returns:
            List of available microstate labels
        """
        subject_path = os.path.join(avg_path, subject_name)

        if not os.path.exists(subject_path):
            return []

        files = [f for f in os.listdir(subject_path) if f.endswith(".npy")]
        microstates = []

        for file in files:
            if file.startswith(f"{subject_name}_"):
                microstate = file.replace(f"{subject_name}_", "").replace(".npy", "")
                microstates.append(microstate)

        return microstates

    def validate_averaged_sources_consistency(self, avg_path, subjects, expected_microstates):
        """Validate that all subjects have consistent averaged source data.

        Args:
            avg_path: Path to averaged sources directory
            subjects: List of subject names
            expected_microstates: List of expected microstate labels

        Returns:
            Tuple of (is_valid, validation_report)
        """
        validation_report = {
            "valid_subjects": [],
            "invalid_subjects": [],
            "missing_microstates": {},
            "shape_inconsistencies": {},
        }

        reference_shapes = {}

        for subject_name in subjects:
            subject_valid = True
            available_microstates = self.get_available_microstates_for_subject(
                avg_path, subject_name
            )

            # Check if all expected microstates are present
            missing = [ms for ms in expected_microstates if ms not in available_microstates]
            if missing:
                validation_report["missing_microstates"][subject_name] = missing
                subject_valid = False

            # Check data shapes
            subject_shapes = {}
            for microstate in expected_microstates:
                if microstate in available_microstates:
                    data = self.load_averaged_sources(avg_path, subject_name, microstate)
                    if data is not None:
                        subject_shapes[microstate] = data.shape

            # Compare with reference shapes (first valid subject)
            if not reference_shapes and subject_shapes:
                reference_shapes = subject_shapes.copy()
            elif subject_shapes:
                for microstate, shape in subject_shapes.items():
                    if microstate in reference_shapes and shape != reference_shapes[microstate]:
                        if subject_name not in validation_report["shape_inconsistencies"]:
                            validation_report["shape_inconsistencies"][subject_name] = {}
                        validation_report["shape_inconsistencies"][subject_name][microstate] = {
                            "expected": reference_shapes[microstate],
                            "actual": shape,
                        }
                        subject_valid = False

            if subject_valid:
                validation_report["valid_subjects"].append(subject_name)
            else:
                validation_report["invalid_subjects"].append(subject_name)

        is_valid = len(validation_report["invalid_subjects"]) == 0
        return is_valid, validation_report
