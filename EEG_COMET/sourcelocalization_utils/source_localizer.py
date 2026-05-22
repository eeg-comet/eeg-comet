"""Source localization routines for EEG-COMET and related helpers."""

import os
from contextlib import redirect_stderr, redirect_stdout

import mne
import numpy as np
from invert import Solver
from scipy import stats

from backfitting_utils.segmentation_io import SegmentationIO
from data_utils.data_io import DataIO
from sourcelocalization_utils.source_io import SourceIO

# Suppress MNE verbose output globally
mne.set_log_level("ERROR")


class SourceLocalizer:
    """Perform source localization on EEG data and manage STC/TESS outputs."""

    def __init__(
        self,
        subjects_dir,
        localized_sources_path,
        preprocessed_data_path,
        segmentation_path,
        use_anatomy,
        extension,
        datatype,
        bem_solver,
        inverse_method,
        spacing,
        microstate_maps,
        nperm,
        logger=None,
        random_seed=None,
        alpha=0.05,
    ):
        """Initialize the SourceLocalizer."""
        self.subjects_dir = subjects_dir
        self.localized_sources_path = localized_sources_path
        stc_path = os.path.join(localized_sources_path, "stc")
        if not os.path.exists(stc_path):
            os.makedirs(stc_path)
        self.stc_path = stc_path
        self.preprocessed_data_path = preprocessed_data_path
        self.segmentation_path = segmentation_path
        self.use_anatomy = use_anatomy
        self.extension = extension
        self.datatype = datatype
        self.bem_solver = bem_solver
        self.inverse_method = inverse_method
        self.spacing = spacing
        self.microstate_maps = microstate_maps
        self.tess_path = None
        self.avg_sources_path = None
        self.nperm = nperm
        self.data_io = DataIO()
        self.source_io = SourceIO()  # Initialize SourceIO
        self.logger = logger
        self.random_seed = random_seed
        self.alpha = alpha

    def _log(self, message, level="info"):
        """Helper method to log messages using the logger if available, otherwise print."""
        if self.logger is not None:
            if level == "info":
                self.logger.processing_info("SOURCE_LOCALIZATION", message)
            elif level == "warning":
                self.logger.warning("SOURCE_LOCALIZATION", message)
            elif level == "error":
                self.logger.error("SOURCE_LOCALIZATION", message)
            elif level == "success":
                self.logger.processing_success("SOURCE_LOCALIZATION", message)
            elif level == "process":
                self.logger.processing_start("SOURCE_LOCALIZATION", message)
        else:
            # Fallback to print if no logger is available
            print(f"[SOURCE_LOCALIZATION] {message}")

    def log_settings(self):
        """Log source localization settings once at the beginning."""
        if self.use_anatomy == "individual":
            self._log("Using individual MRI anatomies")
            self._log("Warning: Individual MRI requires proper subject directory setup", "warning")
        else:
            self._log("Using standard template brain (fsaverage)")
            self._log("Warning: Patient-specific MRI is more accurate", "warning")

        self._log(f"Boundary Element Method: {self.bem_solver}")
        self._log(f"Inverse Method: {self.inverse_method}")
        self._log(f"Source Space Spacing: {self.spacing}")
        self._log("Performing Source Localization...", "process")

    def stc_write(self, stc_data_subject_path, stc_file):
        """Write source time series to disk using SourceIO."""
        self.source_io.write_stc(stc_file, stc_data_subject_path, datatype=self.datatype)

    def stc_read(self, stc_data_subject_path):
        """Read source time series from disk using SourceIO."""
        return self.source_io.read_stc(stc_data_subject_path)

    def export_src_bem_trans(self, subject, src, bem, trans):
        """Export source space, BEM, and coregistration transformations using SourceIO."""
        self.source_io.export_src_bem_trans(
            self.subjects_dir, subject, src, bem, trans, self.spacing
        )

    def load_average_mri(self, eeg_info):
        """Load the standard fsaverage MRI subject and create corresponding BEM and coregistration transformations.

        Args:
            eeg_info: The EEG info object (should already have standard montage applied).

        Returns:
            src: The source space data.
            bem: The BEM data.
            trans: The coregistration transformation data.
        """
        # Download fsaverage files if they don't exist
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)

        # The files live in:
        subject = "fsaverage"
        # Use with fsaverage - the built-in transform handles alignment
        trans = "fsaverage"  # MNE has a built-in fsaverage transformation

        # Check if a different spacing is provided, then create a new source space
        if self.spacing != "ico5":
            with mne.utils.use_log_level("ERROR"):
                src = mne.setup_source_space(
                    subject,
                    spacing=self.spacing,
                    subjects_dir=subjects_dir,
                    add_dist=False,
                    n_jobs=-1,
                    verbose=False,
                )
        else:
            # Use the default 'ico5' source space
            src_path = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
            with mne.utils.use_log_level("ERROR"):
                src = mne.read_source_spaces(src_path, verbose=False)

        # Specify the path to the BEM file
        bem_model = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
        with mne.utils.use_log_level("ERROR"):
            bem = mne.make_bem_solution(bem_model, solver=self.bem_solver, verbose=False)

        # For fsaverage, no coregistration is needed - use the built-in transform
        return src, bem, trans

    def individual_mri(self, subject, raw_info):
        """Perform individual MRI coregistration for a specific subject.

        Args:
            subject: The subject identifier.
            raw_info: The raw EEG info object.

        Returns:
            src: The source space data.
            bem: The BEM data.
            trans: The coregistration transformation data.
        """
        # Get MNI fiducials for the subject
        with mne.utils.use_log_level("ERROR"):
            fiducials = mne.coreg.get_mni_fiducials(
                subject=subject, subjects_dir=self.subjects_dir, verbose=False
            )

        # Perform coregistration based on fiducials and measurement info
        with mne.utils.use_log_level("ERROR"):
            coreg = mne.coreg.Coregistration(
                info=raw_info, subject=subject, subjects_dir=self.subjects_dir, fiducials=fiducials
            )
            coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)

            # Omit head shape points that are too close to the MRI surface
            coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters

            # Compute distances between digitization points and MRI surface in millimeters
            dists = coreg.compute_dig_mri_distances() * 1e3
            self._log(
                f"Distance between HSP and MRI (mean/min/max): {np.mean(dists):.2f} mm / {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
            )

            # Save the transformation matrix to a file
            os.path.join(self.subjects_dir, subject, "mri", "transforms", f"{subject}-trans.fif")
            trans = coreg.trans

        # Set up the source space
        with mne.utils.use_log_level("ERROR"):
            src = mne.setup_volume_source_space(
                subject,
                subjects_dir=self.subjects_dir,
                pos=10.0,
                mri="T1.mgz",
                mindist=5.0,
                verbose=False,
            )

            # Make BEM surfaces and save to a file
            bem_surfaces = mne.make_bem_model(
                subject=subject, subjects_dir=self.subjects_dir, ico=self.spacing[-1], verbose=False
            )

            # Make BEM solution
            bem = mne.make_bem_solution(bem_surfaces, solver=self.bem_solver, verbose=False)

        return src, bem, trans

    def compute_stc(self, src, bem, trans, raw_eeg, raw_info):
        """Compute the source time series using the minimum-norm inverse method.

        Args:
            src: The source space data.
            bem: The BEM data.
            trans: The coregistration transformation data.
            raw_eeg: The raw EEG data (already converted to a standard MNE object).
            raw_info: The raw EEG info object.

        Returns:
            stc_file: The source time series data.
        """
        # Calculate the forward solution (free orientation)
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull), mne.utils.use_log_level("ERROR"):
            fwd = mne.make_forward_solution(
                raw_info, trans, src, bem, eeg=True, mindist=5.0, n_jobs=-1, verbose=False
            )

            # Explicitly convert to fixed orientation
            fwd = mne.convert_forward_solution(
                fwd, surf_ori=True, force_fixed=True, verbose=False
            )

        solver = Solver(self.inverse_method)

        try:
            # Also suppress output during inverse operator creation
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull), mne.utils.use_log_level("ERROR"):
                solver.make_inverse_operator(fwd, raw_eeg)
                return solver.apply_inverse_operator(raw_eeg)
        except Exception as e:
            self._log(f"Error using Solver: {str(e)}", "error")

    def localize_single_file(self, eeg_path, eeg_name):
        """Perform source localization for a single EEG file.

        Parameters:
        -----------
        eeg_path : str
            Path to the EEG file
        eeg_name : str
            Name of the EEG file

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Reduced logging frequency - only log file name, not every processing step
            # This prevents UI freezing from too many thread-safe log updates
            # self._log(f"Processing file: {eeg_name}")

            # Create directory for this subject's source time courses
            stc_subject_path = os.path.join(self.stc_path, eeg_name)
            if not os.path.exists(stc_subject_path):
                os.makedirs(stc_subject_path)

            # Load EEG data
            eeg = self.data_io.load_eeg(eeg_path, self.datatype)

            # Ensure we have valid EEG data
            if eeg is None:
                self._log(f"Could not load EEG data from {eeg_path}", "error")
                return False

            # Extract EEG info
            eeg_info = eeg.info

            # Convert EEGLAB objects to standard MNE objects
            if self.datatype == "raw":
                # Convert RawEEGLAB to standard RawArray
                if not isinstance(eeg, mne.io.fiff.raw.Raw):
                    eeg_data = eeg.get_data()
                    # Suppress MNE verbose output during RawArray creation
                    with mne.utils.use_log_level("ERROR"):
                        standard_eeg = mne.io.RawArray(eeg_data, eeg_info)
                else:
                    standard_eeg = eeg
            else:  # 'epoched'
                # Convert EpochsEEGLAB to standard EpochsArray
                if not isinstance(eeg, mne.epochs.Epochs):
                    eeg_data = eeg.get_data()

                    # Get or create events if needed
                    if hasattr(eeg, "events") and eeg.events is not None:
                        events = eeg.events
                    else:
                        # Create simple events array if not available
                        n_epochs = eeg_data.shape[0]
                        events = np.column_stack(
                            [
                                np.arange(n_epochs),
                                np.zeros(n_epochs, dtype=int),
                                np.ones(n_epochs, dtype=int),
                            ]
                        )

                    # Get or create event_id dictionary
                    if hasattr(eeg, "event_id") and eeg.event_id:
                        event_id = eeg.event_id
                    else:
                        event_id = {"event": 1}

                    # Get time info
                    tmin = eeg.tmin if hasattr(eeg, "tmin") else 0.0

                    # Create standard EpochsArray with suppressed verbose output
                    with mne.utils.use_log_level("ERROR"):
                        standard_eeg = mne.EpochsArray(
                            eeg_data, eeg_info, events=events, event_id=event_id, tmin=tmin
                        )
                else:
                    standard_eeg = eeg

            # Choose appropriate method based on anatomy selection
            if self.use_anatomy == "individual":
                subject = eeg_name
                src, bem, trans = self.individual_mri(subject, eeg_info)
                with mne.utils.use_log_level("ERROR"):
                    self.export_src_bem_trans("fsaverage", src, bem, trans)
                stc_file = self.compute_stc(src, bem, trans, standard_eeg, eeg_info)
                # Morph to fsaverage
                with mne.utils.use_log_level("ERROR"):
                    src_morph = mne.read_source_spaces(src, verbose=False)
                    morph = mne.compute_source_morph(
                        src_morph,
                        subject_from=subject,
                        subject_to="fsaverage",
                        subjects_dir=self.subjects_dir,
                        spacing=self.spacing[-1],
                        verbose=False,
                    )
                # Suppress verbose output during morph save
                with mne.utils.use_log_level("ERROR"):
                    morph.save(
                        os.path.join(
                            self.subjects_dir,
                            subject,
                            f"{subject}-morph.h5",
                        ),
                        overwrite=True,
                    )
                stc_file = morph.apply(stc_file)
            else:
                # For fsaverage, ensure standard montage is applied
                # Apply standard montage to both the data and info for proper alignment
                montage = mne.channels.make_standard_montage('standard_1005')
                with mne.utils.use_log_level("ERROR"):
                    standard_eeg.set_montage(montage, match_case=False, on_missing='ignore')
                    eeg_info.set_montage(montage, match_case=False, on_missing='ignore')
                
                src, bem, trans = self.load_average_mri(eeg_info)
                with mne.utils.use_log_level("ERROR"):
                    self.export_src_bem_trans("fsaverage", src, bem, trans)
                stc_file = self.compute_stc(src, bem, trans, standard_eeg, eeg_info)

            # Reduced logging - export happens silently, success/failure logged in comet.py
            # self._log(f"Exporting source time courses: {eeg_name}")
            self.stc_write(stc_subject_path, stc_file)
            return True
        except Exception as e:
            # Only log errors, not routine processing steps
            self._log(f"Error processing {eeg_name}: {str(e)}", "error")
            return False

    @staticmethod
    def find_t_coeff(sample, maps):
        """Find T coefficients."""
        return np.linalg.lstsq(maps, sample, rcond=None)[0]

    def first_regression(self, sensor_time_series, maps):
        """Perform the first regression to extract T coefficients."""
        return np.apply_along_axis(self.find_t_coeff, 0, sensor_time_series, maps).transpose()

    @staticmethod
    def second_regression(t_coeff, stc_data):
        """Perform the second regression to get beta coefficients."""
        return np.linalg.solve(t_coeff.T @ t_coeff, t_coeff.T @ stc_data)

    def run_tess(self, stc_data, eeg_data, nperm):
        """Run the TESS algorithm.

        Args:
            stc_data: The source time series data.
            eeg_data: The EEG data.
            nperm: The number of permutations.

        Returns:
            p_values: The p-values.
            z_scores: The z-scores.
            filtered_z_scores: The filtered z-scores.
        """
        # TESS Algorithm
        # https://linkinghub.elsevier.com/retrieve/pii/S1053-8119(14)00243-2
        self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
        if not os.path.exists(self.tess_path):
            os.makedirs(self.tess_path)

        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        if self.microstate_maps.shape[0] != eeg_data.shape[0]:
            self.microstate_maps = self.microstate_maps.T

        t_coeff = self.first_regression(eeg_data, self.microstate_maps)
        beta_coeff = self.second_regression(t_coeff, stc_data)
        # Permute beta over t to estimate a null distribution. A seeded
        # Generator and a copy of ``t_coeff`` keep the regression coefficients
        # untouched and the result reproducible.
        rng = np.random.default_rng(self.random_seed)
        z_scores = np.zeros(beta_coeff.shape)
        beta_dist = np.zeros((beta_coeff.shape[0], beta_coeff.shape[1], nperm))
        bonferroni = beta_coeff.shape[1]
        t_shuffle = t_coeff.copy()
        for ii in range(nperm):
            rng.shuffle(t_shuffle, axis=0)
            beta_dist[:, :, ii] = self.second_regression(t_shuffle, stc_data)
            if (ii % 200) == 0 and ii > 0:
                self._log(f"TESS permutation progress: {ii}/{nperm}")
        for idx, x in np.ndenumerate(beta_coeff):
            z_scores[idx] = stats.zscore(np.insert(beta_dist[idx[0]][idx[1]][:], 0, x))[0]
        p_values = bonferroni * stats.norm.sf(abs(z_scores))

        # Bonferroni-corrected significance threshold derived from alpha and nperm.
        significance = self.alpha / max(nperm, 1)
        filtered_z_scores = (p_values < significance) * z_scores

        return p_values, z_scores, filtered_z_scores

    def avg_sources_single_file(self, segmentation_file_path, stc_data):
        """Averages the source data over times matched with each microstate segment for a single file.

        Args:
            segmentation_file_path: The path to the specific segmentation file to process.
            stc_data: The source time series data for this specific file.

        Returns:
            sources_dict: A dictionary containing the averaged source data for each microstate for this file.
        """
        self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")
        if not os.path.exists(self.avg_sources_path):
            os.makedirs(self.avg_sources_path)

        sources_dict = {}

        try:
            # Load using SegmentationIO
            segment_data = SegmentationIO().load_segmentation(
                segmentation_file_path, import_format=".csv"
            )

            if segment_data is None:
                return sources_dict

            # Process the array based on its dimensions
            if segment_data.ndim == 2:
                if segment_data.shape[0] == 1:
                    segmentation_labels = segment_data[0]
                else:
                    segmentation_labels = segment_data.flatten()
            else:
                segmentation_labels = segment_data

            # Handle object arrays
            if segmentation_labels.dtype == object:
                try:
                    segmentation_labels = segmentation_labels.astype(str)
                except Exception:
                    return sources_dict

            # Get unique labels
            unique_labels = np.unique(segmentation_labels)

            # Process each microstate
            for m in unique_labels:
                sources_m_times = np.where(segmentation_labels == m)[0]

                # Check bounds and filter valid time indices
                valid_times = [t for t in sources_m_times if 0 <= t < stc_data.shape[0]]

                if valid_times:
                    sources_m = stc_data[valid_times, :]
                    sources_m_mean = np.mean(sources_m, axis=0)
                    sources_dict[m] = sources_m_mean

        except Exception:
            pass

        return sources_dict

    def identify_sources_single_file(self, eeg_path, eeg_name, source_method):
        """Identify microstate sources for a single EEG file.

        Parameters:
        -----------
        eeg_path : str
            Path to the EEG file
        eeg_name : str
            Name of the EEG file
        source_method : str
            Method to use for source identification ('tess' or 'avg')

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            stc_subject_path = os.path.join(self.stc_path, eeg_name)
            stc_file = self.stc_read(stc_subject_path)

            if not stc_file:
                return False

            stc_data = stc_file[0].data.T

            # Load EEG data
            eeg = self.data_io.load_eeg(eeg_path, self.datatype)

            if eeg is None:
                return False

            # Get EEG data in the right format for source identification
            eeg_data = eeg.get_data() if self.datatype == "raw" else eeg.get_data().mean(axis=0)

            # Check data dimensions and transpose if necessary
            if eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T

            if source_method == "tess":
                p_values, z_scores, filtered_z_scores = self.run_tess(
                    stc_data, eeg_data, self.nperm
                )

                # Save TESS results using SourceIO
                self.source_io.save_tess_results(
                    self.tess_path, eeg_name, z_scores, filtered_z_scores, p_values
                )

            elif source_method == "avg":
                # Find the corresponding segmentation file for this EEG file
                segmentation_file_path = os.path.join(self.segmentation_path, f"{eeg_name}.csv")

                # Check if the segmentation file exists
                if not os.path.exists(segmentation_file_path):
                    base_name = os.path.splitext(eeg_name)[0]
                    segmentation_file_path = os.path.join(
                        self.segmentation_path, f"{base_name}.csv"
                    )

                    if not os.path.exists(segmentation_file_path):
                        return False

                # Process only this single file
                sources_dict = self.avg_sources_single_file(segmentation_file_path, stc_data)

                if sources_dict:
                    # Save averaged sources using SourceIO
                    self.source_io.save_averaged_sources(
                        self.avg_sources_path, eeg_name, sources_dict
                    )

            return True
        except Exception:
            return False
