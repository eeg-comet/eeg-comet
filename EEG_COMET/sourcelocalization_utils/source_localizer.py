import os
import numpy as np
import mne
from scipy import stats
from invert import Solver
from invert.config import all_solvers
from data_utils.data_io import DataIO
from backfitting_utils.segmentation_io import SegmentationIO
from controllers.logging_window import LogWindow


# TODO: send log to COMET
class SourceLocalizer:
    """
    SourceLocalizer class for performing source localization on EEG data.
    """
    def __init__(self, subjects_dir, localized_sources_path, preprocessed_data_path, segmentation_path,
                 use_anatomy, extension, datatype, spacing, inverse_method, microstate_maps, nperm):
        """
        Initialize the SourceLocalizer
        """
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
        self.spacing = spacing
        self.inverse_method = inverse_method
        self.microstate_maps = microstate_maps
        self.tess_path = None
        self.avg_sources_path = None
        self.nperm = nperm
        self.data_io = DataIO()

    def stc_write(self, stc_data_subject_path, stc_file):
        """
        Write source time series to disk.

        Args:
            stc_data_subject_path: The path to the directory where the source time series will be saved.
            stc_file: The source time series data to be saved.
        """

        if self.datatype == "epoched":
            for idx, stc in enumerate(stc_file):
                filename = f'stc_{idx}'
                filepath = os.path.join(stc_data_subject_path, filename)
                stc.save(filepath, ftype='h5', overwrite=True)
        else:
            filepath = os.path.join(stc_data_subject_path, "stc_file")
            stc_file.save(filepath, ftype='h5', overwrite=True)

    def stc_read(self, stc_data_subject_path):
        """
        Read source time series from disk.

        Args:
            stc_data_subject_path: The path to the directory where the source time series is stored.

        Returns:
            stc_file: The loaded source time series data.
        """

        stc_list, _ = self.data_io.find_data(stc_data_subject_path, '.h5', pattern='*')
        return [mne.read_source_estimate(stc_path) for stc_path in stc_list]

    def export_src_bem_trans(self, subject, src, bem, trans):
        """
        Export source space, BEM, and coregistration transformations.

        Args:
            subject: The subject identifier.
            src: The source space data.
            bem: The BEM data.
            trans: The coregistration transformation data.
        """

        mne.write_source_spaces(
            os.path.join(self.subjects_dir, subject, f'{subject}-{self.spacing}-src.fif'), src, overwrite=True)
        mne.write_bem_solution(
            os.path.join(self.subjects_dir, subject, f'{subject}-bem.fif'), bem, overwrite=True)
        mne.write_trans(
            os.path.join(self.subjects_dir, subject, f'{subject}-trans.fif'), trans, overwrite=True)

    def load_average_mri(self, eeg_info):
        """
        Load the standard fsaverage MRI subject and create corresponding BEM and coregistration transformations.

        Args:
            eeg_info: The EEG info object.

        Returns:
            src: The source space data.
            bem: The BEM data.
            trans: The coregistration transformation data.
        """

        # Print information about using the standard template MRI subject -fsaverage-
        print('\nUsing the standard template MRI subject -fsaverage-')
        print('\nWarning, patient-specific MRI is more accurate!')

        # Download fsaverage files if they don't exist
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subjects_dir = os.path.dirname(fs_dir)
        print(f"fsaverage subjects_dir path: {subjects_dir}")

        # The files live in:
        subject = "fsaverage"
        trans = "fsaverage"  # MNE has a built-in fsaverage transformation

        # Check if a different spacing is provided, then create a new source space
        if self.spacing != 'ico5':
            src = mne.setup_source_space(
                subject,
                spacing=self.spacing,
                subjects_dir=subjects_dir,
                add_dist=False,
                n_jobs=-1
            )
        else:
            # Use the default 'ico5' source space
            src_path = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
            src = mne.read_source_spaces(src_path)

        # Specify the path to the BEM file
        bem_model = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
        bem = mne.make_bem_solution(bem_model, solver='mne')

        # Get MNI fiducials for the subject
        fiducials = mne.coreg.get_mni_fiducials(
            subject=subject, subjects_dir=subjects_dir)

        # Perform coregistration based on fiducials and measurement info
        coreg = mne.coreg.Coregistration(
            info=eeg_info,
            subject=subject,
            subjects_dir=subjects_dir,
            fiducials=fiducials
        )
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

        # Omit head shape points that are too close to the MRI surface
        coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
        trans = coreg.trans

        return src, bem, trans

    def individual_mri(self, subject, raw_info):
        """
        Perform individual MRI coregistration for a specific subject.

        Args:
            subject: The subject identifier.
            raw_info: The raw EEG info object.

        Returns:
            src: The source space data.
            bem: The BEM data.
            trans: The coregistration transformation data.
        """

        print(f"\nUsing the individual MRI subject: {subject}")
        print('\nThis may take some time to compute ...')
        # Get MNI fiducials for the subject
        fiducials = mne.coreg.get_mni_fiducials(
            subject=subject, subjects_dir=self.subjects_dir)

        # Perform coregistration based on fiducials and measurement info
        coreg = mne.coreg.Coregistration(
            info=raw_info,
            subject=subject,
            subjects_dir=self.subjects_dir,
            fiducials=fiducials
        )
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

        # Omit head shape points that are too close to the MRI surface
        coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters

        # Compute distances between digitization points and MRI surface in millimeters
        dists = coreg.compute_dig_mri_distances() * 1e3
        print(
            f"Distance between HSP and MRI (mean/min/max): "
            f"{np.mean(dists):.2f} mm / {np.min(dists):.2f} mm / {np.max(dists):.2f} mm")

        # Save the transformation matrix to a file
        trans_file = os.path.join(self.subjects_dir, subject, 'mri', 'transforms', f'{subject}-trans.fif')
        trans = coreg.trans

        # Set up the source space
        src = mne.setup_volume_source_space(
            subject,
            subjects_dir=self.subjects_dir,
            pos=10.0,
            mri='T1.mgz',
            mindist=5.0
        )

        # Make BEM surfaces and save to a file
        bem_surfaces = mne.make_bem_model(
            subject=subject, subjects_dir=self.subjects_dir, ico=self.spacing[-1])

        # bem_path = os.path.join(subjects_dir, subject, 'bem', f'{subject}-bem-sol.fif')
        # mne.write_bem_surfaces(bem_path, bem_surfaces, overwrite=True)

        # Make BEM solution
        bem = mne.make_bem_solution(bem_surfaces)

        return src, bem, trans

    def compute_stc(self, src, bem, trans, raw_eeg, raw_info):
        """
        Compute the source time series using the minimum-norm inverse method.

        Args:
            src: The source space data.
            bem: The BEM data.
            trans: The coregistration transformation data.
            raw_eeg: The raw EEG data (already converted to a standard MNE object).
            raw_info: The raw EEG info object.

        Returns:
            stc_file: The source time series data.
        """
        print('\nPerforming source localization ...')

        # Calculate the forward solution using the specified parameters
        fwd = mne.make_forward_solution(
            raw_info, trans, src, bem, eeg=True, mindist=5.0, n_jobs=-1)

        try:
            from invert import Solver

            print(f"Using Solver with {self.inverse_method} method")
            solver = Solver("MNE")  # Always use MNE for now

            try:
                solver.make_inverse_operator(fwd, raw_eeg)
                stc = solver.apply_inverse_operator(raw_eeg)
                return stc
            except Exception as e:
                print(f"Error using Solver: {str(e)}")
                print("Falling back to standard MNE methods")
        except ImportError:
            print("Solver not available, using standard MNE methods")

    def localize_single_file(self, eeg_path, eeg_name):
        """
        Perform source localization for a single EEG file.

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
            print(f"Source Localizing {eeg_name}")

            # Create directory for this subject's source time courses
            stc_subject_path = os.path.join(self.stc_path, eeg_name)
            if not os.path.exists(stc_subject_path):
                os.makedirs(stc_subject_path)

            # Load EEG data
            eeg = self.data_io.load_eeg(eeg_path, self.datatype)

            # Ensure we have valid EEG data
            if eeg is None:
                print(f"Error: Could not load EEG data from {eeg_path}")
                return False

            # Extract EEG info
            eeg_info = eeg.info

            # Convert EEGLAB objects to standard MNE objects
            if self.datatype == 'raw':
                # Convert RawEEGLAB to standard RawArray
                if not isinstance(eeg, mne.io.fiff.raw.Raw):
                    print(f"Converting {type(eeg).__name__} to standard RawArray")
                    eeg_data = eeg.get_data()
                    standard_eeg = mne.io.RawArray(eeg_data, eeg_info)
                else:
                    standard_eeg = eeg
            else:  # 'epoched'
                # Convert EpochsEEGLAB to standard EpochsArray
                if not isinstance(eeg, mne.epochs.Epochs):
                    print(f"Converting {type(eeg).__name__} to standard EpochsArray")
                    eeg_data = eeg.get_data()

                    # Get or create events if needed
                    if hasattr(eeg, 'events') and eeg.events is not None:
                        events = eeg.events
                    else:
                        # Create simple events array if not available
                        n_epochs = eeg_data.shape[0]
                        events = np.column_stack([
                            np.arange(n_epochs),
                            np.zeros(n_epochs, dtype=int),
                            np.ones(n_epochs, dtype=int)
                        ])

                    # Get or create event_id dictionary
                    if hasattr(eeg, 'event_id') and eeg.event_id:
                        event_id = eeg.event_id
                    else:
                        event_id = {'event': 1}

                    # Get time info
                    if hasattr(eeg, 'tmin'):
                        tmin = eeg.tmin
                    else:
                        tmin = 0.0

                    # Create standard EpochsArray
                    standard_eeg = mne.EpochsArray(eeg_data, eeg_info, events=events,
                                                   event_id=event_id, tmin=tmin)
                else:
                    standard_eeg = eeg

            # Choose appropriate method based on anatomy selection
            if self.use_anatomy == "individual":
                print("\nUsing individual anatomies")
                subject = eeg_name
                src, bem, trans = self.individual_mri(subject, eeg_info)
                self.export_src_bem_trans('fsaverage', src, bem, trans)
                stc_file = self.compute_stc(src, bem, trans, standard_eeg, eeg_info)
                # Morph to fsaverage
                src_morph = mne.read_source_spaces(src)
                morph = mne.compute_source_morph(
                    src_morph,
                    subject_from=subject,
                    subject_to='fsaverage',
                    subjects_dir=self.subjects_dir,
                    spacing=self.spacing[-1]
                )
                morph.save(
                    os.path.join(
                        self.subjects_dir,
                        subject,
                        f'{subject}-morph.h5',
                    ),
                    overwrite=True,
                )
                stc_file = morph.apply(stc_file)
            else:
                print("\nUsing default template brain - fsaverage")
                src, bem, trans = self.load_average_mri(eeg_info)
                self.export_src_bem_trans('fsaverage', src, bem, trans)
                stc_file = self.compute_stc(src, bem, trans, standard_eeg, eeg_info)

            print(f"\nExporting Source Time Courses: {eeg_name}")
            self.stc_write(stc_subject_path, stc_file)
            return True
        except Exception as e:
            import traceback
            print(f"Error processing {eeg_name}: {str(e)}")
            print(traceback.format_exc())  # Print detailed error traceback
            return False

    @staticmethod
    def find_t_coeff(sample, maps):
        """
        Find T coefficients.
        """
        return np.linalg.lstsq(maps, sample, rcond=None)[0]

    def first_regression(self, sensor_time_series, maps):
        """
        Perform the first regression to extract T coefficients.
        """
        return np.apply_along_axis(self.find_t_coeff, 0, sensor_time_series, maps).transpose()

    @staticmethod
    def second_regression(t_coeff, stc_data):
        """
        Perform the second regression to get beta coefficients.
        """
        return np.linalg.solve(t_coeff.T @ t_coeff, t_coeff.T @ stc_data)

    def run_tess(self, stc_data, eeg_data, nperm):
        """
        Run the TESS algorithm.

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
        # Permutation of beta over t to determine significance
        z_scores = np.zeros(beta_coeff.shape)
        beta_dist = np.zeros((beta_coeff.shape[0], beta_coeff.shape[1], nperm))
        bonferroni = beta_coeff.shape[1]
        t_shuffle = t_coeff
        for ii in range(nperm):
            np.random.shuffle(t_shuffle)
            beta_dist[:, :, ii] = self.second_regression(t_shuffle, stc_data)
            if (ii % 50) == 0:
                print(ii)
        for idx, x in np.ndenumerate(beta_coeff):
            z_scores[idx] = stats.zscore(np.insert(beta_dist[idx[0]][idx[1]][:], 0, x))[0]
        p_values = bonferroni * stats.norm.sf(abs(z_scores))

        # Filter z-scores
        significance = 0.005  # assuming the nperm=2000
        filtered_z_scores = (p_values < significance) * z_scores

        return p_values, z_scores, filtered_z_scores

    def avg_sources(self, labelled_data_path, stc_data):
        """
        Averages the source data over times matched with each microstate segment.

        Args:
            labelled_data_path: The path to the directory where the labelled data is stored.
            stc_data: The source time series data.

        Returns:
            all_sources_dict: A dictionary containing the averaged source data for each microstate.
        """

        self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")
        if not os.path.exists(self.avg_sources_path):
            os.makedirs(self.avg_sources_path)

        list_segmented_data, _ = DataIO().find_data(labelled_data_path, '.csv', pattern='*')
        all_sources_dict = {}
        for segmented_path in list_segmented_data:
            segment_data = SegmentationIO().load_segmentation(segmented_path, import_format='.csv')
            for m in segment_data['segmentation'].unique():
                sources_m_times = segment_data.index[segment_data['segmentation'] == m].tolist()
                sources_m = stc_data[sources_m_times, :]
                sources_m_mean = np.mean(sources_m, axis=0)
                # Append the averaged source data to the dictionary with key as 'm'
                if m in all_sources_dict:
                    all_sources_dict[m].append(sources_m_mean)
                else:
                    all_sources_dict[m] = [sources_m_mean]

                # If you want to convert the lists to NumPy arrays for each 'm'
            for m in all_sources_dict:
                all_sources_dict[m] = np.vstack(all_sources_dict[m])

            return all_sources_dict

    def identify_sources_single_file(self, eeg_path, eeg_name, source_method):
        """
        Identify microstate sources for a single EEG file.

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
            print(f"\nLoading Source Time Courses: {eeg_name}")
            stc_subject_path = os.path.join(self.stc_path, eeg_name)
            stc_file = self.stc_read(stc_subject_path)
            stc_data = stc_file[0].data.T

            # Load EEG data
            eeg = self.data_io.load_eeg(eeg_path, self.datatype)

            # Ensure we have valid EEG data
            if eeg is None:
                print(f"Error: Could not load EEG data from {eeg_path}")
                return False

            # Get EEG data in the right format for source identification
            if self.datatype == 'raw':
                eeg_data = eeg.get_data()
            else:  # 'epoched'
                # For epoched data, we'll use the average across epochs
                # This is necessary for methods like TESS that expect 2D data
                eeg_data = eeg.get_data().mean(axis=0)
                print(f"Using mean of {eeg.get_data().shape[0]} epochs for source identification")

            # Check data dimensions and transpose if necessary
            if eeg_data.shape[0] > eeg_data.shape[1]:
                print(f"Transposing EEG data from shape {eeg_data.shape} for source identification")
                eeg_data = eeg_data.T

            if source_method == 'tess':
                p_values, z_scores, filtered_z_scores = self.run_tess(stc_data, eeg_data, self.nperm)
                print('\nExtracting sources associated with each microstate',
                      '\nusing the topographic electrophysiological state source-imaging (TESS) algorithm ...')
                tess_subject_path = os.path.join(self.tess_path, eeg_name)
                if not os.path.exists(tess_subject_path):
                    os.makedirs(tess_subject_path)
                print('\nTess Completed on File ...')
                np.save(os.path.join(tess_subject_path, f'{eeg_name}-filtered_zscore.npy'), filtered_z_scores)
                np.save(os.path.join(tess_subject_path, f'{eeg_name}-zscore.npy'), z_scores)
                np.save(os.path.join(tess_subject_path, f'{eeg_name}-p_values.npy'), p_values)
            elif source_method == 'avg':
                # Average sources over times matched with each microstate
                print('\nAveraging sources over times matched with each microstate ...')
                avg_subject_path = os.path.join(self.avg_sources_path, eeg_name)
                if not os.path.exists(avg_subject_path):
                    os.makedirs(avg_subject_path)

                all_sources_dict = self.avg_sources(self.segmentation_path, stc_data)
                print(f'\nExporting the averaged microstate sources for subject {eeg_name}')
                for m, array_data in all_sources_dict.items():
                    filename = os.path.join(avg_subject_path, f"{eeg_name}_{m}.npy")
                    np.save(filename, array_data)
            return True
        except Exception as e:
            import traceback
            print(f"Error processing {eeg_name}: {str(e)}")
            print(traceback.format_exc())  # Print detailed error traceback
            return False
