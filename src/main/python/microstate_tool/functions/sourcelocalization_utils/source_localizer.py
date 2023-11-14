"""
Description: This class facilitates the localization of neural sources from EEG data.
It handles both the reading and writing of source time courses, as well as exporting
necessary transformations and BEM solutions. Different methods for coregistration
using individual or template MRIs are also available.

"""

import os
import numpy as np
import mne
from scipy import stats
from functions.data_utils.data_io import DataIO
from functions.backfitting_utils.segmentation_io import SegmentationIO


class SourceLocalizer:
    def __init__(self, subjects_dir, localized_sources_path, preprocessed_data_path, segmentation_path,
                 use_anatomy, extension, datatype, spacing, inv_method, microstate_maps, nperm):
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
        self.inv_method = inv_method
        self.microstate_maps = microstate_maps
        self.tess_path = None
        self.avg_sources_path = None
        self.nperm = nperm
        self.data_io = DataIO()

    def stc_write(self, stc_data_subject_path, stc_file):
        """Write source time series to disk."""
        if self.datatype == "epoched":
            for idx, stc in enumerate(stc_file):
                filename = f'stc_{idx}'
                filepath = os.path.join(stc_data_subject_path, filename)
                stc.save(filepath, ftype='h5', overwrite=True)
        else:
            filepath = os.path.join(stc_data_subject_path, "stc_file")
            stc_file.save(filepath, ftype='h5', overwrite=True)

    def stc_read(self, stc_data_subject_path):
        """Read source time series from disk."""
        stc_list, _ = self.data_io.find_data(stc_data_subject_path, '.h5', pattern='*')
        stc_file = [mne.read_source_estimate(stc_path) for stc_path in stc_list]
        return stc_file

    def export_src_bem_trans(self, subject, src, bem, trans):
        """Export source space, BEM, and coregistration transformations."""
        mne.write_source_spaces(os.path.join(self.subjects_dir, subject, f'{subject}-{self.spacing}-src.fif'), src, overwrite=True)
        mne.write_bem_solution(os.path.join(self.subjects_dir, subject, f'{subject}-bem.fif'), bem, overwrite=True)
        mne.write_trans(os.path.join(self.subjects_dir, subject, f'{subject}-trans.fif'), trans, overwrite=True)

    def load_average_mri(self, eeg_info):
        """Load the standard fsaverage MRI subject and create corresponding BEM and coregistration transformations."""
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
        if not self.spacing == 'ico5':
            src = mne.setup_source_space(subject,
                                         spacing=self.spacing,
                                         subjects_dir=subjects_dir,
                                         add_dist=False,
                                         n_jobs=-1)
        else:
            # Use the default 'ico5' source space
            src_path = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
            src = mne.read_source_spaces(src_path)

        # Specify the path to the BEM file
        bem_model = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
        bem = mne.make_bem_solution(bem_model, solver='mne')

        # Get MNI fiducials for the subject
        fiducials = mne.coreg.get_mni_fiducials(subject=subject,
                                                subjects_dir=subjects_dir)

        # Perform coregistration based on fiducials and measurement info
        coreg = mne.coreg.Coregistration(info=eeg_info,
                                         subject=subject,
                                         subjects_dir=subjects_dir,
                                         fiducials=fiducials)
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

        # Omit head shape points that are too close to the MRI surface
        coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
        trans = coreg.trans

        return src, bem, trans


    def individual_mri(self, subject, raw_info):
        """Perform individual MRI coregistration for a specific subject."""
        print(f"\nUsing the individual MRI subject: {subject}")
        print('\nThis may take some time to compute ...')
        # Get MNI fiducials for the subject
        fiducials = mne.coreg.get_mni_fiducials(subject=subject,
                                                subjects_dir=self.subjects_dir)

        # Perform coregistration based on fiducials and measurement info
        coreg = mne.coreg.Coregistration(info=raw_info,
                                         subject=subject,
                                         subjects_dir=self.subjects_dir,
                                         fiducials=fiducials)
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

        # Omit head shape points that are too close to the MRI surface
        coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters

        # Compute distances between digitization points and MRI surface in millimeters
        dists = coreg.compute_dig_mri_distances() * 1e3
        print(
            f"Distance between HSP and MRI (mean/min/max): {np.mean(dists):.2f} mm / {np.min(dists):.2f} mm / {np.max(dists):.2f} mm")

        # Save the transformation matrix to a file
        trans_file = os.path.join(self.subjects_dir, subject, 'mri', 'transforms', f'{subject}-trans.fif')
        trans = coreg.trans

        # Set up the source space
        src = mne.setup_volume_source_space(subject,
                                            subjects_dir=self.subjects_dir,
                                            pos=10.0,  # Distance of sources from the inner skull surface
                                            mri='T1.mgz',  # T1-weighted MRI file
                                            mindist=5.0)  # Minimum distance (in mm) between sources and inner skull surface

        # Make BEM surfaces and save to a file
        bem_surfaces = mne.make_bem_model(subject=subject, subjects_dir=self.subjects_dir, ico=self.spacing[-1])

        # bem_path = os.path.join(subjects_dir, subject, 'bem', f'{subject}-bem-sol.fif')
        # mne.write_bem_surfaces(bem_path, bem_surfaces, overwrite=True)

        # Make BEM solution
        bem = mne.make_bem_solution(bem_surfaces)

        return src, bem, trans


    def compute_stc(self, src, bem, trans, raw, raw_info):
        """Extract the forward solution and apply minimum-norm inverse to obtain the source time series."""
        print('\nPerforming source localization ...')
        # Calculate the forward solution using the specified parameters
        fwd = mne.make_forward_solution(raw_info, trans, src,
                                        bem, eeg=True, mindist=5.0, n_jobs=-1)
        # Compute noise covariance from the raw data
        if self.datatype == 'raw':
            noise_cov = mne.compute_raw_covariance(raw, method='auto', verbose=True, n_jobs=-1)
        elif self.datatype == 'epoched':
            noise_cov = mne.compute_covariance(raw, method='auto', verbose=True, n_jobs=-1)
        # Regularize noise covariance to avoid singularity issues
        noise_cov = mne.cov.regularize(noise_cov, raw_info,
                                       mag=0.1, grad=0.1, eeg=0.1, proj=True)
        # Create the inverse operator
        inverse_operator = mne.minimum_norm.make_inverse_operator(raw_info, fwd, noise_cov)
        # Set the regularization parameter for the inverse solution based on the signal-to-noise ratio (snr)
        snr = 3.
        lambda2 = 1. / snr ** 2
        # Apply minimum-norm inverse to obtain the source time series
        if self.datatype == 'raw':
            # Process raw data
            stc_file = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2,
                                                     method=self.inv_method, pick_ori=None, verbose=True)
        elif self.datatype == 'epoched':
            # Process epoched data
            stc_file = mne.minimum_norm.apply_inverse_epochs(raw, inverse_operator, lambda2,
                                                        method=self.inv_method, pick_ori=None, verbose=True)
        return stc_file

    def run_source_localization(self):
        """Perform source localization for multiple EEG files."""
        list_eeg_path, list_eeg_name = self.data_io.find_data(self.preprocessed_data_path, '.set', '*')
        for idx, (eeg_path, eeg_name) in enumerate(zip(list_eeg_path, list_eeg_name)):
            print(f"Source Localizing {eeg_name} ({idx + 1}/{len(list_eeg_path)})")
            stc_subject_path = os.path.join(self.stc_path, list_eeg_name[idx])
            if not os.path.exists(stc_subject_path):
                os.makedirs(stc_subject_path)

            eeg = self.data_io.load_eegs(eeg_path, self.extension, self.datatype)
            eeg_info = eeg.info

            if self.use_anatomy == "individual":
                print("\nUsing individual anatomies")
                subject = list_eeg_name[idx]
                src, bem, trans = self.individual_mri(subject, eeg_info)
                self.export_src_bem_trans('fsaverage', src, bem, trans)
                stc_file = self.compute_stc(src, bem, trans, eeg, eeg_info)
                # Morph to fsaverage
                src_morph = mne.read_source_spaces(src)
                morph = mne.compute_source_morph(src_morph, subject_from=subject, subject_to='fsaverage',
                                                 subjects_dir=self.individual_subjects_dir, spacing=self.spacing[-1])
                morph.save(os.path.join(self.individual_subjects_dir, subject, subject + '-morph.h5'), overwrite=True)
                stc_file = morph.apply(stc_file)
            else:
                print("\nUsing default template brain - fsaverage")
                src, bem, trans = self.load_average_mri(eeg_info)
                self.export_src_bem_trans('fsaverage', src, bem, trans)
                stc_file = self.compute_stc(src, bem, trans, eeg, eeg_info)

            print(f"\nExporting Source Time Courses: {eeg_name}")
            self.stc_write(stc_subject_path, stc_file)

    def find_t_coeff(self, sample, maps):
        """Find T coefficients."""
        return np.linalg.lstsq(maps, sample, rcond=None)[0]

    def first_regression(self, sensor_time_series, maps):
        """Perform the first regression to extract T coefficients."""
        return np.apply_along_axis(self.find_t_coeff, 0, sensor_time_series, maps).transpose()

    def second_regression(self, t_coeff, stc_data):
        """Perform the second regression to get beta coefficients."""
        return np.linalg.solve(t_coeff.T @ t_coeff, t_coeff.T @ stc_data)

    def run_tess(self, stc_data, eeg_data, nperm=2000):
        """Run the TESS algorithm."""
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
        for ii in range(0, nperm):
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
        """Averages the source data over times matched with each microstate segment."""

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
            for m in all_sources_dict.keys():
                all_sources_dict[m] = np.vstack(all_sources_dict[m])

            return all_sources_dict

    def identify_microstates_sources(self, source_method):
        """Identify microstate source localization for multiple EEG files."""

        list_eeg_path, list_eeg_name = self.data_io.find_data(self.preprocessed_data_path, '.set', '*')
        for idx, (eeg_path, eeg_name) in enumerate(zip(list_eeg_path, list_eeg_name)):
            print(f"\nLoading Source Time Courses: {eeg_name}")
            stc_subject_path = os.path.join(self.stc_path, eeg_name)
            stc_file = self.stc_read(stc_subject_path)
            stc_data = stc_file[0].data.T

            eeg = self.data_io.load_eegs(eeg_path, self.extension, self.datatype)
            eeg_data = eeg.get_data()

            if source_method == 'tess':
                p_values, z_scores, filtered_z_scores = self.run_tess(stc_data, eeg_data, self.nperm)
                print('\nExtracting sources associated with each microstate',
                      '\nusing the topographic electrophysiological state source-imaging (TESS) algorithm ...')
                tess_subject_path = os.path.join(self.tess_path, list_eeg_name[idx])
                if not os.path.exists(tess_subject_path):
                    os.makedirs(tess_subject_path)
                print('\nTess Completed on File ...')
                np.save(os.path.join(tess_subject_path, f'{list_eeg_name[idx]}-filtered_zscore.npy'), filtered_z_scores)
                np.save(
                    os.path.join(tess_subject_path, f'{list_eeg_name[idx]}-zscore.npy'),
                    z_scores)
                np.save(
                    os.path.join(tess_subject_path, f'{list_eeg_name[idx]}-p_values.npy'),
                    p_values)
            elif source_method == 'avg':
                # Average sources over times matched with each microstate
                print('\nAveraging sources over times matched with each microstate ...')
                avg_subject_path = os.path.join(self.avg_sources_path, list_eeg_name[idx])
                if not os.path.exists(avg_subject_path):
                    os.makedirs(avg_subject_path)

                all_sources_dict = self.avg_sources(self.segmentation_path, stc_data)
                print(f'\nExporting the averaged microstate sources for subject {list_eeg_name[idx]}')
                for m, array_data in all_sources_dict.items():
                    filename = os.path.join(avg_subject_path, f"{list_eeg_name[idx]}_{m}.npy")
                    np.save(filename, array_data)