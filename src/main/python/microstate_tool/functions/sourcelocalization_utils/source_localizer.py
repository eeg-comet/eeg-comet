"""
Description: This class facilitates the localization of neural sources from EEG data.
It handles both the reading and writing of source time courses, as well as exporting
necessary transformations and BEM solutions. Different methods for coregistration
using individual or template MRIs are also available.

"""

import os
import numpy as np
import mne
from functions.data_utils.data_io import DataIO


class SourceLocalizer:
    def __init__(self, subjects_dir, stc_path, preprocessed_data_path, extension, datatype, spacing, inv_method):
        self.subjects_dir = subjects_dir
        self.stc_path = stc_path
        self.preprocessed_data_path = preprocessed_data_path
        self.extension = extension
        self.datatype = datatype
        self.spacing = spacing
        self.inv_method = inv_method
        self.data_io = DataIO()

    def stc_write(self, stc_data_subject_path, stc_file):
        """Write source time series to disk."""
        for idx, stc in enumerate(stc_file):
            filename = f'stc_{idx}'
            filepath = os.path.join(stc_data_subject_path, filename)
            stc.save(filepath, ftype='h5')

    def stc_read(self, stc_data_subject_path):
        """Read source time series from disk."""
        stc_list, _ = self.data_io.find_data(stc_data_subject_path, '.stc', pattern='*')
        stc_file = [mne.read_source_estimate(stc_path) for stc_path in stc_list]
        return stc_file

    def export_src_bem_trans(self, subject, src, bem, trans):
        """Export source space, BEM, and coregistration transformations."""
        mne.write_source_spaces(os.path.join(self.subjects_dir, subject, f'{subject}-ico4-src.fif'), src, overwrite=True)
        mne.write_bem_solution(os.path.join(self.subjects_dir, subject, f'{subject}-bem.fif'), bem, overwrite=True)
        mne.write_trans(os.path.join(self.subjects_dir, subject, f'{subject}-trans.fif'), trans, overwrite=True)

    def load_average_mri(self, eeg_info, spacing='ico5'):
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
        if not spacing == 'ico5':
            src = mne.setup_source_space(subject,
                                         spacing=spacing,
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


    def individual_mri(self, subjects_dir, subject, raw_info):
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
        bem_surfaces = mne.make_bem_model(subject=subject, subjects_dir=self.subjects_dir, ico=4)

        # bem_path = os.path.join(subjects_dir, subject, 'bem', f'{subject}-bem-sol.fif')
        # mne.write_bem_surfaces(bem_path, bem_surfaces, overwrite=True)

        # Make BEM solution
        bem = mne.make_bem_solution(bem_surfaces)

        return src, bem, trans


    def extract_forward_transform(self, src, bem, trans, raw, raw_info):
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
            stc = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2,
                                                     method=self.inv_method, pick_ori=None, verbose=True)
            source_time_series = stc.data
        elif self.datatype == 'epoched':
            # Process epoched data
            stc = mne.minimum_norm.apply_inverse_epochs(raw, inverse_operator, lambda2,
                                                        method=self.inv_method, pick_ori=None, verbose=True)
            # Concatenate the source data for all time points across all trials
            n_trials = len(stc)
            n_sources = stc[0].data.shape[0]
            n_timepoints = stc[0].data.shape[1]  # Assuming all stc objects have the same number of time points
            source_time_series = np.empty((n_trials, n_sources, n_timepoints))
            for tr in range(n_trials):
                source_time_series[tr, :, :] = stc[tr].data

        return stc  # , source_time_series


    def run_source_localization(self):
        """Perform source localization for multiple EEG files."""
        list_eeg_path, list_eeg_name = self.data_io.find_data(self.preprocessed_data_path, '.set', '*')

        for idx, (eeg_path, eeg_name) in enumerate(zip(list_eeg_path, list_eeg_name)):
            print(f"Source Localizing {eeg_name} ({idx + 1}/{len(list_eeg_path)})")

            eeg = self.data_io.load_eegs(eeg_path, self.extension, self.datatype)
            eeg_info = eeg.info

            src, bem, trans = self.load_average_mri(eeg_info)
            self.export_src_bem_trans('fsaverage', src, bem, trans)

            stc_file = self.extract_forward_transform(src, bem, trans, eeg, eeg_info)
            stc_subject_path = os.path.join(self.stc_path, eeg_name)
            stc_file.save(stc_subject_path, ftype='h5', overwrite=True)
