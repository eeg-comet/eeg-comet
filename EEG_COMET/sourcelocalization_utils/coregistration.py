"""Coregistration helpers for aligning EEG sensors and MRI anatomy."""

import os

import mne


class Coregistration:
    """Operations to run manual or automatic coregistration."""

    def __init__(self, subjects_dir):
        """Initialize with directory containing subject MRI data."""
        self.subjects_dir = subjects_dir

    @staticmethod
    def find_file(subject_dir, suffix):
        """Return first file in directory tree ending with the given suffix."""
        file_gen = (
            os.path.join(root, file)
            for root, dirs, files in os.walk(subject_dir)
            for file in files
            if file.endswith(suffix)
        )
        return next(file_gen, None)

    def manual_coreg(self, subject):
        """Open MNE coregistration GUI for manual alignment."""
        return mne.gui.coregistration(
            subject=subject, subjects_dir=self.subjects_dir, fullscreen=True, verbose="ERROR"
        )

    def auto_coreg(self, subject, eeg_info):
        """Automatically coregister MRI data with the subject's head shape."""
        subject_dir = os.path.join(self.subjects_dir, subject)
        
        # Handle fsaverage template specially
        if subject == "fsaverage":
            # Use standard montage with fsaverage
            # The built-in 'fsaverage' transform handles alignment automatically
            # No need to save a transform file - just use 'fsaverage' string directly
            # in forward solution and source localization functions
            return
        
        # Standard coregistration for individual subjects
        fid_path = self.find_file(subject_dir, "-fiducials.fif")
        if fid_path is not None:
            fiducials = mne.coreg.get_mni_fiducials(subject=subject, subjects_dir=self.subjects_dir)
        else:
            fiducials = "estimated"
        coreg = mne.coreg.Coregistration(
            info=eeg_info, subject=subject, subjects_dir=self.subjects_dir, fiducials=fiducials
        )
        coreg.fit_fiducials()
        coreg.fit_icp(n_iterations=6, nasion_weight=2.0)
        coreg.omit_head_shape_points(distance=5.0 / 1000)
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
        mne.write_trans(
            fname=os.path.join(subject_dir, "mri", "transforms", f"{subject}-trans.fif"),
            trans=coreg.trans,
            overwrite=True,
            verbose="ERROR",
        )
