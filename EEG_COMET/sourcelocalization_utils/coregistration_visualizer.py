"""Visualization helpers for coregistration overlays using MNE."""

import os

import mne


class CoregistrationVisualizer:
    """Display alignment between MRI and sensor geometry."""

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

    def show_coreg(self, subject, eeg_info):
        """Display head, sensor, and source space alignment in 3D using MNE."""
        subject_dir = os.path.join(self.subjects_dir, subject)
        trans_path = self.find_file(subject_dir, "-trans.fif")
        fid_path = self.find_file(subject_dir, "-fiducials.fif")

        if trans_path is None:
            raise FileNotFoundError(f"No '-trans.fif' file found for subject {subject}")
        trans = mne.read_trans(trans_path)

        mri_fiducials = fid_path is not None

        return mne.viz.plot_alignment(
            info=eeg_info,
            trans=trans,
            subject=subject,
            subjects_dir=self.subjects_dir,
            surfaces="auto",
            coord_frame="auto",
            eeg=["original", "projected"],
            dig=False,
            mri_fiducials=mri_fiducials,
            show_axes=False,
            sensor_colors=None,
        )
