import os
import mne


class CoregistrationVisualizer:
    def __init__(self, subjects_dir):
        """
        Initialize with the directory containing subject MRI data.
        """
        self.subjects_dir = subjects_dir

    @staticmethod
    def find_file(subject_dir, suffix):
        """
        Finds the first file in the given directory (and its subdirectories) that ends with the specified suffix.
        """
        file_gen = (os.path.join(root, file) for root, dirs, files in os.walk(subject_dir) for file in files if
                     file.endswith(suffix))
        file_path = next(file_gen, None)
        return file_path

    def show_coreg(self, subject, eeg_info):
        """
        Display head, sensor, and source space alignment in 3D using MNE-Python.
        """
        subject_dir = os.path.join(self.subjects_dir, subject)
        trans_path = self.find_file(subject_dir, '-trans.fif')
        fid_path = self.find_file(subject_dir, '-fiducials.fif')

        if trans_path is None:
            raise FileNotFoundError(f"No '-trans.fif' file found for subject {subject}")
        trans = mne.read_trans(trans_path)

        if fid_path is None:
            mri_fiducials = False
        else:
            mri_fiducials = True

        pv_fig = mne.viz.plot_alignment(
            info=eeg_info,
            trans=trans,
            subject=subject,
            subjects_dir=self.subjects_dir,
            surfaces='auto',
            coord_frame='auto',
            eeg=['original', 'projected'],
            dig=False,
            mri_fiducials=mri_fiducials,
            show_axes=False,
            sensor_colors=None
        )
        return pv_fig
