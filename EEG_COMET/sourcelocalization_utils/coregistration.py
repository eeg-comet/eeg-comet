import os
import mne


class Coregistration:
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

    def manual_coreg(self, subject):
        """
        Manually coregister an MRI with the subject's head shape using MNE-Python's interactive GUI.
        """
        return mne.gui.coregistration(
            subject=subject,
            subjects_dir=self.subjects_dir,
            fullscreen=True,
            verbose='ERROR')

    def auto_coreg(self, subject, eeg_info):
        """
        Automated approach to coregistering MRI data with a subject’s head shape.
        """
        subject_dir = os.path.join(self.subjects_dir, subject)
        fid_path = self.find_file(subject_dir, '-fiducials.fif')
        if fid_path is not None:
            fiducials = mne.coreg.get_mni_fiducials(
                subject=subject,
                subjects_dir=self.subjects_dir
            )
        else:
            fiducials = "estimated"
        coreg = mne.coreg.Coregistration(
            info=eeg_info,
            subject=subject,
            subjects_dir=self.subjects_dir,
            fiducials=fiducials
        )
        coreg.fit_fiducials()
        coreg.fit_icp(n_iterations=6, nasion_weight=2.0)
        coreg.omit_head_shape_points(distance=5.0 / 1000)
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
        mne.write_trans(
            fname=os.path.join(subject_dir, 'mri', 'transforms', f'{subject}-trans.fif'),
            trans=coreg.trans,
            overwrite=True,
            verbose='ERROR'
        )
