
import os.path
import mne
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QFileDialog
import pyvista as pv
from pyvistaqt import QtInteractor


# TODO: embed 3d visualization into the window
class SourceVisualizationDialog(QDialog):
    def __init__(self, context, parent=None, tbx=None):
        super(SourceVisualizationDialog, self).__init__(parent)

        self.tbx = tbx
        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("SourceVisualizationWindow.ui"), self)

        self.ui.setWindowTitle("Visualization of the localized sources")

        self.ui.subjects_dir_button.clicked.connect(self.locate_subjects_dir)
        self.ui.plot_reconstruction_button.clicked.connect(self.show_brain)
        self.ui.plot_alignment_button.clicked.connect(self.show_alignment)

        self.plotter = QtInteractor(self)
        self.PV_Layout.addWidget(self.plotter)

        self.resize(1000, 800)

    def locate_subjects_dir(self):
        fname = QFileDialog.getExistingDirectory(self, "Select the folder containing raw data")
        self.subjects_dir = fname
        self.ui.subjects_dir_lineedit.setText(fname)

        list_subjects = [folder for folder in os.listdir(self.subjects_dir) if os.path.isdir(os.path.join(self.subjects_dir, folder))]

        for i in range(len(list_subjects)):
            self.ui.subjects_list.addItem(str(list_subjects[i]))

    def show_brain(self):
        subject = 'MUSC-003'
        subjects_dir = 'C:/Users/amin_/OneDrive - Simon Fraser University (1sfu)/MICROSTATES_RESULTS_HEALTHY/test_tess_microstate_paper/subjects_dir/'

        src = mne.read_source_spaces(src_file)
        src.plot(subjects_dir=subjects_dir)


    def show_alignment(self):

        subject = 'MUSC-003'
        subjects_dir = 'C:/Users/amin_/OneDrive - Simon Fraser University (1sfu)/MICROSTATES_RESULTS_HEALTHY/test_tess_microstate_paper/subjects_dir/'
        src_file = os.path.join('C:/Users/amin_/OneDrive - Simon Fraser University (1sfu)/MICROSTATES_RESULTS_HEALTHY/test_tess_microstate_paper/subjects_dir/sub_test/MUSC-003-ico4-src.fif')
        trans_file = os.path.join('C:/Users/amin_/OneDrive - Simon Fraser University (1sfu)/MICROSTATES_RESULTS_HEALTHY/test_tess_microstate_paper/subjects_dir/sub_test/MUSC-003-trans.fif')

        src = mne.read_source_spaces(src_file)
        trans = mne.read_trans(trans_file)

        raw_dir = 'C://Users//amin_//OneDrive - Simon Fraser University (1sfu)//MICROSTATES_RESULTS_HEALTHY//test_tess_microstate_paper//raw_test//musc-m003_v3_notime_tms-rest_1mv_sp_lpfc_clean.set'
        raw = mne.read_epochs_eeglab(raw_dir)
        raw = raw.resample(250)
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()
        raw_info = raw.info
        """
        fig = mne.viz.plot_alignment(
        info=raw_info, subject=subject,
        subjects_dir=subjects_dir,
        src=src, eeg=['original', 'projected'], trans=trans,
        show_axes=True, mri_fiducials=True)
        """
        src = mne.read_source_spaces(src_file)

        meshes = []
        vertices = np.array(src[0]['rr'])
        triangles = np.array(src[0]['tris'])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh = pv.PolyData(vertices, triangles)
        meshes.append(mesh)
        vertices = np.array(src[1]['rr'])
        triangles = np.array(src[1]['tris'])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh = pv.PolyData(vertices, triangles)
        meshes.append(mesh)
        for mesh in meshes:
            self.plotter.add_mesh(mesh)

        #mne_figure = src.plot(subjects_dir=subjects_dir)
        #self.plotter.add_mesh(pyvista_mesh)

        """
        
        Brain = mne.viz.get_brain_class()
        brain = Brain(subject, hemi="split", surf="inflated", cortex="high_contrast",
                      subjects_dir=subjects_dir, figure=fig)
        brain.add_annotation("aparc.a2009s", borders=False)
        """
