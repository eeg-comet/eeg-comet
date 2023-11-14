
import os.path
import mne
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QFileDialog, QAbstractItemView
import pyvista as pv
from pyvistaqt import QtInteractor

# TODO: *** not completed
# TODO: embed 3d visualization into the window
class SourceVisualizationDialog(QDialog):
    def __init__(self, context, parent=None, tbx=None):
        super(SourceVisualizationDialog, self).__init__(parent)

        self.list_subjects = None
        self.subjects_dir = None
        self.tbx = tbx
        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("SourceVisualizationWindow.ui"), self)

        self.ui.setWindowTitle("Visualization of the localized sources")

        # These functions need to be updated before re-enabling in the UI: show_brain, show_alignment
        # self.ui.plot_reconstruction_button.clicked.connect(self.show_brain)
        # self.ui.plot_alignment_button.clicked.connect(self.show_alignment)

        self.ui.plot_sources_button.clicked.connect(self.plot_sources)
        self.ui.load_stc_button.clicked.connect(self.locate_subjects_dir)
        self.ui.subjects_list.itemSelectionChanged.connect(self.handle_new_file_selection)

        self.plotter = QtInteractor(self)
        self.PV_Layout.addWidget(self.plotter)

        self.resize(1000, 800)

    def locate_subjects_dir(self):
        self.ui.subjects_list.clear()
        self.subjects_dir = os.path.join(self.tbx.localized_sources_path, 'stc')

        list_subjects = [folder for folder in os.listdir(self.subjects_dir) if os.path.isdir(os.path.join(self.subjects_dir, folder))]
        self.list_subjects = list_subjects

        for i in range(len(list_subjects)):
            self.ui.subjects_list.addItem(str(list_subjects[i]))

        self.ui.subjects_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def handle_new_file_selection(self):
        # Clear the combo box before adding new items
        self.ui.microstates_list_combobox.clear()
        # Collect selected items in subjects_list:
        selected_items = self.ui.subjects_list.selectedItems()
        self.selected_subjects = [item.text() for item in selected_items]

        self.tess_dir = os.path.join(self.tbx.localized_sources_path, 'tess_sources')
        list_tess_subjects = [folder for folder in self.selected_subjects if os.path.isdir(os.path.join(self.tess_dir, folder))]

        self.avg_dir = os.path.join(self.tbx.localized_sources_path, 'avg_sources')
        list_avg_subjects = [folder for folder in self.selected_subjects if os.path.isdir(os.path.join(self.avg_dir, folder))]

        plottable = False

        if len(list_tess_subjects) == len(self.selected_subjects):
            self.ui.microstates_list_combobox.addItems(["TESS - Filtered Z-Scores", "TESS - Raw Z-Scores"])
            plottable = True

        if len(list_avg_subjects) == len(self.selected_subjects):
            self.ui.microstates_list_combobox.addItems(["AVG - Filtered Z-Scores", "AVG - Raw Z-Scores"])
            plottable = True

        if not plottable:
            self.ui.microstates_list_combobox.addItems(["--Data Unavailable--"])
        
    def show_brain(self):
        # Currently, this function is not in use
        subject = 'MUSC-003'
        subjects_dir = 'C:/Users/amin_/OneDrive - Simon Fraser University (1sfu)/MICROSTATES_RESULTS_HEALTHY/test_tess_microstate_paper/subjects_dir/'

        src = mne.read_source_spaces(src_file)
        src.plot(subjects_dir=subjects_dir)


    def show_alignment(self):
        # Currently, this function is not in use
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

    def preprocess_data(self, mode):
        # [TODO] Need to complete this function

        if mode == "tess_filtered":
            todo = 1
            # [TODO] Load the tess filtered z-score filenames from the selected self.selected_subjects
            # z_filt_files = ... '-filtered_zscore.npy'))
        elif mode == "tess_raw":
            todo = 1
            # [TODO] Load the tess raw z-score filenames from the selected self.selected_subjects
            # z_filt_files = ... '-zscore.npy'))
        elif mode == "avg_filtered":
            todo = 1
            # [TODO] Load the average filtered z-score filenames from the selected self.selected_subjects
        elif mode == "avg_raw":
            todo = 1
            # [TODO] Load the average raw z-score filenames from the selected self.selected_subjects

        # [TODO] Need to load multiple data files and average them (see function normalize_and_average_data()
        #  in sourcelocalization_utils/run_tess_and_visualize.py for reference)...
        # ... z_filt_avg_mean_subtracted = z_filt_avg - np.mean(z_filt_avg, axis=0)

        # [TODO] Create an STC object
        # mystc = mystc = mne.SourceEstimate(z_filt_avg_mean_subtracted.T,
        #                                vertices=[np.arange(z_filt_avg_mean_subtracted.shape[1] / 2),
        #                                np.arange(z_filt_avg_mean_subtracted.shape[1] / 2)],
        #                                tmin=0, tstep=1, subject='fsaverage')

        return mystc, z_filt_avg_mean_subtracted

    def plot_sources(self):
        # [TODO] Need to add plots within the main visualization window

        if self.ui.microstates_list_combobox.currentText() == "TESS - Filtered Z-Scores":
            [mystc, z_filt] = self.preprocess_data(mode="tess_filtered")
        elif self.ui.microstates_list_combobox.currentText() == "TESS - Raw Z-Scores":
            [mystc, z_filt] = self.preprocess_data(mode="tess_raw")
        elif self.ui.microstates_list_combobox.currentText() == "AVG - Filtered Z-Scores":
            [mystc, z_filt] = self.preprocess_data(mode="avg_filtered")
        elif self.ui.microstates_list_combobox.currentText() == "AVG - Raw Z-Scores":
            [mystc, z_filt] = self.preprocess_data(mode="avg_raw")

        # Load fsaverage files if they don't exist
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)

        # Initial time refers to which microstate is displayed (i.e. 0 = microstate A, 6 = microstate G)
        # Loop over all microstates by changing the initial time
        for initial_time in range(0, z_filt.shape[0]):
            brain = mystc.plot(subjects_dir=os.path.dirname(fs_dir),
                               initial_time=initial_time,
                               views=['lateral', 'medial'],
                               cortex='low_contrast',
                               hemi='split',
                               surface='inflated',
                               clim=dict(kind='value', lims=[-0.15, 0, 0.15]),
                               time_viewer=False,
                               background='white',
                               spacing='ico4',
                               size=(1600, 800),
                               colormap='jet',
                               smoothing_steps=15,
                               colorbar=True)

            # brain.add_annotation("HCPMMP1_combined", borders=2)
            brain.save_image(filename=os.path.join(parent_path, f'avg_filtered_zscore_M_{initial_time}.png'))



