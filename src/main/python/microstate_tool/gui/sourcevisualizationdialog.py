
import os.path
import mne
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QFileDialog, QAbstractItemView, QSizePolicy, QWidget
from functions.data_utils.data_io import DataIO
import pyvista as pv
from pyvistaqt import QtInteractor, BackgroundPlotter

# TODO: *** fix the compatibility issue with the individual anatomy

class SourceVisualizationDialog(QDialog):
    def __init__(self, context, parent=None, comet_tbx=None):
        super(SourceVisualizationDialog, self).__init__(parent)

        self.list_subjects = None
        self.comet_tbx = comet_tbx

        if not self.comet_tbx.use_anatomy == "fsaverage":
            self.subjects_dir = self.comet_tbx.individual_subjects_dir
        else:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.subjects_dir = os.path.dirname(fs_dir)
        print(self.subjects_dir)

        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("SourceVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the localized sources")

        for m in self.comet_tbx.micro_labels:
            self.ui.microstate_combobox.addItem(m)

        self.plotter = QtInteractor(self)
        self.Figure_Layout.addWidget(self.plotter)

        self.locate_subjects_dir()
        self.ui.plot_sources_button.clicked.connect(self.plot_sources)
        self.ui.subjects_list.itemSelectionChanged.connect(self.handle_new_file_selection)
        self.ui.subjects_list.itemClicked.connect(self.show_brain)

        self.resize(1000, 800)

    def locate_subjects_dir(self):
        self.ui.subjects_list.clear()
        subjects_dir = os.path.join(self.comet_tbx.localized_sources_path, 'stc')
        list_subjects = [folder for folder in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, folder))]
        self.list_subjects = list_subjects

        for i in range(len(list_subjects)):
            self.ui.subjects_list.addItem(str(list_subjects[i]))

        self.ui.subjects_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def handle_new_file_selection(self):
        self.ui.microstates_list_combobox.clear()
        selected_items = self.ui.subjects_list.selectedItems()
        self.selected_subjects = [item.text() for item in selected_items]

        self.tess_dir = os.path.join(self.comet_tbx.localized_sources_path, 'tess_sources')
        list_tess_subjects = [folder for folder in self.selected_subjects if os.path.isdir(os.path.join(self.tess_dir, folder))]

        self.avg_dir = os.path.join(self.comet_tbx.localized_sources_path, 'avg_sources')
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

        src_filename = 'fsaverage-' + self.comet_tbx.spacing[:-1] + '-' + self.comet_tbx.spacing[-1] + '-src.fif'
        src_filepath = os.path.join(self.subjects_dir, 'fsaverage' ,'bem', src_filename)
        src = mne.read_source_spaces(src_filepath, verbose=False)
        meshes = []
        vertices = np.array(src[0]['rr'])
        triangles = np.array(src[0]['tris'])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh1 = pv.PolyData(vertices, triangles)
        meshes.append(mesh1)
        vertices = np.array(src[1]['rr'])
        triangles = np.array(src[1]['tris'])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh2 = pv.PolyData(vertices, triangles)
        meshes.append(mesh2)

        # Add the source estimate meshes on top of the existing meshes
        # for mesh in meshes:
        #    self.plotter.add_mesh(mesh)
        self.plotter.add_mesh(mesh1, color="red", opacity=0.7)
        self.plotter.add_mesh(mesh2, color="blue", opacity=0.7)

        # Update the plotter to render the changes
        self.plotter.update()

    def normalize_matrix(self, matrix):
        return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

    def normalize_stc_data(self, stc_data):
        return np.array([self.normalize_matrix(row) for row in stc_data])

    def preprocess_data(self, mode):
        if mode in ["tess_filtered", "tess_raw"]:
            self.tess_path = os.path.join(self.comet_tbx.localized_sources_path, 'tess_sources')
            pattern = '*-zscore*' if mode == "tess_raw" else '*filtered_zscore*'
            stc_list_path, _ = DataIO().find_data(self.tess_path, extension='.npy', pattern=pattern)
        elif mode in ["avg_filtered", "avg_raw"]:
            # TODO: complete this
            self.avg_sources_path = os.path.join(self.comet_tbx.localized_sources_path, 'avg_sources')
            stc_list_path, _ = DataIO().find_data(self.tess_path, extension='.npy', pattern='avg')

        selected_items = self.ui.subjects_list.selectedItems()
        selected_subjects = [item.text() for item in selected_items]
        stc_list_selected_paths = [full_path for full_path in stc_list_path if
                                   any(folder in full_path for folder in selected_subjects)]
        print(stc_list_selected_paths)

        stc_sum = np.zeros_like(np.load(stc_list_selected_paths[0])) if stc_list_selected_paths else []
        for stc_path in stc_list_selected_paths:
            stc_data = np.load(stc_path)
            stc_data = self.normalize_stc_data(stc_data)
            stc_sum += stc_data

        stc_avg = stc_sum / len(stc_list_selected_paths) if stc_list_selected_paths else np.zeros_like(stc_sum)

        mystc = mne.SourceEstimate(stc_avg,
                                   vertices=[np.arange(stc_avg.shape[0] // 2), np.arange(stc_avg.shape[0] // 2)],
                                   tmin=0, tstep=1, subject='fsaverage')

        return mystc, stc_avg

    def plot_sources(self):
        # [TODO] Need to add plots within the main visualization window

        if self.ui.microstates_list_combobox.currentText() == "TESS - Filtered Z-Scores":
            [mystc, stc_avg] = self.preprocess_data(mode="tess_filtered")
        elif self.ui.microstates_list_combobox.currentText() == "TESS - Raw Z-Scores":
            [mystc, stc_avg] = self.preprocess_data(mode="tess_raw")
        elif self.ui.microstates_list_combobox.currentText() == "AVG - Filtered Z-Scores":
            [mystc, stc_avg] = self.preprocess_data(mode="avg_filtered")
        elif self.ui.microstates_list_combobox.currentText() == "AVG - Raw Z-Scores":
            [mystc, stc_avg] = self.preprocess_data(mode="avg_raw")

        # Initial time refers to which microstate is displayed (i.e. 0 = microstate A, 6 = microstate G)
        initial_time = self.ui.microstate_combobox.currentIndex()
        print(initial_time)
        #"""
        brain = mystc.plot(subjects_dir=self.subjects_dir,
                           initial_time=initial_time,
                           views=['lateral', 'medial'],
                           cortex='low_contrast',
                           hemi='split',
                           surface='inflated',
                           clim=dict(kind='value', lims=[0, 0.5, 1]),
                           time_viewer=False,
                           background='white',
                           spacing=self.comet_tbx.spacing,
                           size=(800, 800),
                           colormap='jet',
                           smoothing_steps=15,
                           colorbar=True
                           )
        #"""

        # brain.add_annotation("HCPMMP1_combined", borders=2)
        # brain.save_image(filename=os.path.join(parent_path, f'avg_filtered_zscore_M_{initial_time}.png'))

