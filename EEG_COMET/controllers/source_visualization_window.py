
import os.path
import mne
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QAbstractItemView
from pyvistaqt import QtInteractor, BackgroundPlotter
from sourcelocalization_utils.source_visualizer import SourceVisualizer

# TODO: *** fix the compatibility issue with the individual anatomy


class SourceVisualizationWindow(QDialog):
    def __init__(self, context, parent=None, comet_tbx=None):
        super(SourceVisualizationWindow, self).__init__(parent)

        self.list_subjects = None
        self.comet_tbx = comet_tbx

        if not self.comet_tbx.use_anatomy == "fsaverage":
            self.subjects_dir = self.comet_tbx.individual_subjects_dir
        else:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.subjects_dir = os.path.dirname(fs_dir)
        print(self.subjects_dir)

        self.source_visualizer = SourceVisualizer(
            self.comet_tbx.anatomy_subjects_dir,
            self.comet_tbx.spacing,
            self.comet_tbx.localized_sources_path
        )

        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("SourceVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the localized sources")

        for m in self.comet_tbx.micro_labels:
            self.ui.microstate_combobox.addItem(m)

        self.plotter = QtInteractor(self)
        self.Figure_Layout.addWidget(self.plotter)

        self.locate_subjects_dir()
        self.ui.plot_sources_button.clicked.connect(self.show_sources)
        self.ui.subjects_list.itemSelectionChanged.connect(self.handle_new_file_selection)
        self.ui.subjects_list.itemClicked.connect(self.show_meshes)

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

    def show_meshes(self):

        meshes = self.source_visualizer.export_meshes()
        # Add the source estimate meshes on top of the existing meshes
        for mesh in meshes:
            self.plotter.add_mesh(mesh, opacity=0.7)
        # self.plotter.add_mesh(mesh1, color="red", opacity=0.7)
        # self.plotter.add_mesh(mesh2, color="blue", opacity=0.7)

        # Update the plotter to render the changes
        self.plotter.update()

    def show_sources(self):

        if self.ui.microstates_list_combobox.currentText() == "TESS - Filtered Z-Scores":
            source_mode = "tess_filtered"
        elif self.ui.microstates_list_combobox.currentText() == "TESS - Raw Z-Scores":
            source_mode = "tess_raw"
        elif self.ui.microstates_list_combobox.currentText() == "AVG - Filtered Z-Scores":
            source_mode = "avg_filtered"
        elif self.ui.microstates_list_combobox.currentText() == "AVG - Raw Z-Scores":
            source_mode = "avg_raw"

        initial_time = self.ui.microstate_combobox.currentIndex()
        selected_items = self.ui.subjects_list.selectedItems()
        spacing = self.comet_tbx.spacing
        self.source_visualizer.plot_sources(selected_items, source_mode, initial_time, spacing)
