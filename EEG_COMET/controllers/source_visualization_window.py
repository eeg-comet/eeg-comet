
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
        self.comet = comet_tbx

        if self.comet.use_anatomy != "fsaverage":
            self.subjects_dir = self.comet.individual_subjects_dir
        else:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.subjects_dir = os.path.dirname(fs_dir)
        print(self.subjects_dir)

        self.source_visualizer = SourceVisualizer(
            self.comet.anatomy_subjects_dir,
            self.comet.spacing,
            self.comet.localized_sources_path
        )

        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("SourceVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the localized sources")

        for m in self.comet.micro_labels:
            self.ui.microstate_combobox.addItem(m)

        self.plotter = QtInteractor(self)
        self.Figure_Layout.addWidget(self.plotter)

        self.locate_subjects_dir()
        self.setup_connections()
        self.resize(1000, 800)

    def setup_connections(self):
        self.ui.stc_time_slider.item.valueChanged.connect(self.source_localization_controller)
        self.ui.show_stc_button.clicked.connect(self.show_sources)
        self.ui.show_microstate_sources_button.clicked.connect(self.show_microstate_sources)
        self.ui.subjects_list.itemSelectionChanged.connect(self.handle_new_file_selection)
        # self.ui.subjects_list.itemClicked.connect(self.show_meshes)

    def source_localization_controller(self):
        self.ui.stc_time_input.setText(str(self.ui.stc_time_slider.value()))

    def locate_subjects_dir(self):
        self.ui.subjects_list.clear()
        subjects_dir = os.path.join(self.comet.localized_sources_path, 'stc')
        self.list_subjects = [
            folder for folder in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, folder))
        ]
        for list_subject in self.list_subjects:
            self.ui.subjects_list.addItem(str(list_subject))
        self.ui.subjects_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def handle_new_file_selection(self):
        self.ui.microstates_list_combobox.clear()
        selected_items = self.ui.subjects_list.selectedItems()
        self.selected_subjects = [item.text() for item in selected_items]
        self.tess_dir = os.path.join(self.comet.localized_sources_path, 'tess_sources')
        list_tess_subjects = [
            folder for folder in self.selected_subjects if os.path.isdir(os.path.join(self.tess_dir, folder))
        ]
        self.avg_dir = os.path.join(self.comet.localized_sources_path, 'avg_sources')
        list_avg_subjects = [
            folder for folder in self.selected_subjects if os.path.isdir(os.path.join(self.avg_dir, folder))
        ]

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
        initial_time = self.ui.microstate_combobox.currentIndex()
        selected_items = self.ui.subjects_list.selectedItems()
        spacing = self.comet.spacing
        self.source_visualizer.plot_sources(selected_items, source_mode, initial_time, spacing)

    def show_microstate_sources(self):
        return
