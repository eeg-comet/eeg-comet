
import os.path
import mne
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QListWidgetItem, QSizePolicy
from sourcelocalization_utils.coregistration_visualizer import CoregistrationVisualizer
from sourcelocalization_utils.coregistration import Coregistration


class CoregistrationWindow(QDialog):
    def __init__(self, context, parent=None, tbx=None):
        super(CoregistrationWindow, self).__init__(parent)

        self.comet = tbx

        if self.comet.use_anatomy != "fsaverage":
            self.subjects_dir = self.comet.individual_subjects_dir
        else:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.subjects_dir = os.path.dirname(fs_dir)

        self.coregistration = Coregistration(self.subjects_dir)
        self.coregistration_visualizer = CoregistrationVisualizer(self.subjects_dir)

        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("CoregistrationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the head, sensor, and source space alignment")

        self.ui.manual_coreg_button.clicked.connect(self.manual_coregistration)
        self.ui.auto_coreg_button.clicked.connect(self.automatic_coregistration)
        self.ui.plot_alignment_button.clicked.connect(self.show_alignment)
        self.ui.subjects_list.itemSelectionChanged.connect(self.handle_new_file_selection)

        self.populate_subjects_list()

        self.plotter = None
        self.resize(1000, 800)

    def populate_subjects_list(self):
        """
        Populate the subjects list with folder names inside subjects_dir.
        """
        subject_folders = [f for f in os.listdir(self.subjects_dir) if
                           os.path.isdir(os.path.join(self.subjects_dir, f))]
        self.ui.subjects_list.clear()
        for folder in subject_folders:
            self.ui.subjects_list.addItem(QListWidgetItem(folder))

    def handle_new_file_selection(self):
        selected_item = self.ui.subjects_list.currentItem()
        self.selected_subject = selected_item.text()
        self.ui.subject_id_label.setText(self.selected_subject)

    def clean_figure_layout(self):
        """
        Clear and delete all widgets from the figure layout to reset it.
        """
        while self.ui.Figure_Layout.count() > 0:
            item = self.ui.Figure_Layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def manual_coregistration(self):
        """
        Load the MNE-Python's GUI for manual coregistration.
        """
        self.coregistration.manual_coreg(self.selected_subject)

    def automatic_coregistration(self):
        """
        Perform automatic coregistration.
        """
        self.coregistration.auto_coreg(self.selected_subject, self.comet.eeg_info)

    def show_alignment(self):
        """
        Updates the 3D alignment visualization inside the already initialized plotter.
        """
        if self.plotter is not None:
            self.clean_figure_layout()
        self.plotter = self.coregistration_visualizer.show_coreg(self.selected_subject, self.comet.eeg_info).plotter
        self.plotter.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.ui.Figure_Layout.addWidget(self.plotter)
        self.plotter.show()
