"""Coregistration GUI to align EEG sensor space with MRI anatomy and visualize alignment."""

import os.path

import mne
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QListWidgetItem

from eeg_comet.gui_utils.responsive import (
    apply_window_minimum,
    configure_splitter,
    expand_canvas,
)
from eeg_comet.sourcelocalization_utils.coregistration import Coregistration
from eeg_comet.sourcelocalization_utils.coregistration_visualizer import CoregistrationVisualizer


class CoregistrationWindow(QDialog):
    """Dialog to manage and visualize EEG–MRI coregistration tasks.

    Provides helpers to perform manual and automatic coregistration and to
    display 3D alignment between head, sensor, and source spaces.

    Attributes:
      comet: COMET toolbox instance providing configuration and EEG info.
      subjects_dir (str): FreeSurfer subjects directory used for coregistration.
      coregistration (Coregistration): Coregistration helper.
      coregistration_visualizer (CoregistrationVisualizer): Visualizer helper.
      plotter: Embedded 3D plotter widget (initialized on demand).
    """

    def __init__(self, context, parent=None, tbx=None):
        """Initialize window, load UI, and set up coregistration helpers.

        Args:
          context: Resource/context provider used to resolve the `.ui` file.
          parent: Optional parent widget.
          tbx: COMET toolbox instance providing settings and EEG info.

        Returns:
          None
        """
        super().__init__(parent)

        self.comet = tbx
        # Initialize selected subject placeholder
        self.selected_subject = ""

        if self.comet.use_anatomy != "fsaverage":
            self.subjects_dir = self.comet.individual_subjects_dir
        else:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
            self.subjects_dir = os.path.dirname(fs_dir)

        self.coregistration = Coregistration(self.subjects_dir)
        self.coregistration_visualizer = CoregistrationVisualizer(self.subjects_dir)

        self.ui = uic.loadUi(context.get_resource("CoregistrationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the head, sensor, and source space alignment")
        apply_window_minimum(self, "dialog")
        if hasattr(self.ui, "coregistration_label"):
            self.ui.coregistration_label.setProperty("role", "banner")
        if hasattr(self.ui, "splitter"):
            configure_splitter(
                self.ui.splitter,
                ratio=(1, 2),
                save_key="coregistration",
            )

        self.ui.manual_coreg_button.clicked.connect(self.manual_coregistration)
        self.ui.auto_coreg_button.clicked.connect(self.automatic_coregistration)
        self.ui.plot_alignment_button.clicked.connect(self.show_alignment)
        self.ui.subjects_list.itemSelectionChanged.connect(self.handle_new_file_selection)

        self.populate_subjects_list()

        self.plotter = None
        self.resize(1000, 800)

    def populate_subjects_list(self):
        """Populate the subjects list with folder names inside ``subjects_dir``.

        Returns:
          None
        """
        subject_folders = [
            f
            for f in os.listdir(self.subjects_dir)
            if os.path.isdir(os.path.join(self.subjects_dir, f))
        ]
        self.ui.subjects_list.clear()
        for folder in subject_folders:
            self.ui.subjects_list.addItem(QListWidgetItem(folder))

    def handle_new_file_selection(self):
        """Handle selection change in the subjects list and update label.

        Sets the currently selected subject and updates the subject ID label.

        Returns:
          None
        """
        selected_item = self.ui.subjects_list.currentItem()
        self.selected_subject = selected_item.text()
        self.ui.subject_id_label.setText(self.selected_subject)

    def clean_figure_layout(self):
        """Clear and delete all widgets from the figure layout.

        Returns:
          None
        """
        while self.ui.Figure_Layout.count() > 0:
            item = self.ui.Figure_Layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def manual_coregistration(self):
        """Open MNE-Python GUI for manual coregistration.

        Uses the currently selected subject ID.

        Returns:
          None
        """
        self.coregistration.manual_coreg(self.selected_subject)

    def automatic_coregistration(self):
        """Perform automatic coregistration for the selected subject.

        Uses EEG sensor information from ``self.comet.eeg_info``.

        Returns:
          None
        """
        self.coregistration.auto_coreg(self.selected_subject, self.comet.eeg_info)

    def show_alignment(self):
        """Update the embedded 3D alignment visualization.

        Recreates the plotter for the selected subject and adds it to the UI.

        Returns:
          None
        """
        if self.plotter is not None:
            self.clean_figure_layout()
        self.plotter = self.coregistration_visualizer.show_coreg(
            self.selected_subject, self.comet.eeg_info
        ).plotter
        expand_canvas(self.plotter, minimum=(360, 280))
        self.ui.Figure_Layout.addWidget(self.plotter)
        self.plotter.show()
