
import os.path
import mne
import pandas as pd
import numpy as np

from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QFileDialog

from functions.data_utils.data_io import DataIO
from functions.backfitting_utils.segmentation_io import SegmentationIO

class BackfittingVisualizationDialog(QDialog):
    def __init__(self, context, parent=None, tbx=None):
        super(BackfittingVisualizationDialog, self).__init__(parent)

        self.tbx = tbx
        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("BackfittingVisualizationWindow.ui"), self)

        self.ui.setWindowTitle("Visualization of the localized sources")

        self.ui.show_backfitting_button.clicked.connect(self.show_backfitting)

        self.resize(1000, 800)


    def show_backfitting(self):
        selected_file_name = self.ui.eeg_filenames_combobox.currentText()
        eeg_dir = os.path.join(self.preprocessed_data_path, f"{selected_file_name}{self.extension}")
        print(eeg_dir)
        data_io = DataIO()

        eeg = data_io.load_eegs(eeg_dir, self.extension, self.datatype)

        # Create an instance of the SegmentationIO class
        segmentation_io = SegmentationIO()
        segmentation_dir = os.path.join(self.segmentation_path, f"{selected_file_name}{self.export_format}")
        segmentation_array = segmentation_io.load_segmentation(segmentation_dir, import_format=self.export_format)

        print(segmentation_array)