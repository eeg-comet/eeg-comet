
import os.path
import pandas as pd

from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QFileDialog

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

        self.resize(1000, 800)

    def locate_subjects_dir(self):
        fname = QFileDialog.getExistingDirectory(self, "Select the folder containing raw data")
        self.subjects_dir = fname
        self.ui.subjects_dir_lineedit.setText(fname)

        list_subjects = [folder for folder in os.listdir(self.subjects_dir) if os.path.isdir(os.path.join(self.subjects_dir, folder))]

        for i in range(len(list_subjects)):
            self.ui.subjects_list.addItem(str(list_subjects[i]))