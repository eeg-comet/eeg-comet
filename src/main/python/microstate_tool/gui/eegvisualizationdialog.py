import os.path
from PyQt5 import uic
import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from customwidgets.mplwidget import MplWidget

from functions import load_data

class RawVisualizationDialog(QDialog):

    def __init__(self, context, parent=None):
        super(RawVisualizationDialog, self).__init__(parent)
        
        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("rawdatavisualization.ui"), self)
        
        self.ui.setWindowTitle("Explore Raw Data")
        
        self.ui.rawdata_plot_button.clicked.connect(self.plot_EEG)
        self.ui.rawdata_psd_button.clicked.connect(self.plot_PSD)
        self.ui.rawdata_channels_button.clicked.connect(self.plot_CHANNELS)
        
        #self.ui.addToolBar(NavigationToolbar(self.ui.MplWidget.canvas, self))
        
        
    def plot_CHANNELS(self):
        filename = self.ui.rawdata_file_list.currentItem().text()
        extension = self.extension
        data_type = self.data_type
        EEG = load_data.load_eegs(filename, extension, data_type)
        if self.ui.rawdata_show_channel_names_checkbox.isChecked():
            show_names = True
        else:
            show_names = False
        ax = self.ui.MplWidget.canvas.axes
        ax.clear()
        EEG.plot_sensors(ch_type='eeg', show_names=show_names, axes=ax)
        self.ui.MplWidget.canvas.draw()
    
    def plot_EEG(self):
        filename = self.ui.rawdata_file_list.currentItem().text()
        extension = self.extension
        data_type = self.data_type
        ax = self.ui.MplWidget.canvas.axes
        ax.clear()
        EEG = load_data.load_eegs(filename, extension, data_type)
        EEG.plot()
        self.ui.MplWidget.canvas.draw()
        
    def plot_PSD(self):
        filename = self.ui.rawdata_file_list.currentItem().text()
        extension = self.extension
        data_type = self.data_type
        EEG = load_data.load_eegs(filename, extension, data_type)
        fmin = int(self.ui.rawdata_range_psd_min.text())
        fmax = int(self.ui.rawdata_range_psd_max.text())
        ax = self.ui.MplWidget.canvas.axes
        ax.clear()
        EEG.plot_psd(fmin=fmin, fmax=fmax, ax=ax)
        self.ui.MplWidget.canvas.draw()