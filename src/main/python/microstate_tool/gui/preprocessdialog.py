import os.path
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QDialog, QMessageBox

from functions import preprocess

class PreprocessDialog(QDialog):

    def __init__(self, context, parent=None):
        super(PreprocessDialog, self).__init__(parent)

        # load the ui
        self.ui = uic.loadUi(context.get_resource("preprocessdialog.ui"), self)
        
        self.ui.setWindowTitle("Preprocessing Raw Data")
        #self.ui.list_eegs = []
        
        self.ui.no_option_checkbox.clicked.connect(self.preprocess_check_options)
        self.ui.filter_option_checkbox.clicked.connect(self.preprocess_check_options)
        self.ui.downsamp_option_checkbox.clicked.connect(self.preprocess_check_options)

        self.ui.preprocess_data_button.clicked.connect(self.preprocess_data)
        
    def preprocess_check_options(self):
        if self.ui.no_option_checkbox.isChecked():
            self.ui.filter_option_checkbox.setChecked(False)
            self.ui.filter_option_checkbox.setDisabled(True)
            self.ui.filter_data = False
            self.ui.lowcut_freq_input.setDisabled(True)
            self.ui.highcut_freq_input.setDisabled(True)
            self.ui.fir_filtermethod_radio.setDisabled(True)
            self.ui.iir_filtermethod_radio.setDisabled(True)
            self.ui.downsamp_option_checkbox.setChecked(False)
            self.ui.downsamp_option_checkbox.setDisabled(True)
            self.ui.downsample_data = False
            self.ui.downsamp_freq_input.setDisabled(True)
        else:
            self.ui.filter_option_checkbox.setEnabled(True)
            self.ui.downsamp_option_checkbox.setEnabled(True)
            if self.ui.filter_option_checkbox.isChecked():
                self.ui.filter_data = True
                self.ui.lowcut_freq_input.setEnabled(True)
                self.ui.highcut_freq_input.setEnabled(True)
                self.ui.fir_filtermethod_radio.setEnabled(True)
                self.ui.iir_filtermethod_radio.setEnabled(True)
            else:
                self.ui.filter_data = False
                self.ui.lowcut_freq_input.setDisabled(True)
                self.ui.highcut_freq_input.setDisabled(True)
                self.ui.fir_filtermethod_radio.setDisabled(True)
                self.ui.iir_filtermethod_radio.setDisabled(True)
            if self.ui.downsamp_option_checkbox.isChecked():
                self.ui.downsample_data = True
                self.ui.downsamp_freq_input.setEnabled(True)
            else:
                self.ui.downsample_data = False
                self.ui.downsamp_freq_input.setDisabled(True)
    
    
    
    def preprocess_data(self):
        if self.ui.lowcut_freq_input.text() >= self.ui.highcut_freq_input.text():
            QMessageBox.information(self, "Filter Error",
                                        "Please modify the filter range!",
                                        QMessageBox.Ok)
        else:
            print("preprocessing data ...")
            list_eegs = self.ui.list_eegs
            print(list_eegs)
            
            eeg_format = self.ui.extension
            data_type = self.ui.data_type
            
            if self.ui.fir_filtermethod_radio.isChecked():
                filertmethod = 'fir'
            elif self.ui.iir_filtermethod_radio.isChecked():
                filertmethod = 'iir'
            
            for file in list_eegs:
                self.ui.__progress = preprocess.preprocess_eegs(file, list_eegs,
                                                             eeg_format,
                                                             data_type,
                                                             self.ui.filter_data,
                                                             filertmethod,
                                                             int(self.ui.lowcut_freq_input.text()),
                                                             int(self.ui.highcut_freq_input.text()),
                                                             self.ui.downsample_data,
                                                             int(self.ui.downsamp_freq_input.text()),
                                                             self.ui.save_dir)
                print("progress: ", self.ui.__progress)
                self.ui.preprocessing_progress.setValue(int(self.ui.__progress))
            if self.ui.__progress==100:
                self.ui.close()