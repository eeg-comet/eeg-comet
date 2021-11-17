import os.path
from configparser import ConfigParser

from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QDialog, QMessageBox

from functions import preprocess

class PreprocessDialog(QDialog):

    def __init__(self, context, parent=None):
        super(PreprocessDialog, self).__init__(parent)


        # load the ui
        self.ui = uic.loadUi(context.get_resource("PreprocessDialog.ui"), self)

        self.ui.setWindowTitle("Preprocessing Raw Data")
        # self.ui.list_eegs = []

        self.ui.filter_data = False
        self.ui.downsample_data = False
        self.ui.lowcut_freq = 2
        self.ui.highcut_freq = 20
        self.ui.sample_rate = 250

        self.ui.no_option_checkbox.clicked.connect(self.preprocess_check_options)
        self.ui.filter_option_checkbox.clicked.connect(self.preprocess_check_options)
        self.ui.downsamp_option_checkbox.clicked.connect(self.preprocess_check_options)

        self.ui.preprocess_data_button.clicked.connect(self.preprocess_data)

    def preprocess_check_options(self):
        if self.ui.no_option_checkbox.isChecked():
            self.ui.filter_option_checkbox.setChecked(False)
            self.ui.filter_option_checkbox.setDisabled(True)
            self.ui.lowcut_freq_input.setDisabled(True)
            self.ui.highcut_freq_input.setDisabled(True)
            self.ui.fir_filtermethod_radio.setDisabled(True)
            self.ui.iir_filtermethod_radio.setDisabled(True)
            self.ui.downsamp_option_checkbox.setChecked(False)
            self.ui.downsamp_option_checkbox.setDisabled(True)
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
        if self.ui.no_option_checkbox.isChecked():
            self.ui.filter_data = False
            self.ui.lowcut_freq = ''
            self.ui.highcut_freq = ''
            self.ui.downsample_data = False
            self.ui.sample_rate = ''

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
                self.ui.filertmethod = 'fir'
            elif self.ui.iir_filtermethod_radio.isChecked():
                self.ui.filertmethod = 'iir'

            self.ui.lowcut_freq = int(self.ui.lowcut_freq_input.text())
            self.ui.highcut_freq = int(self.ui.highcut_freq_input.text())

            self.ui.sample_rate = int(self.ui.downsamp_freq_input.text())

            for file in list_eegs:
                self.ui.__progress = preprocess.preprocess_eegs(file, list_eegs,
                                                                eeg_format,
                                                                data_type,
                                                                self.ui.filter_data,
                                                                self.ui.filertmethod,
                                                                self.ui.lowcut_freq,
                                                                self.ui.highcut_freq,
                                                                self.ui.downsample_data,
                                                                self.ui.sample_rate,
                                                                self.ui.save_preprocessed_path)
                print("progress: ", self.ui.__progress)
                self.ui.preprocessing_progress.setValue(int(self.ui.__progress))
            if self.ui.__progress == 100:
                list_eegs_save = ','.join(map(str, list_eegs))
                # save config
                config = ConfigParser()
                config_file = os.path.join(self.ui.save_dir, 'config.ini')
                if os.path.isfile(config_file):
                    os.remove(config_file)
                config.read(config_file)
                config.add_section('step1')
                config.set('step1', 'data_extension', eeg_format)
                config.set('step1', 'data_type', data_type)
                config.set('step1', 'list_eegs', list_eegs_save)
                config.set('step1', 'filter_data', str(self.ui.filter_data))
                config.set('step1', 'lowcut_freq', str(self.ui.lowcut_freq))
                config.set('step1', 'highcut_freq', str(self.ui.highcut_freq))
                config.set('step1', 'downsample_data', str(self.ui.downsample_data))
                config.set('step1', 'sample_rate', str(self.ui.sample_rate))
                with open(config_file, 'w+') as f:
                    config.write(f)

                self.ui.close()