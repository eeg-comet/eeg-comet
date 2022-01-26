import os.path
from PyQt5 import uic
import sys
import shutil
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from customwidgets.mplwidget import MplWidget
from configparser import ConfigParser

from functions import find_data, load_data, preprocess

class NewStudyWindow(QDialog):

    def __init__(self, context, parent=None):
        super(NewStudyWindow, self).__init__(parent)
        
        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("NewStudyWindow.ui"), self)
        
        self.ui.setWindowTitle("New Study - Import Raw Data and Preprocess")

        self.ui.step0_load_all_radio.clicked.connect(self.newstudy_controller)
        self.ui.step0_load_pattern_radio.clicked.connect(self.newstudy_controller)

        self.ui.step0_input_path_button.clicked.connect(self.choose_input)
        self.ui.step0_import_raw_button.clicked.connect(self.load_raw)

        self.ui.step0_save_path_button.clicked.connect(self.new_study_save_path)

        self.ui.step0_no_option_checkbox.clicked.connect(self.newstudy_controller)
        self.ui.step0_filter_option_checkbox.clicked.connect(self.newstudy_controller)
        self.ui.step0_downsamp_option_checkbox.clicked.connect(self.newstudy_controller)
        self.ui.step0_preprocess_data_button.clicked.connect(self.preprocess_data)

        self.ui.step0_selected_files_list.itemClicked.connect(self.plot_CHANNELS)
        self.ui.step0_selected_files_list.itemClicked.connect(self.plot_PSD)
        self.ui.rawdata_plot_button.clicked.connect(self.plot_EEG)


    def newstudy_controller(self):
        if self.ui.step0_load_pattern_radio.isChecked():
            self.ui.step0_import_pattern_lineedit.setEnabled(True)
        else:
            self.ui.step0_import_pattern_lineedit.setDisabled(True)

        if self.ui.step0_input_path_lineedit:
            self.input_folder_found = True
        else:
            self.input_folder_found = True

        if self.input_folder_found:
            self.ui.step0_import_raw_button.setEnabled(True)
            if self.ui.step0_selected_files_list.count() == 0:
                self.data_found = False
            else:
                self.data_found = True

        else:
            self.ui.step0_import_raw_button.setDisabled(True)

        if self.data_found and self.ui.step0_study_name_lineedit.text() and self.ui.step0_save_path_lineedit.text():
            self.step0_import_raw_button.setStyleSheet("background-color: lightgreen")
            # Enable Plot Options
            self.ui.rawdata_plot_button.setEnabled(True)
            self.ui.rawdata_show_channel_names_checkbox.setEnabled(True)
            self.ui.rawdata_range_psd_label.setEnabled(True)
            self.ui.rawdata_range_psd_min_label.setEnabled(True)
            self.ui.rawdata_range_psd_max_label.setEnabled(True)
            self.ui.rawdata_range_psd_min.setEnabled(True)
            self.ui.rawdata_range_psd_max.setEnabled(True)
            self.ui.rawdata_range_hz1.setEnabled(True)
            self.ui.rawdata_range_hz2.setEnabled(True)
            # Enable Preprocessing Options
            self.ui.step0_no_option_checkbox.setEnabled(True)
            self.ui.step0_filter_option_checkbox.setEnabled(True)
            self.ui.step0_downsamp_option_checkbox.setEnabled(True)
            self.ui.step0_preprocess_data_button.setEnabled(True)

            if self.ui.step0_no_option_checkbox.isChecked():
                self.ui.step0_preprocessing_progress.setEnabled(True)

                self.ui.step0_filter_option_checkbox.setChecked(False)
                self.ui.step0_filter_option_checkbox.setDisabled(True)
                self.ui.step0_lowcut_freq_input.setDisabled(True)
                self.ui.step0_highcut_freq_input.setDisabled(True)
                self.ui.step0_fir_filtermethod_radio.setDisabled(True)
                self.ui.step0_iir_filtermethod_radio.setDisabled(True)
                self.ui.step0_downsamp_option_checkbox.setChecked(False)
                self.ui.step0_downsamp_option_checkbox.setDisabled(True)
                self.ui.step0_downsamp_freq_input.setDisabled(True)
            else:
                self.ui.step0_filter_option_checkbox.setEnabled(True)
                self.ui.step0_downsamp_option_checkbox.setEnabled(True)
                if self.ui.step0_filter_option_checkbox.isChecked():
                    self.ui.step0_preprocessing_progress.setEnabled(True)
                    self.filter_data = True
                    self.ui.step0_filter_method_label.setEnabled(True)
                    self.ui.step0_fir_filtermethod_radio.setEnabled(True)
                    self.ui.step0_iir_filtermethod_radio.setEnabled(True)
                    self.ui.step0_lowcut_freq_label.setEnabled(True)
                    self.ui.step0_lowcut_freq_input.setEnabled(True)
                    self.ui.step0_filt_hz1.setEnabled(True)
                    self.ui.step0_highcut_freq_label.setEnabled(True)
                    self.ui.step0_highcut_freq_input.setEnabled(True)
                    self.ui.step0_filt_hz2.setEnabled(True)
                else:
                    self.filter_data = False
                    self.ui.step0_filter_method_label.setDisabled(True)
                    self.ui.step0_fir_filtermethod_radio.setDisabled(True)
                    self.ui.step0_iir_filtermethod_radio.setDisabled(True)
                    self.ui.step0_lowcut_freq_label.setDisabled(True)
                    self.ui.step0_lowcut_freq_input.setDisabled(True)
                    self.ui.step0_filt_hz1.setDisabled(True)
                    self.ui.step0_highcut_freq_label.setDisabled(True)
                    self.ui.step0_highcut_freq_input.setDisabled(True)
                    self.ui.step0_filt_hz2.setDisabled(True)
                if self.ui.step0_downsamp_option_checkbox.isChecked():
                    self.ui.step0_preprocessing_progress.setEnabled(True)
                    self.ui.downsample_data = True
                    self.ui.step0_downsamp_freq_label.setEnabled(True)
                    self.ui.step0_downsamp_freq_input.setEnabled(True)
                    self.ui.step0_downsamp_hz.setEnabled(True)
                else:
                    self.downsample_data = False
                    self.ui.step0_downsamp_freq_label.setDisabled(True)
                    self.ui.step0_downsamp_freq_input.setDisabled(True)
                    self.ui.step0_downsamp_hz.setDisabled(True)


    def choose_input(self):
        fname = QFileDialog.getExistingDirectory(self, "Select the folder containing raw data")
        self.input_folder = fname
        self.ui.step0_input_path_lineedit.setText(fname)
        self.newstudy_controller()

    def get_extension(self):
        selected_extension = self.ui.step0_import_format_combobox.currentText()
        if selected_extension == "BrainVision (.vhdr, .vmrk, .eeg)":
            extension = ".vhdr"
        elif selected_extension == "European data format (.edf)":
            extension = ".edf"
        elif selected_extension == "BioSemi data format (.bdf)":
            extension = ".bdf"
        elif selected_extension == "General data format (.gdf)":
            extension = ".gdf"
        elif selected_extension == "Neuroscan CNT (.cnt)":
            extension = ".cnt"
        elif selected_extension == "EGI simple binary (.egi)":
            extension = ".egi"
        elif selected_extension == "EGI MFF (.mff)":
            extension = ".mff"
        elif selected_extension == "EEGLAB files (.set, .fdt)":
            extension = ".set"
        elif selected_extension == "Nicolet (.data)":
            extension = ".data"
        elif selected_extension == "eXimia EEG data (.nxe)":
            extension = ".nxe"
        elif selected_extension == "Persyst EEG data (.lay, .dat)":
            extension = ".lay"
        elif selected_extension == "Nihon Kohden EEG data (.eeg, .21e, .pnt, .log)":
            extension = ".eeg"
        return extension

    def get_data_type(self):
        if self.ui.step0_import_continuous_radio.isChecked():
            data_type = "continuous"
        elif self.ui.step0_import_epoched_radio.isChecked():
            data_type = "epoched"
        return data_type

    def load_raw(self):
        if self.ui.step0_load_all_radio.isChecked():
            pattern = '*'
        if self.ui.step0_load_pattern_radio.isChecked():
            pattern = '*'+self.ui.step0_import_pattern_lineedit.text()
        self.extension = self.get_extension()
        self.data_type = self.get_data_type()
        self.list_eegs = find_data.find_eeg(self.input_folder, self.extension, pattern)
        for i in range(len(self.list_eegs)):
            self.ui.step0_selected_files_list.addItem(str(self.list_eegs[i]))
        # self.ui.foldername_preprocessed_data = os.path.join(self.ui.input_folder, 'output')
        self.newstudy_controller()

    def new_study_save_path(self):
        #step0_study_name_linedit.text()
        path = QFileDialog.getExistingDirectory(self, "Select the folder to save results")
        folder_name = self.ui.step0_study_name_lineedit.text()
        save_directory = os.path.join(path, folder_name)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        else:
            reply = QMessageBox.question(self, "Study exists!",
                                         "Do you want to overwrite an existing study?",
                                         QMessageBox.Yes |
                                         QMessageBox.No)
            if reply == QMessageBox.Yes:
                shutil.rmtree(save_directory)
            else:
                folder_name = folder_name+'_new'
                self.ui.step0_study_name_lineedit.setText(folder_name)
                save_directory = os.path.join(path, folder_name)

        self.save_dir = save_directory
        self.ui.step0_save_path_lineedit.setText(save_directory)
        self.use_raw_data = True
        self.newstudy_controller()



    def preprocess_data(self):

        self.save_preprocessed_path = os.path.join(self.save_dir, 'preprocessed_data')
        if not os.path.exists(self.save_preprocessed_path):
            os.makedirs(self.save_preprocessed_path)

        if self.ui.step0_no_option_checkbox.isChecked():
            self.filter_data = False
            self.lowcut_freq = ''
            self.highcut_freq = ''
            self.downsample_data = False
            self.sample_rate = ''

        if self.ui.step0_lowcut_freq_input.text() >= self.ui.step0_highcut_freq_input.text():
            QMessageBox.information(self, "Filter Error",
                                    "Please modify the filter range!",
                                    QMessageBox.Ok)
        else:
            print("preprocessing data ...")
            list_eegs = self.list_eegs
            print(list_eegs)

            eeg_format = self.extension
            data_type = self.data_type

            if self.ui.step0_fir_filtermethod_radio.isChecked():
                self.filertmethod = 'fir'
            elif self.ui.step0_iir_filtermethod_radio.isChecked():
                self.filertmethod = 'iir'

            self.lowcut_freq = int(self.ui.step0_lowcut_freq_input.text())
            self.highcut_freq = int(self.ui.step0_highcut_freq_input.text())

            self.sample_rate = int(self.ui.step0_downsamp_freq_input.text())

            for file in list_eegs:
                self.ui.__progress = preprocess.preprocess_eegs(file, list_eegs,
                                                                eeg_format,
                                                                data_type,
                                                                self.filter_data,
                                                                self.filertmethod,
                                                                self.lowcut_freq,
                                                                self.highcut_freq,
                                                                self.downsample_data,
                                                                self.sample_rate,
                                                                self.save_preprocessed_path)
                print("progress: ", self.__progress)
                self.ui.step0_preprocessing_progress.setValue(int(self.__progress))
            if self.__progress == 100:
                list_eegs_save = ','.join(map(str, list_eegs))
                # save config
                config = ConfigParser()
                config_file = os.path.join(self.save_dir, 'config.ini')
                if os.path.isfile(config_file):
                    os.remove(config_file)
                config.read(config_file)
                config.add_section('step1')
                config.set('step1', 'study_name', self.ui.step0_study_name_lineedit.text())
                config.set('step1', 'input_folder', self.ui.step0_input_path_lineedit.text())
                config.set('step1', 'save_folder', self.save_dir)
                config.set('step1', 'data_extension', eeg_format)
                config.set('step1', 'data_type', data_type)
                config.set('step1', 'list_eegs', list_eegs_save)
                config.set('step1', 'filter_data', str(self.filter_data))
                config.set('step1', 'lowcut_freq', str(self.lowcut_freq))
                config.set('step1', 'highcut_freq', str(self.highcut_freq))
                config.set('step1', 'downsample_data', str(self.downsample_data))
                config.set('step1', 'sample_rate', str(self.sample_rate))
                with open(config_file, 'w+') as f:
                    config.write(f)

                self.ui.close()



    def plot_CHANNELS(self):
        filename = self.ui.step0_selected_files_list.currentItem().text()
        extension = self.extension
        data_type = self.data_type
        EEG = load_data.load_eegs(filename, extension, data_type)
        if self.ui.rawdata_show_channel_names_checkbox.isChecked():
            show_names = True
        else:
            show_names = False
        ax = self.ui.MplWidget_chan.canvas.axes
        ax.clear()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        EEG.plot_sensors(ch_type='eeg', show_names=show_names, axes=ax)
        self.ui.MplWidget_chan.canvas.draw()
    
    def plot_EEG(self):
        filename = self.ui.step0_selected_files_list.currentItem().text()
        extension = self.extension
        data_type = self.data_type
        EEG = load_data.load_eegs(filename, extension, data_type)
        EEG.plot()

    def plot_PSD(self):
        filename = self.ui.step0_selected_files_list.currentItem().text()
        extension = self.extension
        data_type = self.data_type
        EEG = load_data.load_eegs(filename, extension, data_type)
        fmin = int(self.ui.rawdata_range_psd_min.text())
        fmax = int(self.ui.rawdata_range_psd_max.text())
        ax = self.ui.MplWidget_psd.canvas.axes
        ax.clear()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        EEG.plot_psd(fmin=fmin, fmax=fmax, ax=ax)
        self.ui.MplWidget_psd.canvas.draw()