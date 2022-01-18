import os.path
import shutil
import numpy as np
import pickle
import csv
import json
import pandas as pd
import webbrowser
from configparser import ConfigParser
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDialog, QMessageBox,QAbstractItemView

from gui.eegvisualizationdialog import RawVisualizationDialog
from gui.preprocessdialog import PreprocessDialog
from gui.microstatedialog import MicrostateDialog


from functions import find_data, load_data
from functions.concatenate_data import concatenate_files
from functions.clustering_functions import _pre_clustering, initialize_centers, eegInfo
from functions.clustering_functions import number_of_clusters, clustering_func, clustering_minibatch
from functions import extract_features_functions

# Settings Class
class SettingsModel:

    def __init__(self, settings=None):
        super(SettingsModel, self).__init__()
        self.settings = settings or []

    def show_settings(self):
        return print(self.settings)

# Main Class
class MainMicrostateWindow(QMainWindow):

    def __init__(self, context, parent=None):
        super(MainMicrostateWindow, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("MainMicrostateWindow.ui"), self)

        self.ui.setWindowTitle("Microstate Toolbox")
        #self.ui.setFixedSize(3000, 1500)

        self.ui.RawVisualizationDialog = RawVisualizationDialog(context)
        self.ui.PreprocessDialog = PreprocessDialog(context)
        self.ui.MicrostateDialog = MicrostateDialog()

        self.done_preprocessing = False
        self.done_clustering = False
        self.done_feature_extraction = False
        self.ui.foldername_raw_data = ""
        self.ui.foldername_preprocessed_data = ""
        self.ui.list_eegs = []


        self.ui.open_github_action.triggered.connect(self.open_github)
        self.ui.report_issues_action.triggered.connect(self.report_issues)

        self.ui.load_results_action.triggered.connect(self.load_results)

        self.ui.step0_exploreraw_button.clicked.connect(self.open_exploreraw_dialog)
        self.ui.step0_reset_selectedfiles_button.clicked.connect(self.reset_selected)
        self.ui.step0_reset_selectedfiles_button.clicked.connect(self.mainwindow_check_options)

        self.ui.step1_import_raw_radio.clicked.connect(self.mainwindow_check_options)
        self.ui.step1_import_preprocessed_radio.clicked.connect(self.mainwindow_check_options)

        self.ui.step1_importraw_button.clicked.connect(self.load_raw)
        self.ui.step1_savepath_button.clicked.connect(self.save_path)

        self.ui.step1_load_preprocess_button.clicked.connect(self.open_preprocessing_dialog)

        self.ui.step1_savepath_button.clicked.connect(self.mainwindow_check_options)
        self.ui.step1_load_preprocess_button.clicked.connect(self.mainwindow_check_options)
        self.ui.step1_importraw_button.clicked.connect(self.mainwindow_check_options)
        self.ui.step1_load_all_radio.clicked.connect(self.mainwindow_check_options)
        self.ui.step1_load_pattern_radio.clicked.connect(self.mainwindow_check_options)
        self.ui.step2_clustermethod_combobox.activated.connect(self.mainwindow_check_options)
        self.ui.step2_auto_numberofmaps_radio.clicked.connect(self.mainwindow_check_options)
        self.ui.step2_user_numberofmaps_radio.clicked.connect(self.mainwindow_check_options)
        self.ui.step2_smoothgfp_checkbox.clicked.connect(self.mainwindow_check_options)

        self.ui.step2_clustering_button.clicked.connect(self.do_clustering)
        self.ui.step2_clustering_button.clicked.connect(self.mainwindow_check_options)
        self.ui.step3_visualize_clustering_button.clicked.connect(self.visualize_results)

        self.ui.step3_extractfeatures_button.clicked.connect(self.extract_features)
        self.ui.step3_extractfeatures_button.clicked.connect(self.mainwindow_check_options)

        self.ui.step0_exit_button.clicked.connect(self.exit_msg)

        # Set actions

        self.ui.import_settings_action.triggered.connect(self.import_settings)
        self.ui.export_settings_action.triggered.connect(self.export_settings)

        # Initialize settings
        default_settings = [("Amin", "Is cool"), ("Raaj", "Is nice"), ("Paul", "Is great"), ("Marian", "Is awesome")]
        self.app_settings = SettingsModel(default_settings)

    # Raaj Testing adding a persistent settings store
    def import_settings(self, fname=None):
        if not fname:
            fname = QFileDialog.getOpenFileName(self, "Open file", "", "JSON files (*.json)")
        with open(fname[0], 'r') as f:
            self.app_settings.settings = json.load(f)
        return print("Settings Loaded")

    def export_settings(self):
        fname = QFileDialog.getSaveFileName(self, "Save file", "", "JSON files (*.json)")
        with open(fname[0], 'w') as f:
            data = json.dump(self.app_settings.settings, f)
        return print("Exported Settings")

        # # #

    def open_github(self):
        webbrowser.open('https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction')

    def report_issues(self):
        webbrowser.open('https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction/issues/new')

    def load_results(self):
        fname = QFileDialog.getExistingDirectory(self, "Select the folder containing microstate results")
        self.ui.save_dir = fname
        self.done_clustering = True

    def get_extension(self):
        selected_extension = self.ui.step1_importformat_combobox.currentText()
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
        if self.ui.step1_import_continuous_radio.isChecked():
            data_type = "continuous"
        elif self.ui.step1_import_epoched_radio.isChecked():
            data_type = "epoched"
        return data_type

    def load_raw(self):
        fname = QFileDialog.getExistingDirectory(self, "Select the folder containing raw data")
        self.ui.foldername_raw_data = fname
        self.ui.input_folder = self.ui.foldername_raw_data
        if self.ui.step1_load_all_radio.isChecked():
            pattern = '*'
        if self.ui.step1_load_pattern_radio.isChecked():
            pattern = '*'+self.ui.step1_importpattern_lineedit.text()
        self.ui.extension = self.get_extension()
        self.ui.data_type = self.get_data_type()
        self.ui.list_eegs = find_data.find_eeg(self.ui.input_folder, self.ui.extension, pattern)
        for i in range(len(self.ui.list_eegs)):
            self.ui.step0_selectedfiles_list.addItem(str(self.ui.list_eegs[i]))
        # self.ui.foldername_preprocessed_data = os.path.join(self.ui.input_folder, 'output')

    def save_path(self):
        fname = QFileDialog.getExistingDirectory(self, "Select the folder to save results")
        self.ui.save_dir = fname
        self.ui.step1_savedir_lineedit.setText(fname)

    def reset_selected(self):
        self.ui.step0_selectedfiles_list.clear()
        self.ui.step1_savedir_lineedit.clear()
        self.done_preprocessing = False
        self.done_clustering = False

    def open_exploreraw_dialog(self):
        self.RawVisualizationDialog.extension = self.ui.extension
        self.RawVisualizationDialog.data_type = self.ui.data_type
        list_eegs = self.ui.list_eegs
        for i in range(len(list_eegs)):
            self.ui.RawVisualizationDialog.ui.rawdata_file_list.addItem(str(list_eegs[i]))

        self.ui.RawVisualizationDialog.show()

    def open_preprocessing_dialog(self):

        if self.ui.step1_import_preprocessed_radio.isChecked():
            fname = QFileDialog.getExistingDirectory(self, "Select the folder containing preprocessed data")
            self.ui.save_preprocessed_path = fname

            if not self.ui.step1_savedir_lineedit.text():
                self.ui.save_dir = self.ui.save_preprocessed_path

            # load config
            config = ConfigParser()
            config_file = os.path.join(self.ui.save_dir, 'config.ini')
            config.read(config_file)
            self.ui.extension = config.get(
                'step1', 'data_extension')
            self.ui.data_type = config.get(
                'step1', 'data_type')
            self.ui.list_eegs = config.get(
                'step1', 'list_eegs')
            self.ui.PreprocessDialog.ui.filter_data = config.get(
                'step1', 'filter_data')
            self.ui.PreprocessDialog.ui.lowcut_freq = config.get(
                'step1', 'lowcut_freq')
            self.ui.PreprocessDialog.ui.highcut_freq = config.get(
                'step1', 'highcut_freq')
            self.ui.PreprocessDialog.ui.downsample_data = config.get(
                'step1', 'downsample_data')
            self.ui.PreprocessDialog.ui.sample_rate = config.get(
                'step1', 'sample_rate')

            self.ui.list_eegs = self.ui.list_eegs.split(",")
            print(self.ui.list_eegs)
            for i in range(len(self.ui.list_eegs)):
                self.ui.step0_selectedfiles_list.addItem(str(self.ui.list_eegs[i]))

            # add condition
            self.ui.done_preprocessing = True
        if self.ui.step1_import_raw_radio.isChecked():

            self.ui.save_preprocessed_path = os.path.join(
                self.ui.save_dir, 'preprocessed_data')
            if not os.path.exists(self.ui.save_preprocessed_path):
                os.makedirs(self.ui.save_preprocessed_path)

            self.ui.PreprocessDialog.setWindowModality(QtCore.Qt.ApplicationModal)
            self.ui.PreprocessDialog.ui.list_eegs = self.ui.list_eegs
            self.ui.PreprocessDialog.ui.extension = self.ui.extension
            self.ui.PreprocessDialog.ui.data_type = self.ui.data_type
            self.ui.PreprocessDialog.ui.save_dir = self.ui.save_dir
            self.ui.PreprocessDialog.ui.save_preprocessed_path = self.ui.save_preprocessed_path

            self.ui.PreprocessDialog.show()
            # add condition
            self.done_preprocessing = True

    def mainwindow_check_options(self):

        #
        if self.ui.step1_import_raw_radio.isChecked():
            self.use_raw_data = True
        else:
            self.use_raw_data = False
        if self.ui.step1_import_preprocessed_radio.isChecked():
            self.use_preprocessed_data = True
        else:
            self.use_preprocessed_data = False
        if self.ui.step0_selectedfiles_list.count() == 0:
            self.data_found = False
        else:
            self.data_found = True
        if self.ui.step1_savedir_lineedit.text():
            self.save_directory_selected = True
        else:
            self.save_directory_selected = False
        #

        if self.use_raw_data:
            self.ui.step1_load_preprocess_button.setText("Preprocess Raw Data")
            self.ui.step1_importformat_combobox.setEnabled(True)
            self.ui.step1_load_all_radio.setEnabled(True)
            self.ui.step1_load_pattern_radio.setEnabled(True)
            self.ui.step1_importpattern_lineedit.setEnabled(True)
            self.ui.step1_import_continuous_radio.setEnabled(True)
            self.ui.step1_import_epoched_radio.setEnabled(True)
            self.ui.step1_importraw_button.setEnabled(True)
            if self.ui.step1_load_all_radio.isChecked():
                self.ui.step1_importpattern_lineedit.setDisabled(True)
            if self.ui.step1_load_pattern_radio.isChecked():
                self.ui.step1_importpattern_lineedit.setEnabled(True)

        if self.use_preprocessed_data:
            self.ui.step1_load_preprocess_button.setText("Load Preprocessed Data")
            self.ui.step1_importformat_combobox.setDisabled(True)
            self.ui.step1_load_all_radio.setDisabled(True)
            self.ui.step1_load_pattern_radio.setDisabled(True)
            self.ui.step1_importpattern_lineedit.setDisabled(True)
            self.ui.step1_import_continuous_radio.setDisabled(True)
            self.ui.step1_import_epoched_radio.setDisabled(True)
            self.ui.step1_importraw_button.setDisabled(True)
            self.ui.step1_savedir_lineedit.setEnabled(True)
            self.ui.step1_savepath_button.setEnabled(True)

        if self.data_found:
            self.ui.step0_exploreraw_button.setEnabled(True)
            self.ui.step1_savedir_lineedit.setEnabled(True)
            self.ui.step1_savepath_button.setEnabled(True)
            if self.save_directory_selected:
                self.ui.step1_load_preprocess_button.setEnabled(True)
            else:
                self.ui.step1_load_preprocess_button.setDisabled(True)
        else:
            self.ui.step0_exploreraw_button.setDisabled(True)
            if self.use_preprocessed_data:
                self.ui.step1_load_preprocess_button.setEnabled(True)
            else:
                self.ui.step1_load_preprocess_button.setDisabled(True)

        if self.done_preprocessing:
            self.ui.step1_preprocessed_led_radio.setStyleSheet("QRadioButton::indicator"
                                                               "{"
                                                               "background-color : green;"
                                                               "}")

            self.ui.step2_save_clustering_checkbox.setEnabled(True)
            self.ui.step2_clustering_title_label.setEnabled(True)
            self.ui.step2_clustermethod_combo_label.setEnabled(True)
            self.ui.step2_clustermethod_combobox.setEnabled(True)
            self.ui.step2_numberofmaps_label.setEnabled(True)
            self.ui.step2_auto_numberofmaps_radio.setEnabled(True)
            self.ui.step2_user_numberofmaps_radio.setEnabled(True)
            if self.ui.step2_auto_numberofmaps_radio.isChecked():
                self.ui.step2_user_numberofmaps_input.setDisabled(True)
            if self.ui.step2_user_numberofmaps_radio.isChecked():
                self.ui.step2_user_numberofmaps_input.setEnabled(True)
            if self.ui.step2_smoothgfp_checkbox.isChecked():
                self.ui.step2_kernel_size_input.setEnabled(True)
            else:
                self.ui.step2_kernel_size_input.setDisabled(True)
            self.ui.step2_numberofrepeats_label.setEnabled(True)
            self.ui.step2_user_numberofrepeats_input.setEnabled(True)
            self.ui.step2_other_label.setEnabled(True)
            self.ui.step2_other_options_combobox.setEnabled(True)
            self.ui.step2_initializer_label.setEnabled(True)
            self.ui.step2_random_initializer_radio.setEnabled(True)
            self.ui.step2_kmeans_initializer_radio.setEnabled(True)
            self.ui.step2_stopcondition_label.setEnabled(True)
            self.ui.step2_tolerance_label.setEnabled(True)
            self.ui.step2_stopcondition_input.setEnabled(True)
            self.ui.step2_smoothgfp_checkbox.setEnabled(True)
            self.ui.step2_kernel_size_label.setEnabled(True)
            self.ui.step2_kernel_size_input.setEnabled(True)
            self.ui.step2_performclustering_label.setEnabled(True)
            self.ui.step2_clustering_button.setEnabled(True)
        else:
            self.ui.step1_preprocessed_led_radio.setStyleSheet("QRadioButton::indicator"
                                                               "{"
                                                               "background-color : red;"
                                                               "}")
            self.ui.step2_clustering_title_label.setDisabled(True)
            self.ui.step2_clustermethod_combo_label.setDisabled(True)
            self.ui.step2_clustermethod_combobox.setDisabled(True)
            self.ui.step2_numberofmaps_label.setDisabled(True)
            self.ui.step2_auto_numberofmaps_radio.setDisabled(True)
            self.ui.step2_user_numberofmaps_radio.setDisabled(True)
            self.ui.step2_user_numberofmaps_input.setDisabled(True)
            self.ui.step2_numberofrepeats_label.setDisabled(True)
            self.ui.step2_user_numberofrepeats_input.setDisabled(True)
            self.ui.step2_other_label.setDisabled(True)
            self.ui.step2_other_options_combobox.setDisabled(True)
            self.ui.step2_initializer_label.setDisabled(True)
            self.ui.step2_random_initializer_radio.setDisabled(True)
            self.ui.step2_kmeans_initializer_radio.setDisabled(True)
            self.ui.step2_stopcondition_label.setDisabled(True)
            self.ui.step2_tolerance_label.setDisabled(True)
            self.ui.step2_stopcondition_input.setDisabled(True)
            self.ui.step2_smoothgfp_checkbox.setDisabled(True)
            self.ui.step2_kernel_size_label.setDisabled(True)
            self.ui.step2_kernel_size_input.setDisabled(True)
            self.ui.step2_performclustering_label.setDisabled(True)
            self.ui.step2_clustering_button.setDisabled(True)



        METHOD = self.step2_clustermethod_combobox.currentText()
        if METHOD == "K-MEANS":
            self.ui.step2_other_label.setText("K-Means Distance Metric:")
            self.ui.step2_other_options_combobox.clear()
            self.ui.step2_other_options_combobox.addItem("Euclidean")
            self.ui.step2_other_options_combobox.addItem("Euclidean Square")
            self.ui.step2_other_options_combobox.addItem("Manhattan")
            self.ui.step2_other_options_combobox.addItem("Chebyshev")
            self.ui.step2_other_options_combobox.addItem("Minkowski")
            self.ui.step2_other_options_combobox.setCurrentText("Euclidean Square")
        # elif METHOD == "MINI BATCH K-MEANS":
        #
        elif METHOD == "X-MEANS":
            self.ui.step2_other_label.setText("X-Means Splitting Criterion:")
            self.ui.step2_other_options_combobox.clear()
            self.ui.step2_other_options_combobox.addItem("Bayesian Information Criterion")
            self.ui.step2_other_options_combobox.addItem("Minimum Noiseless Description Length")
            self.ui.step2_other_options_combobox.setCurrentText("Bayesian Information Criterion")
        else:
            self.ui.step2_other_label.setText("Other Options:")
            self.ui.step2_other_options_combobox.clear()

        if self.done_clustering:
            self.ui.step2_clustering_led_radio.setStyleSheet("QRadioButton::indicator"
                                                             "{"
                                                             "background-color : green;"
                                                             "}")
            self.ui.step3_visualize_clustering_button.setEnabled(True)
            self.ui.step3_features_title_label.setEnabled(True)
            self.ui.step3_featurestoextract_label.setEnabled(True)
            self.ui.step3_outputformats_label.setEnabled(True)
            self.ui.step3_coverage_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_foc_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_mmd_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_gev_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_tp_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_save_transitions_checkbox.setEnabled(True)
            self.ui.step3_outputformats_combobox.setEnabled(True)
            self.ui.step3_extractfeatures_button.setEnabled(True)
        else:
            self.ui.step2_clustering_led_radio.setStyleSheet("QRadioButton::indicator"
                                                             "{"
                                                             "background-color : red;"
                                                             "}")
            self.ui.step3_visualize_clustering_button.setDisabled(True)
            self.ui.step3_features_title_label.setDisabled(True)
            self.ui.step3_featurestoextract_label.setDisabled(True)
            self.ui.step3_outputformats_label.setDisabled(True)
            self.ui.step3_coverage_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_foc_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_mmd_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_gev_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_tp_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_save_transitions_checkbox.setDisabled(True)
            self.ui.step3_outputformats_combobox.setDisabled(True)
            self.ui.step3_extractfeatures_button.setDisabled(True)

        if self.done_feature_extraction:
            self.ui.step3_extract_led_radio.setStyleSheet("QRadioButton::indicator"
                                                          "{"
                                                          "background-color : green;"
                                                          "}")
        else:
            self.ui.step3_extract_led_radio.setStyleSheet("QRadioButton::indicator"
                                                          "{"
                                                          "background-color : red;"
                                                          "}")

    def do_clustering(self):

        # Fs = 250 #from previous window
        Fs = self.ui.PreprocessDialog.downsamp_freq_input.text()
        print(Fs)
        # DATA = INPUT_DATA #from previous window
        # print(DATA.shape)

        FOLDER = self.ui.save_preprocessed_path
        print(FOLDER)

        # Concatenate data
        CONCATENATE = True
        DATA, N_CHANNELS, FILENAMES, LENGTH_DATA = concatenate_files(
            FOLDER, self.ui.save_dir)

        self.listoffiles = FILENAMES
        self.lengthoffiles = LENGTH_DATA

        if self.ui.step2_smoothgfp_checkbox.isChecked():
            SMOOTHING = True
            SMOOTHING_KERNEL = int(self.ui.step2_kernel_size_input.text())
        else:
            SMOOTHING = False
            SMOOTHING_KERNEL = []
        print(SMOOTHING_KERNEL)

        if self.ui.step2_random_initializer_radio.isChecked():
            INITIALIZER = "Random"
        elif self.ui.step2_kmeans_initializer_radio.isChecked():
            INITIALIZER = "K-Means++"

        METHOD = self.ui.step2_clustermethod_combobox.currentText()
        print(METHOD)

        TOLERANCE = float(self.ui.step2_stopcondition_input.text())
        print(TOLERANCE)

        if METHOD == "Mini Batch K-MEANS":

            # modify
            if self.ui.step2_user_numberofmaps_radio.isChecked():
                CLUSTERS = "USER"
                N_STATES = int(self.ui.step2_user_numberofmaps_input.text())
                print(N_STATES)

            best_maps, final_segmentation, gev = clustering_minibatch(
                FOLDER,
                Fs,
                SMOOTHING_KERNEL,
                N_STATES,
                INITIALIZER,
                TOLERANCE)


        else:

            MAPS, PEAKS = _pre_clustering(DATA, Fs, SMOOTHING_KERNEL)

            # modify
            if self.ui.step2_auto_numberofmaps_radio.isChecked():
                CLUSTERS = "AUTO"
                N_STATES = number_of_clusters(MAPS)
                print('Using Elbow method to find the optimal number of microstate maps')
            elif self.ui.step2_user_numberofmaps_radio.isChecked():
                CLUSTERS = "USER"
                N_STATES = int(self.ui.step2_user_numberofmaps_input.text())
            print(N_STATES)

            INITIAL_CENTERS = initialize_centers(
                DATA,
                MAPS,
                PEAKS,
                N_STATES,
                INITIALIZER)
            print(INITIAL_CENTERS.shape)

            METRIC = self.ui.step2_other_options_combobox.currentText()
            print(METRIC)

            REPEAT = int(self.ui.step2_user_numberofrepeats_input.text())

            best_maps, final_segmentation, gev = clustering_func(
                DATA,
                N_CHANNELS,
                MAPS,
                METHOD,
                N_STATES,
                INITIAL_CENTERS,
                SMOOTHING_KERNEL,
                REPEAT,
                TOLERANCE,
                METRIC)

        if self.ui.step2_save_clustering_checkbox.isChecked():
            save_raw_path = os.path.join(self.ui.save_dir, 'raw_features')
            if not os.path.exists(save_raw_path):
                os.makedirs(save_raw_path)
            # Save Maps
            with open(os.path.join(FOLDER, 'preprocessed_data', 'EEG_INFO.pickle'), 'rb') as p:
                eeg_info = pickle.load(p)
            save_name = os.path.join(save_raw_path, 'microstate_maps')
            maps_df = pd.DataFrame(best_maps.T, index=eeg_info.ch_names)
            maps_df.to_csv(save_name + '.csv')
            # Save Segmentation
            time = np.arange(0, (1000 / int(Fs)) * len(final_segmentation), (1000 / int(Fs)))
            segmentation_df = pd.DataFrame(final_segmentation,
                                           columns=['segmentation'],
                                           index=time)
            save_name = os.path.join(self.ui.save_dir, 'raw_features',
                                     'raw_segmentation')
            segmentation_df.to_csv(save_name + '.csv')

        # self.logclustering.append("Smoothing: "+str(SMOOTHING))
        # self.logclustering.append("Clustering Method: "+str(METHOD))
        # self.logclustering.append("Number of Maps: "+str(self.n_maps))
        # self.logclustering.append("Tolerance: "+str(TOLERANCE))
        # self.logclustering.append("Center Initializer: "+str(INITIALIZER))
        # self.logclustering.append("Metric: "+str(METRIC))
        # self.logclustering.append("Number of Repeats: "+str(REPEAT))

        # save config
        config = ConfigParser()
        config_file = os.path.join(self.ui.save_dir, 'config.ini')
        config.read(config_file)
        if config.has_section('step2'):
            config.remove_section('step2')
        config.add_section('step2')
        LENGTH_DATA_save = ','.join(map(str, LENGTH_DATA))

        config.set('step2', 'clustering_method', METHOD)
        #config.set('step2', 'option', METRIC)
        config.set('step2', 'choose_number_of_maps', str(CLUSTERS))
        config.set('step2', 'number_of_maps', str(N_STATES))
        config.set('step2', 'initializer', INITIALIZER)
        config.set('step2', 'smoothing_gfp', str(SMOOTHING))
        config.set('step2', 'smoothing_kernel_size', str(SMOOTHING_KERNEL))
        config.set('step2', 'tolerance', str(TOLERANCE))
        config.set('step2', 'concatenate_data', str(CONCATENATE))
        config.set('step2', 'length_data', LENGTH_DATA_save)
        #config.set('step2', 'number_of_repeats', str(REPEAT))
        with open(config_file, 'w+') as f:
            config.write(f)

        self.final_segmentation = final_segmentation
        self.final_maps = best_maps

        self.MicrostateDialog.save_dir = self.ui.save_dir
        self.MicrostateDialog.plot_maps(best_maps, gev, eegInfo(FOLDER))
        self.MicrostateDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.MicrostateDialog.show()
        self.done_clustering = True

    def visualize_results(self):
        Segmentation = self.final_segmentation.tolist()
        Segmentation = list(map(str, Segmentation))
        micro_labels = self.MicrostateDialog.micro_labels
        for i in range(len(micro_labels)):
            Segmentation = np.char.replace(Segmentation, str(i), micro_labels[i])

        extract_features_functions.transition_matrix(Segmentation,
                                                     visualize=True,
                                                     colormap='Blues')

    def extract_features(self):

        print("Extracting Features ...")

        '''
        if self.each_cb.isChecked():

        if self.all_cb.isChecked():
        '''

        self.ui.save_raw_path = os.path.join(self.ui.save_dir, 'raw_features')
        if not os.path.exists(self.ui.save_raw_path):
            os.makedirs(self.ui.save_raw_path)

        self.ui.save_features_path = os.path.join(self.ui.save_dir, 'extracted_features')
        if not os.path.exists(self.ui.save_features_path):
            os.makedirs(self.ui.save_features_path)

        outputformat = self.ui.step3_outputformats_combobox.currentText()
        if outputformat == "Comma-Separated Values (.csv)":
            saveformat = 'csv'
        elif outputformat == "Pickle (.pkl)":
            saveformat = 'pkl'
        elif outputformat == "Hierarchical Data Format (.h5)":
            saveformat = 'hdf'
        elif outputformat == "Java Script Object Notation (.json)":
            saveformat = 'json'

        Segmentation = self.final_segmentation.tolist()
        Segmentation = list(map(str, Segmentation))
        micro_labels = self.MicrostateDialog.micro_labels
        for i in range(len(micro_labels)):
            Segmentation = np.char.replace(Segmentation, str(i), micro_labels[i])

        Fs = int(self.ui.PreprocessDialog.downsamp_freq_input.text())
        ListOfFiles = self.listoffiles
        LengthOfFiles = self.lengthoffiles
        Extracted_Maps = self.final_maps

        if self.ui.step3_save_transitions_checkbox.isChecked():
            save_transitions = True
        else:
            save_transitions = False
        save_maps = True
        save_segmentation = True
        extract_features_functions.save_raw_results(ListOfFiles, LengthOfFiles, Fs,
                                                    save_segmentation, Segmentation,
                                                    save_maps, Extracted_Maps,
                                                    micro_labels,
                                                    save_transitions,
                                                    self.ui.save_preprocessed_path,
                                                    saveformat, self.ui.save_raw_path)

        '''
        if self.ui.step2_save_clustering_checkbox.isChecked():
            if saveformat == 'csv':
                with open(os.path.join(self.ui.foldername_preprocessed_data,"Segmentation.csv"), "w") as f:
                    write = csv.writer(f)
                    write.writerows(Segmentation)
                    #np.savetxt("Segmentation.csv", np.asarray(Segmentation), delimiter=",") 
            elif saveformat == 'pkl':
                with open(os.path.join(self.ui.foldername_preprocessed_data,"Segmentation.pkl"), "wb") as f:
                    pickle.dump(Segmentation, f)
        '''

        # print(Segmentation)

        Features = []
        if self.ui.step3_coverage_featurestoextract_checkbox.isChecked():
            Features.append("COVERAGE")
        if self.ui.step3_foc_featurestoextract_checkbox.isChecked():
            Features.append("FOC")
        if self.ui.step3_mmd_featurestoextract_checkbox.isChecked():
            Features.append("MMD")
        if self.ui.step3_tp_featurestoextract_checkbox.isChecked():
            Features.append("TP")
        if self.ui.step3_gev_featurestoextract_checkbox.isChecked():
            Features.append("GEV")

        extracted_features_df = extract_features_functions.extract_features(
            ListOfFiles, LengthOfFiles, Segmentation,
            Extracted_Maps, Fs, Features)
        # print(extracted_features_df)

        # Save Features
        extract_features_functions.save_features(extracted_features_df,
                                                 saveformat, self.ui.save_features_path)
        print('finished')

        # save config
        config = ConfigParser()
        config_file = os.path.join(self.ui.save_dir, 'config.ini')
        config.read(config_file)
        if config.has_section('step3'):
            config.remove_section('step3')
        config.add_section('step3')
        Features_save = ','.join(map(str, Features))
        config.set('step3', 'features', Features_save)
        config.set('step3', 'save_raw_segmentation', str(save_segmentation))
        config.set('step3', 'save_microstate_maps', str(save_maps))
        config.set('step3', 'save_transition_matrices', str(save_transitions))
        config.set('step3', 'output_format', str(saveformat))
        with open(config_file, 'w+') as f:
            config.write(f)

        self.done_feature_extraction = True

    def exit_msg(self, event):
        reply = QMessageBox.question(self, "Quit",
                                     "Are you sure you want to quit?",
                                     QMessageBox.Yes |
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
