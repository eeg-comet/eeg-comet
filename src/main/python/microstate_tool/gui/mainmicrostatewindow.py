import os.path
import numpy as np
import pickle
import json
import pandas as pd
import webbrowser
from configparser import ConfigParser
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox

from gui.newstudywindow import NewStudyWindow
from gui.microstatedialog import MicrostateDialog

from functions.concatenate_data import concatenate_files
from functions.clustering_functions import pre_clustering, initialize_centers, eegInfo
from functions.clustering_functions import number_of_clusters, clustering_func, clustering_minibatch
from functions.extract_features_functions import save_raw_results, extract_features, transition_matrix, save_features

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
        self.ui.showMaximized()

        self.ui.NewStudyWindow = NewStudyWindow(context)
        self.ui.MicrostateDialog = MicrostateDialog()

        self.done_preprocessing = False
        self.done_clustering = False
        self.done_feature_extraction = False
        self.ui.foldername_raw_data = ""
        self.ui.foldername_preprocessed_data = ""
        self.list_eegs = []

        self.ui.open_github_action.triggered.connect(self.open_github)
        self.ui.report_issues_action.triggered.connect(self.report_issues)
        self.ui.update_action.triggered.connect(self.update_toolbox)

        self.ui.step0_new_study_button.clicked.connect(self.open_new_study_dialog)
        self.ui.step0_load_study_button.clicked.connect(self.load_study)
        self.ui.step0_save_log_button.clicked.connect(self.save_log)

        self.ui.step2_clustermethod_combobox.activated.connect(self.mainwindow_controller)
        self.ui.step2_auto_numberofmaps_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step2_user_numberofmaps_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step2_smoothgfp_checkbox.clicked.connect(self.mainwindow_controller)

        self.ui.step2_clustering_button.clicked.connect(self.do_clustering)
        self.ui.step3_visualize_clustering_button.clicked.connect(self.visualize_results)

        self.ui.step3_extractfeatures_button.clicked.connect(self.extract_features)

        self.ui.step0_exit_button.clicked.connect(self.exit_msg)

        # Set actions

        self.ui.import_settings_action.triggered.connect(self.import_settings)
        self.ui.export_settings_action.triggered.connect(self.export_settings)

        # Initialize settings
        default_settings = [("Load", "Raw"), ("Format", ".set"), ("Type", "Continuous"), ("Files", "All"),
                            ("Method", "MODIFIED K-MEANS"), ("Choose_Maps", "User"), ("Maps", "5"),("Repeats", "5"),
                            ("Option", ""), ("Initializer", "Random"), ("Tolerance", "1E-5"), ("Smooth", "True"),
                            ("Kernel", "10"), ("Save_Raw", "True"), ("Features", ["MMD", "FOC"]), ("Output_Format", ".csv")]
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

    def update_toolbox(self):
        ret = QMessageBox.question(self, 'MessageBox', "Download toolbox?",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
        if ret == QMessageBox.Yes:
            webbrowser.open(
                'https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction/archive/refs/heads/main.zip')

    def open_new_study_dialog(self):
        self.ui.step0_log_textbrowser.clear()
        self.ui.step0_study_name_mainwin_lineedit.clear()
        self.done_preprocessing = False
        self.done_clustering = False
        self.done_feature_extraction = False
        self.ui.NewStudyWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.NewStudyWindow.showMaximized()
        self.mainwindow_controller()

    def load_study(self):
        def str2bool(v):
            return v.lower() in ("True", "yes", "1")

        self.save_dir = QFileDialog.getExistingDirectory(self, "Select the folder containing preprocessed data")
        if self.save_dir != "":
            # Load settings log
            config = ConfigParser()
            config_file = os.path.join(self.save_dir, 'settings_log.ini')
            config.read(config_file)
            self.study_name = config.get('input_settings', 'study_name')
            self.ui.step0_study_name_mainwin_lineedit.setText(self.study_name)
            self.ui.step0_log_textbrowser.insertPlainText("Study Name: " + self.study_name + "\n")
            self.ui.step0_log_textbrowser.insertPlainText(20*"* " + "\n")
            self.input_folder = config.get('input_settings', 'input_folder')
            self.save_dir = config.get('input_settings', 'save_folder')
            self.extension = config.get('input_settings', 'data_extension')
            self.data_type = config.get('input_settings', 'data_type')
            self.filter_data = config.get('input_settings', 'filter_data')
            self.filter_method = config.get('input_settings', 'filter_method')
            self.lowcut_freq = config.get('input_settings', 'lowcut_freq')
            self.highcut_freq = config.get('input_settings', 'highcut_freq')
            self.downsample_data = config.get('input_settings', 'downsample_data')
            self.sample_rate = config.get('input_settings', 'sample_rate')

            for dirpath, dirnames, filenames in os.walk(self.save_dir):
                for filename in [f for f in filenames if f.startswith("EEG_INFO")]:
                    eegInfo_path = os.path.join(dirpath, filename)

            with open(eegInfo_path, 'rb') as f:
                self.eeg_info = pickle.load(f)
            if str2bool(self.downsample_data):
                print('downsampling true')
                self.Fs = float(self.sample_rate)
            else:
                self.Fs = float(self.eeg_info['sfreq'])

            # Load data log
            config = ConfigParser()
            config_file = os.path.join(self.save_dir, 'data_log.ini')
            config.read(config_file)
            if config.has_option('input_data', 'list_eegs'):
                self.list_eegs_str = config.get('input_data', 'list_eegs')
                self.list_eegs = self.list_eegs_str.split(",")
                self.ui.step0_log_textbrowser.insertPlainText("\n"
                                                              + str(len(self.list_eegs))
                                                              + " EEG files are found:\n\n"
                                                              + "\n".join(self.list_eegs)+"\n")
                self.done_preprocessing = True
                self.listoffiles = self.list_eegs

            if self.filter_data:
                self.ui.step0_log_textbrowser.insertPlainText(
                    "\n" + "Filtered data using " + self.filter_method.upper() + " between " +
                    self.lowcut_freq + " Hz " + self.highcut_freq + " Hz\n")
            if self.downsample_data:
                self.ui.step0_log_textbrowser.insertPlainText(
                    "\n" + "Downsampled data to " + self.sample_rate + " Hz\n")

            self.ui.step0_log_textbrowser.insertPlainText("\n" + 20*"* ")
            self.ui.step0_log_textbrowser.insertPlainText("\n" + "Preprocessing is done.\n")
            self.ui.step0_log_textbrowser.insertPlainText(20*"* " + "\n")

            if config.has_section('clustering_results'):
                self.use_saved_results = True
                self.done_clustering = True
                self.lengthoffiles = config.get('input_data', 'length_data')
                self.lengthoffiles = self.lengthoffiles.split(",")
                self.lengthoffiles = list(map(int, self.lengthoffiles))
                # Load raw clustering results
                raw_results_path = os.path.join(self.save_dir, 'raw_features')
                self.final_segmentation = pd.read_csv(os.path.join(raw_results_path, 'raw_segmentation.csv'))
                self.final_segmentation = self.final_segmentation.iloc[:, 1:]
                self.final_segmentation = self.final_segmentation['segmentation'].to_list()
                self.final_maps = pd.read_csv(os.path.join(raw_results_path, 'microstate_maps.csv'))
                self.final_maps = self.final_maps.iloc[:, 1:]
                self.final_maps = self.final_maps.values.T
            if config.has_option('clustering_results', 'gev'):
                self.gev = config.get('clustering_results', 'gev')
                self.ui.step0_log_textbrowser.insertPlainText("\n" + "Global Explained Variance: " + self.gev + "\n")
                self.ui.step0_log_textbrowser.insertPlainText("\n" + 20*"* ")
                self.ui.step0_log_textbrowser.insertPlainText("\n"+ "Clustering is done.\n")
                self.ui.step0_log_textbrowser.insertPlainText(20*"* " + "\n")
            if config.has_option('clustering_results', 'micro_labels'):
                self.micro_labels = config.get('clustering_results', 'micro_labels')
                self.ui.step0_log_textbrowser.insertPlainText("\n" + "Microstate labels: " + self.micro_labels + "\n")
                self.micro_labels = self.micro_labels.split(",")

            self.mainwindow_controller()

    def save_log(self):
        text = self.step0_log_textbrowser.toPlainText()
        save_log_path = os.path.join(self.save_dir, self.study_name+'_log.txt')
        with open(save_log_path, 'w') as f:
            f.write(text)


    def mainwindow_controller(self):
        if self.done_preprocessing:

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
            self.step2_clustering_title_label.setStyleSheet("background-color: lightgreen")
            self.ui.step3_visualize_clustering_button.setEnabled(True)
            self.ui.step3_features_title_label.setEnabled(True)
            self.ui.step3_featurestoextract_label.setEnabled(True)
            self.ui.step3_outputformats_label.setEnabled(True)
            self.ui.step3_coverage_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_foc_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_mmd_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_gev_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_tp_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_complexity_featurestoextract_checkbox.setEnabled(True)
            self.ui.step3_save_segmentations_checkbox.setEnabled(True)
            self.ui.step3_save_transitions_checkbox.setEnabled(True)
            self.ui.step3_outputformats_combobox.setEnabled(True)
            self.ui.step3_extractfeatures_button.setEnabled(True)
        else:
            self.ui.step3_visualize_clustering_button.setDisabled(True)
            self.ui.step3_features_title_label.setDisabled(True)
            self.ui.step3_featurestoextract_label.setDisabled(True)
            self.ui.step3_outputformats_label.setDisabled(True)
            self.ui.step3_coverage_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_foc_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_mmd_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_gev_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_tp_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_complexity_featurestoextract_checkbox.setDisabled(True)
            self.ui.step3_save_segmentations_checkbox.setDisabled(True)
            self.ui.step3_save_transitions_checkbox.setDisabled(True)
            self.ui.step3_outputformats_combobox.setDisabled(True)
            self.ui.step3_extractfeatures_button.setDisabled(True)

        if self.done_feature_extraction:
            self.step3_features_title_label.setStyleSheet("background-color: lightgreen")

    def do_clustering(self):
        self.use_saved_results = False

        # Concatenate data
        CONCATENATE = True
        DATA, N_CHANNELS, FILENAMES, LENGTH_DATA = concatenate_files(self.save_dir)

        # Save data log
        config = ConfigParser()
        config_file = os.path.join(self.save_dir, 'data_log.ini')
        config.read(config_file)
        if config.has_option('input_data', 'length_data'):
            LENGTH_DATA_save = config.get('input_data', 'length_data')
            config.remove_option('input_data', 'length_data')
        else:
            LENGTH_DATA_save = ','.join(map(str, LENGTH_DATA))
            self.listoffiles = FILENAMES
            self.lengthoffiles = LENGTH_DATA
        config.set('input_data', 'length_data', LENGTH_DATA_save)
        with open(config_file, 'w+') as f:
            config.write(f)

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

            best_maps, final_segmentation, self.gev = clustering_minibatch(
                self.input_folder,
                self.Fs,
                SMOOTHING_KERNEL,
                N_STATES,
                INITIALIZER,
                TOLERANCE)

        else:
            MAPS, PEAKS = pre_clustering(DATA, self.Fs, SMOOTHING_KERNEL)
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

            best_maps, final_segmentation, self.gev = clustering_func(
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

        save_raw_path = os.path.join(self.save_dir, 'raw_features')
        if not os.path.exists(save_raw_path):
            os.makedirs(save_raw_path)
        # Save Maps
        #with open(os.path.join(self.input_folder, 'EEG_INFO.pickle'), 'rb') as p:
        #    eeg_info = pickle.load(p)
        save_name = os.path.join(save_raw_path, 'microstate_maps')
        maps_df = pd.DataFrame(best_maps.T, index=self.eeg_info.ch_names)
        maps_df.to_csv(save_name + '.csv')
        # Save Segmentation
        time = np.arange(0, (1000 / int(self.Fs)) * len(final_segmentation), (1000 / int(self.Fs)))
        segmentation_df = pd.DataFrame(final_segmentation,
                                       columns=['segmentation'],
                                       index=time)
        save_name = os.path.join(self.save_dir, 'raw_features',
                                 'raw_segmentation')
        segmentation_df.to_csv(save_name + '.csv')

        # Save settings log
        config = ConfigParser()
        config_file = os.path.join(self.save_dir, 'settings_log.ini')
        config.read(config_file)
        if config.has_section('clustering_settings'):
            config.remove_section('clustering_settings')
        config.add_section('clustering_settings')
        config.set('clustering_settings', 'clustering_method', METHOD)
        config.set('clustering_settings', 'option', METRIC)
        config.set('clustering_settings', 'choose_number_of_maps', str(CLUSTERS))
        config.set('clustering_settings', 'number_of_maps', str(N_STATES))
        config.set('clustering_settings', 'initializer', INITIALIZER)
        config.set('clustering_settings', 'smoothing_gfp', str(SMOOTHING))
        config.set('clustering_settings', 'smoothing_kernel_size', str(SMOOTHING_KERNEL))
        config.set('clustering_settings', 'tolerance', str(TOLERANCE))
        config.set('clustering_settings', 'concatenate_data', str(CONCATENATE))
        config.set('clustering_settings', 'number_of_repeats', str(REPEAT))
        with open(config_file, 'w+') as f:
            config.write(f)

        self.final_segmentation = final_segmentation
        self.final_maps = best_maps

        self.MicrostateDialog.save_dir = self.save_dir
        self.MicrostateDialog.plot_maps(best_maps, self.gev, self.eeg_info)
        self.MicrostateDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.MicrostateDialog.showMaximized()
        self.done_clustering = True
        self.use_saved_results = True
        self.ui.step0_log_textbrowser.insertPlainText("\n" + "Global Explained Variance: " + str(self.gev) + "\n")
        self.ui.step0_log_textbrowser.insertPlainText("\n" + 20 * "* ")
        self.ui.step0_log_textbrowser.insertPlainText("\n" + "Clustering is done.\n")
        self.ui.step0_log_textbrowser.insertPlainText(20 * "* " + "\n")

        self.mainwindow_controller()

    def visualize_results(self):
        if self.use_saved_results:
            Segmentation = self.final_segmentation
        else:
            Segmentation = self.final_segmentation.tolist()
            Segmentation = list(map(str, Segmentation))
            self.micro_labels = self.MicrostateDialog.micro_labels
            for i in range(len(self.micro_labels)):
                Segmentation = np.char.replace(Segmentation, str(i), self.micro_labels[i])
        transition_matrix(Segmentation, visualize=True, colormap='Blues')

    def extract_features(self):
        print("Extracting Features ...")

        self.ui.save_raw_path = os.path.join(self.save_dir, 'raw_features')
        if not os.path.exists(self.ui.save_raw_path):
            os.makedirs(self.ui.save_raw_path)

        self.ui.save_features_path = os.path.join(self.save_dir, 'extracted_features')
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

        if self.use_saved_results:
            Segmentation = self.final_segmentation
        else:
            Segmentation = self.final_segmentation.tolist()
            self.micro_labels = self.MicrostateDialog.micro_labels

        Segmentation = list(map(str, Segmentation))
        for i in range(len(self.micro_labels)):
            Segmentation = np.char.replace(Segmentation, str(i), self.micro_labels[i])

        if self.ui.step3_save_segmentations_checkbox.isChecked():
            save_segmentation = True
        else:
            save_segmentation = False
        if self.ui.step3_save_transitions_checkbox.isChecked():
            save_transitions = True
        else:
            save_transitions = False
        save_maps = True

        save_raw_results(self.listoffiles, self.lengthoffiles, self.Fs,
                            save_segmentation, Segmentation,
                            save_maps, self.final_maps,
                            self.micro_labels,
                            save_transitions,
                            self.save_dir,
                            saveformat, self.ui.save_raw_path)


        Features = []
        if self.ui.step3_coverage_featurestoextract_checkbox.isChecked():
            Features.append("COVERAGE")
        if self.ui.step3_foc_featurestoextract_checkbox.isChecked():
            Features.append("FOC")
        if self.ui.step3_mmd_featurestoextract_checkbox.isChecked():
            Features.append("MMD")
        if self.ui.step3_tp_featurestoextract_checkbox.isChecked():
            Features.append("TP")
        if self.ui.step3_complexity_featurestoextract_checkbox.isChecked():
            Features.append("LZC")
        if self.ui.step3_gev_featurestoextract_checkbox.isChecked():
            Features.append("GEV")

        extracted_features_df = extract_features( self.listoffiles, self.lengthoffiles,
                                                  Segmentation, self.final_maps, self.Fs, Features)

        # Save Features
        save_features(extracted_features_df, saveformat, self.ui.save_features_path)
        print('finished')

        # save config
        config = ConfigParser()
        config_file = os.path.join(self.save_dir, 'settings_log.ini')
        config.read(config_file)
        if config.has_section('features_settings'):
            config.remove_section('features_settings')
        config.add_section('features_settings')
        Features_save = ','.join(map(str, Features))
        config.set('features_settings', 'features', Features_save)
        config.set('features_settings', 'save_raw_segmentation', str(save_segmentation))
        config.set('features_settings', 'save_microstate_maps', str(save_maps))
        config.set('features_settings', 'save_transition_matrices', str(save_transitions))
        config.set('features_settings', 'output_format', str(saveformat))
        with open(config_file, 'w+') as f:
            config.write(f)

        self.done_feature_extraction = True
        self.ui.step0_log_textbrowser.insertPlainText("\n" + 20 * "* ")
        self.ui.step0_log_textbrowser.insertPlainText("\n" + "Features are extracted.\n")

        self.mainwindow_controller()

    def exit_msg(self, event):
        reply = QMessageBox.question(self, "Quit",
                                     "Are you sure you want to quit?",
                                     QMessageBox.Yes |
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
