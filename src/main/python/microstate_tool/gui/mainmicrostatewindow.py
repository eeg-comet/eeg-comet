import os.path
import numpy as np
import json
import pandas as pd
import webbrowser
from matplotlib import pyplot as plt
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox

from gui.newstudywindow import NewStudyWindow
from gui.numbermapsdialog import NumberMapsDialog
from gui.microstatedialog import MicrostateDialog
from gui.visualizationdialog import VisualizationDialog
from gui.microsegdialog import MicroSegDialog


from functions.utils.load_save_config import load_config, save_config
from functions.utils.load_save_eeg_info import load_eeg_info
from functions.utils.import_hdf_data import import_hdf_data
from functions.utils.find_data import find_data
from functions.utils.backfit_func import backfit_func
from functions.utils.micro_segments_data import micro_segments_data
from functions.clustering_functions import number_of_clusters, clustering_func
from functions.modified_kmeans import run_minibatch_modified_kmeans
from functions.features.extract_features_functions import extract_segments, save_segmentation_results,\
    save_transitions, save_raw_results, extract_features, transition_matrix, save_features
from functions.features.source_localization_tess import run_source_localization, visualize_sources

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
        self.ui.NumberMapsDialog = NumberMapsDialog(context)
        self.ui.MicrostateDialog = MicrostateDialog()
        self.ui.MicroSegDialog = MicroSegDialog()
        self.ui.VisualizationDialog = VisualizationDialog(context)

        self.done_preprocessing = False
        self.done_clustering = False
        self.done_labeling_microstates = False
        self.done_backfitting = False
        self.done_extracting_features = False
        self.done_source_localization = False
        self.done_extracting_microsegments = False
        self.ui.foldername_raw_data = ""
        self.ui.foldername_preprocessed_data = ""

        self.ui.open_github_action.triggered.connect(self.open_github)
        self.ui.report_issues_action.triggered.connect(self.report_issues)
        self.ui.update_action.triggered.connect(self.update_toolbox)

        self.ui.step0_new_study_action.triggered.connect(self.open_new_study_dialog)
        self.ui.step0_new_study_button.clicked.connect(self.open_new_study_dialog)
        self.ui.step0_load_study_action.triggered.connect(self.load_study)
        self.ui.step0_load_study_button.clicked.connect(self.load_study)
        self.ui.step0_save_log_button.clicked.connect(self.save_log)

        self.ui.step2_clustermethod_combobox.activated.connect(self.mainwindow_controller)
        self.ui.step2_auto_numberofmaps_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step2_user_numberofmaps_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step3_backfit_all_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step3_backfit_peaks_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step3_filter_segments_checkbox.clicked.connect(self.mainwindow_controller)

        self.ui.step2_numberofmaps_elbow_button.clicked.connect(self.plot_elbow)
        self.ui.step2_clustering_button.clicked.connect(self.do_clustering)
        self.ui.step3_label_maps_button.clicked.connect(self.label_maps)
        self.ui.step3_backfit_button.clicked.connect(self.do_backfitting)
        self.ui.step4_extractfeatures_button.clicked.connect(self.extract_features)
        self.ui.step4_visualizefeatures_button.clicked.connect(self.open_visualize_features_dialog)
        self.ui.step5_estimate_sources_button.clicked.connect(self.source_localize_microstates)
        self.ui.step5_visualize_sources_button.clicked.connect(self.visualize_source_localized_microstates)
        self.ui.step6_extract_microseg_button.clicked.connect(self.extract_microsegments)
        self.ui.step4_visualize_sensor_microseg_button.clicked.connect(self.visualize_microsegments)

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
        self.done_labeling_microstates = False
        self.done_backfitting = False
        self.done_extracting_features = False
        self.done_source_localization = False
        self.done_extracting_microsegments = False
        self.ui.NewStudyWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.NewStudyWindow.showMaximized()
        self.mainwindow_controller()

    def open_visualize_features_dialog(self):
        self.ui.VisualizationDialog.extracted_features_path = self.extracted_features_path
        self.ui.VisualizationDialog.save_folder = self.save_folder
        self.ui.VisualizationDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.VisualizationDialog.showMaximized()
        self.ui.VisualizationDialog.load_filenames()
        self.mainwindow_controller()

    def load_study(self):

        self.save_folder = QFileDialog.getExistingDirectory(self, "Select the folder containing preprocessed data")
        config_file = os.path.join(self.save_folder, 'log.ini')
        if not os.path.exists(config_file):
            QMessageBox.information(self, "Load error",
                                    "The selected folder does not contain a valid study!",
                                    QMessageBox.Ok)
        else:
            # Define global directories
            self.preprocessed_data_path = os.path.join(self.save_folder, 'preprocessed_data')
            self.raw_features_path = os.path.join(self.save_folder, 'raw_features')
            #self.raw_segmentation_path = os.path.join(self.raw_features_path, 'raw_segmentation')
            self.raw_transitions_path = os.path.join(self.raw_features_path, 'raw_transitions')
            self.extracted_features_path = os.path.join(self.save_folder, 'extracted_features')
            self.micro_segments_path = os.path.join(self.save_folder, 'micro_segments')
            self.eeg_info_path = os.path.join(self.save_folder, 'eeg_info.pkl')
            self.final_maps_path = os.path.join(self.raw_features_path, 'microstate_maps.csv')
            self.localized_sources_path = os.path.join(self.raw_features_path, 'localized_sources')
            self.stc_path = os.path.join(self.localized_sources_path, 'stc_data.npy')

            # Load config
            config = load_config(config_file)
            # Load "progress" from config
            self.done_preprocessing = config.getboolean('progress', 'done_preprocessing')
            self.done_clustering = config.getboolean('progress', 'done_clustering')
            self.done_labeling_microstates = config.getboolean('progress', 'done_labeling_microstates')
            self.done_backfitting = config.getboolean('progress', 'done_backfitting')
            self.done_extracting_features = config.getboolean('progress', 'done_extracting_features')
            self.done_extracting_microsegments = config.getboolean('progress', 'done_extracting_microsegments')
            self.done_source_localization = config.getboolean('progress', 'done_source_localization')

            # Load preprocessing information
            if self.done_preprocessing:
                # Load "study info" from config
                self.study_name = config['study info']['study_name']
                self.input_folder = config['study info']['input_folder']
                self.extension = config['study info']['input_data_extension']
                self.data_type = config['study info']['input_data_type']
                filenames_str = config['study info']['input_filenames']
                self.list_eegs = filenames_str.split(",")
                self.save_folder = config['study info']['save_folder']
                # Load "preprocessing settings" from config
                self.filter_data = config.getboolean('preprocessing settings', 'filter_data')
                if self.filter_data:
                    self.filter_method = config['preprocessing settings']['filter_method']
                    self.lowcut_freq = config.getint('preprocessing settings', 'lowcut_freq')
                    self.highcut_freq = config.getint('preprocessing settings', 'highcut_freq')
                self.downsample_data = config.getboolean('preprocessing settings', 'downsample_data')
                if self.downsample_data:
                    self.sample_rate = config.getint('preprocessing settings', 'sample_rate')
                else:
                    self.eeg_info = load_eeg_info(self.eeg_info_path)
                    self.sample_rate = self.eeg_info['sfreq']

                self.channels2remove = config['preprocessing settings']['channels2remove']
                # Load "preprocessing results" from config
                length_data = config['preprocessing results']['length_data']
                length_data = length_data.split(",")
                length_data = list(map(float, length_data))
                self.length_data = list(map(int, length_data))
                self.n_chan = config.getint('preprocessing results', 'n_chan')
                ch_names_str = config['preprocessing results']['ch_names']
                self.ch_names = ch_names_str.split(",")

                self.hdf_concatenated_data_path = os.path.join(self.save_folder, self.study_name+'_concatenated_data.hdf')
                self.hf_segmentation_path = os.path.join(self.raw_features_path, self.study_name+'_segmentation.hdf')
                ###
                #self.listofh5files = find_data(self.preprocessed_data_path, '.h5', '*')

                # Load EEG info
                #self.eeg_info = load_eeg_info(self.eeg_info_path)

                # Show study name on the main window
                self.ui.step0_study_name_mainwin_lineedit.setText(self.study_name)
                self.ui.step0_log_textbrowser.insertPlainText("Study Name: " + self.study_name + "\n")
                self.ui.step0_log_textbrowser.insertPlainText(20 * "* " + "\n")


                #if self.downsample_data:
                #    self.sample_rate = self.sample_rate
                #else:
                #    self.sample_rate = self.eeg_info['sfreq']

                self.ui.step0_log_textbrowser.insertPlainText("\n"
                                                              + str(len(self.list_eegs))
                                                              + " EEG files are found:\n\n"
                                                              + "\n".join(self.list_eegs)+"\n")

                if self.filter_data:
                    self.ui.step0_log_textbrowser.insertPlainText(
                        "\n" + "Filtered data using " + self.filter_method.upper() + " between " +
                        str(self.lowcut_freq) + " Hz " + str(self.highcut_freq) + " Hz\n")
                if self.downsample_data:
                    self.ui.step0_log_textbrowser.insertPlainText(
                        "\n" + "Downsampled data to " + str(self.sample_rate) + " Hz\n")
                self.ui.step0_log_textbrowser.insertPlainText("\n" + 20*"* ")
                self.ui.step0_log_textbrowser.insertPlainText("\n" + "Preprocessing is done.\n")
                self.ui.step0_log_textbrowser.insertPlainText(20*"* " + "\n")

            # Load clustering information
            if self.done_clustering:
                self.use_saved_results = True
                # Load "clustering settings" from config
                self.clustering_method = config['clustering settings']['clustering_method']
                self.clustering_option = config['clustering settings']['clustering_option']
                self.choose_number_of_maps = config['clustering settings']['choose_number_of_maps']
                self.number_of_maps = config.getint('clustering settings', 'number_of_maps')
                self.initializer = config['clustering settings']['initializer']
                self.smoothing_gfp = config.getboolean('clustering settings', 'smoothing_gfp')
                self.min_distance_size = config.getint('clustering settings', 'min_distance_size')
                self.clustering_tolerance = config.getfloat('clustering settings', 'clustering_tolerance')
                self.number_of_repeats = config.getint('clustering settings', 'number_of_repeats')
                # Load "clustering results" from config
                self.gev = config.getfloat('clustering results', 'gev')
                if self.done_labeling_microstates:
                    micro_labels_str = config['clustering results']['micro_labels']
                    self.micro_labels = micro_labels_str.split(",")

                    self.ui.step0_log_textbrowser.insertPlainText(
                        "\n" + "Microstate labels: " + micro_labels_str + "\n")

                # Load raw clustering results
                self.final_maps_path = os.path.join(self.raw_features_path, 'microstate_maps.csv')

                if os.path.exists(self.final_maps_path):
                    self.final_maps = pd.read_csv(self.final_maps_path)
                    self.final_maps = self.final_maps.iloc[:, 1:]
                    self.final_maps = self.final_maps.values.T


                self.ui.step0_log_textbrowser.insertPlainText("\n" + "Global Explained Variance: " + str(self.gev) + "\n")
                self.ui.step0_log_textbrowser.insertPlainText("\n" + 20*"* ")
                self.ui.step0_log_textbrowser.insertPlainText("\n"+ "Clustering is done.\n")
                self.ui.step0_log_textbrowser.insertPlainText(20*"* " + "\n")

                # Load "backfitting settings" from config
                if self.done_backfitting:
                    self.backfit_to = config['backfitting settings']['backfit_to']
                    self.output_format = config['backfitting settings']['output_format']
                    self.filter_segments = config.getboolean('backfitting settings', 'filter_segments')
                    if self.filter_segments:
                        self.filter_segments_option = config['backfitting settings']['filter_segments_option']
                        self.remove_segments_less_than = config.getint('backfitting settings',
                                                                       'remove_segments_less_than')
                    else:
                        self.filter_segments_option = []
                        self.remove_segments_less_than = []

                    if self.done_extracting_features:
                        # Load "feature extraction settings" from config
                        features2extract_str = config['feature extraction settings']['features2extract']
                        self.features2extract = features2extract_str.split(",")


                if self.done_source_localization:
                    with open(self.stc_path, 'rb') as f:
                        self.stc_data = np.load(f)
                    # Load "source localization settings" from config
                    self.inverse_method = config['source localization settings']['inverse_method']
                    self.nperm = config['source localization settings']['permutations']
                    self.spacing = config['source localization settings']['spacing']


            # Update GUI
            self.update_mainwindow_gui()

            self.mainwindow_controller()

    def save_log(self):
        text = self.step0_log_textbrowser.toPlainText()
        save_log_path = os.path.join(self.save_folder, self.study_name+'_log.txt')
        with open(save_log_path, 'w') as f:
            f.write(text)


    def mainwindow_controller(self):
        if self.done_preprocessing:
            self.ui.step0_study_name_mainwin_lineedit.setStyleSheet("background-color: lightgreen")
            self.ui.step2_clustering_title_label.setEnabled(True)
            self.ui.step2_clustermethod_combo_label.setEnabled(True)
            self.ui.step2_clustermethod_combobox.setEnabled(True)
            self.ui.step2_numberofmaps_label.setEnabled(True)
            self.ui.step2_auto_numberofmaps_radio.setEnabled(True)
            self.ui.step2_user_numberofmaps_radio.setEnabled(True)
            self.ui.step2_numberofmaps_elbow_button.setEnabled(True)
            if self.ui.step2_auto_numberofmaps_radio.isChecked():
                self.ui.step2_user_numberofmaps_input.setDisabled(True)
            if self.ui.step2_user_numberofmaps_radio.isChecked():
                self.ui.step2_user_numberofmaps_input.setEnabled(True)
            self.ui.step2_kernel_size_input.setEnabled(True)

            self.ui.step2_numberofrepeats_label.setEnabled(True)
            self.ui.step2_user_numberofrepeats_input.setEnabled(True)
            self.ui.step2_other_label.setEnabled(True)
            self.ui.step2_other_options_combobox.setEnabled(True)
            self.ui.step2_initializer_label.setEnabled(True)
            self.ui.step2_random_initializer_radio.setEnabled(True)
            self.ui.step2_kmeans_initializer_radio.setEnabled(True)
            self.ui.step2_stopcondition_label.setEnabled(True)
            self.ui.step2_stopcondition_input.setEnabled(True)
            self.ui.step2_kernel_size_label.setEnabled(True)
            self.ui.step2_kernel_size_input.setEnabled(True)
            self.ui.step2_kernel_size_label_2.setEnabled(True)
            self.ui.step2_performclustering_label.setEnabled(True)
            self.ui.step2_performclustering_concat_radio.setEnabled(True)
            self.ui.step2_performclustering_each_radio.setEnabled(True)
            self.ui.step2_clustering_button.setEnabled(True)

        else:
            self.ui.step0_study_name_mainwin_lineedit.setStyleSheet("background-color: none")
            self.ui.step2_clustering_title_label.setDisabled(True)
            self.ui.step2_clustermethod_combo_label.setDisabled(True)
            self.ui.step2_clustermethod_combobox.setDisabled(True)
            self.ui.step2_numberofmaps_label.setDisabled(True)
            self.ui.step2_auto_numberofmaps_radio.setDisabled(True)
            self.ui.step2_user_numberofmaps_radio.setDisabled(True)
            self.ui.step2_user_numberofmaps_input.setDisabled(True)
            self.ui.step2_numberofmaps_elbow_button.setDisabled(True)
            self.ui.step2_numberofrepeats_label.setDisabled(True)
            self.ui.step2_user_numberofrepeats_input.setDisabled(True)
            self.ui.step2_other_label.setDisabled(True)
            self.ui.step2_other_options_combobox.setDisabled(True)
            self.ui.step2_initializer_label.setDisabled(True)
            self.ui.step2_random_initializer_radio.setDisabled(True)
            self.ui.step2_kmeans_initializer_radio.setDisabled(True)
            self.ui.step2_stopcondition_label.setDisabled(True)
            self.ui.step2_stopcondition_input.setDisabled(True)
            self.ui.step2_kernel_size_label.setDisabled(True)
            self.ui.step2_kernel_size_input.setDisabled(True)
            self.ui.step2_kernel_size_label_2.setDisabled(True)
            self.ui.step2_performclustering_label.setDisabled(True)
            self.ui.step2_performclustering_concat_radio.setDisabled(True)
            self.ui.step2_performclustering_each_radio.setDisabled(True)
            self.ui.step2_clustering_button.setDisabled(True)

        self.clustering_method = self.step2_clustermethod_combobox.currentText()
        if self.clustering_method == "K-means":
            self.ui.step2_other_label.setText("K-means distance metric:")
            self.ui.step2_other_options_combobox.clear()
            self.ui.step2_other_options_combobox.addItem("Euclidean")
            self.ui.step2_other_options_combobox.addItem("Euclidean Square")
            self.ui.step2_other_options_combobox.addItem("Cosine Similarity")
            self.ui.step2_other_options_combobox.addItem("Spatial Correlation")
            self.ui.step2_other_options_combobox.setCurrentText("Cosine Similarity")
        elif self.clustering_method == "Agglomerative hierarchical clustering":
            self.ui.step2_other_label.setText("Type of link between clusters:")
            self.ui.step2_other_options_combobox.clear()
            self.ui.step2_other_options_combobox.addItem("Single Link")
            self.ui.step2_other_options_combobox.addItem("Complete Link")
            self.ui.step2_other_options_combobox.addItem("Average Link")
            self.ui.step2_other_options_combobox.addItem("Centroid Link")
            self.ui.step2_other_options_combobox.setCurrentText("Single Link")
        elif self.clustering_method == "X-means":
            self.ui.step2_other_label.setText("X-means splitting criterion:")
            self.ui.step2_other_options_combobox.clear()
            self.ui.step2_other_options_combobox.addItem("Bayesian Information Criterion")
            self.ui.step2_other_options_combobox.addItem("Minimum Noiseless Description Length")
            self.ui.step2_other_options_combobox.setCurrentText("Bayesian Information Criterion")
        else:
            self.ui.step2_other_label.setText("Other options:")
            self.ui.step2_other_options_combobox.clear()

        if self.done_clustering:
            self.ui.step2_clustering_button.setStyleSheet("background-color: lightgreen")

            outputformat = self.ui.step4_outputformats_combobox.currentText()
            if outputformat == "Comma-Separated Values (.csv)":
                self.output_format = '.csv'
            elif outputformat == "Pickle (.pkl)":
                self.output_format = '.pkl'
            elif outputformat == "Hierarchical Data Format (.hdf)":
                self.output_format = '.hdf'
            elif outputformat == "Java Script Object Notation (.json)":
                self.output_format = '.json'

            self.ui.step3_label_maps_button.setEnabled(True)
            self.ui.step3_backfit_title_label.setEnabled(True)
            self.ui.step3_backfit_all_radio.setEnabled(True)
            self.ui.step3_backfit_peaks_radio.setEnabled(True)
            self.ui.step3_backfit_button.setEnabled(True)

            if self.ui.step3_backfit_all_radio.isChecked():
                self.ui.step3_filter_segments_checkbox.setEnabled(True)
                if self.ui.step3_filter_segments_checkbox.isChecked():
                    self.ui.step3_filter_segments_input.setEnabled(True)
                    self.ui.step3_filter_segments_label.setEnabled(True)
                    self.ui.step3_replace_segments_radio.setEnabled(True)
                    self.ui.step3_remove_segments_radio.setEnabled(True)
                elif not self.ui.step3_filter_segments_checkbox.isChecked():
                    self.ui.step3_filter_segments_input.setDisabled(True)
                    self.ui.step3_filter_segments_label.setDisabled(True)
                    self.ui.step3_replace_segments_radio.setDisabled(True)
                    self.ui.step3_remove_segments_radio.setDisabled(True)
            if self.ui.step3_backfit_peaks_radio.isChecked():
                self.ui.step3_filter_segments_input.setDisabled(True)
                self.ui.step3_filter_segments_label.setDisabled(True)
                self.ui.step3_filter_segments_checkbox.setDisabled(True)


        else:
            self.done_labeling_microstates = False
            self.done_backfitting = False
            self.done_extracting_features = False
            self.done_source_localization = False
            self.done_extracting_microsegments = False

            self.ui.step2_clustering_button.setStyleSheet("background-color: none")
            self.ui.step3_label_maps_button.setDisabled(True)
            self.ui.step3_backfit_title_label.setDisabled(True)
            self.ui.step3_backfit_all_radio.setDisabled(True)
            self.ui.step3_backfit_peaks_radio.setDisabled(True)
            self.ui.step3_backfit_button.setDisabled(True)
            self.ui.step3_label_maps_button.setDisabled(True)
            self.ui.step3_filter_segments_checkbox.setDisabled(True)
            self.ui.step3_filter_segments_input.setDisabled(True)
            self.ui.step3_filter_segments_label.setDisabled(True)
            self.ui.step3_replace_segments_radio.setDisabled(True)
            self.ui.step3_remove_segments_radio.setDisabled(True)

        if self.done_labeling_microstates:
            self.ui.step3_label_maps_button.setStyleSheet("background-color: lightgreen")
        else:
            self.ui.step3_label_maps_button.setStyleSheet("background-color: none")
            self.done_backfitting = False
            self.done_extracting_features = False
            self.done_source_localization = False
            self.done_extracting_microsegments = False

        if self.done_backfitting:
            self.ui.step3_backfit_button.setStyleSheet("background-color: lightgreen")
            self.ui.step4_features_title_label.setEnabled(True)
            self.ui.step4_featurestoextract_label.setEnabled(True)
            self.ui.step4_coverage_featurestoextract_checkbox.setEnabled(True)
            self.ui.step4_foc_featurestoextract_checkbox.setEnabled(True)
            self.ui.step4_mmd_featurestoextract_checkbox.setEnabled(True)
            self.ui.step4_gev_featurestoextract_checkbox.setEnabled(True)
            self.ui.step4_tp_featurestoextract_checkbox.setEnabled(True)
            self.ui.step4_complexity_featurestoextract_checkbox.setEnabled(True)
            self.ui.step4_extractfeatures_button.setEnabled(True)
            self.ui.step4_outputformats_label.setEnabled(True)
            self.ui.step4_outputformats_combobox.setEnabled(True)
            self.ui.step6_extract_microseg_button.setEnabled(True)
            self.ui.step6_extract_microsegments_title_label.setEnabled(True)
            self.ui.step5_source_localization_title_label.setEnabled(True)
            self.ui.step5_inverse_method_label.setEnabled(True)
            self.ui.step5_inverse_method_combobox.setEnabled(True)
            self.ui.step5_permutations_label.setEnabled(True)
            self.ui.step5_permutations_input.setEnabled(True)
            self.ui.step5_spacing_label.setEnabled(True)
            self.ui.step5_spacing_combobox.setEnabled(True)
            self.ui.step5_estimate_sources_button.setEnabled(True)

        else:
            self.done_extracting_features = False
            self.done_extracting_microsegments = False

            self.ui.step3_backfit_button.setStyleSheet("background-color: none")
            self.ui.step4_features_title_label.setDisabled(True)
            self.ui.step4_featurestoextract_label.setDisabled(True)
            self.ui.step4_coverage_featurestoextract_checkbox.setDisabled(True)
            self.ui.step4_foc_featurestoextract_checkbox.setDisabled(True)
            self.ui.step4_mmd_featurestoextract_checkbox.setDisabled(True)
            self.ui.step4_gev_featurestoextract_checkbox.setDisabled(True)
            self.ui.step4_tp_featurestoextract_checkbox.setDisabled(True)
            self.ui.step4_complexity_featurestoextract_checkbox.setDisabled(True)
            self.ui.step4_extractfeatures_button.setDisabled(True)
            self.ui.step4_outputformats_label.setDisabled(True)
            self.ui.step4_outputformats_combobox.setDisabled(True)
            self.ui.step6_extract_microseg_button.setDisabled(True)
            self.ui.step6_extract_microsegments_title_label.setDisabled(True)
            self.ui.step5_source_localization_title_label.setDisabled(True)
            self.ui.step5_inverse_method_label.setDisabled(True)
            self.ui.step5_inverse_method_combobox.setDisabled(True)
            self.ui.step5_permutations_label.setDisabled(True)
            self.ui.step5_permutations_input.setDisabled(True)
            self.ui.step5_spacing_label.setDisabled(True)
            self.ui.step5_spacing_combobox.setDisabled(True)
            self.ui.step5_estimate_sources_button.setDisabled(True)

        if self.done_extracting_features:
            self.ui.step4_extractfeatures_button.setStyleSheet("background-color: lightgreen")
            self.ui.step4_visualizefeatures_button.setEnabled(True)
        else:
            self.ui.step4_extractfeatures_button.setStyleSheet("background-color: none")
            self.ui.step4_visualizefeatures_button.setDisabled(True)

        if self.done_extracting_microsegments:
            self.ui.step6_extract_microseg_button.setStyleSheet("background-color: lightgreen")
            self.ui.step4_visualize_sensor_microseg_button.setEnabled(True)
        else:
            self.ui.step6_extract_microseg_button.setStyleSheet("background-color: none")
            self.ui.step4_visualize_sensor_microseg_button.setDisabled(True)

        if self.done_source_localization:
            self.ui.step5_estimate_sources_button.setStyleSheet("background-color: lightgreen")
            self.ui.step5_visualize_sources_button.setEnabled(True)
            self.ui.step6_visualize_source_microseg_button.setEnabled(True)
        else:
            self.ui.step5_estimate_sources_button.setStyleSheet("background-color: none")
            self.ui.step5_visualize_sources_button.setDisabled(True)
            self.ui.step6_visualize_source_microseg_button.setDisabled(True)

    def update_mainwindow_gui(self):
        if self.done_clustering:
            self.ui.step2_clustermethod_combobox.setCurrentText(self.clustering_method)
            #self.clustering_option
            if self.choose_number_of_maps == 'Auto':
                self.ui.step2_auto_numberofmaps_radio.setChecked(True)
            elif self.choose_number_of_maps == 'User':
                self.ui.step2_user_numberofmaps_radio.setChecked(True)
            self.ui.step2_user_numberofmaps_input.setText(str(self.number_of_maps))
            if self.initializer == 'Random':
                self.ui.step2_random_initializer_radio.setChecked(True)
            elif self.initializer == 'K-Means++':
                self.ui.step2_kmeans_initializer_radio.setChecked(True)
            #self.smoothing_gfp
            self.ui.step2_kernel_size_input.setText(str(self.min_distance_size))
            self.ui.step2_stopcondition_input.setText(str(self.clustering_tolerance))
            self.ui.step2_user_numberofrepeats_input.setText(str(self.number_of_repeats))

        if self.done_backfitting:
            if self.filter_segments:
                self.ui.step3_filter_segments_checkbox.setChecked(True)
            else:
                self.ui.step3_filter_segments_checkbox.setChecked(False)
            if self.backfit_to == 'all':
                self.ui.step3_backfit_all_radio.setChecked(True)
            elif self.backfit_to == 'peaks':
                self.ui.step3_backfit_peaks_radio.setChecked(True)
            #self.ui.step4_outputformats_combobox.setCurrentText(self.output_format)
        if self.done_extracting_features:
            if 'COV' in self.features2extract:
                self.ui.step4_coverage_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_coverage_featurestoextract_checkbox.setChecked(False)
            if 'MMD' in self.features2extract:
                self.ui.step4_mmd_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_mmd_featurestoextract_checkbox.setChecked(False)
            if 'FOC' in self.features2extract:
                self.ui.step4_foc_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_foc_featurestoextract_checkbox.setChecked(False)
            if 'GEV' in self.features2extract:
                self.ui.step4_gev_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_gev_featurestoextract_checkbox.setChecked(False)
            if 'TP' in self.features2extract:
                self.ui.step4_tp_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_tp_featurestoextract_checkbox.setChecked(False)
            if 'LZC' in self.features2extract:
                self.ui.step4_complexity_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_complexity_featurestoextract_checkbox.setChecked(False)

    def plot_elbow(self):
        self.NumberMapsDialog.data = import_hdf_data(self.hdf_concatenated_data_path)
        self.NumberMapsDialog.min_distance_size = int(int(self.ui.step2_kernel_size_input.text())/(1000/self.sample_rate))
        self.NumberMapsDialog.clustering_tolerance = float(self.ui.step2_stopcondition_input.text())
        self.NumberMapsDialog.number_of_repeats = int(self.ui.step2_user_numberofrepeats_input.text())
        self.NumberMapsDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.NumberMapsDialog.showMaximized()

    def do_clustering(self):

        # Check if the analysis is already done.
        if self.done_clustering:
            ret = QMessageBox.question(self, 'MessageBox', "Data has been clustered once,"
                                                           " do you want to redo the analysis?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_clustering_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_clustering_from_scratch = True

                # Remove the previous log
                config_file = os.path.join(self.save_folder, 'log.ini')
                config = load_config(config_file)
                config['clustering settings'] = {}
                config['clustering results'] = {}
                config['backfitting settings'] = {}
                config['feature extraction settings'] = {}
                config['feature visualization groups'] = {}
                config['source localization settings'] = {}
                save_config(config_file, config)

        else:
            self.do_clustering_from_scratch = True

        if self.do_clustering_from_scratch:
            self.done_clustering = False
            self.done_labeling_microstates = False
            self.done_backfitting = False
            self.done_extracting_features = False
            self.done_extracting_microsegments = False
            self.done_source_localization = False
            # Remove the previous log
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['progress']['done_clustering'] = str(self.done_clustering)
            config['progress']['done_labeling_microstates'] = str(self.done_labeling_microstates)
            config['progress']['done_backfitting'] = str(self.done_backfitting)
            config['progress']['done_extracting_features'] = str(self.done_extracting_features)
            config['progress']['done_extracting_microsegments'] = str(self.done_extracting_microsegments)
            config['progress']['done_source_localization'] = str(self.done_source_localization)
            config['clustering settings'] = {}
            config['clustering results'] = {}
            config['backfitting settings'] = {}
            config['feature extraction settings'] = {}
            config['feature visualization groups'] = {}
            config['source localization settings'] = {}
            save_config(config_file, config)
            self.mainwindow_controller()

            self.use_saved_results = False
            if not os.path.exists(self.raw_features_path):
                os.makedirs(self.raw_features_path)

            # Concatenate data
            if self.ui.step2_performclustering_concat_radio.isChecked():
                self.concat_data = True
            elif self.ui.step2_performclustering_each_radio.isChecked():
                self.concat_data = False

            if self.ui.step2_kernel_size_input.text():
                self.smoothing_gfp = True
                self.min_distance_size = int(int(self.ui.step2_kernel_size_input.text())/(1000/self.sample_rate))
            else:
                self.smoothing_gfp = False
                self.min_distance_size = []
            #print(self.min_distance_size)

            if self.ui.step2_auto_numberofmaps_radio.isChecked():
                self.choose_number_of_maps = "auto"
            elif self.ui.step2_user_numberofmaps_radio.isChecked():
                self.choose_number_of_maps = "user"
                self.number_of_maps = int(self.ui.step2_user_numberofmaps_input.text())
            #print(self.number_of_maps)

            if self.ui.step2_random_initializer_radio.isChecked():
                self.initializer = "Random"
            elif self.ui.step2_kmeans_initializer_radio.isChecked():
                self.initializer = "K-Means++"

            self.clustering_method = self.ui.step2_clustermethod_combobox.currentText()
            #print(self.clustering_method)

            self.clustering_tolerance = float(self.ui.step2_stopcondition_input.text())
            #print(self.clustering_tolerance)

            self.clustering_option = self.ui.step2_other_options_combobox.currentText()
            #print(self.clustering_option)

            self.number_of_repeats = int(self.ui.step2_user_numberofrepeats_input.text())

            if self.concat_data:
                #if not self.concat_data_available:
                #    HF, PREPROCESSED_DATA, N_CHANNELS, FILENAMES = concatenate_files(self.study_name, self.save_folder)
                #    self.concatenated_data = PREPROCESSED_DATA
                print("Loading the concatenated data ...")
                self.concatenated_data = import_hdf_data(self.hdf_concatenated_data_path)
                best_maps, self.gev, _ = clustering_func(
                    self.concatenated_data,
                    self.n_chan,
                    self.clustering_method,
                    self.number_of_maps,
                    self.initializer,
                    self.min_distance_size,
                    self.number_of_repeats,
                    self.clustering_tolerance,
                    self.clustering_option)
            else:
                # minibatch
                print("minibatch")
                best_maps, self.gev, _ = run_minibatch_modified_kmeans(self.study_name,
                                                                       self.save_folder,
                                                                       self.min_distance_size,
                                                                       self.number_of_maps,
                                                                       self.clustering_tolerance,
                                                                       self.number_of_repeats,
                                                                       self.initializer)



            self.microstate_maps = best_maps

            # Save Maps
            #with open(os.path.join(self.input_folder, 'EEG_INFO.pickle'), 'rb') as p:
            #    eeg_info = pickle.load(p)
            save_name = os.path.join(self.raw_features_path, 'microstate_maps')
            maps_df = pd.DataFrame(best_maps.T, index=self.ch_names)
            maps_df.to_csv(save_name + '.csv')

            # Save settings log
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['clustering settings']['clustering_method'] = self.clustering_method
            config['clustering settings']['clustering_option'] = self.clustering_option
            config['clustering settings']['choose_number_of_maps'] = str(self.choose_number_of_maps)
            config['clustering settings']['number_of_maps'] = str(self.number_of_maps)
            config['clustering settings']['initializer'] = self.initializer
            config['clustering settings']['smoothing_gfp'] = str(self.smoothing_gfp)
            config['clustering settings']['min_distance_size'] = str(self.min_distance_size)
            config['clustering settings']['clustering_tolerance'] = str(self.clustering_tolerance)
            config['clustering settings']['number_of_repeats'] = str(self.number_of_repeats)
            save_config(config_file, config)

            self.final_maps = best_maps
            self.label_maps()
            self.done_clustering = True
            # load and save config
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['progress']['done_clustering'] = str(self.done_clustering)
            save_config(config_file, config)

            self.use_saved_results = True
            self.ui.step0_log_textbrowser.insertPlainText("\n" + "Global Explained Variance: " + str(self.gev) + "\n")
            self.ui.step0_log_textbrowser.insertPlainText("\n" + 20 * "* ")
            self.ui.step0_log_textbrowser.insertPlainText("\n" + "Clustering is done.\n")
            self.ui.step0_log_textbrowser.insertPlainText(20 * "* " + "\n")

            self.mainwindow_controller()

    def do_backfitting(self):
        # Load config
        config_file = os.path.join(self.save_folder, 'log.ini')
        config = load_config(config_file)
        self.done_backfitting = config.getboolean('progress', 'done_backfitting')

        if self.done_backfitting:
            ret = QMessageBox.question(self, 'MessageBox', "Microstates have been backfitted to data once,"
                                                           " do you want to redo backfitting?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_backfitting_from_scratch = False
            if ret == QMessageBox.Yes:
                 self.do_backfitting_from_scratch = True
        else:
            self.do_backfitting_from_scratch = True

        if self.do_backfitting_from_scratch:
            self.done_extracting_features = False
            self.done_extracting_microsegments = False
            self.done_source_localization = False
            # Remove the previous log
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['progress']['done_backfitting'] = str(self.done_backfitting)
            config['progress']['done_extracting_features'] = str(self.done_extracting_features)
            config['progress']['done_extracting_microsegments'] = str(self.done_extracting_microsegments)
            config['progress']['done_source_localization'] = str(self.done_source_localization)
            config['backfitting settings'] = {}
            config['feature extraction settings'] = {}
            config['feature visualization groups'] = {}
            config['source localization settings'] = {}
            save_config(config_file, config)

            # load and save config
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            if config.has_section('backfitting settings'):
                config.remove_section('backfitting settings')
            config['backfitting settings'] = {}

            if self.ui.step3_backfit_all_radio.isChecked():
                self.backfit_to = 'all'
            elif self.ui.step3_backfit_peaks_radio.isChecked():
                self.backfit_to = 'peaks'
            config['backfitting settings']['backfit_to'] = self.backfit_to

            if self.ui.step3_filter_segments_checkbox.isChecked():
                self.filter_segments = True
                self.remove_segments_less_than = self.ui.step3_filter_segments_input.text()
                if self.ui.step3_replace_segments_radio.isChecked():
                    self.filter_segments_option = 'replace'
                elif self.ui.step3_remove_segments_radio.isChecked():
                    self.filter_segments_option = 'remove'
            else:
                self.filter_segments = False
                self.remove_segments_less_than = []
            config['backfitting settings']['filter_segments'] = str(self.filter_segments)
            config['backfitting settings']['filter_segments_option'] = str(self.filter_segments_option)
            config['backfitting settings']['remove_segments_less_than'] = str(self.remove_segments_less_than)
            config['backfitting settings']['output_format'] = self.output_format
            save_config(config_file, config)

            #if not self.concat_data_available:
            #    self.hf_group_data, self.concatenated_data, _, _ = concatenate_files(self.study_name,
            #                                                                         self.save_folder)
            #else:
            print("Loading the concatenated data ...")
            self.concatenated_data = import_hdf_data(self.hdf_concatenated_data_path)
            # Load microstate labels
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            micro_labels_str = config['clustering results']['micro_labels']
            self.micro_labels = micro_labels_str.split(",")

            print('\nBackfitting Maps to Data ...')
            final_segmentation = backfit_func(self.study_name,
                                              self.preprocessed_data_path,
                                              self.final_maps,
                                              self.backfit_to,
                                              self.sample_rate,
                                              self.filter_segments_option,
                                              self.remove_segments_less_than,
                                              self.micro_labels,
                                              self.raw_features_path)

            # Save Segmentation
            '''
            save_segmentation_results(self.list_eegs,
                                      self.length_data,
                                      self.sample_rate,
                                      final_segmentation,
                                      self.output_format,
                                      self.raw_features_path)
            '''
            # Add segment info to h5 data
            #extract_segments(self.listofh5files, self.raw_features_path, self.micro_labels)

            self.done_backfitting = True
            # load and save config
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['progress']['done_backfitting'] = str(self.done_backfitting)
            save_config(config_file, config)
            print('done')

            self.mainwindow_controller()

    def label_maps(self):
        # Load config
        config_file = os.path.join(self.save_folder, 'log.ini')
        config = load_config(config_file)
        self.done_labeling_microstates = config.getboolean('progress', 'done_labeling_microstates')

        if self.done_labeling_microstates:
            ret = QMessageBox.question(self, 'MessageBox', "Microstates have been labeled once,"
                                                           " do you want to relabel microstates?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_labeling_microstates_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_labeling_microstates_from_scratch = True
        else:
            self.do_labeling_microstates_from_scratch = True

        if self.do_labeling_microstates_from_scratch:
            self.done_labeling_microstates = False
            self.done_backfitting = False
            self.done_extracting_features = False
            self.done_extracting_microsegments = False
            self.done_source_localization = False
            # Remove the previous log
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['progress']['done_labeling_microstates'] = str(self.done_labeling_microstates)
            config['progress']['done_backfitting'] = str(self.done_backfitting)
            config['progress']['done_extracting_features'] = str(self.done_extracting_features)
            config['progress']['done_extracting_microsegments'] = str(self.done_extracting_microsegments)
            config['progress']['done_source_localization'] = str(self.done_source_localization)
            config['backfitting settings'] = {}
            config['feature extraction settings'] = {}
            config['feature visualization groups'] = {}
            config['source localization settings'] = {}
            save_config(config_file, config)
            # Load EEG info
            self.eeg_info = load_eeg_info(self.eeg_info_path)
            # Load Microstate Dialog
            self.MicrostateDialog.save_dir = self.save_folder
            self.MicrostateDialog.plot_maps(self.final_maps, self.gev, self.eeg_info)
            self.MicrostateDialog.setWindowModality(QtCore.Qt.ApplicationModal)
            self.MicrostateDialog.showMaximized()
            self.done_labeling_microstates = self.MicrostateDialog.done_labeling
            self.mainwindow_controller()

    def extract_microsegments(self):
        # Load config
        config_file = os.path.join(self.save_folder, 'log.ini')
        config = load_config(config_file)
        self.done_extracting_microsegments = config.getboolean('progress', 'done_extracting_microsegments')

        if self.done_extracting_microsegments:
            ret = QMessageBox.question(self, 'MessageBox', "Microstates have been labeled once,"
                                                           " do you want to relabel microstates?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_extracting_microsegments_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_extracting_microsegments_from_scratch = True
        else:
            self.do_extracting_microsegments_from_scratch = True

        if self.do_extracting_microsegments_from_scratch:
            self.done_extracting_microsegments = False
            # Remove the previous log
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['progress']['done_extracting_microsegments'] = str(self.done_extracting_microsegments)
            save_config(config_file, config)
            self.mainwindow_controller()

            print("\nExtracting Segments ...\n")
            if not os.path.exists(self.micro_segments_path):
                os.makedirs(self.micro_segments_path)
            # Extract data segments
            micro_segments_data(self.hf_data_path,
                                self.hf_segmentation_path,
                                self.n_chan,
                                self.micro_labels,
                                self.micro_segments_path)
            self.done_extracting_microsegments = True
            # Write "feature extraction settings" to config
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['progress']['done_extracting_microsegments'] = str(self.done_extracting_microsegments)
            save_config(config_file, config)
            self.mainwindow_controller()

    def visualize_microsegments(self):
        # UNDER DEVELOPMENT
        print("under development ...")
        '''
        self.MicroSegDialog.number_of_maps = int(self.number_of_maps)
        self.MicroSegDialog.micro_segments_path = self.micro_segments_path
        self.MicroSegDialog.ch_names = self.ch_names
        self.MicroSegDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.MicroSegDialog.showMaximized()
        if self.MicroSegDialog.done_extracting_microsegments:
            self.done_extracting_microsegments = True
        '''

    def extract_features(self):
        # Load config
        config_file = os.path.join(self.save_folder, 'log.ini')
        config = load_config(config_file)
        self.done_extracting_features = config.getboolean('progress', 'done_extracting_features')

        if self.done_extracting_features:
            ret = QMessageBox.question(self, 'MessageBox', "Features have been extracted once,"
                                                           " do you want to extract features again?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_extracting_features_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_extracting_features_from_scratch = True
        else:
            self.do_extracting_features_from_scratch = True

        if self.do_extracting_features_from_scratch:
            self.done_extracting_features = False
            # Remove the previous log
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['progress']['done_extracting_features'] = str(self.done_extracting_features)
            config['feature extraction settings'] = {}
            config['feature visualization groups'] = {}
            save_config(config_file, config)
            self.mainwindow_controller()

            print("\nExtracting Features ...\n")
            if not os.path.exists(self.extracted_features_path):
                os.makedirs(self.extracted_features_path)
            self.Features = []
            if self.ui.step4_coverage_featurestoextract_checkbox.isChecked():
                self.Features.append("COV")
            if self.ui.step4_foc_featurestoextract_checkbox.isChecked():
                self.Features.append("FOC")
            if self.ui.step4_mmd_featurestoextract_checkbox.isChecked():
                self.Features.append("MMD")
            if self.ui.step4_tp_featurestoextract_checkbox.isChecked():
                self.Features.append("TP")
                self.save_transitions = True
            else:
                self.save_transitions = False
            if self.ui.step4_complexity_featurestoextract_checkbox.isChecked():
                self.Features.append("LZC")
            if self.ui.step4_gev_featurestoextract_checkbox.isChecked():
                self.Features.append("GEV")
            extracted_features_df = extract_features(self.preprocessed_data_path,
                                                     self.hf_segmentation_path,
                                                     self.final_maps,
                                                     self.micro_labels,
                                                     self.sample_rate,
                                                     self.Features,
                                                     np.min(self.length_data))
            # Save Features
            save_features(extracted_features_df, 'extracted_features',
                          self.output_format,
                          self.extracted_features_path)
            # Save Transition Matrices
            if self.save_transitions:
                if not os.path.exists(self.raw_transitions_path):
                    os.makedirs(self.raw_transitions_path)
                save_transitions(self.raw_features_path,
                                 self.micro_labels,
                                 self.output_format,
                                 self.raw_transitions_path)
            print('\n*** Finished ***')
            # Write "feature extraction settings" to config
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            Features_save = ','.join(map(str, self.Features))
            config['feature extraction settings']['features2extract'] = Features_save
            config['feature extraction settings']['save_transition_matrices'] = str(self.save_transitions)
            config['feature extraction settings']['output_format'] = str(self.output_format)
            self.done_extracting_features = True
            config['progress']['done_extracting_features'] = str(self.done_extracting_features)
            save_config(config_file, config)
            self.ui.step0_log_textbrowser.insertPlainText("\n" + 20 * "* ")
            self.ui.step0_log_textbrowser.insertPlainText("\n" + "Features are extracted.\n")
            self.mainwindow_controller()


    def source_localize_microstates(self):
        # Load config
        config_file = os.path.join(self.save_folder, 'log.ini')
        config = load_config(config_file)
        self.done_source_localization = config.getboolean('progress', 'done_source_localization')

        if self.done_source_localization:
            ret = QMessageBox.question(self, 'MessageBox', "Microstates have been source localized once,"
                                                           " do you want to source localize microstates again?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_source_localization_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_source_localization_from_scratch = True
        else:
            self.do_source_localization_from_scratch = True

        if self.do_source_localization_from_scratch:
            self.done_source_localization = False
            # Remove the previous log
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['progress']['done_source_localization'] = str(self.done_source_localization)
            config['source localization settings'] = {}
            save_config(config_file, config)
            self.mainwindow_controller()

            print("Source Localizing Microstates ...")

            self.eeg_info = load_eeg_info(self.eeg_info_path)
            microstate_maps_df = pd.read_csv(self.final_maps_path)
            self.microstate_maps = np.asarray(microstate_maps_df.iloc[:, 1:])

            inverse_method = self.ui.step5_inverse_method_combobox.currentText()
            self.inv_method = inverse_method[inverse_method.find("(") + 1:inverse_method.find(")")]
            self.nperm = int(self.ui.step5_permutations_input.text())
            spacing = self.ui.step5_spacing_combobox.currentText()
            self.spacing = spacing[spacing.find("(") + 1:spacing.find(")")].lower()

            self.stc_data = run_source_localization(self.study_name,
                                                    self.eeg_info,
                                                    self.hf_data_path,
                                                    self.microstate_maps,
                                                    self.inv_method,
                                                    self.nperm,
                                                    self.spacing)
            if not os.path.exists(self.localized_sources_path):
                os.makedirs(self.localized_sources_path)
            self.stc_path = os.path.join(self.localized_sources_path, 'stc_data.npy')
            with open(self.stc_path, 'wb') as f:
                np.save(f, self.stc_data)
            self.done_source_localization = True
            # Write "source localization settings" to config
            config_file = os.path.join(self.save_folder, 'log.ini')
            config = load_config(config_file)
            config['source localization settings']['inverse_method'] = str(self.inv_method)
            config['source localization settings']['permutations'] = str(self.nperm)
            config['source localization settings']['spacing'] = str(self.spacing)
            config['progress']['done_source_localization'] = str(self.done_source_localization)
            save_config(config_file, config)

    def visualize_source_localized_microstates(self):
        visualize_sources(self.stc_data, self.spacing)

    def exit_msg(self, event):
        reply = QMessageBox.question(self, "Quit",
                                     "Are you sure you want to quit?",
                                     QMessageBox.Yes |
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
