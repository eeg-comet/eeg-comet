
import os.path
import webbrowser
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from functions.gui_utils.CheckableComboBox import CheckableComboBox

from gui.newstudywindow import NewStudyWindow
from gui.microstate_visualization_dialog import MicrostateVisualizationDialog
from gui.numbermapsdialog import NumberMapsDialog
from gui.visualizationdialog import VisualizationDialog
from gui.sourcevisualizationdialog import SourceVisualizationDialog

import pickle
import re

from functions.data_utils.data_io import load_eeg_info, find_data
from functions.gui_utils.config_io import load_config, save_config
from functions.gui_utils.set_widgets_status import set_widgets_status
from functions.utils.micro_segments_data import micro_segments_data
#from functions.features_utils.extract_features_functions import extract_segments, save_segmentation_results,\
#    save_transitions, save_raw_results, extract_features, transition_matrix, save_features, extract_dynamic_features
#from functions.features_utils.source_localization_functions import run_source_localization

from ToolBox import ToolBox

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

        self.tbx = ToolBox()
        self.context = context
        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("MainMicrostateWindow.ui"), self)

        self.ui.setWindowTitle("Microstate Toolbox")
        self.ui.showMaximized()

        self.ui.NewStudyWindow = NewStudyWindow(context, main_window=self, tbx=self.tbx)
        self.ui.MicrostateVisualizationDialog = MicrostateVisualizationDialog(context, main_window=self, tbx=self.tbx)
        self.ui.NumberMapsDialog = NumberMapsDialog(context)
        self.ui.SourceVisualizationDialog = SourceVisualizationDialog(context)

        # self.ui.VisualizationDialog = VisualizationDialog(context, tbx=self.tbx)

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

        self.ui.step4_features2extract_combobox = CheckableComboBox()
        self.CheckableComboBox_Layout.addWidget(self.ui.step4_features2extract_combobox)
        list_features = [
            "Microstate Coverage (COV)",
            "Microstate Duration (DUR)",
            "Microstate Occurrence (OCC)",
            "Global Explained Variance (GEV)",
            "Transition Probability (TP)",
            "Microstate Complexity (LZC)"
        ]
        self.ui.step4_features2extract_combobox.addItems(list_features)

        self.ui.step0_new_study_action.triggered.connect(self.open_new_study_dialog)
        self.ui.step0_new_study_button.clicked.connect(self.open_new_study_dialog)
        self.ui.step0_load_study_action.triggered.connect(self.load_study)
        self.ui.step0_load_study_button.clicked.connect(self.load_study)
        self.ui.step0_save_log_button.clicked.connect(self.save_log)

        self.ui.step2_clustermethod_combobox.activated.connect(self.mainwindow_controller)
        self.ui.step2_auto_k_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step2_user_k_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step2_advanced_checkbox.clicked.connect(self.mainwindow_controller)
        self.ui.step2_use_percent_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step2_use_peaks_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step3_backfit_all_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step3_backfit_peaks_radio.clicked.connect(self.mainwindow_controller)
        self.ui.step3_filter_segments_checkbox.clicked.connect(self.mainwindow_controller)
        self.ui.step3_filter_segments_method_combobox.activated.connect(self.mainwindow_controller)

        self.ui.step2_numberofmaps_elbow_button.clicked.connect(self.plot_elbow)
        self.ui.step2_clustering_button.clicked.connect(self.do_clustering)
        self.ui.step3_label_maps_button.clicked.connect(self.label_maps)
        self.ui.step3_backfit_button.clicked.connect(self.do_backfitting)
        self.ui.step4_extractfeatures_button.clicked.connect(self.extract_features)
        self.ui.step4_visualizefeatures_button.clicked.connect(self.open_visualize_features_dialog)
        self.ui.step5_estimate_sources_button.clicked.connect(self.source_localize_microstates)
        self.ui.step5_visualize_sources_button.clicked.connect(self.visualize_source_localized_microstates)
        # self.ui.step6_extract_microseg_button.clicked.connect(self.extract_microsegments)
        # self.ui.step4_visualize_sensor_microseg_button.clicked.connect(self.visualize_microsegments)

        self.ui.step0_exit_button.clicked.connect(self.exit_msg)

        # Set actions

        # self.ui.import_settings_action.triggered.connect(self.import_settings)
        # self.ui.export_settings_action.triggered.connect(self.export_settings)

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
        self.tbx.done_preprocessing = False
        self.tbx.done_clustering = False
        self.tbx.done_labeling_microstates = False
        self.tbx.done_backfitting = False
        self.tbx.done_extracting_features = False
        self.tbx.done_source_localization = False
        self.tbx.done_extracting_microsegments = False
        self.ui.NewStudyWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.NewStudyWindow.showMaximized()
        self.mainwindow_controller()

    def open_visualize_features_dialog(self):
        self.ui.VisualizationDialog = VisualizationDialog(self.context, tbx=self.tbx)
        self.ui.VisualizationDialog.extracted_features_path = self.tbx.extracted_features_path
        self.ui.VisualizationDialog.save_folder = self.tbx.save_dir
        self.ui.VisualizationDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.VisualizationDialog.showMaximized()
        self.ui.VisualizationDialog.load_filenames()
        self.mainwindow_controller()

    def load_study(self, save_folder=None):

        if self.tbx.done_preprocessing == False:  
            self.save_folder = QFileDialog.getExistingDirectory(self, "Select the folder containing preprocessed data")
            tbx_object = os.path.join(self.save_folder, 'tbx_object.pkl')
            if not os.path.exists(tbx_object):
                QMessageBox.information(self, "Load error",
                                        "The selected folder does not contain a valid study!",
                                        QMessageBox.Ok)
                return
            with open(tbx_object, 'rb') as input_tbx:
                self.tbx = pickle.load(input_tbx)
        # Define global directories
        # self.preprocessed_data_path = os.path.join(self.save_folder, 'preprocessed_data')
        self.tbx.raw_features_path = os.path.join(self.tbx.save_dir, 'raw_features')
        self.tbx.raw_transitions_path = os.path.join(self.tbx.raw_features_path, 'raw_transitions')
        self.tbx.extracted_features_path = os.path.join(self.tbx.save_dir, 'extracted_features')
        self.tbx.microstate_maps_path = os.path.join(self.tbx.raw_features_path, 'microstate_maps.csv')
        self.tbx.localized_sources_path = os.path.join(self.tbx.raw_features_path, 'localized_sources')
        self.tbx.stc_path = os.path.join(self.tbx.localized_sources_path, 'stc_data.npy')

        # Load preprocessing information
        if self.tbx.done_preprocessing:
            self.tbx.segmentation_path = os.path.join(self.tbx.raw_features_path, self.tbx.study_name+'_segmentation.hdf')

            # Show study name on the main window
            self.ui.step0_study_name_mainwin_lineedit.setText(self.tbx.study_name)
            self.ui.step0_log_textbrowser.insertPlainText("Study Name: " + self.tbx.study_name + "\n")
            self.ui.step0_log_textbrowser.insertPlainText(20 * "* " + "\n")

            self.ui.step0_log_textbrowser.insertPlainText("\n"
                                                          + str(len(self.tbx.list_eegs))
                                                          + " EEG files are found:\n\n"
                                                          + "\n".join(self.tbx.list_eegs)+"\n")

            if self.tbx.filter_data:
                self.ui.step0_log_textbrowser.insertPlainText(
                    f"\nFiltered data using {self.tbx.filter_method.upper()} between {self.tbx.lowcut_freq} Hz and {self.tbx.highcut_freq} Hz\n")
            if self.tbx.downsample_data:
                self.ui.step0_log_textbrowser.insertPlainText(
                    f"\nDownsampled data to {self.tbx.sample_rate} Hz\n")
            self.ui.step0_log_textbrowser.insertPlainText("\n" + 20*"* ")
            self.ui.step0_log_textbrowser.insertPlainText("\nPreprocessing is done.\n")
            self.ui.step0_log_textbrowser.insertPlainText(20*"* " + "\n")

        # Load clustering_utils information
        if self.tbx.done_clustering:
            if self.tbx.done_labeling_microstates:
                # micro_labels_str = config['clustering_utils results']['micro_labels']
                # self.micro_labels = micro_labels_str.split(",")
                micro_labels_str = ','.join(self.tbx.micro_labels)
                self.ui.step0_log_textbrowser.insertPlainText(
                    f"\nMicrostate labels: {micro_labels_str}\n")

            self.ui.step0_log_textbrowser.insertPlainText(f"\nGlobal Explained Variance: {self.tbx.gev}\n")
            self.ui.step0_log_textbrowser.insertPlainText("\n" + 20*"* ")
            self.ui.step0_log_textbrowser.insertPlainText("\nClustering is done.\n")
            self.ui.step0_log_textbrowser.insertPlainText(20*"* " + "\n")

        # Update GUI
        self.update_mainwindow_gui()

        self.mainwindow_controller()

    def save_log(self):
        text = self.step0_log_textbrowser.toPlainText()
        save_log_path = os.path.join(self.tbx.save_dir, f'{self.tbx.study_name}_log.txt')
        with open(save_log_path, 'w') as f:
            f.write(text)

    def reset_option_box(self, box, options=[], current=None):
        '''
        box: the option box that we want to reset
        options: list of strings, the name of options
        current: str, default option
        
        '''
        box.clear()
        for item in options:
            box.addItem(item)
        if current:
            box.setCurrentText(current)

    def mainwindow_controller(self):

        after_preprocessing_widgets = [
            self.ui.step2_clustering_title_label,
            self.ui.step2_clustermethod_combo_label,
            self.ui.step2_clustermethod_combobox,
            self.ui.step2_numberofmaps_label,
            self.ui.step2_advanced_checkbox,
            self.ui.step2_auto_k_radio,
            self.ui.step2_user_k_radio,
            self.ui.step2_numberofmaps_elbow_button,
            self.ui.step2_advanced_checkbox,
            self.ui.step2_clustering_button
        ]

        elbow_widgets = [
            self.ui.step2_auto_target_label,
            self.ui.step2_stopping_threshold_label,
            self.ui.step2_stopping_threshold_percentage_label,
            self.ui.step2_stopping_threshold_input,
            self.ui.step2_auto_range_kmin_combobox,
            self.ui.step2_auto_range_kmax_combobox,
            self.ui.step2_auto_k_method_combobox,
            self.ui.step2_auto_range_label
        ]

        advanced_widgets = [
            self.ui.step2_other_label,
            self.ui.step2_other_options_combobox,
            self.ui.step2_initializer_label,
            self.ui.step2_random_initializer_radio,
            self.ui.step2_kmeans_initializer_radio,
            self.ui.step2_select_times_label,
            self.ui.step2_use_peaks_radio,
            self.ui.step2_kernel_size_label,
            self.ui.step2_kernel_size_input,
            self.ui.step2_kernel_size_label_2,
            self.ui.step2_use_percent_radio,
            self.ui.step2_percent_label,
            self.ui.step2_percent_combobox,
            self.ui.step2_percent_label_2,
            self.ui.step2_maxiter_label,
            self.ui.step2_maxiter_input,
            self.ui.step2_stopcondition_label,
            self.ui.step2_stopcondition_input,
            self.ui.step2_numberofrepeats_label,
            self.ui.step2_user_numberofrepeats_input
        ]

        peaks2use_widgets = [
            self.ui.step2_kernel_size_label,
            self.step2_kernel_size_input,
            self.step2_kernel_size_label_2
        ]

        rand2use_widgets = [
            self.ui.step2_percent_label,
            self.step2_percent_combobox,
            self.step2_percent_label_2
        ]

        after_clustering_widgets = [
            self.ui.step3_label_maps_button,
            self.ui.step3_backfit_title_label,
            self.ui.step3_backfit_all_radio,
            self.ui.step3_backfit_peaks_radio,
            self.ui.step3_filter_segments_checkbox,
            self.ui.step3_backfit_button
        ]

        filter_segments_widgets = [
            self.ui.step3_filter_segments_input,
            self.ui.step3_filter_segments_label,
            self.ui.step3_filter_segments_method_combobox,
            self.ui.step3_filter_segments_label,
            self.ui.step3_filter_segments_label_2
        ]

        smooth_segments_widgets = [
            self.ui.step3_smooth_segments_epsilon_label,
            self.ui.step3_smooth_segments_epsilon_input,
            self.ui.step3_smooth_segments_lambda_label,
            self.ui.step3_smooth_segments_lambda_input
        ]

        feature_extraction_widgets = [
            self.ui.step4_features_title_label,
            self.ui.step4_featurestoextract_label,
            self.ui.step4_features2extract_combobox,
            self.ui.step4_duration_of_window_input,
            self.ui.step4_duration_of_window_label,
            self.ui.step4_duration_of_window_label_2,
            self.ui.step4_static_features_checkbox,
            self.ui.step4_dynamic_features_checkbox,
            self.ui.step4_extractfeatures_button,
            self.ui.step4_outputformats_label,
            self.ui.step4_outputformats_combobox
            ]

        source_localization_widgets = [
            self.ui.step5_source_localization_title_label,
            self.ui.step5_inverse_method_label,
            self.ui.step5_inverse_method_combobox,
            self.ui.step5_permutations_label,
            self.ui.step5_permutations_input,
            self.ui.step5_spacing_label,
            self.ui.step5_spacing_combobox,
            self.ui.step5_estimate_sources_button
        ]

        if self.tbx.done_preprocessing:
            self.ui.step0_study_name_mainwin_lineedit.setStyleSheet("background-color: lightgreen")
            set_widgets_status(after_preprocessing_widgets, mode='enable')
            if self.ui.step2_auto_k_radio.isChecked():
                k_log = 'will be automatically determined.'
                self.ui.step2_user_k_input.setDisabled(True)
                self.ui.step2_numberofmaps_elbow_button.setDisabled(True)
                set_widgets_status(elbow_widgets, mode='enable')
            if self.ui.step2_user_k_radio.isChecked():
                k_log = 'is user-predefined.'
                self.ui.step2_user_k_input.setEnabled(True)
                self.ui.step2_numberofmaps_elbow_button.setEnabled(True)
                set_widgets_status(elbow_widgets, mode='disable')

            if self.ui.step2_advanced_checkbox.isChecked():
                set_widgets_status(advanced_widgets, mode='enable')
                set_widgets_status(advanced_widgets, mode='show')
            else:
                set_widgets_status(advanced_widgets, mode='disable')
                set_widgets_status(advanced_widgets, mode='hide')

            self.tbx.clustering_method = self.step2_clustermethod_combobox.currentText()
            if self.tbx.clustering_method == "K-means":
                self.ui.step2_other_label.setText("Similarity metric:")
                options = ['Cosine Similarity', 'Spatial Correlation']
                self.reset_option_box(self.ui.step2_other_options_combobox, options, 'Spatial Correlation')

            elif self.tbx.clustering_method == "Agglomerative hierarchical clustering_utils":
                self.ui.step2_other_label.setText("Type of link between clusters:")
                options = ['Single Link', 'Complete Link', 'Average Link', 'Centroid Link']
                self.reset_option_box(self.ui.step2_other_options_combobox, options, 'Single Link')

            elif self.tbx.clustering_method == "X-means":
                self.ui.step2_other_label.setText("X-means splitting criterion:")
                options = ['Bayesian Information Criterion', 'Minimum Noiseless Description Length']
                self.reset_option_box(self.ui.step2_other_options_combobox, options, 'Bayesian Information Criterion')

            else:
                self.ui.step2_other_label.setText("Other options:")
                self.reset_option_box(self.ui.step2_other_options_combobox)

            if self.ui.step2_use_peaks_radio.isChecked():
                cluster_data_log = 'the local peaks of the global field power.'
                set_widgets_status(peaks2use_widgets, mode='enable')
                set_widgets_status(rand2use_widgets, mode='disable')
            elif self.ui.step2_use_percent_radio.isChecked():
                cluster_data_log = 'a randomly selected subset of the data.'
                set_widgets_status(rand2use_widgets, mode='enable')
                set_widgets_status(peaks2use_widgets, mode='disable')

            # Update the clustering_utils log
            self.step2_clustering_log_textedit.clear()
            self.step2_clustering_log_textedit.appendPlainText(
                f"EEG microstates will be identified using {self.tbx.clustering_method} clustering_utils algorithm")
            self.step2_clustering_log_textedit.appendPlainText(f"The number of maps to extract {k_log}")
            self.step2_clustering_log_textedit.appendPlainText(f"Clustering will be performed on {cluster_data_log}")

        else:
            self.ui.step0_study_name_mainwin_lineedit.setStyleSheet("background-color: none")
            set_widgets_status(after_preprocessing_widgets, mode='disable')
            self.ui.step2_user_k_input.setDisabled(True)
        
        if self.tbx.done_clustering:
            self.ui.step2_clustering_button.setStyleSheet("background-color: lightgreen")
            set_widgets_status(after_clustering_widgets, mode='enable')

            outputformat = self.ui.step4_outputformats_combobox.currentText()
            self.tbx.export_format = outputformat.split(".")[-1].split(")")[0]

            if self.ui.step3_filter_segments_checkbox.isChecked():
                set_widgets_status(filter_segments_widgets, mode='enable')
                filter_segments_method = self.ui.step3_filter_segments_method_combobox.currentText()
                if filter_segments_method == "Smooth segments":
                    set_widgets_status(smooth_segments_widgets, mode='enable')
                    set_widgets_status(smooth_segments_widgets, mode='show')
                else:
                    set_widgets_status(smooth_segments_widgets, mode='disable')
                    set_widgets_status(smooth_segments_widgets, mode='hide')
            else:
                set_widgets_status(filter_segments_widgets, mode='disable')
                set_widgets_status(smooth_segments_widgets, mode='disable')

        else:
            self.tbx.done_labeling_microstates = False
            self.tbx.done_backfitting = False
            self.tbx.done_extracting_features = False
            self.tbx.done_source_localization = False
            self.tbx.done_extracting_microsegments = False

            self.ui.step2_clustering_button.setStyleSheet("background-color: none")
            # set ALL disabled
            set_widgets_status(self.ui.step3_filter_segments_checkbox, mode='disable')
            set_widgets_status(after_clustering_widgets, mode='disable')
            set_widgets_status(filter_segments_widgets, mode='disable')

        if self.tbx.done_labeling_microstates:
            self.ui.step3_label_maps_button.setStyleSheet("background-color: lightgreen")
        else:
            self.ui.step3_label_maps_button.setStyleSheet("background-color: none")
            self.tbx.done_backfitting = False
            self.tbx.done_extracting_features = False
            self.tbx.done_source_localization = False
            self.tbx.done_extracting_microsegments = False

        if self.tbx.done_backfitting:
            self.ui.step3_backfit_button.setStyleSheet("background-color: lightgreen")
            set_widgets_status(feature_extraction_widgets, mode='enable')
            set_widgets_status(source_localization_widgets, mode='enable')

        else:
            self.tbx.done_extracting_features = False
            self.tbx.done_extracting_microsegments = False
            self.ui.step3_backfit_button.setStyleSheet("background-color: none")
            set_widgets_status(feature_extraction_widgets, mode='disable')
            set_widgets_status(source_localization_widgets, mode='disable')

        if self.tbx.done_extracting_features:
            self.ui.step4_extractfeatures_button.setStyleSheet("background-color: lightgreen")
            self.ui.step4_visualizefeatures_button.setEnabled(True)
        else:
            self.ui.step4_extractfeatures_button.setStyleSheet("background-color: none")
            self.ui.step4_visualizefeatures_button.setDisabled(True)

        # if self.done_extracting_microsegments:
        #     self.ui.step6_extract_microseg_button.setStyleSheet("background-color: lightgreen")
        #     self.ui.step4_visualize_sensor_microseg_button.setEnabled(True)
        # else:
        #     self.ui.step6_extract_microseg_button.setStyleSheet("background-color: none")
        #     self.ui.step4_visualize_sensor_microseg_button.setDisabled(True)

        if self.tbx.done_source_localization:
            self.ui.step5_estimate_sources_button.setStyleSheet("background-color: lightgreen")
            self.ui.step5_visualize_sources_button.setEnabled(True)
            # self.ui.step6_visualize_source_microseg_button.setEnabled(True)
        else:
            self.ui.step5_estimate_sources_button.setStyleSheet("background-color: none")
            self.ui.step5_visualize_sources_button.setDisabled(True)
            # self.ui.step6_visualize_source_microseg_button.setDisabled(True)

    def update_mainwindow_gui(self):
        if self.tbx.done_clustering:
            self.ui.step2_clustermethod_combobox.setCurrentText(self.tbx.clustering_method)
            #self.clustering_option
            if self.tbx.choose_number_of_maps == 'Auto':
                self.ui.step2_auto_k_radio.setChecked(True)
            elif self.tbx.choose_number_of_maps == 'User':
                self.ui.step2_user_k_radio.setChecked(True)
                self.ui.step2_user_k_input.setText(str(self.tbx.number_of_maps))
            if self.tbx.initializer == 'Random':
                self.ui.step2_random_initializer_radio.setChecked(True)
            elif self.tbx.initializer == 'K-Means++':
                self.ui.step2_kmeans_initializer_radio.setChecked(True)
            #self.smoothing_gfp
            self.ui.step2_kernel_size_input.setText(str(self.tbx.min_distance_size))
            self.ui.step2_stopcondition_input.setText(str(self.tbx.clustering_tolerance))
            self.ui.step2_user_numberofrepeats_input.setText(str(self.tbx.number_of_repeats))

        if self.tbx.done_backfitting:
            if self.tbx.filter_segments:
                self.ui.step3_filter_segments_checkbox.setChecked(True)
            else:
                self.ui.step3_filter_segments_checkbox.setChecked(False)
            if self.tbx.backfit_to == 'all':
                self.ui.step3_backfit_all_radio.setChecked(True)
            elif self.tbx.backfit_to == 'peaks':
                self.ui.step3_backfit_peaks_radio.setChecked(True)
            #self.ui.step4_outputformats_combobox.setCurrentText(self.output_format)
        """
        if self.tbx.done_extracting_features:
            if 'COV' in self.tbx.Features:
                self.ui.step4_coverage_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_coverage_featurestoextract_checkbox.setChecked(False)
            if 'MMD' in self.tbx.Features:
                self.ui.step4_mmd_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_mmd_featurestoextract_checkbox.setChecked(False)
            if 'OCC' in self.tbx.Features:
                self.ui.step4_foc_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_foc_featurestoextract_checkbox.setChecked(False)
            if 'GEV' in self.tbx.Features:
                self.ui.step4_gev_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_gev_featurestoextract_checkbox.setChecked(False)
            if 'TP' in self.tbx.Features:
                self.ui.step4_tp_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_tp_featurestoextract_checkbox.setChecked(False)
            if 'LZC' in self.tbx.Features:
                self.ui.step4_complexity_featurestoextract_checkbox.setChecked(True)
            else:
                self.ui.step4_complexity_featurestoextract_checkbox.setChecked(False)
            """
    def plot_elbow(self):
        self.NumberMapsDialog.preprocessed_data_path = self.tbx.preprocessed_data_path
        self.NumberMapsDialog.extension = self.tbx.extension
        self.NumberMapsDialog.datatype = self.tbx.datatype
        if self.ui.step2_use_percent_radio.isChecked():
            self.tbx.use_percentages = self.ui.step2_percent_combobox.currentText()
        else:
            self.tbx.use_percentages = None
        self.NumberMapsDialog.use_percentages = self.tbx.use_percentages
        self.NumberMapsDialog.min_distance_size = int(int(self.ui.step2_kernel_size_input.text())/(1000/self.tbx.sample_rate))
        self.NumberMapsDialog.clustering_tolerance = float(self.ui.step2_stopcondition_input.text())
        self.NumberMapsDialog.number_of_repeats = int(self.ui.step2_user_numberofrepeats_input.text())
        self.NumberMapsDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.NumberMapsDialog.showMaximized()

    def do_clustering(self):

        # Check if the analysis is already done.
        if self.tbx.done_clustering:
            ret = QMessageBox.question(self, 'MessageBox', "Data has been clustered once,"
                                                           " do you want to redo the analysis?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_clustering_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_clustering_from_scratch = True
                # Remove the previous log
                self.tbx.done_clustering = False
                self.tbx.done_labeling_microstates = False
                self.tbx.done_backfitting = False
                self.tbx.done_extracting_features = False
                # self.done_extracting_microsegments = False
                self.tbx.done_source_localization = False
        else:
            self.do_clustering_from_scratch = True

        if self.do_clustering_from_scratch:
            if not os.path.exists(self.tbx.raw_features_path):
                os.makedirs(self.tbx.raw_features_path)
            if self.ui.step2_kernel_size_input.text():
                self.tbx.smoothing_gfp = True
                self.tbx.smoothing_distance = int(self.ui.step2_kernel_size_input.text())
                # self.tbx.min_distance_size = int(int(self.ui.step2_kernel_size_input.text())/(1000/self.sample_rate))
            else:
                self.smoothing_gfp = False
                self.tbx.smoothing_distance = ''
                # self.min_distance_size = []
            if self.ui.step2_auto_k_radio.isChecked():
                self.tbx.choose_number_of_maps = "auto"
                # TODO: should automatically set a self.number_of_maps
                # if self.ui.step2_stopping_traditional_radio.isChecked():
                #     self.tbx.elbow_version = 'traditional'
                # else:
                #     self.tbx.elbow_version = 'modified'
                self.tbx.kmin = int(self.ui.step2_auto_range_kmin_combobox.currentText())
                self.tbx.kmax = int(self.ui.step2_auto_range_kmax_combobox.currentText())

                auto_k_method = self.ui.step2_auto_k_method_combobox.currentText()
                if auto_k_method == 'Global Explained Variance':
                    self.tbx.stopping_mode = 'gev'
                elif auto_k_method == 'Residual Variance':
                    self.tbx.stopping_mode = 'res'
                elif auto_k_method == 'Silhouette Score':
                    self.tbx.stopping_mode = 'sil'

                self.tbx.stopping_parameter = float(self.ui.step2_stopping_threshold_input.text())
                self.tbx.number_of_maps = 'auto'
            elif self.ui.step2_user_k_radio.isChecked():
                self.tbx.choose_number_of_maps = "user"
                self.tbx.stopping_mode = ''
                self.tbx.stopping_parameter = ''
                self.tbx.kmin = ''
                self.tbx.kmax = ''
                self.tbx.number_of_maps = int(self.ui.step2_user_k_input.text())
            #print(self.number_of_maps)
            if self.ui.step2_random_initializer_radio.isChecked():
                self.tbx.initializer = "Random"
            elif self.ui.step2_kmeans_initializer_radio.isChecked():
                self.tbx.initializer = "K-Means++"
            self.tbx.clustering_method = self.ui.step2_clustermethod_combobox.currentText()
            print(self.tbx.clustering_method)

            if self.ui.step2_use_percent_radio.isChecked():
                self.tbx.use_percentages = self.ui.step2_percent_combobox.currentText()
            else:
                self.tbx.use_percentages = None
            self.tbx.max_iterations = int(self.ui.step2_maxiter_input.text())
            self.tbx.clustering_tolerance = float(self.ui.step2_stopcondition_input.text())
            #print(self.clustering_tolerance)
            self.tbx.clustering_option = self.ui.step2_other_options_combobox.currentText()
            #print(self.clustering_option)
            self.tbx.number_of_repeats = int(self.ui.step2_user_numberofrepeats_input.text())

            self.tbx.do_clustering()

            self.label_maps()
            self.tbx.done_clustering = True
            self.tbx.save_tbx()
            
            # self.use_saved_results = True
            self.ui.step0_log_textbrowser.insertPlainText("\n" + "Global Explained Variance: " + str(self.tbx.gev) + "\n")
            self.ui.step0_log_textbrowser.insertPlainText("\n" + 20 * "* ")
            self.ui.step0_log_textbrowser.insertPlainText("\n" + "Clustering is done.\n")
            self.ui.step0_log_textbrowser.insertPlainText(20 * "* " + "\n")

            self.mainwindow_controller()

    def do_backfitting(self):
        # Load config
        # config_file = os.path.join(self.save_folder, 'log.ini')
        # config = load_config(config_file)
        # self.done_backfitting = config.getboolean('progress', 'done_backfitting')

        if self.tbx.done_backfitting:
            ret = QMessageBox.question(self, 'MessageBox', "Microstates have been backfitted to data once,"
                                                           " do you want to redo backfitting_utils?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_backfitting_from_scratch = False
            if ret == QMessageBox.Yes:
                 self.do_backfitting_from_scratch = True
        else:
            self.do_backfitting_from_scratch = True

        if self.do_backfitting_from_scratch:
            self.tbx.done_extracting_features = False
            self.tbx.done_extracting_microsegments = False
            self.tbx.done_source_localization = False
            
            if self.ui.step3_backfit_all_radio.isChecked():
                self.tbx.backfit_to = 'all'
            elif self.ui.step3_backfit_peaks_radio.isChecked():
                self.tbx.backfit_to = 'peaks'

            self.tbx.epsilon = ''
            self.tbx.b = ''
            self.tbx.lamb = ''
            if self.ui.step3_filter_segments_checkbox.isChecked():
                self.tbx.filter_segments = True
                self.tbx.remove_segments_less_than = int(int(self.ui.step3_filter_segments_input.text())/(1000/self.tbx.sample_rate))
                filter_segments_method = self.ui.step3_filter_segments_method_combobox.currentText()
                if filter_segments_method == 'Replace short segments: nearby dominant microstate':
                    self.tbx.filter_segments_option = 'replace_high'
                elif filter_segments_method == 'Replace short segments: half and half':
                    self.tbx.filter_segments_option = 'replace_half'
                elif filter_segments_method == 'Remove short segments':
                    self.tbx.filter_segments_option = 'remove'
                elif filter_segments_method == 'Smooth segments':
                    self.tbx.filter_segments_option = 'smooth'
                    self.tbx.epsilon = float(self.ui.step3_smooth_segments_epsilon_input.text())
                    self.tbx.b = self.tbx.remove_segments_less_than
                    self.tbx.lamb = int(self.ui.step3_smooth_segments_lambda_input.text())

            else:
                self.tbx.filter_segments = False
                self.tbx.filter_segments_option = ''
                self.tbx.remove_segments_less_than = []

            self.tbx.do_backfitting()
            self.tbx.save_tbx()
            self.mainwindow_controller()

    def label_maps(self):
        # Load config
        # config_file = os.path.join(self.save_folder, 'log.ini')
        # config = load_config(config_file)
        # self.done_labeling_microstates = config.getboolean('progress', 'done_labeling_microstates')
        just_show_labels = False

        if self.tbx.done_labeling_microstates:
            ret = QMessageBox.question(self, 'MessageBox', "Microstates have been labeled once,"
                                                           " do you want to relabel microstates?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.show_labelling_window = False
            if ret == QMessageBox.Yes:
                self.show_labelling_window = True
                just_show_labels = False
            elif ret == QMessageBox.No:
                self.show_labelling_window = True
                just_show_labels = True
        else:
            self.show_labelling_window = True

        if self.show_labelling_window:
            if just_show_labels == True:

                #self.MicrostateVisualizationDialog.save_dir = self.tbx.save_dir
                self.MicrostateVisualizationDialog.n_states = self.tbx.best_maps.shape[0]
                self.MicrostateVisualizationDialog.microstate_maps = self.tbx.best_maps
                self.MicrostateVisualizationDialog.eeg_info = self.tbx.eeg_info
                self.MicrostateVisualizationDialog.microstates_combobox.addItems([str(i) for i in range(self.tbx.best_maps.shape[0])])
                self.MicrostateVisualizationDialog.microstates_image_path = os.path.join(self.tbx.save_dir, f"{self.tbx.study_name}_microstates.png")
                self.MicrostateVisualizationDialog.set_layout(self.tbx.micro_labels)
                self.MicrostateVisualizationDialog.plot_maps()
                self.MicrostateVisualizationDialog.setWindowModality(QtCore.Qt.ApplicationModal)
                self.MicrostateVisualizationDialog.showMaximized()

                self.mainwindow_controller()
            else:
                self.tbx.done_labeling_microstates = False
                self.tbx.done_backfitting = False
                self.tbx.done_extracting_features = False
                # self.tbx.done_extracting_microsegments = False
                self.tbx.done_source_localization = False
                """
                self.MicrostateVisualizationDialog.save_dir = self.tbx.save_dir
                self.MicrostateVisualizationDialog.plot_maps(self.tbx.best_maps, self.tbx.eeg_info)
                self.MicrostateVisualizationDialog.setWindowModality(QtCore.Qt.ApplicationModal)
                self.MicrostateVisualizationDialog.showMaximized()
                self.mainwindow_controller()
                """
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
            micro_segments_data(self.hdf_concatenated_data_path,
                                self.segmentation_path,
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


    def extract_features(self):
        # Load config
        # config_file = os.path.join(self.save_folder, 'log.ini')
        # config = load_config(config_file)
        # self.done_extracting_features = config.getboolean('progress', 'done_extracting_features')

        if self.done_extracting_features:
            ret = QMessageBox.question(self, 'MessageBox', "Features have been extracted once,"
                                                           " do you want to extract features_utils again?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_extracting_features_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_extracting_features_from_scratch = True
        else:
            self.do_extracting_features_from_scratch = True

        if self.do_extracting_features_from_scratch:
            self.tbx.done_extracting_features = False
            self.mainwindow_controller()

            self.tbx.feature_list = []
            features2extract = self.ui.step4_features2extract_combobox.currentData()
            for feature in features2extract:
                match = re.search(r'\((\w+)\)', feature)
                if match:
                    self.tbx.feature_list.append(match.group(1))

            self.tbx.feature_mode = []
            if self.ui.step4_static_features_checkbox.isChecked():
                self.tbx.feature_mode.append('static')
            if self.ui.step4_dynamic_features_checkbox.isChecked():
                self.tbx.feature_mode.append('dynamic')

            self.tbx.window_size = int(self.ui.step4_duration_of_window_input.text())

            print(self.tbx.export_format)

            self.tbx.extract_features()
            self.tbx.save_tbx()
            
            self.ui.step0_log_textbrowser.insertPlainText("\n" + 20 * "* ")
            self.ui.step0_log_textbrowser.insertPlainText("\n" + "Features are extracted.\n")
            self.mainwindow_controller()


    def source_localize_microstates(self):
        
        if self.tbx.done_source_localization:
            ret = QMessageBox.question(self, 'MessageBox', "Microstates have been source localized once,"
                                                           " do you want to source localize microstates again?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_source_localization_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_source_localization_from_scratch = True
        else:
            self.do_source_localization_from_scratch = True

        if self.do_source_localization_from_scratch:
            self.tbx.done_source_localization = False
            self.mainwindow_controller()

            inverse_method = self.ui.step5_inverse_method_combobox.currentText()
            self.tbx.inverse_method = inverse_method[inverse_method.find("(") + 1:inverse_method.find(")")]
            self.tbx.nperm = int(self.ui.step5_permutations_input.text())
            spacing = self.ui.step5_spacing_combobox.currentText()
            self.tbx.spacing = spacing[spacing.find("(") + 1:spacing.find(")")].lower()

            self.tbx.source_localize_microstates()
            self.tbx.save_tbx()
            self.mainwindow_controller()
            
    def visualize_source_localized_microstates(self):
        # TODO
        self.SourceVisualizationDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.SourceVisualizationDialog.showMaximized()
        #visualize_sources(self.tbx.localized_sources_path, self.tbx.spacing)

    def exit_msg(self, event):
        reply = QMessageBox.question(self, "Quit",
                                     "Are you sure you want to quit?",
                                     QMessageBox.Yes |
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
