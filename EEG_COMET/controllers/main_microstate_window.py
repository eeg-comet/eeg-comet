
import os.path
import webbrowser
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QComboBox, QSpinBox, QMessageBox
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

from controllers.new_study_window import NewStudyWindow
from controllers.compare_studies_window import CompareStudiesWindow
from controllers.microstate_visualization_window import MicrostateVisualizationWindow
from controllers.optimizer_visualization_window import OptimizerVisualizationWindow
from controllers.backfitting_visualization_window import BackfittingVisualizationWindow
from controllers.feature_visualization_window import FeatureVisualizationWindow
from controllers.coregistration_window import CoregistrationWindow
from controllers.source_visualization_window import SourceVisualizationWindow
from gui_utils.set_widgets_status import set_widgets_status
from comet import COMET


class MainMicrostateWindow(QMainWindow):
    def __init__(self, context, parent=None):
        super(MainMicrostateWindow, self).__init__(parent)

        # Initialize key components
        self.comet_tbx = COMET()
        self.comet_tbx.initialize_log_window()
        self.comet_tbx.LogWindow.show()
        self.context = context
        
        # Load the UI from the .ui file
        self.ui = uic.loadUi(context.get_resource("MainMicrostateWindow.ui"), self)
        self.ui.setWindowTitle("EEG-COMET")
        self.ui.showMaximized()

        # Initialize dialogs, flags, UI components, and connections
        self.init_dialogs()
        self.init_flags()
        self.init_ui_components()
        self.setup_connections()

    @staticmethod
    def open_github():
        """
        Open the GitHub page in the default web browser
        """
        webbrowser.open('https://github.com/eBrainLab/EEG-COMET')

    @staticmethod
    def report_issues():
        """
        Open the GitHub issues page in the default web browser
        """
        # webbrowser.open('https://github.com/eBrainLab/EEG-COMET/issues/new')
        webbrowser.open('https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction/issues/new')

    def update_toolbox(self):
        """
        Ask the user if they want to download the toolbox
        """
        ret = QMessageBox.question(self, 'MessageBox', "Do you want to download the toolbox?",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
        if ret == QMessageBox.Yes:
            webbrowser.open(
                'https://github.com/eBrainLab/EEG-COMET/archive/refs/heads/main.zip')

    def init_dialogs(self):
        # Create and initialize NewStudyWindow
        self.ui.NewStudyWindow = NewStudyWindow(
            self.context,
            main_window=self,
            comet_tbx=self.comet_tbx
        )

        self.ui.CompareStudiesWindow = CompareStudiesWindow(self.context)

        # Create and initialize MicrostateVisualizationWindow
        # self.ui.MicrostateVisualizationWindow = MicrostateVisualizationWindow(
        #     self.context,
        #     main_window=self,
        #     tbx=self.comet_tbx
        # )
        # Create and initialize OptimizerVisualizationWindow
        self.ui.OptimizerVisualizationWindow = OptimizerVisualizationWindow(
            self.context
        )
        # Create and initialize BackfittingVisualizationWindow
        self.ui.BackfittingVisualizationWindow = BackfittingVisualizationWindow(
            self.context
        )
        # Create and initialize FeatureVisualizationWindow
        self.ui.FeatureVisualizationWindow = FeatureVisualizationWindow(
            self.context,
            tbx=self.comet_tbx
        )
        # Create and initialize CoregistrationWindow
        self.ui.CoregistrationWindow = CoregistrationWindow(
            self.context,
            tbx=self.comet_tbx
        )

    def reset_processing_flags(self, processing_flags, value=False):
        """
        Set processing flags.
        """
        for flag in processing_flags:
            setattr(self.comet_tbx, flag, value)

    def init_flags(self):
        """
        Initialize flags for tracking processing steps
        """
        # Reset all processing flags to False
        processing_flags = [
            'done_preprocessing', 'done_clustering', 'done_labeling_microstates',
            'done_backfitting', 'done_extracting_features', 'done_source_localization',
            'done_source_microstate_correlation'
        ]
        self.reset_processing_flags(processing_flags)
        self.ui.foldername_raw_data = ""
        self.ui.foldername_preprocessed_data = ""

    def init_ui_components(self):

        # Add EEG-COMET Logo
        icon_path = self.context.get_resource("eeg_comet_logo.png")
        self.pixmap = QPixmap(icon_path)
        pixmap = self.pixmap.scaled(512, 512, Qt.KeepAspectRatio)
        self.ui.comet_logo.setPixmap(pixmap)

        set_widgets_status(self.ui.main_tab, mode='hide')

    def setup_connections(self):
        # Controlling the visibility and state of various UI components based on user interactions
        control_items = [
            self.ui.step2_auto_k_radio,
            self.ui.step2_user_k_radio,
            self.ui.step2_advanced_checkbox,
            self.ui.step2_use_percent_radio,
            self.ui.step2_use_peaks_radio,
            self.ui.step2_clustermethod_combobox,
            self.ui.step2_auto_k_method_combobox,
            self.ui.step2_auto_range_kmin_spinbox,
            self.ui.step2_auto_range_kmax_spinbox,
            self.ui.step3_backfit_all_radio,
            self.ui.step3_backfit_peaks_radio,
            self.ui.step3_filter_segments_checkbox,
            self.ui.step3_identify_short_checkbox,
            self.ui.step3_filter_segments_method_combobox,
            self.ui.step4_feature_occ_checkbox,
            self.ui.step4_feature_dur_checkbox,
            self.ui.step4_feature_cov_checkbox,
            self.ui.step4_feature_gev_checkbox,
            self.ui.step4_feature_tp_checkbox,
            self.ui.step4_feature_se_checkbox,
            self.ui.step4_feature_lzc_checkbox,
            self.ui.step4_feature_er_checkbox,
            self.ui.step4_feature_rof_checkbox,
            self.ui.step4_feature_rtf_checkbox,
            self.ui.step4_averaged_features_checkbox,
            self.ui.step4_sliding_features_checkbox,
            self.ui.step5_use_tess_radio,
            self.ui.step5_use_avg_radio
        ]
        for item in control_items:
            if isinstance(item, QComboBox):
                item.activated.connect(self.mainwindow_controller)
            elif isinstance(item, QSpinBox):
                item.valueChanged.connect(self.mainwindow_controller)
            else:
                item.clicked.connect(self.mainwindow_controller)
        # Button connections for performing specific tasks
        click_actions = [
            (self.ui.step0_show_hide_log_window_button, self.comet_tbx.LogWindow.show_hide_log_window),
            (self.ui.step0_auto_pilot_button, self.run_autopilot),
            (self.ui.step0_load_study_button, self.load_study),
            (self.ui.step0_new_study_button, self.open_new_study_dialog),
            (self.ui.step0_compare_studies_button, self.open_compare_studies_window),
            (self.ui.step2_numberofmaps_elbow_button, self.visualize_elbow),
            (self.ui.step2_clustering_button, self.do_clustering),
            (self.ui.step3_label_maps_button, self.visualize_microstates),
            (self.ui.step3_backfit_button, self.do_backfitting),
            (self.ui.step4_extractfeatures_button, self.extract_features),
            (self.ui.step4_visualizefeatures_button, self.visualize_microstate_features),
            (self.ui.step3_backfit_visualization_button, self.visualize_microstate_segmentation),
            (self.ui.step5_coreg_button, self.coregister),# TODO
            (self.ui.step5_estimate_sources_button, self.source_localize_microstates),
            (self.ui.step5_compute_source_microstate_correlation_button, self.source_microstates_correlation),
            (self.ui.step5_visualize_sources_button, self.visualize_source_localized_microstates),
            (self.ui.step0_exit_button, self.exit_msg)
        ]
        trigger_actions = [
            (self.ui.open_github_action, self.open_github),
            (self.ui.report_issues_action, self.report_issues),
            (self.ui.update_action, self.update_toolbox),
            (self.ui.step0_new_study_action, self.open_new_study_dialog),
            (self.ui.step0_load_study_action, self.load_study),
            (self.ui.step0_reopen_log_window, self.comet_tbx.LogWindow.show_hide_log_window)
        ]
        for button, action in click_actions:
            button.clicked.connect(action)
        for item, action in trigger_actions:
            item.triggered.connect(action)

    def open_new_study_dialog(self):
        # Clear study name and reset processing flags
        self.ui.step0_study_name_mainwin_lineedit.clear()
        # Reset all processing flags to False
        processing_flags = [
            'done_preprocessing', 'done_clustering', 'done_labeling_microstates',
            'done_backfitting', 'done_extracting_features', 'done_source_localization',
            'done_source_microstate_correlation'
        ]
        self.reset_processing_flags(processing_flags)
        # Set NewStudyWindow modality and show maximized
        self.ui.NewStudyWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.NewStudyWindow.showMaximized()
        # Update the main window
        self.mainwindow_controller()

    def load_study_helper(self):
        # Get the folder containing preprocessed data
        self.save_folder = QFileDialog.getExistingDirectory(
            self, "Please choose the folder where the EEG-COMET study is located."
        )
        # Check if the eeg_comet_parameters.pkl file exists
        params_path = os.path.join(self.save_folder, 'eeg_comet_parameters.pkl')
        # Handle the case where loading the study fails
        if not os.path.exists(params_path):
            QMessageBox.information(self, "Load error",
                                    "The selected folder does not contain a valid study!",
                                    QMessageBox.Ok)
            return
        else:
            # Load the COMET parameters
            if self.comet_tbx.load_params(params_path):
                # Initialize the log window if needed
                if not hasattr(self.comet_tbx, 'LogWindow') or self.comet_tbx.LogWindow is None:
                    self.comet_tbx.initialize_log_window()

                self.comet_tbx.LogWindow.show()

                # Update the log window with loaded study information
                if hasattr(self.comet_tbx, 'log_text') and self.comet_tbx.log_text:
                    self.comet_tbx.LogWindow.replace_log(self.comet_tbx.log_text)

                self.comet_tbx.LogWindow.append_log(f"Study Loaded - ✓ Study Name: {self.comet_tbx.study_name}")
                return True
            else:
                QMessageBox.information(self, "Load error",
                                        "Failed to load study parameters!",
                                        QMessageBox.Ok)
                return False

    def load_study(self, from_new_study=False):
        if from_new_study:
            # No need to manually set paths - COMET already manages them
            # Just ensure paths are properly set up
            if hasattr(self.comet_tbx, 'reset_directories'):
                self.comet_tbx.reset_directories()

            # Initialize log window if needed
            if not hasattr(self.comet_tbx, 'LogWindow') or self.comet_tbx.LogWindow is None:
                self.comet_tbx.initialize_log_window()

            self.comet_tbx.LogWindow.show()
            self.comet_tbx.LogWindow.append_log(f"Study Created - ✓ Study Name: {self.comet_tbx.study_name}")
        else:
            if self.comet_tbx.done_preprocessing:
                ret = QMessageBox.question(self, 'MessageBox', f"The {self.comet_tbx.study_name} is already loaded,"
                                                               " do you want to load another study?",
                                           QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
                if ret == QMessageBox.Yes:
                    if self.load_study_helper():
                        # Ensure directories are properly set after loading
                        if hasattr(self.comet_tbx, 'reset_directories'):
                            self.comet_tbx.reset_directories()
            else:
                self.load_study_helper()

        # Update the main window
        self.mainwindow_controller()

    def open_compare_studies_window(self):
        self.ui.CompareStudiesWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.CompareStudiesWindow.showMaximized()

    @staticmethod
    def reset_option_box(box, options=None, current=None):
        """
        Reset the content of a combo box.
        """
        if options is None:
            options = []
        box.clear()
        box.addItems(options)
        if current:
            box.setCurrentText(current)

    def mainwindow_controller(self):
        # TODO: optimize the function
        """
        Control the visibility and enable/disable state of UI widgets based on conditions
        """
        current_tab = self.ui.main_tab.currentWidget().objectName()
        self.ui.main_tab.setStyleSheet("QTabBar::tab:selected { font-weight: bold; }")

        after_preprocessing_widgets = [
            self.ui.step2_line1,
            self.ui.step2_line2,
            self.ui.step2_line3,
            self.ui.step2_number_maps_label,
            self.ui.step0_auto_pilot_button,
            self.ui.step2_clustermethod_combo_label,
            self.ui.step2_clustermethod_combobox,
            self.ui.step2_advanced_checkbox,
            self.ui.step2_auto_k_radio,
            self.ui.step2_user_k_radio,
            self.ui.step2_numberofmaps_elbow_button,
            self.ui.step2_advanced_checkbox,
            self.ui.step2_clustering_button
        ]

        user_k_widgets = [
            self.ui.step2_user_k_input,
            self.ui.step2_numberofmaps_elbow_button
        ]

        auto_k_widgets = [
            self.ui.step2_auto_target_label,
            self.ui.step2_auto_target_parameter_label,
            self.ui.step2_stopping_threshold_input,
            self.ui.step2_auto_range_kmin_spinbox,
            self.ui.step2_auto_range_kmax_spinbox,
            self.ui.step2_auto_k_method_combobox,
            self.ui.step2_auto_range_label
        ]

        advanced_widgets = [
            self.ui.step2_line4,
            self.ui.step2_line5,
            self.ui.step2_line6,
            self.ui.step2_line7,
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
            self.ui.step2_convergence_label,
            self.ui.step2_maxiter_label,
            self.ui.step2_maxiter_input,
            self.ui.step2_stopcondition_label,
            self.ui.step2_stopcondition_input,
            self.ui.step2_numberofrepeats_label,
            self.ui.step2_user_numberofrepeats_input
        ]

        pca_widgets = [
            self.ui.step2_npca_label,
            self.ui.step2_npca_input
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
            self.ui.step3_line1,
            self.ui.step3_line2,
            self.ui.step3_line3,
            self.ui.step3_line4,
            self.ui.step3_backfit_label,
            self.ui.step3_label_maps_button,
            self.ui.step3_backfit_all_radio,
            self.ui.step3_backfit_peaks_radio,
            self.ui.step3_filter_segments_checkbox,
            self.ui.step3_identify_short_checkbox,
            self.ui.step3_backfit_button,
            self.ui.step3_backfit_visualization_button
        ]

        filter_segments_widgets = [
            self.ui.step3_identify_short_checkbox,
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

        identify_short_widgets = [
            self.ui.step3_filter_segments_input,
            self.ui.step3_filter_segments_label,
            self.ui.step3_filter_segments_label,
            self.ui.step3_filter_segments_label_2,
        ]

        feature_extraction_widgets = [
            self.ui.step4_line1,
            self.ui.step4_line2,
            self.ui.step4_line3,
            self.ui.step4_line4,
            self.ui.step4_features_extract_label,
            self.ui.step4_features_type_label,
            self.ui.step4_feature_occ_checkbox,
            self.ui.step4_feature_dur_checkbox,
            self.ui.step4_feature_cov_checkbox,
            self.ui.step4_feature_gev_checkbox,
            self.ui.step4_feature_tp_checkbox,
            self.ui.step4_feature_se_checkbox,
            self.ui.step4_feature_lzc_checkbox,
            self.ui.step4_feature_er_checkbox,
            self.ui.step4_features_epoched_label,
            self.ui.step4_feature_rof_checkbox,
            self.ui.step4_feature_rtf_checkbox,
            self.ui.step4_averaged_features_checkbox,
            self.ui.step4_sliding_features_checkbox,
            self.ui.step4_synthetic_checkbox,
            self.ui.step4_sliding_window_raw_label_0,
            self.ui.step4_sliding_window_raw_label_1,
            self.ui.step4_sliding_window_raw_input,
            self.ui.step4_sliding_window_epoched_label_0,
            self.ui.step4_sliding_window_epoched_label_1,
            self.ui.step4_sliding_window_epoched_label_2,
            self.ui.step4_sliding_window_epoched_input_pre,
            self.ui.step4_sliding_window_epoched_input_post,
            self.step4_word_size_label1,
            self.step4_word_size_label2,
            self.step4_word_size_label3,
            self.step4_word_size_min_input,
            self.step4_word_size_max_input,
            self.ui.step4_extractfeatures_button,
            self.ui.step4_outputformats_label,
            self.ui.step4_outputformats_combobox,
            self.ui.step4_visualizefeatures_button
            ]

        source_localization_widgets = [
            self.ui.step5_line1,
            self.ui.step5_line2,
            self.ui.step5_line3,
            self.ui.step5_line4,
            self.ui.step5_stc_label,
            self.ui.step5_anatomical_label,
            self.ui.step5_use_fsaverage_radio,
            self.ui.step5_use_individual_radio,
            self.ui.step5_inverse_method_label,
            self.ui.step5_inverse_method_combobox,
            self.ui.step5_use_tess_radio,
            self.ui.step5_use_avg_radio,
            self.ui.step5_spacing_label,
            self.ui.step5_spacing_combobox,
            self.ui.step5_coreg_button,
            self.ui.step5_estimate_sources_button,
            self.ui.step5_source_microstate_correlation_label,
            self.ui.step5_compute_source_microstate_correlation_button
        ]

        source_microstates_widgets = [
            self.ui.step5_source_microstate_correlation_label,
            self.ui.step5_use_tess_radio,
            self.ui.step5_use_avg_radio,
            self.ui.step5_permutations_label,
            self.ui.step5_permutations_input,
            self.ui.step5_compute_source_microstate_correlation_button,
        ]

        tess_widgets = [
            self.ui.step5_permutations_label,
            self.ui.step5_permutations_input,
        ]


        font_steps = QFont()
        font_steps.setPointSize(16)
        if self.comet_tbx.done_preprocessing:
            set_widgets_status(self.ui.main_tab, mode='show')
            self.ui.main_tab.setTabEnabled(0, True)

            if current_tab == "clustering_tab":

                # Show/Hide Widgets
                # set_widgets_status((after_preprocessing_widgets + user_k_widgets + auto_k_widgets), mode='show')
                set_widgets_status(after_preprocessing_widgets, mode='enable')

                widgets_to_rm = (
                        after_clustering_widgets +
                        filter_segments_widgets +
                        smooth_segments_widgets +
                        feature_extraction_widgets +
                        source_localization_widgets +
                        tess_widgets
                )
                # set_widgets_status(widgets_to_rm, mode='hide')
                set_widgets_status(widgets_to_rm, mode='disable')

                # Show/Hide Buttons
                # set_widgets_status([self.ui.step2_clustering_button, self.ui.step3_label_maps_button], mode='show')
                # set_widgets_status([self.ui.step3_backfit_button,
                #                     self.ui.step3_backfit_visualization_button,
                #                     self.ui.step4_extractfeatures_button,
                #                     self.ui.step4_visualizefeatures_button,
                #                     self.ui.step5_compute_source_microstate_correlation_button,
                #                     self.ui.step5_visualize_sources_button], mode='hide')

                # Hide the logo and show next steps
                self.ui.comet_label.setText("EEG-COMET")
                set_widgets_status(self.ui.comet_logo, mode='hide')

                self.ui.step0_study_name_mainwin_lineedit.setText(self.comet_tbx.study_name)
                self.ui.step0_study_name_mainwin_lineedit.setStyleSheet("background-color: lightgreen")
                if self.ui.step2_auto_k_radio.isChecked():
                    set_widgets_status(user_k_widgets, mode='disable')
                    set_widgets_status(auto_k_widgets, mode='enable')

                    kmin_value = int(self.ui.step2_auto_range_kmin_spinbox.value())
                    kmax_value = int(self.ui.step2_auto_range_kmax_spinbox.value())
                    if kmax_value <= kmin_value:
                        self.ui.step2_auto_range_kmax_spinbox.setValue(kmin_value + 1)

                    auto_k_method = self.ui.step2_auto_k_method_combobox.currentText()
                    if auto_k_method == 'Gap Statistic':
                        step2_auto_target_parameter_text = 'Random datasets:'
                    elif auto_k_method == 'Cross Validation':
                        step2_auto_target_parameter_text = 'Folds:'
                    elif auto_k_method in ['Elbow - Global Explained Variance', 'Elbow - Residual Variance']:
                        step2_auto_target_parameter_text = 'Threshold (%):'
                    else:
                        step2_auto_target_parameter_text = ''
                        # set_widgets_status([self.ui.step2_auto_target_parameter_label,
                        #                     self.ui.step2_stopping_threshold_input], mode='hide')
                    #    set_widgets_status(after_preprocessing_widgets, mode='hide')
                    self.ui.step2_auto_target_parameter_label.setText(step2_auto_target_parameter_text)

                if self.ui.step2_user_k_radio.isChecked():
                    set_widgets_status(user_k_widgets, mode='enable')
                    set_widgets_status(auto_k_widgets, mode='disable')

                if self.ui.step2_advanced_checkbox.isChecked():
                    set_widgets_status(advanced_widgets, mode='enable')
                    set_widgets_status(advanced_widgets, mode='show')
                    if self.ui.step2_use_peaks_radio.isChecked():
                        set_widgets_status(peaks2use_widgets, mode='enable')
                        set_widgets_status(rand2use_widgets, mode='disable')
                    elif self.ui.step2_use_percent_radio.isChecked():
                        set_widgets_status(rand2use_widgets, mode='enable')
                        set_widgets_status(peaks2use_widgets, mode='disable')

                    self.comet_tbx.clustering_method = self.step2_clustermethod_combobox.currentText()
                    if self.comet_tbx.clustering_method in ["K-Means Clustering",
                                                            'PCA + K-Means Clustering',
                                                            'Autoencoder + K-Means Clustering']:
                        self.ui.step2_other_label.setText("Similarity metric:")
                        options = ['Cosine Similarity', 'Spatial Correlation']
                        self.reset_option_box(self.ui.step2_other_options_combobox, options, 'Spatial Correlation')
                        if self.comet_tbx.clustering_method == 'PCA + K-Means Clustering':
                            set_widgets_status(pca_widgets, mode='enable')
                            # set_widgets_status(pca_widgets, mode='show')
                        else:
                            set_widgets_status(pca_widgets, mode='disable')
                            # set_widgets_status(pca_widgets, mode='hide')
                    elif self.comet_tbx.clustering_method == "X-Means Clustering":
                        self.ui.step2_other_label.setText("X-means splitting criterion:")
                        options = ['Bayesian Information Criterion', 'Minimum Noiseless Description Length']
                        self.reset_option_box(self.ui.step2_other_options_combobox,
                                              options, 'Bayesian Information Criterion')
                    # else:
                        # set_widgets_status([self.ui.step2_other_options_combobox,
                        #                     self.ui.step2_other_label], mode='hide')

                else:
                    set_widgets_status((advanced_widgets + pca_widgets), mode='disable')
                    set_widgets_status((advanced_widgets + pca_widgets), mode='hide')

        else:
            self.ui.step0_study_name_mainwin_lineedit.setStyleSheet("background-color: none")
            set_widgets_status(self.ui.main_tab, mode='hide')
            # set_widgets_status((after_preprocessing_widgets + user_k_widgets + auto_k_widgets), mode='hide')
            set_widgets_status(after_preprocessing_widgets, mode='disable')
            # Show/Hide Buttons
            # set_widgets_status([self.ui.step2_clustering_button,
            #                     self.ui.step3_label_maps_button
            #                     ], mode='hide')

        if self.comet_tbx.done_clustering:
            self.ui.main_tab.setTabEnabled(1, True)
            set_widgets_status(self.ui.step3_label_maps_button, mode='enable')

            if current_tab == "backfitting_tab":
                self.ui.step2_clustering_button.setStyleSheet("background-color: lightgreen")

                # set_widgets_status(after_clustering_widgets, mode='show')
                set_widgets_status(after_clustering_widgets, mode='enable')

                widgets_to_rm = (
                        after_preprocessing_widgets +
                        pca_widgets +
                        user_k_widgets +
                        auto_k_widgets +
                        advanced_widgets +
                        feature_extraction_widgets +
                        source_localization_widgets +
                        tess_widgets
                )
                # set_widgets_status(widgets_to_rm, mode='hide')
                set_widgets_status(widgets_to_rm, mode='disable')

                # Show/Hide Buttons
                # set_widgets_status([self.ui.step3_backfit_button,
                #                     self.ui.step3_backfit_visualization_button], mode='show')
                # set_widgets_status(identify_short_widgets, mode='show')
                # set_widgets_status(filter_segments_widgets, mode='show')
                # set_widgets_status(smooth_segments_widgets, mode='show')
                # set_widgets_status([self.ui.step2_clustering_button,
                #                     self.ui.step3_label_maps_button,
                #                     self.ui.step4_extractfeatures_button,
                #                     self.ui.step4_visualizefeatures_button,
                #                     self.ui.step5_compute_source_microstate_correlation_button,
                #                     self.ui.step5_visualize_sources_button], mode='hide')

                outputformat = self.ui.step4_outputformats_combobox.currentText()
                opening_parenthesis = outputformat.find("(")
                closing_parenthesis = outputformat.find(")")
                if opening_parenthesis != -1 and closing_parenthesis != -1:
                    self.comet_tbx.export_format = outputformat[opening_parenthesis + 1: closing_parenthesis]

                if self.ui.step3_backfit_peaks_radio.isChecked():
                    self.ui.step3_filter_segments_checkbox.setChecked(False)
                    set_widgets_status(self.ui.step3_filter_segments_checkbox, mode='disable')
                else:
                    set_widgets_status(self.ui.step3_filter_segments_checkbox, mode='enable')

                if self.ui.step3_filter_segments_checkbox.isChecked():
                    set_widgets_status(filter_segments_widgets, mode='enable')
                    filter_segments_method = self.ui.step3_filter_segments_method_combobox.currentText()
                    if filter_segments_method == "Smooth segments":
                        set_widgets_status(smooth_segments_widgets, mode='enable')
                    else:
                        set_widgets_status(smooth_segments_widgets, mode='disable')

                    if not self.ui.step3_identify_short_checkbox.isChecked():
                        set_widgets_status(identify_short_widgets, mode='enable')
                    else:
                        set_widgets_status(identify_short_widgets, mode='disable')
                        if filter_segments_method == "Smooth segments":
                            set_widgets_status(smooth_segments_widgets, mode='disable')
                else:
                    set_widgets_status(filter_segments_widgets, mode='disable')
                    set_widgets_status(smooth_segments_widgets, mode='disable')

        else:
            # Reset next steps processing flags to False
            processing_flags = [
                'done_labeling_microstates', 'done_backfitting', 'done_extracting_features',
                'done_source_localization', 'done_source_microstate_correlation'
            ]
            self.reset_processing_flags(processing_flags)
            self.ui.step2_clustering_button.setStyleSheet("background-color: none")
            self.comet_tbx.best_maps, self.comet_tbx.micro_labels = None, []
            # set ALL disabled
            self.ui.main_tab.setTabEnabled(2, False)
            self.ui.main_tab.setTabEnabled(3, False)
            set_widgets_status(self.ui.step3_filter_segments_checkbox, mode='disable')
            set_widgets_status(after_clustering_widgets, mode='disable')
            set_widgets_status(filter_segments_widgets, mode='disable')

        if self.comet_tbx.done_labeling_microstates:
            self.ui.step3_label_maps_button.setStyleSheet("background-color: lightgreen")
        else:
            self.ui.step3_label_maps_button.setStyleSheet("background-color: none")
            # Reset next steps processing flags to False
            processing_flags = [
                'done_backfitting', 'done_extracting_features',
                'done_source_localization', 'done_source_microstate_correlation'
            ]
            self.reset_processing_flags(processing_flags)

        if self.comet_tbx.done_backfitting:
            self.ui.step3_backfit_button.setStyleSheet("background-color: lightgreen")
            self.ui.main_tab.setTabEnabled(2, True)
            self.ui.main_tab.setTabEnabled(3, True)
            self.ui.step3_backfit_visualization_button.setEnabled(True)

            if current_tab == "feature_tab":

                # set_widgets_status(feature_extraction_widgets, mode='show')
                set_widgets_status(feature_extraction_widgets, mode='enable')

                widgets_to_rm = (
                        after_preprocessing_widgets +
                        after_clustering_widgets +
                        user_k_widgets +
                        auto_k_widgets +
                        advanced_widgets +
                        filter_segments_widgets +
                        smooth_segments_widgets +
                        source_localization_widgets +
                        tess_widgets
                )
                # set_widgets_status(widgets_to_rm, mode='hide')
                set_widgets_status(widgets_to_rm, mode='disable')

                # Show/Hide Buttons
                # set_widgets_status([self.ui.step4_extractfeatures_button,
                #                     self.ui.step4_visualizefeatures_button], mode='show')
                # set_widgets_status([self.ui.step2_clustering_button,
                #                     self.ui.step3_label_maps_button,
                #                     self.ui.step3_backfit_button,
                #                     self.ui.step3_backfit_visualization_button,
                #                     self.ui.step5_compute_source_microstate_correlation_button,
                #                     self.ui.step5_visualize_sources_button], mode='hide')

                sliding_feature_extraction_raw_widgets = [
                    self.ui.step4_sliding_window_raw_label_0,
                    self.ui.step4_sliding_window_raw_label_1,
                    self.ui.step4_sliding_window_raw_input
                ]

                feature_extraction_epoched_widgets = [
                    self.ui.step4_features_epoched_label,
                    self.ui.step4_feature_rof_checkbox,
                    self.ui.step4_feature_rtf_checkbox
                ]

                sliding_feature_extraction_epoched_widgets = [
                    self.ui.step4_sliding_window_epoched_label_0,
                    self.ui.step4_sliding_window_epoched_label_1,
                    self.ui.step4_sliding_window_epoched_label_2,
                    self.ui.step4_sliding_window_epoched_input_pre,
                    self.ui.step4_sliding_window_epoched_input_post
                ]

                if self.comet_tbx.datatype == 'epoched':
                    self.ui.step4_sliding_features_checkbox.setText("Extract Features Before and After TMS per Subject")
                    set_widgets_status(feature_extraction_epoched_widgets, mode='enable')
                    self.ui.step4_feature_rof_checkbox.setChecked(True)
                    self.ui.step4_feature_rtf_checkbox.setChecked(True)
                    set_widgets_status(sliding_feature_extraction_raw_widgets, mode='disable')
                    # set_widgets_status(sliding_feature_extraction_raw_widgets, mode='hide')
                    if self.ui.step4_sliding_features_checkbox.isChecked():
                        set_widgets_status(sliding_feature_extraction_epoched_widgets, mode='enable')
                    else:
                        set_widgets_status(sliding_feature_extraction_epoched_widgets, mode='disable')
                else:
                    set_widgets_status(feature_extraction_epoched_widgets, mode='disable')
                    self.ui.step4_feature_rof_checkbox.setChecked(False)
                    self.ui.step4_feature_rtf_checkbox.setChecked(False)
                    set_widgets_status(sliding_feature_extraction_epoched_widgets, mode='disable')
                    # set_widgets_status(sliding_feature_extraction_epoched_widgets, mode='hide')
                    if self.ui.step4_sliding_features_checkbox.isChecked():
                        set_widgets_status(sliding_feature_extraction_raw_widgets, mode='enable')
                    else:
                        set_widgets_status(sliding_feature_extraction_raw_widgets, mode='disable')

                microsynt_feature_extraction_widgets = [
                    self.step4_word_size_label1,
                    self.step4_word_size_label2,
                    self.step4_word_size_label3,
                    self.step4_word_size_min_input,
                    self.step4_word_size_max_input
                ]
                if self.ui.step4_feature_er_checkbox.isChecked():
                    set_widgets_status(microsynt_feature_extraction_widgets, mode='enable')
                else:
                    set_widgets_status(microsynt_feature_extraction_widgets, mode='disable')

                feature_checkboxes1 = [
                    self.ui.step4_feature_occ_checkbox,
                    self.ui.step4_feature_dur_checkbox,
                    self.ui.step4_feature_cov_checkbox,
                    self.ui.step4_feature_gev_checkbox,
                    self.ui.step4_feature_tp_checkbox,
                    self.ui.step4_feature_se_checkbox,
                    self.ui.step4_feature_lzc_checkbox,
                    self.ui.step4_feature_er_checkbox,
                    self.ui.step4_feature_rof_checkbox,
                    self.ui.step4_feature_rtf_checkbox
                ]
                feature_checkboxes2 = [
                    self.ui.step4_averaged_features_checkbox,
                    self.ui.step4_sliding_features_checkbox,
                    self.ui.step4_synthetic_checkbox
                ]
                self.ui.step4_extractfeatures_button.setDisabled(
                    not any(checkbox.isChecked() for checkbox in feature_checkboxes1) or
                    not any(checkbox.isChecked() for checkbox in feature_checkboxes2))

                if self.comet_tbx.done_extracting_features:
                    self.ui.step4_extractfeatures_button.setStyleSheet("background-color: lightgreen")
                    self.ui.step4_visualizefeatures_button.setEnabled(True)
                else:
                    self.ui.step4_extractfeatures_button.setStyleSheet("background-color: none")
                    self.ui.step4_visualizefeatures_button.setDisabled(True)
            else:
                set_widgets_status(feature_extraction_widgets, mode='disable')

            if current_tab == "source_tab":
                # set_widgets_status((source_localization_widgets + tess_widgets), mode='show')
                set_widgets_status(source_localization_widgets, mode='enable')

                widgets_to_rm = (
                        after_preprocessing_widgets +
                        after_clustering_widgets +
                        user_k_widgets +
                        auto_k_widgets +
                        advanced_widgets +
                        filter_segments_widgets +
                        smooth_segments_widgets +
                        feature_extraction_widgets
                )
                # set_widgets_status(widgets_to_rm, mode='hide')
                set_widgets_status(widgets_to_rm, mode='disable')

                # Show/Hide Buttons
                # set_widgets_status([self.ui.step5_compute_source_microstate_correlation_button,
                #                     self.ui.step5_visualize_sources_button], mode='show')
                # set_widgets_status([self.ui.step2_clustering_button,
                #                     self.ui.step3_label_maps_button,
                #                     self.ui.step3_backfit_button,
                #                     self.ui.step3_backfit_visualization_button,
                #                     self.ui.step4_extractfeatures_button,
                #                     self.ui.step4_visualizefeatures_button], mode='hide')

                # Determine whether to use fsaverage or individual anatomy
                if self.step5_use_individual_radio.isChecked():
                    self.comet_tbx.use_anatomy = "individual"
                else:
                    self.comet_tbx.use_anatomy = "fsaverage"
                    # self.comet_tbx.individual_subjects_dir = QFileDialog.getExistingDirectory(
                    #     self,
                    #     "Locate Folder with Individual Anatomical Reconstructions"
                    # )

                if self.comet_tbx.done_source_localization:
                    self.ui.step5_estimate_sources_button.setStyleSheet("background-color: lightgreen")
                    set_widgets_status(source_microstates_widgets, mode='enable')
                    if not self.ui.step5_use_tess_radio.isChecked():
                        set_widgets_status(tess_widgets, mode='disable')
                    else:
                        set_widgets_status(tess_widgets, mode='enable')
                else:
                    self.ui.step5_estimate_sources_button.setStyleSheet("background-color: none")
                    set_widgets_status(source_microstates_widgets, mode='disable')

                if self.comet_tbx.done_source_microstate_correlation:
                    self.ui.step5_compute_source_microstate_correlation_button.setStyleSheet(
                        "background-color: lightgreen")
                    self.ui.step5_visualize_sources_button.setEnabled(True)
                else:
                    self.ui.step5_compute_source_microstate_correlation_button.setStyleSheet("background-color: none")
                    self.ui.step5_visualize_sources_button.setDisabled(True)

            else:
                # set_widgets_status(source_localization_widgets, mode='hide')
                set_widgets_status(source_localization_widgets, mode='disable')
        else:
            # Reset next steps processing flags to False
            self.reset_processing_flags('done_extracting_features')
            self.ui.step3_backfit_button.setStyleSheet("background-color: none")
            self.ui.step3_backfit_visualization_button.setDisabled(True)
            set_widgets_status((feature_extraction_widgets +
                                source_localization_widgets), mode='disable')
            # set_widgets_status((feature_extraction_widgets +
            #                     source_localization_widgets), mode='hide')

    def visualize_elbow(self):
        """
        Initiates the visualization of elbow plots for optimizing the number of clusters,
        based on clustering results and user-specified parameters.
        """
        # Check if clustering is done and the number of maps is chosen automatically
        if self.comet_tbx.done_clustering and self.comet_tbx.choose_number_of_maps == "auto":
            # Set attributes in OptimizerVisualizationWindow to indicate optimization is done
            setattr(self.OptimizerVisualizationWindow, f'optimizer_{self.comet_tbx.stopping_mode}_done', True)
            # Initialize optimizer_results dictionary if not already done
            self.OptimizerVisualizationWindow.optimizer_results = {}
            # Store clustering results in optimizer_results dictionary
            self.OptimizerVisualizationWindow.optimizer_results[self.comet_tbx.stopping_mode] = {
                'optimal_k': self.comet_tbx.optimal_k,
                'k_values': self.comet_tbx.k_values,
                'target_values': self.comet_tbx.target_values
            }
        else:
            # Initialize optimizer_results dictionary if clustering is not done
            self.OptimizerVisualizationWindow.optimizer_results = {}
        # Set attributes in OptimizerVisualizationWindow related to data and visualization
        self.OptimizerVisualizationWindow.preprocessed_data_path = self.comet_tbx.preprocessed_data_path
        self.OptimizerVisualizationWindow.extension = self.comet_tbx.extension
        self.OptimizerVisualizationWindow.datatype = self.comet_tbx.datatype
        # Check if using percentages and set the value accordingly
        if self.ui.step2_use_percent_radio.isChecked():
            self.comet_tbx.use_percentages = self.ui.step2_percent_combobox.currentText()
        else:
            self.comet_tbx.use_percentages = None
        # Set attributes in OptimizerVisualizationWindow
        self.OptimizerVisualizationWindow.use_percentages = self.comet_tbx.use_percentages
        self.OptimizerVisualizationWindow.min_distance_size = int(int(self.ui.step2_kernel_size_input.text()) /
                                                                  (1000/self.comet_tbx.sample_rate))
        self.OptimizerVisualizationWindow.clustering_tolerance = float(self.ui.step2_stopcondition_input.text())
        self.OptimizerVisualizationWindow.number_of_repeats = int(self.ui.step2_user_numberofrepeats_input.text())
        self.comet_tbx.max_iterations = int(self.ui.step2_maxiter_input.text())
        self.OptimizerVisualizationWindow.max_iterations = self.comet_tbx.max_iterations
        # Set modality and show the OptimizerVisualizationWindow
        self.OptimizerVisualizationWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.OptimizerVisualizationWindow.showMaximized()

    def run_autopilot(self):
        """

        """
        # TODO
        self.comet_tbx.do_autopilot()

    def do_clustering(self):
        """
        Initiates the clustering process based on user-specified parameters and performs clustering on EEG data.
        """
        # Check if the analysis is already done.
        if self.comet_tbx.done_clustering:
            ret = QMessageBox.question(self, 'MessageBox', "Data has been clustered once,"
                                                           " do you want to redo the analysis?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_clustering_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_clustering_from_scratch = True
                # Reset all processing flags to False
                processing_flags = [
                    'done_clustering', 'done_labeling_microstates',
                    'done_backfitting', 'done_extracting_features', 'done_source_localization',
                    'done_source_microstate_correlation'
                ]
                self.reset_processing_flags(processing_flags)
        else:
            self.do_clustering_from_scratch = True

        if self.do_clustering_from_scratch:
            # Process user-specified clustering parameters
            if self.ui.step2_kernel_size_input.text():
                self.comet_tbx.smoothing_gfp = True
                self.comet_tbx.smoothing_distance = int(self.ui.step2_kernel_size_input.text())
            else:
                self.comet_tbx.smoothing_gfp = False
                self.comet_tbx.smoothing_distance = ''
                # self.min_distance_size = []
            if self.ui.step2_auto_k_radio.isChecked():
                self.comet_tbx.choose_number_of_maps = "auto"
                self.comet_tbx.kmin = int(self.ui.step2_auto_range_kmin_spinbox.value())
                self.comet_tbx.kmax = int(self.ui.step2_auto_range_kmax_spinbox.value())
                auto_k_method = self.ui.step2_auto_k_method_combobox.currentText()
                if auto_k_method == 'Gap Statistic':
                    self.comet_tbx.stopping_mode = 'gs'
                elif auto_k_method == 'Cross Validation':
                    self.comet_tbx.stopping_mode = 'cv'
                elif auto_k_method == 'Elbow - Global Explained Variance':
                    self.comet_tbx.stopping_mode = 'gev'
                elif auto_k_method == 'Elbow - Residual Variance':
                    self.comet_tbx.stopping_mode = 'res'
                elif auto_k_method == 'Silhouette Method':
                    self.comet_tbx.stopping_mode = 'sil'
                elif auto_k_method == 'Calinski-Harabasz Method':
                    self.comet_tbx.stopping_mode = 'ch'
                elif auto_k_method == 'Davies-Bouldin Method':
                    self.comet_tbx.stopping_mode = 'db'
                self.comet_tbx.stopping_parameter = float(self.ui.step2_stopping_threshold_input.text())
                self.comet_tbx.number_of_maps = 'auto'
            elif self.ui.step2_user_k_radio.isChecked():
                self.comet_tbx.choose_number_of_maps = "user"
                self.comet_tbx.stopping_mode = ''
                self.comet_tbx.stopping_parameter = ''
                self.comet_tbx.kmin = ''
                self.comet_tbx.kmax = ''
                self.comet_tbx.number_of_maps = int(self.ui.step2_user_k_input.text())
            if self.ui.step2_random_initializer_radio.isChecked():
                self.comet_tbx.initializer = "Random"
            elif self.ui.step2_kmeans_initializer_radio.isChecked():
                self.comet_tbx.initializer = "K-Means++"
            self.comet_tbx.clustering_method = self.ui.step2_clustermethod_combobox.currentText()
            self.comet_tbx.n_pca = int(self.ui.step2_npca_input.text())
            if self.ui.step2_use_percent_radio.isChecked():
                self.comet_tbx.use_percentages = self.ui.step2_percent_combobox.currentText()
            else:
                self.comet_tbx.use_percentages = None
            self.comet_tbx.max_iterations = int(self.ui.step2_maxiter_input.text())
            self.comet_tbx.clustering_tolerance = float(self.ui.step2_stopcondition_input.text())
            self.comet_tbx.clustering_option = self.ui.step2_other_options_combobox.currentText()
            self.comet_tbx.number_of_repeats = int(self.ui.step2_user_numberofrepeats_input.text())
            self.comet_tbx.microstate_maps_path = os.path.join(self.comet_tbx.save_dir, 'microstate_maps.csv')

            # Perform clustering
            self.comet_tbx.do_clustering()
            # Label the maps and save the state
            self.visualize_microstates()
            # self.comet_tbx.save_tbx()
            # Update the main window
            self.ui.main_tab.setTabEnabled(1, True)
            self.mainwindow_controller()

    def do_backfitting(self):
        """
        Perform microstate backfitting.
        """
        # Check if backfitting has already been done
        if self.comet_tbx.done_backfitting:
            ret = QMessageBox.question(self, 'MessageBox', "Microstates have been backfitted to data once,"
                                                           " do you want to redo backfitting_utils?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_backfitting_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_backfitting_from_scratch = True
        else:
            self.do_backfitting_from_scratch = True

        if self.do_backfitting_from_scratch:
            # Reset next steps processing flags to False
            processing_flags = [
                'done_extracting_features', 'done_source_localization', 'done_source_microstate_correlation'
            ]
            self.reset_processing_flags(processing_flags)
            # Set backfitting parameters based on user input
            if self.ui.step3_backfit_all_radio.isChecked():
                self.comet_tbx.backfit_to = 'all'
                if self.ui.step3_identify_short_checkbox.isChecked():
                    self.comet_tbx.identify_short_window = True
                else:
                    self.comet_tbx.identify_short_window = False
            elif self.ui.step3_backfit_peaks_radio.isChecked():
                self.comet_tbx.backfit_to = 'peaks'
                self.comet_tbx.identify_short_window = False
            self.comet_tbx.epsilon = ''
            self.comet_tbx.b = ''
            self.comet_tbx.lamb = ''
            if self.ui.step3_filter_segments_checkbox.isChecked():
                self.comet_tbx.filter_segments = True
                self.comet_tbx.remove_segments_less_than = int(int(self.ui.step3_filter_segments_input.text()) /
                                                               (1000/self.comet_tbx.sample_rate))
                filter_segments_method = self.ui.step3_filter_segments_method_combobox.currentText()
                if filter_segments_method == 'Replace short segments: nearby dominant microstate':
                    self.comet_tbx.filter_segments_option = 'replace_high'
                elif filter_segments_method == 'Replace short segments: half and half':
                    self.comet_tbx.filter_segments_option = 'replace_half'
                elif filter_segments_method == 'Remove short segments':
                    self.comet_tbx.filter_segments_option = 'remove'
                elif filter_segments_method == 'Smooth segments':
                    self.comet_tbx.filter_segments_option = 'smooth'
                    self.comet_tbx.epsilon = float(self.ui.step3_smooth_segments_epsilon_input.text())
                    self.comet_tbx.b = self.comet_tbx.remove_segments_less_than
                    self.comet_tbx.lamb = int(self.ui.step3_smooth_segments_lambda_input.text())
            else:
                self.comet_tbx.filter_segments = False
                self.comet_tbx.filter_segments_option = ''
                self.comet_tbx.remove_segments_less_than = []
            # Log and perform backfitting
            self.comet_tbx.do_backfitting()
            # Save the state
            self.comet_tbx.save_tbx()
            # Update the main window
            self.ui.main_tab.setTabEnabled(2, True)
            self.mainwindow_controller()

    def visualize_microstates(self):
        """
        Visualize microstate maps for labeling.
        """
        self.ui.MicrostateVisualizationWindow = MicrostateVisualizationWindow(
            self.context,
            main_window=self,
            tbx=self.comet_tbx
        )
        self.ui.MicrostateVisualizationWindow.plot_maps()
        self.ui.MicrostateVisualizationWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.MicrostateVisualizationWindow.showMaximized()

    def visualize_microstate_segmentation(self):
        """
        Open BackfittingVisualizationWindow to visualize microstate segmentation.
        """
        # Set relevant paths and parameters for visualization
        self.BackfittingVisualizationWindow.preprocessed_data_path = self.comet_tbx.preprocessed_data_path
        self.BackfittingVisualizationWindow.extension = self.comet_tbx.extension
        self.BackfittingVisualizationWindow.datatype = self.comet_tbx.datatype
        self.BackfittingVisualizationWindow.eeg_filenames_combobox.addItems([i for i in self.comet_tbx.list_eegs])
        self.BackfittingVisualizationWindow.segmentation_path = self.comet_tbx.segmentation_path
        self.BackfittingVisualizationWindow.export_format = self.comet_tbx.export_format
        # Display the Backfitting Visualization Dialog
        self.BackfittingVisualizationWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.BackfittingVisualizationWindow.showMaximized()

    def extract_features(self):
        """
        Extract features from the backfitted data.
        """
        # Check if features have been extracted before
        if self.comet_tbx.done_extracting_features:
            ret = QMessageBox.question(self, 'MessageBox', "Features have been extracted once,"
                                                           " do you want to extract features_utils again?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_extracting_features_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_extracting_features_from_scratch = True
        else:
            self.do_extracting_features_from_scratch = True
        # Proceed with feature extraction
        if self.do_extracting_features_from_scratch:
            # Reset processing flags and update the main window
            self.comet_tbx.done_extracting_features = False
            self.mainwindow_controller()

            # Define features to extract based on user selection
            self.comet_tbx.feature_list = []
            if self.ui.step4_feature_occ_checkbox.isChecked():
                self.comet_tbx.feature_list.append("OCC")
            if self.ui.step4_feature_dur_checkbox.isChecked():
                self.comet_tbx.feature_list.append("DUR")
            if self.ui.step4_feature_cov_checkbox.isChecked():
                self.comet_tbx.feature_list.append("COV")
            if self.ui.step4_feature_gev_checkbox.isChecked():
                self.comet_tbx.feature_list.append("GEV")
            if self.ui.step4_feature_tp_checkbox.isChecked():
                self.comet_tbx.feature_list.append("TP")
            if self.ui.step4_feature_se_checkbox.isChecked():
                self.comet_tbx.feature_list.append("SE")
            if self.ui.step4_feature_lzc_checkbox.isChecked():
                self.comet_tbx.feature_list.append("LZC")
            if self.ui.step4_feature_er_checkbox.isChecked():
                self.comet_tbx.feature_list.append("ER")
                self.comet_tbx.word_size = int(self.ui.step4_word_size_min_input.text())
            else:
                self.comet_tbx.word_size = 2


            # Define feature extraction modes
            self.comet_tbx.feature_mode = []
            if self.ui.step4_averaged_features_checkbox.isChecked():
                self.comet_tbx.feature_mode.append("averaged")
            if self.ui.step4_sliding_features_checkbox.isChecked():
                self.comet_tbx.feature_mode.append("sliding")
            if self.ui.step4_synthetic_checkbox.isChecked():
                self.comet_tbx.feature_types = ['real', 'surrogate', 'random']
            else:
                self.comet_tbx.feature_types = ['real']

            if self.comet_tbx.datatype == 'epoched':
                # Set window size for pre post features
                self.comet_tbx.pre_window_size = int(self.ui.step4_sliding_window_epoched_input_pre.text())
                self.comet_tbx.post_window_size = int(self.ui.step4_sliding_window_epoched_input_post.text())
                if self.ui.step4_feature_rof_checkbox.isChecked():
                    self.comet_tbx.feature_list.append("ROF")
                if self.ui.step4_feature_rtf_checkbox.isChecked():
                    self.comet_tbx.feature_list.append("RTF")
            else:
                # Set sliding window size for dynamic features
                self.comet_tbx.sliding_window_size = int(self.ui.step4_sliding_window_raw_input.text())
            # Log and perform feature extraction
            self.comet_tbx.extract_features()
            # Update flags and save the state
            self.comet_tbx.done_extracting_features = True
            self.comet_tbx.save_tbx()
            # Update the main window
            self.mainwindow_controller()

    def visualize_microstate_features(self):
        """
        Open FeatureVisualizationWindow to visualize microstate features.
        """
        # Set relevant paths and parameters for visualization
        self.ui.FeatureVisualizationWindow.extracted_features_path = self.comet_tbx.extracted_features_path
        self.ui.FeatureVisualizationWindow.export_format = self.comet_tbx.export_format
        self.ui.FeatureVisualizationWindow.feature_mode = self.comet_tbx.feature_mode
        self.ui.FeatureVisualizationWindow.feature_combo.clear()
        self.ui.FeatureVisualizationWindow.feature_combo.addItems([i for i in self.comet_tbx.feature_list])
        self.ui.FeatureVisualizationWindow.list_eegs = self.comet_tbx.list_eegs
        self.ui.FeatureVisualizationWindow.reset_groups()
        # Display the Feature Visualization Dialog
        self.ui.FeatureVisualizationWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.FeatureVisualizationWindow.showMaximized()

    def coregister(self):
        """
        Open CoregistrationWindow to align EEG Sensors to head source space.
        """
        self.ui.CoregistrationWindow = CoregistrationWindow(self.context, tbx=self.comet_tbx)
        self.ui.CoregistrationWindow.showMaximized()

    def source_localize_microstates(self):
        """
        Perform source localization of data.
        """
        # Check if source localization is already done
        if self.comet_tbx.done_source_localization:
            ret = QMessageBox.question(self, 'MessageBox', "Source time series have been extracted once,"
                                                           " do you want to extract source time series again?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_source_localization_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_source_localization_from_scratch = True
        else:
            self.do_source_localization_from_scratch = True

        # If source localization needs to be done from scratch
        if self.do_source_localization_from_scratch:
            # Reset next steps processing flags to False
            processing_flags = [
                'done_source_localization', 'done_source_microstate_correlation'
            ]
            self.reset_processing_flags(processing_flags)
            # Update the main window
            self.mainwindow_controller()
            # Determine whether to use fsaverage or individual anatomy
            if self.step5_use_fsaverage_radio.isChecked():
                self.comet_tbx.use_anatomy = "fsaverage"
            elif self.step5_use_individual_radio.isChecked():
                self.comet_tbx.use_anatomy = "individual"
                self.comet_tbx.individual_subjects_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Locate Folder with Individual Anatomical Reconstructions"
                )
            # Get the inverse method and spacing options
            inverse_method = self.ui.step5_inverse_method_combobox.currentText()
            self.comet_tbx.inverse_method = inverse_method[inverse_method.find("(") + 1:inverse_method.find(")")]
            spacing = self.ui.step5_spacing_combobox.currentText()
            self.comet_tbx.spacing = spacing[spacing.find("(") + 1:spacing.find(")")].lower()
            self.comet_tbx.nperm = int(self.ui.step5_permutations_input.text())
            # Perform the source localization and save the state
            self.comet_tbx.source_localize_microstates()
            self.comet_tbx.save_tbx()
            # Update the main window
            self.mainwindow_controller()

    def source_microstates_correlation(self):
        """
        Identify sources correlated to each microstate.
        """
        # Check if microstate source localization is already done
        if self.comet_tbx.done_source_microstate_correlation:
            ret = QMessageBox.question(self, 'MessageBox',
                                       "Source-microstate correlations have already been calculated,"
                                       " do you want to run this step again?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_source_microstate_correlation_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_source_microstate_correlation_from_scratch = True
        else:
            self.do_source_microstate_correlation_from_scratch = True
        # If microstate source localization need to be calculated from scratch
        if self.do_source_microstate_correlation_from_scratch:
            # TODO: Add missing options here
            self.comet_tbx.nperm = int(self.ui.step5_permutations_input.text())
            # Perform the microstate source localization and save the state
            self.comet_tbx.source_microstate_correlation()
            self.comet_tbx.done_source_microstate_correlation = True
            self.comet_tbx.save_tbx()
            # Update the main window
            self.mainwindow_controller()

    def visualize_source_localized_microstates(self):
        """
        Open SourceVisualizationWindow to visualize microstates localized sources.
        """
        # TODO
        # Set relevant paths and parameters for visualization
        self.ui.SourceVisualizationWindow = SourceVisualizationWindow(self.context, comet_tbx=self.comet_tbx)
        #self.SourceVisualizationWindow.tess_path = os.path.join(self.comet_tbx.localized_sources_path, "tess_sources")
        # Display the Source Visualization Dialog
        self.SourceVisualizationWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.SourceVisualizationWindow.showMaximized()

    def exit_msg(self):
        """
        Display a confirmation message before quitting the application.
        """
        # Show a confirmation dialog
        reply = QMessageBox.question(self, "Quit",
                                     "Are you sure you want to quit?",
                                     QMessageBox.Yes |
                                     QMessageBox.No)
        # Check the user's response
        if reply == QMessageBox.Yes:
            # Close the log window
            self.comet_tbx.LogWindow.close()
            # Close the main window
            self.close()
