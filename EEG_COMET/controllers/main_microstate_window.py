import os.path
import webbrowser
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QComboBox, QSpinBox, QSlider, QMessageBox,
                             QGraphicsDropShadowEffect)
from PyQt5.QtGui import QPixmap, QFont, QColor
from PyQt5.QtCore import Qt

from .new_study_window import NewStudyWindow
from .compare_studies_window import CompareStudiesWindow
from .microstate_visualization_window import MicrostateVisualizationWindow
from .optimizer_visualization_window import OptimizerVisualizationWindow
from .backfitting_visualization_window import BackfittingVisualizationWindow
from .feature_visualization_window import FeatureVisualizationWindow
from .coregistration_window import CoregistrationWindow
from .source_visualization_window import SourceVisualizationWindow
from gui_utils.set_widgets_status import set_widgets_status
from comet import COMET


class MainMicrostateWindow(QMainWindow):
    def __init__(self, context, parent=None):
        super(MainMicrostateWindow, self).__init__(parent)

        # Initialize key components
        self.comet = COMET()
        self.comet.initialize_log_window()
        self.comet.LogWindow.show()
        self.context = context
        
        # Load the UI from the .ui file
        self.ui = uic.loadUi(context.get_resource("MainMicrostateWindow.ui"), self)
        self.ui.setWindowTitle("EEG-COMET")
        # self.ui.showMaximized()

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
        webbrowser.open('https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction')

    @staticmethod
    def report_issues():
        """
        Open the GitHub issues page in the default web browser
        """
        webbrowser.open('https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction/issues/new')

    def update_toolbox(self):
        """
        Ask the user if they want to download the toolbox
        """
        ret = QMessageBox.question(self, 'MessageBox', "Do you want to download the toolbox?",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
        if ret == QMessageBox.Yes:
            webbrowser.open(
                'https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction/archive/refs/heads/main.zip')

    def init_dialogs(self):
        # Create and initialize NewStudyWindow
        self.ui.NewStudyWindow = NewStudyWindow(
            self.context,
            main_window=self,
            comet_tbx=self.comet
        )

        self.ui.CompareStudiesWindow = CompareStudiesWindow(self.context)

        # Create and initialize MicrostateVisualizationWindow
        # self.ui.MicrostateVisualizationWindow = MicrostateVisualizationWindow(
        #     self.context,
        #     main_window=self,
        #     tbx=self.comet
        # )
        # Create and initialize OptimizerVisualizationWindow
        self.ui.OptimizerVisualizationWindow = OptimizerVisualizationWindow(
            self.context,
            self.comet
        )

        # Create and initialize BackfittingVisualizationWindow
        self.ui.BackfittingVisualizationWindow = BackfittingVisualizationWindow(
            self.context
        )
        # Create and initialize FeatureVisualizationWindow
        self.ui.FeatureVisualizationWindow = FeatureVisualizationWindow(
            self.context,
            tbx=self.comet
        )
        # Create and initialize CoregistrationWindow
        self.ui.CoregistrationWindow = CoregistrationWindow(
            self.context,
            tbx=self.comet
        )

    def reset_processing_flags(self, processing_flags, value=False):
        """
        Set processing flags.
        """
        for flag in processing_flags:
            setattr(self.comet, flag, value)

    def init_flags(self):
        """
        Initialize flags for tracking processing steps
        """
        # Reset all processing flags to False
        processing_flags = [
            'done_preprocessing', 'done_clustering', 'done_microstate_labeling',
            'done_backfitting', 'done_extracting_features', 'done_source_localization',
            'done_identifying_microstate_sources'
        ]
        self.reset_processing_flags(processing_flags)
        self.ui.foldername_raw_data = ""
        self.ui.foldername_preprocessed_data = ""

    def init_ui_components(self):
        # Add EEG-COMET Logo with enhanced graphics
        icon_path = self.context.get_resource("eeg_comet_logo.png")
        self.pixmap = QPixmap(icon_path)

        # Use high-quality scaling
        pixmap = self.pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(5)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.ui.comet_logo.setGraphicsEffect(shadow)

        # Set the pixmap
        self.ui.comet_logo.setPixmap(pixmap)

        set_widgets_status(self.ui.main_tab, mode='hide')

    def setup_connections(self):
        # Controlling the visibility and state of various UI components based on user interactions
        control_items = [
            self.ui.step2_auto_k_radio,
            self.ui.step2_user_k_radio,
            self.ui.step2_use_percent_radio,
            self.ui.step2_use_peaks_radio,
            self.ui.step2_clustermethod_combobox,
            self.ui.step2_auto_k_method_combobox,
            self.ui.step2_auto_range_kmin_spinbox,
            self.ui.step2_auto_range_kmax_spinbox,
            self.ui.step2_percent_slider,
            self.ui.step2_batch_checkbox,
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
            self.ui.step5_use_fsaverage_radio,
            self.ui.step5_use_individual_radio,
            self.ui.step5_use_tess_radio,
            self.ui.step5_use_avg_radio
        ]
        for item in control_items:
            if isinstance(item, QComboBox):
                item.activated.connect(self.mainwindow_controller)
            elif isinstance(item, (QSpinBox, QSlider)):
                item.valueChanged.connect(self.mainwindow_controller)
            else:
                item.clicked.connect(self.mainwindow_controller)
        # Button connections for performing specific tasks
        click_actions = [
            (self.ui.step0_show_hide_log_window_button, self.comet.LogWindow.show_hide_log_window),
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
            (self.ui.step5_use_individual_radio, self.locate_individual_subjects_dir),
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
            (self.ui.step0_compare_studies_action, self.open_compare_studies_window),
            (self.ui.step0_reopen_log_window, self.comet.LogWindow.show_hide_log_window)
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
            'done_preprocessing', 'done_clustering', 'done_microstate_labeling',
            'done_backfitting', 'done_extracting_features', 'done_source_localization',
            'done_identifying_microstate_sources'
        ]
        self.reset_processing_flags(processing_flags)
        # Set NewStudyWindow modality and show maximized
        self.ui.NewStudyWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.NewStudyWindow.showMaximized()
        # Update the main window
        self.mainwindow_controller()

    def load_study_helper(self):
        # Get the folder containing preprocessed data
        self.comet.save_dir = QFileDialog.getExistingDirectory(
            self, "Please choose the folder where the EEG-COMET study is located."
        )
        # Check if the eeg_comet_config.ini file exists
        config_path = os.path.join(self.comet.save_dir, 'eeg_comet_config.ini')
        # Handle the case where loading the study fails
        if not os.path.exists(config_path):
            QMessageBox.information(self, "Load error",
                                    "The selected folder does not contain a valid config file!",
                                    QMessageBox.Ok)
            return False
        else:
            try:
                # Load the COMET parameters
                self.comet.config = self.comet.load_config(config_path)
                self.comet.load_config_values()  # Process config and update instance variables
                self.comet.reset_directories()  # Update directory paths based on new config
                self.comet.load_eeg_info()
                self.comet.load_maps()
                self.comet.load_clean()

                # Initialize the log window if needed
                if not hasattr(self.comet, 'LogWindow') or self.comet.LogWindow is None:
                    self.comet.initialize_log_window()

                self.comet.LogWindow.show()

                # Update the log window with loaded study information
                if hasattr(self.comet, 'log_text') and self.comet.log_text:
                    self.comet.LogWindow.replace_log(self.comet.log_text)

                self.comet.LogWindow.append_log(f"Study Loaded - ✓ Study Name: {self.comet.study_name}")
                return True
            except Exception as e:
                QMessageBox.information(self, "Load error",
                                        f"Failed to load study parameters: {e}",
                                        QMessageBox.Ok)
                return False

    def load_study(self, from_new_study=False):
        if from_new_study:
            # No need to manually set paths - COMET already manages them
            # Just ensure paths are properly set up
            if hasattr(self.comet, 'reset_directories'):
                self.comet.reset_directories()

            # Initialize log window if needed
            if not hasattr(self.comet, 'LogWindow') or self.comet.LogWindow is None:
                self.comet.initialize_log_window()

            self.comet.LogWindow.show()
            self.comet.LogWindow.append_log(f"Study Created - ✓ Study Name: {self.comet.study_name}")
        else:
            if self.comet.done_preprocessing:
                ret = QMessageBox.question(self, 'MessageBox', f"The {self.comet.study_name} is already loaded,"
                                                               " do you want to load another study?",
                                           QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
                if ret == QMessageBox.Yes:
                    if self.load_study_helper():
                        # Ensure directories are properly set after loading
                        if hasattr(self.comet, 'reset_directories'):
                            self.comet.reset_directories()
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

        hide_after_loading_widgets = [
            self.ui.comet_logo,
            self.ui.step0_new_study_button,
            self.ui.step0_load_study_button,
            self.ui.step0_compare_studies_button,
            self.ui.step0_show_hide_log_window_button
        ]

        after_preprocessing_widgets = [
            self.ui.step2_line1,
            self.ui.step2_line2,
            self.ui.step2_line3,
            self.ui.step2_line4,
            self.ui.step2_line5,
            self.ui.step2_line6,
            self.ui.step2_line7,
            self.ui.step2_similarity_label,
            self.ui.step2_similarity_combobox,
            self.ui.step2_initializer_label,
            self.ui.step2_initialization_method_label,
            self.ui.step2_random_initializer_radio,
            self.ui.step2_kmeans_initializer_radio,
            self.ui.step2_select_times_label,
            self.ui.step2_use_peaks_radio,
            self.ui.step2_kernel_size_label,
            self.ui.step2_kernel_size_input,
            self.ui.step2_use_percent_radio,
            self.ui.step2_percent_label,
            self.ui.step2_percent_input,
            self.ui.step2_percent_slider,
            self.ui.step2_convergence_label,
            self.ui.step2_maxiter_label,
            self.ui.step2_maxiter_input,
            self.ui.step2_stopcondition_label,
            self.ui.step2_stopcondition_input,
            self.ui.step2_numberofrepeats_label,
            self.ui.step2_user_numberofrepeats_input,
            self.ui.step2_number_maps_label,
            self.ui.step2_clustermethod_combo_label,
            self.ui.step2_clustermethod_combobox,
            self.ui.step2_auto_k_radio,
            self.ui.step2_user_k_radio,
            self.ui.step2_batch_checkbox,
            self.ui.step2_numberofmaps_elbow_button,
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

        similarity_widgets = [
            self.ui.step2_similarity_label,
            self.ui.step2_similarity_combobox
        ]

        batch_widgets = [
            self.ui.step2_batch_label,
            self.ui.step2_batch_input
        ]

        peaks2use_widgets = [
            self.ui.step2_kernel_size_label,
            self.step2_kernel_size_input
        ]

        rand2use_widgets = [
            self.ui.step2_percent_label,
            self.ui.step2_percent_input,
            self.ui.step2_percent_slider,
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
            self.ui.step3_filter_segments_method_combobox,
        ]

        smooth_segments_widgets = [
            self.ui.step3_smooth_segments_epsilon_label,
            self.ui.step3_smooth_segments_epsilon_input,
            self.ui.step3_smooth_segments_lambda_label,
            self.ui.step3_smooth_segments_lambda_input
        ]

        identify_short_widgets = [
            self.ui.step3_filter_segments_input,
            self.ui.step3_window_segments_label,
            self.ui.step3_filter_segments_method_label,
            self.ui.step3_filter_segments_param_label
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
            self.ui.step4_sliding_window_raw_input,
            self.ui.step4_sliding_window_epoched_label_0,
            self.ui.step4_sliding_window_epoched_label_1,
            self.ui.step4_sliding_window_epoched_label_2,
            self.ui.step4_sliding_window_epoched_input_pre,
            self.ui.step4_sliding_window_epoched_input_post,
            self.ui.step4_extractfeatures_button,
            self.ui.step4_outputformats_label,
            self.ui.step4_outputformats_combobox,
            self.ui.step4_visualizefeatures_button
            ]

        sliding_feature_extraction_raw_widgets = [
            self.ui.step4_sliding_window_raw_label_0,
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

        microsynt_feature_extraction_widgets = [
            self.ui.step4_word_size_label1,
            self.ui.step4_word_size_label2,
            self.ui.step4_word_size_label3,
            self.ui.step4_word_size_min_input,
            self.ui.step4_word_size_max_input
        ]

        source_localization_widgets = [
            self.ui.step5_line1,
            self.ui.step5_line2,
            self.ui.step5_line3,
            self.ui.step5_line4,
            self.ui.step5_stc_settings_label,
            self.ui.step5_bem_method_label,
            self.ui.step5_bem_mne_radio,
            self.ui.step5_bem_openmeeg_radio,
            self.ui.step5_anatomy_label,
            self.ui.step5_subjects_dir_label,
            self.ui.step5_use_fsaverage_radio,
            self.ui.step5_use_individual_radio,
            self.ui.step5_inverse_method_label,
            self.ui.step5_inverse_method_combobox,
            self.ui.step5_spacing_label,
            self.ui.step5_spacing_combobox,
            self.ui.step5_coreg_button,
            self.ui.step5_estimate_sources_button,
        ]

        source_indivisual_widgets = [
            self.ui.step5_subjects_dir_lineedit
        ]

        source_microstates_widgets = [
            self.ui.step5_source_microstate_settings_label,
            self.ui.step5_source_microstate_method_label,
            self.ui.step5_use_tess_radio,
            self.ui.step5_use_avg_radio,
            self.ui.step5_compute_source_microstate_correlation_button,
        ]

        tess_widgets = [
            self.ui.step5_permutations_label,
            self.ui.step5_permutations_input,
        ]


        font_steps = QFont()
        font_steps.setPointSize(16)

        if not self.comet.done_preprocessing:
            # NOT done_preprocessing
            self.ui.main_tab.setTabEnabled(0, False)  # Clustering tab
            self.ui.main_tab.setTabEnabled(1, False)  # Backfitting tab
            self.ui.main_tab.setTabEnabled(2, False)  # Feature tab
            self.ui.main_tab.setTabEnabled(3, False)  # Source tab
            set_widgets_status(hide_after_loading_widgets, mode='show')
            set_widgets_status(self.ui.main_tab, mode='hide')
            set_widgets_status(after_preprocessing_widgets, mode='disable')
            self.ui.step0_study_name_mainwin_lineedit.setStyleSheet("background-color: none")
        else:
            # done_preprocessing
            self.ui.comet_label.setText("EEG-COMET")
            set_widgets_status(hide_after_loading_widgets, mode='hide')
            set_widgets_status(self.ui.main_tab, mode='show')
            self.ui.main_tab.setTabEnabled(0, True)  # Clustering tab
            set_widgets_status(after_preprocessing_widgets, mode='enable')

            self.ui.step0_study_name_mainwin_lineedit.setText(self.comet.study_name)
            self.ui.step0_study_name_mainwin_lineedit.setStyleSheet("background-color: lightgreen")
            self.comet.clustering_method = self.step2_clustermethod_combobox.currentText()

            if not self.comet.clustering_method == 'Modified K-Means Clustering (Pascual-Marqui et al. 1995)':
                set_widgets_status(similarity_widgets, mode='enable')
                set_widgets_status(similarity_widgets, mode='show')
            else:
                set_widgets_status(similarity_widgets, mode='disable')
                set_widgets_status(similarity_widgets, mode='hide')

            if self.ui.step2_auto_k_radio.isChecked():
                set_widgets_status(user_k_widgets, mode='disable')
                set_widgets_status(auto_k_widgets, mode='enable')
                set_widgets_status(auto_k_widgets, mode='show')

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
                self.ui.step2_auto_target_parameter_label.setText(step2_auto_target_parameter_text)

            if self.ui.step2_user_k_radio.isChecked():
                set_widgets_status(user_k_widgets, mode='enable')
                set_widgets_status(auto_k_widgets, mode='disable')
                set_widgets_status(auto_k_widgets, mode='hide')

            if self.ui.step2_batch_checkbox.isChecked():
                set_widgets_status(batch_widgets, mode='enable')
                if not self.ui.step2_batch_input.text():
                    self.ui.step2_batch_input.setText("1000")
                self.comet.batch_size = int(self.ui.step2_batch_input.text())
            else:
                set_widgets_status(batch_widgets, mode='disable')
                self.comet.batch_size = None

            if self.ui.step2_use_peaks_radio.isChecked():
                set_widgets_status(peaks2use_widgets, mode='enable')
                set_widgets_status(rand2use_widgets, mode='disable')
            elif self.ui.step2_use_percent_radio.isChecked():
                set_widgets_status(rand2use_widgets, mode='enable')
                set_widgets_status(peaks2use_widgets, mode='disable')
                self.ui.step2_percent_input.setText(str(self.ui.step2_percent_slider.value()))

            if not self.comet.done_clustering:
                self.ui.main_tab.setTabEnabled(1, False)  # Backfitting tab
                self.ui.main_tab.setTabEnabled(2, False)  # Feature tab
                self.ui.main_tab.setTabEnabled(3, False)  # Source tab

                # Reset next steps processing flags to False
                processing_flags = [
                    'done_microstate_labeling', 'done_backfitting', 'done_extracting_features',
                    'done_source_localization', 'done_identifying_microstate_sources'
                ]
                self.reset_processing_flags(processing_flags)
                self.ui.step2_clustering_button.setStyleSheet("background-color: none")
                self.comet.best_maps, self.comet.micro_labels = None, []
                # set ALL disabled
                set_widgets_status(self.ui.step3_label_maps_button, mode='disable')
                set_widgets_status(after_clustering_widgets, mode='disable')
                set_widgets_status(filter_segments_widgets, mode='disable')
            else:
                # done_preprocessing & done_clustering
                set_widgets_status(self.ui.step3_label_maps_button, mode='enable')

                if not self.comet.done_microstate_labeling:
                    self.ui.step3_label_maps_button.setStyleSheet("background-color: none")
                    # Reset next steps processing flags to False
                    processing_flags = [
                        'done_backfitting', 'done_extracting_features',
                        'done_source_localization', 'done_identifying_microstate_sources'
                    ]
                    self.reset_processing_flags(processing_flags)

                else:
                    # done_preprocessing & done_clustering & done_microstate_labeling
                    self.ui.step3_label_maps_button.setStyleSheet("background-color: lightgreen")
                    self.ui.main_tab.setTabEnabled(1, True)  # Backfitting tab
                    set_widgets_status(after_clustering_widgets, mode='enable')

                    outputformat = self.ui.step4_outputformats_combobox.currentText()
                    opening_parenthesis = outputformat.find("(")
                    closing_parenthesis = outputformat.find(")")
                    if opening_parenthesis != -1 and closing_parenthesis != -1:
                        self.comet.export_format = outputformat[opening_parenthesis + 1: closing_parenthesis]

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

            if not self.comet.done_backfitting:
                self.ui.main_tab.setTabEnabled(2, False)  # Feature tab
                self.ui.main_tab.setTabEnabled(3, False)  # Source tab
                set_widgets_status((feature_extraction_widgets +
                                    source_localization_widgets), mode='disable')
                self.reset_processing_flags('done_extracting_features')
                self.ui.step3_backfit_button.setStyleSheet("background-color: none")
                self.ui.step3_backfit_visualization_button.setDisabled(True)

            else:
                # done_preprocessing & done_clustering & done_microstate_labeling & done_backfitting
                self.ui.step3_backfit_button.setStyleSheet("background-color: lightgreen")
                self.ui.main_tab.setTabEnabled(2, True)  # Feature tab
                self.ui.main_tab.setTabEnabled(3, True)  # Source tab
                self.ui.step3_backfit_visualization_button.setEnabled(True)
                set_widgets_status(feature_extraction_widgets, mode='enable')
                set_widgets_status(source_localization_widgets, mode='enable')

                if self.comet.datatype == 'epoched':
                    self.ui.step4_sliding_features_checkbox.setText(
                        "Extract Features Before and After TMS per Subject")
                    set_widgets_status(feature_extraction_epoched_widgets, mode='enable')
                    self.ui.step4_feature_rof_checkbox.setChecked(True)
                    self.ui.step4_feature_rtf_checkbox.setChecked(True)
                    set_widgets_status(sliding_feature_extraction_raw_widgets, mode='disable')
                    if self.ui.step4_sliding_features_checkbox.isChecked():
                        set_widgets_status(sliding_feature_extraction_epoched_widgets, mode='enable')
                    else:
                        set_widgets_status(sliding_feature_extraction_epoched_widgets, mode='disable')
                else:
                    set_widgets_status(feature_extraction_epoched_widgets, mode='disable')
                    self.ui.step4_feature_rof_checkbox.setChecked(False)
                    self.ui.step4_feature_rtf_checkbox.setChecked(False)
                    set_widgets_status(sliding_feature_extraction_epoched_widgets, mode='disable')
                    if self.ui.step4_sliding_features_checkbox.isChecked():
                        set_widgets_status(sliding_feature_extraction_raw_widgets, mode='enable')
                    else:
                        set_widgets_status(sliding_feature_extraction_raw_widgets, mode='disable')

                if self.ui.step4_feature_er_checkbox.isChecked():
                    set_widgets_status(microsynt_feature_extraction_widgets, mode='enable')
                else:
                    set_widgets_status(microsynt_feature_extraction_widgets, mode='disable')

                self.ui.step4_extractfeatures_button.setDisabled(
                    not any(checkbox.isChecked() for checkbox in feature_checkboxes1) or
                    not any(checkbox.isChecked() for checkbox in feature_checkboxes2))

                if self.comet.done_extracting_features:
                    self.ui.step4_extractfeatures_button.setStyleSheet("background-color: lightgreen")
                    self.ui.step4_visualizefeatures_button.setEnabled(True)
                else:
                    self.ui.step4_extractfeatures_button.setStyleSheet("background-color: none")
                    self.ui.step4_visualizefeatures_button.setDisabled(True)

                if self.ui.step5_use_individual_radio.isChecked():
                    self.comet.use_anatomy = "individual"
                    set_widgets_status(source_indivisual_widgets, mode='enable')
                else:  # self.ui.step5_use_fsaverage_radio.isChecked():
                    self.comet.use_anatomy = "fsaverage"
                    set_widgets_status(source_indivisual_widgets, mode='disable')

                if self.ui.step5_use_tess_radio.isChecked():
                    set_widgets_status(tess_widgets, mode='enable')
                    self.comet.source_localization_method = 'tess'
                else:  # self.ui.step5_use_avg_radio.isChecked()
                    set_widgets_status(tess_widgets, mode='disable')
                    self.comet.source_localization_method = 'avg'

                if not self.comet.done_source_localization:
                    self.ui.step5_estimate_sources_button.setStyleSheet("background-color: none")
                    set_widgets_status(source_microstates_widgets, mode='disable')
                else:
                    self.ui.step5_estimate_sources_button.setStyleSheet("background-color: lightgreen")
                    set_widgets_status(source_microstates_widgets, mode='enable')

                    if self.comet.done_identifying_microstate_sources:
                        self.ui.step5_compute_source_microstate_correlation_button.setStyleSheet(
                            "background-color: lightgreen")
                        self.ui.step5_visualize_sources_button.setEnabled(True)
                    else:
                        self.ui.step5_compute_source_microstate_correlation_button.setStyleSheet(
                            "background-color: none")
                        self.ui.step5_visualize_sources_button.setDisabled(True)

    def visualize_elbow(self):
        """
        Initiates the visualization of optimization plots for determining the optimal number of clusters.
        Works with the new ClustererOptimizer implementation.
        """
        # Update COMET parameters based on current UI settings
        self._update_comet_clustering_parameters()

        # Check if we already have optimization results from previous automatic clustering
        if (hasattr(self.comet, 'optimization_results') and
                self.comet.optimization_results and
                self.comet.choose_number_of_maps == "auto"):

            # Pass the existing results to the visualization window
            self.ui.OptimizerVisualizationWindow.results_cache = {}

            # Convert COMET's optimization results to the format expected by the visualization window
            for method_code, result in self.comet.optimization_results.items():
                if method_code != 'majority_vote':  # Skip majority vote for individual visualizations
                    cache_key = f"{method_code}_None"
                    self.ui.OptimizerVisualizationWindow.results_cache[cache_key] = {
                        'method': method_code,
                        'result': result,
                        'optimal_k': result.optimal_k,
                        'k_values': result.k_values,
                        'scores': result.scores
                    }
        else:
            # Clear any previous results
            self.ui.OptimizerVisualizationWindow.results_cache = {}

        # Update parameters in the visualization window
        self.ui.OptimizerVisualizationWindow._load_comet_parameters()

        # Update the min/max range spinboxes based on current COMET settings
        if hasattr(self.comet, 'kmin'):
            self.ui.OptimizerVisualizationWindow.ui.optimizer_min_input.setText(str(self.comet.kmin))
        if hasattr(self.comet, 'kmax'):
            self.ui.OptimizerVisualizationWindow.ui.optimizer_max_input.setText(str(self.comet.kmax))

        # Reset the optimizer to ensure fresh computation if parameters changed
        self.ui.OptimizerVisualizationWindow.optimizer = None

        # Set modality and show the OptimizerVisualizationWindow
        self.ui.OptimizerVisualizationWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.OptimizerVisualizationWindow.showMaximized()

    def _update_comet_clustering_parameters(self):
        """
        Update COMET instance with current UI clustering parameters.
        This ensures the optimizer uses the correct settings.
        """
        # Update smoothing parameters
        if self.ui.step2_kernel_size_input.text():
            self.comet.smoothing_gfp = True
            self.comet.smoothing_distance = int(self.ui.step2_kernel_size_input.text())
            self.comet.min_distance_size = int(self.comet.smoothing_distance / (1000 / self.comet.sample_rate))
        else:
            self.comet.smoothing_gfp = False
            self.comet.smoothing_distance = 0
            self.comet.min_distance_size = None

        # Update use_percentages based on radio button selection
        if self.ui.step2_use_percent_radio.isChecked():
            self.comet.use_percentages = int(self.ui.step2_percent_slider.value())
        else:
            self.comet.use_percentages = None

        # Update clustering parameters
        self.comet.clustering_tolerance = float(self.ui.step2_stopcondition_input.text())
        self.comet.max_iterations = int(self.ui.step2_maxiter_input.text())
        self.comet.number_of_repeats = int(self.ui.step2_user_numberofrepeats_input.text())

        # Update k range for auto mode
        if self.ui.step2_auto_k_radio.isChecked():
            self.comet.kmin = int(self.ui.step2_auto_range_kmin_spinbox.value())
            self.comet.kmax = int(self.ui.step2_auto_range_kmax_spinbox.value())

    def do_clustering(self):
        """
        Initiates the clustering process based on user-specified parameters and performs clustering on EEG data.
        Updated to work with the new ClustererOptimizer.
        """
        # Check if the analysis is already done.
        if self.comet.done_clustering:
            ret = QMessageBox.question(self, 'MessageBox', "Data has been clustered once,"
                                                           " do you want to redo the analysis?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_clustering_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_clustering_from_scratch = True
                # Reset all processing flags to False
                processing_flags = [
                    'done_clustering', 'done_microstate_labeling',
                    'done_backfitting', 'done_extracting_features', 'done_source_localization',
                    'done_identifying_microstate_sources'
                ]
                self.reset_processing_flags(processing_flags)
                # Clear previous optimization results
                if hasattr(self.comet, 'optimization_results'):
                    self.comet.optimization_results = None
        else:
            self.do_clustering_from_scratch = True

        if self.do_clustering_from_scratch:
            # Update all clustering parameters from UI
            self._update_comet_clustering_parameters()

            # Process user-specified clustering parameters
            if self.ui.step2_auto_k_radio.isChecked():
                self.comet.choose_number_of_maps = "auto"
                self.comet.number_of_maps = 'auto'

                # Get the optimization method settings
                auto_k_method = self.ui.step2_auto_k_method_combobox.currentText()
                method_map = {
                    'Gap Statistic': 'gs',
                    'Cross Validation': 'cv',
                    'Elbow - Global Explained Variance': 'gev',
                    'Elbow - Residual Variance': 'res',
                    'Silhouette Method': 'sil',
                    'Calinski-Harabasz Method': 'ch',
                    'Davies-Bouldin Method': 'db'
                }
                self.comet.stopping_mode = method_map.get(auto_k_method, 'gev')
                self.comet.stopping_parameter = float(self.ui.step2_stopping_threshold_input.text())

            elif self.ui.step2_user_k_radio.isChecked():
                self.comet.choose_number_of_maps = "user"
                self.comet.stopping_mode = ''
                self.comet.stopping_parameter = ''
                self.comet.kmin = ''
                self.comet.kmax = ''
                self.comet.number_of_maps = int(self.ui.step2_user_k_input.text())

            # Set initializer
            if self.ui.step2_random_initializer_radio.isChecked():
                self.comet.initializer = "Random"
            elif self.ui.step2_kmeans_initializer_radio.isChecked():
                self.comet.initializer = "K-Means++"

            # Set clustering method and similarity metric
            self.comet.clustering_method = self.ui.step2_clustermethod_combobox.currentText()
            self.comet.similarity_metric = self.ui.step2_similarity_combobox.currentText()

            # Set batch size if enabled
            if self.ui.step2_batch_checkbox.isChecked():
                self.comet.batch_size = int(
                    self.ui.step2_batch_input.text()) if self.ui.step2_batch_input.text() else 1000
            else:
                self.comet.batch_size = None

            # Set paths
            self.comet.microstate_maps_path = os.path.join(self.comet.save_dir, 'microstate_maps.csv')

            # Perform clustering
            self.comet.run_clustering()
            self.comet.done_clustering = True
            self.mainwindow_controller()

    def do_backfitting(self):
        """
        Perform microstate backfitting.
        """
        # Check if backfitting has already been done
        if self.comet.done_backfitting:
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
                'done_extracting_features', 'done_source_localization', 'done_identifying_microstate_sources'
            ]
            self.reset_processing_flags(processing_flags)
            # Set backfitting parameters based on user input
            if self.ui.step3_backfit_all_radio.isChecked():
                self.comet.backfit_to = 'all'
                if self.ui.step3_identify_short_checkbox.isChecked():
                    self.comet.identify_short_window = True
                else:
                    self.comet.identify_short_window = False
            elif self.ui.step3_backfit_peaks_radio.isChecked():
                self.comet.backfit_to = 'peaks'
                self.comet.identify_short_window = False
            self.comet.epsilon = ''
            self.comet.b = ''
            self.comet.lamb = ''
            if self.ui.step3_filter_segments_checkbox.isChecked():
                self.comet.filter_segments = True
                self.comet.remove_segments_less_than = int(int(self.ui.step3_filter_segments_input.text()) /
                                                               (1000/self.comet.sample_rate))
                filter_segments_method = self.ui.step3_filter_segments_method_combobox.currentText()
                if filter_segments_method == 'Replace short segments: nearby dominant microstate':
                    self.comet.filter_segments_option = 'replace_high'
                elif filter_segments_method == 'Replace short segments: half and half':
                    self.comet.filter_segments_option = 'replace_half'
                elif filter_segments_method == 'Remove short segments':
                    self.comet.filter_segments_option = 'remove'
                elif filter_segments_method == 'Smooth segments':
                    self.comet.filter_segments_option = 'smooth'
                    self.comet.epsilon = float(self.ui.step3_smooth_segments_epsilon_input.text())
                    self.comet.b = self.comet.remove_segments_less_than
                    self.comet.lamb = int(self.ui.step3_smooth_segments_lambda_input.text())
            else:
                self.comet.filter_segments = False
                self.comet.filter_segments_option = ''
                self.comet.remove_segments_less_than = []
            # Log and perform backfitting
            self.comet.run_backfitting()
            self.ui.main_tab.setCurrentIndex(2)
            self.mainwindow_controller()

    def visualize_microstates(self):
        """
        Visualize microstate maps for labeling.
        """
        self.ui.MicrostateVisualizationWindow = MicrostateVisualizationWindow(
            self.context,
            main_window=self,
            tbx=self.comet
        )
        self.ui.MicrostateVisualizationWindow.plot_maps()
        self.ui.MicrostateVisualizationWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.MicrostateVisualizationWindow.showMaximized()

    def visualize_microstate_segmentation(self):
        """
        Open BackfittingVisualizationWindow to visualize microstate segmentation.
        """
        # Set relevant paths and parameters for visualization
        self.BackfittingVisualizationWindow.preprocessed_data_path = self.comet.preprocessed_data_path
        self.BackfittingVisualizationWindow.extension = self.comet.extension
        self.BackfittingVisualizationWindow.datatype = self.comet.datatype
        self.BackfittingVisualizationWindow.eeg_filenames_combobox.addItems([i for i in self.comet.list_eegs])
        self.BackfittingVisualizationWindow.segmentation_path = self.comet.segmentation_path
        self.BackfittingVisualizationWindow.export_format = self.comet.export_format
        # Display the Backfitting Visualization Dialog
        self.BackfittingVisualizationWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.BackfittingVisualizationWindow.showMaximized()

    def extract_features(self):
        """
        Extract features from the backfitted data.
        """
        # Check if features have been extracted before
        if self.comet.done_extracting_features:
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
            self.comet.done_extracting_features = False
            self.mainwindow_controller()

            # Define features to extract based on user selection
            self.comet.feature_list = []
            if self.ui.step4_feature_occ_checkbox.isChecked():
                self.comet.feature_list.append("OCC")
            if self.ui.step4_feature_dur_checkbox.isChecked():
                self.comet.feature_list.append("DUR")
            if self.ui.step4_feature_cov_checkbox.isChecked():
                self.comet.feature_list.append("COV")
            if self.ui.step4_feature_gev_checkbox.isChecked():
                self.comet.feature_list.append("GEV")
            if self.ui.step4_feature_tp_checkbox.isChecked():
                self.comet.feature_list.append("TP")
            if self.ui.step4_feature_se_checkbox.isChecked():
                self.comet.feature_list.append("SE")
            if self.ui.step4_feature_lzc_checkbox.isChecked():
                self.comet.feature_list.append("LZC")
            if self.ui.step4_feature_er_checkbox.isChecked():
                self.comet.feature_list.append("ER")
                self.comet.word_size = int(self.ui.step4_word_size_min_input.text())
            else:
                self.comet.word_size = 2

            # Define feature extraction modes
            self.comet.feature_mode = []
            if self.ui.step4_averaged_features_checkbox.isChecked():
                self.comet.feature_mode.append("averaged")
            if self.ui.step4_sliding_features_checkbox.isChecked():
                self.comet.feature_mode.append("sliding")
            if self.ui.step4_synthetic_checkbox.isChecked():
                self.comet.feature_types = ['real', 'surrogate', 'random']
            else:
                self.comet.feature_types = ['real']

            if self.comet.datatype == 'epoched':
                # Set window size for pre post features
                self.comet.pre_window_size = int(self.ui.step4_sliding_window_epoched_input_pre.text())
                self.comet.post_window_size = int(self.ui.step4_sliding_window_epoched_input_post.text())
                if self.ui.step4_feature_rof_checkbox.isChecked():
                    self.comet.feature_list.append("ROF")
                if self.ui.step4_feature_rtf_checkbox.isChecked():
                    self.comet.feature_list.append("RTF")
            else:
                # Set sliding window size for dynamic features
                self.comet.sliding_window_size = int(self.ui.step4_sliding_window_raw_input.text())
            # Log and perform feature extraction
            self.comet.run_feature_extraction()
            # Update flags and save the state
            self.comet.done_extracting_features = True
            # Update the main window
            self.mainwindow_controller()

    def visualize_microstate_features(self):
        """
        Open FeatureVisualizationWindow to visualize microstate features.
        """
        # Set relevant paths and parameters for visualization
        self.ui.FeatureVisualizationWindow.extracted_features_path = self.comet.extracted_features_path
        self.ui.FeatureVisualizationWindow.export_format = self.comet.export_format
        self.ui.FeatureVisualizationWindow.feature_mode = self.comet.feature_mode
        self.ui.FeatureVisualizationWindow.feature_combo.clear()
        self.ui.FeatureVisualizationWindow.feature_combo.addItems([i for i in self.comet.feature_list])
        self.ui.FeatureVisualizationWindow.list_eegs = self.comet.list_eegs
        self.ui.FeatureVisualizationWindow.reset_groups()
        # Display the Feature Visualization Dialog
        self.ui.FeatureVisualizationWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.FeatureVisualizationWindow.showMaximized()

    def coregister(self):
        """
        Open CoregistrationWindow to align EEG Sensors to head source space.
        """
        self.ui.CoregistrationWindow = CoregistrationWindow(self.context, tbx=self.comet)
        self.ui.CoregistrationWindow.showMaximized()

    def locate_individual_subjects_dir(self):
        self.comet.individual_subjects_dir = QFileDialog.getExistingDirectory(
            self,
            "Locate Folder with Individual Anatomical Reconstructions"
        )
        if self.comet.individual_subjects_dir:
            self.ui.step5_subjects_dir_lineedit.setText(self.comet.individual_subjects_dir)
        else:
            self.ui.step5_use_fsaverage_radio.setChecked(True)

    def source_localize_microstates(self):
        """
        Perform source localization of data.
        """
        # Check if source localization is already done
        if self.comet.done_source_localization:
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
                'done_source_localization', 'done_identifying_microstate_sources'
            ]
            self.reset_processing_flags(processing_flags)
            # Update the main window
            self.mainwindow_controller()

            # Get the inverse method and spacing options
            if self.ui.step5_bem_openmeeg_radio.isChecked():
                self.comet.bem_solver = 'openmeeg'
            else:  # self.ui.step5_bem_mne_radio.isChecked():
                self.comet.bem_solver = 'mne'
            inverse_method = self.ui.step5_inverse_method_combobox.currentText()
            self.comet.inverse_method = inverse_method[inverse_method.find("(") + 1:inverse_method.find(")")]
            spacing = self.ui.step5_spacing_combobox.currentText()
            self.comet.spacing = spacing[spacing.find("(") + 1:spacing.find(")")].lower()
            self.comet.nperm = int(self.ui.step5_permutations_input.text())
            # Perform the source localization and save the state
            self.comet.run_source_localization()
            # Update the main window
            self.mainwindow_controller()

    def source_microstates_correlation(self):
        """
        Identify sources correlated to each microstate.
        """
        # Check if microstate source localization is already done
        if self.comet.done_identifying_microstate_sources:
            ret = QMessageBox.question(self, 'MessageBox',
                                       "Source-microstate correlations have already been calculated,"
                                       " do you want to run this step again?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            self.do_identifying_microstate_sources_from_scratch = False
            if ret == QMessageBox.Yes:
                self.do_identifying_microstate_sources_from_scratch = True
        else:
            self.do_identifying_microstate_sources_from_scratch = True
        # If microstate source localization need to be calculated from scratch
        if self.do_identifying_microstate_sources_from_scratch:
            # TODO: Add missing options here
            self.comet.nperm = int(self.ui.step5_permutations_input.text())
            # Perform the microstate source localization and save the state
            self.comet.run_identifying_microstate_sources()
            self.comet.done_identifying_microstate_sources = True
            # Update the main window
            self.mainwindow_controller()

    def visualize_source_localized_microstates(self):
        """
        Open SourceVisualizationWindow to visualize microstates localized sources.
        """
        # TODO
        # Set relevant paths and parameters for visualization
        self.ui.SourceVisualizationWindow = SourceVisualizationWindow(self.context, comet_tbx=self.comet)
        #self.SourceVisualizationWindow.tess_path = os.path.join(self.comet.localized_sources_path, "tess_sources")
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
            self.comet.LogWindow.close()
            # Close the main window
            self.close()
