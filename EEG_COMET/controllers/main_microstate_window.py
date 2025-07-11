import os.path
import webbrowser
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QComboBox, QSpinBox, QSlider,
    QMessageBox, QGraphicsDropShadowEffect, QWidget,
    QVBoxLayout, QGridLayout
)
from PyQt5.QtGui import QPixmap, QColor, QKeySequence, QFont
from PyQt5.QtCore import Qt, pyqtSignal

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


class WidgetMode(Enum):
    """Enum for widget status modes"""
    ENABLE = 'enable'
    DISABLE = 'disable'
    SHOW = 'show'
    HIDE = 'hide'


@dataclass
class ProcessingFlags:
    """Data class to manage processing flags"""
    done_preprocessing: bool = False
    done_clustering: bool = False
    done_microstate_labeling: bool = False
    done_backfitting: bool = False
    done_extracting_features: bool = False
    done_source_localization: bool = False
    done_identifying_microstate_sources: bool = False

    def reset_all(self):
        """Reset all flags to False"""
        for field in self.__dataclass_fields__:
            setattr(self, field, False)

    def reset_from(self, flag_name: str):
        """Reset flags from a specific step onwards"""
        flag_order = list(self.__dataclass_fields__.keys())
        if flag_name in flag_order:
            start_index = flag_order.index(flag_name)
            for flag in flag_order[start_index:]:
                setattr(self, flag, False)


class WidgetGroups:
    """Centralized widget group management"""

    def __init__(self, ui):
        self.ui = ui
        self._initialize_groups()
        self._cache_original_properties()

    def _initialize_groups(self):
        """Initialize all widget groups"""
        self.groups = {
            'hide_after_loading': [
                self.ui.comet_label,
                self.ui.comet_logo,
                self.ui.step0_new_study_button,
                self.ui.step0_load_study_button,
                self.ui.step0_compare_studies_button
            ],
            'after_preprocessing': self._get_preprocessing_widgets(),
            'user_k': [
                self.ui.step2_user_k_input,
                self.ui.step2_numberofmaps_elbow_button
            ],
            'auto_k': self._get_auto_k_widgets(),
            'similarity': [
                self.ui.step2_similarity_label,
                self.ui.step2_similarity_combobox
            ],
            'batch': [
                self.ui.step2_batch_label,
                self.ui.step2_batch_input
            ],
            'peaks_use': [
                self.ui.step2_kernel_size_label,
                self.ui.step2_kernel_size_input
            ],
            'rand_use': [
                self.ui.step2_percent_label,
                self.ui.step2_percent_input,
                self.ui.step2_percent_slider
            ],
            'convergence': self._get_convergence_widgets(),
            'after_clustering': self._get_clustering_widgets(),
            'filter_segments': [
                self.ui.step3_filter_segments_method_label,
                self.ui.step3_identify_short_checkbox,
                self.ui.step3_filter_segments_method_combobox
            ],
            'smooth_segments': self._get_smooth_segments_widgets(),
            'identify_short': self._get_identify_short_widgets(),
            'feature_extraction': self._get_feature_extraction_widgets(),
            'feature_checkboxes': self._get_feature_checkboxes(),
            'feature_modes': [
                self.ui.step4_averaged_features_checkbox,
                self.ui.step4_sliding_features_checkbox,
                self.ui.step4_synthetic_checkbox
            ],
            'source_localization': self._get_source_localization_widgets(),
            'source_individual': [self.ui.step5_subjects_dir_lineedit],
            'source_microstates': self._get_source_microstate_widgets(),
            'tess': [
                self.ui.step5_permutations_label,
                self.ui.step5_permutations_input
            ]
        }

    def _cache_original_properties(self):
        """Cache original widget properties to prevent formatting issues"""
        self.original_properties = {}
        for group_widgets in self.groups.values():
            for widget in group_widgets:
                if widget and hasattr(widget, 'font'):
                    self.original_properties[widget] = {
                        'font': widget.font(),
                        'minimumHeight': widget.minimumHeight() if hasattr(widget, 'minimumHeight') else None,
                        'maximumHeight': widget.maximumHeight() if hasattr(widget, 'maximumHeight') else None
                    }

    def restore_properties(self, widgets: List[QWidget]):
        """Restore original properties for widgets"""
        for widget in widgets:
            if widget in self.original_properties:
                props = self.original_properties[widget]
                if props['font']:
                    widget.setFont(props['font'])
                if props['minimumHeight'] is not None:
                    widget.setMinimumHeight(props['minimumHeight'])
                if props['maximumHeight'] is not None:
                    widget.setMaximumHeight(props['maximumHeight'])

    def set_group_status(self, group_name: str, mode: WidgetMode):
        """Set status for a widget group"""
        if group_name in self.groups:
            widgets = self.groups[group_name]
            set_widgets_status(widgets, mode=mode.value)
            # Restore properties after status change
            if mode in [WidgetMode.ENABLE, WidgetMode.SHOW]:
                self.restore_properties(widgets)

    # Widget group initialization methods
    def _get_preprocessing_widgets(self):
        """Get preprocessing-related widgets"""
        return [
            self.ui.step2_line1, self.ui.step2_line2, self.ui.step2_line3,
            self.ui.step2_line4, self.ui.step2_line5, self.ui.step2_line6,
            self.ui.step2_line7, self.ui.step2_similarity_label,
            self.ui.step2_similarity_combobox, self.ui.step2_initializer_label,
            self.ui.step2_initialization_method_label, self.ui.step2_random_initializer_radio,
            self.ui.step2_kmeans_initializer_radio, self.ui.step2_select_times_label,
            self.ui.step2_use_peaks_radio, self.ui.step2_kernel_size_label,
            self.ui.step2_kernel_size_input, self.ui.step2_use_percent_radio,
            self.ui.step2_percent_label, self.ui.step2_percent_input,
            self.ui.step2_percent_slider, self.ui.step2_number_maps_label,
            self.ui.step2_clustermethod_combo_label, self.ui.step2_clustermethod_combobox,
            self.ui.step2_auto_k_radio, self.ui.step2_user_k_radio,
            self.ui.step2_batch_checkbox, self.ui.step2_numberofmaps_elbow_button,
            self.ui.step2_clustering_button
        ]

    def _get_auto_k_widgets(self):
        """Get auto-k related widgets"""
        return [
            self.ui.step2_auto_target_label, self.ui.step2_auto_target_parameter_label,
            self.ui.step2_stopping_threshold_input, self.ui.step2_auto_range_kmin_spinbox,
            self.ui.step2_auto_range_kmax_spinbox, self.ui.step2_auto_k_method_combobox,
            self.ui.step2_auto_range_label
        ]

    def _get_convergence_widgets(self):
        """Get convergence-related widgets"""
        return [
            self.ui.step2_convergence_label, self.ui.step2_maxiter_label,
            self.ui.step2_maxiter_input, self.ui.step2_stopcondition_label,
            self.ui.step2_stopcondition_input, self.ui.step2_numberofrepeats_label,
            self.ui.step2_numberofrepeats_input
        ]

    def _get_clustering_widgets(self):
        """Get clustering-related widgets"""
        return [
            self.ui.step3_line1, self.ui.step3_line2, self.ui.step3_line3,
            self.ui.step3_line4, self.ui.step3_backfit_label,
            self.ui.step3_backfit_all_radio, self.ui.step3_backfit_peaks_radio,
            self.ui.step3_filter_segments_checkbox, self.ui.step3_identify_short_checkbox,
            self.ui.step3_backfit_button
        ]

    def _get_smooth_segments_widgets(self):
        """Get smooth segments widgets"""
        return [
            self.ui.step3_smooth_segments_epsilon_label,
            self.ui.step3_smooth_segments_epsilon_input,
            self.ui.step3_smooth_segments_lambda_label,
            self.ui.step3_smooth_segments_lambda_input
        ]

    def _get_identify_short_widgets(self):
        """Get identify short widgets"""
        return [
            self.ui.step3_filter_segments_input,
            self.ui.step3_window_segments_label,
            self.ui.step3_filter_segments_param_label
        ]

    def _get_feature_extraction_widgets(self):
        """Get feature extraction widgets"""
        base_widgets = [
            self.ui.step4_line1, self.ui.step4_line2, self.ui.step4_line3,
            self.ui.step4_line4, self.ui.step4_features_extract_label,
            self.ui.step4_features_type_label, self.ui.step4_outputformats_label,
            self.ui.step4_outputformats_combobox, self.ui.step4_extractfeatures_button
        ]

        # Add all feature checkboxes and mode checkboxes
        feature_widgets = self._get_feature_checkboxes() + [
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
            self.ui.step4_features_epoched_label,
            self.ui.step4_word_size_label1,
            self.ui.step4_word_size_label2,
            self.ui.step4_word_size_label3,
            self.ui.step4_word_size_min_input,
            self.ui.step4_word_size_max_input
        ]

        return base_widgets + feature_widgets

    def _get_feature_checkboxes(self):
        """Get feature checkboxes"""
        return [
            self.ui.step4_feature_occ_checkbox, self.ui.step4_feature_dur_checkbox,
            self.ui.step4_feature_cov_checkbox, self.ui.step4_feature_gev_checkbox,
            self.ui.step4_feature_tp_checkbox, self.ui.step4_feature_se_checkbox,
            self.ui.step4_feature_lzc_checkbox, self.ui.step4_feature_er_checkbox,
            self.ui.step4_feature_rof_checkbox, self.ui.step4_feature_rtf_checkbox
        ]

    def _get_source_localization_widgets(self):
        """Get source localization widgets"""
        return [
            self.ui.step5_line1, self.ui.step5_line2, self.ui.step5_line3,
            self.ui.step5_line4, self.ui.step5_stc_settings_label,
            self.ui.step5_bem_method_label, self.ui.step5_bem_mne_radio,
            self.ui.step5_bem_openmeeg_radio, self.ui.step5_anatomy_label,
            self.ui.step5_subjects_dir_label, self.ui.step5_use_fsaverage_radio,
            self.ui.step5_use_individual_radio, self.ui.step5_inverse_method_label,
            self.ui.step5_inverse_method_combobox, self.ui.step5_spacing_label,
            self.ui.step5_spacing_combobox, self.ui.step5_coreg_button,
            self.ui.step5_estimate_sources_button
        ]

    def _get_source_microstate_widgets(self):
        """Get source microstate widgets"""
        return [
            self.ui.step5_source_microstate_settings_label,
            self.ui.step5_use_tess_radio,
            self.ui.step5_use_avg_radio,
            self.ui.step5_compute_source_microstate_correlation_button
        ]


class MainMicrostateWindow(QMainWindow):
    """Optimized main window for EEG-COMET microstate analysis"""

    # Signals for better event handling
    processing_state_changed = pyqtSignal(str)

    def __init__(self, context, parent=None):
        super().__init__(parent)

        # Initialize core components
        self.context = context
        self.comet = COMET()
        self.comet.initialize_log_window()
        self.comet.LogWindow.show()

        # Load UI
        self.ui = uic.loadUi(context.get_resource("MainMicrostateWindow.ui"), self)
        self.ui.setWindowTitle("EEG-COMET")

        # Theme stylesheets
        # Dark theme stylesheet
        self.dark_style = (
            "QWidget {"
            " background-color: #2b2b2b;"
            " color: #f0f0f0;"
            " }"
            " QPushButton { background-color: #3498db; color: #ffffff; border-radius: 6px; padding: 6px 12px; }"
            " QPushButton:hover { background-color: #2980b9; }"
            " QPushButton:disabled { background-color: #95a5a6; color: #ecf0f1; }"
            " QLineEdit, QComboBox { background-color: #ecf0f1; color: #2c3e50; border: 1px solid #7f8c8d; border-radius: 4px; padding: 4px; }"
            " QLineEdit:read-only { background-color: #bdc3c7; }"
            " QCheckBox, QRadioButton { color: #f0f0f0; }"
            " QTabWidget::pane { border: 1px solid #34495e; }"
            " QTabBar::tab { background: #34495e; color: #ecf0f1; padding: 6px 10px; }"
            " QTabBar::tab:selected { background: #2c3e50; font-weight: bold; }"
            " QListWidget { background-color: #1e272e; color: #ecf0f1; border: 1px solid #7f8c8d; border-radius: 4px; }"
            " QSplitter::handle { background: #34495e; }"
            " QMenuBar, QMenu { font-family: 'Calibri'; font-size: 12pt; background-color: #2b2b2b; color: #f0f0f1; }"
            " QMenuBar::item:selected, QMenu::item:selected { background-color: #2980b9; }"
        )

        # Light theme stylesheet (only enforce menu fonts; otherwise default Qt look)
        self.light_style = (
            "QMenuBar, QMenu { font-family: 'Calibri'; font-size: 12pt; }"
        )

        # Initialize components
        self._init_processing_flags()
        self._init_dialogs()
        self._init_ui_components()
        self._init_widget_groups()
        self._setup_connections()

        # Initialize UI state
        self._update_ui_state()

        # Base font parameters for scaling
        self._base_font_pt = 14
        self._base_width = 1200  # reference width

        # Set default application font
        QApplication.instance().setFont(QFont("Calibri", self._base_font_pt))

        # Apply initial theme (light)
        QApplication.instance().setStyleSheet(self.light_style)

        # Initial font scaling
        self._update_font_sizes()

    def toggle_theme(self, checked: bool):
        """Toggle application-wide theme."""
        if checked:
            QApplication.instance().setStyleSheet(self.dark_style)
        else:
            QApplication.instance().setStyleSheet(self.light_style)

        # Re-apply current font after stylesheet change
        self._update_font_sizes()

    # ---------------- Font Scaling ----------------
    def _update_font_sizes(self):
        """Scale global application font based on window width."""
        scale = max(0.8, min(2.0, self.width() / self._base_width))  # cap scaling
        new_size = int(self._base_font_pt * scale)
        QApplication.instance().setFont(QFont("Calibri", new_size))

    def resizeEvent(self, event):
        """Override resizeEvent to adjust fonts dynamically."""
        super().resizeEvent(event)
        self._update_font_sizes()

    def _init_processing_flags(self):
        """Initialize processing flags"""
        self.processing_flags = ProcessingFlags()

        # Sync with COMET instance
        for flag in self.processing_flags.__dataclass_fields__:
            setattr(self.comet, flag, False)

    def _init_dialogs(self):
        """Initialize all dialog windows"""
        self.dialogs = {
            'new_study': NewStudyWindow(self.context, main_window=self, comet_tbx=self.comet),
            'compare_studies': CompareStudiesWindow(self.context),
            'optimizer': OptimizerVisualizationWindow(self.context, self.comet),
            'backfitting': BackfittingVisualizationWindow(self.context),
            'features': FeatureVisualizationWindow(self.context, tbx=self.comet),
            'coregistration': CoregistrationWindow(self.context, tbx=self.comet)
        }

        # Assign to UI for backward compatibility
        for name, dialog in self.dialogs.items():
            setattr(self.ui, f"{name.replace('_', '').title()}Window", dialog)

    def _init_ui_components(self):
        """Initialize UI components with optimized settings"""
        # Logo setup with shadow effect
        self._setup_logo()

        # Ensure tabs use available space and have consistent sizing
        # This helps long labels (e.g., "Clustering") fit even when they turn bold
        if hasattr(self.ui, 'main_tab') and self.ui.main_tab is not None:
            self.ui.main_tab.tabBar().setExpanding(True)

        # Initial visibility
        set_widgets_status([
            self.ui.main_tab,
            self.ui.step0_study_name_mainwin_lineedit
        ], mode='hide')

        # Set fixed layout properties to prevent resizing issues
        self._setup_layout_properties()

    def _setup_logo(self):
        """Setup logo with effects"""
        icon_path = self.context.get_resource("eeg_comet_logo.png")
        pixmap = QPixmap(icon_path).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(5)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 80))

        self.ui.comet_logo.setGraphicsEffect(shadow)
        self.ui.comet_logo.setPixmap(pixmap)

    def _setup_layout_properties(self):
        """Setup fixed layout properties to prevent resizing"""
        # Set fixed spacing for layouts
        for widget in self.findChildren(QWidget):
            if hasattr(widget, 'layout') and widget.layout():
                layout = widget.layout()
                if isinstance(layout, (QVBoxLayout, QGridLayout)):
                    layout.setSpacing(10)
                    layout.setContentsMargins(10, 10, 10, 10)

    def _init_widget_groups(self):
        """Initialize widget groups manager"""
        self.widget_groups = WidgetGroups(self.ui)

    def _setup_connections(self):
        """Setup all signal-slot connections"""
        # Control widgets
        control_mappings = self._get_control_mappings()
        for widget, handler in control_mappings.items():
            self._connect_control_widget(widget, handler)

        # Button actions
        button_mappings = self._get_button_mappings()
        for button, action in button_mappings.items():
            button.clicked.connect(action)

        # Menu actions
        menu_mappings = self._get_menu_mappings()
        for action, (handler, shortcut) in menu_mappings.items():
            action.triggered.connect(handler)
            action.setShortcut(shortcut)

        # Dark mode checkbox connection
        if hasattr(self.ui, 'dark_mode_checkbox'):
            self.ui.dark_mode_checkbox.toggled.connect(self.toggle_theme)

    def _get_control_mappings(self) -> Dict[QWidget, callable]:
        """Get control widget to handler mappings"""
        return {
            self.ui.step2_auto_k_radio: self._update_ui_state,
            self.ui.step2_user_k_radio: self._update_ui_state,
            self.ui.step2_use_percent_radio: self._update_ui_state,
            self.ui.step2_use_peaks_radio: self._update_ui_state,
            self.ui.step2_clustermethod_combobox: self._update_ui_state,
            self.ui.step2_auto_k_method_combobox: self._update_ui_state,
            self.ui.step2_auto_range_kmin_spinbox: self._update_ui_state,
            self.ui.step2_auto_range_kmax_spinbox: self._update_ui_state,
            self.ui.step2_percent_slider: self._update_ui_state,
            self.ui.step2_batch_checkbox: self._update_ui_state,
            self.ui.step3_backfit_all_radio: self._update_ui_state,
            self.ui.step3_backfit_peaks_radio: self._update_ui_state,
            self.ui.step3_filter_segments_checkbox: self._update_ui_state,
            self.ui.step3_identify_short_checkbox: self._update_ui_state,
            self.ui.step3_filter_segments_method_combobox: self._update_ui_state,
            self.ui.step4_feature_occ_checkbox: self._update_ui_state,
            self.ui.step4_feature_dur_checkbox: self._update_ui_state,
            self.ui.step4_feature_cov_checkbox: self._update_ui_state,
            self.ui.step4_feature_gev_checkbox: self._update_ui_state,
            self.ui.step4_feature_tp_checkbox: self._update_ui_state,
            self.ui.step4_feature_se_checkbox: self._update_ui_state,
            self.ui.step4_feature_lzc_checkbox: self._update_ui_state,
            self.ui.step4_feature_er_checkbox: self._update_ui_state,
            self.ui.step4_feature_rof_checkbox: self._update_ui_state,
            self.ui.step4_feature_rtf_checkbox: self._update_ui_state,
            self.ui.step4_averaged_features_checkbox: self._update_ui_state,
            self.ui.step4_sliding_features_checkbox: self._update_ui_state,
            self.ui.step5_use_fsaverage_radio: self._update_ui_state,
            self.ui.step5_use_individual_radio: self._update_ui_state,
            self.ui.step5_use_tess_radio: self._update_ui_state,
            self.ui.step5_use_avg_radio: self._update_ui_state
        }

    def _connect_control_widget(self, widget: QWidget, handler: callable):
        """Connect control widget to handler based on widget type"""
        if isinstance(widget, QComboBox):
            widget.activated.connect(handler)
        elif isinstance(widget, (QSpinBox, QSlider)):
            widget.valueChanged.connect(handler)
        else:
            widget.clicked.connect(handler)

    def _get_button_mappings(self) -> Dict[QWidget, callable]:
        """Get button to action mappings"""
        return {
            self.ui.step0_load_study_button: self.load_study,
            self.ui.step0_new_study_button: self.open_new_study_dialog,
            self.ui.step0_compare_studies_button: self.open_compare_studies_window,
            self.ui.step2_numberofmaps_elbow_button: self.visualize_elbow,
            self.ui.step2_clustering_button: self.do_clustering,
            self.ui.step3_backfit_button: self.do_backfitting,
            self.ui.step4_extractfeatures_button: self.extract_features,
            self.ui.step5_coreg_button: self.coregister,
            self.ui.step5_use_individual_radio: self.locate_individual_subjects_dir,
            self.ui.step5_estimate_sources_button: self.source_localize_microstates,
            self.ui.step5_compute_source_microstate_correlation_button: self.source_microstates_correlation,
            self.ui.step0_exit_button: self.exit_msg
        }

    def _get_menu_mappings(self) -> Dict[Any, Tuple[callable, QKeySequence]]:
        """Get menu action to handler and shortcut mappings"""
        return {
            self.ui.open_github_action: (self.open_github, QKeySequence("F1")),
            self.ui.report_issues_action: (self.report_issues, QKeySequence("F2")),
            self.ui.update_action: (self.update_toolbox, QKeySequence("F3")),
            self.ui.step0_new_study_action: (self.open_new_study_dialog, QKeySequence("Ctrl+N")),
            self.ui.step0_load_study_action: (self.load_study, QKeySequence("Ctrl+L")),
            self.ui.step0_compare_studies_action: (self.open_compare_studies_window, QKeySequence("Ctrl+Shift+C")),
            self.ui.step0_reopen_log_window: (self.comet.LogWindow.show_hide_log_window, QKeySequence("F12")),
            self.ui.view_microstates_action: (self.visualize_microstates, QKeySequence("Shift+M")),
            self.ui.view_backfitting_action: (self.visualize_microstate_segmentation, QKeySequence("Shift+B")),
            self.ui.view_features_action: (self.visualize_microstate_features, QKeySequence("Shift+F")),
            self.ui.view_sources_action: (self.visualize_source_localized_microstates, QKeySequence("Shift+S")),
        }

    def _update_ui_state(self):
        """Centralized UI state update method"""
        # Defer the actual update to prevent layout thrashing
        QtCore.QTimer.singleShot(0, self._do_update_ui_state)

    def _do_update_ui_state(self):
        """Actual UI state update implementation"""
        # Store current focus to restore later
        current_focus = self.focusWidget()

        # Update tab styling with enough padding / width so text fits even when bold
        self.ui.main_tab.setStyleSheet(
            """
            QTabBar::tab {
                min-width: 120px;        /* ensure room for bold text */
                padding: 6px 12px;       /* vertical, horizontal */
            }
            QTabBar::tab:selected {
                font-weight: bold;       /* keep highlighting */
            }
            """
        )

        # Sync flags with COMET
        self._sync_processing_flags()

        # Update UI based on processing state
        if not self.comet.done_preprocessing:
            self._handle_preprocessing_state()
        else:
            self._handle_post_preprocessing_state()

        # Restore focus
        if current_focus:
            current_focus.setFocus()

        # Force layout update
        self.centralWidget().updateGeometry()
        QtCore.QCoreApplication.processEvents()

    def _sync_processing_flags(self):
        """Sync processing flags between UI and COMET"""
        for flag in self.processing_flags.__dataclass_fields__:
            setattr(self.processing_flags, flag, getattr(self.comet, flag))

    def _handle_preprocessing_state(self):
        """Handle UI state when preprocessing is not done"""
        # Disable all tabs
        for i in range(self.ui.main_tab.count()):
            self.ui.main_tab.setTabEnabled(i, False)

        # Show initial widgets
        self.widget_groups.set_group_status('hide_after_loading', WidgetMode.SHOW)
        set_widgets_status([self.ui.main_tab, self.ui.step0_study_name_mainwin_lineedit], mode='hide')
        self.widget_groups.set_group_status('after_preprocessing', WidgetMode.DISABLE)

    def _handle_post_preprocessing_state(self):
        """Handle UI state after preprocessing is done"""
        # Hide initial widgets
        self.widget_groups.set_group_status('hide_after_loading', WidgetMode.HIDE)
        set_widgets_status([self.ui.main_tab, self.ui.step0_study_name_mainwin_lineedit], mode='show')

        # Update study name display
        self.ui.step0_study_name_mainwin_lineedit.setText(self.comet.study_name)
        self.ui.step0_study_name_mainwin_lineedit.setStyleSheet("background-color: lightgreen")

        # Enable clustering tab
        self.ui.main_tab.setTabEnabled(0, True)
        self.widget_groups.set_group_status('after_preprocessing', WidgetMode.ENABLE)

        # Handle clustering method specific settings
        self._handle_clustering_method_settings()

        # Handle number of maps settings
        self._handle_number_of_maps_settings()

        # Handle batch processing settings
        self._handle_batch_processing_settings()

        # Handle data selection settings
        self._handle_data_selection_settings()

        # Handle post-clustering state
        if self.comet.done_clustering:
            self._handle_post_clustering_state()
        else:
            self._disable_post_clustering_features()

    def _handle_clustering_method_settings(self):
        """Handle clustering method specific UI settings"""
        self.comet.clustering_method = self.ui.step2_clustermethod_combobox.currentText()
        is_taahc = "Topographic Atomize and Agglomerate Hierarchical Clustering" in self.comet.clustering_method

        if is_taahc:
            # Force batch processing for TAAHC
            self.ui.step2_batch_checkbox.setChecked(True)
            self.ui.step2_batch_checkbox.setEnabled(False)
            self.widget_groups.set_group_status('convergence', WidgetMode.HIDE)
            self.widget_groups.set_group_status('batch', WidgetMode.ENABLE)
            if not self.ui.step2_batch_input.text():
                self.ui.step2_batch_input.setText("10000")
        else:
            self.ui.step2_batch_checkbox.setEnabled(True)
            self.widget_groups.set_group_status('convergence', WidgetMode.SHOW)
            self.widget_groups.set_group_status('convergence', WidgetMode.ENABLE)

            # Handle similarity metrics
            if self.comet.clustering_method != 'Modified K-Means Clustering (Pascual-Marqui et al. 1995)':
                self.widget_groups.set_group_status('similarity', WidgetMode.SHOW)
            else:
                self.widget_groups.set_group_status('similarity', WidgetMode.HIDE)

    def _handle_number_of_maps_settings(self):
        """Handle number of maps UI settings"""
        if self.ui.step2_auto_k_radio.isChecked():
            self.widget_groups.set_group_status('user_k', WidgetMode.DISABLE)
            self.widget_groups.set_group_status('auto_k', WidgetMode.ENABLE)

            # Validate k range
            kmin = int(self.ui.step2_auto_range_kmin_spinbox.value())
            kmax = int(self.ui.step2_auto_range_kmax_spinbox.value())
            if kmax <= kmin:
                self.ui.step2_auto_range_kmax_spinbox.setValue(kmin + 1)

            # Update target parameter label
            self._update_auto_k_parameter_label()
        else:
            self.widget_groups.set_group_status('user_k', WidgetMode.ENABLE)
            self.widget_groups.set_group_status('auto_k', WidgetMode.DISABLE)

    def _update_auto_k_parameter_label(self):
        """Update auto-k parameter label based on method"""
        method = self.ui.step2_auto_k_method_combobox.currentText()
        label_map = {
            'Gap Statistic': 'Random datasets:',
            'Cross Validation': 'Folds:',
            'Elbow - Global Explained Variance': 'Threshold (%):',
            'Elbow - Residual Variance': 'Threshold (%):'
        }
        label_text = label_map.get(method, '')
        self.ui.step2_auto_target_parameter_label.setText(label_text)

    def _handle_batch_processing_settings(self):
        """Handle batch processing UI settings"""
        is_taahc = "Topographic Atomize and Agglomerate Hierarchical Clustering" in self.comet.clustering_method

        if not is_taahc and self.ui.step2_batch_checkbox.isChecked():
            self.widget_groups.set_group_status('batch', WidgetMode.ENABLE)
            if not self.ui.step2_batch_input.text():
                self.ui.step2_batch_input.setText("1000")
        elif not is_taahc:
            self.widget_groups.set_group_status('batch', WidgetMode.DISABLE)

        # Ensure batch input is never empty
        if not self.ui.step2_batch_input.text().strip():
            default_batch = "10000" if is_taahc else "1000"
            self.ui.step2_batch_input.setText(default_batch)

        self.comet.batch_size = int(
            self.ui.step2_batch_input.text()) if self.ui.step2_batch_checkbox.isChecked() else None

    def _handle_data_selection_settings(self):
        """Handle data selection UI settings"""
        if self.ui.step2_use_peaks_radio.isChecked():
            self.widget_groups.set_group_status('peaks_use', WidgetMode.ENABLE)
            self.widget_groups.set_group_status('rand_use', WidgetMode.DISABLE)
        else:
            self.widget_groups.set_group_status('rand_use', WidgetMode.ENABLE)
            self.widget_groups.set_group_status('peaks_use', WidgetMode.DISABLE)
            self.ui.step2_percent_input.setText(str(self.ui.step2_percent_slider.value()))

    def _handle_post_clustering_state(self):
        """Handle UI state after clustering is done"""
        self.widget_groups.set_group_status('after_clustering', WidgetMode.ENABLE)

        # Enable microstate visualization action immediately after clustering
        self.ui.view_microstates_action.setEnabled(True)

        if self.comet.done_microstate_labeling:
            self._handle_post_labeling_state()
        else:
            self._reset_post_labeling_features()
        
        # Automatically open microstate visualization window ONLY after clustering just finished
        # (not when loading a study that already has clustered data)
        if (hasattr(self.comet, 'best_maps') and self.comet.best_maps is not None and
            getattr(self, '_clustering_just_finished', False)):
            QtCore.QTimer.singleShot(100, self.visualize_microstates)
            # Reset the flag so it doesn't auto-open again
            self._clustering_just_finished = False

    def _handle_post_labeling_state(self):
        """Handle UI state after microstate labeling"""
        self.ui.view_microstates_action.setEnabled(True)
        self.ui.main_tab.setTabEnabled(1, True)  # Enable backfitting tab

        # Update export format
        self._update_export_format()

        # Handle filter segments settings
        self._handle_filter_segments_settings()

        if self.comet.done_backfitting:
            self._handle_post_backfitting_state()
        else:
            self._disable_post_backfitting_features()

    def _update_export_format(self):
        """Update export format from UI"""
        output_format = self.ui.step4_outputformats_combobox.currentText()
        start = output_format.find("(") + 1
        end = output_format.find(")")
        if start > 0 and end > start:
            self.comet.export_format = output_format[start:end]

    def _handle_filter_segments_settings(self):
        """Handle filter segments UI settings"""
        if self.ui.step3_filter_segments_checkbox.isChecked():
            self.widget_groups.set_group_status('filter_segments', WidgetMode.ENABLE)

            method = self.ui.step3_filter_segments_method_combobox.currentText()
            if method == "Smooth segments":
                self.widget_groups.set_group_status('smooth_segments', WidgetMode.ENABLE)
            else:
                self.widget_groups.set_group_status('smooth_segments', WidgetMode.DISABLE)

            if not self.ui.step3_identify_short_checkbox.isChecked():
                self.widget_groups.set_group_status('identify_short', WidgetMode.ENABLE)
            else:
                self.widget_groups.set_group_status('identify_short', WidgetMode.DISABLE)
                if method == "Smooth segments":
                    self.widget_groups.set_group_status('smooth_segments', WidgetMode.DISABLE)
        else:
            self.widget_groups.set_group_status('filter_segments', WidgetMode.DISABLE)
            self.widget_groups.set_group_status('smooth_segments', WidgetMode.DISABLE)
            self.widget_groups.set_group_status('identify_short', WidgetMode.DISABLE)

    def _handle_post_backfitting_state(self):
        """Handle UI state after backfitting"""
        self.ui.step3_backfit_button.setStyleSheet("background-color: lightgreen")
        self.ui.main_tab.setTabEnabled(2, True)  # Enable feature tab
        self.ui.main_tab.setTabEnabled(3, True)  # Enable source tab
        self.ui.view_backfitting_action.setEnabled(True)

        self.widget_groups.set_group_status('feature_extraction', WidgetMode.ENABLE)
        self.widget_groups.set_group_status('source_localization', WidgetMode.ENABLE)

        # Handle epoched vs raw data features
        self._handle_data_type_features()

        # Handle feature extraction settings
        self._handle_feature_extraction_settings()

        # Handle source localization settings
        self._handle_source_localization_settings()

    def _disable_post_backfitting_features(self):
        """Disable features that require backfitting to be done"""
        self.ui.main_tab.setTabEnabled(2, False)  # Disable feature tab
        self.ui.main_tab.setTabEnabled(3, False)  # Disable source tab

        self.processing_flags.done_extracting_features = False
        self._sync_flags_to_comet()

        self.ui.step3_backfit_button.setStyleSheet("background-color: none")
        self.ui.view_backfitting_action.setDisabled(True)

    def _handle_data_type_features(self):
        """Handle features based on data type"""
        if self.comet.datatype == 'epoched':
            self.ui.step4_sliding_features_checkbox.setText("Extract Features Before and After TMS per Subject")

            # Enable epoched-specific features
            epoched_widgets = [
                self.ui.step4_features_epoched_label,
                self.ui.step4_feature_rof_checkbox,
                self.ui.step4_feature_rtf_checkbox
            ]
            set_widgets_status(epoched_widgets, mode='enable')
            self.ui.step4_feature_rof_checkbox.setChecked(True)
            self.ui.step4_feature_rtf_checkbox.setChecked(True)

            # Handle sliding window settings
            raw_widgets = [self.ui.step4_sliding_window_raw_label_0, self.ui.step4_sliding_window_raw_input]
            set_widgets_status(raw_widgets, mode='disable')

            if self.ui.step4_sliding_features_checkbox.isChecked():
                epoched_sliding = [
                    self.ui.step4_sliding_window_epoched_label_0,
                    self.ui.step4_sliding_window_epoched_label_1,
                    self.ui.step4_sliding_window_epoched_label_2,
                    self.ui.step4_sliding_window_epoched_input_pre,
                    self.ui.step4_sliding_window_epoched_input_post
                ]
                set_widgets_status(epoched_sliding, mode='enable')
            else:
                epoched_sliding = [
                    self.ui.step4_sliding_window_epoched_label_0,
                    self.ui.step4_sliding_window_epoched_label_1,
                    self.ui.step4_sliding_window_epoched_label_2,
                    self.ui.step4_sliding_window_epoched_input_pre,
                    self.ui.step4_sliding_window_epoched_input_post
                ]
                set_widgets_status(epoched_sliding, mode='disable')
        else:
            # Disable epoched-specific features
            epoched_widgets = [
                self.ui.step4_features_epoched_label,
                self.ui.step4_feature_rof_checkbox,
                self.ui.step4_feature_rtf_checkbox
            ]
            set_widgets_status(epoched_widgets, mode='disable')
            self.ui.step4_feature_rof_checkbox.setChecked(False)
            self.ui.step4_feature_rtf_checkbox.setChecked(False)

            # Handle sliding window settings
            epoched_sliding = [
                self.ui.step4_sliding_window_epoched_label_0,
                self.ui.step4_sliding_window_epoched_label_1,
                self.ui.step4_sliding_window_epoched_label_2,
                self.ui.step4_sliding_window_epoched_input_pre,
                self.ui.step4_sliding_window_epoched_input_post
            ]
            set_widgets_status(epoched_sliding, mode='disable')

            if self.ui.step4_sliding_features_checkbox.isChecked():
                raw_widgets = [self.ui.step4_sliding_window_raw_label_0, self.ui.step4_sliding_window_raw_input]
                set_widgets_status(raw_widgets, mode='enable')
            else:
                raw_widgets = [self.ui.step4_sliding_window_raw_label_0, self.ui.step4_sliding_window_raw_input]
                set_widgets_status(raw_widgets, mode='disable')

    def _handle_feature_extraction_settings(self):
        """Handle feature extraction UI settings"""
        # Handle ER feature word size settings
        if self.ui.step4_feature_er_checkbox.isChecked():
            er_widgets = [
                self.ui.step4_word_size_label1,
                self.ui.step4_word_size_label2,
                self.ui.step4_word_size_label3,
                self.ui.step4_word_size_min_input,
                self.ui.step4_word_size_max_input
            ]
            set_widgets_status(er_widgets, mode='enable')
        else:
            er_widgets = [
                self.ui.step4_word_size_label1,
                self.ui.step4_word_size_label2,
                self.ui.step4_word_size_label3,
                self.ui.step4_word_size_min_input,
                self.ui.step4_word_size_max_input
            ]
            set_widgets_status(er_widgets, mode='disable')

        # Check if extract button should be enabled
        feature_checks = any(cb.isChecked() for cb in self.widget_groups.groups['feature_checkboxes'])
        mode_checks = any(cb.isChecked() for cb in self.widget_groups.groups['feature_modes'])

        self.ui.step4_extractfeatures_button.setEnabled(feature_checks and mode_checks)

        # Update button styles based on processing state
        if self.comet.done_extracting_features:
            self.ui.step4_extractfeatures_button.setStyleSheet("background-color: lightgreen")
            self.ui.view_features_action.setEnabled(True)
        else:
            self.ui.step4_extractfeatures_button.setStyleSheet("background-color: none")
            self.ui.view_features_action.setDisabled(True)

    def _handle_source_localization_settings(self):
        """Handle source localization UI settings"""
        # Handle anatomy settings
        if self.ui.step5_use_individual_radio.isChecked():
            self.comet.use_anatomy = "individual"
            self.widget_groups.set_group_status('source_individual', WidgetMode.ENABLE)
        else:
            self.comet.use_anatomy = "fsaverage"
            self.widget_groups.set_group_status('source_individual', WidgetMode.DISABLE)

        # Handle source localization method
        if self.ui.step5_use_tess_radio.isChecked():
            self.widget_groups.set_group_status('tess', WidgetMode.ENABLE)
            self.comet.source_localization_method = 'tess'
        else:
            self.widget_groups.set_group_status('tess', WidgetMode.DISABLE)
            self.comet.source_localization_method = 'avg'

        # Handle post source localization state
        if not self.comet.done_source_localization:
            self.ui.step5_estimate_sources_button.setStyleSheet("background-color: none")
            self.widget_groups.set_group_status('source_microstates', WidgetMode.DISABLE)
        else:
            self.ui.step5_estimate_sources_button.setStyleSheet("background-color: lightgreen")
            self.widget_groups.set_group_status('source_microstates', WidgetMode.ENABLE)

            if self.comet.done_identifying_microstate_sources:
                self.ui.step5_compute_source_microstate_correlation_button.setStyleSheet("background-color: lightgreen")
                self.ui.view_sources_action.setEnabled(True)
            else:
                self.ui.step5_compute_source_microstate_correlation_button.setStyleSheet("background-color: none")
                self.ui.view_sources_action.setDisabled(True)

    def _reset_post_labeling_features(self):
        """Reset features that depend on microstate labeling"""
        self.processing_flags.reset_from('done_backfitting')
        self._sync_flags_to_comet()

    def _sync_flags_to_comet(self):
        """Sync processing flags to COMET instance"""
        for flag in self.processing_flags.__dataclass_fields__:
            setattr(self.comet, flag, getattr(self.processing_flags, flag))

    # Static methods
    @staticmethod
    def open_github():
        """Open the GitHub page in the default web browser"""
        webbrowser.open('https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction')

    @staticmethod
    def report_issues():
        """Open the GitHub issues page in the default web browser"""
        webbrowser.open('https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction/issues/new')

    def update_toolbox(self):
        """Ask the user if they want to download the toolbox"""
        reply = QMessageBox.question(
            self, 'Update Toolbox',
            "Do you want to download the latest version of the toolbox?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            webbrowser.open(
                'https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction/archive/refs/heads/main.zip')

    # Dialog methods
    def open_new_study_dialog(self):
        """Open new study dialog"""
        self.ui.step0_study_name_mainwin_lineedit.clear()
        self.processing_flags.reset_all()
        self._sync_flags_to_comet()

        self.dialogs['new_study'].setWindowModality(QtCore.Qt.ApplicationModal)
        self.dialogs['new_study'].showMaximized()
        self._update_ui_state()

    def open_compare_studies_window(self):
        """Open compare studies window"""
        self.dialogs['compare_studies'].setWindowModality(QtCore.Qt.ApplicationModal)
        self.dialogs['compare_studies'].showMaximized()

    # Study management methods
    def load_study(self, from_new_study=False):
        """Load a study with improved error handling"""
        if from_new_study:
            self._handle_new_study_load()
        else:
            self._handle_existing_study_load()

        self._update_ui_state()

    def _handle_new_study_load(self):
        """Handle loading from new study"""
        if hasattr(self.comet, 'reset_directories'):
            self.comet.reset_directories()

        if not hasattr(self.comet, 'LogWindow') or self.comet.LogWindow is None:
            self.comet.initialize_log_window()

        self.comet.LogWindow.show()
        
        # Save the configuration to persist the initial log
        if self.comet.auto_save:
            self.comet.save_config()

    def _handle_existing_study_load(self):
        """Handle loading existing study"""
        if self.comet.done_preprocessing:
            reply = QMessageBox.question(
                self, 'Load Study',
                f"The {self.comet.study_name} is already loaded. Do you want to load another study?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._load_study_from_folder()
        else:
            self._load_study_from_folder()

    def _load_study_from_folder(self):
        """Load study configuration from selected folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Study Folder", 
            self.comet.output_folder if hasattr(self.comet, 'output_folder') else ""
        )
        
        if not folder:
            return
            
        # Look for config file
        config_path = os.path.join(folder, "eeg_comet_config.ini")
        if not os.path.exists(config_path):
            QMessageBox.warning(
                self, "Config Not Found",
                f"No configuration file found in:\n{folder}",
                QMessageBox.Ok
            )
            return

        try:
            # Load configuration
            self.comet.config = self.comet.load_config(config_path)
            self.comet.load_config_values()
            
            # Reset directories first to ensure log_file_path points to correct location
            self.comet.reset_directories()
            
            # Now restore logs from the correct location
            if hasattr(self.comet, 'LogWindow') and self.comet.LogWindow:
                restored = self.comet.restore_logs_if_available()
            else:
                self.comet.initialize_log_window()
                self.comet.restore_logs_if_available()
            
            self.comet.load_eeg_info()
            self.comet.load_maps()
            self.comet.load_clean()

            self.comet.LogWindow.show()

            # Add study loading messages
            self.comet.LogWindow.append_log("Study Loading", log_type='section')
            self.comet.LogWindow.append_log(f"Study '{self.comet.study_name}' loaded successfully from: {folder}", log_type='success')

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load study parameters: {e}",
                QMessageBox.Ok
            )

    # Processing methods
    def _update_comet_clustering_parameters(self):
        """Update COMET instance with current UI clustering parameters"""
        # Smoothing parameters
        if self.ui.step2_kernel_size_input.text():
            self.comet.smoothing_gfp = True
            self.comet.smoothing_distance = int(self.ui.step2_kernel_size_input.text())
            self.comet.min_distance_size = int(self.comet.smoothing_distance / (1000 / self.comet.sample_rate))
        else:
            self.comet.smoothing_gfp = False
            self.comet.smoothing_distance = 0
            self.comet.min_distance_size = None

        # Data selection parameters
        if self.ui.step2_use_percent_radio.isChecked():
            self.comet.use_percentages = int(self.ui.step2_percent_slider.value())
        else:
            self.comet.use_percentages = None

        # Clustering parameters
        self.comet.clustering_tolerance = float(self.ui.step2_stopcondition_input.text())
        self.comet.max_iterations = int(self.ui.step2_maxiter_input.text())
        self.comet.number_of_repeats = int(self.ui.step2_numberofrepeats_input.text())

        # K range for auto mode
        if self.ui.step2_auto_k_radio.isChecked():
            self.comet.kmin = int(self.ui.step2_auto_range_kmin_spinbox.value())
            self.comet.kmax = int(self.ui.step2_auto_range_kmax_spinbox.value())

    def visualize_elbow(self):
        """Visualize optimization plots for determining optimal clusters"""
        self._update_comet_clustering_parameters()

        # Check for existing optimization results
        if (hasattr(self.comet, 'optimization_results') and
                self.comet.optimization_results and
                self.comet.choose_number_of_maps == "auto"):

            # Use existing results
            self.dialogs['optimizer'].results_cache = {}

            for method_code, result in self.comet.optimization_results.items():
                if method_code != 'majority_vote':
                    cache_key = f"{method_code}_None"
                    self.dialogs['optimizer'].results_cache[cache_key] = {
                        'method': method_code,
                        'result': result,
                        'optimal_k': result.optimal_k,
                        'k_values': result.k_values,
                        'scores': result.scores
                    }
        else:
            self.dialogs['optimizer'].results_cache = {}

        # Update parameters
        self.dialogs['optimizer']._load_comet_parameters()

        if hasattr(self.comet, 'kmin'):
            self.dialogs['optimizer'].ui.optimizer_min_input.setText(str(self.comet.kmin))
        if hasattr(self.comet, 'kmax'):
            self.dialogs['optimizer'].ui.optimizer_max_input.setText(str(self.comet.kmax))

        self.dialogs['optimizer'].optimizer = None
        self.dialogs['optimizer'].setWindowModality(QtCore.Qt.ApplicationModal)
        self.dialogs['optimizer'].showMaximized()

    def do_clustering(self):
        """Perform clustering with confirmation dialog"""
        if self.comet.done_clustering:
            reply = QMessageBox.question(
                self, 'Redo Clustering',
                "Data has been clustered once. Do you want to redo the analysis?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

            # Reset flags
            self.processing_flags.reset_from('done_clustering')
            self._sync_flags_to_comet()

            # Clear previous results
            if hasattr(self.comet, 'optimization_results'):
                self.comet.optimization_results = None

        # Update parameters
        self._update_comet_clustering_parameters()

        # Set clustering method parameters
        if self.ui.step2_auto_k_radio.isChecked():
            self._set_auto_k_parameters()
        else:
            self._set_user_k_parameters()

        # Set other parameters
        self._set_clustering_parameters()

        # Set up callback to update UI when clustering finishes
        self.comet.clustering_completed_callback = self._on_clustering_finished

        # Perform clustering
        self.comet.run_clustering()
        self._update_ui_state()

    def _on_clustering_finished(self):
        """Called when clustering is finished to update UI state"""
        # Set flag to indicate clustering just completed (for auto-opening visualization)
        self._clustering_just_finished = True
        
        # Update the UI state now that clustering is complete
        self._update_ui_state()

    def _set_auto_k_parameters(self):
        """Set parameters for automatic k selection"""
        self.comet.choose_number_of_maps = "auto"
        self.comet.number_of_maps = 'auto'

        method_map = {
            'Gap Statistic': 'gs',
            'Cross Validation': 'cv',
            'Elbow - Global Explained Variance': 'gev',
            'Elbow - Residual Variance': 'res',
            'Silhouette Method': 'sil',
            'Calinski-Harabasz Method': 'ch',
            'Davies-Bouldin Method': 'db'
        }

        method = self.ui.step2_auto_k_method_combobox.currentText()
        self.comet.stopping_mode = method_map.get(method, 'gev')
        self.comet.stopping_parameter = float(self.ui.step2_stopping_threshold_input.text())

    def _set_user_k_parameters(self):
        """Set parameters for user-defined k"""
        self.comet.choose_number_of_maps = "user"
        self.comet.stopping_mode = ''
        self.comet.stopping_parameter = ''
        self.comet.kmin = ''
        self.comet.kmax = ''
        self.comet.number_of_maps = int(self.ui.step2_user_k_input.text())

    def _set_clustering_parameters(self):
        """Set general clustering parameters"""
        # Initializer
        if self.ui.step2_random_initializer_radio.isChecked():
            self.comet.initializer = "Random"
        else:
            self.comet.initializer = "K-Means++"

        # Method and metric
        self.comet.clustering_method = self.ui.step2_clustermethod_combobox.currentText()
        self.comet.similarity_metric = self.ui.step2_similarity_combobox.currentText()

        # Batch size
        if self.ui.step2_batch_checkbox.isChecked():
            self.comet.batch_size = int(self.ui.step2_batch_input.text()) if self.ui.step2_batch_input.text() else 1000
        else:
            self.comet.batch_size = None

        # Paths
        self.comet.microstate_maps_path = os.path.join(self.comet.save_dir, 'microstate_maps.csv')

    def visualize_microstates(self):
        """Visualize microstate maps for labeling"""
        # Check if microstate maps are available
        if not hasattr(self.comet, 'best_maps') or self.comet.best_maps is None:
            QMessageBox.warning(
                self,
                "No Microstate Maps Available",
                "No microstate maps have been generated yet. Please complete the clustering process first."
            )
            return

        # ---------------------------------------------
        # Re-use an existing visualization window if it
        # is already open instead of opening duplicates
        # ---------------------------------------------
        if hasattr(self, "_microstate_window") and self._microstate_window is not None:
            if self._microstate_window.isVisible():
                # Refresh maps/labels inside the existing window
                self._microstate_window.plot_maps()
                self._microstate_window.raise_()
                self._microstate_window.activateWindow()
                return  # Skip creating a new window

        # Create new instance to ensure fresh state
        microstate_window = MicrostateVisualizationWindow(
            self.context,
            main_window=self,
            tbx=self.comet
        )
        microstate_window.plot_maps()
        microstate_window.setWindowModality(QtCore.Qt.ApplicationModal)
        microstate_window.showMaximized()

        # Cache reference so we can reuse/refresh it later
        self._microstate_window = microstate_window
        # Ensure cache is cleared when window is closed
        microstate_window.destroyed.connect(lambda: setattr(self, "_microstate_window", None))

    def do_backfitting(self):
        """Perform microstate backfitting"""
        if self.comet.done_backfitting:
            reply = QMessageBox.question(
                self, 'Redo Backfitting',
                "Microstates have been backfitted to data once. Do you want to redo backfitting?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Reset flags
        self.processing_flags.reset_from('done_extracting_features')
        self._sync_flags_to_comet()

        # Clear any previous callbacks
        self.comet.clustering_completed_callback = None

        # Set backfitting parameters
        self._set_backfitting_parameters()

        # Perform backfitting
        self.comet.run_backfitting()
        self.ui.main_tab.setCurrentIndex(2)
        self._update_ui_state()

    def _set_backfitting_parameters(self):
        """Set backfitting parameters from UI"""
        # Backfit target
        if self.ui.step3_backfit_all_radio.isChecked():
            self.comet.backfit_to = 'all'
            self.comet.identify_short_window = self.ui.step3_identify_short_checkbox.isChecked()
        else:
            self.comet.backfit_to = 'peaks'
            self.comet.identify_short_window = False

        # Filter segments
        if self.ui.step3_filter_segments_checkbox.isChecked():
            self.comet.filter_segments = True
            self.comet.remove_segments_less_than = int(
                int(self.ui.step3_filter_segments_input.text()) / (1000 / self.comet.sample_rate)
            )

            # Set filter method
            method_map = {
                'Replace short segments: nearby dominant microstate': 'replace_high',
                'Replace short segments: half and half': 'replace_half',
                'Remove short segments': 'remove',
                'Smooth segments': 'smooth'
            }
            method = self.ui.step3_filter_segments_method_combobox.currentText()
            self.comet.filter_segments_option = method_map.get(method, 'remove')

            # Set smooth parameters if needed
            if self.comet.filter_segments_option == 'smooth':
                self.comet.epsilon = float(self.ui.step3_smooth_segments_epsilon_input.text())
                self.comet.b = self.comet.remove_segments_less_than
                self.comet.lamb = int(self.ui.step3_smooth_segments_lambda_input.text())
            else:
                self.comet.epsilon = ''
                self.comet.b = ''
                self.comet.lamb = ''
        else:
            self.comet.filter_segments = False
            self.comet.filter_segments_option = ''
            self.comet.remove_segments_less_than = []
            self.comet.epsilon = ''
            self.comet.b = ''
            self.comet.lamb = ''

    def visualize_microstate_segmentation(self):
        """Open backfitting visualization window"""
        viz_window = self.dialogs['backfitting']

        # Set parameters
        viz_window.preprocessed_data_path = self.comet.preprocessed_data_path
        viz_window.extension = self.comet.extension
        viz_window.datatype = self.comet.datatype
        viz_window.eeg_filenames_combobox.clear()
        viz_window.eeg_filenames_combobox.addItems(self.comet.list_eegs)
        viz_window.segmentation_path = self.comet.segmentation_path
        viz_window.export_format = self.comet.export_format

        # Show window
        viz_window.setWindowModality(QtCore.Qt.ApplicationModal)
        viz_window.showMaximized()

    def extract_features(self):
        """Extract features from backfitted data"""
        if self.comet.done_extracting_features:
            reply = QMessageBox.question(
                self, 'Re-extract Features',
                "Features have been extracted once. Do you want to extract features again?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Reset flag and update UI
        self.comet.done_extracting_features = False
        self._update_ui_state()

        # Clear any previous callbacks
        self.comet.clustering_completed_callback = None

        # Set feature parameters
        self._set_feature_extraction_parameters()

        # Perform feature extraction
        self.comet.run_feature_extraction()
        self.comet.done_extracting_features = True
        self._update_ui_state()

    def _set_feature_extraction_parameters(self):
        """Set feature extraction parameters from UI"""
        # Feature list
        feature_map = {
            self.ui.step4_feature_occ_checkbox: "OCC",
            self.ui.step4_feature_dur_checkbox: "DUR",
            self.ui.step4_feature_cov_checkbox: "COV",
            self.ui.step4_feature_gev_checkbox: "GEV",
            self.ui.step4_feature_tp_checkbox: "TP",
            self.ui.step4_feature_se_checkbox: "SE",
            self.ui.step4_feature_lzc_checkbox: "LZC",
            self.ui.step4_feature_er_checkbox: "ER"
        }

        self.comet.feature_list = [
            feature for checkbox, feature in feature_map.items()
            if checkbox.isChecked()
        ]

        # Feature modes
        self.comet.feature_mode = []
        if self.ui.step4_averaged_features_checkbox.isChecked():
            self.comet.feature_mode.append("averaged")
        if self.ui.step4_sliding_features_checkbox.isChecked():
            self.comet.feature_mode.append("sliding")

        # Feature types
        if self.ui.step4_synthetic_checkbox.isChecked():
            self.comet.feature_types = ['real', 'surrogate', 'random']
        else:
            self.comet.feature_types = ['real']

        # Word size for ER
        if self.ui.step4_feature_er_checkbox.isChecked():
            self.comet.word_size = int(self.ui.step4_word_size_min_input.text())
        else:
            self.comet.word_size = 2

        # Data type specific parameters
        if self.comet.datatype == 'epoched':
            self.comet.pre_window_size = int(self.ui.step4_sliding_window_epoched_input_pre.text())
            self.comet.post_window_size = int(self.ui.step4_sliding_window_epoched_input_post.text())

            if self.ui.step4_feature_rof_checkbox.isChecked():
                self.comet.feature_list.append("ROF")
            if self.ui.step4_feature_rtf_checkbox.isChecked():
                self.comet.feature_list.append("RTF")
        else:
            self.comet.sliding_window_size = int(self.ui.step4_sliding_window_raw_input.text())

    def visualize_microstate_features(self):
        """Open feature visualization window"""
        viz_window = self.dialogs['features']

        # Set parameters
        viz_window.extracted_features_path = self.comet.extracted_features_path
        viz_window.export_format = self.comet.export_format
        viz_window.feature_mode = self.comet.feature_mode
        viz_window.feature_combo.clear()
        viz_window.feature_combo.addItems(self.comet.feature_list)
        viz_window.list_eegs = self.comet.list_eegs
        viz_window.reset_groups()

        # Show window
        viz_window.setWindowModality(QtCore.Qt.ApplicationModal)
        viz_window.showMaximized()

    def coregister(self):
        """Open coregistration window"""
        coreg_window = CoregistrationWindow(self.context, tbx=self.comet)
        coreg_window.showMaximized()

    def locate_individual_subjects_dir(self):
        """Select individual subjects directory"""
        subjects_dir = QFileDialog.getExistingDirectory(
            self,
            "Locate Folder with Individual Anatomical Reconstructions"
        )

        if subjects_dir:
            self.comet.individual_subjects_dir = subjects_dir
            self.ui.step5_subjects_dir_lineedit.setText(subjects_dir)
        else:
            self.ui.step5_use_fsaverage_radio.setChecked(True)

    def source_localize_microstates(self):
        """Perform source localization"""
        if self.comet.done_source_localization:
            reply = QMessageBox.question(
                self, 'Redo Source Localization',
                "Source time series have been extracted once. Do you want to extract source time series again?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Reset flags
        self.processing_flags.reset_from('done_source_localization')
        self._sync_flags_to_comet()
        self._update_ui_state()

        # Set source localization parameters
        self._set_source_localization_parameters()

        # Perform source localization
        self.comet.run_source_localization()
        self._update_ui_state()

    def _set_source_localization_parameters(self):
        """Set source localization parameters from UI"""
        # BEM solver
        if self.ui.step5_bem_openmeeg_radio.isChecked():
            self.comet.bem_solver = 'openmeeg'
        else:
            self.comet.bem_solver = 'mne'

        # Inverse method
        method_text = self.ui.step5_inverse_method_combobox.currentText()
        start = method_text.find("(") + 1
        end = method_text.find(")")
        self.comet.inverse_method = method_text[start:end] if start > 0 and end > start else 'MNE'

        # Spacing
        spacing_text = self.ui.step5_spacing_combobox.currentText()
        start = spacing_text.find("(") + 1
        end = spacing_text.find(")")
        self.comet.spacing = spacing_text[start:end].lower() if start > 0 and end > start else 'ico4'

        # Permutations
        self.comet.nperm = int(self.ui.step5_permutations_input.text())

    def source_microstates_correlation(self):
        """Calculate source-microstate correlations"""
        if self.comet.done_identifying_microstate_sources:
            reply = QMessageBox.question(
                self, 'Recalculate Correlations',
                "Source-microstate correlations have already been calculated. Do you want to run this step again?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Set parameters
        self.comet.nperm = int(self.ui.step5_permutations_input.text())

        # Perform calculation
        self.comet.run_identifying_microstate_sources()
        self.comet.done_identifying_microstate_sources = True
        self._update_ui_state()

    def visualize_source_localized_microstates(self):
        """Open source visualization window"""
        source_window = SourceVisualizationWindow(self.context, comet_tbx=self.comet)
        source_window.setWindowModality(QtCore.Qt.ApplicationModal)
        source_window.showMaximized()

    def exit_msg(self):
        """Display confirmation before quitting"""
        reply = QMessageBox.question(
            self, "Quit",
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if hasattr(self.comet, 'LogWindow') and self.comet.LogWindow:
                self.comet.LogWindow.close()
            self.close()

    # Override close event for cleanup
    def closeEvent(self, event):
        """Handle window close event"""
        # Clear any callbacks
        self.comet.clustering_completed_callback = None
        if hasattr(self.comet, 'LogWindow') and self.comet.LogWindow is not None:
            self.comet.LogWindow.close()

        # Close all dialogs
        for dialog in self.dialogs.values():
            if dialog and hasattr(dialog, 'close'):
                dialog.close()

        event.accept()

    # Utility method for updating the main window controller
    def mainwindow_controller(self):
        """Legacy method for backward compatibility"""
        self._update_ui_state()

    def _disable_post_clustering_features(self):
        """Disable features that require clustering to be done"""
        for i in range(1, 4):  # Disable backfitting, feature, and source tabs
            self.ui.main_tab.setTabEnabled(i, False)

        self.processing_flags.reset_from('done_microstate_labeling')
        self._sync_flags_to_comet()

        self.ui.step2_clustering_button.setStyleSheet("background-color: none")
        self.comet.best_maps, self.comet.micro_labels = None, []

        self.widget_groups.set_group_status('after_clustering', WidgetMode.DISABLE)
