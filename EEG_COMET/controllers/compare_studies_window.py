"""UI to compare two EEG-COMET studies across features and maps."""

import os.path
import re
import traceback
from collections import Counter, defaultdict
from contextlib import suppress
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QSizePolicy
from scipy.stats import f, pearsonr, ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests

from clustering_utils.microstate_visualizer import show_microstate
from comet import COMET
from features_utils.feature_io import FeatureIO
from gui_utils.set_widgets_status import set_widgets_status
from gui_utils.terminal_logger import get_logger


class CompareStudiesWindow(QDialog):
    """Dialog window to compare two EEG-COMET studies.

    Provides tools to load two studies, visualize their microstate maps, and
    compare static/dynamic features including statistical summaries.

    Attributes:
        study1_loaded: Whether the first study has been loaded.
        study2_loaded: Whether the second study has been loaded.
        comet_tbx_study1: COMET toolbox instance for study 1.
        comet_tbx_study2: COMET toolbox instance for study 2.
        study1_path: File path to study 1.
        study2_path: File path to study 2.
        study2_name: Display name for study 2.
        synthetic_type: Type of synthetic comparison ('surrogate' or 'random').
        feature_list_dictionary: Dictionary mapping feature codes to full names.
        pattern_info: Information about detected filename patterns.
    """

    # Constants
    FEATURE_TYPES = ["real", "surrogate", "random"]
    FEATURE_MODES = ["static", "dynamic", "averaged", "sliding"]
    EXPORT_FORMATS = [".csv", ".pkl", ".hdf", ".json"]
    CONFIG_FILENAME = "eeg_comet_config.ini"

    # UI Text Constants
    STUDY_NAMES = {
        "surrogate": "Surrogate Study",
        "random": "Random Study",
        "within": "Within Study",
    }

    ERROR_MESSAGES = {
        "no_config": (
            "The selected folder does not contain a valid EEG-COMET study!\n"
            "Please select a folder containing 'eeg_comet_config.ini' file."
        ),
        "no_features": "No compatible features found",
        "no_synthetic": "No synthetic features available",
        "no_within_features": "No features available for within-study comparison",
    }

    def __init__(self, context, parent: Optional = None):
        """Initialize the dialog and wire up the UI.

        Args:
            context: Resource/context provider to resolve UI resources.
            parent: Optional parent widget.
        """
        super().__init__(parent)

        # Initialize flags and data
        self.study1_loaded = False
        self.study2_loaded = False
        self.comet_tbx_study1: Optional[COMET] = None
        self.comet_tbx_study2: Optional[COMET] = None
        self.study1_path: Optional[str] = None
        self.study2_path: Optional[str] = None
        self.study2_name: Optional[str] = None
        self.synthetic_type: Optional[str] = None
        self.feature_list_dictionary: Optional[Dict] = None
        self.pattern_info: Optional[Dict] = None

        # Setup UI components and connections
        self._setup_ui(context)
        self._connect_ui_signals()
        self._create_matplotlib_components()
        self.update_ui()

    # ==================== UI INITIALIZATION ====================

    def _setup_ui(self, context) -> None:
        """Set up static UI components from the Qt Designer file.

        Args:
            context: Resource/context provider to resolve the `.ui` path.
        """
        self.ui = uic.loadUi(context.get_resource("CompareStudiesWindow.ui"), self)
        self.ui.setWindowTitle("Comparison of two EEG-COMET studies")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        
        # Initialize checkboxes
        self.ui.paired_test_checkbox.setChecked(False)
        self.ui.show_features_checkbox.setChecked(False)
        # show_microstates_checkbox is set to True by default in UI file, keep that setting
        
        # Initialize widget and label visibility based on checkbox states
        features_visible = self.ui.show_features_checkbox.isChecked()
        microstates_visible = self.ui.show_microstates_checkbox.isChecked()
        
        self.ui.plot_label.setVisible(features_visible)
        self.ui.verticalLayoutWidget.setVisible(features_visible)
        
        self.ui.microstate_label.setVisible(microstates_visible)
        self.ui.layoutWidget.setVisible(microstates_visible)

    def _connect_ui_signals(self) -> None:
        """Connect UI signals to their handlers."""
        # Study loading buttons
        self.ui.load_study1_button.clicked.connect(self.load_study1)
        self.ui.load_study2_button.clicked.connect(self.load_study2)

        # Radio button connections
        compare_widgets = [
            self.ui.compare_two_studies_radio,
            self.ui.compare_surrogate_radio,
            self.ui.compare_random_radio,
            self.ui.compare_within_study_radio,
        ]
        for widget in compare_widgets:
            widget.clicked.connect(self.update_ui)

        # Feature and analysis connections
        self.ui.feature_combo.currentTextChanged.connect(self._update_plot_label)
        self.ui.feature_combo.currentTextChanged.connect(self._on_feature_selection_changed)
        self.ui.show_features_checkbox.toggled.connect(self._on_show_features_toggled)
        self.ui.show_microstates_checkbox.toggled.connect(self._on_show_microstates_toggled)
        self.ui.compare_features_button.clicked.connect(self.update_feature_stats)
        self.ui.plot_testretest_button.clicked.connect(self.perform_test_retest_analysis)
        self.ui.match_subjects_button.clicked.connect(self.match_subjects_by_pattern)

    def _create_matplotlib_components(self) -> None:
        """Create matplotlib figures and canvases for visualization."""
        # Study 1 microstates
        self.figure_microstates_study1 = Figure(tight_layout=True)
        self.canvas_microstates_study1 = self._create_canvas(
            self.figure_microstates_study1
        )
        self.ui.Figure_Microstates_Study1_Layout.addWidget(
            self.canvas_microstates_study1
        )

        # Study 2 microstates
        self.figure_microstates_study2 = Figure(tight_layout=True)
        self.canvas_microstates_study2 = self._create_canvas(
            self.figure_microstates_study2
        )
        self.ui.Figure_Microstates_Study2_Layout.addWidget(
            self.canvas_microstates_study2
        )

        # Features plot
        self.figure_features = Figure(tight_layout=True)
        self.canvas_features = self._create_canvas(self.figure_features)
        self.ui.Figure_Features_Layout.addWidget(self.canvas_features)

    def _create_canvas(self, figure: Figure) -> FigureCanvasQTAgg:
        """Create a matplotlib canvas with standard settings.

        Args:
            figure: The matplotlib figure to attach to the canvas.

        Returns:
            Configured FigureCanvasQTAgg instance.
        """
        canvas = FigureCanvasQTAgg(figure)
        canvas.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        canvas.setMinimumHeight(100)
        return canvas

    # ==================== STUDY LOADING ====================

    def load_study1(self) -> None:
        """Load the first EEG-COMET study via configuration discovery."""
        self.study1_path = self._select_study_folder(
            "Select the folder containing an EEG-COMET study."
        )
        if not self.study1_path:
            return

        try:
            self.comet_tbx_study1 = self._load_comet_study(self.study1_path)
            self.study1_loaded = True
            self._update_study_ui(
                self.comet_tbx_study1,
                self.ui.study1_file_list,
                self.figure_microstates_study1,
                self.canvas_microstates_study1,
            )
            self.update_ui()
            self._enable_match_button_if_applicable()
            self._log_study_completion_status(self.comet_tbx_study1, "Study 1")

        except Exception as e:
            self._show_error("Load Error", f"Failed to load study: {str(e)}")

    def load_study2(self) -> None:
        """Load the second EEG-COMET study via configuration discovery."""
        self.study2_path = self._select_study_folder(
            "Select the folder containing an EEG-COMET study."
        )
        if not self.study2_path:
            return

        try:
            self.comet_tbx_study2 = self._load_comet_study(self.study2_path)
            self.study2_loaded = True
            self._update_study_ui(
                self.comet_tbx_study2,
                self.ui.study2_file_list,
                self.figure_microstates_study2,
                self.canvas_microstates_study2,
            )
            self.update_ui()
            self._log_study_completion_status(self.comet_tbx_study2, "Study 2")

        except Exception as e:
            self._show_error("Load Error", f"Failed to load study: {str(e)}")

    def _select_study_folder(self, caption: str) -> Optional[str]:
        """Select a study folder via dialog."""
        return QFileDialog.getExistingDirectory(self, caption)

    def _load_comet_study(self, study_path: str) -> COMET:
        """Load a COMET study from the given path.

        Args:
            study_path: Path to the study directory.

        Returns:
            Loaded COMET instance.

        Raises:
            FileNotFoundError: If config file is not found.
            Exception: If loading fails.
        """
        config_path = os.path.join(study_path, self.CONFIG_FILENAME)
        if not os.path.exists(config_path):
            raise FileNotFoundError(self.ERROR_MESSAGES["no_config"])

        # Create and configure COMET instance
        comet_tbx = COMET(auto_save=False)
        comet_tbx.config = comet_tbx.load_config(config_path)
        comet_tbx.load_config_values()
        comet_tbx.reset_directories()

        # Load study data
        comet_tbx.load_eeg_info()

        # Load maps if clustering is complete
        if comet_tbx.done_clustering:
            with suppress(FileNotFoundError):
                comet_tbx.load_maps()

        comet_tbx.load_clean()
        return comet_tbx

    def _update_study_ui(
        self, tbx: COMET, listwidget, figure: Figure, canvas: FigureCanvasQTAgg
    ) -> None:
        """Update UI components with loaded study information.

        Args:
            tbx: COMET study instance.
            listwidget: Target QListWidget to receive entries.
            figure: Figure for plotting maps.
            canvas: Canvas for rendering.
        """
        self._populate_file_list(tbx, listwidget)
        # Only plot microstates if checkbox is checked
        if self.ui.show_microstates_checkbox.isChecked():
            self._plot_microstate_maps(tbx, figure, canvas)
        else:
            self._hide_microstate_maps(figure, canvas)

    def _populate_file_list(self, tbx: COMET, listwidget) -> None:
        """Populate a list widget with study file names.

        Args:
            tbx: COMET study instance.
            listwidget: Target QListWidget to receive entries.
        """
        for eeg in tbx.list_eegs:
            listwidget.addItem(str(eeg))

    def _enable_match_button_if_applicable(self) -> None:
        """Enable match button if within-study comparison is selected and files are available."""
        if (
            self.ui.compare_within_study_radio.isChecked()
            and hasattr(self.comet_tbx_study1, "list_eegs")
            and len(self.comet_tbx_study1.list_eegs) > 0
        ):
            self.ui.match_subjects_button.setEnabled(True)

    # ==================== UI STATE MANAGEMENT ====================

    def update_ui(self) -> None:
        """Update UI element states based on loaded studies and selected comparison mode."""
        self._clear_ui_elements()
        self._configure_comparison_mode()
        self._update_feature_controls()
        self._update_plot_button_text()

    def _clear_ui_elements(self) -> None:
        """Clear previous selections and texts."""
        self.ui.feature_combo.clear()
        self.ui.plot_label.clear()
        self.ui.stats_textedit.clear()
        # Reset checkboxes when UI elements are cleared
        self.ui.paired_test_checkbox.setChecked(False)
        self.ui.show_features_checkbox.setChecked(False)
        # Hide plots, labels, and widgets
        self._hide_features_plot()
        self.ui.plot_label.setVisible(False)
        self.ui.verticalLayoutWidget.setVisible(False)
        # Don't reset show_microstates_checkbox, but refresh the display and widget visibility
        if self.ui.show_microstates_checkbox.isChecked():
            self._show_all_microstates()
            self.ui.microstate_label.setVisible(True)
            self.ui.layoutWidget.setVisible(True)
        else:
            self._hide_all_microstates()
            self.ui.microstate_label.setVisible(False)
            self.ui.layoutWidget.setVisible(False)

    def _configure_comparison_mode(self) -> None:
        """Configure UI based on selected comparison mode."""
        if self.ui.compare_two_studies_radio.isChecked():
            self._configure_two_studies_mode()
        elif self.ui.compare_within_study_radio.isChecked():
            self._configure_within_study_mode()
        else:
            self._configure_synthetic_mode()

    def _configure_two_studies_mode(self) -> None:
        """Configure UI for two-studies comparison mode."""
        # Show Study 2 widgets
        widgets_to_show = [
            self.ui.load_study2_button,
            self.ui.study2_name_label,
            self.ui.study2_file_list,
            self.ui.match_subjects_button,
            self.canvas_microstates_study2,
        ]
        widgets_to_hide = [self.ui.study1_to_substudy1, self.ui.substudy1_to_study1]

        self._set_widget_visibility(widgets_to_show, True)
        self._set_widget_visibility(widgets_to_hide, False)

        # Enable Study 2 controls
        self.ui.load_study2_button.setEnabled(True)
        self.ui.study2_name_label.setEnabled(True)
        self.ui.study2_file_list.setEnabled(True)
        self.canvas_microstates_study2.setEnabled(True)

        # Set Study 2 name
        if self.study2_loaded:
            self.study2_name = self.comet_tbx_study2.study_name
            self.ui.study2_name_label.setText(self.study2_name)
        else:
            self.ui.study2_name_label.setText("Study 2")

    def _configure_within_study_mode(self) -> None:
        """Configure UI for within-study comparison mode."""
        # Show within-study widgets
        widgets_to_show = [
            self.ui.study1_to_substudy1,
            self.ui.substudy1_to_study1,
            self.ui.study2_name_label,
            self.ui.study2_file_list,
            self.ui.match_subjects_button,
        ]
        widgets_to_hide = [self.ui.load_study2_button, self.canvas_microstates_study2]

        self._set_widget_visibility(widgets_to_show, True)
        self._set_widget_visibility(widgets_to_hide, False)

        # Configure widget states
        self.ui.load_study2_button.setEnabled(False)
        self.ui.study2_name_label.setEnabled(True)
        self.ui.study2_file_list.setEnabled(True)
        self.canvas_microstates_study2.setEnabled(False)
        self.ui.plot_testretest_button.setEnabled(False)

        # Enable controls based on study1 status
        if (
            self.study1_loaded
            and hasattr(self.comet_tbx_study1, "list_eegs")
            and len(self.comet_tbx_study1.list_eegs) > 0
        ):
            self.ui.match_subjects_button.setEnabled(True)
        else:
            self.ui.match_subjects_button.setEnabled(False)

        self.ui.study1_to_substudy1.setEnabled(True)
        self.ui.substudy1_to_study1.setEnabled(True)

        # Set labels
        self.ui.study2_name_label.setText(self.STUDY_NAMES["within"])
        self.study2_name = "within_study"

    def _configure_synthetic_mode(self) -> None:
        """Configure UI for synthetic comparison mode."""
        # Hide Study 2 widgets
        widgets_to_hide = [
            self.ui.load_study2_button,
            self.ui.study2_name_label,
            self.ui.study2_file_list,
            self.ui.study1_to_substudy1,
            self.ui.substudy1_to_study1,
            self.ui.match_subjects_button,
            self.canvas_microstates_study2,
        ]

        self._set_widget_visibility(widgets_to_hide, False)

        # Disable controls
        self.ui.load_study2_button.setEnabled(False)
        self.ui.study2_name_label.setEnabled(True)
        self.ui.study2_file_list.setEnabled(False)
        self.canvas_microstates_study2.setEnabled(False)
        self.ui.plot_testretest_button.setEnabled(False)
        self.ui.match_subjects_button.setEnabled(False)
        self.ui.study1_to_substudy1.setEnabled(False)
        self.ui.substudy1_to_study1.setEnabled(False)

        # Set Study 2 name based on comparison type
        if self.ui.compare_surrogate_radio.isChecked():
            self.ui.study2_name_label.setText(self.STUDY_NAMES["surrogate"])
            self.synthetic_type = "surrogate"
        elif self.ui.compare_random_radio.isChecked():
            self.ui.study2_name_label.setText(self.STUDY_NAMES["random"])
            self.synthetic_type = "random"
        self.study2_name = self.synthetic_type

    def _set_widget_visibility(self, widgets: List, visible: bool) -> None:
        """Set visibility for a list of widgets.

        Args:
            widgets: List of widgets to modify.
            visible: Whether widgets should be visible.
        """
        for widget in widgets:
            widget.setVisible(visible)

    def _update_feature_controls(self) -> None:
        """Update feature-related controls based on loaded studies."""
        if not self.study1_loaded:
            self._disable_feature_buttons()
            self._update_plot_button_text()
            return

        # Display Study 1 information
        self.ui.study1_name_label.setText(self.comet_tbx_study1.study_name)
        self.feature_list_dictionary = self.comet_tbx_study1.feature_list_dictionary

        if self.ui.compare_two_studies_radio.isChecked():
            self._update_two_studies_features()
        elif self.ui.compare_within_study_radio.isChecked():
            self._update_within_study_features()
        else:
            self._update_synthetic_features()
            
        self._update_plot_button_text()

    def _update_two_studies_features(self) -> None:
        """Update features for two-studies comparison."""
        if self.study1_loaded and self.study2_loaded:
            set_widgets_status(self.ui.compare_features_button, mode="enable")

            # Check for test-retest capability
            if len(self.comet_tbx_study1.list_eegs) == len(
                self.comet_tbx_study2.list_eegs
            ):
                set_widgets_status(self.ui.plot_testretest_button, mode="enable")
            else:
                set_widgets_status(self.ui.plot_testretest_button, mode="disable")

            # Populate common features
            feature_type, feature_mode = self._get_compatible_features()
            if feature_type and feature_mode:
                self._populate_common_features()
            else:
                self._set_no_features_message()
        else:
            self._disable_feature_buttons()

    def _update_within_study_features(self) -> None:
        """Update features for within-study comparison."""
        feature_type, feature_mode = self._get_compatible_features()
        if feature_type and feature_mode:
            set_widgets_status(self.ui.compare_features_button, mode="enable")
            self._populate_study_features()
        else:
            self._set_no_features_message(self.ERROR_MESSAGES["no_within_features"])

    def _update_synthetic_features(self) -> None:
        """Update features for synthetic comparison."""
        feature_type, feature_mode = self._get_compatible_features()
        if feature_type and feature_mode:
            set_widgets_status(self.ui.compare_features_button, mode="enable")
            self._populate_study_features()
        else:
            self._set_no_features_message(self.ERROR_MESSAGES["no_synthetic"])

    def _populate_common_features(self) -> None:
        """Populate feature combo with features common to both studies."""
        common_features = [
            feat
            for feat in self.comet_tbx_study1.feature_list
            if feat in self.comet_tbx_study2.feature_list
        ]
        self._populate_feature_combo(common_features)

    def _populate_study_features(self) -> None:
        """Populate feature combo with features from study 1."""
        self._populate_feature_combo(self.comet_tbx_study1.feature_list)

    def _populate_feature_combo(self, features: List[str]) -> None:
        """Populate the feature combo box with full feature names.

        Args:
            features: List of feature codes.
        """
        self.ui.feature_combo.clear()
        full_feature_names = [
            self.feature_list_dictionary.get(feat, feat) for feat in features
        ]
        self.ui.feature_combo.addItems(full_feature_names)

    def _set_no_features_message(self, message: str = None) -> None:
        """Set a 'no features' message in the combo box.

        Args:
            message: Custom message to display.
        """
        self.ui.feature_combo.clear()
        message = message or self.ERROR_MESSAGES["no_features"]
        self.ui.feature_combo.addItem(message)
        set_widgets_status(self.ui.compare_features_button, mode="disable")

    def _disable_feature_buttons(self) -> None:
        """Disable all feature-related buttons."""
        set_widgets_status(self.ui.compare_features_button, mode="disable")
        set_widgets_status(self.ui.plot_testretest_button, mode="disable")
        
    def _update_plot_button_text(self) -> None:
        """Update the plot features button text based on selected comparison mode."""
        if self.ui.compare_two_studies_radio.isChecked() or self.ui.compare_within_study_radio.isChecked():
            # Check if both file lists have content before showing comparison text
            study1_files = self.ui.study1_file_list.count()
            study2_files = self.ui.study2_file_list.count()
            
            if study1_files > 0 and study2_files > 0:
                study1_name = self.ui.study1_name_label.text()
                study2_name = self.ui.study2_name_label.text()
                button_text = f"Statistical Analysis - {study1_name} vs. {study2_name}"
            else:
                button_text = "Statistical Analysis"
        elif self.ui.compare_surrogate_radio.isChecked():
            button_text = "Statistical Analysis - Surrogate Features"
        elif self.ui.compare_random_radio.isChecked():
            button_text = "Statistical Analysis - Random Features"
        else:
            button_text = "Statistical Analysis"
            
        self.ui.compare_features_button.setText(button_text)

    def _update_plot_label(self) -> None:
        """Update the plot label based on the currently selected feature."""
        selected_feature_full_name = self.ui.feature_combo.currentText()

        # Check if this is a placeholder/error message
        if (
            not selected_feature_full_name
            or selected_feature_full_name in self.ERROR_MESSAGES.values()
        ):
            self.ui.plot_label.clear()
            return

        self.ui.plot_label.setText(selected_feature_full_name)

    # ==================== FEATURE HANDLING ====================

    def _get_selected_feature_code(self) -> str:
        """Return the feature short code for the selected full name.

        Returns:
            Feature short code (e.g., "COV", "OCC"), or the full name if not found.
        """
        selected_full_name = self.ui.feature_combo.currentText()

        if hasattr(self, "feature_list_dictionary"):
            reverse_dict = {v: k for k, v in self.feature_list_dictionary.items()}
            return reverse_dict.get(selected_full_name, selected_full_name)

        return selected_full_name

    def _get_compatible_features(self) -> Tuple[Optional[str], Optional[str]]:
        """Return features that are compatible between the selected studies.

        Returns:
            Tuple of (feature_type, feature_mode) suitable for comparison,
            or (None, None) if not available.
        """
        if not self.study1_loaded:
            return None, None

        study1_features = self._discover_available_features(self.comet_tbx_study1)

        if self.ui.compare_two_studies_radio.isChecked():
            return self._get_two_studies_compatible_features(study1_features)
        elif self.ui.compare_within_study_radio.isChecked():
            return self._get_within_study_compatible_features(study1_features)
        else:
            return self._get_synthetic_compatible_features(study1_features)

    def _get_two_studies_compatible_features(
        self, study1_features: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get compatible features for two-studies comparison."""
        if not self.study2_loaded:
            return None, None

        study2_features = self._discover_available_features(self.comet_tbx_study2)
        common_types = set(study1_features.keys()) & set(study2_features.keys())

        if not common_types:
            return None, None

        feature_type = "real" if "real" in common_types else list(common_types)[0]
        common_modes = set(study1_features[feature_type]) & set(
            study2_features[feature_type]
        )

        if not common_modes:
            return None, None

        feature_mode = "static" if "static" in common_modes else list(common_modes)[0]
        return feature_type, feature_mode

    def _get_within_study_compatible_features(
        self, study1_features: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get compatible features for within-study comparison."""
        if "real" not in study1_features:
            return None, None

        available_modes = study1_features["real"]
        if not available_modes:
            return None, None

        feature_mode = "static" if "static" in available_modes else available_modes[0]
        return "real", feature_mode

    def _get_synthetic_compatible_features(
        self, study1_features: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get compatible features for synthetic comparison."""
        if "real" not in study1_features:
            return None, None

        if self.synthetic_type not in study1_features:
            return None, None

        common_modes = set(study1_features["real"]) & set(
            study1_features[self.synthetic_type]
        )
        if not common_modes:
            return None, None

        feature_mode = "static" if "static" in common_modes else list(common_modes)[0]
        return "real", feature_mode

    @staticmethod
    def _discover_available_features(tbx: COMET) -> Dict[str, List[str]]:
        """Discover what feature files are actually available in a study.

        Args:
            tbx: The COMET toolbox object for the study.

        Returns:
            Dictionary with feature types as keys and lists of available modes as values.
        """
        available_features = {}

        if not hasattr(tbx, "extracted_features_path") or not os.path.exists(
            tbx.extracted_features_path
        ):
            return available_features

        for feature_type in CompareStudiesWindow.FEATURE_TYPES:
            available_modes = []
            for feature_mode in CompareStudiesWindow.FEATURE_MODES:
                for export_format in CompareStudiesWindow.EXPORT_FORMATS:
                    feature_filename = (
                        f"{feature_type}_{feature_mode}_features{export_format}"
                    )
                    feature_path = os.path.join(
                        tbx.extracted_features_path, feature_filename
                    )
                    if (
                        os.path.exists(feature_path)
                        and feature_mode not in available_modes
                    ):
                        available_modes.append(feature_mode)

            if available_modes:
                available_features[feature_type] = available_modes

        return available_features

    @staticmethod
    def _load_features_safely(
        tbx: COMET, feature_type: str, feature_mode: str
    ) -> Optional[pd.DataFrame]:
        """Safely load features with proper error handling.

        Args:
            tbx: COMET toolbox object.
            feature_type: Type of features ('real', 'surrogate', 'random').
            feature_mode: Mode ('static', 'dynamic', 'averaged', 'sliding').

        Returns:
            Loaded features or None if loading fails.
        """
        if not hasattr(tbx, "extracted_features_path") or not os.path.exists(
            tbx.extracted_features_path
        ):
            return None

        for export_format in CompareStudiesWindow.EXPORT_FORMATS:
            feature_filename = f"{feature_type}_{feature_mode}_features{export_format}"
            feature_path = os.path.join(tbx.extracted_features_path, feature_filename)

            if os.path.exists(feature_path):
                try:
                    return FeatureIO().import_features(feature_path, export_format)
                except Exception as e:
                    print(f"Error loading {feature_path}: {e}")
                    continue

        return None

    def _get_common_features(self) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        """Return a DataFrame of common features and their columns.

        Returns:
            Tuple of (combined DataFrame, common columns) or (None, None) on failure.
        """
        feature_type, feature_mode = self._get_compatible_features()

        if feature_type is None or feature_mode is None:
            self._show_feature_loading_error()
            return None, None

        # Load features
        features_df_study1 = self._load_features_safely(
            self.comet_tbx_study1, feature_type, feature_mode
        )
        if features_df_study1 is None:
            self._show_error(
                "Feature Loading Error",
                f"Could not load {feature_type} {feature_mode} features from Study 1.",
            )
            return None, None

        features_df_study2 = self._get_study2_features(
            feature_type, feature_mode, features_df_study1
        )
        if features_df_study2 is None:
            return None, None

        # Filter features based on files currently shown in the UI lists
        features_df_study1_filtered = self._filter_features_by_filelist(
            features_df_study1, self.ui.study1_file_list
        )
        features_df_study2_filtered = self._filter_features_by_filelist(
            features_df_study2, self.ui.study2_file_list
        )

        return self._combine_feature_dataframes(features_df_study1_filtered, features_df_study2_filtered)

    def _filter_features_by_filelist(
        self, features_df: pd.DataFrame, file_list_widget
    ) -> pd.DataFrame:
        """Filter features DataFrame to include only files currently shown in the file list.
        
        Args:
            features_df: The full features DataFrame.
            file_list_widget: QListWidget containing the files to include.
            
        Returns:
            Filtered DataFrame containing only the specified files.
        """
        if features_df is None or features_df.empty:
            return features_df
            
        # Get list of filenames from the UI widget
        filenames_in_list = []
        for i in range(file_list_widget.count()):
            item = file_list_widget.item(i)
            if item is not None:
                filenames_in_list.append(item.text())
                
        if not filenames_in_list:
            return features_df
            
        # Filter the dataframe to include only these files
        if "Filename" in features_df.columns:
            filtered_df = features_df[features_df["Filename"].isin(filenames_in_list)].copy()
        else:
            # If no Filename column, return the original dataframe
            filtered_df = features_df.copy()
            
        return filtered_df

    def _get_study2_features(
        self, feature_type: str, feature_mode: str, features_df_study1: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Get features for study 2 based on comparison mode."""
        if self.ui.compare_two_studies_radio.isChecked():
            features_df = self._load_features_safely(
                self.comet_tbx_study2, feature_type, feature_mode
            )
            if features_df is None:
                self._show_error(
                    "Feature Loading Error",
                    f"Could not load {feature_type} {feature_mode} features from "
                    "Study 2.",
                )
            return features_df

        elif self.ui.compare_within_study_radio.isChecked():
            return features_df_study1.copy()

        else:  # Synthetic comparison
            features_df = self._load_features_safely(
                self.comet_tbx_study1, self.synthetic_type, feature_mode
            )
            if features_df is None:
                self._show_error(
                    "Feature Loading Error",
                    f"Could not load {self.synthetic_type} {feature_mode} features "
                    "from Study 1.",
                )
            return features_df

    def _combine_feature_dataframes(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        """Combine two feature DataFrames for comparison."""
        # Identify common columns
        common_columns = sorted([col for col in df1.columns if col in df2.columns])

        if not common_columns:
            self._show_error(
                "Feature Comparison Error",
                "No common feature columns found between the datasets.",
            )
            return None, None

        # Prepare DataFrames with Study information
        study1_df = df1[common_columns].copy()
        study1_df["Study"] = self.comet_tbx_study1.study_name

        study2_df = df2[common_columns].copy()
        study2_df["Study"] = self.study2_name

        # Concatenate DataFrames
        common_features_df = pd.concat([study1_df, study2_df], ignore_index=True)
        return common_features_df, common_columns

    def _show_feature_loading_error(self) -> None:
        """Show detailed error message for feature loading failures."""
        study1_features = self._discover_available_features(self.comet_tbx_study1)
        error_msg = "No compatible features found between studies.\n\n"
        error_msg += f"Study 1 available features: {study1_features}\n"

        if (
            self.ui.compare_two_studies_radio.isChecked()
            and self.study2_loaded
        ):
            study2_features = self._discover_available_features(self.comet_tbx_study2)
            error_msg += f"Study 2 available features: {study2_features}\n"

        if self.ui.compare_within_study_radio.isChecked():
            error_msg += (
                "\nNote: Within-study comparison selected. "
                "Only Study 1 features are needed."
            )

        self._show_error(
            "Feature Loading Error",
            error_msg + "\nPlease ensure the study has completed feature extraction.",
        )

    # ==================== PLOTTING AND VISUALIZATION ====================

    def _plot_microstate_maps(
        self, tbx: COMET, figure: Figure, canvas: FigureCanvasQTAgg
    ) -> None:
        """Plot microstate maps for a study on the provided canvas.

        Args:
            tbx: COMET study instance providing maps, labels and EEG info.
            figure: Target figure.
            canvas: Target canvas.
        """
        # Clear existing axes
        for ax in figure.get_axes():
            self._clear_axis(ax)

        # Check if microstate maps are available
        if not self._has_microstate_maps(tbx):
            self._plot_no_maps_message(figure, canvas)
            return

        # Create subplots for each microstate
        axs = [
            figure.add_subplot(1, len(tbx.micro_labels), idx + 1)
            for idx in range(len(tbx.micro_labels))
        ]

        # Plot each microstate with its label
        for idx, ax in enumerate(axs):
            self._plot_single_microstate(
                tbx.best_maps[idx, :], tbx.micro_labels[idx], tbx.eeg_info, ax
            )

        figure.tight_layout()
        canvas.draw()

    @staticmethod
    def _clear_axis(ax) -> None:
        """Clear and reset axis properties."""
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    @staticmethod
    def _has_microstate_maps(tbx: COMET) -> bool:
        """Check if microstate maps are available."""
        return (
            hasattr(tbx, "best_maps")
            and tbx.best_maps is not None
            and hasattr(tbx, "micro_labels")
            and tbx.micro_labels
        )

    def _plot_no_maps_message(
        self, figure: Figure, canvas: FigureCanvasQTAgg
    ) -> None:
        """Plot a message when no microstate maps are available."""
        ax = figure.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            "No microstate maps available\nClustering not completed yet",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        figure.tight_layout()
        canvas.draw()

    @staticmethod
    def _plot_single_microstate(
        microstate: np.ndarray, micro_label: str, eeg_info, ax
    ) -> None:
        """Plot a microstate topomap and its label on the provided axis.

        Args:
            microstate: Map values.
            micro_label: Label to display below the map.
            eeg_info: EEG info structure (e.g., MNE info) for plotting.
            ax: Target axes.
        """
        show_microstate(microstate, eeg_info, ax, log_callback=None)  # No logging in comparison view
        ax.axis("off")
        ax.text(
            0.5,
            -0.2,
            micro_label.upper(),
            transform=ax.transAxes,
            fontsize=16,
            ha="center",
            va="center",
        )
        ax.text(0, 0, "", transform=ax.transAxes)

    def plot_features(self) -> None:
        """Plot a violin plot (with swarm overlay) for the selected feature."""
        selected_feature = self._get_selected_feature_code()

        try:
            result = self._get_common_features()
            if result is None or result[0] is None:
                return

            common_features_df, _ = result
            plot_data, feature_list = self._organize_data_for_plotting(
                common_features_df, selected_feature
            )

            ax = self.canvas_features.figure.gca()
            self._prepare_plot_axis(ax)
            self._create_violin_plot(ax, plot_data, selected_feature)
            self._set_plot_labels_and_ticks(ax, feature_list, selected_feature)
            self.canvas_features.draw()

        except Exception as e:
            self._show_error("Plotting Error", f"Failed to plot features: {str(e)}")

    @staticmethod
    def _organize_data_for_plotting(
        common_features_df: pd.DataFrame, selected_feature: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Organize a combined DataFrame for plotting a specific feature.

        Args:
            common_features_df: Combined features from both studies.
            selected_feature: Short code of the selected feature.

        Returns:
            Tuple of (melted plot data, feature column list).
        """
        columns_to_keep = ["Filename", "Study"] + [
            col for col in common_features_df.columns if col.startswith(selected_feature)
        ]
        filtered_df = common_features_df[columns_to_keep]
        feature_list = [
            col for col in filtered_df.columns if col not in ["Study", "Filename"]
        ]

        plot_data = pd.melt(
            filtered_df.reset_index(),
            id_vars=["Filename", "Study"],
            value_vars=feature_list,
        )
        plot_data.columns = ["Filename", "Study", "Feature", selected_feature]
        return plot_data, feature_list

    @staticmethod
    def _prepare_plot_axis(ax) -> None:
        """Clear axes and apply default font sizes."""
        ax.clear()
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(18)

    def _create_violin_plot(
        self, ax, plot_data: pd.DataFrame, selected_feature: str
    ) -> None:
        """Create violin plot with swarm overlay."""
        # Create violin plot
        sns.violinplot(
            x="Feature",
            y=selected_feature,
            hue="Study",
            data=plot_data,
            ax=ax,
            hue_order=[self.comet_tbx_study1.study_name, self.study2_name],
        )

        # Overlay swarm plot for individual data points
        sns.swarmplot(
            x="Feature",
            y=selected_feature,
            hue="Study",
            data=plot_data,
            ax=ax,
            color="white",
            size=10,
            marker="o",
            dodge=True,
            legend=False,
        )

    def _set_plot_labels_and_ticks(
        self, ax, filter_cols: List[str], feature: str
    ) -> None:
        """Set labels and tick marks for the current feature plot."""
        ax.set_xlabel("Microstate")
        xticklabels = ["_".join(col.split("_")[1:]) for col in filter_cols]
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(self.feature_list_dictionary[feature])
        self.ui.plot_label.setText(f"{self.feature_list_dictionary[feature]}")

    def _on_feature_selection_changed(self) -> None:
        """Handle feature selection change - auto-plot if checkbox is checked."""
        if self.ui.show_features_checkbox.isChecked():
            self._auto_plot_features()

    def _on_show_features_toggled(self, checked: bool) -> None:
        """Handle show features checkbox toggle."""
        if checked:
            self._auto_plot_features()
            self.ui.plot_label.setVisible(True)
            self.ui.verticalLayoutWidget.setVisible(True)
        else:
            self._hide_features_plot()
            self.ui.plot_label.setVisible(False)
            self.ui.verticalLayoutWidget.setVisible(False)

    def _auto_plot_features(self) -> None:
        """Automatically plot features if conditions are met."""
        # Check if we have the necessary conditions for plotting
        if (self.study1_loaded and 
            self.ui.feature_combo.currentText() and 
            self.ui.feature_combo.currentText() not in self.ERROR_MESSAGES.values()):
            try:
                self.plot_features()
                # Show the plot label and widget when plotting succeeds
                self.ui.plot_label.setVisible(True)
                self.ui.verticalLayoutWidget.setVisible(True)
            except Exception as e:
                # If plotting fails, don't show error to user since this is auto-plotting
                print(f"Auto-plotting failed: {e}")

    def _hide_features_plot(self) -> None:
        """Hide the features plot by clearing the figure."""
        self.figure_features.clear()
        self.canvas_features.draw()

    def _on_show_microstates_toggled(self, checked: bool) -> None:
        """Handle show microstates checkbox toggle."""
        if checked:
            self._show_all_microstates()
            self.ui.microstate_label.setVisible(True)
            self.ui.layoutWidget.setVisible(True)
        else:
            self._hide_all_microstates()
            self.ui.microstate_label.setVisible(False)
            self.ui.layoutWidget.setVisible(False)

    def _show_all_microstates(self) -> None:
        """Show microstate maps for all loaded studies."""
        if self.study1_loaded:
            try:
                self._plot_microstate_maps(
                    self.comet_tbx_study1, 
                    self.figure_microstates_study1, 
                    self.canvas_microstates_study1
                )
            except Exception as e:
                print(f"Failed to plot microstates for study 1: {e}")
        
        # Only show study2 microstates for two-studies comparison
        if (self.study2_loaded and 
            self.ui.compare_two_studies_radio.isChecked()):
            try:
                self._plot_microstate_maps(
                    self.comet_tbx_study2, 
                    self.figure_microstates_study2, 
                    self.canvas_microstates_study2
                )
            except Exception as e:
                print(f"Failed to plot microstates for study 2: {e}")

    def _hide_all_microstates(self) -> None:
        """Hide all microstate maps."""
        self._hide_microstate_maps(self.figure_microstates_study1, self.canvas_microstates_study1)
        self._hide_microstate_maps(self.figure_microstates_study2, self.canvas_microstates_study2)

    def _hide_microstate_maps(self, figure: Figure, canvas: FigureCanvasQTAgg) -> None:
        """Hide microstate maps by clearing the figure."""
        figure.clear()
        canvas.draw()

    # ==================== STATISTICAL ANALYSIS ====================

    def update_feature_stats(self) -> None:
        """Update feature comparison t-test statistics in the UI."""
        self.ui.stats_textedit.clear()
        
        # Get analysis information
        (
            feature_list,
            t_test_results,
            p_values,
            adjusted_p_values,
        ) = self._perform_feature_comparison_analysis()

        if not feature_list:
            self.ui.stats_textedit.appendPlainText(
                "No features available for statistical comparison."
            )
            return

        # Display header information
        self._display_analysis_header()
        
        # Display main results table
        self._display_statistical_results_table(
            feature_list, t_test_results, p_values, adjusted_p_values
        )
        
        # Display summary statistics
        self._display_statistical_summary(p_values, adjusted_p_values)

    def _display_analysis_header(self) -> None:
        """Display header information for the statistical analysis."""
        # Get comparison information
        comparison_info = self._get_comparison_info()
        selected_feature = self._get_selected_feature_code()
        feature_full_name = self.feature_list_dictionary.get(selected_feature, selected_feature)
        
        # Analysis header
        header_text = (
            f"Statistical Analysis Report\n"
            f"{'=' * 60}\n"
            f"Feature: {feature_full_name}\n"
            f"Comparison: {comparison_info['comparison_type']}\n"
            f"Study 1: {comparison_info['study1_name']} ({comparison_info['study1_files']} files)\n"
            f"Study 2: {comparison_info['study2_name']} ({comparison_info['study2_files']} files)\n"
            f"Test Type: {'Paired' if self.ui.paired_test_checkbox.isChecked() else 'Independent'} t-test\n"
            f"Multiple Testing Correction: {self.ui.multiple_test_method_combo.currentText()}\n"
            f"{'=' * 60}\n"
        )
        self.ui.stats_textedit.appendPlainText(header_text)

    def _display_statistical_results_table(
        self, 
        feature_list: List[str], 
        t_test_results: Dict[str, float], 
        p_values: List[float], 
        adjusted_p_values: List[float]
    ) -> None:
        """Display the main statistical results in a formatted table."""
        # Table header
        self.ui.stats_textedit.appendPlainText("Statistical Results:")
        self.ui.stats_textedit.appendPlainText(
            f"{'Microstate Feature':<35} {'t-statistic':>12} {'p-value':>12} {'adj. p-value':>12} {'Significance':>12}"
        )
        self.ui.stats_textedit.appendPlainText("-" * 85)
        
        # Display results for each feature
        for i, feat in enumerate(feature_list):
            t_statistic = t_test_results[feat]
            p_value = p_values[i]
            adjusted_p_value = adjusted_p_values[i]
            
            # Determine significance
            significance = self._get_significance_level(adjusted_p_value)
            
            # Format feature name (remove feature code prefix for cleaner display)
            display_name = "_".join(feat.split("_")[1:]) if "_" in feat else feat
            
            # Format p-values with scientific notation for very small values
            p_value_str = f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}"
            adj_p_value_str = f"{adjusted_p_value:.2e}" if adjusted_p_value < 0.001 else f"{adjusted_p_value:.4f}"
            
            result_line = (
                f"{display_name:<35} {t_statistic:>12.4f} {p_value_str:>12} "
                f"{adj_p_value_str:>12} {significance:>12}"
            )
            self.ui.stats_textedit.appendPlainText(result_line)
        
        self.ui.stats_textedit.appendPlainText("")  # Add spacing

    def _display_statistical_summary(self, p_values: List[float], adjusted_p_values: List[float]) -> None:
        """Display intuitive summary of significant differences."""
        if not p_values:
            return
            
        # Get the analysis results for detailed interpretation
        result = self._get_common_features()
        if result is None or result[0] is None:
            return
            
        common_features_df, _ = result
        selected_feature = self._get_selected_feature_code()
        feature_full_name = self.feature_list_dictionary.get(selected_feature, selected_feature)
        
        # Organize data for interpretation
        study1_df, study2_df = self._separate_studies_data(common_features_df, selected_feature)
        feature_list = [col for col in study1_df.columns if col not in ["Study", "Filename"]]
        
        comparison_info = self._get_comparison_info()
        study1_name = comparison_info['study1_name']
        study2_name = comparison_info['study2_name']
        
        # Generate intuitive summary
        self.ui.stats_textedit.appendPlainText("Interpretation Summary:")
        self.ui.stats_textedit.appendPlainText("-" * 60)
        
        significant_findings = []
        non_significant_findings = []
        
        for i, feat in enumerate(feature_list):
            adjusted_p_value = adjusted_p_values[i]
            
            if adjusted_p_value < 0.05:  # Significant
                # Calculate means for interpretation
                study1_mean = study1_df[feat].mean()
                study2_mean = study2_df[feat].mean()
                
                # Extract microstate from feature name
                microstate = "_".join(feat.split("_")[1:]) if "_" in feat else feat
                
                # Determine direction of effect
                if study2_mean > study1_mean:
                    direction = "significantly greater in"
                    higher_study = study2_name
                    lower_study = study1_name
                    higher_mean = study2_mean
                    lower_mean = study1_mean
                else:
                    direction = "significantly greater in"
                    higher_study = study1_name
                    lower_study = study2_name
                    higher_mean = study1_mean
                    lower_mean = study2_mean
                
                # Format p-value
                p_value_str = f"{adjusted_p_value:.2e}" if adjusted_p_value < 0.001 else f"{adjusted_p_value:.4f}"
                
                # Create interpretation text
                finding = (
                    f"• {feature_full_name} of Microstate {microstate} is {direction} "
                    f"{higher_study} compared to {lower_study}\n"
                    f"  ({higher_study}: {higher_mean:.4f}, {lower_study}: {lower_mean:.4f}, "
                    f"adj. p = {p_value_str})"
                )
                significant_findings.append(finding)
            else:
                microstate = "_".join(feat.split("_")[1:]) if "_" in feat else feat
                p_value_str = f"{adjusted_p_value:.2e}" if adjusted_p_value < 0.001 else f"{adjusted_p_value:.4f}"
                
                finding = (
                    f"• {feature_full_name} of Microstate {microstate}: "
                    f"No significant difference (adj. p = {p_value_str})"
                )
                non_significant_findings.append(finding)
        
        # Display significant findings first
        if significant_findings:
            self.ui.stats_textedit.appendPlainText("Significant Differences:")
            for finding in significant_findings:
                self.ui.stats_textedit.appendPlainText(finding)
                self.ui.stats_textedit.appendPlainText("")
        
        # Display non-significant findings
        if non_significant_findings:
            if significant_findings:
                self.ui.stats_textedit.appendPlainText("Non-Significant Differences:")
            for finding in non_significant_findings:
                self.ui.stats_textedit.appendPlainText(finding)
        
        # Add overall summary
        n_significant = len(significant_findings)
        n_total = len(feature_list)
        
        self.ui.stats_textedit.appendPlainText("")
        self.ui.stats_textedit.appendPlainText(f"Overall: {n_significant}/{n_total} microstate features show significant differences")
        self.ui.stats_textedit.appendPlainText("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")

    def _get_comparison_info(self) -> Dict[str, str]:
        """Get information about the current comparison for display."""
        study1_name = self.ui.study1_name_label.text()
        study2_name = self.ui.study2_name_label.text()
        study1_files = self.ui.study1_file_list.count()
        study2_files = self.ui.study2_file_list.count()
        
        if self.ui.compare_two_studies_radio.isChecked():
            comparison_type = "Two Independent Studies"
        elif self.ui.compare_within_study_radio.isChecked():
            comparison_type = "Within-Study Comparison"
        elif self.ui.compare_surrogate_radio.isChecked():
            comparison_type = "Real vs. Surrogate Data"
        elif self.ui.compare_random_radio.isChecked():
            comparison_type = "Real vs. Random Data"
        else:
            comparison_type = "Unknown"
            
        return {
            "comparison_type": comparison_type,
            "study1_name": study1_name,
            "study2_name": study2_name,
            "study1_files": study1_files,
            "study2_files": study2_files
        }

    def _get_significance_level(self, p_value: float) -> str:
        """Return significance level string based on p-value."""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"

    def _perform_feature_comparison_analysis(
        self,
    ) -> Tuple[List[str], Dict[str, float], List[float], List[float]]:
        """Perform t-test-based feature comparison between two studies.

        Returns:
            Tuple of (feature_list, t_test_results, p_values, adjusted_p_values).
        """
        result = self._get_common_features()
        if result is None or result[0] is None:
            return [], {}, [], []

        common_features_df, _ = result
        selected_feature = self._get_selected_feature_code()

        # Organize data for the selected feature
        study1_df, study2_df = self._separate_studies_data(
            common_features_df, selected_feature
        )

        # Perform t-tests
        feature_list = [
            col for col in study1_df.columns if col not in ["Study", "Filename"]
        ]
        t_test_results, p_values = self._calculate_t_tests(
            study1_df, study2_df, feature_list
        )

        # Adjust p-values for multiple testing
        adjusted_p_values = multipletests(
            p_values, method=self.ui.multiple_test_method_combo.currentText().lower()
        )[1]

        return feature_list, t_test_results, p_values, adjusted_p_values

    def _separate_studies_data(
        self, common_features_df: pd.DataFrame, selected_feature: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate combined features DataFrame by study."""
        columns_to_keep = ["Filename", "Study"] + [
            col for col in common_features_df.columns if col.startswith(selected_feature)
        ]
        filtered_df = common_features_df[columns_to_keep]

        study1_df = filtered_df[
            filtered_df["Study"] == self.comet_tbx_study1.study_name
        ].sort_values(by="Filename")
        study2_df = filtered_df[
            filtered_df["Study"] == self.study2_name
        ].sort_values(by="Filename")

        return study1_df, study2_df

    def _calculate_t_tests(
        self,
        study1_df: pd.DataFrame,
        study2_df: pd.DataFrame,
        feature_list: List[str],
    ) -> Tuple[Dict[str, float], List[float]]:
        """Calculate t-tests for each feature."""
        t_test_results = {}
        p_values = []

        for feat in feature_list:
            study1_values = study1_df[feat].tolist()
            study2_values = study2_df[feat].tolist()

            if self.ui.paired_test_checkbox.isChecked():
                t_statistic, p_value = ttest_rel(study1_values, study2_values)
            else:
                t_statistic, p_value = ttest_ind(study1_values, study2_values)

            t_test_results[feat] = t_statistic
            p_values.append(p_value)

        return t_test_results, p_values

    # ==================== TEST-RETEST ANALYSIS ====================

    def perform_test_retest_analysis(self) -> None:
        """Perform test-retest ICC analysis for the selected feature."""
        try:
            self.ui.stats_textedit.clear()

            # Load and validate features
            features_study1, features_study2 = self._load_test_retest_features()
            if features_study1 is None or features_study2 is None:
                return

            # Match subjects and perform analysis
            common_subjects = self._match_test_retest_subjects(
                features_study1, features_study2
            )
            if not common_subjects:
                return

            self._perform_icc_analysis(features_study1, features_study2, common_subjects)

        except Exception as e:
            self._show_error(
                "Test-Retest Analysis Error",
                f"Failed to perform test-retest analysis: {str(e)}",
            )
            traceback.print_exc()

    def _load_test_retest_features(
        self,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load features for test-retest analysis."""
        feature_type, feature_mode = self._get_compatible_features()

        if feature_type is None or feature_mode is None:
            self.ui.stats_textedit.appendPlainText(
                "No compatible features found for test-retest analysis.\n"
                "Please ensure both studies have completed feature extraction."
            )
            return None, None

        features_study1 = self._load_features_safely(
            self.comet_tbx_study1, feature_type, feature_mode
        )
        features_study2 = self._load_features_safely(
            self.comet_tbx_study2, feature_type, feature_mode
        )

        if features_study1 is None or features_study2 is None:
            self.ui.stats_textedit.appendPlainText(
                "Failed to load features from one or both studies."
            )
            return None, None

        # Filter features based on files currently shown in the UI lists
        features_study1_filtered = self._filter_features_by_filelist(
            features_study1, self.ui.study1_file_list
        )
        features_study2_filtered = self._filter_features_by_filelist(
            features_study2, self.ui.study2_file_list
        )

        return features_study1_filtered, features_study2_filtered

    def _match_test_retest_subjects(
        self, features_study1: pd.DataFrame, features_study2: pd.DataFrame
    ) -> List[str]:
        """Match subjects between two studies for test-retest analysis."""
        # Extract subject IDs
        study1_subjects = {
            self._extract_subject_id(row["Filename"]): idx
            for idx, row in features_study1.iterrows()
        }
        study2_subjects = {
            self._extract_subject_id(row["Filename"]): idx
            for idx, row in features_study2.iterrows()
        }

        # Find common subjects
        common_subjects = sorted(
            list(set(study1_subjects.keys()) & set(study2_subjects.keys()))
        )

        if len(common_subjects) == 0:
            self.ui.stats_textedit.appendPlainText(
                "No matching subjects found between studies.\n"
                "Please ensure both studies contain data from the same subjects.\n"
                f"Study 1 subjects: {list(study1_subjects.keys())[:5]}...\n"
                f"Study 2 subjects: {list(study2_subjects.keys())[:5]}..."
            )
            return []

        return common_subjects

    def _perform_icc_analysis(
        self,
        features_study1: pd.DataFrame,
        features_study2: pd.DataFrame,
        common_subjects: List[str],
    ) -> None:
        """Perform ICC analysis for matched subjects."""
        feature_type, feature_mode = self._get_compatible_features()
        selected_feature = self._get_selected_feature_code()

        # Print analysis header
        self._print_icc_header(len(common_subjects), feature_type, feature_mode)

        # Get feature columns for selected feature
        feature_columns = [
            col
            for col in features_study1.columns
            if col not in ["Filename", "Study"] and col.startswith(selected_feature)
        ]

        if not feature_columns:
            self.ui.stats_textedit.appendPlainText(
                f"No columns found for selected feature: {selected_feature}"
            )
            return

        # Calculate ICC for each sub-feature
        icc_results = self._calculate_icc_results(
            features_study1, features_study2, common_subjects, feature_columns
        )

        # Display and summarize results
        self._display_icc_results(icc_results)
        self._summarize_icc_results(icc_results)

    def _print_icc_header(
        self, num_subjects: int, feature_type: str, feature_mode: str
    ) -> None:
        """Print header information for ICC analysis."""
        header_text = (
            f"Test-Retest Reliability Analysis (ICC)\n"
            f"{'=' * 60}\n"
            f"Number of matched subjects: {num_subjects}\n"
            f"Feature type: {feature_type}\n"
            f"Feature mode: {feature_mode}\n"
            f"{'=' * 60}\n"
        )
        self.ui.stats_textedit.appendPlainText(header_text)

    def _calculate_icc_results(
        self,
        features_study1: pd.DataFrame,
        features_study2: pd.DataFrame,
        common_subjects: List[str],
        feature_columns: List[str],
    ) -> List[Dict]:
        """Calculate ICC results for all feature columns."""
        # Create subject ID mappings
        study1_subjects = {
            self._extract_subject_id(row["Filename"]): idx
            for idx, row in features_study1.iterrows()
        }
        study2_subjects = {
            self._extract_subject_id(row["Filename"]): idx
            for idx, row in features_study2.iterrows()
        }

        icc_results = []

        for feature in feature_columns:
            # Create paired data matrix
            ratings = np.zeros((len(common_subjects), 2))

            for i, subject_id in enumerate(common_subjects):
                idx1 = study1_subjects[subject_id]
                idx2 = study2_subjects[subject_id]
                ratings[i, 0] = features_study1.loc[idx1, feature]
                ratings[i, 1] = features_study2.loc[idx2, feature]

            # Calculate ICC
            if np.any(np.isnan(ratings)):
                icc_value, ci_low, ci_high = np.nan, np.nan, np.nan
            else:
                icc_value, ci_low, ci_high = self._calculate_icc(ratings)

            # Get full feature name
            feature_full_name = self._get_full_feature_name(feature)

            icc_results.append(
                {
                    "Feature": feature,
                    "Feature_Full": feature_full_name,
                    "ICC": icc_value,
                    "CI_Low": ci_low,
                    "CI_High": ci_high,
                }
            )

        # Sort results by ICC value (descending)
        icc_results.sort(
            key=lambda x: x["ICC"] if not np.isnan(x["ICC"]) else -1, reverse=True
        )
        return icc_results

    def _get_full_feature_name(self, feature: str) -> str:
        """Get the full feature name from the feature code."""
        feature_parts = feature.split("_")
        if feature_parts[0] in self.feature_list_dictionary:
            feature_full_name = self.feature_list_dictionary[feature_parts[0]]
            if len(feature_parts) > 1:
                feature_full_name += f" ({'_'.join(feature_parts[1:])})"
        else:
            feature_full_name = feature
        return feature_full_name

    def _display_icc_results(self, icc_results: List[Dict]) -> None:
        """Display ICC results in formatted table."""
        self.ui.stats_textedit.appendPlainText(
            f"{'Feature':<40} {'ICC':>8} {'95% CI':>20}\n" + f"{'-' * 70}"
        )

        for result in icc_results:
            if not np.isnan(result["ICC"]):
                reliability = self._interpret_icc(result["ICC"])
                ci_str = f"[{result['CI_Low']:.3f}, {result['CI_High']:.3f}]"
                self.ui.stats_textedit.appendPlainText(
                    f"{result['Feature_Full']:<40} {result['ICC']:>8.3f} "
                    f"{ci_str:>20} {reliability}"
                )
            else:
                self.ui.stats_textedit.appendPlainText(
                    f"{result['Feature_Full']:<40} {'N/A':>8} {'N/A':>20}"
                )

    def _summarize_icc_results(self, icc_results: List[Dict]) -> None:
        """Display summary statistics for ICC results."""
        valid_iccs = [r["ICC"] for r in icc_results if not np.isnan(r["ICC"])]
        if not valid_iccs:
            return

        summary_text = (
            f"\n{'=' * 60}\n"
            f"Summary Statistics (Selected Feature):\n"
            f"Mean ICC: {np.mean(valid_iccs):.3f}\n"
            f"Median ICC: {np.median(valid_iccs):.3f}\n"
            f"Min ICC: {np.min(valid_iccs):.3f}\n"
            f"Max ICC: {np.max(valid_iccs):.3f}\n"
            f"Features with excellent reliability (ICC > 0.75): "
            f"{sum(1 for icc in valid_iccs if icc > 0.75)}/{len(valid_iccs)}\n"
            f"Features with good reliability (ICC > 0.60): "
            f"{sum(1 for icc in valid_iccs if icc > 0.60)}/{len(valid_iccs)}"
        )
        self.ui.stats_textedit.appendPlainText(summary_text)

    @staticmethod
    def _calculate_icc(
        ratings: np.ndarray, icc_type: str = "ICC(2,1)"
    ) -> Tuple[float, float, float]:
        """Calculate the Intraclass Correlation Coefficient (ICC).

        Args:
            ratings: Array of shape (n_subjects, n_raters); raters=2 for test-retest.
            icc_type: ICC type (e.g., 'ICC(2,1)') used for description only.

        Returns:
            Tuple of (icc_value, ci_low, ci_high) for 95% CI.
        """
        n_subjects, n_raters = ratings.shape

        # Calculate mean squares
        row_means = np.mean(ratings, axis=1)
        col_means = np.mean(ratings, axis=0)
        grand_mean = np.mean(ratings)

        # Sum of squares
        ss_total = np.sum((ratings - grand_mean) ** 2)
        ss_between_rows = n_raters * np.sum((row_means - grand_mean) ** 2)
        ss_between_cols = n_subjects * np.sum((col_means - grand_mean) ** 2)
        ss_error = ss_total - ss_between_rows - ss_between_cols

        # Degrees of freedom
        df_rows = n_subjects - 1
        df_cols = n_raters - 1
        df_error = df_rows * df_cols

        # Mean squares
        ms_rows = ss_between_rows / df_rows
        ms_cols = ss_between_cols / df_cols
        ms_error = ss_error / df_error if df_error > 0 else 0

        # ICC(2,1) calculation
        if ms_error == 0:
            icc_value = 1.0 if ms_rows == ms_error else 0.0
        else:
            icc_value = (ms_rows - ms_error) / (
                ms_rows
                + (n_raters - 1) * ms_error
                + n_raters * (ms_cols - ms_error) / n_subjects
            )

        # Calculate 95% confidence interval
        ci_low, ci_high = CompareStudiesWindow._calculate_icc_confidence_interval(
            ms_rows, ms_error, df_rows, df_error, n_raters, icc_value
        )

        return icc_value, ci_low, ci_high

    @staticmethod
    def _calculate_icc_confidence_interval(
        ms_rows: float,
        ms_error: float,
        df_rows: int,
        df_error: int,
        n_raters: int,
        icc_value: float,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for ICC."""
        alpha = 0.05

        if ms_error > 0:
            f_stat = ms_rows / ms_error
            f_low = f.ppf(alpha / 2, df_rows, df_error)
            f_high = f.ppf(1 - alpha / 2, df_rows, df_error)

            fl = f_stat / f_high
            fu = f_stat / f_low

            ci_low = (fl - 1) / (fl + n_raters - 1)
            ci_high = (fu - 1) / (fu + n_raters - 1)

            # Ensure bounds are within [0, 1]
            ci_low = max(0, min(1, ci_low))
            ci_high = max(0, min(1, ci_high))
        else:
            ci_low = ci_high = icc_value

        return ci_low, ci_high

    @staticmethod
    def _extract_subject_id(filename: str) -> str:
        """Extract a subject identifier from a filename using common patterns.

        Args:
            filename: Source filename.

        Returns:
            Extracted subject identifier (best effort), or the filename if no match.
        """
        # Pattern 1: sub-XX format
        match = re.search(r"sub-(\d+)", filename)
        if match:
            return f"sub-{match.group(1)}"

        # Pattern 2: subjectXX or subjXX format
        match = re.search(r"subj(?:ect)?(\d+)", filename, re.IGNORECASE)
        if match:
            return f"subject{match.group(1)}"

        # Pattern 3: SXX format
        match = re.search(r"S(\d+)", filename)
        if match:
            return f"S{match.group(1)}"

        # Pattern 4: Just numbers at the beginning
        match = re.search(r"^(\d+)", filename)
        if match:
            return match.group(1)

        return filename

    @staticmethod
    def _interpret_icc(icc_value: float) -> str:
        """Return a qualitative interpretation of ICC per Koo & Li (2016).

        Args:
            icc_value: ICC value in [0, 1].

        Returns:
            One of '(Poor)', '(Moderate)', '(Good)', '(Excellent)'.
        """
        if icc_value < 0.50:
            return "(Poor)"
        if icc_value < 0.75:
            return "(Moderate)"
        if icc_value < 0.90:
            return "(Good)"
        return "(Excellent)"

    # ==================== PATTERN MATCHING ====================

    def match_subjects_by_pattern(self) -> None:
        """Detect patterns in filenames and split them into two groups.

        Analyzes filenames to find common patterns that repeat exactly twice,
        then splits the files into two groups and populates study2_file_list.
        Updates study labels to reflect the detected patterns.
        """
        if not self._validate_pattern_matching_prerequisites():
            return

        filenames = [str(eeg) for eeg in self.comet_tbx_study1.list_eegs]

        if len(filenames) < 2:
            self._show_error(
                "Pattern Matching Error",
                "At least 2 files are required for pattern matching.",
            )
            return

        # Try to detect patterns
        pattern_result = self._detect_filename_patterns(filenames)

        if pattern_result is None:
            self._show_pattern_detection_error()
            return

        self._apply_detected_pattern(pattern_result)

    def _validate_pattern_matching_prerequisites(self) -> bool:
        """Validate prerequisites for pattern matching."""
        if not self.study1_loaded or not hasattr(self.comet_tbx_study1, "list_eegs"):
            self._show_error(
                "Pattern Matching Error",
                "No study loaded or no files available for pattern matching.",
            )
            return False
        return True

    def _show_pattern_detection_error(self) -> None:
        """Show error message when pattern detection fails."""
        error_msg = (
            "Could not detect any repeating patterns in the filenames.\n"
            "Expected patterns like:\n"
            "- sub-01_task-eyesclosed_eeg\n"
            "- sub-01_task-eyesopen_eeg\n"
            "where each subject appears exactly twice with different conditions."
        )
        self._show_error("Pattern Matching Error", error_msg)

    def _apply_detected_pattern(self, pattern_result: Tuple) -> None:
        """Apply the detected pattern to update UI."""
        (
            common_pattern,
            group1_pattern,
            group2_pattern,
            group1_files,
            group2_files,
        ) = pattern_result

        # Update file lists
        self.ui.study2_file_list.clear()
        for filename in group2_files:
            self.ui.study2_file_list.addItem(filename)

        self.ui.study1_file_list.clear()
        for filename in group1_files:
            self.ui.study1_file_list.addItem(filename)

        # Update study labels
        base_study_name = self.comet_tbx_study1.study_name
        self.ui.study1_name_label.setText(f"{group1_pattern}")
        self.ui.study2_name_label.setText(f"{group2_pattern}")

        # Store pattern information
        self.pattern_info = {
            "common_pattern": common_pattern,
            "group1_pattern": group1_pattern,
            "group2_pattern": group2_pattern,
            "group1_files": group1_files,
            "group2_files": group2_files,
        }

        success_msg = (
            f"Successfully detected pattern: {common_pattern}\n"
            f"Group 1 ({group1_pattern}): {len(group1_files)} files\n"
            f"Group 2 ({group2_pattern}): {len(group2_files)} files"
        )
        self._show_info("Pattern Matching Successful", success_msg)
        
        # Check the paired test checkbox since subjects have been successfully matched
        self.ui.paired_test_checkbox.setChecked(True)
        
        # Update plot button text now that study2_file_list has been populated
        self._update_plot_button_text()

    @staticmethod
    def _detect_filename_patterns(filenames: List[str]) -> Optional[Tuple]:
        """Detect common patterns in filenames that repeat exactly twice.

        Uses a robust, order-independent approach that parses filename components
        semantically rather than assuming fixed positions.

        Args:
            filenames: List of filenames to analyze.

        Returns:
            Tuple of (common_pattern, group1_pattern, group2_pattern, group1_files,
            group2_files) or None if no suitable pattern is found.
        """
        # Try semantic approach first
        result = CompareStudiesWindow._detect_semantic_patterns(filenames)
        if result is not None:
            return result

        # Fallback to flexible delimiter-based approach
        return CompareStudiesWindow._detect_flexible_delimiter_patterns(filenames)

    @staticmethod
    def _detect_semantic_patterns(filenames: List[str]) -> Optional[Tuple]:
        """Detect patterns using semantic understanding of filename components."""
        # Parse each filename into semantic components
        parsed_files = []
        for filename in filenames:
            components = CompareStudiesWindow._parse_filename_components(filename)
            if components:
                parsed_files.append((filename, components))

        if len(parsed_files) < 4:  # Need at least 4 files for 2x2 pattern
            return None

        # Find components that vary vs those that are constant
        all_components = {}
        for filename, components in parsed_files:
            for comp_type, comp_value in components.items():
                if comp_type not in all_components:
                    all_components[comp_type] = set()
                all_components[comp_type].add(comp_value)

        # Look for component types with exactly 2 values (potential conditions)
        condition_candidates = [
            comp_type
            for comp_type, values in all_components.items()
            if len(values) == 2
        ]

        # Try each potential condition component
        for condition_type in condition_candidates:
            result = CompareStudiesWindow._try_condition_grouping(
                parsed_files, condition_type, all_components[condition_type]
            )
            if result is not None:
                return result

        return None

    @staticmethod
    def _try_condition_grouping(
        parsed_files: List[Tuple], condition_type: str, condition_values: set
    ) -> Optional[Tuple]:
        """Try grouping files by a specific condition type."""
        condition_values = list(condition_values)

        # Group files by all components except the condition
        groups = defaultdict(list)
        for filename, components in parsed_files:
            if condition_type in components:
                # Create grouping key from all components except condition
                group_key_parts = []
                for comp_type, comp_value in sorted(components.items()):
                    if comp_type != condition_type:
                        group_key_parts.append(f"{comp_type}:{comp_value}")
                group_key = "|".join(group_key_parts)

                condition_value = components[condition_type]
                groups[group_key].append((filename, condition_value))

        # Check if we have a valid 2x2 pattern
        valid_groups = []
        for group_key, file_list in groups.items():
            if len(file_list) == 2:  # Exactly 2 files per group
                conditions_in_group = [item[1] for item in file_list]
                if len(set(conditions_in_group)) == 2:  # Both conditions present
                    valid_groups.append((group_key, file_list))

        # If we have at least 2 valid groups (4+ files total), we found a pattern
        if len(valid_groups) >= 2:
            total_files = sum(len(file_list) for _, file_list in valid_groups)
            if total_files >= 4:
                return CompareStudiesWindow._organize_pattern_groups(
                    valid_groups, condition_values, condition_type
                )

        return None

    @staticmethod
    def _organize_pattern_groups(
        valid_groups: List, condition_values: List[str], condition_type: str
    ) -> Tuple:
        """Organize detected pattern groups into final result."""
        group1_files = []
        group2_files = []

        for _, file_list in valid_groups:
            for filename, condition_value in file_list:
                if condition_value == condition_values[0]:
                    group1_files.append(filename)
                else:
                    group2_files.append(filename)

        if len(group1_files) == len(group2_files) and len(group1_files) >= 2:
            common_pattern = (
                f"{condition_type}:[{condition_values[0]}|{condition_values[1]}]"
            )
            return (
                common_pattern,
                condition_values[0],
                condition_values[1],
                group1_files,
                group2_files,
            )

        return None

    @staticmethod
    def _parse_filename_components(filename: str) -> Dict[str, str]:
        """Parse a filename into semantic components.

        Recognizes common EEG filename components regardless of order.

        Args:
            filename: Filename to parse.

        Returns:
            Dictionary of component_type -> component_value.
        """
        components = {}
        filename_lower = filename.lower()

        # Remove common file extensions
        clean_name = re.sub(
            r"\.(eeg|set|fdt|edf|bdf|cnt|vhdr|vmrk|fif|gz)$", "", filename_lower
        )

        # Parse various components
        CompareStudiesWindow._parse_subject_components(clean_name, components)
        CompareStudiesWindow._parse_session_components(clean_name, components)
        CompareStudiesWindow._parse_task_components(clean_name, components)
        CompareStudiesWindow._parse_run_components(clean_name, components)
        CompareStudiesWindow._parse_condition_components(clean_name, components)

        return components

    @staticmethod
    def _parse_subject_components(clean_name: str, components: Dict) -> None:
        """Parse subject-related components."""
        subject_patterns = [
            r"sub-?(\d+)",
            r"subject-?(\d+)",
            r"s(\d+)(?=[_\-\.]|$)",
            r"participant-?(\d+)",
            r"p(\d+)(?=[_\-\.]|$)",
        ]
        for pattern in subject_patterns:
            match = re.search(pattern, clean_name)
            if match:
                components["subject"] = f"sub-{match.group(1).zfill(2)}"
                break

    @staticmethod
    def _parse_session_components(clean_name: str, components: Dict) -> None:
        """Parse session-related components."""
        session_patterns = [
            r"ses-?session(\d+)",
            r"ses-?(\d+)",
            r"session-?(\d+)",
            r"sess-?(\d+)",
        ]
        for pattern in session_patterns:
            match = re.search(pattern, clean_name)
            if match:
                components["session"] = f"ses-{match.group(1).zfill(2)}"
                break

    @staticmethod
    def _parse_task_components(clean_name: str, components: Dict) -> None:
        """Parse task-related components."""
        condition_keywords = {
            "eyesclosed",
            "eyesopen",
            "closed",
            "open",
            "ec",
            "eo",
            "rest",
            "resting",
            "active",
            "baseline",
            "stimulation",
            "stim",
            "pre",
            "post",
            "before",
            "after",
            "control",
            "treatment",
            "a",
            "b",
            "1",
            "2",
            "cond1",
            "cond2",
            "condition1",
            "condition2",
        }

        task_patterns = [r"task-([a-zA-Z0-9]+)", r"task_([a-zA-Z0-9]+)"]

        for pattern in task_patterns:
            match = re.search(pattern, clean_name)
            if match:
                task_candidate = match.group(1).lower()
                if task_candidate not in condition_keywords:
                    components["task"] = match.group(1)
                    break

    @staticmethod
    def _parse_run_components(clean_name: str, components: Dict) -> None:
        """Parse run-related components."""
        run_patterns = [r"run-?(\d+)", r"r(\d+)(?=[_\-\.]|$)"]
        for pattern in run_patterns:
            match = re.search(pattern, clean_name)
            if match:
                components["run"] = f"run-{match.group(1).zfill(2)}"
                break

    @staticmethod
    def _parse_condition_components(clean_name: str, components: Dict) -> None:
        """Parse condition-related components."""
        # Task-specific condition patterns
        task_condition_patterns = [
            (r"task-?(eyesclosed|eyes?closed|ec)", "eyesclosed"),
            (r"task-?(eyesopen|eyes?open|eo)", "eyesopen"),
            (r"task-?(rest|resting)", "rest"),
            (r"task-?(active)", "active"),
            (r"task-?(baseline|base)", "baseline"),
            (r"task-?(stimulation|stim)", "stimulation"),
        ]

        for pattern, condition_name in task_condition_patterns:
            if re.search(pattern, clean_name):
                components["condition"] = condition_name
                return

        # Standalone condition patterns
        standalone_patterns = [
            (r"\b(eyesclosed|eyes?closed|ec)\b", "eyesclosed"),
            (r"\b(eyesopen|eyes?open|eo)\b", "eyesopen"),
            (r"\b(restingstate|resting|rest)\b", "rest"),
            (r"\b(baseline|base)\b", "baseline"),
            (r"\b(stimulation|stim)\b", "stimulation"),
            (r"\b(pre)\b", "pre"),
            (r"\b(post)\b", "post"),
            (r"\b(active)\b", "active"),
            (r"\b(passive)\b", "passive"),
            (r"\b(control|ctrl)\b", "control"),
            (r"\b(treatment|treat)\b", "treatment"),
        ]

        for pattern, condition_name in standalone_patterns:
            if re.search(pattern, clean_name):
                components["condition"] = condition_name
                return

        # Numeric conditions
        CompareStudiesWindow._parse_numeric_conditions(clean_name, components)

    @staticmethod
    def _parse_numeric_conditions(clean_name: str, components: Dict) -> None:
        """Parse numeric condition patterns."""
        numeric_conditions = re.findall(r"(?:cond|condition)(\d+)", clean_name)
        if numeric_conditions:
            components["condition"] = f"cond{numeric_conditions[0]}"
            return

        # Single letters or numbers as conditions
        letter_matches = re.findall(r"[_\-]([a-z])(?=[_\-]|$)", clean_name)
        if len(letter_matches) == 1 and letter_matches[0] in "abcdefgh":
            components["condition"] = letter_matches[0]
            return

        number_matches = re.findall(r"[_\-](\d)(?=[_\-]|$)", clean_name)
        if len(number_matches) == 1:
            components["condition"] = f"cond{number_matches[0]}"

    @staticmethod
    def _detect_flexible_delimiter_patterns(filenames: List[str]) -> Optional[Tuple]:
        """Flexible delimiter-based pattern detection."""
        delimiters = ["_", "-", ".", " "]

        for delimiter in delimiters:
            all_components = set()
            split_filenames = []

            for filename in filenames:
                parts = [p for p in filename.split(delimiter) if p]
                split_filenames.append((filename, parts))
                all_components.update(parts)

            # Try each component as the potential varying element
            for varying_component in all_components:
                result = CompareStudiesWindow._try_delimiter_grouping(
                    split_filenames, varying_component, delimiter
                )
                if result is not None:
                    return result

        return None

    @staticmethod
    def _try_delimiter_grouping(
        split_filenames: List[Tuple], varying_component: str, delimiter: str
    ) -> Optional[Tuple]:
        """Try grouping by delimiter-based pattern."""
        pattern_groups = defaultdict(list)

        for filename, parts in split_filenames:
            if varying_component in parts:
                normalized_parts = [
                    "VARY" if part == varying_component else part for part in parts
                ]
                pattern_key = delimiter.join(normalized_parts)
                pattern_groups[pattern_key].append((filename, varying_component))

        # Check each pattern group for 2-way splits
        for pattern_key, file_info in pattern_groups.items():
            if len(file_info) >= 4 and len(file_info) % 2 == 0:
                varying_parts = [info[1] for info in file_info]
                varying_counts = Counter(varying_parts)

                if len(varying_counts) == 2 and all(
                    count == len(file_info) // 2 for count in varying_counts.values()
                ):
                    group1_pattern, group2_pattern = list(varying_counts.keys())
                    group1_files = [
                        info[0] for info in file_info if info[1] == group1_pattern
                    ]
                    group2_files = [
                        info[0] for info in file_info if info[1] == group2_pattern
                    ]
                    common_pattern = pattern_key.replace(
                        "VARY", f"[{group1_pattern}|{group2_pattern}]"
                    )

                    return (
                        common_pattern,
                        group1_pattern,
                        group2_pattern,
                        group1_files,
                        group2_files,
                    )

        return None

    # ==================== LOGGING AND ERROR HANDLING ====================

    @staticmethod
    def _log_study_completion_status(comet_instance: COMET, study_name: str) -> None:
        """Log the completion status of all processing steps for a study.

        Args:
            comet_instance: COMET instance to inspect.
            study_name: Label used in logging output.
        """
        try:
            logger = get_logger()

            logger.section_header("STUDY_LOADING")
            logger.processing_success(
                "STUDY_LOADING",
                f"Study '{comet_instance.study_name}' loaded successfully",
            )
            logger.section_header("STUDY_STATUS")

            # Check each processing step
            steps = [
                ("done_preprocessing", "Data Preprocessing"),
                ("done_clustering", "Microstate Clustering"),
                ("done_microstate_labeling", "Microstate Labeling"),
                ("done_backfitting", "Microstate Backfitting"),
                ("done_extracting_features", "Feature Extraction"),
                ("done_source_localization", "Source Localization"),
                (
                    "done_identifying_microstate_sources",
                    "Source-Microstate Correlation",
                ),
            ]

            for attr, step_name in steps:
                if getattr(comet_instance, attr, False):
                    logger.processing_success("STUDY_STATUS", f"{step_name} - COMPLETED")
                else:
                    logger.warning("STUDY_STATUS", f"{step_name} - NOT COMPLETED")

            # Log additional info for clustering
            if comet_instance.done_clustering:
                if (
                    hasattr(comet_instance, "best_gev")
                    and comet_instance.best_gev is not None
                ):
                    logger.processing_info(
                        "STUDY_STATUS",
                        f"Best GEV: {100 * comet_instance.best_gev:.3f}%",
                    )
                if (
                    hasattr(comet_instance, "number_of_maps")
                    and comet_instance.number_of_maps is not None
                ):
                    logger.processing_info(
                        "STUDY_STATUS", f"Number of Maps: {comet_instance.number_of_maps}"
                    )

        except ImportError:
            # Fallback if terminal_logger is not available
            print(f"Study '{comet_instance.study_name}' loaded successfully")

    def _show_error(self, title: str, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, title, message, QMessageBox.Ok)

    def _show_info(self, title: str, message: str) -> None:
        """Show information message dialog."""
        QMessageBox.information(self, title, message, QMessageBox.Ok)
