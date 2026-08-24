"""UI to compare two EEG-COMET studies across features and maps."""

import os.path
import re
import traceback
import warnings
from collections import Counter, defaultdict
from contextlib import suppress
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from eeg_comet.data_utils.safe_io import safe_pd_read_pickle, safe_pickle_load
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import QDialog, QFileDialog, QListWidgetItem, QMessageBox, QSizePolicy

from eeg_comet.gui_utils.responsive import (
    apply_window_minimum,
    expand_canvas,
    scroll_wrap_all_tabs,
)
from scipy.stats import (
    f,
    levene,
    mannwhitneyu,
    pearsonr,
    t,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)
from scipy.ndimage import label as scipy_label
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian, Gamma
from statsmodels.genmod.families.links import Identity, Log
from statsmodels.genmod.cov_struct import Exchangeable, Independence

from eeg_comet.clustering_utils.microstate_visualizer import show_microstate
from eeg_comet.comet import COMET
from eeg_comet.features_utils.feature_io import FeatureIO
from eeg_comet.gui_utils.set_widgets_status import set_widgets_status
from eeg_comet.gui_utils.terminal_logger import get_logger
from mne.stats import permutation_cluster_1samp_test

# The correction combo shows conventional hyphenated labels, but statsmodels
# spells the FDR methods with underscores and rejects anything else.
_CORRECTION_METHOD_ALIASES = {
    "fdr-bh": "fdr_bh",
    "fdr-tsbh": "fdr_tsbh",
    "fdr-tsbky": "fdr_tsbky",
}


def _correction_method(label):
    """Translate a correction combo label into a ``multipletests`` method name."""
    normalized = (label or "").strip().lower()
    return _CORRECTION_METHOD_ALIASES.get(normalized, normalized)


# Levene's test is only informative with enough observations per group; below
# this the variance estimate is too noisy to act on, so Student's test is kept.
_MIN_N_FOR_LEVENE = 3


def independent_ttest(sample_a, sample_b):
    """Independent-samples t-test, using Welch's correction when variances differ.

    Levene's test decides between the two. Student's pooled-variance test is
    invalid when the groups have unequal variance and unequal size, which is the
    common case for microstate features across cohorts of different sizes.

    Returns:
        tuple: ``(statistic, p_value, used_welch)``.
    """
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)

    use_welch = False
    if len(sample_a) >= _MIN_N_FOR_LEVENE and len(sample_b) >= _MIN_N_FOR_LEVENE:
        try:
            _, levene_p = levene(sample_a, sample_b, center="median")
            use_welch = bool(levene_p < 0.05)
        except ValueError:
            # Degenerate input (e.g. zero variance in both groups).
            use_welch = False

    statistic, p_value = ttest_ind(sample_a, sample_b, equal_var=not use_welch)
    return statistic, p_value, use_welch


def gee_family_from_model_name(model_name):
    """Pick the GEE family implied by the model combo entry.

    Gamma with a log link suits the strictly positive, right-skewed outcomes
    typical of trial-level coverage and duration; Gaussian with an identity link
    is the default for roughly symmetric measures.
    """
    return "gamma" if "gamma" in (model_name or "").lower() else "gaussian"


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
    FEATURE_MODES = ["static", "dynamic", "averaged", "sliding", "pre_post"]
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
        self.setup_checkable_font_styling()

    # ==================== UI INITIALIZATION ====================

    def _setup_ui(self, context) -> None:
        """Set up static UI components from the Qt Designer file.

        Args:
            context: Resource/context provider to resolve the `.ui` path.
        """
        self.ui = uic.loadUi(context.get_resource("CompareStudiesWindow.ui"), self)
        self.ui.setWindowTitle("Comparison of two EEG-COMET studies")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        apply_window_minimum(self, "tool")
        if hasattr(self.ui, "comparison_study_label"):
            self.ui.comparison_study_label.setProperty("role", "banner")
        if hasattr(self.ui, "tabWidget"):
            scroll_wrap_all_tabs(self.ui.tabWidget)

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

        # Radio button connections for comparison mode
        compare_widgets = [
            self.ui.compare_two_studies_radio,
            self.ui.compare_surrogate_radio,
            self.ui.compare_random_radio,
            self.ui.compare_within_study_radio,
        ]
        for widget in compare_widgets:
            widget.clicked.connect(self.update_ui)

        # Feature combo connections - sync both combos
        self.ui.feature2visualize_combo.currentTextChanged.connect(self._on_visualize_feature_changed)
        self.ui.feature2compare_combo.currentTextChanged.connect(self._on_compare_feature_changed)
        
        # Visualization connections
        self.ui.show_features_checkbox.toggled.connect(self._on_show_features_toggled)
        self.ui.show_microstates_checkbox.toggled.connect(self._on_show_microstates_toggled)
        
        # Statistical analysis connections
        self.ui.compare_features_button.clicked.connect(self.update_feature_stats)
        self.ui.match_subjects_button.clicked.connect(self.match_subjects_by_pattern)
        
        # File list transfer buttons (for within-study comparison)
        self.ui.study1_to_substudy1.clicked.connect(self._move_to_study2_list)
        self.ui.substudy1_to_study1.clicked.connect(self._move_to_study1_list)
        
        # Enable drag-and-drop sorting on study2_file_list
        self.ui.study2_file_list.setDragDropMode(self.ui.study2_file_list.InternalMove)
        self.ui.study2_file_list.setDefaultDropAction(Qt.MoveAction)
        
        # Connect to model changes to update colors/numbers after drag-drop reorder
        self.ui.study2_file_list.model().rowsMoved.connect(self._on_study2_list_reordered)
        
        # Analysis option radio buttons - update model combo when changed
        analysis_radio_buttons = [
            self.ui.analyze_scope_full_radio,
            self.ui.analyze_scope_prepost_radio,
            self.ui.analyze_level_subject_radio,
            self.ui.analyze_level_trial_radio,
            self.ui.analyze_design_paired_radio,
            self.ui.analyze_design_independent_radio,
            self.ui.analyze_type_parametric_radio,
            self.ui.analyze_type_nonparametric_radio,
        ]
        for radio in analysis_radio_buttons:
            radio.toggled.connect(self._update_analyze_model_combo)
        
        # Initialize the model combo
        self._update_analyze_model_combo()

    def _on_visualize_feature_changed(self, text: str) -> None:
        """Handle feature2visualize_combo change - sync with feature2compare_combo."""
        # Prevent infinite loop by checking if already synced
        if self.ui.feature2compare_combo.currentText() != text:
            index = self.ui.feature2compare_combo.findText(text)
            if index >= 0:
                self.ui.feature2compare_combo.blockSignals(True)
                self.ui.feature2compare_combo.setCurrentIndex(index)
                self.ui.feature2compare_combo.blockSignals(False)
        
        # Trigger feature selection changed logic
        self._on_feature_selection_changed()

    def _on_compare_feature_changed(self, text: str) -> None:
        """Handle feature2compare_combo change - sync with feature2visualize_combo."""
        # Prevent infinite loop by checking if already synced
        if self.ui.feature2visualize_combo.currentText() != text:
            index = self.ui.feature2visualize_combo.findText(text)
            if index >= 0:
                self.ui.feature2visualize_combo.blockSignals(True)
                self.ui.feature2visualize_combo.setCurrentIndex(index)
                self.ui.feature2visualize_combo.blockSignals(False)
        
        # Trigger feature selection changed logic
        self._on_feature_selection_changed()

    def _update_analyze_model_combo(self, checked: bool = True) -> None:
        """Update the analyze_model_combo based on current radio button selections.
        
        Args:
            checked: Whether the radio button is checked (from toggled signal).
                     Only update when a button is checked, not unchecked.
        """
        # Only update when a radio button is checked, not when unchecked
        if not checked:
            return
        
        # If ROF is selected, don't update - keep TFCE
        if hasattr(self, 'ui') and self._is_rof_feature_selected():
            return
            
        # Get current selections
        is_subject_level = self.ui.analyze_level_subject_radio.isChecked()
        is_trial_level = self.ui.analyze_level_trial_radio.isChecked()
        is_paired = self.ui.analyze_design_paired_radio.isChecked()
        is_independent = self.ui.analyze_design_independent_radio.isChecked()
        is_parametric = self.ui.analyze_type_parametric_radio.isChecked()
        is_nonparametric = self.ui.analyze_type_nonparametric_radio.isChecked()
        
        # Clear current items
        self.ui.analyze_model_combo.clear()
        
        if is_subject_level:
            if is_paired:
                if is_parametric:
                    self.ui.analyze_model_combo.addItems([
                        "Paired t-Test",
                        "Repeated Measures ANOVA"
                    ])
                else:  # non-parametric
                    self.ui.analyze_model_combo.addItems([
                        "Wilcoxon Signed-Rank Test",
                        "Friedman Test"
                    ])
            else:  # independent
                if is_parametric:
                    self.ui.analyze_model_combo.addItems([
                        "Independent t-Test",
                        "One-Way ANOVA"
                    ])
                else:  # non-parametric
                    self.ui.analyze_model_combo.addItems([
                        "Mann-Whitney U Test",
                        "Kruskal-Wallis H Test"
                    ])
        else:  # trial-level
            if is_paired:
                if is_parametric:
                    self.ui.analyze_model_combo.addItems([
                        "Linear Mixed Model (LMM)",
                        "Generalized Estimating Equations (GEE, Gaussian/identity)"
                    ])
                else:  # non-parametric
                    self.ui.analyze_model_combo.addItems([
                        "Generalized Estimating Equations (GEE, Gamma/log)",
                        "Permutation Test with Clustering"
                    ])
            else:  # independent
                if is_parametric:
                    self.ui.analyze_model_combo.addItems([
                        "Linear Mixed Model (LMM)",
                        "Generalized Estimating Equations (GEE, Gaussian/identity)"
                    ])
                else:  # non-parametric
                    self.ui.analyze_model_combo.addItems([
                        "Generalized Estimating Equations (GEE, Gamma/log)",
                        "Bootstrap Resampling"
                    ])

    def update_widget_font_weight(self, checked_or_widget=None):
        """Update font weight of a radio button or checkbox based on its checked state.

        Args:
            checked_or_widget: Either a boolean (from toggled signal) or a widget object.
                              If boolean or None, uses self.sender() to get the widget.

        Makes the widget bold when checked, normal when unchecked.
        """
        if checked_or_widget is None or isinstance(checked_or_widget, bool):
            widget = self.sender()
        else:
            widget = checked_or_widget

        if widget is None:
            return

        if widget.isChecked():
            widget.setStyleSheet("font-weight: bold;")
        else:
            widget.setStyleSheet("font-weight: normal;")

    def setup_checkable_font_styling(self):
        """Set up font weight styling for all radio buttons and checkboxes.

        Connects toggled signal to update font weight and initializes current states.
        """
        checkable_widgets = [
            # Comparison mode radios
            "compare_two_studies_radio",
            "compare_surrogate_radio",
            "compare_random_radio",
            "compare_within_study_radio",
            # Visualization checkboxes
            "show_features_checkbox",
            "show_microstates_checkbox",
            # Analysis option radios
            "analyze_scope_full_radio",
            "analyze_scope_prepost_radio",
            "analyze_level_subject_radio",
            "analyze_level_trial_radio",
            "analyze_design_paired_radio",
            "analyze_design_independent_radio",
            "analyze_type_parametric_radio",
            "analyze_type_nonparametric_radio",
        ]

        for widget_name in checkable_widgets:
            widget = getattr(self.ui, widget_name, None)
            if widget is not None:
                widget.toggled.connect(self.update_widget_font_weight)
                self.update_widget_font_weight(widget)

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
        expand_canvas(canvas, minimum=(360, 240))
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
        # Clear existing items before adding new ones
        listwidget.clear()
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

    def _move_to_study2_list(self) -> None:
        """Move selected items from study1_file_list to study2_file_list."""
        selected_items = self.ui.study1_file_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            # Get the row of the item
            row = self.ui.study1_file_list.row(item)
            # Take the item from study1 list
            taken_item = self.ui.study1_file_list.takeItem(row)
            if taken_item:
                # Create a new item for study2 list (preserving data)
                new_item = QListWidgetItem(taken_item.text())
                # Copy the original filename from UserRole if it exists
                original_filename = taken_item.data(Qt.UserRole)
                if original_filename:
                    new_item.setData(Qt.UserRole, original_filename)
                # Copy the background color if set
                if taken_item.background().style() != Qt.NoBrush:
                    new_item.setBackground(taken_item.background())
                # Add to study2 list
                self.ui.study2_file_list.addItem(new_item)

    def _move_to_study1_list(self) -> None:
        """Move selected items from study2_file_list back to study1_file_list."""
        selected_items = self.ui.study2_file_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            # Get the row of the item
            row = self.ui.study2_file_list.row(item)
            # Take the item from study2 list
            taken_item = self.ui.study2_file_list.takeItem(row)
            if taken_item:
                # Create a new item for study1 list (preserving data)
                new_item = QListWidgetItem(taken_item.text())
                # Copy the original filename from UserRole if it exists
                original_filename = taken_item.data(Qt.UserRole)
                if original_filename:
                    new_item.setData(Qt.UserRole, original_filename)
                # Copy the background color if set
                if taken_item.background().style() != Qt.NoBrush:
                    new_item.setBackground(taken_item.background())
                # Add to study1 list
                self.ui.study1_file_list.addItem(new_item)

    def _on_study2_list_reordered(self) -> None:
        """Handle reordering of study2_file_list after drag-drop."""
        self._update_paired_list_display()

    def _update_paired_list_display(self) -> None:
        """Update colors and numbers for both lists to reflect current pairing.
        
        Items at the same index in both lists are considered paired and will have
        matching colors and prefix numbers.
        """
        study1_count = self.ui.study1_file_list.count()
        study2_count = self.ui.study2_file_list.count()
        
        # Generate colors for the number of pairs
        max_pairs = max(study1_count, study2_count)
        if max_pairs == 0:
            return
        
        # Generate pastel colors using golden ratio for good distribution
        colors = []
        golden_ratio = 0.618033988749895
        hue = 0.0
        for i in range(max_pairs):
            hue = (hue + golden_ratio) % 1.0
            # Create pastel color (high brightness, medium saturation)
            color = QColor.fromHsvF(hue, 0.35, 0.95)
            colors.append(color)
        
        # Update study1_file_list
        for i in range(study1_count):
            item = self.ui.study1_file_list.item(i)
            if item:
                # Get original filename
                original_filename = item.data(Qt.UserRole)
                if original_filename is None:
                    # Extract from current text if no UserRole data
                    current_text = item.text()
                    # Remove existing number prefix if present
                    if ". " in current_text and current_text.split(". ")[0].isdigit():
                        original_filename = ". ".join(current_text.split(". ")[1:])
                    else:
                        original_filename = current_text
                    item.setData(Qt.UserRole, original_filename)
                
                # Update display with new number
                item.setText(f"{i + 1}. {original_filename}")
                # Set color
                item.setBackground(QBrush(colors[i]))
        
        # Update study2_file_list
        for i in range(study2_count):
            item = self.ui.study2_file_list.item(i)
            if item:
                # Get original filename
                original_filename = item.data(Qt.UserRole)
                if original_filename is None:
                    # Extract from current text if no UserRole data
                    current_text = item.text()
                    # Remove existing number prefix if present
                    if ". " in current_text and current_text.split(". ")[0].isdigit():
                        original_filename = ". ".join(current_text.split(". ")[1:])
                    else:
                        original_filename = current_text
                    item.setData(Qt.UserRole, original_filename)
                
                # Update display with new number
                item.setText(f"{i + 1}. {original_filename}")
                # Set color (matching with study1 at same index)
                item.setBackground(QBrush(colors[i]))

    # ==================== UI STATE MANAGEMENT ====================

    def update_ui(self) -> None:
        """Update UI element states based on loaded studies and selected comparison mode."""
        self._clear_ui_elements()
        self._configure_comparison_mode()
        self._update_feature_controls()
        self._update_plot_button_text()

    def _clear_ui_elements(self) -> None:
        """Clear previous selections and texts."""
        self.ui.feature2visualize_combo.clear()
        self.ui.feature2compare_combo.clear()
        self.ui.plot_label.clear()
        self.ui.stats_textedit.clear()
        # Reset checkboxes when UI elements are cleared
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
        # Clear pattern info from within-study matching
        self.pattern_info = None
        
        # Reset load button text for two-studies mode
        self.ui.load_study1_button.setText("Load Study 1")
        
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
        # Check if we're dealing with event-related features by checking the selected feature name
        is_event_related = "(Pre/Post Event)" in self.ui.feature2visualize_combo.currentText()
        
        # Change load button text for within-study mode
        self.ui.load_study1_button.setText("Load Study")
        
        # Show within-study widgets
        widgets_to_show = [
            self.ui.study1_to_substudy1,
            self.ui.substudy1_to_study1,
            self.ui.study2_name_label,
            self.ui.study2_file_list,
            self.ui.match_subjects_button,
        ]
        # Hide Study 2 load button and related widgets
        widgets_to_hide = [
            self.ui.load_study2_button, 
            self.canvas_microstates_study2,
        ]

        self._set_widget_visibility(widgets_to_show, True)
        self._set_widget_visibility(widgets_to_hide, False)

        # Configure widget states
        self.ui.load_study2_button.setEnabled(False)
        self.ui.study2_name_label.setEnabled(True)
        self.ui.study2_file_list.setEnabled(True)
        self.canvas_microstates_study2.setEnabled(False)

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

        # Set labels based on whether this is event-related or pattern-matched
        if is_event_related:
            # For event-related features, we're comparing pre vs post within the same files
            self.ui.study2_name_label.setText("Post-Event")
            self.ui.study1_name_label.setText("Pre-Event")
            self.study2_name = "post_event"
            # Clear the file list since we're not separating files
            self.ui.study2_file_list.clear()
            self.ui.study2_file_list.addItem("Same files (Pre vs Post comparison)")
        elif self.pattern_info is not None:
            # Pattern matching was done - preserve the detected condition labels
            self.ui.study1_name_label.setText(self.pattern_info["group1_pattern"])
            self.ui.study2_name_label.setText(self.pattern_info["group2_pattern"])
            self.study2_name = self.pattern_info["group2_pattern"]
        else:
            # For regular within-study comparisons, use the standard labels
            self.ui.study2_name_label.setText(self.STUDY_NAMES["within"])
            self.study2_name = "within_study"

    def _configure_synthetic_mode(self) -> None:
        """Configure UI for synthetic comparison mode."""
        # Clear pattern info from within-study matching
        self.pattern_info = None
        
        # Reset load button text for synthetic mode
        self.ui.load_study1_button.setText("Load Study 1")
        
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
        # Get actual extracted features from files rather than config
        study1_extracted_features = self._get_extracted_feature_codes(self.comet_tbx_study1)
        study2_extracted_features = self._get_extracted_feature_codes(self.comet_tbx_study2)
        
        common_features = [
            feat
            for feat in study1_extracted_features
            if feat in study2_extracted_features
        ]
        self._populate_feature_combo(common_features)

    def _populate_study_features(self) -> None:
        """Populate feature combo with features from study 1."""
        # Get actual extracted features from files rather than config
        extracted_features = self._get_extracted_feature_codes(self.comet_tbx_study1)
        self._populate_feature_combo(extracted_features)

    def _get_extracted_feature_codes(self, tbx: COMET) -> List[str]:
        """Get list of feature codes that have been actually extracted from feature files.
        
        This reads the actual feature file columns rather than relying on the config,
        ensuring all extracted features (including new variability features) are discovered.
        
        Returns features from all available modes (averaged, sliding, etc.) with mode suffix.
        
        Args:
            tbx: COMET toolbox object to check for extracted features.
            
        Returns:
            List of available feature codes with mode suffix 
            (e.g., ["OCC", "DUR", "COV", "OCC (Sliding)", "DUR (Sliding)", etc.]).
        """
        extracted_features = set()
        extracted_features_with_mode = {}  # feature_code -> set of modes
        
        if not hasattr(tbx, "extracted_features_path") or not os.path.exists(
            tbx.extracted_features_path
        ):
            # Fallback to config feature_list if no extracted features path
            return getattr(tbx, "feature_list", [])
        
        # Check what feature files are available for this study
        available_features = self._discover_available_features(tbx)
        
        if not available_features:
            # Fallback to config feature_list
            return getattr(tbx, "feature_list", [])
        
        # Try to load all available feature files to discover column features
        # Track which modes each feature appears in
        for feature_type in self.FEATURE_TYPES:
            if feature_type not in available_features:
                continue
                
            for feature_mode in available_features[feature_type]:
                for export_format in self.EXPORT_FORMATS:
                    # Try both standard and variability feature files
                    filenames_to_try = [
                        f"{feature_type}_{feature_mode}_features{export_format}",
                        f"{feature_type}_{feature_mode}_variability_features{export_format}"
                    ]
                    
                    for feature_filename in filenames_to_try:
                        feature_path = os.path.join(tbx.extracted_features_path, feature_filename)
                        
                        if os.path.exists(feature_path):
                            try:
                                # Load just the header to check columns
                                df = None
                                if export_format == ".csv":
                                    df = pd.read_csv(feature_path, nrows=0)
                                elif export_format == ".pkl":
                                    df = safe_pd_read_pickle(feature_path)
                                elif export_format == ".hdf":
                                    df = pd.read_hdf(feature_path, key="features")
                                elif export_format == ".json":
                                    df = pd.read_json(feature_path, lines=True, nrows=0)
                                
                                if df is not None:
                                    # Extract feature codes from column names
                                    for col in df.columns:
                                        if col in ["Filename", "Study", "Window_index", "Trial", 
                                                   "Window_Type", "Event_name", "base_filename",
                                                   "window_start_idx", "window_end_idx", "window_duration_ms",
                                                   "N", "Time"]:
                                            continue
                                        # Extract feature code from column name
                                        # e.g., "COV_A" -> "COV", "TP_A_B" -> "TP", "DUR_SD_A" -> "DUR_SD"
                                        parts = col.split("_")
                                        
                                        # Handle multi-part feature codes like DUR_SD, DUR_RMSSD, etc.
                                        if len(parts) >= 2 and parts[1] in ["SD", "RMSSD"]:
                                            feature_code = f"{parts[0]}_{parts[1]}"
                                        else:
                                            feature_code = parts[0]
                                        
                                        extracted_features.add(feature_code)
                                        
                                        # Track which mode this feature appears in
                                        if feature_code not in extracted_features_with_mode:
                                            extracted_features_with_mode[feature_code] = set()
                                        extracted_features_with_mode[feature_code].add(feature_mode)
                                    
                            except Exception as e:
                                print(f"Warning: Could not read feature columns from {feature_path}: {e}")
                                continue
        
        # Check for special ROF and RTF features (exported as separate files)
        for feature_type in self.FEATURE_TYPES:
            # Check for ROF features
            for export_format in self.EXPORT_FORMATS:
                rof_filenames = [
                    f"{feature_type}_pre_post_rof_data{export_format}",
                    f"ROF_timeseries{export_format}",
                    f"ROF_timeseries_{feature_type}{export_format}",
                    f"{feature_type}_ROF_timeseries{export_format}",
                    f"rof_timeseries{export_format}",
                ]
                
                for rof_filename in rof_filenames:
                    rof_path = os.path.join(tbx.extracted_features_path, rof_filename)
                    if os.path.exists(rof_path):
                        extracted_features.add("ROF")
                        if "ROF" not in extracted_features_with_mode:
                            extracted_features_with_mode["ROF"] = set()
                        extracted_features_with_mode["ROF"].add("pre_post")
                        break
            
            # Check for RTF features
            for export_format in self.EXPORT_FORMATS:
                rtf_filenames = [
                    f"{feature_type}_pre_post_rtf_data{export_format}",
                    f"RTF_averages{export_format}",
                    f"RTF_averages_{feature_type}{export_format}",
                    f"{feature_type}_RTF_averages{export_format}",
                    f"rtf_averages{export_format}",
                ]
                
                for rtf_filename in rtf_filenames:
                    rtf_path = os.path.join(tbx.extracted_features_path, rtf_filename)
                    if os.path.exists(rtf_path):
                        extracted_features.add("RTF")
                        if "RTF" not in extracted_features_with_mode:
                            extracted_features_with_mode["RTF"] = set()
                        extracted_features_with_mode["RTF"].add("pre_post")
                        break
        
        # Build final feature list with mode suffixes where needed
        final_feature_list = []
        for feature_code in sorted(extracted_features):
            modes = extracted_features_with_mode.get(feature_code, set())
            
            # Check if this is a variability feature (SD, RMSSD)
            # These are computed FROM sliding but result in single value per file
            is_variability_feature = (
                feature_code.endswith("_SD") or 
                feature_code.endswith("_RMSSD") or
                "_SD_" in feature_code or
                "_RMSSD_" in feature_code
            )
            
            # Special handling for ROF and RTF features
            if feature_code in ["ROF", "RTF"]:
                # ROF and RTF are always event-related features
                final_feature_list.append(feature_code)
                continue
            
            # If feature appears in multiple modes, add separate entries for each mode
            if len(modes) > 1:
                # Add averaged mode first (if available)
                if "averaged" in modes:
                    final_feature_list.append(feature_code)  # No suffix for averaged (default)
                # Add sliding mode (but not for variability features - they're always aggregated)
                if "sliding" in modes and not is_variability_feature:
                    final_feature_list.append(f"{feature_code} (Sliding)")
                # Add pre_post mode
                if "pre_post" in modes:
                    final_feature_list.append(f"{feature_code} (Pre/Post Event)")
                # Add other modes
                for mode in sorted(modes):
                    if mode not in ["averaged", "sliding", "pre_post"]:
                        final_feature_list.append(f"{feature_code} ({mode.capitalize()})")
            else:
                # Feature only in one mode
                # Variability features are always treated as aggregated, even if only in sliding mode
                if "sliding" in modes and not is_variability_feature:
                    final_feature_list.append(f"{feature_code} (Sliding)")
                elif "pre_post" in modes:
                    final_feature_list.append(f"{feature_code} (Pre/Post Event)")
                else:
                    # Regular feature or variability feature - no suffix
                    final_feature_list.append(feature_code)

        # Return all accumulated features
        if final_feature_list:
            return final_feature_list

        # If we couldn't read any files, fallback to config feature_list
        return getattr(tbx, "feature_list", [])

    def _populate_feature_combo(self, features: List[str]) -> None:
        """Populate both feature combo boxes with full feature names.

        Args:
            features: List of feature codes (may include mode suffixes like "COV (Sliding)").
        """
        # Block signals to prevent sync loops during population
        self.ui.feature2visualize_combo.blockSignals(True)
        self.ui.feature2compare_combo.blockSignals(True)
        
        self.ui.feature2visualize_combo.clear()
        self.ui.feature2compare_combo.clear()
        full_feature_names = []

        for feat in features:
            # Check if this has a mode suffix
            if " (" in feat and feat.endswith(")"):
                # Extract feature code and mode
                feature_code = feat.split(" (")[0]
                mode = feat.split(" (")[1].rstrip(")")

                # Get full feature name
                full_name = self.feature_list_dictionary.get(feature_code, feature_code)

                # Handle special mode names
                if mode == "Pre/Post Event":
                    mode_display = "Pre/Post Event"
                elif mode == "Sliding":
                    mode_display = "Sliding Window"
                else:
                    mode_display = mode

                # Append mode with dash separator
                full_feature_name = f"{full_name} - {mode_display}"
            else:
                # No mode suffix, just get full name
                full_feature_name = self.feature_list_dictionary.get(feat, feat)

            full_feature_names.append(full_feature_name)

        # Add items to both combo boxes
        self.ui.feature2visualize_combo.addItems(full_feature_names)
        self.ui.feature2compare_combo.addItems(full_feature_names)
        
        # Re-enable signals
        self.ui.feature2visualize_combo.blockSignals(False)
        self.ui.feature2compare_combo.blockSignals(False)

    def _set_no_features_message(self, message: str = None) -> None:
        """Set a 'no features' message in both combo boxes.

        Args:
            message: Custom message to display.
        """
        self.ui.feature2visualize_combo.clear()
        self.ui.feature2compare_combo.clear()
        message = message or self.ERROR_MESSAGES["no_features"]
        self.ui.feature2visualize_combo.addItem(message)
        self.ui.feature2compare_combo.addItem(message)
        set_widgets_status(self.ui.compare_features_button, mode="disable")

    def _disable_feature_buttons(self) -> None:
        """Disable all feature-related buttons."""
        set_widgets_status(self.ui.compare_features_button, mode="disable")

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
        selected_feature_full_name = self.ui.feature2visualize_combo.currentText()

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
        # Use the compare combo as this is used for analysis
        selected_full_name = self.ui.feature2compare_combo.currentText()

        # Remove mode suffix if present (e.g., "Microstate Coverage (%) - Sliding" -> "Microstate Coverage (%)")
        if " - " in selected_full_name:
            selected_full_name = selected_full_name.split(" - ")[0]

        if hasattr(self, "feature_list_dictionary"):
            reverse_dict = {v: k for k, v in self.feature_list_dictionary.items()}
            return reverse_dict.get(selected_full_name, selected_full_name)

        return selected_full_name

    def _is_sliding_feature_selected(self) -> bool:
        """Check if the currently selected feature is a sliding feature.

        Returns:
            True if selected feature has "- Sliding" suffix, False otherwise.
        """
        selected_full_name = self.ui.feature2compare_combo.currentText()
        return " - Sliding" in selected_full_name or " - Sliding Window" in selected_full_name

    def _is_pre_post_event_feature_selected(self) -> bool:
        """Check if the currently selected feature is a pre/post event feature.

        Returns:
            True if selected feature has "- Pre/Post Event" suffix, False otherwise.
        """
        selected_full_name = self.ui.feature2compare_combo.currentText()
        return " - Pre/Post Event" in selected_full_name

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

        # Check if this is an event-related feature selection by checking the selected feature name
        is_pre_post_selected = self._is_pre_post_event_feature_selected()

        # For pre/post event features, use "pre_post" mode
        if is_pre_post_selected and "pre_post" in available_modes:
            return "real", "pre_post"

        # For sliding features, use "sliding" mode
        is_sliding_selected = self._is_sliding_feature_selected()
        if is_sliding_selected and "sliding" in available_modes:
            return "real", "sliding"

        # Default to averaged/static mode
        feature_mode = "averaged" if "averaged" in available_modes else available_modes[0]
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
                    # Check for standard feature files
                    feature_filename = (
                        f"{feature_type}_{feature_mode}_features{export_format}"
                    )
                    feature_path = os.path.join(
                        tbx.extracted_features_path, feature_filename
                    )

                    # Also check for variability feature files
                    variability_filename = (
                        f"{feature_type}_{feature_mode}_variability_features{export_format}"
                    )
                    variability_path = os.path.join(
                        tbx.extracted_features_path, variability_filename
                    )

                    if (
                        (os.path.exists(feature_path) or os.path.exists(variability_path))
                        and feature_mode not in available_modes
                    ):
                        available_modes.append(feature_mode)

            if available_modes:
                available_features[feature_type] = available_modes

        # Check for special ROF and RTF features (exported as separate files)
        special_features = {}
        for feature_type in CompareStudiesWindow.FEATURE_TYPES:
            special_modes = []

            # Check for ROF features
            for export_format in CompareStudiesWindow.EXPORT_FORMATS:
                rof_filenames = [
                    f"{feature_type}_pre_post_rof_data{export_format}",
                    f"ROF_timeseries{export_format}",
                    f"ROF_timeseries_{feature_type}{export_format}",
                    f"{feature_type}_ROF_timeseries{export_format}",
                    f"rof_timeseries{export_format}",
                ]

                for rof_filename in rof_filenames:
                    rof_path = os.path.join(tbx.extracted_features_path, rof_filename)
                    if os.path.exists(rof_path) and "pre_post" not in special_modes:
                        special_modes.append("pre_post")
                        break

            # Check for RTF features
            for export_format in CompareStudiesWindow.EXPORT_FORMATS:
                rtf_filenames = [
                    f"{feature_type}_pre_post_rtf_data{export_format}",
                    f"RTF_averages{export_format}",
                    f"RTF_averages_{feature_type}{export_format}",
                    f"{feature_type}_RTF_averages{export_format}",
                    f"rtf_averages{export_format}",
                ]

                for rtf_filename in rtf_filenames:
                    rtf_path = os.path.join(tbx.extracted_features_path, rtf_filename)
                    if os.path.exists(rtf_path) and "pre_post" not in special_modes:
                        special_modes.append("pre_post")
                        break

            if special_modes:
                special_features[feature_type] = special_modes

        # Merge special features with regular features
        for feature_type, modes in special_features.items():
            if feature_type in available_features:
                # Add any new modes that aren't already present
                for mode in modes:
                    if mode not in available_features[feature_type]:
                        available_features[feature_type].append(mode)
            else:
                available_features[feature_type] = modes

        return available_features

    @staticmethod
    def _load_features_safely(
        tbx: COMET, feature_type: str, feature_mode: str
    ) -> Optional[pd.DataFrame]:
        """Safely load features with proper error handling.

        Loads both standard and variability features if both exist, and merges them.

        Args:
            tbx: COMET toolbox object.
            feature_type: Type of features ('real', 'surrogate', 'random').
            feature_mode: Mode ('static', 'dynamic', 'averaged', 'sliding', 'pre_post').

        Returns:
            Loaded features or None if loading fails.
        """
        if not hasattr(tbx, "extracted_features_path") or not os.path.exists(
            tbx.extracted_features_path
        ):
            return None

        standard_df = None
        variability_df = None

        for export_format in CompareStudiesWindow.EXPORT_FORMATS:
            # Try to load standard feature file
            feature_filename = f"{feature_type}_{feature_mode}_features{export_format}"
            feature_path = os.path.join(tbx.extracted_features_path, feature_filename)

            if os.path.exists(feature_path):
                try:
                    standard_df = FeatureIO().import_features(feature_path, export_format)
                except Exception as e:
                    print(f"Error loading {feature_path}: {e}")

            # Try to load variability feature file from the same mode
            variability_filename = f"{feature_type}_{feature_mode}_variability_features{export_format}"
            variability_path = os.path.join(tbx.extracted_features_path, variability_filename)

            if os.path.exists(variability_path):
                try:
                    variability_df = FeatureIO().import_features(variability_path, export_format)
                except Exception as e:
                    print(f"Error loading {variability_path}: {e}")

            # If variability features don't exist for current mode, try sliding mode
            # (variability features are often only extracted for sliding windows)
            if variability_df is None and feature_mode != "sliding":
                variability_sliding_filename = f"{feature_type}_sliding_variability_features{export_format}"
                variability_sliding_path = os.path.join(tbx.extracted_features_path, variability_sliding_filename)

                if os.path.exists(variability_sliding_path):
                    try:
                        # Load the sliding variability features
                        variability_sliding_df = FeatureIO().import_features(variability_sliding_path, export_format)

                        # For averaged mode, we need to average the sliding variability features per file
                        if feature_mode == "averaged" and "Filename" in variability_sliding_df.columns:
                            # Group by Filename and calculate mean for all feature columns
                            feature_cols = [col for col in variability_sliding_df.columns
                                          if col not in ['Filename', 'Study', 'Window_index', 'Trial',
                                                        'Window_Type', 'Event_name', 'base_filename',
                                                        'window_start_idx', 'window_end_idx', 'window_duration_ms']]

                            if feature_cols:
                                # Average across windows for each file
                                variability_df = variability_sliding_df.groupby('Filename')[feature_cols].mean().reset_index()
                        else:
                            # For other modes, just use the sliding data as-is
                            variability_df = variability_sliding_df

                    except Exception as e:
                        print(f"Error loading or processing {variability_sliding_path}: {e}")

            # If we found at least one file with this export format, process and return
            if standard_df is not None or variability_df is not None:
                # Merge standard and variability features if both exist
                if standard_df is not None and variability_df is not None:
                    # Identify metadata columns to merge on (not feature columns)
                    # Common metadata columns in feature files
                    potential_merge_cols = ['Filename', 'Study', 'Window_index', 'Trial',
                                           'Window_Type', 'Event_name', 'base_filename',
                                           'window_start_idx', 'window_end_idx', 'window_duration_ms']

                    # Find which metadata columns exist in both dataframes
                    merge_cols = [col for col in potential_merge_cols
                                  if col in standard_df.columns and col in variability_df.columns]

                    if merge_cols:
                        merged_df = pd.merge(standard_df, variability_df,
                                           on=merge_cols, how='outer')
                        return merged_df
                    else:
                        # If no common metadata columns, just return standard features
                        return standard_df
                elif standard_df is not None:
                    return standard_df
                elif variability_df is not None:
                    return variability_df

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
        # Check for original filename in UserRole (used when numbering is applied)
        filenames_in_list = []
        for i in range(file_list_widget.count()):
            item = file_list_widget.item(i)
            if item is not None:
                # Try to get original filename from UserRole data
                original_filename = item.data(Qt.UserRole)
                if original_filename:
                    filenames_in_list.append(original_filename)
                else:
                    # Fallback to display text (strip number prefix if present)
                    text = item.text()
                    # Remove "N. " prefix if present (e.g., "1. filename" -> "filename")
                    if re.match(r'^\d+\.\s+', text):
                        text = re.sub(r'^\d+\.\s+', '', text)
                    filenames_in_list.append(text)

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
            figure.add_subplot(1, len(tbx.microstate_labels), idx + 1)
            for idx in range(len(tbx.microstate_labels))
        ]

        # Plot each microstate with its label
        for idx, ax in enumerate(axs):
            self._plot_single_microstate(
                tbx.best_maps[idx, :], tbx.microstate_labels[idx], tbx.eeg_info, ax
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
            and hasattr(tbx, "microstate_labels")
            and tbx.microstate_labels
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
        # Match columns that start with the feature code followed by underscore
        # This ensures "DUR" doesn't match "DUR_SD_A" but matches "DUR_A"
        # Also handle single-column features (no underscore suffix)
        feature_columns = []
        for col in common_features_df.columns:
            if col == selected_feature:  # Exact match for single-column features
                feature_columns.append(col)
            elif col.startswith(selected_feature + "_"):  # Must have underscore after feature code
                feature_columns.append(col)

        columns_to_keep = ["Filename", "Study"] + feature_columns
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
        # Get display names from UI labels
        study1_display_name = self.ui.study1_name_label.text()
        study2_display_name = self.ui.study2_name_label.text()

        # Create a copy of plot_data and replace study names with display names
        plot_data_display = plot_data.copy()
        plot_data_display['Study'] = plot_data_display['Study'].replace({
            self.comet_tbx_study1.study_name: study1_display_name,
            self.study2_name: study2_display_name
        })

        # Create violin plot with display names
        sns.violinplot(
            x="Feature",
            y=selected_feature,
            hue="Study",
            data=plot_data_display,
            ax=ax,
            hue_order=[study1_display_name, study2_display_name],
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
        # Reconfigure comparison mode based on selected feature
        self._configure_comparison_mode()
        
        # Update analysis options based on selected feature
        self._update_analysis_options_for_feature()

        if self.ui.show_features_checkbox.isChecked():
            self._auto_plot_features()

    def _is_rof_feature_selected(self) -> bool:
        """Check if the currently selected feature is a ROF (Rate of Occurrence Function) feature.
        
        Returns:
            True if selected feature contains "ROF", False otherwise.
        """
        selected_full_name = self.ui.feature2compare_combo.currentText()
        return "ROF" in selected_full_name.upper()

    def _update_analysis_options_for_feature(self) -> None:
        """Update analysis option widgets based on the selected feature.
        
        For ROF features, disable all options and set model to TFCE.
        For other features, enable all options and update model normally.
        """
        is_rof = self._is_rof_feature_selected()
        
        # List of analysis option widgets to enable/disable
        analysis_widgets = [
            self.ui.analyze_scope_full_radio,
            self.ui.analyze_scope_prepost_radio,
            self.ui.analyze_level_subject_radio,
            self.ui.analyze_level_trial_radio,
            self.ui.analyze_design_paired_radio,
            self.ui.analyze_design_independent_radio,
            self.ui.analyze_type_parametric_radio,
            self.ui.analyze_type_nonparametric_radio,
        ]
        
        # Enable/disable based on whether ROF is selected
        for widget in analysis_widgets:
            widget.setEnabled(not is_rof)
        
        # Update model combo
        if is_rof:
            self.ui.analyze_model_combo.clear()
            self.ui.analyze_model_combo.addItem("TFCE (Threshold-Free Cluster Enhancement)")
        else:
            # Re-update model combo based on current selections
            self._update_analyze_model_combo()

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
            self.ui.feature2visualize_combo.currentText() and
            self.ui.feature2visualize_combo.currentText() not in self.ERROR_MESSAGES.values()):
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
        """Update feature comparison statistics in the UI with appropriate statistical method.
        
        Routes analysis based on:
        - Feature type (ROF uses TFCE)
        - Data scope (Full Recording vs Event-Related)
        - Analysis level (Subject-Level vs Trial-Level)
        - Study design (Paired vs Independent)
        - Test type (Parametric vs Non-parametric)
        """
        # Switch to Summary Statistics tab (index 3)
        self.ui.tabWidget.setCurrentIndex(3)
        
        self.ui.stats_textedit.clear()

        selected_feature = self._get_selected_feature_code()
        
        # Get current analysis settings
        is_rof = self._is_rof_feature_selected()
        is_prepost = self.ui.analyze_scope_prepost_radio.isChecked()
        is_subject_level = self.ui.analyze_level_subject_radio.isChecked()
        is_trial_level = self.ui.analyze_level_trial_radio.isChecked()
        is_paired = self.ui.analyze_design_paired_radio.isChecked()
        is_parametric = self.ui.analyze_type_parametric_radio.isChecked()
        selected_model = self.ui.analyze_model_combo.currentText()

        # 1. ROF features - always use TFCE
        if is_rof and self._has_rof_data():
            self.ui.stats_textedit.appendPlainText(
                "ROF Feature Analysis\n"
                "=" * 60 + "\n"
                "Using TFCE (Threshold-Free Cluster Enhancement) with sign-flipping permutation...\n"
            )
            self._perform_rof_tfce_analysis()
            return

        # 2. Pre/Post Event scope with trial-level data
        if is_prepost or self._is_pre_post_event_feature_selected():
            if is_trial_level:
                self.ui.stats_textedit.appendPlainText(
                    f"Event-Related Trial-Level Analysis\n"
                    f"{'=' * 60}\n"
                    f"Model: {selected_model}\n\n"
                )
                self._perform_trial_level_analysis(is_paired, is_parametric, selected_model)
            else:
                self.ui.stats_textedit.appendPlainText(
                    f"Event-Related Subject-Level Analysis\n"
                    f"{'=' * 60}\n"
                    f"Model: {selected_model}\n\n"
                )
                self._perform_pre_post_event_analysis()
            return

        # 3. Sliding window features
        if self._is_sliding_feature_selected():
            if is_trial_level:
                self.ui.stats_textedit.appendPlainText(
                    f"Sliding Window Trial-Level Analysis\n"
                    f"{'=' * 60}\n"
                    f"Model: {selected_model}\n\n"
                )
                self._perform_trial_level_analysis(is_paired, is_parametric, selected_model)
            else:
                self.ui.stats_textedit.appendPlainText(
                    f"Sliding Window Subject-Level Analysis\n"
                    f"{'=' * 60}\n"
                    f"Model: {selected_model}\n\n"
                )
                self._perform_sliding_feature_analysis()
            return

        # 4. Standard features - route by analysis level and test type
        if is_trial_level:
            self.ui.stats_textedit.appendPlainText(
                f"Trial-Level Analysis\n"
                f"{'=' * 60}\n"
                f"Model: {selected_model}\n\n"
            )
            self._perform_trial_level_analysis(is_paired, is_parametric, selected_model)
            return

        # 5. Subject-level analysis (default)
        self._perform_subject_level_analysis(is_paired, is_parametric, selected_model)

    def _perform_subject_level_analysis(
        self, is_paired: bool, is_parametric: bool, model_name: str
    ) -> None:
        """Perform subject-level statistical analysis.
        
        Args:
            is_paired: Whether to use paired tests
            is_parametric: Whether to use parametric tests
            model_name: Name of the selected statistical model
        """
        # Get analysis information
        (
            feature_list,
            test_results,
            p_values,
            adjusted_p_values,
        ) = self._perform_feature_comparison_analysis(is_paired, is_parametric)

        if not feature_list:
            selected_feature = self._get_selected_feature_code()
            error_msg = (
                f"No features available for statistical comparison.\n\n"
                f"Selected feature: {selected_feature}\n\n"
                f"This may happen if:\n"
                f"1. The selected feature '{selected_feature}' is not present in the loaded feature files\n"
                f"2. The feature files don't contain data for this feature\n"
                f"3. The files need to be re-extracted with updated feature extraction code\n"
            )
            self.ui.stats_textedit.appendPlainText(error_msg)
            return

        # Display header information
        self._display_analysis_header()

        # Display main results table
        self._display_statistical_results_table(
            feature_list, test_results, p_values, adjusted_p_values
        )

        # Display summary statistics
        self._display_statistical_summary(p_values, adjusted_p_values)

    def _perform_trial_level_analysis(
        self, is_paired: bool, is_parametric: bool, model_name: str
    ) -> None:
        """Perform trial-level statistical analysis using GEE or LMM.
        
        Args:
            is_paired: Whether subjects are matched across conditions
            is_parametric: Whether to use parametric models
            model_name: Name of the selected statistical model
        """
        result = self._get_common_features()
        if result is None or result[0] is None:
            self.ui.stats_textedit.appendPlainText("No features available for analysis.\n")
            return

        common_features_df, _ = result
        selected_feature = self._get_selected_feature_code()
        
        # Get feature columns
        feature_columns = [
            col for col in common_features_df.columns
            if col == selected_feature or col.startswith(selected_feature + "_")
        ]
        
        if not feature_columns:
            self.ui.stats_textedit.appendPlainText(
                f"No columns found for feature: {selected_feature}\n"
            )
            return

        # Display header
        self._display_analysis_header()
        
        # Perform GEE or LMM analysis
        if "GEE" in model_name or "Linear Mixed" in model_name:
            self._perform_gee_analysis(
                common_features_df,
                feature_columns,
                is_paired,
                family_name=gee_family_from_model_name(model_name),
            )
        else:
            # Fallback to aggregated subject-level analysis
            self.ui.stats_textedit.appendPlainText(
                f"Note: {model_name} will be implemented in a future update.\n"
                f"Currently using subject-level aggregation.\n\n"
            )
            self._perform_subject_level_analysis(is_paired, is_parametric, model_name)

    def _perform_gee_analysis(
        self,
        data_df: pd.DataFrame,
        feature_columns: List[str],
        is_paired: bool,
        family_name: str = "gaussian",
    ) -> None:
        """Perform Generalized Estimating Equations analysis.
        
        Args:
            data_df: DataFrame with feature data
            feature_columns: List of feature column names to analyze
            is_paired: Whether design is paired (within-subject)
            family_name: ``'gaussian'`` for an identity link, or ``'gamma'`` for a
                log link suited to strictly positive, right-skewed outcomes such
                as trial-level coverage.
        """
        try:
            # Prepare data for GEE
            # Extract subject ID from filename
            data_df = data_df.copy()
            data_df['Subject'] = data_df['Filename'].apply(self._extract_subject_id)
            
            results_text = []
            p_values = []
            
            for feat_col in feature_columns:
                # Create long-format data for this feature
                analysis_df = data_df[['Subject', 'Study', 'Filename', feat_col]].copy()
                analysis_df = analysis_df.dropna(subset=[feat_col])
                
                if analysis_df.empty:
                    continue
                
                # Check if we have enough data
                n_subjects = analysis_df['Subject'].nunique()
                n_per_group = analysis_df.groupby('Study').size()
                
                if n_subjects < 3:
                    results_text.append(f"{feat_col}: Insufficient subjects (n={n_subjects})")
                    continue
                
                try:
                    # Fit GEE model
                    # Create condition variable (0 = Study1, 1 = Study2)
                    study1_name = self.comet_tbx_study1.study_name
                    analysis_df['Condition'] = (analysis_df['Study'] != study1_name).astype(int)
                    
                    if family_name == "gamma":
                        # A log link requires a strictly positive response, so
                        # shift the whole feature if any value is <= 0. Shifting
                        # is only valid for a log link; under identity it would
                        # silently bias the intercept.
                        min_val = analysis_df[feat_col].min()
                        if min_val <= 0:
                            analysis_df[feat_col] = analysis_df[feat_col] - min_val + 0.001
                        family = Gamma(link=Log())
                    else:
                        family = Gaussian(link=Identity())

                    # Fit GEE with exchangeable correlation structure
                    formula = f"`{feat_col}` ~ Condition"
                    model = GEE.from_formula(
                        formula=formula,
                        groups="Subject",
                        data=analysis_df,
                        family=family,
                        cov_struct=Exchangeable() if is_paired else Independence(),
                    )
                    gee_result = model.fit()
                    
                    # Extract results
                    coef = gee_result.params.get('Condition', np.nan)
                    stderr = gee_result.bse.get('Condition', np.nan)
                    z_val = gee_result.tvalues.get('Condition', np.nan)
                    p_val = gee_result.pvalues.get('Condition', np.nan)
                    
                    p_values.append(p_val)
                    
                    sig = self._get_significance_level(p_val)
                    results_text.append(
                        f"{feat_col}: β={coef:.4f}, SE={stderr:.4f}, z={z_val:.2f}, p={p_val:.4f} {sig}"
                    )
                    
                except Exception as e:
                    results_text.append(f"{feat_col}: Analysis failed - {str(e)[:50]}")
            
            # Display results
            self.ui.stats_textedit.appendPlainText("GEE Analysis Results\n" + "-" * 40 + "\n")
            self.ui.stats_textedit.appendPlainText("\n".join(results_text))
            
            # Apply multiple testing correction
            if p_values:
                correction_method = _correction_method(self.ui.analyze_correction_combo.currentText())

                try:
                    _, adj_p_values, _, _ = multipletests(p_values, method=correction_method)
                    
                    self.ui.stats_textedit.appendPlainText(
                        f"\n\nMultiple Testing Correction ({self.ui.analyze_correction_combo.currentText()}):\n"
                    )
                    n_sig = sum(1 for p in adj_p_values if p < 0.05)
                    self.ui.stats_textedit.appendPlainText(
                        f"Significant features after correction: {n_sig}/{len(adj_p_values)}\n"
                    )
                except Exception as e:
                    self.ui.stats_textedit.appendPlainText(
                        f"\nCould not apply correction: {str(e)}\n"
                    )
            
        except Exception as e:
            self.ui.stats_textedit.appendPlainText(
                f"GEE Analysis Error: {str(e)}\n"
                f"Falling back to standard t-test analysis.\n\n"
            )
            traceback.print_exc()
            self._perform_subject_level_analysis(is_paired, True, "t-test")

    def _display_analysis_header(self) -> None:
        """Display header information for the statistical analysis."""
        # Get comparison information
        comparison_info = self._get_comparison_info()
        selected_feature = self._get_selected_feature_code()
        feature_full_name = self.feature_list_dictionary.get(selected_feature, selected_feature)
        
        # Get test type description
        is_paired = self.ui.analyze_design_paired_radio.isChecked()
        is_parametric = self.ui.analyze_type_parametric_radio.isChecked()
        is_subject_level = self.ui.analyze_level_subject_radio.isChecked()
        selected_model = self.ui.analyze_model_combo.currentText()
        
        # Determine analysis level description
        level_desc = "Subject-Level" if is_subject_level else "Trial-Level"
        design_desc = "Paired" if is_paired else "Independent"
        type_desc = "Parametric" if is_parametric else "Non-parametric"

        # Analysis header
        header_text = (
            f"Statistical Analysis Report\n"
            f"{'=' * 60}\n"
            f"Feature: {feature_full_name}\n"
            f"Comparison: {comparison_info['comparison_type']}\n"
            f"Study 1: {comparison_info['study1_name']} ({comparison_info['study1_files']} files)\n"
            f"Study 2: {comparison_info['study2_name']} ({comparison_info['study2_files']} files)\n"
            f"Analysis Level: {level_desc}\n"
            f"Study Design: {design_desc}\n"
            f"Test Type: {type_desc}\n"
            f"Statistical Model: {selected_model}\n"
            f"Multiple Testing Correction: {self.ui.analyze_correction_combo.currentText()}\n"
            f"{'=' * 60}\n"
        )
        self.ui.stats_textedit.appendPlainText(header_text)

    def _get_statistic_column_name(self) -> str:
        """Get the appropriate column name for the test statistic."""
        is_parametric = self.ui.analyze_type_parametric_radio.isChecked()
        is_paired = self.ui.analyze_design_paired_radio.isChecked()
        
        if is_parametric:
            return "t-statistic"
        else:
            if is_paired:
                return "W-statistic"  # Wilcoxon
            else:
                return "U-statistic"  # Mann-Whitney

    def _display_statistical_results_table(
        self,
        feature_list: List[str],
        t_test_results: Dict[str, float],
        p_values: List[float],
        adjusted_p_values: List[float]
    ) -> None:
        """Display the main statistical results in a formatted table."""
        stat_col_name = self._get_statistic_column_name()
        
        # Table header
        self.ui.stats_textedit.appendPlainText("Statistical Results:")
        self.ui.stats_textedit.appendPlainText(
            f"{'Microstate Feature':<35} {stat_col_name:>12} {'p-value':>12} {'adj. p-value':>12} {'Significance':>12}"
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

        # Check if this is a pre/post event comparison
        is_pre_post_event = self._is_pre_post_event_feature_selected()

        if self.ui.compare_two_studies_radio.isChecked():
            comparison_type = "Two Independent Studies"
        elif self.ui.compare_within_study_radio.isChecked():
            if is_pre_post_event:
                comparison_type = "Pre vs Post Event Comparison"
                # For event-related comparisons, we're comparing within the same files
                study1_files = "Same files (Pre-event)"
                study2_files = "Same files (Post-event)"
            else:
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
        is_paired: bool = None,
        is_parametric: bool = None,
    ) -> Tuple[List[str], Dict[str, float], List[float], List[float]]:
        """Perform statistical comparison between two studies.

        Args:
            is_paired: Use paired tests (if None, reads from UI)
            is_parametric: Use parametric tests (if None, reads from UI)

        Returns:
            Tuple of (feature_list, test_results, p_values, adjusted_p_values).
        """
        # Get settings from UI if not provided
        if is_paired is None:
            is_paired = self.ui.analyze_design_paired_radio.isChecked()
        if is_parametric is None:
            is_parametric = self.ui.analyze_type_parametric_radio.isChecked()

        result = self._get_common_features()
        if result is None or result[0] is None:
            return [], {}, [], []

        common_features_df, _ = result
        selected_feature = self._get_selected_feature_code()

        # Organize data for the selected feature
        study1_df, study2_df = self._separate_studies_data(
            common_features_df, selected_feature
        )

        # Get feature columns
        feature_list = [
            col for col in study1_df.columns if col not in ["Study", "Filename"]
        ]

        # Check if we have any features to compare
        if not feature_list:
            return [], {}, [], []

        # Perform statistical tests based on settings
        test_results, p_values = self._calculate_statistical_tests(
            study1_df, study2_df, feature_list, is_paired, is_parametric
        )

        # Adjust p-values for multiple testing only if we have p-values
        if p_values:
            adjusted_p_values = multipletests(
                p_values, method=_correction_method(self.ui.analyze_correction_combo.currentText())
            )[1]
        else:
            adjusted_p_values = []

        return feature_list, test_results, p_values, adjusted_p_values

    def _separate_studies_data(
        self, common_features_df: pd.DataFrame, selected_feature: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate combined features DataFrame by study."""
        # Match columns that start with the feature code followed by underscore
        # This ensures "DUR" doesn't match "DUR_SD_A" but matches "DUR_A"
        # Also handle single-column features (no underscore suffix)
        feature_columns = []
        for col in common_features_df.columns:
            if col == selected_feature:  # Exact match for single-column features
                feature_columns.append(col)
            elif col.startswith(selected_feature + "_"):  # Must have underscore after feature code
                feature_columns.append(col)

        columns_to_keep = ["Filename", "Study"] + feature_columns
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
        """Calculate t-tests for each feature using the current UI settings."""
        is_paired = self.ui.analyze_design_paired_radio.isChecked()
        is_parametric = self.ui.analyze_type_parametric_radio.isChecked()
        return self._calculate_statistical_tests(
            study1_df, study2_df, feature_list, is_paired, is_parametric
        )

    def _calculate_statistical_tests(
        self,
        study1_df: pd.DataFrame,
        study2_df: pd.DataFrame,
        feature_list: List[str],
        is_paired: bool,
        is_parametric: bool,
    ) -> Tuple[Dict[str, float], List[float]]:
        """Calculate statistical tests for each feature.
        
        Args:
            study1_df: DataFrame with Study 1 data
            study2_df: DataFrame with Study 2 data
            feature_list: List of feature column names
            is_paired: Use paired tests
            is_parametric: Use parametric tests
            
        Returns:
            Tuple of (test_statistic_dict, p_values_list)
        """
        test_results = {}
        p_values = []

        for feat in feature_list:
            study1_values = np.array(study1_df[feat].dropna().tolist())
            study2_values = np.array(study2_df[feat].dropna().tolist())
            
            # Skip if insufficient data
            if len(study1_values) < 2 or len(study2_values) < 2:
                test_results[feat] = np.nan
                p_values.append(1.0)
                continue

            try:
                if is_parametric:
                    # Parametric tests
                    if is_paired:
                        # Paired t-test (requires equal lengths)
                        min_len = min(len(study1_values), len(study2_values))
                        statistic, p_value = ttest_rel(
                            study1_values[:min_len], study2_values[:min_len]
                        )
                    else:
                        # Independent t-test, Welch-corrected if variances differ
                        statistic, p_value, _ = independent_ttest(
                            study1_values, study2_values
                        )
                else:
                    # Non-parametric tests
                    if is_paired:
                        # Wilcoxon signed-rank test (requires equal lengths)
                        min_len = min(len(study1_values), len(study2_values))
                        statistic, p_value = wilcoxon(
                            study1_values[:min_len], study2_values[:min_len]
                        )
                    else:
                        # Mann-Whitney U test
                        statistic, p_value = mannwhitneyu(
                            study1_values, study2_values, alternative='two-sided'
                        )

                test_results[feat] = statistic
                p_values.append(p_value)
                
            except Exception as e:
                # Handle test failures gracefully
                test_results[feat] = np.nan
                p_values.append(1.0)

        return test_results, p_values

    # ==================== PRE/POST EVENT ANALYSIS ====================

    def _perform_pre_post_event_analysis(self) -> None:
        """Perform mixed-effects analysis for pre/post event features.

        This method specifically handles pre/post event features extracted from epoched data.
        Uses mixed-effects models to account for:
        - Within-subject correlation (multiple trials per subject)
        - Variable number of trials per subject
        """
        selected_feature = self._get_selected_feature_code()

        # Get the appropriate feature type and mode (should be pre_post)
        feature_type, feature_mode = self._get_compatible_features()
        if feature_type is None or feature_mode is None or feature_mode != "pre_post":
            self.ui.stats_textedit.appendPlainText(
                f"Could not load pre/post event features for {selected_feature}.\n"
                f"Expected mode: pre_post, got: {feature_mode}\n"
                f"Please ensure pre/post event features have been extracted."
            )
            return

        try:
            # Load pre/post features
            features_df = self._load_features_safely(
                self.comet_tbx_study1, feature_type, feature_mode
            )

            if features_df is None or features_df.empty:
                self.ui.stats_textedit.appendPlainText(
                    f"Could not load pre/post event features for {selected_feature}.\n"
                    f"Please ensure pre/post event features have been extracted."
                )
                return

            # Filter by selected files if needed
            selected_files = [item.text() for item in self.ui.study1_file_list.selectedItems()]
            if selected_files:
                # For pre/post features, filenames might have _Pre or _Post suffix
                # We need to match the base filename
                def extract_base_filename(filename):
                    # Remove _Pre or _Post suffix
                    if filename.endswith("_Pre") or filename.endswith("_Post"):
                        return filename.rsplit("_", 1)[0]
                    return filename

                features_df["base_filename"] = features_df["Filename"].apply(extract_base_filename)
                selected_base = [extract_base_filename(f) for f in selected_files]
                features_df = features_df[features_df["base_filename"].isin(selected_base)]

            if features_df.empty:
                self.ui.stats_textedit.appendPlainText(
                    "No pre/post event features found for selected files."
                )
                return

            # Check if Window_Type column exists
            if "Window_Type" not in features_df.columns:
                self.ui.stats_textedit.appendPlainText(
                    "Error: Pre/post event features missing 'Window_Type' column.\n"
                    "Please re-extract features with updated code."
                )
                return

                # Get feature columns for the selected feature
            filter_cols = [col for col in features_df.columns if col.startswith(selected_feature + "_")]
            if not filter_cols and selected_feature in features_df.columns:
                filter_cols = [selected_feature]
            filter_cols.sort()

            if not filter_cols:
                self.ui.stats_textedit.appendPlainText(
                    f"No features found for '{selected_feature}' in pre/post event data."
                )
                return

            # Display analysis header
            comparison_info = self._get_comparison_info()
            header_text = (
                f"Pre/Post Event Mixed-Effects Analysis\n"
                f"{'=' * 80}\n"
                f"Feature: {self.feature_list_dictionary.get(selected_feature, selected_feature)}\n"
                f"Comparison: {comparison_info['comparison_type']}\n"
                f"Analysis: Mixed-effects model accounting for within-subject correlation\n"
                f"Microstates analyzed: {len(filter_cols)}\n"
                f"{'=' * 80}\n\n"
            )
            self.ui.stats_textedit.appendPlainText(header_text)

            # Perform mixed-effects analysis for each microstate
            results = []
            for feat_col in filter_cols:
                microstate_label = "_".join(feat_col.split("_")[1:]) if "_" in feat_col else feat_col

                # Prepare data for mixed-effects model
                data_list = []

                # Extract subject ID from filename (remove _Pre/_Post and trial info)
                def extract_subject_id(filename):
                    # Remove _Pre/_Post suffix
                    base = filename.rsplit("_Pre", 1)[0].rsplit("_Post", 1)[0]
                    # Remove trial info if present
                    base = re.sub(r'_trial\d+', '', base)
                    return base

                for idx, row in features_df.iterrows():
                    if pd.notna(row[feat_col]):
                        # Code Pre as 0, Post as 1
                        window_type = str(row['Window_Type']).lower()
                        if 'pre' in window_type:
                            group = 0
                        elif 'post' in window_type:
                            group = 1
                        else:
                            continue  # Skip unknown window types

                        data_list.append({
                            'Subject': extract_subject_id(row['Filename']),
                            'Group': group,
                            'Value': row[feat_col]
                        })

                if len(data_list) == 0:
                    continue

                # Create dataframe for mixed-effects model
                analysis_df = pd.DataFrame(data_list)

                # Get summary statistics
                n_subjects = len(analysis_df['Subject'].unique())
                n_pre = len(analysis_df[analysis_df['Group'] == 0])
                n_post = len(analysis_df[analysis_df['Group'] == 1])
                mean_pre = analysis_df[analysis_df['Group'] == 0]['Value'].mean()
                mean_post = analysis_df[analysis_df['Group'] == 1]['Value'].mean()

                if n_subjects < 2:
                    continue

                try:
                    # Fit mixed-effects model
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=Warning)

                        model = smf.mixedlm("Value ~ Group", data=analysis_df,
                                            groups=analysis_df["Subject"],
                                            re_formula="1")
                        result = model.fit(method='lbfgs', maxiter=100)

                    # Extract results
                    coef = result.params['Group']
                    p_val = result.pvalues['Group']
                    ci_low, ci_high = result.conf_int().loc['Group']

                    # Calculate effect size
                    pooled_sd = np.sqrt(result.scale)
                    cohens_d = coef / pooled_sd if pooled_sd > 0 else 0

                    results.append({
                        "microstate": microstate_label,
                        "n_subjects": n_subjects,
                        "n_pre": n_pre,
                        "n_post": n_post,
                        "mean_pre": mean_pre,
                        "mean_post": mean_post,
                        "diff": coef,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_val": p_val,
                        "cohens_d": cohens_d,
                        "converged": result.converged
                    })

                except Exception as e:
                    self.ui.stats_textedit.appendPlainText(
                        f"Error fitting model for {feat_col}: {str(e)}"
                    )
                    continue

            if not results:
                self.ui.stats_textedit.appendPlainText("No valid data for analysis.")
                return

            # Apply multiple testing correction
            p_values = [r["p_val"] for r in results]
            _, adj_p_values, _, _ = multipletests(
                p_values, method=_correction_method(self.ui.analyze_correction_combo.currentText())
            )

            for i, result in enumerate(results):
                result["adj_p_val"] = adj_p_values[i]

            # Display results
            self.ui.stats_textedit.appendPlainText(
                "\nMixed Effects Model Results (Value ~ Group + (1|Subject)):\n"
                f"{'-' * 130}\n"
                f"{'Microstate':<12} {'N_Subj':<10} {'N_Obs':<16} {'Mean(Pre)':<12} {'Mean(Post)':<12} "
                f"{'Diff':<10} {'95% CI':<26} {'p-value':<12} {'Adj.p':<12} {'d':<8}\n"
                f"{'-' * 130}"
            )

            for result in results:
                sig_marker = " ***" if result['adj_p_val'] < 0.001 else " **" if result['adj_p_val'] < 0.01 else " *" if \
                result['adj_p_val'] < 0.05 else ""
                conv_marker = "" if result.get('converged', True) else " [!]"

                n_obs_str = f"{result['n_pre']}/{result['n_post']}"
                ci_str = f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"

                p_str = f"{result['p_val']:.2e}" if result['p_val'] < 0.001 else f"{result['p_val']:.6f}"
                adj_p_str = f"{result['adj_p_val']:.2e}" if result[
                                                                'adj_p_val'] < 0.001 else f"{result['adj_p_val']:.6f}"

                self.ui.stats_textedit.appendPlainText(
                    f"{result['microstate']:<12} {result['n_subjects']:<10} {n_obs_str:<16} "
                    f"{result['mean_pre']:<12.4f} {result['mean_post']:<12.4f} "
                    f"{result['diff']:<10.4f} {ci_str:<26} "
                    f"{p_str:<12} {adj_p_str:<12} {result['cohens_d']:<8.3f}"
                    f"{sig_marker}{conv_marker}"
                )

            # Summary
            n_sig = sum(1 for r in results if r["adj_p_val"] < 0.05)
            n_converged = sum(r.get('converged', True) for r in results)

            self.ui.stats_textedit.appendPlainText(
                f"\n{'-' * 130}\n"
                f"Summary:\n"
                f"  Significant differences (adjusted p < 0.05): {n_sig}/{len(results)}\n"
                f"  Models converged: {n_converged}/{len(results)}\n"
                f"  Multiple testing correction: {self.ui.analyze_correction_combo.currentText()}\n\n"
                f"Interpretation:\n"
                f"  Model: Value ~ Group + (1|Subject) accounts for within-subject correlation\n"
                f"  Group: Pre (0) vs Post (1) event window\n"
                f"  N_Subj: Number of subjects analyzed\n"
                f"  N_Obs: Number of observations (Pre/Post)\n"
                f"  Diff: Mean difference (Post - Pre)\n"
                f"  95% CI: Confidence interval for difference\n"
                f"  d: Cohen's d effect size\n"
                f"  Significance: *** p<0.001, ** p<0.01, * p<0.05\n"
                f"  [!]: Model convergence warning\n"
            )

        except Exception as e:
            self._show_error(
                "Pre/Post Event Analysis Error",
                f"Failed to perform pre/post event analysis:\n{str(e)}\n\n{traceback.format_exc()}"
            )

            # ==================== SLIDING FEATURE (REPEATED MEASURES) ANALYSIS ====================

    def _perform_sliding_feature_analysis(self) -> None:
        """Perform repeated measures analysis for sliding window features.

        This method handles two scenarios:
        1. Fixed sliding windows: Each window is treated as a repeated measure
        2. Event-based windows: Pre/post event comparisons
        """
        selected_feature = self._get_selected_feature_code()

        # Get the appropriate feature type and mode
        feature_type, feature_mode = self._get_compatible_features()
        if feature_type is None or feature_mode is None:
            self.ui.stats_textedit.appendPlainText(
                f"Could not determine appropriate feature type and mode for {selected_feature}.\n"
                f"Please ensure the feature has been extracted."
            )
            return

        try:
                # Load features using the determined mode
                features_df_study1 = self._load_features_safely(
                    self.comet_tbx_study1, feature_type, feature_mode
                )

                if features_df_study1 is None:
                    self.ui.stats_textedit.appendPlainText(
                        f"Could not load sliding features for {selected_feature}.\n"
                        f"Please ensure sliding features have been extracted."
                    )
                    return

                # Get study2 features based on comparison mode
                if self.ui.compare_two_studies_radio.isChecked():
                    # Two separate studies
                    features_df_study2 = self._load_features_safely(
                        self.comet_tbx_study2, feature_type, feature_mode
                    )
                    if features_df_study2 is None:
                        self.ui.stats_textedit.appendPlainText(
                            "Could not load sliding features from Study 2."
                        )
                        return
                    # Filter each study by its respective file list
                    features_df_study1 = self._filter_features_by_filelist(
                        features_df_study1, self.ui.study1_file_list
                    )
                    features_df_study2 = self._filter_features_by_filelist(
                        features_df_study2, self.ui.study2_file_list
                    )
                elif self.ui.compare_within_study_radio.isChecked():
                    # Within-study comparison: both groups come from the same data
                    # Keep the original data for filtering by each group separately
                    features_df_full = features_df_study1.copy()

                    # Check if this is an event-related feature (pre/post comparison)
                    if "Window_Type" in features_df_full.columns:
                        # For event-related features, split by Window_Type (Pre vs Post)
                        features_df_study1 = features_df_full[
                            features_df_full['Window_Type'].isin(['Pre', 'pre'])
                        ].copy()
                        features_df_study2 = features_df_full[
                            features_df_full['Window_Type'].isin(['Post', 'post'])
                        ].copy()
                    else:
                        # For regular within-study comparisons, filter by file lists
                        features_df_study1 = self._filter_features_by_filelist(
                            features_df_full, self.ui.study1_file_list
                        )
                        features_df_study2 = self._filter_features_by_filelist(
                            features_df_full, self.ui.study2_file_list
                        )
                else:
                    self.ui.stats_textedit.appendPlainText(
                        "Repeated measures analysis for synthetic comparisons is not yet implemented."
                    )
                    return

                # Detect if this is event-based or fixed windows
                is_event_based = "Event_name" in features_df_study1.columns

                if is_event_based:
                    self._analyze_event_based_sliding(
                        features_df_study1, features_df_study2, selected_feature
                    )
                else:
                    self._analyze_fixed_window_sliding(
                        features_df_study1, features_df_study2, selected_feature
                    )

        except Exception as e:
            self._show_error(
                "Sliding Feature Analysis Error",
                f"Failed to perform repeated measures analysis:\n{str(e)}\n\n{traceback.format_exc()}"
            )

    def _analyze_fixed_window_sliding(
            self, df1: pd.DataFrame, df2: pd.DataFrame, selected_feature: str
    ) -> None:
        """Analyze fixed sliding window features using repeated measures.

        For fixed windows with within-subject design, we need to match subjects
        between conditions and use a paired analysis.
        """
        # Get feature columns - match exactly (feature_microstate format)
        # This ensures we only get COV_A, COV_B, etc., not other columns
        feature_cols = []
        for col in df1.columns:
            if col == selected_feature:  # Exact match
                feature_cols.append(col)
            elif col.startswith(selected_feature + "_"):
                    # Make sure it's in the format FEATURE_MICROSTATE (e.g., COV_A)
                    # and not something else like COV_SD_A
                    parts = col.split("_")
                    if len(parts) == 2 and parts[0] == selected_feature:
                        feature_cols.append(col)

            if not feature_cols:
                self.ui.stats_textedit.appendPlainText(
                    f"No feature columns found for '{selected_feature}' in sliding data.\n"
                    f"Available columns: {list(df1.columns)[:20]}"
                )
                return

            # Check if dataframes have data
            if df1.empty:
                self.ui.stats_textedit.appendPlainText(
                    f"Study 1 dataframe is empty after filtering."
                )
                return

            if df2.empty:
                self.ui.stats_textedit.appendPlainText(
                    f"Study 2 dataframe is empty after filtering."
                )
                return

            # Print header
            comparison_info = self._get_comparison_info()
            header_text = (
                f"Repeated Measures Analysis - Fixed Sliding Windows\n"
                f"{'=' * 60}\n"
                f"Feature: {self.feature_list_dictionary.get(selected_feature, selected_feature)}\n"
                f"Comparison: {comparison_info['comparison_type']}\n"
                f"Study 1: {comparison_info['study1_name']} ({comparison_info['study1_files']} files)\n"
                f"Study 2: {comparison_info['study2_name']} ({comparison_info['study2_files']} files)\n"
                f"Analysis: Repeated measures across {len(feature_cols)} microstate(s)\n"
                f"Microstate columns: {feature_cols}\n"
                f"Study 1 data: {len(df1)} windows\n"
                f"Study 2 data: {len(df2)} windows\n"
                f"{'=' * 60}\n\n"
            )
            self.ui.stats_textedit.appendPlainText(header_text)

            # Check if this is a within-subject design
            is_within_subject = self.ui.compare_within_study_radio.isChecked()

            # For each microstate, perform analysis
            results = []
            for feat_col in feature_cols:
                # Get microstate label
                microstate_label = "_".join(feat_col.split("_")[1:]) if "_" in feat_col else feat_col

                if is_within_subject:
                    # For within-subject: aggregate per subject, then paired t-test
                    # Extract subject IDs and aggregate
                    def extract_subject_id(filename):
                        """Extract subject ID from filename (remove condition suffix)."""
                        # Handle BIDS format (e.g., sub-01_ses-session1_task-eyesopen_...)
                        # Remove task specification onwards
                        match = re.search(r'(.*?)_task-[^_]+', filename)
                        if match:
                            return match.group(1)

                        # Fallback: Remove common suffixes like _eyesopen, _eyesclosed, etc.
                        base = re.sub(r'_(eyesopen|eyesclosed|open|closed|pre|post).*', '', filename)
                        return base

                    # Aggregate windows per subject
                    subject_means_1 = df1.groupby('Filename')[feat_col].mean()
                    subject_means_2 = df2.groupby('Filename')[feat_col].mean()

                    # Extract subject IDs
                    subj_ids_1 = {extract_subject_id(fname): val for fname, val in subject_means_1.items()}
                    subj_ids_2 = {extract_subject_id(fname): val for fname, val in subject_means_2.items()}

                    # Find matched subjects
                    matched_subjects = set(subj_ids_1.keys()) & set(subj_ids_2.keys())

                    if len(matched_subjects) < 3:
                        self.ui.stats_textedit.appendPlainText(
                            f"Skipping {feat_col}: Insufficient matched subjects ({len(matched_subjects)})"
                        )
                        continue

                    # Get matched pairs
                    values_1 = [subj_ids_1[subj] for subj in matched_subjects]
                    values_2 = [subj_ids_2[subj] for subj in matched_subjects]

                    # Remove any NaN values
                    valid_pairs = [(v1, v2) for v1, v2 in zip(values_1, values_2)
                                   if pd.notna(v1) and pd.notna(v2)]

                    if len(valid_pairs) < 3:
                        continue

                    values_1 = [p[0] for p in valid_pairs]
                    values_2 = [p[1] for p in valid_pairs]

                    # Perform paired t-test
                    t_stat, p_val = ttest_rel(values_1, values_2)

                    mean1 = np.mean(values_1)
                    mean2 = np.mean(values_2)
                    diff = mean2 - mean1

                    # Calculate paired Cohen's d
                    differences = np.array(values_2) - np.array(values_1)
                    cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences,
                                                                                            ddof=1) > 0 else 0

                    # Calculate 95% CI for the difference
                    se_diff = np.std(differences, ddof=1) / np.sqrt(len(differences))
                    ci_low = diff - t.ppf(0.975, len(differences) - 1) * se_diff
                    ci_high = diff + t.ppf(0.975, len(differences) - 1) * se_diff

                    n_windows_1 = len(df1[df1[feat_col].notna()])
                    n_windows_2 = len(df2[df2[feat_col].notna()])

                    results.append({
                        "microstate": microstate_label,
                        "n_subjects": len(valid_pairs),
                        "n_windows_1": n_windows_1,
                        "n_windows_2": n_windows_2,
                        "mean1": mean1,
                        "mean2": mean2,
                        "diff": diff,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "t_stat": t_stat,
                        "p_val": p_val,
                        "cohens_d": cohens_d,
                        "is_paired": True
                    })

                else:
                    # For independent samples: mixed effects model
                    data_list = []

                    # Add Study 1 data
                    if 'Filename' in df1.columns:
                        for idx, row in df1.iterrows():
                            if pd.notna(row[feat_col]):
                                data_list.append({
                                    'Subject': row['Filename'],
                                    'Group': 0,
                                    'Value': row[feat_col]
                                })

                    # Add Study 2 data
                    if 'Filename' in df2.columns:
                        for idx, row in df2.iterrows():
                            if pd.notna(row[feat_col]):
                                data_list.append({
                                    'Subject': row['Filename'],
                                    'Group': 1,
                                    'Value': row[feat_col]
                                })

                    if len(data_list) == 0:
                        continue

                    analysis_df = pd.DataFrame(data_list)

                    n_subjects_1 = len(analysis_df[analysis_df['Group'] == 0]['Subject'].unique())
                    n_subjects_2 = len(analysis_df[analysis_df['Group'] == 1]['Subject'].unique())
                    n_windows_1 = len(analysis_df[analysis_df['Group'] == 0])
                    n_windows_2 = len(analysis_df[analysis_df['Group'] == 1])
                    mean1 = analysis_df[analysis_df['Group'] == 0]['Value'].mean()
                    mean2 = analysis_df[analysis_df['Group'] == 1]['Value'].mean()

                    if n_subjects_1 < 2 or n_subjects_2 < 2:
                        continue

                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=Warning)

                            model = smf.mixedlm("Value ~ Group", data=analysis_df,
                                                groups=analysis_df["Subject"],
                                                re_formula="1")
                            result = model.fit(method='lbfgs', maxiter=100)

                        coef = result.params['Group']
                        p_val = result.pvalues['Group']
                        ci_low, ci_high = result.conf_int().loc['Group']

                        pooled_sd = np.sqrt(result.scale)
                        cohens_d = coef / pooled_sd if pooled_sd > 0 else 0

                        results.append({
                            "microstate": microstate_label,
                            "n_subjects_1": n_subjects_1,
                            "n_subjects_2": n_subjects_2,
                            "n_windows_1": n_windows_1,
                            "n_windows_2": n_windows_2,
                            "mean1": mean1,
                            "mean2": mean2,
                            "diff": coef,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "t_stat": None,
                            "p_val": p_val,
                            "cohens_d": cohens_d,
                            "converged": result.converged,
                            "is_paired": False
                        })

                    except Exception as e:
                        self.ui.stats_textedit.appendPlainText(
                            f"Error fitting model for {feat_col}: {str(e)}"
                        )
                        continue

            if not results:
                self.ui.stats_textedit.appendPlainText(
                    "\nNo valid data for analysis.\n"
                    f"This may indicate that the file filtering is not matching correctly.\n"
                    f"Study 1 files in list: {[self.ui.study1_file_list.item(i).text() for i in range(min(3, self.ui.study1_file_list.count()))]}\n"
                    f"Study 2 files in list: {[self.ui.study2_file_list.item(i).text() for i in range(min(3, self.ui.study2_file_list.count()))]}\n"
                    f"Filenames in sliding data: {df1['Filename'].unique()[:3].tolist() if 'Filename' in df1.columns else 'No Filename column'}"
                )
                return

            # Apply multiple testing correction
            p_values = [r["p_val"] for r in results]
            _, adj_p_values, _, _ = multipletests(
                p_values, method=_correction_method(self.ui.analyze_correction_combo.currentText())
            )

            for i, result in enumerate(results):
                result["adj_p_val"] = adj_p_values[i]

            # Display results header based on analysis type
            if is_within_subject:
                self.ui.stats_textedit.appendPlainText(
                    "\nPaired Analysis Results (Within-Subject Design):\n"
                    f"{'-' * 130}\n"
                    f"{'Microstate':<12} {'N_Pairs':<10} {'N_Windows':<16} {'Mean(Cond1)':<13} {'Mean(Cond2)':<13} "
                    f"{'Diff':<10} {'95% CI':<26} {'t-stat':<10} {'p-value':<12} {'Adj.p':<12} {'d':<8}\n"
                    f"{'-' * 130}"
                )

                for result in results:
                    sig_marker = " ***" if result['adj_p_val'] < 0.001 else " **" if result[
                                                                                         'adj_p_val'] < 0.01 else " *" if \
                    result['adj_p_val'] < 0.05 else ""

                    n_win_str = f"{result['n_windows_1']}/{result['n_windows_2']}"
                    ci_str = f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"

                    p_str = f"{result['p_val']:.2e}" if result['p_val'] < 0.001 else f"{result['p_val']:.6f}"
                    adj_p_str = f"{result['adj_p_val']:.2e}" if result[
                                                                    'adj_p_val'] < 0.001 else f"{result['adj_p_val']:.6f}"

                    self.ui.stats_textedit.appendPlainText(
                        f"{result['microstate']:<12} {result['n_subjects']:<10} {n_win_str:<16} "
                        f"{result['mean1']:<13.4f} {result['mean2']:<13.4f} "
                        f"{result['diff']:<10.4f} {ci_str:<26} "
                        f"{result['t_stat']:<10.4f} {p_str:<12} {adj_p_str:<12} {result['cohens_d']:<8.3f}"
                        f"{sig_marker}"
                    )

                n_sig = sum(1 for r in results if r["adj_p_val"] < 0.05)

                self.ui.stats_textedit.appendPlainText(
                    f"\n{'-' * 130}\n"
                    f"Significant differences (adjusted p < 0.05): {n_sig}/{len(results)}\n"
                    f"Multiple testing correction: {self.ui.analyze_correction_combo.currentText()}\n\n"
                    f"Note:\n"
                    f"  - Analysis: Paired t-test on subject-averaged values (windows aggregated per subject)\n"
                    f"  - N_Pairs: Number of matched subject pairs\n"
                    f"  - N_Windows: Total windows used for aggregation (Cond1/Cond2)\n"
                    f"  - Diff: Mean difference (Condition2 - Condition1)\n"
                    f"  - 95% CI: Confidence interval for mean difference\n"
                    f"  - d: Cohen's d effect size (for paired samples)\n"
                    f"  - Significance: *** p<0.001, ** p<0.01, * p<0.05\n"
                )
            else:
                self.ui.stats_textedit.appendPlainText(
                    "\nMixed Effects Model Results (Independent Groups):\n"
                    f"{'-' * 130}\n"
                    f"{'Microstate':<12} {'N_Subj':<12} {'N_Windows':<16} {'Mean(S1)':<12} {'Mean(S2)':<12} "
                    f"{'Diff':<10} {'95% CI':<26} {'p-value':<12} {'Adj.p':<12} {'d':<8}\n"
                    f"{'-' * 130}"
                )

                for result in results:
                    sig_marker = " ***" if result['adj_p_val'] < 0.001 else " **" if result[
                                                                                         'adj_p_val'] < 0.01 else " *" if \
                    result['adj_p_val'] < 0.05 else ""
                    conv_marker = "" if result.get('converged', True) else " [!]"

                    n_subj_str = f"{result['n_subjects_1']}/{result['n_subjects_2']}"
                    n_win_str = f"{result['n_windows_1']}/{result['n_windows_2']}"
                    ci_str = f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"

                    p_str = f"{result['p_val']:.2e}" if result['p_val'] < 0.001 else f"{result['p_val']:.6f}"
                    adj_p_str = f"{result['adj_p_val']:.2e}" if result[
                                                                    'adj_p_val'] < 0.001 else f"{result['adj_p_val']:.6f}"

                    self.ui.stats_textedit.appendPlainText(
                        f"{result['microstate']:<12} {n_subj_str:<12} {n_win_str:<16} "
                        f"{result['mean1']:<12.4f} {result['mean2']:<12.4f} "
                        f"{result['diff']:<10.4f} {ci_str:<26} "
                        f"{p_str:<12} {adj_p_str:<12} {result['cohens_d']:<8.3f}"
                        f"{sig_marker}{conv_marker}"
                    )

                n_sig = sum(1 for r in results if r["adj_p_val"] < 0.05)
                n_converged = sum(r.get('converged', True) for r in results)

                self.ui.stats_textedit.appendPlainText(
                    f"\n{'-' * 130}\n"
                    f"Significant differences (adjusted p < 0.05): {n_sig}/{len(results)}\n"
                    f"Models converged: {n_converged}/{len(results)}\n"
                    f"Multiple testing correction: {self.ui.analyze_correction_combo.currentText()}\n\n"
                    f"Note:\n"
                    f"  - Model: Value ~ Group + (1|Subject) - accounts for within-subject correlation\n"
                    f"  - N_Subj: Number of subjects (Study1/Study2)\n"
                    f"  - N_Windows: Total windows analyzed (Study1/Study2)\n"
                    f"  - Diff: Group effect (Study2 - Study1)\n"
                    f"  - 95% CI: Confidence interval for group difference\n"
                    f"  - d: Cohen's d effect size\n"
                    f"  - Significance: *** p<0.001, ** p<0.01, * p<0.05\n"
                    f"  - [!]: Model convergence warning\n"
                )

    def _analyze_event_based_sliding(
            self, df1: pd.DataFrame, df2: pd.DataFrame, selected_feature: str
    ) -> None:
        """Analyze event-based sliding window features.

        For event-based data, we compare features across different events or
        pre/post event windows. Uses mixed-effects models to account for:
        - Within-subject correlation (multiple trials per subject)
        - Variable number of trials per subject
        - Pre vs Post event comparisons
        """
        # Get feature columns - match exactly (feature_microstate format)
        feature_cols = []
        for col in df1.columns:
            if col == selected_feature:  # Exact match
                feature_cols.append(col)
            elif col.startswith(selected_feature + "_"):
                    # Make sure it's in the format FEATURE_MICROSTATE (e.g., COV_A)
                    parts = col.split("_")
                    if len(parts) == 2 and parts[0] == selected_feature:
                        feature_cols.append(col)

            if not feature_cols:
                self.ui.stats_textedit.appendPlainText(
                    f"No feature columns found for '{selected_feature}' in event-based data.\n"
                    f"Available columns: {list(df1.columns)[:20]}"
                )
                return

            # Check if we have Window_Type column (Pre/Post distinction)
            has_window_type = "Window_Type" in df1.columns

            # Check if this is a Pre vs Post comparison within the same dataset
            is_pre_post_comparison = False
            if has_window_type:
                window_types_1 = df1['Window_Type'].unique() if not df1.empty else []
                window_types_2 = df2['Window_Type'].unique() if not df2.empty else []
                # If both datasets have Pre and Post, or if one has Pre and other has Post
                has_pre = any('Pre' in str(wt) or 'pre' in str(wt) for wt in window_types_1) or \
                          any('Pre' in str(wt) or 'pre' in str(wt) for wt in window_types_2)
                has_post = any('Post' in str(wt) or 'post' in str(wt) for wt in window_types_1) or \
                           any('Post' in str(wt) or 'post' in str(wt) for wt in window_types_2)
                is_pre_post_comparison = has_pre and has_post

            # Print header
            comparison_info = self._get_comparison_info()

            analysis_description = "Event-Based Pre/Post Comparison" if is_pre_post_comparison else "Event-Based Repeated Measures"

            header_text = (
                f"Repeated Measures Analysis - {analysis_description}\n"
                f"{'=' * 80}\n"
                f"Feature: {self.feature_list_dictionary.get(selected_feature, selected_feature)}\n"
                f"Comparison: {comparison_info['comparison_type']}\n"
                f"Study 1: {comparison_info['study1_name']} ({comparison_info['study1_files']} files)\n"
                f"Study 2: {comparison_info['study2_name']} ({comparison_info['study2_files']} files)\n"
                f"Analysis: Mixed-effects model accounting for within-subject correlation\n"
                f"Microstates analyzed: {len(feature_cols)}\n"
            )

            if has_window_type:
                window_types_combined = set(list(df1['Window_Type'].unique()) + list(
                    df2['Window_Type'].unique())) if not df1.empty and not df2.empty else []
                header_text += f"Window Types: {sorted(window_types_combined)}\n"

            if is_pre_post_comparison:
                header_text += (
                    f"\nNote: Pre vs Post event comparison detected.\n"
                    f"      Mixed-effects models account for:\n"
                    f"      - Multiple trials per subject (within-subject correlation)\n"
                    f"      - Variable number of trials across subjects\n"
                    f"      - Temporal dependencies within event-related responses\n"
                )

            header_text += f"{'=' * 80}\n\n"
            self.ui.stats_textedit.appendPlainText(header_text)

            # Perform mixed effects analysis for each microstate
            results = []
            for feat_col in feature_cols:
                microstate_label = "_".join(feat_col.split("_")[1:]) if "_" in feat_col else feat_col

                # Prepare data for mixed effects model
                data_list = []

                # Add Study 1 data
                if 'Filename' in df1.columns:
                    for idx, row in df1.iterrows():
                        if pd.notna(row[feat_col]):
                            data_list.append({
                                'Subject': row['Filename'],
                                'Group': 0,
                                'Value': row[feat_col]
                            })

                # Add Study 2 data
                if 'Filename' in df2.columns:
                    for idx, row in df2.iterrows():
                        if pd.notna(row[feat_col]):
                            data_list.append({
                                'Subject': row['Filename'],
                                'Group': 1,
                                'Value': row[feat_col]
                            })

                if len(data_list) == 0:
                    continue

                # Create dataframe for mixed effects model
                analysis_df = pd.DataFrame(data_list)

                # Get summary statistics
                n_subjects_1 = len(analysis_df[analysis_df['Group'] == 0]['Subject'].unique())
                n_subjects_2 = len(analysis_df[analysis_df['Group'] == 1]['Subject'].unique())
                n_events_1 = len(analysis_df[analysis_df['Group'] == 0])
                n_events_2 = len(analysis_df[analysis_df['Group'] == 1])
                mean1 = analysis_df[analysis_df['Group'] == 0]['Value'].mean()
                mean2 = analysis_df[analysis_df['Group'] == 1]['Value'].mean()

                if n_subjects_1 < 2 or n_subjects_2 < 2:
                    continue

                try:
                    # Fit mixed effects model
                    model = smf.mixedlm("Value ~ Group", data=analysis_df,
                                        groups=analysis_df["Subject"],
                                        re_formula="1")
                    result = model.fit(method='lbfgs', maxiter=100)

                    # Extract results
                    coef = result.params['Group']
                    p_val = result.pvalues['Group']
                    ci_low, ci_high = result.conf_int().loc['Group']

                    # Calculate effect size
                    pooled_sd = np.sqrt(result.scale)
                    cohens_d = coef / pooled_sd if pooled_sd > 0 else 0

                    results.append({
                        "microstate": microstate_label,
                        "n_subjects_1": n_subjects_1,
                        "n_subjects_2": n_subjects_2,
                        "n_events_1": n_events_1,
                        "n_events_2": n_events_2,
                        "mean1": mean1,
                        "mean2": mean2,
                        "coef": coef,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_val": p_val,
                        "cohens_d": cohens_d,
                        "converged": result.converged
                    })

                except Exception as e:
                    self.ui.stats_textedit.appendPlainText(
                        f"Error fitting model for {feat_col}: {str(e)}"
                    )
                    continue

            if not results:
                self.ui.stats_textedit.appendPlainText("No valid data for analysis.")
                return

            # Apply multiple testing correction
            p_values = [r["p_val"] for r in results]
            _, adj_p_values, _, _ = multipletests(
                p_values, method=_correction_method(self.ui.analyze_correction_combo.currentText())
            )

            for i, result in enumerate(results):
                result["adj_p_val"] = adj_p_values[i]

            # Display results
            self.ui.stats_textedit.appendPlainText(
                "\nMixed Effects Model Results (Value ~ Group + (1|Subject)):\n"
                f"{'-' * 130}\n"
                f"{'Microstate':<12} {'N_Subj':<12} {'N_Events':<16} {'Mean(S1)':<12} {'Mean(S2)':<12} "
                f"{'Diff':<10} {'95% CI':<26} {'p-value':<12} {'Adj.p':<12} {'d':<8}\n"
                f"{'-' * 130}"
            )

            for result in results:
                sig_marker = " ***" if result['adj_p_val'] < 0.001 else " **" if result['adj_p_val'] < 0.01 else " *" if \
                result['adj_p_val'] < 0.05 else ""
                conv_marker = "" if result.get('converged', True) else " [!]"

                n_subj_str = f"{result['n_subjects_1']}/{result['n_subjects_2']}"
                n_events_str = f"{result['n_events_1']}/{result['n_events_2']}"
                ci_str = f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"

                p_str = f"{result['p_val']:.2e}" if result['p_val'] < 0.001 else f"{result['p_val']:.6f}"
                adj_p_str = f"{result['adj_p_val']:.2e}" if result[
                                                                'adj_p_val'] < 0.001 else f"{result['adj_p_val']:.6f}"

                self.ui.stats_textedit.appendPlainText(
                    f"{result['microstate']:<12} {n_subj_str:<12} {n_events_str:<16} "
                    f"{result['mean1']:<12.4f} {result['mean2']:<12.4f} "
                    f"{result['coef']:<10.4f} {ci_str:<26} "
                    f"{p_str:<12} {adj_p_str:<12} {result['cohens_d']:<8.3f}"
                    f"{sig_marker}{conv_marker}"
                )

            # Summary
            n_sig = sum(1 for r in results if r["adj_p_val"] < 0.05)
            n_converged = sum(r.get('converged', True) for r in results)

            self.ui.stats_textedit.appendPlainText(
                f"\n{'-' * 130}\n"
                f"Significant differences (adjusted p < 0.05): {n_sig}/{len(results)}\n"
                f"Models converged: {n_converged}/{len(results)}\n"
                f"Multiple testing correction: {self.ui.analyze_correction_combo.currentText()}\n\n"
                f"Note:\n"
                f"  - N_Subj: Number of subjects (Study1/Study2)\n"
                f"  - N_Events: Total event windows analyzed (Study1/Study2)\n"
                f"  - Diff: Group effect (Study2 - Study1)\n"
                f"  - 95% CI: Confidence interval for group difference\n"
                f"  - d: Cohen's d effect size\n"
                f"  - Significance: *** p<0.001, ** p<0.01, * p<0.05\n"
                f"  - [!]: Model convergence warning\n"
            )

            # ==================== RTF ANALYSIS ====================

    def _perform_rtf_analysis(self) -> None:
        """Perform statistical analysis for RTF (Rate of Time in Field) features.

        RTF represents the rate of transitions into specific microstates and is typically
        computed per trial/window. Uses t-tests with multiple comparison correction.
        """
        try:
                # Load RTF features
                result = self._get_common_features()
                if result is None or result[0] is None:
                    self.ui.stats_textedit.appendPlainText(
                        "Could not load RTF features for analysis.\n"
                        "Please ensure RTF features have been extracted."
                    )
                    return

                common_features_df, _ = result
                selected_feature = self._get_selected_feature_code()

                # Separate studies
                study1_df, study2_df = self._separate_studies_data(
                    common_features_df, selected_feature
                )

                # Get feature columns
                feature_cols = [
                    col for col in study1_df.columns if col not in ["Study", "Filename"]
                ]

                if not feature_cols:
                    self.ui.stats_textedit.appendPlainText(
                        f"No RTF feature columns found for '{selected_feature}'."
                    )
                    return

                # Display analysis header
                comparison_info = self._get_comparison_info()
                header_text = (
                    f"RTF (Rate of Time in Field) Statistical Analysis\n"
                    f"{'=' * 80}\n"
                    f"Feature: {self.feature_list_dictionary.get(selected_feature, selected_feature)}\n"
                    f"Comparison: {comparison_info['comparison_type']}\n"
                    f"Study 1: {comparison_info['study1_name']} ({comparison_info['study1_files']} files)\n"
                    f"Study 2: {comparison_info['study2_name']} ({comparison_info['study2_files']} files)\n"
                    f"Test Type: {'Paired' if self.ui.analyze_design_paired_radio.isChecked() else 'Independent'} t-test\n"
                    f"Multiple Testing Correction: {self.ui.analyze_correction_combo.currentText()}\n"
                    f"Number of comparisons: {len(feature_cols)}\n"
                    f"{'=' * 80}\n\n"
                )
                self.ui.stats_textedit.appendPlainText(header_text)

                # Perform t-tests for each microstate
                results = []
                p_values = []

                for feat_col in feature_cols:
                    microstate_label = "_".join(feat_col.split("_")[1:]) if "_" in feat_col else feat_col

                    study1_values = study1_df[feat_col].dropna().values
                    study2_values = study2_df[feat_col].dropna().values

                    if len(study1_values) < 2 or len(study2_values) < 2:
                        continue

                    # Perform t-test
                    if self.ui.analyze_design_paired_radio.isChecked():
                        if len(study1_values) != len(study2_values):
                            self.ui.stats_textedit.appendPlainText(
                                f"Warning: Skipping {feat_col} - unequal sample sizes for paired test\n"
                            )
                            continue
                        t_stat, p_val = ttest_rel(study1_values, study2_values)
                    else:
                        t_stat, p_val, _ = independent_ttest(
                            study1_values, study2_values
                        )

                    # Calculate effect size (Cohen's d)
                    mean1 = np.mean(study1_values)
                    mean2 = np.mean(study2_values)

                    if self.ui.analyze_design_paired_radio.isChecked():
                        # Paired Cohen's d
                        diff = study2_values - study1_values
                        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                    else:
                        # Independent Cohen's d
                        pooled_std = np.sqrt(
                            ((len(study1_values) - 1) * np.var(study1_values, ddof=1) +
                             (len(study2_values) - 1) * np.var(study2_values, ddof=1)) /
                            (len(study1_values) + len(study2_values) - 2)
                        )
                        cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

                    # Calculate 95% CI
                    se_diff = np.sqrt(np.var(study1_values, ddof=1) / len(study1_values) +
                                      np.var(study2_values, ddof=1) / len(study2_values))
                    df = len(study1_values) + len(study2_values) - 2
                    ci_margin = t.ppf(0.975, df) * se_diff
                    mean_diff = mean2 - mean1
                    ci_low = mean_diff - ci_margin
                    ci_high = mean_diff + ci_margin

                    results.append({
                        'microstate': microstate_label,
                        'n1': len(study1_values),
                        'n2': len(study2_values),
                        'mean1': mean1,
                        'mean2': mean2,
                        'diff': mean_diff,
                        'ci_low': ci_low,
                        'ci_high': ci_high,
                        't_stat': t_stat,
                        'p_val': p_val,
                        'cohens_d': cohens_d
                    })
                    p_values.append(p_val)

                if not results:
                    self.ui.stats_textedit.appendPlainText("No valid data for RTF analysis.")
                    return

                # Apply multiple testing correction
                _, adj_p_values, _, _ = multipletests(
                    p_values, method=_correction_method(self.ui.analyze_correction_combo.currentText())
                )

                for i, result in enumerate(results):
                    result['adj_p_val'] = adj_p_values[i]

                # Display results table
                self.ui.stats_textedit.appendPlainText(
                    "Statistical Results:\n"
                    f"{'-' * 120}\n"
                    f"{'Microstate':<12} {'N(S1/S2)':<12} {'Mean(S1)':<12} {'Mean(S2)':<12} "
                    f"{'Diff':<10} {'95% CI':<26} {'t-stat':<10} {'p-value':<12} {'Adj.p':<12} {'d':<8}\n"
                    f"{'-' * 120}"
                )

                for result in results:
                    sig_marker = " ***" if result['adj_p_val'] < 0.001 else " **" if result[
                                                                                         'adj_p_val'] < 0.01 else " *" if \
                    result['adj_p_val'] < 0.05 else ""

                    n_str = f"{result['n1']}/{result['n2']}"
                    ci_str = f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
                    p_str = f"{result['p_val']:.2e}" if result['p_val'] < 0.001 else f"{result['p_val']:.6f}"
                    adj_p_str = f"{result['adj_p_val']:.2e}" if result[
                                                                    'adj_p_val'] < 0.001 else f"{result['adj_p_val']:.6f}"

                    self.ui.stats_textedit.appendPlainText(
                        f"{result['microstate']:<12} {n_str:<12} {result['mean1']:<12.4f} {result['mean2']:<12.4f} "
                        f"{result['diff']:<10.4f} {ci_str:<26} "
                        f"{result['t_stat']:<10.4f} {p_str:<12} {adj_p_str:<12} {result['cohens_d']:<8.3f}"
                        f"{sig_marker}"
                    )

                # Summary
                n_sig = sum(1 for r in results if r['adj_p_val'] < 0.05)

                self.ui.stats_textedit.appendPlainText(
                    f"\n{'-' * 120}\n"
                    f"Summary:\n"
                    f"  Significant differences (adjusted p < 0.05): {n_sig}/{len(results)}\n"
                    f"  Multiple testing correction: {self.ui.analyze_correction_combo.currentText()}\n\n"
                    f"Interpretation:\n"
                    f"  RTF measures the rate of transitions into specific microstates.\n"
                    f"  Higher values indicate more frequent transitions into that microstate.\n"
                    f"  Diff: Mean difference (Study2 - Study1)\n"
                    f"  d: Cohen's d effect size\n"
                    f"  Significance: *** p<0.001, ** p<0.01, * p<0.05\n"
                )

        except Exception as e:
            self._show_error(
                "RTF Analysis Error",
                f"Failed to perform RTF analysis:\n{str(e)}\n\n{traceback.format_exc()}"
            )

            # ==================== ROF TFCE CLUSTER PERMUTATION ANALYSIS ====================

    def _has_rof_data(self) -> bool:
        """Check if ROF data is available for analysis.

        Returns:
            True if ROF data is loaded and available, False otherwise.
        """
        if not self.study1_loaded:
            return False

        # Check if Study 1 has ROF data
            feature_type, feature_mode = self._get_compatible_features()
            if feature_type is None or feature_mode is None:
                return False

            # Load features to check for rof_data
            features_df = self._load_features_safely(self.comet_tbx_study1, feature_type, feature_mode)
            if features_df is None:
                return False

            # Check if there's ROF data in the feature extraction results
            # ROF data is stored separately from regular features
            return hasattr(self.comet_tbx_study1, 'extracted_features_path')

    def _perform_rof_tfce_analysis(self) -> None:
            """Perform TFCE-based cluster permutation testing for ROF time courses.

            This method implements the statistical analysis described in:
            - One-sample t-tests against zero at each time point
            - Sign-flipping permutation approach
            - TFCE for multiple comparison correction
            - Cohen's d for effect sizes
            """
            try:
                # Load ROF data
                rof_data_study1, rof_data_study2 = self._load_rof_data()

                if rof_data_study1 is None:
                    self.ui.stats_textedit.appendPlainText(
                        "Could not load ROF data for analysis.\n"
                        "Please ensure ROF features have been extracted for event-related data."
                    )
                    return

                # Display analysis header
                self._display_rof_analysis_header(rof_data_study1, rof_data_study2)

                # Determine analysis type (one-sample vs two-sample)
                if rof_data_study2 is None or self.ui.compare_surrogate_radio.isChecked() or self.ui.compare_random_radio.isChecked():
                    # One-sample test against zero
                    self._perform_rof_one_sample_tfce(rof_data_study1)
                else:
                    # Two-sample comparison (difference between conditions)
                    self._perform_rof_two_sample_tfce(rof_data_study1, rof_data_study2)

            except Exception as e:
                self._show_error(
                    "ROF TFCE Analysis Error",
                    f"Failed to perform TFCE cluster permutation testing:\n{str(e)}\n\n{traceback.format_exc()}"
                )

    def _load_rof_data(self) -> Tuple[Optional[Dict], Optional[Dict]]:
            """Load ROF data from feature extraction results.

            Returns:
                Tuple of (rof_data_study1, rof_data_study2) or (rof_data, None) for single study.
            """
            feature_type, feature_mode = self._get_compatible_features()

            if feature_type is None or feature_mode is None:
                return None, None

            # For ROF, we need to load the full time-resolved data, not just summary statistics
            # This data is stored separately during feature extraction

            # Try to load from the extracted features path
            rof_data_study1 = self._load_rof_from_features_path(
                self.comet_tbx_study1, feature_type, feature_mode
            )

            rof_data_study2 = None
            if self.ui.compare_two_studies_radio.isChecked() and self.study2_loaded:
                rof_data_study2 = self._load_rof_from_features_path(
                    self.comet_tbx_study2, feature_type, feature_mode
                )
            elif self.ui.compare_within_study_radio.isChecked():
                # For within-study comparison, we need to split the data by file lists
                rof_data_study2 = self._load_rof_from_features_path(
                    self.comet_tbx_study1, feature_type, feature_mode
                )

            return rof_data_study1, rof_data_study2

    def _load_rof_from_features_path(
            self, tbx: COMET, feature_type: str, feature_mode: str
    ) -> Optional[Dict]:
            """Load ROF data from the features path.

            ROF data includes:
            - time: time array in milliseconds
            - microstates: list of microstate labels
            - occurrences_clr_bc: baseline-corrected CLR-transformed occurrence frequencies
            - occurrences_clr: CLR-transformed occurrence frequencies (pre-baseline correction)
            - baseline_median: median values during baseline period
            - filenames: list of subject filenames

            Args:
                tbx: COMET toolbox instance
                feature_type: Feature type ('real', 'surrogate', 'random')
                feature_mode: Feature mode ('averaged', 'sliding', etc.)

            Returns:
                Dictionary containing ROF data or None if not available
            """
            if not hasattr(tbx, "extracted_features_path") or not os.path.exists(
                    tbx.extracted_features_path
            ):
                return None

            # Try multiple naming patterns
            rof_filenames_to_try = [
                # Standard naming pattern
                f"{feature_type}_{feature_mode}_rof_data",
                # Alternative naming patterns
                "ROF_timeseries",
                f"ROF_timeseries_{feature_type}",
                f"{feature_type}_ROF_timeseries",
                "rof_timeseries",
            ]

            for base_filename in rof_filenames_to_try:
                for export_format in self.EXPORT_FORMATS:
                    rof_path = os.path.join(tbx.extracted_features_path, base_filename + export_format)

                    if os.path.exists(rof_path):
                        try:
                            rof_data = None

                            if export_format == ".csv":
                                # Load CSV file - special handling for ROF timeseries
                                df = pd.read_csv(rof_path)
                                rof_data = self._parse_rof_csv(df)

                            elif export_format == ".pkl":
                                rof_data = safe_pickle_load(rof_path)

                            elif export_format == ".hdf":
                                import h5py
                                rof_data = {}
                                with h5py.File(rof_path, 'r') as f:
                                    # Load all data from HDF5 file
                                    for key in f.keys():
                                        rof_data[key] = f[key][()]

                            elif export_format == ".json":
                                import json
                                with open(rof_path, 'r') as f:
                                    rof_data = json.load(f)
                                # Convert lists back to numpy arrays
                                if 'time' in rof_data:
                                    rof_data['time'] = np.array(rof_data['time'])
                                if 'occurrences_clr_bc' in rof_data:
                                    for ms in rof_data['occurrences_clr_bc']:
                                        rof_data['occurrences_clr_bc'][ms] = np.array(
                                            rof_data['occurrences_clr_bc'][ms]
                                        )

                            if rof_data is not None:
                                return rof_data

                        except Exception:
                            continue

            return None

    def _parse_rof_csv(self, df: pd.DataFrame) -> Dict:
            """Parse ROF data from CSV format.

            Expected CSV format:
            - Columns: Time, Filename, MS_A, MS_B, MS_C, MS_D (or similar microstate columns)
            - Each row represents one time point for one subject/trial

            Args:
                df: DataFrame loaded from CSV

            Returns:
                Dictionary with ROF data structure
            """
            rof_data = {}

            # Extract time array (unique time points)
            time_col = None
            for col in df.columns:
                if 'time' in col.lower():
                    time_col = col
                    break

            if time_col:
                rof_data['time'] = np.array(sorted(df[time_col].unique()))
            else:
                # Assume time is in first column if not labeled
                rof_data['time'] = np.array(sorted(df.iloc[:, 0].unique()))

            # Extract filenames/subjects
            filename_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['file', 'subj', 'participant']):
                    filename_col = col
                    break

            if filename_col:
                rof_data['filenames'] = list(df[filename_col].unique())

            # Extract microstate columns - be more flexible
            # Skip known metadata columns
            metadata_keywords = ['time', 'file', 'subj', 'trial', 'window', 'event',
                                 'condition', 'group', 'session', 'participant']

            microstate_columns = []
            for col in df.columns:
                # Skip if it's a metadata column
                if any(keyword in col.lower() for keyword in metadata_keywords):
                    continue

                # Accept as microstate column if:
                # 1. Starts with MS_ or MS- or MS (like MS_A, MSA, MS-A)
                # 2. Is ROF_ or ROF- prefix (like ROF_A, ROFA, ROF-A)
                # 3. Is a single uppercase letter (A, B, C, D, etc.)
                # 4. Looks like a microstate label
                col_upper = col.upper()
                if (col.startswith('MS') or
                        col.startswith('ROF') or
                        'MICROSTATE' in col_upper or
                        (len(col) == 1 and col.isalpha() and col.isupper()) or
                        col_upper in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
                    microstate_columns.append(col)

            if not microstate_columns:
                # Print helpful error with all columns
                raise ValueError(
                    f"No microstate columns found in ROF CSV file.\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Expected columns like: MS_A, MS_B, MS_C, MS_D or A, B, C, D or ROF_A, ROF_B, etc.\n"
                    f"Metadata columns (automatically skipped): {metadata_keywords}"
                )

            # Extract microstate labels
            rof_data['microstates'] = []
            for col in microstate_columns:
                # Clean up the label
                label = col
                # Remove common prefixes
                for prefix in ['MS_', 'MS-', 'MS', 'ROF_', 'ROF-', 'ROF', 'MICROSTATE_', 'MICROSTATE-']:
                    if label.startswith(prefix):
                        label = label[len(prefix):]
                        break
                rof_data['microstates'].append(label.upper())

            # Organize ROF data by microstate
            # Structure: occurrences_clr_bc[microstate] = array of shape (n_subjects, n_timepoints)
            rof_data['occurrences_clr_bc'] = {}

            if 'filenames' in rof_data:
                n_subjects = len(rof_data['filenames'])
            else:
                # Estimate number of subjects
                n_subjects = len(df) // len(rof_data['time'])

            n_timepoints = len(rof_data['time'])

            for i, ms_col in enumerate(microstate_columns):
                ms_label = rof_data['microstates'][i]

                # Initialize array
                rof_array = np.zeros((n_subjects, n_timepoints))

                # Fill array with data
                if 'filenames' in rof_data and filename_col:
                    for subj_idx, filename in enumerate(rof_data['filenames']):
                        # Get data for this subject
                        subj_data = df[df[filename_col] == filename].copy()

                        # Sort by time
                        if time_col:
                            subj_data = subj_data.sort_values(time_col)

                        # Extract values
                        values = subj_data[ms_col].values
                        if len(values) >= n_timepoints:
                            rof_array[subj_idx, :] = values[:n_timepoints]
                        else:
                            rof_array[subj_idx, :len(values)] = values
                else:
                    # Fallback: assume data is organized sequentially
                    try:
                        rof_array = df[ms_col].values.reshape(n_subjects, n_timepoints)
                    except ValueError:
                        # Try transposed
                        rof_array = df[ms_col].values.reshape(n_timepoints, n_subjects).T

                rof_data['occurrences_clr_bc'][ms_label] = rof_array

            return rof_data

    def _display_rof_analysis_header(
            self, rof_data_study1: Dict, rof_data_study2: Optional[Dict]
    ) -> None:
            """Display header information for ROF TFCE analysis."""
            comparison_info = self._get_comparison_info()

            header_text = (
                f"ROF Cluster-Based Permutation Testing (TFCE)\n"
                f"{'=' * 80}\n"
                f"Feature: Relative Occurrence Frequency (ROF)\n"
                f"Analysis: Post-event dynamics testing against zero baseline\n"
                f"Comparison: {comparison_info['comparison_type']}\n"
                f"Study 1: {comparison_info['study1_name']} ({comparison_info['study1_files']} files)\n"
            )

            if rof_data_study2 is not None:
                header_text += f"Study 2: {comparison_info['study2_name']} ({comparison_info['study2_files']} files)\n"

            if 'microstates' in rof_data_study1:
                header_text += f"Microstates: {', '.join(rof_data_study1['microstates'])}\n"

            if 'time' in rof_data_study1:
                time = rof_data_study1['time']
                header_text += f"Time range: {time[0]:.1f} to {time[-1]:.1f} ms\n"
                header_text += f"Time points: {len(time)}\n"

            header_text += (
                f"\nStatistical Method:\n"
                f"  - Threshold-Free Cluster Enhancement (TFCE) for multiple comparison correction\n"
                f"  - Sign-flipping permutation test (5000 permutations)\n"
                f"  - Preserves temporal dependencies in event-related data\n"
                f"  - Tests null hypothesis: Post-event ROF does not differ from zero\n"
                f"  - Significance level: p < 0.05 (cluster-corrected)\n"
                f"\nROF Baseline Correction:\n"
                f"  - ROF values are baseline-corrected (subtracted pre-event median)\n"
                f"  - CLR-transformed to handle compositional nature of microstate data\n"
                f"  - Analysis period: 20-1000 ms post-event\n"
                f"{'=' * 80}\n\n"
            )

            self.ui.stats_textedit.appendPlainText(header_text)

    def _perform_rof_one_sample_tfce(self, rof_data: Dict) -> None:
            """Perform one-sample TFCE test for ROF against zero.

            Tests the null hypothesis that ROF does not differ from zero following the event.
            Uses sign-flipping permutation to preserve temporal dependencies - this is the
            appropriate method for testing post-event dynamics against baseline.

            The sign-flipping approach randomly flips the sign of each subject's time series
            in each permutation, which:
            - Preserves the temporal autocorrelation structure within each subject
            - Tests against the null hypothesis of zero mean response
            - Controls family-wise error rate across all time points

            Args:
                rof_data: Dictionary containing ROF time course data
            """
            self.ui.stats_textedit.appendPlainText(
                "Analysis Type: One-sample t-test against zero (sign-flipping permutation)\n"
                "Null hypothesis: Post-event ROF does not differ from zero baseline\n"
                "Method: Sign-flipping preserves temporal dependencies within subjects\n\n"
            )

            microstates = rof_data.get('microstates', [])
            time = rof_data.get('time', [])
            occurrences_bc = rof_data.get('occurrences_clr_bc', {})

            if not microstates or len(time) == 0 or not occurrences_bc:
                self.ui.stats_textedit.appendPlainText(
                    "Error: ROF data is incomplete or malformed."
                )
                return

            # Filter to post-event period (20 ms to 1000 ms)
            post_event_mask = (time >= 20) & (time <= 1000)
            time_post = time[post_event_mask]

            if len(time_post) == 0:
                self.ui.stats_textedit.appendPlainText(
                    "Error: No time points found in post-event period (20-1000 ms)."
                )
                return

            # Perform TFCE test for each microstate
            results = []
            for microstate in microstates:
                if microstate not in occurrences_bc:
                    continue

                # Get ROF data for this microstate: shape (n_subjects, n_timepoints)
                rof_timecourse = occurrences_bc[microstate]

                # Filter by file list if needed
                rof_timecourse = self._filter_rof_by_filelist(
                    rof_timecourse, rof_data, self.ui.study1_file_list
                )

                if rof_timecourse is None or len(rof_timecourse) < 3:
                    continue

                # Filter to post-event period
                rof_timecourse_post = rof_timecourse[:, post_event_mask]

                # Perform TFCE cluster permutation test
                t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
                    rof_timecourse_post,
                    n_permutations=5000,
                    threshold=dict(start=0, step=0.2),  # TFCE parameters
                    tail=0,  # Two-tailed test
                    n_jobs=-1,  # Use all available cores
                    out_type='mask',
                    verbose=False
                )

                # Find significant clusters
                sig_clusters = [i for i, pv in enumerate(cluster_pv) if pv < 0.05]

                if len(sig_clusters) > 0:
                    # Calculate effect sizes for significant clusters
                    cluster_results = []
                    for cluster_idx in sig_clusters:
                        cluster_mask = clusters[cluster_idx]
                        cluster_times = time_post[cluster_mask]
                        cluster_rof = rof_timecourse_post[:, cluster_mask]

                        # Calculate Cohen's d for the cluster
                        mean_rof = np.mean(cluster_rof)
                        std_rof = np.std(cluster_rof, ddof=1)
                        cohens_d = mean_rof / std_rof if std_rof > 0 else 0

                        cluster_results.append({
                            'cluster_idx': cluster_idx,
                            'p_value': cluster_pv[cluster_idx],
                            'time_start': cluster_times[0],
                            'time_end': cluster_times[-1],
                            'n_timepoints': len(cluster_times),
                            'mean_rof': mean_rof,
                            'cohens_d': cohens_d,
                            'direction': 'positive' if mean_rof > 0 else 'negative'
                        })

                    results.append({
                        'microstate': microstate,
                        'n_subjects': len(rof_timecourse),
                        'significant': True,
                        'n_clusters': len(sig_clusters),
                        'clusters': cluster_results,
                        'avg_cohens_d': np.mean([c['cohens_d'] for c in cluster_results])
                    })
                else:
                    results.append({
                        'microstate': microstate,
                        'n_subjects': len(rof_timecourse),
                        'significant': False,
                        'n_clusters': 0
                    })

            # Display results
            self._display_rof_tfce_results(results)

    def _perform_rof_two_sample_tfce(
            self, rof_data_study1: Dict, rof_data_study2: Dict
    ) -> None:
            """Perform two-sample TFCE test for ROF differences between conditions.

            Tests the null hypothesis that ROF does not differ between two conditions.
            Uses sign-flipping permutation on the difference between conditions.

            Args:
                rof_data_study1: Dictionary containing ROF data for condition 1
                rof_data_study2: Dictionary containing ROF data for condition 2
            """
            self.ui.stats_textedit.appendPlainText(
                "Analysis Type: Two-sample comparison (difference between conditions)\n"
                "Null hypothesis: No difference in ROF between conditions\n\n"
            )

            microstates = rof_data_study1.get('microstates', [])
            time = rof_data_study1.get('time', [])
            occurrences_bc_1 = rof_data_study1.get('occurrences_clr_bc', {})
            occurrences_bc_2 = rof_data_study2.get('occurrences_clr_bc', {})

            if not microstates or len(time) == 0 or not occurrences_bc_1 or not occurrences_bc_2:
                self.ui.stats_textedit.appendPlainText(
                    "Error: ROF data is incomplete or malformed for both conditions."
                )
                return

            # Filter to post-event period (20 ms to 1000 ms)
            post_event_mask = (time >= 20) & (time <= 1000)
            time_post = time[post_event_mask]

            if len(time_post) == 0:
                self.ui.stats_textedit.appendPlainText(
                    "Error: No time points found in post-event period (20-1000 ms)."
                )
                return

            # Perform TFCE test for each microstate
            results = []
            for microstate in microstates:
                if microstate not in occurrences_bc_1 or microstate not in occurrences_bc_2:
                    continue

                # Get ROF data for both conditions
                rof_1 = occurrences_bc_1[microstate]
                rof_2 = occurrences_bc_2[microstate]

                # For paired comparison, filter and match subjects
                if self.ui.analyze_design_paired_radio.isChecked():
                    # Filter by file lists and get corresponding filenames
                    rof_1_result = self._filter_rof_by_filelist(
                        rof_1, rof_data_study1, self.ui.study1_file_list, return_filenames=True
                    )
                    rof_2_result = self._filter_rof_by_filelist(
                        rof_2, rof_data_study2, self.ui.study2_file_list, return_filenames=True
                    )

                    if rof_1_result[0] is None or rof_2_result[0] is None:
                        continue

                    rof_1, filenames_1 = rof_1_result
                    rof_2, filenames_2 = rof_2_result

                    # Match subjects by extracting subject IDs from filenames
                    subj_to_idx1 = {}
                    for idx, fname in enumerate(filenames_1):
                        subj_id = self._extract_subject_id(fname)
                        subj_to_idx1[subj_id] = idx

                    subj_to_idx2 = {}
                    for idx, fname in enumerate(filenames_2):
                        subj_id = self._extract_subject_id(fname)
                        subj_to_idx2[subj_id] = idx

                    # Find matched subjects
                    matched_subjects = sorted(set(subj_to_idx1.keys()) & set(subj_to_idx2.keys()))

                    if len(matched_subjects) < 3:
                        self.ui.stats_textedit.appendPlainText(
                            f"Warning: Microstate {microstate} - insufficient matched subjects ({len(matched_subjects)}). Skipping.\n"
                        )
                        continue

                    # Extract matched data in correct order
                    matched_rof_1 = np.array([rof_1[subj_to_idx1[subj], :] for subj in matched_subjects])
                    matched_rof_2 = np.array([rof_2[subj_to_idx2[subj], :] for subj in matched_subjects])

                    # Calculate difference: Study2 - Study1 (for matched pairs)
                    rof_diff = matched_rof_2 - matched_rof_1

                    # Filter to post-event period
                    rof_diff_post = rof_diff[:, post_event_mask]

                    # Perform one-sample TFCE test on difference (testing if difference != 0)
                    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
                        rof_diff_post,
                        n_permutations=5000,
                        threshold=dict(start=0, step=0.2),  # TFCE parameters
                        tail=0,  # Two-tailed test
                        n_jobs=-1,
                        out_type='mask',
                        verbose=False
                    )
                else:
                    # Independent samples comparison - filter without matching
                    rof_1 = self._filter_rof_by_filelist(
                        rof_1, rof_data_study1, self.ui.study1_file_list
                    )
                    rof_2 = self._filter_rof_by_filelist(
                        rof_2, rof_data_study2, self.ui.study2_file_list
                    )

                    if rof_1 is None or rof_2 is None:
                        continue

                    # For independent samples TFCE, we would need to use a different MNE function
                    # This is not yet implemented
                    self.ui.stats_textedit.appendPlainText(
                        "Note: Independent samples TFCE comparison not yet implemented.\n"
                        "Please use paired test checkbox for matched subjects.\n"
                    )
                    continue

                # Find significant clusters
                sig_clusters = [i for i, pv in enumerate(cluster_pv) if pv < 0.05]

                if len(sig_clusters) > 0:
                    # Calculate effect sizes for significant clusters
                    cluster_results = []
                    for cluster_idx in sig_clusters:
                        cluster_mask = clusters[cluster_idx]
                        cluster_times = time_post[cluster_mask]
                        cluster_diff = rof_diff_post[:, cluster_mask]

                        # Calculate Cohen's d for the cluster (paired)
                        mean_diff = np.mean(cluster_diff)
                        std_diff = np.std(cluster_diff, ddof=1)
                        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

                        cluster_results.append({
                            'cluster_idx': cluster_idx,
                            'p_value': cluster_pv[cluster_idx],
                            'time_start': cluster_times[0],
                            'time_end': cluster_times[-1],
                            'n_timepoints': len(cluster_times),
                            'mean_diff': mean_diff,
                            'cohens_d': cohens_d,
                            'direction': 'Study2>Study1' if mean_diff > 0 else 'Study1>Study2'
                        })

                    results.append({
                        'microstate': microstate,
                        'n_pairs': len(rof_diff),
                        'significant': True,
                        'n_clusters': len(sig_clusters),
                        'clusters': cluster_results,
                        'avg_cohens_d': np.mean([c['cohens_d'] for c in cluster_results])
                    })
                else:
                    results.append({
                        'microstate': microstate,
                        'n_pairs': len(rof_diff),
                        'significant': False,
                        'n_clusters': 0
                    })

            # Display results
            self._display_rof_tfce_results(results, comparison_type='paired')

    def _filter_rof_by_filelist(
            self, rof_timecourse: np.ndarray, rof_data: Dict, file_list_widget, return_filenames: bool = False
    ) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], List[str]]]:
            """Filter ROF timecourse data by files in the file list widget.

            Args:
                rof_timecourse: ROF data array (n_subjects, n_timepoints) or dict mapping filenames
                rof_data: Full ROF data dictionary potentially containing filename mapping
                file_list_widget: QListWidget containing filenames to include
                return_filenames: If True, return (filtered_data, filtered_filenames) tuple

            Returns:
                Filtered ROF timecourse array (or tuple with filenames) or None if filtering fails
            """
            # Get list of filenames from UI widget
            filenames_in_list = []
            for i in range(file_list_widget.count()):
                item = file_list_widget.item(i)
                if item is not None:
                    filenames_in_list.append(item.text())

            if not filenames_in_list:
                if return_filenames:
                    return rof_timecourse, rof_data.get('filenames', [])
                return rof_timecourse

            # If ROF data has filename mapping, use it to filter
            if 'filenames' in rof_data:
                filenames_in_data = rof_data['filenames']
                # Find indices and corresponding filenames that are in the list
                indices_to_keep = []
                filtered_filenames = []
                for i, fname in enumerate(filenames_in_data):
                    if fname in filenames_in_list:
                        indices_to_keep.append(i)
                        filtered_filenames.append(fname)

                if len(indices_to_keep) == 0:
                    if return_filenames:
                        return None, []
                    return None

                # Filter the timecourse data
                if isinstance(rof_timecourse, np.ndarray):
                    filtered_data = rof_timecourse[indices_to_keep, :]
                    if return_filenames:
                        return filtered_data, filtered_filenames
                    return filtered_data

            # If no filename mapping, return original data
            if return_filenames:
                return rof_timecourse, rof_data.get('filenames', [])
            return rof_timecourse

    def _display_rof_tfce_results(
            self, results: List[Dict], comparison_type: str = 'one_sample'
    ) -> None:
            """Display results from ROF TFCE cluster permutation testing.

            Args:
                results: List of result dictionaries for each microstate
                comparison_type: Type of comparison ('one_sample' or 'paired')
            """
            # Summary header
            n_sig = sum(1 for r in results if r['significant'])
            self.ui.stats_textedit.appendPlainText(
                f"Results Summary: {n_sig}/{len(results)} microstates show significant event effects\n"
                f"{'-' * 80}\n"
            )

            # Display results for each microstate
            for result in results:
                microstate = result['microstate']

                if result['significant']:
                    self.ui.stats_textedit.appendPlainText(
                        f"\nMicrostate {microstate}: SIGNIFICANT EVENT EFFECT ***\n"
                    )

                    if comparison_type == 'one_sample':
                        self.ui.stats_textedit.appendPlainText(
                            f"  Subjects: {result['n_subjects']}\n"
                        )
                    else:
                        self.ui.stats_textedit.appendPlainText(
                            f"  Matched pairs: {result['n_pairs']}\n"
                        )

                    self.ui.stats_textedit.appendPlainText(
                        f"  Number of significant clusters: {result['n_clusters']}\n"
                        f"  Average effect size (Cohen's d): {result['avg_cohens_d']:.3f}\n"
                    )

                    # Display details for each cluster
                    for i, cluster in enumerate(result['clusters'], 1):
                        self.ui.stats_textedit.appendPlainText(
                            f"\n  Cluster {i}:\n"
                            f"    Time window: {cluster['time_start']:.1f} - {cluster['time_end']:.1f} ms\n"
                            f"    Duration: {cluster['n_timepoints']} time points\n"
                            f"    Direction: {cluster['direction']}\n"
                            f"    Cohen's d: {cluster['cohens_d']:.3f}\n"
                            f"    p-value: {cluster['p_value']:.4f}\n"
                        )

                        if comparison_type == 'one_sample':
                            self.ui.stats_textedit.appendPlainText(
                                f"    Mean ROF: {cluster['mean_rof']:.4f}\n"
                            )
                        else:
                            self.ui.stats_textedit.appendPlainText(
                                f"    Mean difference: {cluster['mean_diff']:.4f}\n"
                            )
                else:
                    self.ui.stats_textedit.appendPlainText(
                        f"\nMicrostate {microstate}: No significant event effect (ns)\n"
                    )

                    if comparison_type == 'one_sample':
                        self.ui.stats_textedit.appendPlainText(
                            f"  Subjects: {result['n_subjects']}\n"
                        )
                    else:
                        self.ui.stats_textedit.appendPlainText(
                            f"  Matched pairs: {result['n_pairs']}\n"
                        )

            # Overall interpretation
            self.ui.stats_textedit.appendPlainText(
                f"\n{'=' * 80}\n"
                f"Interpretation:\n"
            )

            if n_sig > 0:
                self.ui.stats_textedit.appendPlainText(
                    f"Significant event-related changes in microstate occurrence were detected for "
                    f"{n_sig} out of {len(results)} microstates. The TFCE method identified temporal "
                    f"clusters where the ROF significantly differed from baseline, while controlling "
                    f"for multiple comparisons across time points.\n\n"
                    f"Effect sizes (Cohen's d) are averaged across all significant clusters for each "
                    f"microstate, as recommended by Sassenhagen & Draschkow (2019). Positive effects "
                    f"indicate increased occurrence post-event, while negative effects indicate decreased "
                    f"occurrence relative to baseline.\n"
                )
            else:
                self.ui.stats_textedit.appendPlainText(
                    f"No significant event-related changes in microstate occurrence were detected. "
                    f"This suggests that the event did not produce consistent, temporally-extended "
                    f"modulations of microstate dynamics that survived multiple comparison correction.\n"
                )

            self.ui.stats_textedit.appendPlainText(
                f"\nMethod Notes:\n"
                f"- Analysis period: 20-1000 ms post-event\n"
                f"- Permutations: 5000 (sign-flipping to preserve temporal dependencies)\n"
                f"- Threshold-Free Cluster Enhancement (TFCE) for multiple comparison correction\n"
                f"- Significance threshold: p < 0.05\n"
                f"- Effect sizes calculated using Cohen's d for standardized mean differences\n"
            )

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

            Prioritizes BIDS format (sub-XX) and handles various naming conventions.

            Args:
                filename: Source filename.

            Returns:
                Extracted subject identifier (normalized), or the filename if no match.
            """
            # Pattern 1: BIDS format sub-XX (most common, highest priority)
            # Matches: sub-01, sub-001, sub-1, etc.
            match = re.search(r"sub-(\d+)", filename, re.IGNORECASE)
            if match:
                # Normalize to sub-XX format with zero-padding
                return f"sub-{match.group(1).zfill(2)}"

            # Pattern 2: subjectXX or subjXX format
            match = re.search(r"subj(?:ect)?[-_]?(\d+)", filename, re.IGNORECASE)
            if match:
                return f"subject-{match.group(1).zfill(2)}"

            # Pattern 3: participantXX or partXX format
            match = re.search(r"part(?:icipant)?[-_]?(\d+)", filename, re.IGNORECASE)
            if match:
                return f"participant-{match.group(1).zfill(2)}"

            # Pattern 4: SXX or S_XX format (common in some labs)
            match = re.search(r"[^a-zA-Z]S[-_]?(\d+)", filename)
            if match:
                return f"S{match.group(1).zfill(2)}"

            # Pattern 5: PXX format (participant shorthand)
            match = re.search(r"[^a-zA-Z]P[-_]?(\d+)", filename)
            if match:
                return f"P{match.group(1).zfill(2)}"

            # Pattern 6: Just numbers at the beginning
            match = re.search(r"^(\d+)", filename)
            if match:
                return match.group(1).zfill(2)

            # Fallback: return the filename unchanged
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
            """Apply the detected pattern to update UI with color-coded paired subjects."""
            (
                common_pattern,
                group1_pattern,
                group2_pattern,
                group1_files,
                group2_files,
            ) = pattern_result

            # Create subject ID to number mapping (sorted for consistent ordering)
            subject_ids = sorted(set(
                self._extract_subject_id(f) for f in group1_files + group2_files
            ))
            subject_to_number = {subj: idx + 1 for idx, subj in enumerate(subject_ids)}

            # Generate colors for each subject pair
            subject_colors = self._generate_subject_colors(group1_files, group2_files)

            # Sort files by subject number for consistent display
            def sort_key(filename):
                return subject_to_number.get(self._extract_subject_id(filename), 999)

            group1_files_sorted = sorted(group1_files, key=sort_key)
            group2_files_sorted = sorted(group2_files, key=sort_key)

            # Update file lists with numbering and color coding
            self.ui.study1_file_list.clear()
            for filename in group1_files_sorted:
                subject_id = self._extract_subject_id(filename)
                number = subject_to_number.get(subject_id, 0)
                display_name = f"{number}. {filename}"
                item = QListWidgetItem(display_name)
                item.setData(Qt.UserRole, filename)  # Store original filename
                if subject_id in subject_colors:
                    item.setBackground(QBrush(subject_colors[subject_id]))
                self.ui.study1_file_list.addItem(item)

            self.ui.study2_file_list.clear()
            for filename in group2_files_sorted:
                subject_id = self._extract_subject_id(filename)
                number = subject_to_number.get(subject_id, 0)
                display_name = f"{number}. {filename}"
                item = QListWidgetItem(display_name)
                item.setData(Qt.UserRole, filename)  # Store original filename
                if subject_id in subject_colors:
                    item.setBackground(QBrush(subject_colors[subject_id]))
                self.ui.study2_file_list.addItem(item)
            
            # Update the stored file lists with sorted order (original filenames)
            group1_files = group1_files_sorted
            group2_files = group2_files_sorted

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

            # Set paired design radio since subjects have been successfully matched
            self.ui.analyze_design_paired_radio.setChecked(True)

            # Update plot button text now that study2_file_list has been populated
            self._update_plot_button_text()

    def _generate_subject_colors(
            self, group1_files: List[str], group2_files: List[str]
    ) -> Dict[str, QColor]:
            """Generate unique pastel colors for each subject pair.

            Args:
                group1_files: List of filenames in group 1.
                group2_files: List of filenames in group 2.

            Returns:
                Dictionary mapping subject IDs to QColor objects.
            """
            # Collect all unique subject IDs
            subject_ids = set()
            for filename in group1_files + group2_files:
                subject_id = self._extract_subject_id(filename)
                subject_ids.add(subject_id)

            # Generate pastel colors using golden ratio for good distribution
            subject_colors = {}
            golden_ratio = 0.618033988749895
            hue = 0.0

            for subject_id in sorted(subject_ids):
                # Generate pastel color (high saturation ~0.4, high value ~0.95)
                hue = (hue + golden_ratio) % 1.0
                # Convert HSV to RGB
                color = QColor.fromHsvF(hue, 0.35, 0.95)
                subject_colors[subject_id] = color

            return subject_colors

    @staticmethod
    def _detect_filename_patterns(filenames: List[str]) -> Optional[Tuple]:
            """Detect common patterns in filenames that repeat exactly twice.

            Uses a robust, multi-strategy approach:
            1. Subject-based grouping (most reliable for sub-# patterns)
            2. Semantic component parsing
            3. Flexible delimiter-based fallback

            Args:
                filenames: List of filenames to analyze.

            Returns:
                Tuple of (common_pattern, group1_pattern, group2_pattern, group1_files,
                group2_files) or None if no suitable pattern is found.
            """
            # Strategy 1: Subject-based approach (most robust for sub-# patterns)
            result = CompareStudiesWindow._detect_subject_based_patterns(filenames)
            if result is not None:
                return result

            # Strategy 2: Semantic approach
            result = CompareStudiesWindow._detect_semantic_patterns(filenames)
            if result is not None:
                return result

            # Strategy 3: Fallback to flexible delimiter-based approach
            return CompareStudiesWindow._detect_flexible_delimiter_patterns(filenames)

    @staticmethod
    def _detect_subject_based_patterns(filenames: List[str]) -> Optional[Tuple]:
            """Detect patterns by grouping files by subject ID.

            This is the most robust approach for standard naming conventions
            where subjects are identified by sub-#, subject-#, S#, etc.
            Each subject should have exactly 2 files with different conditions.

            Args:
                filenames: List of filenames to analyze.

            Returns:
                Tuple of (common_pattern, group1_pattern, group2_pattern, group1_files,
                group2_files) or None if no suitable pattern is found.
            """
            # Group files by subject ID
            subject_groups = defaultdict(list)
            for filename in filenames:
                subject_id = CompareStudiesWindow._extract_subject_id(filename)
                subject_groups[subject_id].append(filename)

            # Filter to subjects with exactly 2 files
            valid_subjects = {
                subj: files for subj, files in subject_groups.items()
                if len(files) == 2
            }

            if len(valid_subjects) < 2:  # Need at least 2 subjects with pairs
                return None

            # For each subject pair, find what component varies
            condition_to_files = defaultdict(list)  # condition -> list of filenames
            condition_pairs_found = []  # Track (cond1, cond2) pairs for validation

            for subject_id, files in valid_subjects.items():
                file1, file2 = sorted(files)  # Sort for consistency

                # Find the varying conditions between the two filenames
                cond1, cond2 = CompareStudiesWindow._find_varying_conditions(file1, file2)

                if cond1 is None or cond2 is None:
                    continue

                # Normalize to ensure consistent ordering (alphabetically)
                if cond1 > cond2:
                    cond1, cond2 = cond2, cond1
                    file1, file2 = file2, file1

                condition_to_files[cond1].append(file1)
                condition_to_files[cond2].append(file2)
                condition_pairs_found.append((cond1, cond2))

            # Check if we have exactly 2 conditions
            if len(condition_to_files) != 2:
                return None

            # Verify all subjects have the same two conditions
            if condition_pairs_found:
                first_pair = condition_pairs_found[0]
                if not all(pair == first_pair for pair in condition_pairs_found):
                    return None

            condition_values = sorted(list(condition_to_files.keys()))
            group1_files = condition_to_files[condition_values[0]]
            group2_files = condition_to_files[condition_values[1]]

            # Verify we have equal groups with at least 2 files each
            if len(group1_files) < 2 or len(group1_files) != len(group2_files):
                return None

            common_pattern = f"condition:[{condition_values[0]}|{condition_values[1]}]"
            return (
                common_pattern,
                condition_values[0],
                condition_values[1],
                sorted(group1_files),
                sorted(group2_files),
            )

    @staticmethod
    def _find_varying_conditions(file1: str, file2: str) -> Tuple[Optional[str], Optional[str]]:
            """Find what condition/component varies between two filenames from the same subject.

            Args:
                file1: First filename.
                file2: Second filename.

            Returns:
                Tuple of (condition1, condition2) representing the differing component,
                or (None, None) if no clear condition difference found.
            """
            # Known condition keywords and their normalized forms
            condition_keywords = {
                # Eyes open/closed variations
                'eyesclosed': 'eyesclosed', 'eyesclose': 'eyesclosed', 'eyeclose': 'eyesclosed',
                'ec': 'eyesclosed', 'closed': 'eyesclosed', 'close': 'eyesclosed',
                'eyesopen': 'eyesopen', 'eyeopen': 'eyesopen',
                'eo': 'eyesopen', 'open': 'eyesopen',
                # Pre/post variations
                'pre': 'pre', 'post': 'post', 'before': 'pre', 'after': 'post',
                # Active/baseline variations
                'active': 'active', 'baseline': 'baseline', 'base': 'baseline',
                'rest': 'rest', 'resting': 'rest', 'task': 'task',
                # Treatment variations
                'sham': 'sham', 'real': 'real', 'verum': 'real',
                'control': 'control', 'treatment': 'treatment',
                'placebo': 'placebo', 'drug': 'drug',
                # Generic labels
                'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd',
                '1': '1', '2': '2', '3': '3', '4': '4',
                'cond1': 'cond1', 'cond2': 'cond2',
                'condition1': 'cond1', 'condition2': 'cond2',
                # Stimulation variations
                'stim': 'stim', 'nostim': 'nostim', 'stimulation': 'stim',
                # Session/run variations that might indicate conditions
                'session1': 'session1', 'session2': 'session2',
                'run1': 'run1', 'run2': 'run2',
            }

            def normalize_filename(filename):
                """Normalize filename for comparison."""
                name = filename.lower()
                # Remove common file extensions
                name = re.sub(r'\.(eeg|set|fdt|edf|bdf|cnt|vhdr|vmrk|fif|gz|mat)$', '', name)
                # Remove common processing suffixes
                name = re.sub(r'[_\-](eeg|clean|proc|preproc|processed|raw|filt|filtered|ica|epoch|avg)$', '', name)
                return name

            clean1 = normalize_filename(file1)
            clean2 = normalize_filename(file2)

            # Strategy 1: Check for BIDS-style task-condition or acq-condition format
            def extract_bids_condition(clean_name):
                """Extract condition from BIDS-style naming (task-X or acq-X)."""
                # Match task-<condition> or acq-<condition>
                for prefix in ['task', 'acq']:
                    match = re.search(rf'{prefix}-([a-zA-Z0-9]+)', clean_name)
                    if match:
                        value = match.group(1).lower()
                        # Check for known condition keywords
                        if value in condition_keywords:
                            return condition_keywords[value]
                        # Check if value contains a known condition
                        for keyword, normalized in condition_keywords.items():
                            if keyword in value and len(keyword) >= 2:  # Avoid single char matches
                                return normalized
                        return value
                return None

            bids_cond1 = extract_bids_condition(clean1)
            bids_cond2 = extract_bids_condition(clean2)

            if bids_cond1 and bids_cond2 and bids_cond1 != bids_cond2:
                return bids_cond1, bids_cond2

            # Strategy 2: Split by delimiters and find differing parts
            parts1 = re.split(r'[_\-\.]', clean1)
            parts2 = re.split(r'[_\-\.]', clean2)

            # Remove empty parts and subject identifiers
            def filter_parts(parts):
                filtered = []
                for p in parts:
                    if not p:
                        continue
                    # Skip subject identifier parts
                    if re.match(r'^(sub|subject|subj|s|p|participant)$', p, re.IGNORECASE):
                        continue
                    if re.match(r'^\d{1,3}$', p):  # Skip pure numbers (likely subject IDs)
                        continue
                    filtered.append(p)
                return filtered

            parts1 = filter_parts(parts1)
            parts2 = filter_parts(parts2)

            # Find parts unique to each filename
            set1 = set(parts1)
            set2 = set(parts2)

            unique_to_1 = set1 - set2
            unique_to_2 = set2 - set1

            def extract_condition_from_parts(parts_set):
                """Extract condition from a set of filename parts."""
                # First, check for exact matches with known conditions
                for part in parts_set:
                    if part in condition_keywords:
                        return condition_keywords[part]

                # Check for partial matches (condition contained in part)
                for part in parts_set:
                    for keyword, normalized in condition_keywords.items():
                        # Require minimum length to avoid false matches
                        if len(keyword) >= 2 and keyword in part:
                            return normalized

                # If no known condition found, return the first unique part
                if parts_set:
                    # Prefer longer parts (more likely to be meaningful)
                    sorted_parts = sorted(parts_set, key=len, reverse=True)
                    return sorted_parts[0]

                return None

            cond1 = extract_condition_from_parts(unique_to_1)
            cond2 = extract_condition_from_parts(unique_to_2)

            if cond1 and cond2 and cond1 != cond2:
                return cond1, cond2

            # Strategy 3: Compare parts position by position
            min_len = min(len(parts1), len(parts2))
            for i in range(min_len):
                if parts1[i] != parts2[i]:
                    p1, p2 = parts1[i], parts2[i]
                    # Normalize if they're known conditions
                    c1 = condition_keywords.get(p1, p1)
                    c2 = condition_keywords.get(p2, p2)
                    if c1 != c2:
                        return c1, c2

            return None, None

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
            Supports BIDS format and custom naming conventions.

            Args:
                filename: Filename to parse.

            Returns:
                Dictionary of component_type -> component_value.
            """
            components = {}
            filename_lower = filename.lower()

            # Remove common file extensions and suffixes
            clean_name = re.sub(
                r"[_\-](eeg|clean|proc|preproc|processed|raw|filt|filtered|ica|epoch|avg)$",
                "",
                filename_lower
            )
            clean_name = re.sub(
                r"\.(eeg|set|fdt|edf|bdf|cnt|vhdr|vmrk|fif|gz|mat)$", "", clean_name
            )

            # Parse various components (order matters - more specific first)
            CompareStudiesWindow._parse_subject_components(clean_name, components)
            CompareStudiesWindow._parse_session_components(clean_name, components)
            CompareStudiesWindow._parse_task_components(clean_name, components)
            CompareStudiesWindow._parse_acquisition_components(clean_name, components)
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
    def _parse_acquisition_components(clean_name: str, components: Dict) -> None:
            """Parse acquisition-related components (BIDS acq- field).

            This handles patterns like:
            - acq-dlpfcactive → sets condition='active'
            - acq-dlpfcsham → sets condition='sham'
            - acq-active → sets condition='active'
            - acq-sham → sets condition='sham'
            - acq-rest → sets condition='rest'

            Note: Sets 'condition' directly (not 'acquisition') to avoid duplicate components.
            """
            # BIDS acquisition field pattern: acq-<label>
            acq_pattern = r"acq-([a-zA-Z0-9]+)"
            match = re.search(acq_pattern, clean_name)

            if match:
                acq_value = match.group(1).lower()

                # Extract meaningful condition from acquisition label
                # Handle compound labels like "dlpfcactive" -> "active"
                condition_keywords = {
                    'active': 'active',
                    'sham': 'sham',
                    'rest': 'rest',
                    'baseline': 'baseline',
                    'pre': 'pre',
                    'post': 'post',
                    'stim': 'stimulation',
                    'nostim': 'nostimulation',
                    'real': 'real',
                    'control': 'control',
                    'treatment': 'treatment'
                }

                # Try to find a condition keyword in the acquisition value
                detected_condition = None
                for keyword, condition_name in condition_keywords.items():
                    if keyword in acq_value:
                        detected_condition = condition_name
                        break

                # Set condition directly (not acquisition) to avoid duplicate components
                if detected_condition:
                    components["condition"] = detected_condition
                else:
                    components["condition"] = acq_value

                return

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
            """Parse condition-related components.

            Note: If acquisition field already set a condition, don't override it.
            """
            # If condition already set by acquisition parsing, don't override
            if "condition" in components:
                return

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
                (r"\b(sham)\b", "sham"),
                (r"\b(real)\b", "real"),
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
                            hasattr(comet_instance, "n_maps")
                            and comet_instance.n_maps is not None
                    ):
                        logger.processing_info(
                            "STUDY_STATUS", f"Number of Maps: {comet_instance.n_maps}"
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