"""UI for visualizing microstate features in tables and plots."""

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAbstractItemView, QActionGroup, QMainWindow, QSizePolicy

from features_utils.feature_io import FeatureIO
from gui_utils.set_widgets_status import set_widgets_status
from gui_utils.export_utils import get_save_file_path, save_matplotlib_figure
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


class FeatureVisualizationWindow(QMainWindow):
    """Window for viewing microstate feature tables and plots.

    Provides controls to visualize static and dynamic feature summaries, compare
    groups, and export figures with configurable styles.

    Attributes:
      comet: COMET toolbox instance providing study configuration and metadata.
      feature_mode (list[str]): Available feature modes (e.g., ["averaged"], "sliding").
      figure (matplotlib.figure.Figure): Figure used for plotting.
      canvas (matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg): Canvas displaying the figure.
      toolbar (matplotlib.backends.backend_qt5agg.NavigationToolbar2QT): Matplotlib toolbar for plot interaction.
      current_colormap (str): Current matplotlib colormap name.
      current_font_family (str): Current font family used for labels and titles.
      display_options (dict[str, bool]): Visibility settings for legend, grid, axes.
    """

    def __init__(self, context, parent=None, tbx=None):
        """Initialize the feature visualization window and UI.

        Args:
          context: Resource/context provider used to resolve the `.ui` file.
          parent: Optional parent widget.
          tbx: COMET toolbox instance for the active study.

        Returns:
          None
        """
        super().__init__(parent)

        self.comet = tbx
        self.ui = self.load_ui(context)

        # Initialize feature_mode with a default value (will be set from main window)
        self.feature_mode = ["averaged"]  # Default fallback
        
        # Store all available features (to restore when switching modes)
        self.all_available_features = []

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas.setMinimumHeight(100)
        
        # Add navigation toolbar for zoom, pan, save, etc.
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ui.Figure_Layout.addWidget(self.toolbar)
        self.ui.Figure_Layout.addWidget(self.canvas)

        self.setup_window()
        self.setup_style_actions()
        self.bind_events()
        
        # Update radio button text for epoched data
        self.update_radio_text_for_datatype()
        
        self.feature_visualization_controller()

    def update_radio_text_for_datatype(self):
        """Update radio button text based on data type.
        
        Returns:
          None
        """
        if hasattr(self.comet, "datatype") and self.comet.datatype == "epoched":
            # For epoched data, "sliding" mode shows pre/post event features per trial
            if hasattr(self.ui, "dynamic_radio"):
                self.ui.dynamic_radio.setText("Pre/Post Event Features")
        else:
            # For non-epoched data, use standard text
            if hasattr(self.ui, "dynamic_radio"):
                self.ui.dynamic_radio.setText("Windowed Features")
    
    def get_selected_feature_code(self):
        """Return the short code for the selected feature full name.

        Returns:
          str: Feature short code (e.g., "COV", "OCC") or the full name if not found.
        """
        selected_full_name = self.ui.feature_combo.currentText()

        # Create reverse mapping from full names to short codes using comet's dictionary
        if hasattr(self.comet, "feature_list_dictionary") and self.comet.feature_list_dictionary:
            reverse_dict = {v: k for k, v in self.comet.feature_list_dictionary.items()}
            return reverse_dict.get(selected_full_name, selected_full_name)

        return selected_full_name

    def _update_radio_button_states(self):
        """Enable/disable static and dynamic radio buttons based on available feature modes.
        
        Returns:
          None
        """
        # Check if we have any feature modes available
        if not self.feature_mode:
            # No feature modes available - disable both radios
            if hasattr(self.ui, "static_radio"):
                self.ui.static_radio.setEnabled(False)
            if hasattr(self.ui, "dynamic_radio"):
                self.ui.dynamic_radio.setEnabled(False)
            return
        
        # Enable static radio if averaged or pre_post features are available
        static_available = "averaged" in self.feature_mode or "pre_post" in self.feature_mode
        if hasattr(self.ui, "static_radio"):
            self.ui.static_radio.setEnabled(static_available)
            # If static becomes unavailable and it was selected, switch to dynamic
            if not static_available and self.ui.static_radio.isChecked():
                if hasattr(self.ui, "dynamic_radio") and "sliding" in self.feature_mode:
                    self.ui.dynamic_radio.setChecked(True)
        
        # Enable dynamic radio if sliding or pre_post features are available
        dynamic_available = "sliding" in self.feature_mode or "pre_post" in self.feature_mode
        if hasattr(self.ui, "dynamic_radio"):
            self.ui.dynamic_radio.setEnabled(dynamic_available)
            # If dynamic becomes unavailable and it was selected, switch to static
            if not dynamic_available and self.ui.dynamic_radio.isChecked():
                if hasattr(self.ui, "static_radio") and ("averaged" in self.feature_mode or "pre_post" in self.feature_mode):
                    self.ui.static_radio.setChecked(True)
        
        # If only one mode is available, select it automatically
        if static_available and not dynamic_available and hasattr(self.ui, "static_radio"):
            self.ui.static_radio.setChecked(True)
        elif dynamic_available and not static_available and hasattr(self.ui, "dynamic_radio"):
            self.ui.dynamic_radio.setChecked(True)
        elif static_available and dynamic_available:
            # If both are available and none are selected, default to static
            if (hasattr(self.ui, "static_radio") and hasattr(self.ui, "dynamic_radio") 
                and not self.ui.static_radio.isChecked() and not self.ui.dynamic_radio.isChecked()):
                self.ui.static_radio.setChecked(True)

    def update_all_available_features(self):
        """Update the stored list of all available features from the dropdown.
        
        This should be called after the main window populates the feature combo.
        
        Returns:
          None
        """
        if not hasattr(self.ui, "feature_combo"):
            return
        
        # Store all features currently in dropdown
        self.all_available_features = []
        for i in range(self.ui.feature_combo.count()):
            self.all_available_features.append(self.ui.feature_combo.itemText(i))
    
    def _filter_features_by_mode(self):
        """Filter feature dropdown to show only compatible features for the current mode.
        
        For dynamic mode with pre_post features (epoched data), ROF and RTF are not compatible
        with the pre/post bar chart visualization and should be hidden.
        
        Returns:
          None
        """
        if not hasattr(self.ui, "feature_combo"):
            return
        
        # Update stored list if it's empty (first call)
        if not self.all_available_features:
            self.update_all_available_features()
        
        # If no features stored, nothing to filter
        if not self.all_available_features:
            return
        
        # Determine current mode
        static_mode = hasattr(self.ui, "static_radio") and self.ui.static_radio.isChecked()
        dynamic_mode = hasattr(self.ui, "dynamic_radio") and self.ui.dynamic_radio.isChecked()
        
        # Check if this is epoched data with only pre_post features in dynamic mode
        is_epoched = hasattr(self.comet, "datatype") and self.comet.datatype == "epoched"
        has_only_pre_post = "pre_post" in self.feature_mode and "sliding" not in self.feature_mode
        
        # Store the currently selected feature
        current_selection = self.ui.feature_combo.currentText()
        
        # Determine which features to show from the complete list
        features_to_show = []
        for full_feature_name in self.all_available_features:
            # Get feature code from full name
            if hasattr(self.comet, "feature_list_dictionary"):
                reverse_dict = {v: k for k, v in self.comet.feature_list_dictionary.items()}
                feature_code = reverse_dict.get(full_feature_name, full_feature_name)
            else:
                feature_code = full_feature_name
            
            # Filter logic:
            # In dynamic mode with pre_post features only (no sliding), exclude ROF and RTF
            if dynamic_mode and is_epoched and has_only_pre_post:
                if feature_code in ["ROF", "RTF"]:
                    continue  # Skip ROF and RTF in this mode
            
            features_to_show.append(full_feature_name)
        
        # Get current features in dropdown
        current_features = []
        for i in range(self.ui.feature_combo.count()):
            current_features.append(self.ui.feature_combo.itemText(i))
        
        # Update dropdown if the list changed
        if set(features_to_show) != set(current_features):
            # Block signals to prevent triggering controller during update
            self.ui.feature_combo.blockSignals(True)
            self.ui.feature_combo.clear()
            self.ui.feature_combo.addItems(features_to_show)
            
            # Restore selection if still available, otherwise select first
            if current_selection in features_to_show:
                index = self.ui.feature_combo.findText(current_selection)
                self.ui.feature_combo.setCurrentIndex(index)
            elif features_to_show:
                self.ui.feature_combo.setCurrentIndex(0)
            
            self.ui.feature_combo.blockSignals(False)

    def setup_style_actions(self):
        """Set up colormap, font family, and style action groups.

        Returns:
          None
        """
        # Create action group for colormap actions
        self.colormap_action_group = QActionGroup(self)
        self.colormap_action_group.setExclusive(True)

        # Dictionary mapping action names to colormap names
        self.colormap_mapping = {
            "cmap_set1": "Set1",
            "cmap_set2": "Set2",
            "cmap_tab10": "tab10",
            "cmap_dark2": "Dark2",
            "cmap_accent": "Accent",
        }

        # Add colormap actions to the group and make them checkable
        for action_name, colormap_name in self.colormap_mapping.items():
            if hasattr(self.ui, action_name):
                action = getattr(self.ui, action_name)
                action.setCheckable(True)
                self.colormap_action_group.addAction(action)

                # Connect action to colormap change handler
                action.triggered.connect(lambda checked, cm=colormap_name: self.change_colormap(cm))

        self.ui.cmap_tab10.setChecked(True)
        self.current_colormap = "tab10"

        # Create action group for font family actions
        self.font_action_group = QActionGroup(self)
        self.font_action_group.setExclusive(True)

        # Dictionary mapping action names to font family names
        self.font_mapping = {
            "font_arial": "Arial",
            "font_calibri": "Calibri",
            "font_times": "Times New Roman",
        }

        # Add font actions to the group and make them checkable
        for action_name, font_name in self.font_mapping.items():
            if hasattr(self.ui, action_name):
                action = getattr(self.ui, action_name)
                action.setCheckable(True)
                self.font_action_group.addAction(action)

                # Connect action to font change handler
                action.triggered.connect(
                    lambda checked, font=font_name: self.change_font_family(font)
                )

        self.ui.font_arial.setChecked(True)
        self.current_font_family = "Arial"

        # Setup independent checkable actions (not mutually exclusive)
        self.setup_display_options()

        # Setup font size actions (if using submenu approach)
        self.setup_font_size_actions()

    def setup_display_options(self):
        """Set up independent display options (legend, grid, axes).

        Returns:
          None
        """
        # Dictionary mapping action names to their default states
        self.display_options = {
            "show_legend": True,  # Default: show legend
            "show_grid": False,  # Default: no grid
            "show_axes": True,  # Default: show axes
        }

        # Setup each display option action
        for action_name, default_state in self.display_options.items():
            if hasattr(self.ui, action_name):
                action = getattr(self.ui, action_name)
                action.setCheckable(True)
                action.setChecked(default_state)

                # Connect action to display option change handler
                action.triggered.connect(
                    lambda checked, option=action_name: self.change_display_option(option, checked)
                )

    def setup_font_size_actions(self):
        """Set up font size actions for title, label, tick, and legend.

        Each category has its own mutually exclusive group.

        Returns:
          None
        """
        # Define font size mappings for each category
        self.font_size_categories = {
            "title": {
                "title_font_small": 14,
                "title_font_medium": 18,
                "title_font_large": 22,
                "title_font_xlarge": 26,
            },
            "label": {
                "label_font_small": 10,
                "label_font_medium": 14,
                "label_font_large": 18,
                "label_font_xlarge": 22,
            },
            "tick": {
                "tick_font_small": 10,
                "tick_font_medium": 14,
                "tick_font_large": 18,
                "tick_font_xlarge": 22,
            },
            "legend": {
                "legend_font_small": 10,
                "legend_font_medium": 14,
                "legend_font_large": 18,
                "legend_font_xlarge": 22,
            },
        }

        # Create action groups for each font category
        self.font_size_action_groups = {}

        for category, font_mapping in self.font_size_categories.items():
            # Create action group for this category
            action_group = QActionGroup(self)
            action_group.setExclusive(True)
            self.font_size_action_groups[category] = action_group

            # Add actions to the group
            for action_name, font_size in font_mapping.items():
                if hasattr(self.ui, action_name):
                    action = getattr(self.ui, action_name)
                    action.setCheckable(True)
                    action_group.addAction(action)

                    # Connect action to font size change handler
                    action.triggered.connect(
                        lambda checked, cat=category, size=font_size: self.change_font_size_category(
                            cat, size
                        )
                    )

            # Set default font size (medium for all categories)
            default_action_name = f"{category}_font_medium"
            if hasattr(self.ui, default_action_name):
                getattr(self.ui, default_action_name).setChecked(True)

    def change_colormap(self, colormap_name):
        """Handle colormap change from menu actions.

        Args:
          colormap_name (str): Matplotlib colormap name.

        Returns:
          None
        """
        self.current_colormap = colormap_name
        self._maybe_refresh_rof_plot()
        self._maybe_refresh_rtf_plot()

    def change_font_family(self, font_name):
        """Handle font family change from menu actions.

        Args:
          font_name (str): Font family name (e.g., "Arial").

        Returns:
          None
        """
        self.current_font_family = font_name
        self._maybe_refresh_rof_plot()
        self._maybe_refresh_rtf_plot()

    def change_display_option(self, option_name, checked):
        """Handle display option change from menu actions.

        Args:
          option_name (str): Option key ("show_legend", "show_grid", "show_axes").
          checked (bool): New state of the option.

        Returns:
          None
        """
        # print(f"Display option '{option_name}' set to: {checked}")
        self._maybe_refresh_rof_plot()
        self._maybe_refresh_rtf_plot()

    def change_font_size_category(self, category, font_size):
        """Handle font size change for a specific category.

        Args:
          category (str): One of {"title", "label", "tick", "legend"}.
          font_size (int): Selected font size value.

        Returns:
          None
        """
        # Update the UI input fields if they exist
        if category == "title" and hasattr(self.ui, "font_size_input"):
            self.ui.font_size_input.setText(str(font_size))
        elif category == "tick" and hasattr(self.ui, "label_size_input"):
            self.ui.label_size_input.setText(str(font_size))

        self._maybe_refresh_rof_plot()
        self._maybe_refresh_rtf_plot()

    def _maybe_refresh_rof_plot(self):
        """Replot ROF immediately if selected and visible.

        Returns:
          None
        """
        if hasattr(self.ui, "plot_rof_button") and self.get_selected_feature_code() == "ROF":
            # Directly call plot function so updates are instant
            self.plot_rof_timeseries()
    
    def _maybe_refresh_rtf_plot(self):
        """Replot RTF immediately if selected and visible.

        Returns:
          None
        """
        if hasattr(self.ui, "plot_rtf_button") and self.get_selected_feature_code() == "RTF":
            # Directly call plot function so updates are instant
            self.plot_rtf_heatmap()

    # ------------------------- Button highlighting -------------------------
    def _reset_plot_button_styles(self):
        """Reset styles for all plot buttons.

        Returns:
          None
        """
        for btn_name in [
            "plot_static_violin_plot_button",
            "plot_static_box_plot_button",
            "plot_all_dynamic_button",
            "plot_heatmap_button",
            "plot_rof_button",
            "plot_rtf_button",
        ]:
            if hasattr(self.ui, btn_name):
                getattr(self.ui, btn_name).setStyleSheet("")

    def _highlight_button(self, button_attr_name):
        """Set the given button background to green and reset others.

        Args:
          button_attr_name (str): Attribute name of the QPushButton on `self.ui`.

        Returns:
          None
        """
        self._reset_plot_button_styles()
        if hasattr(self.ui, button_attr_name):
            getattr(self.ui, button_attr_name).setStyleSheet(
                "background-color: #4CAF50; color: white;"
            )

    def get_selected_colormap(self):
        """Get the currently selected colormap from the action group.

        Returns:
          str: Colormap name; falls back to current/default if none selected.
        """
        checked_action = self.colormap_action_group.checkedAction()
        if checked_action:
            # Find the colormap name from the action
            for action_name, colormap_name in self.colormap_mapping.items():
                if (
                    hasattr(self.ui, action_name)
                    and getattr(self.ui, action_name) == checked_action
                ):
                    return colormap_name

        # Fallback to combobox or default
        return getattr(self, "current_colormap", "viridis")

    def get_selected_font_family(self):
        """Get the currently selected font family from the action group.

        Returns:
          str: Font family name; falls back to current/default if none selected.
        """
        checked_action = self.font_action_group.checkedAction()
        if checked_action:
            # Find the font name from the action
            for action_name, font_name in self.font_mapping.items():
                if (
                    hasattr(self.ui, action_name)
                    and getattr(self.ui, action_name) == checked_action
                ):
                    return font_name

        # Fallback to default
        return getattr(self, "current_font_family", "Arial")

    def get_display_options(self):
        """Get the current state of display options.

        Returns:
          dict[str, bool]: Mapping of option name to enabled state.
        """
        options = {}
        for option_name in self.display_options:
            if hasattr(self.ui, option_name):
                action = getattr(self.ui, option_name)
                options[option_name] = action.isChecked()
            else:
                # Use default if action doesn't exist
                options[option_name] = self.display_options[option_name]

        return options

    def get_selected_font_size_from_menu(self):
        """Get the selected font sizes for all categories from action groups.

        Returns:
          dict[str, int]: Mapping of category to selected font size.
        """
        font_sizes = {}

        for category, action_group in self.font_size_action_groups.items():
            checked_action = action_group.checkedAction()
            if checked_action:
                # Find the font size from the action
                for action_name, font_size in self.font_size_categories[category].items():
                    if (
                        hasattr(self.ui, action_name)
                        and getattr(self.ui, action_name) == checked_action
                    ):
                        font_sizes[category] = font_size
                        break

            # Set default if no action is checked
            if category not in font_sizes:
                defaults = {"title": 18, "label": 14, "tick": 12, "legend": 12}
                font_sizes[category] = defaults.get(category, 12)

        return font_sizes

    def load_ui(self, context):
        """Load and return the Qt Designer UI for this window.

        Args:
          context: Resource/context provider used to resolve the `.ui` path.

        Returns:
          Any: Loaded UI bound to this window instance.
        """
        os.path.dirname(__file__)
        return uic.loadUi(context.get_resource("FeatureVisualizationWindow.ui"), self)

    def setup_window(self):
        """Configure window title and flags.

        Returns:
          None
        """
        self.setWindowTitle("Visualization of the extracted features")
        # Add maximize button to the window
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

    def bind_events(self):
        """Connect UI signals to their handlers.

        Returns:
          None
        """
        self.ui.plot_static_violin_plot_button.clicked.connect(self.show_static_violin_all)
        self.ui.plot_static_box_plot_button.clicked.connect(self.show_static_box_plot)
        self.ui.plot_all_dynamic_button.clicked.connect(self.show_dynamic_line_all)

        self.ui.plot_heatmap_button.clicked.connect(self.show_tp_heatmap)
        self.ui.plot_groups_button.clicked.connect(self.compare_groups)
        self.ui.add_group_a_button.clicked.connect(self.move_file_from_all_to_a)
        self.ui.add_group_b_button.clicked.connect(self.move_file_from_all_to_b)
        self.ui.remove_group_a_button.clicked.connect(self.move_file_from_a_to_all)
        self.ui.remove_group_b_button.clicked.connect(self.move_file_from_b_to_all)
        self.ui.reset_groups_button.clicked.connect(self.reset_groups)

        # Connect export feature image button and add keyboard shortcut
        self.ui.export_feature_image_button.triggered.connect(self.export_feature_image)
        self.ui.export_feature_image_button.setShortcut(QKeySequence("Ctrl+S"))

        self.ui.feature_combo.currentTextChanged.connect(self.feature_visualization_controller)
        self.ui.all_files_list.itemSelectionChanged.connect(self.feature_visualization_controller)
        buttons = [
            self.ui.add_group_a_button,
            self.ui.add_group_b_button,
            self.ui.remove_group_a_button,
            self.ui.remove_group_b_button,
            self.ui.reset_groups_button,
        ]
        for button in buttons:
            button.clicked.connect(self.feature_visualization_controller)

        # Bind ROF plot button if exists in UI
        if hasattr(self.ui, "plot_rof_button"):
            self.ui.plot_rof_button.clicked.connect(self.plot_rof_timeseries)
        
        # Bind RTF plot button if exists in UI
        if hasattr(self.ui, "plot_rtf_button"):
            self.ui.plot_rtf_button.clicked.connect(self.plot_rtf_heatmap)

        # Trigger controller when feature selection changes
        if hasattr(self.ui, "feature_combo"):
            self.ui.feature_combo.currentIndexChanged.connect(self.feature_visualization_controller)

        # Trigger controller when static/dynamic radio toggled
        if hasattr(self.ui, "static_radio"):
            self.ui.static_radio.toggled.connect(self.feature_visualization_controller)
        if hasattr(self.ui, "dynamic_radio"):
            self.ui.dynamic_radio.toggled.connect(self.feature_visualization_controller)

        # Bind selection helper widgets if they exist
        if hasattr(self.ui, "select_all_checkbox"):
            self.ui.select_all_checkbox.stateChanged.connect(self.handle_select_all_checkbox)
        if hasattr(self.ui, "select_pattern_checkbox"):
            self.ui.select_pattern_checkbox.stateChanged.connect(self.apply_pattern_selection)
        if hasattr(self.ui, "select_pattern_input"):
            self.ui.select_pattern_input.textChanged.connect(self.apply_pattern_selection)

        # Update selected file lineedit initially
        if hasattr(self.ui, "selected_file_lineedit"):
            self.update_selected_file_lineedit()

    def export_feature_image(self):
        """Export feature visualization image to a chosen file path.

        Returns:
          None
        """
        # Get study name for default filename
        study_name = getattr(self.comet, "study_name", "features")

        # Get current feature name for filename
        current_feature = self.ui.feature_combo.currentText()
        if current_feature:
            default_filename = f"{study_name}_{current_feature}_visualization.pdf"
        else:
            default_filename = f"{study_name}_features.pdf"

        # Get default directory from comet object (if provided)
        default_dir = getattr(self.comet, "save_dir", "")
        default_path = os.path.join(default_dir, default_filename) if default_dir else default_filename

        # Reuse common dialog and save logic
        file_name = get_save_file_path(
            self,
            default_path,
            "Choose a location and filename to save the feature visualization",
        )

        if file_name:
            save_matplotlib_figure(
                figure=self.figure,
                canvas=self.canvas,
                file_name=file_name,
                title_text=None,
                title_fontsize=18,
                font_family="Arial",
            )

    def feature_visualization_controller(self):
        """Control the behavior of the window based on user selections.

        Updates widget enablement/visibility and triggers appropriate plotting.

        Returns:
          None
        """
        # Update radio button states based on available feature modes
        self._update_radio_button_states()
        
        # Filter features based on current mode (must be done before determining mode for plotting)
        self._filter_features_by_mode()
        
        # Determine current visualization mode based on radio buttons
        static_mode = hasattr(self.ui, "static_radio") and self.ui.static_radio.isChecked()
        dynamic_mode = hasattr(self.ui, "dynamic_radio") and self.ui.dynamic_radio.isChecked()
        
        # Check if this is epoched data
        is_epoched = hasattr(self.comet, "datatype") and self.comet.datatype == "epoched"

        # Adjust list selection behavior
        if dynamic_mode:
            # For epoched data pre/post event features, allow multiple selections
            # For non-epoched time-series, only single file
            if is_epoched:
                self.ui.all_files_list.setSelectionMode(QAbstractItemView.MultiSelection)
            else:
                self.ui.all_files_list.setSelectionMode(QAbstractItemView.SingleSelection)
        else:
            # Allow multiple selections for global (averaged) features
            self.ui.all_files_list.setSelectionMode(QAbstractItemView.MultiSelection)

        # Enable/disable selection helper widgets based on mode
        # For epoched data pre/post event features, allow selection helpers in dynamic mode too
        enable_selection_helpers = static_mode or (dynamic_mode and is_epoched)
        
        if hasattr(self.ui, "select_all_checkbox"):
            self.ui.select_all_checkbox.setEnabled(enable_selection_helpers)
        if hasattr(self.ui, "select_pattern_checkbox"):
            self.ui.select_pattern_checkbox.setEnabled(enable_selection_helpers)
        if hasattr(self.ui, "select_pattern_input"):
            self.ui.select_pattern_input.setEnabled(enable_selection_helpers)
            # If switching to non-epoched dynamic mode, clear pattern selection
            if dynamic_mode and not is_epoched:
                self.ui.select_pattern_input.clear()
                if hasattr(self.ui, "select_all_checkbox"):
                    self.ui.select_all_checkbox.setChecked(False)
                if hasattr(self.ui, "select_pattern_checkbox"):
                    self.ui.select_pattern_checkbox.setChecked(False)

        # Helper variables for currently selected items
        num_selected = len(self.ui.all_files_list.selectedItems())
        has_selection = num_selected > 0
        single_selection = num_selected == 1

        # --- Static buttons ---
        self.ui.plot_static_violin_plot_button.setEnabled(static_mode and has_selection)
        self.ui.plot_static_box_plot_button.setEnabled(static_mode and has_selection)

        # Heatmap (only for Transition Probability feature in static mode)
        if static_mode and self.get_selected_feature_code() == "TP":
            set_widgets_status(self.ui.plot_heatmap_button, mode="enable")
            set_widgets_status(self.ui.plot_heatmap_button, mode="show")
        else:
            set_widgets_status(self.ui.plot_heatmap_button, mode="disable")
            set_widgets_status(self.ui.plot_heatmap_button, mode="show")

        # --- Dynamic button ---
        dynamic_available = "sliding" in self.feature_mode
        # For epoched data, allow multiple selections for pre/post event features
        # For non-epoched data, require single selection for time-series
        if is_epoched:
            self.ui.plot_all_dynamic_button.setEnabled(
                dynamic_mode and has_selection and dynamic_available
            )
        else:
            self.ui.plot_all_dynamic_button.setEnabled(
                dynamic_mode and single_selection and dynamic_available
            )

        # Enable/disable ROF time-series plot button (always enabled when ROF is selected)
        current_feature = self.get_selected_feature_code()
        
        if hasattr(self.ui, "plot_rof_button"):
            if current_feature == "ROF":
                self.ui.plot_rof_button.setEnabled(True)
                self.ui.plot_rof_button.setVisible(True)
            else:
                self.ui.plot_rof_button.setEnabled(False)
                self.ui.plot_rof_button.setVisible(True)
        
        # Enable/disable RTF heatmap plot button (always enabled when RTF is selected)
        if hasattr(self.ui, "plot_rtf_button"):
            if current_feature == "RTF":
                self.ui.plot_rtf_button.setEnabled(True)
                self.ui.plot_rtf_button.setVisible(True)
            else:
                self.ui.plot_rtf_button.setEnabled(False)
                self.ui.plot_rtf_button.setVisible(True)

        # Update comparison widgets status based on list contents
        count_group_a = self.ui.group_a_files_list.count()
        count_group_b = self.ui.group_b_files_list.count()
        self.ui.add_group_a_button.setEnabled(len(self.ui.all_files_list.selectedItems()) != 0)
        self.ui.add_group_b_button.setEnabled(len(self.ui.all_files_list.selectedItems()) != 0)
        self.ui.remove_group_a_button.setEnabled(count_group_a != 0)
        self.ui.remove_group_b_button.setEnabled(count_group_b != 0)
        self.ui.reset_groups_button.setEnabled(count_group_a > 0 or count_group_b > 0)
        self.ui.plot_groups_button.setEnabled(count_group_a > 0 and count_group_b > 0)

        # Enable/disable other plot buttons based on ROF or RTF selection
        if self.get_selected_feature_code() in ["ROF", "RTF"]:
            # Disable unrelated plot buttons
            for btn_name in [
                "plot_static_violin_plot_button",
                "plot_static_box_plot_button",
                "plot_all_dynamic_button",
                "plot_heatmap_button",
            ]:
                if hasattr(self.ui, btn_name):
                    set_widgets_status(getattr(self.ui, btn_name), mode="disable")
                    set_widgets_status(getattr(self.ui, btn_name), mode="show")
        else:
            # Re-enable standard buttons visibility (status handled earlier)
            for btn_name in [
                "plot_static_violin_plot_button",
                "plot_static_box_plot_button",
                "plot_all_dynamic_button",
            ]:
                if hasattr(self.ui, btn_name):
                    # ensure visible
                    set_widgets_status(getattr(self.ui, btn_name), mode="show")

        # Automatically draw plot for newly selected feature
        self.auto_plot_selected_feature()

        # Refresh ROF/RTF plots if relevant
        self._maybe_refresh_rof_plot()
        self._maybe_refresh_rtf_plot()

        # Update selected file display
        self.update_selected_file_lineedit()

        # Ensure any pattern / select all logic stays in sync
        if (
            getattr(self.ui, "select_pattern_checkbox", None)
            and self.ui.select_pattern_checkbox.isChecked()
        ):
            self.apply_pattern_selection(run_controller=False)

    # ------------------------- Selection helper methods -------------------------
    def update_selected_file_lineedit(self):
        """Display selected file(s) name or count in the read-only line-edit.

        Returns:
          None
        """
        if not hasattr(self.ui, "selected_file_lineedit"):
            return
        selected_items = self.ui.all_files_list.selectedItems()
        if len(selected_items) == 1:
            self.ui.selected_file_lineedit.setText(selected_items[0].text())
        elif len(selected_items) > 1:
            self.ui.selected_file_lineedit.setText(f"{len(selected_items)} files selected")
        else:
            self.ui.selected_file_lineedit.clear()

    def handle_select_all_checkbox(self, state):
        """Select or deselect all files when the checkbox state changes.

        Args:
          state (int): Qt checkbox state.

        Returns:
          None
        """
        list_widget = self.ui.all_files_list
        if state == Qt.Checked:
            # Select all items
            list_widget.selectAll()
            # Prevent conflicts with pattern-selection
            if hasattr(self.ui, "select_pattern_checkbox"):
                self.ui.select_pattern_checkbox.setChecked(False)
        else:
            # Unchecked: clear all selections
            list_widget.clearSelection()
        self.update_selected_file_lineedit()
        # Re-run controller to update button states
        self.feature_visualization_controller()

    def apply_pattern_selection(self, *_, run_controller: bool = True):
        """Select files matching the pattern when the checkbox is active.

        Args:
          run_controller (bool): Whether to run the controller after selection.

        Returns:
          None
        """
        if not (
            hasattr(self.ui, "select_pattern_checkbox") and hasattr(self.ui, "select_pattern_input")
        ):
            return
        if not self.ui.select_pattern_checkbox.isChecked():
            return
        pattern = self.ui.select_pattern_input.text().strip().lower()
        if not pattern:
            return
        list_widget = self.ui.all_files_list

        # Temporarily block signals to prevent recursive triggers while selecting
        list_widget.blockSignals(True)
        try:
            list_widget.clearSelection()
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if pattern in item.text().lower():
                    item.setSelected(True)
        finally:
            list_widget.blockSignals(False)

        # Uncheck select all checkbox to avoid ambiguity
        if hasattr(self.ui, "select_all_checkbox"):
            # Block its signals briefly to avoid unintended slot calls
            self.ui.select_all_checkbox.blockSignals(True)
            self.ui.select_all_checkbox.setChecked(False)
            self.ui.select_all_checkbox.blockSignals(False)

        self.update_selected_file_lineedit()
        if run_controller:
            self.feature_visualization_controller()

    def auto_plot_selected_feature(self):
        """Automatically plot the currently selected feature.

        Returns:
          None
        """
        feature_code = self.get_selected_feature_code()
        
        # Validate that feature_code is not empty before attempting to plot
        if not feature_code or not feature_code.strip():
            return

        # Determine which visualization mode is active (needed for all branches)
        static_mode = hasattr(self.ui, "static_radio") and self.ui.static_radio.isChecked()
        dynamic_mode = hasattr(self.ui, "dynamic_radio") and self.ui.dynamic_radio.isChecked()

        if feature_code == "ROF":
            self.plot_rof_timeseries()
        elif feature_code == "RTF":
            self.plot_rtf_heatmap()
        elif feature_code == "TP":
            # If at least one file selected
            if self.ui.all_files_list.selectedItems():
                self.show_tp_heatmap()
        else:
            # Handle visualization based on mode and available features
            if static_mode:
                if self.ui.all_files_list.selectedItems():
                    # In static mode, prioritize averaged features, then pre_post
                    if "averaged" in self.feature_mode:
                        self.show_static_box_plot()
                    elif "pre_post" in self.feature_mode:
                        self._plot_pre_post_features()
            elif dynamic_mode:
                if self.ui.all_files_list.selectedItems():
                    # In dynamic mode, prioritize sliding time-series, then pre_post
                    if "sliding" in self.feature_mode:
                        self.show_dynamic_line_all()
                    elif "pre_post" in self.feature_mode:
                        # Show pre_post in dynamic mode as bar plot (comparison over time windows)
                        self._plot_pre_post_features()

    def _plot_pre_post_features(self):
        """Plot pre/post event features using bar plot visualization.
        
        Returns:
          None
        """
        # Load pre_post features
        try:
            features_df = self.load_features("pre_post")
        except Exception as e:
            # If loading fails, show error message
            self.figure.clear()
            ax = self.canvas.figure.gca()
            ax.text(0.5, 0.5, f'Failed to load pre/post features:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            self.canvas.draw()
            return
        
        # Filter by selected files
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        if selected_files:
            # Extract base filenames for filtering (handle _Pre and _Post suffixes)
            def extract_base_filename(filename):
                # Remove _Pre or _Post suffix
                if filename.endswith("_Pre") or filename.endswith("_Post"):
                    return filename.rsplit("_", 1)[0]
                return filename
            
            # Create base_filename column if it doesn't exist
            if "base_filename" not in features_df.columns:
                features_df["base_filename"] = features_df["Filename"].apply(extract_base_filename)
            
            # Filter by selected base filenames
            selected_base = [extract_base_filename(f) for f in selected_files]
            features_df = features_df[features_df["base_filename"].isin(selected_base)]
        
        if features_df.empty:
            self.figure.clear()
            ax = self.canvas.figure.gca()
            ax.text(0.5, 0.5, 'No pre/post features found for selected files', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            self.canvas.draw()
            return
        
        # Get plot parameters
        feature_code = self.get_selected_feature_code()
        font_sizes = self.get_selected_font_size_from_menu()
        colormap = self.get_selected_colormap()
        font_family = self.get_selected_font_family()
        display_options = self.get_display_options()
        
        # Plot pre/post event bar chart
        self.plot_pre_post_event_bar(
            features_df, feature_code, font_sizes, colormap, font_family, display_options
        )
    
    @staticmethod
    def move_file_between_lists(source_list, target_list):
        """Move currently selected items from source list to target list.

        Args:
          source_list: Source QListWidget.
          target_list: Destination QListWidget.

        Returns:
          None
        """
        selected_items = source_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            # Preserve text then remove from source and add to target
            text = item.text()
            source_list.takeItem(source_list.row(item))
            target_list.addItem(text)

    def move_file_from_all_to_a(self):
        """Move selected items from 'All' to Group A.

        Returns:
          None
        """
        self.move_file_between_lists(self.ui.all_files_list, self.ui.group_a_files_list)

    def move_file_from_all_to_b(self):
        """Move selected items from 'All' to Group B.

        Returns:
          None
        """
        self.move_file_between_lists(self.ui.all_files_list, self.ui.group_b_files_list)

    def move_file_from_a_to_all(self):
        """Move selected items from Group A back to 'All'.

        Returns:
          None
        """
        self.move_file_between_lists(self.ui.group_a_files_list, self.ui.all_files_list)

    def move_file_from_b_to_all(self):
        """Move selected items from Group B back to 'All'.

        Returns:
          None
        """
        self.move_file_between_lists(self.ui.group_b_files_list, self.ui.all_files_list)

    def reset_groups(self):
        """Clear all file lists and reset the plot.

        Returns:
          None
        """
        self.clear_all_lists()
        self.figure.clear()
        self.canvas.draw()
        # Clear figure title
        self.figure.suptitle("")

    def get_plot_parameters(self):
        """Get font sizes, colormap, font family and display options.

        Returns:
          tuple[dict[str, int], str, str, dict[str, bool]]: (font_sizes, colormap,
          font_family, display_options).
        """
        # Get plot parameters from UI components
        # Check if font sizes should come from menu
        menu_font_sizes = self.get_selected_font_size_from_menu()

        # Use menu font sizes if available, otherwise fall back to UI inputs
        title_size = menu_font_sizes.get("title", 18)
        label_size = menu_font_sizes.get("label", 14)
        tick_size = menu_font_sizes.get("tick", 12)
        legend_size = menu_font_sizes.get("legend", 12)

        # Fallback to UI input fields if menu not available
        if not hasattr(self, "font_size_action_groups"):
            try:
                title_size = (
                    int(self.ui.font_size_input.text())
                    if hasattr(self.ui, "font_size_input")
                    else 18
                )
                tick_size = (
                    int(self.ui.label_size_input.text())
                    if hasattr(self.ui, "label_size_input")
                    else 12
                )
                label_size = title_size - 4
                legend_size = tick_size
            except (ValueError, AttributeError):
                pass

        # Use colormap and font family from menu actions
        colormap = self.get_selected_colormap()
        font_family = self.get_selected_font_family()

        # Get display options
        display_options = self.get_display_options()

        # Return font sizes as a dictionary for clarity
        font_sizes = {
            "title": title_size,
            "label": label_size,
            "tick": tick_size,
            "legend": legend_size,
        }

        return font_sizes, colormap, font_family, display_options

    def show_static_violin_all(self):
        """Display violin plots of all static features for selected files.

        Returns:
          None
        """
        selected_feature = self.get_selected_feature_code()
        # Get the selected files from the all_files_list
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        
        # Check if this is a variability feature
        is_variability = selected_feature.endswith("_SD") or selected_feature.endswith("_RMSSD")
        
        # Load appropriate features
        if is_variability:
            all_features_df = self.load_features("variability")
        else:
            all_features_df = self.load_features("averaged")
        
        # Filter features_df based on selected files
        features_df = all_features_df[all_features_df["Filename"].isin(selected_files)]
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_violin(
            features_df, selected_feature, font_sizes, colormap, font_family, display_options
        )
        self._highlight_button("plot_static_violin_plot_button")

    def show_static_box_plot(self):
        """Display box plots of all static features for selected files.

        Returns:
          None
        """
        selected_feature = self.get_selected_feature_code()
        # Get the selected files from the all_files_list
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        
        # Check if this is a variability feature
        is_variability = selected_feature.endswith("_SD") or selected_feature.endswith("_RMSSD")
        
        # Load appropriate features
        if is_variability:
            all_features_df = self.load_features("variability")
        else:
            all_features_df = self.load_features("averaged")
        
        # Filter features_df based on selected files
        features_df = all_features_df[all_features_df["Filename"].isin(selected_files)]
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_box(
            features_df, selected_feature, font_sizes, colormap, font_family, display_options
        )
        self._highlight_button("plot_static_box_plot_button")

    def show_dynamic_line_all(self):
        """Display line plots of all dynamic features for the selected file.

        Returns:
          None
        """
        selected_feature = self.get_selected_feature_code()
        
        # Check if this is epoched data with pre/post event features
        is_epoched = hasattr(self.comet, "datatype") and self.comet.datatype == "epoched"
        
        if is_epoched:
            # For epoched data, show pre/post event bar plot
            selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
            if not selected_files:
                return
            features_df = self.load_features("sliding")
            
            # Filter by selected files - need to match base filename (without _trial#_Pre/Post suffix)
            # Extract base filename by removing trial-specific suffixes
            def extract_base_filename(full_filename):
                # Remove _trial#_Pre or _trial#_Post suffix (also handle old preTMS/postTMS format)
                import re
                base = re.sub(r'_trial\d+_(Pre|Post|preTMS|postTMS)$', '', full_filename)
                return base
            
            features_df['base_filename'] = features_df['Filename'].apply(extract_base_filename)
            features_df = features_df[features_df['base_filename'].isin(selected_files)]
            
            font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
            self.plot_pre_post_event_bar(
                features_df, selected_feature, font_sizes, colormap, font_family, display_options
            )
        else:
            # For non-epoched data, show standard time-series line plot
            selected_file = self.ui.all_files_list.currentItem().text()
            features_df = self.load_features("sliding").query(f'Filename == "{selected_file}"')
            font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
            self.plot_line(
                features_df, selected_feature, font_sizes, colormap, font_family, display_options
            )
        
        self._highlight_button("plot_all_dynamic_button")

    def show_tp_heatmap(self):
        """Display the heatmap for transition probabilities.

        Returns:
          None
        """
        # Get the selected files from the all_files_list
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        # Load all features
        all_features_df = self.load_features("averaged")
        # Filter features_df based on selected files
        features_df = all_features_df[all_features_df["Filename"].isin(selected_files)]
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_heatmap(features_df, font_sizes, font_family, display_options)
        self._highlight_button("plot_heatmap_button")

        # Set figure title
        self.figure.suptitle(
            "Transition Probability Heatmap", fontsize=font_sizes["title"], fontfamily=font_family
        )

    def compare_groups(self):
        """Compare selected features between Group A and Group B.

        Returns:
          None
        """
        selected_feature = self.get_selected_feature_code()
        group_a_name, group_b_name = self.get_group_names()
        
        # Check if this is a variability feature
        is_variability = selected_feature.endswith("_SD") or selected_feature.endswith("_RMSSD")
        
        # Load appropriate features
        if is_variability:
            features_df = self.load_features("variability")
        else:
            features_df = self.load_features("averaged")
        
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_group_comparison(
            features_df,
            selected_feature,
            group_a_name,
            group_b_name,
            font_sizes,
            colormap,
            font_family,
            display_options,
        )

        # Also compute paired t-tests per microstate and display in the text area
        self._compute_and_display_paired_ttests(features_df, selected_feature, group_a_name, group_b_name)

    def load_features(self, mode):
        """Load features for a given mode.

        Args:
          mode (str): Feature mode (e.g., "averaged", "sliding", "variability", "pre_post").

        Returns:
          pandas.DataFrame: Loaded features for the given mode.
        """
        if mode == "variability":
            # Load variability features separately
            feature_path = os.path.join(
                self.extracted_features_path, f"real_sliding_variability_features{self.export_format}"
            )
        elif mode == "pre_post":
            # Load pre/post event features
            feature_path = os.path.join(
                self.extracted_features_path, f"real_pre_post_features{self.export_format}"
            )
        else:
            feature_path = os.path.join(
                self.extracted_features_path, f"real_{mode}_features{self.export_format}"
            )
        return FeatureIO().import_features(feature_path, self.export_format)

    def clear_all_lists(self):
        """Clear all file lists.

        Returns:
          None
        """
        self.ui.all_files_list.clear()
        self.ui.group_a_files_list.clear()
        self.ui.group_b_files_list.clear()
        self.populate_all_files_list()

    def populate_all_files_list(self):
        """Populate the all_files_list with EEG file names.

        Returns:
          None
        """
        for eeg_file in self.list_eegs:
            self.ui.all_files_list.addItem(str(eeg_file))

    def set_labels_ticks(self, filter_cols, feature, ax, font_sizes, font_family, display_options):
        """Set x/y labels, tick labels and apply display options.

        Args:
          filter_cols (list[str]): Feature columns plotted on the x-axis.
          feature (str): Feature short code.
          ax (matplotlib.axes.Axes): Target axes.
          font_sizes (dict[str, int]): Title/label/tick/legend sizes.
          font_family (str): Font family to apply.
          display_options (dict[str, bool]): Visibility options for axes/grid.

        Returns:
          None
        """

        # Helper to convert numerical state indices to letters (1->A, 2->B, ...)
        def _state_to_letter(state_str: str) -> str:
            try:
                state_idx = int(state_str)
                # Ensure 1-based mapping; fall back gracefully
                letter_idx = max(state_idx - 1, 0)
                return chr(ord("A") + letter_idx)
            except (ValueError, TypeError):
                # If conversion fails, just return the original string
                return state_str

        # Determine appropriate x-tick labels
        if feature in ["TP", "RTF"]:
            # Expect column names like "TP_1_2" → "A-B" or "RTF_A_B"
            xticklabels = []
            for col in filter_cols:
                parts = col.split("_")
                if len(parts) >= 3:
                    from_letter = _state_to_letter(parts[-2])
                    to_letter = _state_to_letter(parts[-1])
                    xticklabels.append(f"{from_letter}-{to_letter}")
                else:
                    # Fallback to original logic if format unexpected
                    xticklabels.append(parts[-1])
            ax.set_xlabel("Transition", fontsize=font_sizes["label"], fontfamily=font_family)
        else:
            # Default behaviour for other features
            xticklabels = [col.split("_")[-1] for col in filter_cols]
            ax.set_xlabel("Microstate", fontsize=font_sizes["label"], fontfamily=font_family)

        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        
        # Get y-axis label from feature dictionary, with fallback for empty or missing features
        ylabel = self.comet.feature_list_dictionary.get(feature, feature) if feature else "Feature"
        ax.set_ylabel(
            ylabel,
            fontsize=font_sizes["label"],
            fontfamily=font_family,
        )
        ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])

        # Set font family for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes["tick"])

        # Apply display options
        if not display_options.get("show_axes", True):
            ax.axis("off")

        if display_options.get("show_grid", False):
            ax.grid(True, alpha=0.3)

        # Set figure title instead of plot_label
        self.figure.suptitle(
            f"{self.comet.feature_list_dictionary[feature]}",
            fontsize=font_sizes["title"],
            fontfamily=font_family,
        )

    def set_labels_ticks_sliding(
        self, filter_cols, feature, ax, font_sizes, font_family, display_options
    ):
        """Set labels/ticks and options for sliding window feature plots.

        Args:
          filter_cols (list[str]): Feature columns plotted.
          feature (str): Feature short code.
          ax (matplotlib.axes.Axes): Target axes.
          font_sizes (dict[str, int]): Title/label/tick/legend sizes.
          font_family (str): Font family to apply.
          display_options (dict[str, bool]): Visibility options.

        Returns:
          None
        """

        # Helper to convert numerical state indices to letters (1->A, 2->B, ...)
        def _state_to_letter(state_str: str) -> str:
            try:
                state_idx = int(state_str)
                # Ensure 1-based mapping; fall back gracefully
                letter_idx = max(state_idx - 1, 0)
                return chr(ord("A") + letter_idx)
            except (ValueError, TypeError):
                # If conversion fails, just return the original string
                return state_str

        # For sliding window features, x-axis represents time in seconds
        ax.set_xlabel("Time (seconds)", fontsize=font_sizes["label"], fontfamily=font_family)

        # Set y-axis label based on feature type
        if feature in ["TP", "RTF"]:
            ax.set_ylabel(
                "Transition Probability", fontsize=font_sizes["label"], fontfamily=font_family
            )
        else:
            ax.set_ylabel(
                self.comet.feature_list_dictionary[feature],
                fontsize=font_sizes["label"],
                fontfamily=font_family,
            )

        ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])

        # Set font family for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes["tick"])

        # Apply display options
        if not display_options.get("show_axes", True):
            ax.axis("off")

        if display_options.get("show_grid", False):
            ax.grid(True, alpha=0.3)

        # Set figure title
        self.figure.suptitle(
            f"{self.comet.feature_list_dictionary[feature]} (Sliding Window)",
            fontsize=font_sizes["title"],
            fontfamily=font_family,
        )

    def plot_violin(self, features_df, feature, font_sizes, colormap, font_family, display_options):
        """Plot a violin plot for the selected static feature.

        Args:
          features_df (pandas.DataFrame): Features to plot.
          feature (str): Feature short code.
          font_sizes (dict[str, int]): Title/label/tick/legend sizes.
          colormap (str): Matplotlib colormap name.
          font_family (str): Font family to apply.
          display_options (dict[str, bool]): Visibility options.

        Returns:
          None
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        # Fallback for features computed for the whole sequence (single column without suffix)
        if not filter_cols and feature in features_df.columns:
            filter_cols = [feature]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=["Filename"], value_vars=filter_cols)
        plot_data.columns = ["Filename", "Feature", feature]

        self.clear_and_set_fonts(ax, font_family, font_sizes)

        # Use a color palette for the violins based on the number of features
        num_features = len(filter_cols)
        color_palette = sns.color_palette(colormap, num_features)

        sns.violinplot(x="Feature", y=feature, data=plot_data, ax=ax, palette=color_palette)
        sns.swarmplot(
            x="Feature", y=feature, data=plot_data, ax=ax, color="white", size=10, marker="o"
        )
        self.set_labels_ticks(filter_cols, feature, ax, font_sizes, font_family, display_options)

        # Handle legend display
        legend = ax.get_legend()
        if legend:
            if display_options.get("show_legend", True):
                legend.set_visible(True)
                # Update legend font
                for text in legend.get_texts():
                    text.set_fontsize(font_sizes["legend"])
                    text.set_fontfamily(font_family)
            else:
                legend.remove()

        self.canvas.draw()

    def plot_line(self, features_df, feature, font_sizes, colormap, font_family, display_options):
        """Plot a line plot for the selected dynamic feature.

        Args:
          features_df (pandas.DataFrame): Features to plot.
          feature (str): Feature short code.
          font_sizes (dict[str, int]): Title/label/tick/legend sizes.
          colormap (str): Matplotlib colormap name.
          font_family (str): Font family to apply.
          display_options (dict[str, bool]): Visibility options.

        Returns:
          None
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        filter_cols.sort()
        
        # Check if there are any features to plot
        if not filter_cols:
            ax.text(0.5, 0.5, f'No features found starting with "{feature}"', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            self.canvas.draw()
            return

        # Convert window indices to time in seconds for sliding window features
        if "Window_index" in features_df.columns:
            # Check if this is event-based sliding (has Event_name column)
            is_event_based = "Event_name" in features_df.columns
            
            # Get sampling rate and window size from COMET object
            sampling_rate = getattr(self.comet, "sampling_rate", 250)  # Default to 250 Hz
            sliding_window_size = getattr(
                self.comet, "sliding_window_size", 1
            )  # Default to 1 second

            self.clear_and_set_fonts(ax, font_family, font_sizes)

            # Plot horizontal lines for each window instead of connected points
            # Get unique features for coloring
            unique_features = [col for col in filter_cols]
            colors = sns.color_palette(n_colors=len(unique_features))

            if is_event_based and len(unique_features) > 0:
                # For event-based sliding, show time-based plot with event markers
                # Get event timing information if available
                has_timing = all(col in features_df.columns for col in ['window_start_idx', 'window_end_idx', 'window_duration_ms'])
                
                if has_timing:
                    # Calculate time positions from timing information
                    # Convert from sample indices to seconds
                    window_starts = features_df['window_start_idx'].values / sampling_rate
                    window_ends = features_df['window_end_idx'].values / sampling_rate
                    event_names = features_df['Event_name'].values
                    
                    # Plot features as horizontal lines for each event window
                    for i, col in enumerate(unique_features):
                        feature_values = features_df[col].values
                        
                        for window_idx, (start_time, end_time, value, event_name) in enumerate(
                            zip(window_starts, window_ends, feature_values, event_names)
                        ):
                            if not pd.isna(value):
                                # Draw horizontal line for this event window
                                ax.plot(
                                    [start_time, end_time],
                                    [value, value],
                                    color=colors[i],
                                    linewidth=2,
                                    label=col if window_idx == 0 else "",
                                )
                                
                                # Connect to previous window if exists
                                if window_idx > 0 and not pd.isna(feature_values[window_idx - 1]):
                                    prev_value = feature_values[window_idx - 1]
                                    ax.plot(
                                        [window_starts[window_idx], window_starts[window_idx]],
                                        [prev_value, value],
                                        color=colors[i],
                                        linewidth=1,
                                        alpha=0.7,
                                    )
                    
                    # Add vertical lines at event boundaries with labels
                    unique_event_times = []
                    unique_event_names = []
                    for start_time, event_name in zip(window_starts, event_names):
                        if start_time not in unique_event_times:
                            unique_event_times.append(start_time)
                            unique_event_names.append(event_name)
                    
                    # Update plot to ensure y-axis includes space for labels
                    ax.relim()
                    ax.autoscale_view()
                    
                    # Get y-axis limits after plotting
                    y_min, y_max = ax.get_ylim()
                    y_range = y_max - y_min
                    
                    # Add vertical lines and labels for events
                    for event_time, event_name in zip(unique_event_times, unique_event_names):
                        ax.axvline(x=event_time, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                        # Add event name as text annotation
                        # Position text slightly above the plot area
                        ax.text(event_time, y_max + y_range * 0.02, event_name, 
                               rotation=90, va='bottom', ha='right', 
                               fontsize=font_sizes.get("tick", 10) - 2, 
                               color='gray', alpha=0.8)
                    
                    # Extend y-axis slightly to accommodate event labels
                    ax.set_ylim(y_min, y_max + y_range * 0.15)
                    
                    ax.set_xlabel("Time (seconds)", fontsize=font_sizes["label"], fontfamily=font_family)
                else:
                    # Fallback to window index based plotting if timing info not available
                    # This is similar to regular sliding but shows event boundaries
                    for i, col in enumerate(unique_features):
                        feature_values = features_df[col].values
                        event_names = features_df['Event_name'].values
                        
                        for window_idx, (value, event_name) in enumerate(zip(feature_values, event_names)):
                            if not pd.isna(value):
                                # Use window index as proxy for time
                                window_start = window_idx
                                window_end = window_idx + 1
                                
                                ax.plot(
                                    [window_start, window_end],
                                    [value, value],
                                    color=colors[i],
                                    linewidth=2,
                                    label=col if window_idx == 0 else "",
                                )
                                
                                if window_idx > 0 and not pd.isna(feature_values[window_idx - 1]):
                                    prev_value = feature_values[window_idx - 1]
                                    ax.plot(
                                        [window_start, window_start],
                                        [prev_value, value],
                                        color=colors[i],
                                        linewidth=1,
                                        alpha=0.7,
                                    )
                    
                    ax.set_xlabel("Event Window Index", fontsize=font_sizes["label"], fontfamily=font_family)
                
            else:
                # Original time-based sliding window visualization
                for i, col in enumerate(unique_features):
                    # Get the feature values for this column
                    feature_values = features_df[col].values

                    # Create horizontal lines for each window
                    for window_idx, value in enumerate(feature_values):
                        if not pd.isna(value):  # Skip NaN values
                            # Calculate window start and end times
                            window_start = window_idx * sliding_window_size
                            window_end = (window_idx + 1) * sliding_window_size

                            # Draw horizontal line for this window
                            ax.plot(
                                [window_start, window_end],
                                [value, value],
                                color=colors[i],
                                linewidth=2,
                                label=col if window_idx == 0 else "",
                            )

                            # Add vertical connectors between windows (optional, for continuity)
                            if window_idx > 0 and not pd.isna(feature_values[window_idx - 1]):
                                prev_value = feature_values[window_idx - 1]
                                ax.plot(
                                    [window_start, window_start],
                                    [prev_value, value],
                                    color=colors[i],
                                    linewidth=1,
                                    alpha=0.7,
                                )

                # Set x-axis label
                ax.set_xlabel("Time (seconds)", fontsize=font_sizes["label"], fontfamily=font_family)
            self.set_labels_ticks_sliding(
                filter_cols, feature, ax, font_sizes, font_family, display_options
            )

            # Create legend manually for the sliding window plot
            if display_options.get("show_legend", True):
                # Get handles and labels, removing duplicates
                handles, labels = ax.get_legend_handles_labels()
                # Remove empty labels from the vertical connectors
                filtered_handles = []
                filtered_labels = []
                for h, label in zip(handles, labels):
                    if label and label not in filtered_labels:  # Only add non-empty, unique labels
                        filtered_handles.append(h)
                        filtered_labels.append(label)

                if filtered_handles:
                    legend = ax.legend(filtered_handles, filtered_labels)
                    for text in legend.get_texts():
                        text.set_fontsize(font_sizes["legend"])
                        text.set_fontfamily(font_family)
        else:
            # Fallback to original behavior if no Window_index column
            plot_data = pd.melt(
                features_df.reset_index(), id_vars=["Window_index"], value_vars=filter_cols
            )
            plot_data.columns = ["Window_index", "Feature", feature]

            self.clear_and_set_fonts(ax, font_family, font_sizes)
            sns.lineplot(x="Window_index", y=feature, hue="Feature", data=plot_data, ax=ax)
            self.set_labels_ticks(
                filter_cols, feature, ax, font_sizes, font_family, display_options
            )

            # Handle legend display for non-sliding plots
            legend = ax.get_legend()
            if legend:
                if display_options.get("show_legend", True):
                    legend.set_visible(True)
                    # Update legend font
                    for text in legend.get_texts():
                        text.set_fontsize(font_sizes["legend"])
                        text.set_fontfamily(font_family)
                else:
                    legend.remove()

        self.canvas.draw()

    def plot_heatmap(self, features_df, font_sizes, font_family, display_options):
        """Plot a heatmap for transition probabilities.

        Args:
          features_df (pandas.DataFrame): Features to plot.
          font_sizes (dict[str, int]): Title/label/tick/legend sizes.
          font_family (str): Font family to apply.
          display_options (dict[str, bool]): Visibility options.

        Returns:
          None
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        self.clear_and_set_fonts(ax, font_family, font_sizes)

        # Extract unique states from column names
        states = sorted(
            set(col.split("_")[1] for col in features_df.columns if col.startswith("TP"))
        )

        # Initialize transition matrix
        transition_matrix = np.zeros((len(states), len(states)))

        # Fill transition matrix
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                if f"TP_{from_state}_{to_state}" in features_df.columns:
                    transition_matrix[i, j] = features_df[f"TP_{from_state}_{to_state}"].mean()

        # Plot heatmap and capture the image object for the colorbar
        im = ax.matshow(transition_matrix, cmap="YlGnBu")

        # Add a colorbar to illustrate the colormap scale
        cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=font_sizes["tick"])
        for label in cbar.ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes["tick"])

        # Annotate values with percentage
        for i in range(len(states)):
            for j in range(len(states)):
                if i != j:
                    value = transition_matrix[i, j]
                    ax.text(
                        j,
                        i,
                        f"{100 * value:.2f}%",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=font_sizes["label"],
                        fontfamily=font_family,
                    )

        # Set ticks and labels
        ax.set_xticks(np.arange(len(states)))
        ax.set_yticks(np.arange(len(states)))
        ax.set_xticklabels(states)
        ax.set_yticklabels(states)
        ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])

        # Set font family and size for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes["tick"])

        # Set labels and title
        ax.set_xlabel("To", fontsize=font_sizes["label"], fontfamily=font_family)
        ax.set_ylabel("From", fontsize=font_sizes["label"], fontfamily=font_family)

        # Apply display options
        if not display_options.get("show_axes", True):
            ax.axis("off")

        if display_options.get("show_grid", False):
            ax.grid(True, alpha=0.3)

        # Show the plot
        self.canvas.draw()

    def get_group_names(self):
        """Return the names of Group A and Group B.

        Returns:
          tuple[str, str]: (group_a_name, group_b_name).
        """
        return self.ui.group_a_lineedit.text(), self.ui.group_b_lineedit.text()

    @staticmethod
    def clear_and_set_fonts(ax, font_family, font_sizes):
        """Clear the plot and set the font sizes and family.

        Args:
          ax (matplotlib.axes.Axes): Target axes.
          font_family (str): Font family to apply.
          font_sizes (dict[str, int]): Title/label/tick sizes.

        Returns:
          None
        """
        ax.clear()
        # Set default font properties for all text elements
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(font_sizes["title"])
            item.set_fontfamily(font_family)

        # Set font for tick labels
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(font_sizes["tick"])
            item.set_fontfamily(font_family)

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def plot_group_comparison(
        self,
        features_df,
        selected_feature,
        group_a_name,
        group_b_name,
        font_sizes,
        colormap,
        font_family,
        display_options,
    ):
        """Plot a comparison of features between two groups.

        Args:
          features_df (pandas.DataFrame): Features to plot.
          selected_feature (str): Feature short code.
          group_a_name (str): Name of Group A.
          group_b_name (str): Name of Group B.
          font_sizes (dict[str, int]): Title/label/tick/legend sizes.
          colormap (str): Matplotlib colormap name.
          font_family (str): Font family to apply.
          display_options (dict[str, bool]): Visibility options.

        Returns:
          None
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(selected_feature)]
        # Fallback for features computed for the whole sequence
        if not filter_cols and selected_feature in features_df.columns:
            filter_cols = [selected_feature]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=["Filename"], value_vars=filter_cols)
        plot_data.columns = ["Filename", "Feature", selected_feature]

        listItems_group_a = [
            self.ui.group_a_files_list.item(x).text()
            for x in range(self.ui.group_a_files_list.count())
        ]
        listItems_group_b = [
            self.ui.group_b_files_list.item(x).text()
            for x in range(self.ui.group_b_files_list.count())
        ]

        group_a_data = plot_data.query(f"Filename in {listItems_group_a}")
        group_a_data["Group"] = group_a_name

        group_b_data = plot_data.query(f"Filename in {listItems_group_b}")
        group_b_data["Group"] = group_b_name

        comparison_data = pd.concat([group_a_data, group_b_data], axis=0).reset_index()

        self.clear_and_set_fonts(ax, font_family, font_sizes)
        color_palette = sns.color_palette(colormap, 2)
        sns.boxplot(
            x="Feature",
            y=selected_feature,
            hue="Group",
            data=comparison_data,
            ax=ax,
            palette=color_palette,
        )
        sns.swarmplot(
            x="Feature",
            y=selected_feature,
            hue="Group",
            data=comparison_data,
            ax=ax,
            color="black",
            size=10,
            marker="o",
            dodge=True,
            legend=False,
        )
        self.set_labels_ticks(
            filter_cols, selected_feature, ax, font_sizes, font_family, display_options
        )

        # Handle legend display
        legend = ax.get_legend()
        if legend:
            if display_options.get("show_legend", True):
                legend.set_visible(True)
                # Update legend font
                for text in legend.get_texts():
                    text.set_fontsize(font_sizes["legend"])
                    text.set_fontfamily(font_family)
            else:
                legend.remove()

        self.canvas.draw()

    def _format_feature_label_from_column(self, column_name: str, feature_code: str) -> str:
        """Return a short display label for a feature column.

        For microstate-indexed features like "COV_1" returns "A"; for
        transition-like features such as "TP_1_2" returns "A-B".

        Args:
          column_name: Full column name (e.g., "COV_1", "TP_1_2").
          feature_code: Selected feature short code (e.g., "COV", "TP").

        Returns:
          Readable label string for display.
        """
        def _state_to_letter(idx_str: str) -> str:
            try:
                state_idx = int(idx_str)
                letter_idx = max(state_idx - 1, 0)
                return chr(ord("A") + letter_idx)
            except Exception:
                return idx_str

        parts = column_name.split("_")
        if feature_code in ["TP", "RTF"] and len(parts) >= 3:
            return f"{_state_to_letter(parts[-2])}-{_state_to_letter(parts[-1])}"
        # If feature is whole-sequence (no suffix), return feature code as label
        if feature_code == column_name:
            return feature_code
        if len(parts) >= 2:
            return _state_to_letter(parts[-1])
        return column_name

    def _compute_and_display_paired_ttests(
        self,
        features_df: pd.DataFrame,
        selected_feature: str,
        group_a_name: str,
        group_b_name: str,
    ) -> None:
        """Compute paired t-tests per microstate and write results to the text area.

        Pairing strategy: pairs are formed by list order between Group A and Group B
        file lists. If lengths differ, only the first min(len(A), len(B)) items are used.
        Rows missing in the features table are skipped. NaNs are dropped pairwise.
        Bonferroni correction is applied across the microstate comparisons.
        """
        if not hasattr(self.ui, "compare_groups_textedit"):
            return

        # Collect file lists in the order displayed in the UI
        group_a_files = [self.ui.group_a_files_list.item(i).text() for i in range(self.ui.group_a_files_list.count())]
        group_b_files = [self.ui.group_b_files_list.item(i).text() for i in range(self.ui.group_b_files_list.count())]

        num_pairs_planned = min(len(group_a_files), len(group_b_files))

        # Identify microstate columns for the selected feature
        feature_columns = [col for col in features_df.columns if col.startswith(selected_feature + "_")]
        # Fallback for whole-sequence features that have a single column equal to the code
        if not feature_columns and selected_feature in features_df.columns:
            feature_columns = [selected_feature]
        feature_columns.sort()

        # Prepare containers
        raw_p_values: list[float] = []
        test_results = []  # list of dicts per column

        # Pre-index features by filename for quick lookup
        features_by_file = features_df.set_index("Filename")

        for feature_column in feature_columns:
            values_a = []
            values_b = []

            # Build paired values from list order
            for idx in range(num_pairs_planned):
                file_a = group_a_files[idx]
                file_b = group_b_files[idx]

                # Skip if any filename is not present in the dataframe
                if file_a not in features_by_file.index or file_b not in features_by_file.index:
                    continue

                val_a = features_by_file.at[file_a, feature_column]
                val_b = features_by_file.at[file_b, feature_column]

                # Drop pairs with NaNs
                if pd.isna(val_a) or pd.isna(val_b):
                    continue

                values_a.append(float(val_a))
                values_b.append(float(val_b))

            # Perform paired test if we have at least 2 pairs
            if len(values_a) >= 2:
                t_stat, p_val = ttest_rel(values_a, values_b)
                raw_p_values.append(float(p_val))
                test_results.append(
                    {
                        "column": feature_column,
                        "label": self._format_feature_label_from_column(feature_column, selected_feature),
                        "t": float(t_stat),
                        "p": float(p_val),
                        "n_pairs": len(values_a),
                        "mean_a": float(np.mean(values_a)) if values_a else None,
                        "mean_b": float(np.mean(values_b)) if values_b else None,
                    }
                )
            else:
                test_results.append(
                    {
                        "column": feature_column,
                        "label": self._format_feature_label_from_column(feature_column, selected_feature),
                        "t": None,
                        "p": None,
                        "n_pairs": len(values_a),
                        "mean_a": float(np.mean(values_a)) if values_a else None,
                        "mean_b": float(np.mean(values_b)) if values_b else None,
                    }
                )

        # Apply Bonferroni correction only if there are multiple tests
        corrected_map = {}
        multiple_testing = len(raw_p_values) > 1
        if multiple_testing and raw_p_values:
            _, pvals_corr, _, _ = multipletests(raw_p_values, method="bonferroni")
            # Map back in order to test_results entries with non-None p
            corr_iter = iter(pvals_corr.tolist())
            for res in test_results:
                if res["p"] is not None:
                    corrected_map[res["column"]] = next(corr_iter)

        # Build output text (well-formatted report)
        lines = []
        title = "Paired t-test Report"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append(
            f"Feature: {self.comet.feature_list_dictionary.get(selected_feature, selected_feature)} ({selected_feature})"
        )
        lines.append(f"Groups: {group_a_name} vs {group_b_name}")
        lines.append(f"Planned pairs: {num_pairs_planned}")
        if multiple_testing:
            lines.append("Multiple testing: Bonferroni correction")
        lines.append("")
        lines.append("Results by microstate")
        lines.append("---------------------")

        # Table header
        if multiple_testing:
            header = (
                f"{'Microstate':<12} {'n':>3}  {'mean('+group_a_name+')':>12}  "
                f"{'mean('+group_b_name+')':>12}  {'Δ(A-B)':>9}  {'t':>7}  {'p':>9}  {'p_bonf':>9}  {'sig':>3}"
            )
        else:
            header = (
                f"{'Microstate':<12} {'n':>3}  {'mean('+group_a_name+')':>12}  "
                f"{'mean('+group_b_name+')':>12}  {'Δ(A-B)':>9}  {'t':>7}  {'p':>9}  {'sig':>3}"
            )
        lines.append(header)
        lines.append("-" * len(header))

        insufficient = []
        alpha = 0.05
        significant = []
        for res in test_results:
            label = res["label"]
            n_pairs = res["n_pairs"]
            mean_a = res.get("mean_a")
            mean_b = res.get("mean_b")
            if res["p"] is None:
                insufficient.append((label, n_pairs))
                continue
            p_corr = corrected_map.get(res["column"], res["p"]) if multiple_testing else res["p"]
            t_stat = res["t"]
            diff = None
            sig_mark = "*" if p_corr < alpha else ""
            if mean_a is not None and mean_b is not None:
                diff = mean_a - mean_b
            if multiple_testing:
                lines.append(
                    f"{label:<12} {n_pairs:>3}  "
                    f"{(mean_a if mean_a is not None else float('nan')):>12.3f}  "
                    f"{(mean_b if mean_b is not None else float('nan')):>12.3f}  "
                    f"{(diff if diff is not None else float('nan')):>9.3f}  "
                    f"{t_stat:>7.3f}  {res['p']:>9.2g}  {p_corr:>9.2g}  {sig_mark:>3}"
                )
            else:
                lines.append(
                    f"{label:<12} {n_pairs:>3}  "
                    f"{(mean_a if mean_a is not None else float('nan')):>12.3f}  "
                    f"{(mean_b if mean_b is not None else float('nan')):>12.3f}  "
                    f"{(diff if diff is not None else float('nan')):>9.3f}  "
                    f"{t_stat:>7.3f}  {p_corr:>9.2g}  {sig_mark:>3}"
                )

            if p_corr < alpha and mean_a is not None and mean_b is not None:
                direction = (
                    f"{group_a_name} > {group_b_name}" if mean_a > mean_b else f"{group_b_name} > {group_a_name}"
                )
                significant.append(
                    {
                        "label": label,
                        "p_corr": p_corr,
                        "direction": direction,
                        "diff": (mean_a - mean_b),
                    }
                )

        lines.append("")
        if multiple_testing:
            lines.append(f"Summary (significant after Bonferroni, alpha={alpha})")
            lines.append("-----------------------------------------------")
        else:
            lines.append(f"Summary (alpha={alpha})")
            lines.append("----------------------")
        if not significant:
            lines.append("  No significant differences.")
        else:
            # Sort by corrected p-value ascending
            significant.sort(key=lambda x: x["p_corr"]) 
            for s in significant:
                if multiple_testing:
                    lines.append(
                        f"  {s['label']}: {s['direction']} (p_bonf={s['p_corr']:.4g}, Δ={s['diff']:.3g})"
                    )
                else:
                    lines.append(
                        f"  {s['label']}: {s['direction']} (p={s['p_corr']:.4g}, Δ={s['diff']:.3g})"
                    )

        if insufficient:
            lines.append("")
            lines.append("Insufficient data")
            lines.append("-----------------")
            for label, n_pairs in insufficient:
                lines.append(f"  {label}: insufficient pairs (n={n_pairs})")

        self.ui.compare_groups_textedit.setPlainText("\n".join(lines))

    def plot_box(self, features_df, feature, font_sizes, colormap, font_family, display_options):
        """Plot a box plot for the selected static feature.

        Args:
          features_df (pandas.DataFrame): Features to plot.
          feature (str): Feature short code.
          font_sizes (dict[str, int]): Title/label/tick/legend sizes.
          colormap (str): Matplotlib colormap name.
          font_family (str): Font family to apply.
          display_options (dict[str, bool]): Visibility options.

        Returns:
          None
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        # Fallback for features computed for the whole sequence
        if not filter_cols and feature in features_df.columns:
            filter_cols = [feature]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=["Filename"], value_vars=filter_cols)
        plot_data.columns = ["Filename", "Feature", feature]

        self.clear_and_set_fonts(ax, font_family, font_sizes)

        # Use a color palette for the box plots based on the number of features
        num_features = len(filter_cols)
        color_palette = sns.color_palette(colormap, num_features)

        sns.boxplot(x="Feature", y=feature, data=plot_data, ax=ax, palette=color_palette)
        sns.swarmplot(
            x="Feature", y=feature, data=plot_data, ax=ax, color="black", size=10, marker="o"
        )
        self.set_labels_ticks(filter_cols, feature, ax, font_sizes, font_family, display_options)

        # Handle legend display
        legend = ax.get_legend()
        if legend:
            if display_options.get("show_legend", True):
                legend.set_visible(True)
                # Update legend font
                for text in legend.get_texts():
                    text.set_fontsize(font_sizes["legend"])
                    text.set_fontfamily(font_family)
            else:
                legend.remove()

        self.canvas.draw()

    def plot_rof_timeseries(self):
        """Plot average baseline-corrected ROF time-series with 95% CI.

        Returns:
          None
        """
        # Determine file path
        if not hasattr(self.comet, "extracted_features_path"):
            print("[ERROR] COMET extracted_features_path not set")
            return

        rof_file = os.path.join(
            self.comet.extracted_features_path, f"ROF_timeseries{self.comet.export_format}"
        )
        if not os.path.exists(rof_file):
            print(f"[ERROR] ROF time-series file not found: {rof_file}")
            return

        # Load data
        try:
            if self.comet.export_format == ".csv":
                rof_df = pd.read_csv(rof_file)
            elif self.comet.export_format == ".pkl":
                rof_df = pd.read_pickle(rof_file)
            elif self.comet.export_format == ".hdf":
                rof_df = pd.read_hdf(rof_file, key="rof")
            elif self.comet.export_format == ".json":
                rof_df = pd.read_json(rof_file, orient="records", lines=True)
            else:
                print("[ERROR] Unsupported export format for ROF file")
                return
        except Exception as e:
            print(f"[ERROR] Failed to load ROF file: {e}")
            return

        # Filter by selected files if any
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        if selected_files:
            rof_df = rof_df[rof_df["Filename"].isin(selected_files)]

        # Determine microstate columns
        ms_cols = [c for c in rof_df.columns if c.startswith("ROF_")]
        if not ms_cols:
            print("[ERROR] No ROF_* columns found in data")
            return

        # Filter desired time window
        rof_df = rof_df[(rof_df["Time_ms"] >= -100) & (rof_df["Time_ms"] <= 600)].copy()
        if rof_df.empty:
            print("[ERROR] No data in requested time window")
            return

        # Prepare plotting data: group by Time_ms, compute mean and 95% CI across filenames
        grouped = rof_df.groupby("Time_ms")

        # Get current style parameters from UI/menu
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()

        # Reset figure
        self.figure.clear()
        ax = self.canvas.figure.gca()

        # Apply base font settings and clear axes
        self.clear_and_set_fonts(ax, font_family, font_sizes)

        palette = sns.color_palette(colormap, n_colors=len(ms_cols))

        for idx, ms in enumerate(ms_cols):
            means = grouped[ms].mean()
            means = means.dropna()
            counts = grouped[ms].count()
            stds = grouped[ms].std(ddof=1)
            se = stds / np.sqrt(counts)
            ci95 = 1.96 * se

            times = means.index.values.astype(float)

            ax.plot(times, means.values, label=ms, color=palette[idx])
            ax.fill_between(times, means - ci95, means + ci95, color=palette[idx], alpha=0.3)

        # Styling
        ax.set_xlim(-100, 600)
        ax.set_xlabel("Time (ms)", fontsize=font_sizes["label"], fontfamily=font_family)
        ax.set_ylabel(
            "Relative Occurrence Frequency", fontsize=font_sizes["label"], fontfamily=font_family
        )
        ax.set_title(
            "Average ROF across subjects", fontsize=font_sizes["title"], fontfamily=font_family
        )

        # Add vertical line at t=0
        ax.axvline(0, color="red", linewidth=2, zorder=5)
        # Add horizontal line at y=0
        ax.axhline(0, color="black", linewidth=2, linestyle="--", zorder=5)

        # Handle legend display
        legend = ax.legend()
        if legend:
            if display_options.get("show_legend", True):
                for text in legend.get_texts():
                    text.set_fontsize(font_sizes["legend"])
                    text.set_fontfamily(font_family)
            else:
                legend.remove()

        # Grid and axes visibility
        if not display_options.get("show_axes", True):
            ax.axis("off")

        if display_options.get("show_grid", False):
            ax.grid(True, alpha=0.3)

        # Update tick label fonts
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(font_sizes["tick"])
            label.set_fontfamily(font_family)

        self.canvas.draw()

        # Highlight ROF button
        self._highlight_button("plot_rof_button")

    def plot_rtf_heatmap(self):
        """Plot baseline-corrected RTF as heatmaps for different time windows.

        Returns:
          None
        """
        # Determine file path
        if not hasattr(self.comet, "extracted_features_path"):
            print("[ERROR] COMET extracted_features_path not set")
            return

        rtf_file = os.path.join(
            self.comet.extracted_features_path, f"RTF_averages{self.comet.export_format}"
        )
        if not os.path.exists(rtf_file):
            print(f"[ERROR] RTF averages file not found: {rtf_file}")
            return

        # Load data
        try:
            if self.comet.export_format == ".csv":
                rtf_df = pd.read_csv(rtf_file)
            elif self.comet.export_format == ".pkl":
                rtf_df = pd.read_pickle(rtf_file)
            elif self.comet.export_format == ".hdf":
                rtf_df = pd.read_hdf(rtf_file, key="rtf")
            elif self.comet.export_format == ".json":
                rtf_df = pd.read_json(rtf_file, orient="records", lines=True)
            else:
                print("[ERROR] Unsupported export format for RTF file")
                return
        except Exception as e:
            print(f"[ERROR] Failed to load RTF file: {e}")
            return

        # Filter by selected files if any
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        if selected_files:
            rtf_df = rtf_df[rtf_df["Filename"].isin(selected_files)]

        if rtf_df.empty:
            print("[ERROR] No RTF data available for selected files")
            return

        # Get current style parameters from UI/menu
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()

        # Reset figure
        self.figure.clear()

        # Determine microstates from column names
        transition_cols = [c for c in rtf_df.columns if c.startswith("RTF_")]
        
        if not transition_cols:
            print("[ERROR] No RTF transition columns found")
            return

        # Extract unique microstates from transition column names
        # Format can be either:
        # 1. Simple: RTF_A_B (3 parts) - just transition, no time window
        # 2. With window: RTF_post_tms_A_B (4+ parts) - includes time window
        microstates = set()
        time_windows = set()
        
        for col in transition_cols:
            parts = col.split("_")
            if len(parts) == 3:
                # Simple format: RTF_from_to (no time window info)
                # Use a default window name
                time_windows.add("baseline_corrected")
                microstates.add(parts[1])
                microstates.add(parts[2])
            elif len(parts) >= 4:
                # With window: RTF_window_from_to format
                window_name = "_".join(parts[1:-2])
                time_windows.add(window_name)
                microstates.add(parts[-2])
                microstates.add(parts[-1])
        
        microstates = sorted(list(microstates))
        time_windows = sorted(list(time_windows))
        n_windows = len(time_windows)
        n_states = len(microstates)

        if n_windows == 0 or n_states == 0:
            print(f"[ERROR] Could not parse RTF data: microstates={microstates}, windows={time_windows}")
            return

        # Create single heatmap (RTF data typically has one baseline-corrected matrix)
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        
        # Build transition matrix (already in percentage from RTF calculation)
        transition_matrix = np.zeros((n_states, n_states))
        transition_matrix_raw = np.zeros((n_states, n_states))  # Keep raw values for display
        
        for i, from_state in enumerate(microstates):
            for j, to_state in enumerate(microstates):
                # Try simple format first
                col_name = f"RTF_{from_state}_{to_state}"
                if col_name in rtf_df.columns:
                    # Average across files (already in percentage)
                    raw_value = rtf_df[col_name].mean()
                    transition_matrix_raw[i, j] = raw_value
                else:
                    # Try with window name
                    for window_name in time_windows:
                        col_name = f"RTF_{window_name}_{from_state}_{to_state}"
                        if col_name in rtf_df.columns:
                            raw_value = rtf_df[col_name].mean()
                            transition_matrix_raw[i, j] = raw_value
                            break
        
        # For visualization, cap extreme values at ±200% for better color scaling
        # But keep raw values for annotation
        transition_matrix = np.clip(transition_matrix_raw, -200, 200)
        
        # Determine colorbar range using percentiles for robustness
        # Use 95th percentile to avoid extreme outliers affecting the scale
        non_zero_values = transition_matrix[transition_matrix != 0]
        if len(non_zero_values) > 0:
            percentile_95 = np.percentile(np.abs(non_zero_values), 95)
            # Round up to nice values
            if percentile_95 < 10:
                vmax = 10
            elif percentile_95 < 25:
                vmax = 25
            elif percentile_95 < 50:
                vmax = 50
            elif percentile_95 < 100:
                vmax = 100
            else:
                vmax = 200
        else:
            vmax = 50
        
        # Plot heatmap with percentage scale
        im = ax.matshow(transition_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        
        # Add colorbar with percentage label
        cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Change (%)', fontsize=font_sizes["label"] - 2, fontfamily=font_family)
        cbar.ax.tick_params(labelsize=font_sizes["tick"])
        for label in cbar.ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes["tick"])
        
        # Annotate with raw percentage values (not capped)
        for i in range(n_states):
            for j in range(n_states):
                if i != j:  # Skip self-transitions
                    raw_value = transition_matrix_raw[i, j]
                    capped_value = transition_matrix[i, j]
                    # Use adaptive threshold based on colorbar range
                    text_color_threshold = vmax * 0.4  # 40% of max for better contrast
                    
                    # Show raw value with asterisk if it was capped for color scaling
                    if abs(raw_value) > 200:
                        display_text = f"{raw_value:.0f}%*"  # Asterisk indicates capped for color
                    else:
                        display_text = f"{raw_value:.1f}%"
                    
                    ax.text(
                        j, i, display_text,
                        ha="center", va="center",
                        color="black" if abs(capped_value) < text_color_threshold else "white",
                        fontsize=font_sizes["tick"] - 1,
                        fontfamily=font_family,
                    )
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_states))
        ax.set_yticks(np.arange(n_states))
        ax.set_xticklabels(microstates, fontsize=font_sizes["tick"], fontfamily=font_family)
        ax.set_yticklabels(microstates, fontsize=font_sizes["tick"], fontfamily=font_family)
        
        # Set axis labels
        ax.set_xlabel("To", fontsize=font_sizes["label"], fontfamily=font_family)
        ax.set_ylabel("From", fontsize=font_sizes["label"], fontfamily=font_family)

        # Set overall figure title
        n_subjects = rtf_df["Filename"].nunique() if "Filename" in rtf_df.columns else len(rtf_df)
        
        self.figure.suptitle(
            f"Relative Transition Frequency - Post-Event % Change from Baseline\n({n_subjects} subjects)",
            fontsize=font_sizes["title"],
            fontfamily=font_family,
        )

        self.figure.tight_layout()
        self.canvas.draw()

        # Highlight RTF button
        self._highlight_button("plot_rtf_button")

    def plot_pre_post_event_bar(self, features_df, feature, font_sizes, colormap, font_family, display_options):
        """Plot bar plot comparing pre-event vs post-event features for epoched data.

        Args:
          features_df (pandas.DataFrame): Features with Window_Type column, optionally Trial column.
          feature (str): Feature short code.
          font_sizes (dict[str, int]): Title/label/tick/legend sizes.
          colormap (str): Matplotlib colormap name.
          font_family (str): Font family to apply.
          display_options (dict[str, bool]): Visibility options.

        Returns:
          None
        """
        # Reset figure
        self.figure.clear()
        ax = self.canvas.figure.gca()
        
        # Check if Window_Type column exists (indicates pre/post event features)
        if "Window_Type" not in features_df.columns:
            error_msg = 'No pre/post event features found (missing Window_Type column)'
            ax.text(0.5, 0.5, error_msg, 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            self.canvas.draw()
            return
        
        # Check if DataFrame is empty after filtering
        if features_df.empty:
            error_msg = 'No data found for selected files.\nPlease re-extract features with the updated code.'
            ax.text(0.5, 0.5, error_msg, 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            self.canvas.draw()
            return
        
        # Get feature columns for the selected feature
        filter_cols = [col for col in features_df.columns if col.startswith(feature + "_")]
        if not filter_cols and feature in features_df.columns:
            filter_cols = [feature]
        filter_cols.sort()
        
        if not filter_cols:
            ax.text(0.5, 0.5, f'No features found for "{feature}"', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            self.canvas.draw()
            return
        
        # Prepare data for plotting - handle both with and without Trial column
        has_trial_column = "Trial" in features_df.columns
        has_ntrials_column = "N_Trials" in features_df.columns
        
        # Build id_vars list dynamically
        id_vars = ["Filename", "Window_Type"]
        if has_trial_column:
            id_vars.append("Trial")
        if has_ntrials_column:
            id_vars.append("N_Trials")
        
        plot_data = pd.melt(
            features_df, 
            id_vars=id_vars, 
            value_vars=filter_cols,
            var_name="Microstate",
            value_name="Value"  # Use generic name to avoid conflicts
        )
        
        # Clean up microstate labels (remove feature prefix)
        plot_data["Microstate"] = plot_data["Microstate"].str.replace(f"{feature}_", "")
        
        self.clear_and_set_fonts(ax, font_family, font_sizes)
        
        # Create grouped bar plot
        num_microstates = len(filter_cols)
        color_palette = sns.color_palette(colormap, num_microstates)
        
        # Plot using seaborn
        sns.barplot(
            data=plot_data,
            x="Window_Type",
            y="Value",
            hue="Microstate",
            ax=ax,
            palette=color_palette,
            errorbar="sd",  # Show standard deviation
            capsize=0.1,
        )
        
        # Set labels and title
        ax.set_xlabel("Event Window", fontsize=font_sizes["label"], fontfamily=font_family)
        ax.set_ylabel(
            self.comet.feature_list_dictionary.get(feature, feature),
            fontsize=font_sizes["label"],
            fontfamily=font_family,
        )
        
        # Set figure title with correct subject and trial counts
        # Count unique base filenames for subject count
        if "base_filename" in features_df.columns:
            n_subjects = features_df["base_filename"].nunique()
        else:
            # Remove _Pre and _Post suffixes from Filename to get unique subjects
            unique_subjects = features_df["Filename"].apply(
                lambda x: x.rsplit("_Pre", 1)[0].rsplit("_Post", 1)[0]
            ).nunique()
            n_subjects = unique_subjects
        
        # Count total trials across all subjects
        if "N_Trials" in plot_data.columns:
            # For pre_post features: sum the N_Trials column from unique filenames
            # Remove _Pre/_Post suffix to avoid double counting (each subject has Pre and Post rows)
            plot_data["base_subject"] = plot_data["Filename"].apply(
                lambda x: x.rsplit("_Pre", 1)[0].rsplit("_Post", 1)[0]
            )
            unique_files = plot_data.drop_duplicates(subset=["base_subject"])
            n_unique_trials = int(unique_files["N_Trials"].sum())
        elif "base_filename" in plot_data.columns and "Trial" in plot_data.columns:
            # For per-trial features: count unique (subject, trial) pairs
            n_unique_trials = plot_data.groupby(["base_filename", "Trial"]).ngroups
        elif "Trial" in plot_data.columns:
            n_unique_trials = plot_data["Trial"].nunique()
        else:
            # Fallback: estimate from number of observations
            n_unique_trials = len(features_df) // 2
        
        # Total observations (includes both Pre and Post for each subject)
        n_total_observations = len(features_df)
        
        self.figure.suptitle(
            f"{self.comet.feature_list_dictionary.get(feature, feature)} - Pre/Post Event\n"
            f"({n_subjects} subjects, {n_unique_trials} trials, {n_total_observations} observations)",
            fontsize=font_sizes["title"],
            fontfamily=font_family,
        )
        
        # Update tick fonts
        ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes["tick"])
        
        # Handle legend
        legend = ax.get_legend()
        if legend:
            if display_options.get("show_legend", True):
                legend.set_visible(True)
                legend.set_title("Microstate", prop={'size': font_sizes["legend"], 'family': font_family})
                for text in legend.get_texts():
                    text.set_fontsize(font_sizes["legend"])
                    text.set_fontfamily(font_family)
            else:
                legend.remove()
        
        # Apply display options
        if not display_options.get("show_axes", True):
            ax.axis("off")
        if display_options.get("show_grid", False):
            ax.grid(True, alpha=0.3, axis='y')
        
        self.canvas.draw()
