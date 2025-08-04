import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QActionGroup, QFileDialog, QAbstractItemView
from PyQt5.QtGui import QKeySequence
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from features_utils.feature_io import FeatureIO
from gui_utils.set_widgets_status import set_widgets_status


class FeatureVisualizationWindow(QMainWindow):
    def __init__(self, context, parent=None, tbx=None):
        super().__init__(parent)

        self.comet = tbx
        self.ui = self.load_ui(context)

        # Initialize feature_mode with a default value (will be set from main window)
        self.feature_mode = ['averaged']  # Default fallback

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas.setMinimumHeight(100)
        self.ui.Figure_Layout.addWidget(self.canvas)

        self.setup_window()
        self.setup_style_actions()
        self.bind_events()
        self.feature_visualization_controller()

    def get_selected_feature_code(self):
        """
        Get the feature short code corresponding to the selected full name in the combo box.
        
        Returns:
            str: Feature short code (e.g., 'COV', 'OCC') or the full name if not found
        """
        selected_full_name = self.ui.feature_combo.currentText()
        
        # Create reverse mapping from full names to short codes using comet's dictionary
        if hasattr(self.comet, 'feature_list_dictionary') and self.comet.feature_list_dictionary:
            reverse_dict = {v: k for k, v in self.comet.feature_list_dictionary.items()}
            return reverse_dict.get(selected_full_name, selected_full_name)
        
        return selected_full_name

    def setup_style_actions(self):
        """
        Setup font size and display actions for different text elements
        """
        # Create action group for colormap actions
        self.colormap_action_group = QActionGroup(self)
        self.colormap_action_group.setExclusive(True)

        # Dictionary mapping action names to colormap names
        self.colormap_mapping = {
            'cmap_set1': 'Set1',
            'cmap_set2': 'Set2',
            'cmap_tab10': 'tab10',
            'cmap_dark2': 'Dark2',
            'cmap_accent': 'Accent'
        }

        # Add colormap actions to the group and make them checkable
        for action_name, colormap_name in self.colormap_mapping.items():
            if hasattr(self.ui, action_name):
                action = getattr(self.ui, action_name)
                action.setCheckable(True)
                self.colormap_action_group.addAction(action)

                # Connect action to colormap change handler
                action.triggered.connect(
                    lambda checked, cm=colormap_name: self.change_colormap(cm)
                )

        self.ui.cmap_tab10.setChecked(True)
        self.current_colormap = 'tab10'

        # Create action group for font family actions
        self.font_action_group = QActionGroup(self)
        self.font_action_group.setExclusive(True)

        # Dictionary mapping action names to font family names
        self.font_mapping = {
            'font_arial': 'Arial',
            'font_calibri': 'Calibri',
            'font_times': 'Times New Roman'
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
        self.current_font_family = 'Arial'

        # Setup independent checkable actions (not mutually exclusive)
        self.setup_display_options()

        # Setup font size actions (if using submenu approach)
        self.setup_font_size_actions()

    def setup_display_options(self):
        """
        Setup independent display options (legend, grid, axes)
        """
        # Dictionary mapping action names to their default states
        self.display_options = {
            'show_legend': True,  # Default: show legend
            'show_grid': False,  # Default: no grid
            'show_axes': True  # Default: show axes
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
        """
        Setup font size actions for different text elements (title, label, tick, legend)
        Each category has its own mutually exclusive group
        """
        # Define font size mappings for each category
        self.font_size_categories = {
            'title': {
                'title_font_small': 14,
                'title_font_medium': 18,
                'title_font_large': 22,
                'title_font_xlarge': 26
            },
            'label': {
                'label_font_small': 10,
                'label_font_medium': 14,
                'label_font_large': 18,
                'label_font_xlarge': 22
            },
            'tick': {
                'tick_font_small': 10,
                'tick_font_medium': 14,
                'tick_font_large': 18,
                'tick_font_xlarge': 22
            },
            'legend': {
                'legend_font_small': 10,
                'legend_font_medium': 14,
                'legend_font_large': 18,
                'legend_font_xlarge': 22
            }
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
                        lambda checked, cat=category, size=font_size: self.change_font_size_category(cat, size)
                    )

            # Set default font size (medium for all categories)
            default_action_name = f"{category}_font_medium"
            if hasattr(self.ui, default_action_name):
                getattr(self.ui, default_action_name).setChecked(True)

    def change_colormap(self, colormap_name):
        """
        Handle colormap change from menu actions
        """
        self.current_colormap = colormap_name
        self._maybe_refresh_rof_plot()

    def change_font_family(self, font_name):
        """
        Handle font family change from menu actions
        """
        self.current_font_family = font_name
        self._maybe_refresh_rof_plot()

    def change_display_option(self, option_name, checked):
        """
        Handle display option change from menu actions
        """
        # print(f"Display option '{option_name}' set to: {checked}")
        self._maybe_refresh_rof_plot()

    def change_font_size_category(self, category, font_size):
        """
        Handle font size change for specific category from menu actions
        """

        # Update the UI input fields if they exist
        if category == 'title' and hasattr(self.ui, 'font_size_input'):
            self.ui.font_size_input.setText(str(font_size))
        elif category == 'tick' and hasattr(self.ui, 'label_size_input'):
            self.ui.label_size_input.setText(str(font_size))

        self._maybe_refresh_rof_plot()

    def _maybe_refresh_rof_plot(self):
        """Replot ROF immediately if it is the selected feature and plot is visible."""
        if hasattr(self.ui, 'plot_rof_button') and self.get_selected_feature_code() == 'ROF':
            # Directly call plot function so updates are instant
            self.plot_rof_timeseries()

    # ------------------------- Button highlighting -------------------------
    def _reset_plot_button_styles(self):
        """Reset styles for all plot buttons."""
        for btn_name in ['plot_static_violin_plot_button', 'plot_static_box_plot_button',
                         'plot_all_dynamic_button', 'plot_heatmap_button', 'plot_rof_button']:
            if hasattr(self.ui, btn_name):
                getattr(self.ui, btn_name).setStyleSheet("")

    def _highlight_button(self, button_attr_name):
        """Set given button background to green and reset others."""
        self._reset_plot_button_styles()
        if hasattr(self.ui, button_attr_name):
            getattr(self.ui, button_attr_name).setStyleSheet("background-color: #4CAF50; color: white;")

    def get_selected_colormap(self):
        """
        Get the currently selected colormap from action group
        """
        checked_action = self.colormap_action_group.checkedAction()
        if checked_action:
            # Find the colormap name from the action
            for action_name, colormap_name in self.colormap_mapping.items():
                if hasattr(self.ui, action_name) and getattr(self.ui, action_name) == checked_action:
                    return colormap_name

        # Fallback to combobox or default
        return getattr(self, 'current_colormap', 'viridis')

    def get_selected_font_family(self):
        """
        Get the currently selected font family from action group
        """
        checked_action = self.font_action_group.checkedAction()
        if checked_action:
            # Find the font name from the action
            for action_name, font_name in self.font_mapping.items():
                if hasattr(self.ui, action_name) and getattr(self.ui, action_name) == checked_action:
                    return font_name

        # Fallback to default
        return getattr(self, 'current_font_family', 'Arial')

    def get_display_options(self):
        """
        Get the current state of display options
        """
        options = {}
        for option_name in self.display_options.keys():
            if hasattr(self.ui, option_name):
                action = getattr(self.ui, option_name)
                options[option_name] = action.isChecked()
            else:
                # Use default if action doesn't exist
                options[option_name] = self.display_options[option_name]

        return options

    def get_selected_font_size_from_menu(self):
        """
        Get the currently selected font sizes for all categories from action groups
        """
        font_sizes = {}

        for category, action_group in self.font_size_action_groups.items():
            checked_action = action_group.checkedAction()
            if checked_action:
                # Find the font size from the action
                for action_name, font_size in self.font_size_categories[category].items():
                    if hasattr(self.ui, action_name) and getattr(self.ui, action_name) == checked_action:
                        font_sizes[category] = font_size
                        break

            # Set default if no action is checked
            if category not in font_sizes:
                defaults = {'title': 18, 'label': 14, 'tick': 12, 'legend': 12}
                font_sizes[category] = defaults.get(category, 12)

        return font_sizes

    def load_ui(self, context):
        basepath = os.path.dirname(__file__)
        return uic.loadUi(context.get_resource("FeatureVisualizationWindow.ui"), self)

    def setup_window(self):
        self.setWindowTitle("Visualization of the extracted features")
        # Add maximize button to the window
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

    def bind_events(self):
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
            self.ui.reset_groups_button
        ]
        for button in buttons:
            button.clicked.connect(self.feature_visualization_controller)

        # Bind ROF plot button if exists in UI
        if hasattr(self.ui, 'plot_rof_button'):
            self.ui.plot_rof_button.clicked.connect(self.plot_rof_timeseries)

        # Trigger controller when feature selection changes
        if hasattr(self.ui, 'feature_combo'):
            self.ui.feature_combo.currentIndexChanged.connect(self.feature_visualization_controller)

        # Trigger controller when static/dynamic radio toggled
        if hasattr(self.ui, 'static_radio'):
            self.ui.static_radio.toggled.connect(self.feature_visualization_controller)
        if hasattr(self.ui, 'dynamic_radio'):
            self.ui.dynamic_radio.toggled.connect(self.feature_visualization_controller)

        # Bind selection helper widgets if they exist
        if hasattr(self.ui, 'select_all_checkbox'):
            self.ui.select_all_checkbox.stateChanged.connect(self.handle_select_all_checkbox)
        if hasattr(self.ui, 'select_pattern_checkbox'):
            self.ui.select_pattern_checkbox.stateChanged.connect(self.apply_pattern_selection)
        if hasattr(self.ui, 'select_pattern_input'):
            self.ui.select_pattern_input.textChanged.connect(self.apply_pattern_selection)

        # Update selected file lineedit initially
        if hasattr(self.ui, 'selected_file_lineedit'):
            self.update_selected_file_lineedit()

    def export_feature_image(self):
        """Export feature visualization images to file"""
        # Get study name for default filename
        study_name = getattr(self.comet, 'study_name', 'features')

        # Get current feature name for filename
        current_feature = self.ui.feature_combo.currentText()
        if current_feature:
            default_filename = f"{study_name}_{current_feature}_visualization.pdf"
        else:
            default_filename = f"{study_name}_features.pdf"

        # Get default directory from comet object
        default_dir = getattr(self.comet, 'save_dir', '')

        # Combine directory and filename for full default path
        if default_dir:
            default_path = os.path.join(default_dir, default_filename)
        else:
            default_path = default_filename

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Choose a location and filename to save the feature visualization",
            default_path,  # Use the full default path (directory + filename)
            "PDF Files (*.pdf);;PNG Files (*.png);;JPG Files (*.jpg);;SVG Files (*.svg);;All Files (*)",
            options=options
        )

        if file_name:
            extension = os.path.splitext(file_name)[-1].lower()

            # Ensure that the file has an extension
            if not extension:
                file_name += '.pdf'  # Default to PDF for vector format

            # Save with appropriate settings for vector formats
            if extension in ['.pdf', '.svg']:
                # Vector formats - save with high DPI and vector-friendly settings
                self.figure.savefig(file_name, dpi=300, bbox_inches='tight',
                                    facecolor='white', edgecolor='none')
            else:
                # Raster formats
                self.figure.savefig(file_name, dpi=300, bbox_inches='tight')

    def feature_visualization_controller(self):
        """
        Control the behavior of the Feature Visualization window based on user selections.
        """

        # Determine current visualization mode based on radio buttons
        static_mode = hasattr(self.ui, 'static_radio') and self.ui.static_radio.isChecked()
        dynamic_mode = hasattr(self.ui, 'dynamic_radio') and self.ui.dynamic_radio.isChecked()

        # Adjust list selection behavior
        if dynamic_mode:
            # Only a single file can be selected for dynamic (windowed) features
            self.ui.all_files_list.setSelectionMode(QAbstractItemView.SingleSelection)
        else:
            # Allow multiple selections for global (averaged) features
            self.ui.all_files_list.setSelectionMode(QAbstractItemView.MultiSelection)

        # Enable/disable selection helper widgets based on mode
        if hasattr(self.ui, 'select_all_checkbox'):
            self.ui.select_all_checkbox.setEnabled(static_mode)
        if hasattr(self.ui, 'select_pattern_checkbox'):
            self.ui.select_pattern_checkbox.setEnabled(static_mode)
        if hasattr(self.ui, 'select_pattern_input'):
            self.ui.select_pattern_input.setEnabled(static_mode)
            # If switching to dynamic mode, clear pattern selection and unchecked boxes
            if not static_mode:
                self.ui.select_pattern_input.clear()
                if hasattr(self.ui, 'select_all_checkbox'):
                    self.ui.select_all_checkbox.setChecked(False)
                if hasattr(self.ui, 'select_pattern_checkbox'):
                    self.ui.select_pattern_checkbox.setChecked(False)

        # Helper variables for currently selected items
        num_selected = len(self.ui.all_files_list.selectedItems())
        has_selection = num_selected > 0
        single_selection = num_selected == 1

        # --- Static buttons ---
        self.ui.plot_static_violin_plot_button.setEnabled(static_mode and has_selection)
        self.ui.plot_static_box_plot_button.setEnabled(static_mode and has_selection)

        # Heatmap (only for Transition Probability feature in static mode)
        if static_mode and self.get_selected_feature_code() == 'TP':
            set_widgets_status(self.ui.plot_heatmap_button, mode='enable')
            set_widgets_status(self.ui.plot_heatmap_button, mode='show')
        else:
            set_widgets_status(self.ui.plot_heatmap_button, mode='disable')
            set_widgets_status(self.ui.plot_heatmap_button, mode='show')

        # --- Dynamic button ---
        dynamic_available = "sliding" in self.feature_mode
        self.ui.plot_all_dynamic_button.setEnabled(dynamic_mode and single_selection and dynamic_available)

        # Enable/disable ROF time-series plot button
        if hasattr(self.ui, 'plot_rof_button'):
            if self.get_selected_feature_code() == 'ROF':
                set_widgets_status(self.ui.plot_rof_button, mode='enable')
                set_widgets_status(self.ui.plot_rof_button, mode='show')
            else:
                set_widgets_status(self.ui.plot_rof_button, mode='disable')
                set_widgets_status(self.ui.plot_rof_button, mode='show')

        # Update comparison widgets status based on list contents
        count_group_a = self.ui.group_a_files_list.count()
        count_group_b = self.ui.group_b_files_list.count()
        self.ui.add_group_a_button.setEnabled(not len(self.ui.all_files_list.selectedItems()) == 0)
        self.ui.add_group_b_button.setEnabled(not len(self.ui.all_files_list.selectedItems()) == 0)
        self.ui.remove_group_a_button.setEnabled(not count_group_a == 0)
        self.ui.remove_group_b_button.setEnabled(not count_group_b == 0)
        self.ui.reset_groups_button.setEnabled(count_group_a > 0 or count_group_b > 0)
        self.ui.plot_groups_button.setEnabled(count_group_a > 0 and count_group_b > 0)

        # Enable/disable other plot buttons based on ROF selection
        if self.get_selected_feature_code() == 'ROF':
            # Disable unrelated plot buttons
            for btn_name in ['plot_static_violin_plot_button', 'plot_static_box_plot_button',
                             'plot_all_dynamic_button', 'plot_heatmap_button']:
                if hasattr(self.ui, btn_name):
                    set_widgets_status(getattr(self.ui, btn_name), mode='disable')
                    set_widgets_status(getattr(self.ui, btn_name), mode='show')
        else:
            # Re-enable standard buttons visibility (status handled earlier)
            for btn_name in ['plot_static_violin_plot_button', 'plot_static_box_plot_button',
                             'plot_all_dynamic_button']:
                if hasattr(self.ui, btn_name):
                    # ensure visible
                    set_widgets_status(getattr(self.ui, btn_name), mode='show')

        # Automatically draw plot for newly selected feature
        self.auto_plot_selected_feature()

        # Refresh ROF plot if relevant
        self._maybe_refresh_rof_plot()

        # Update selected file display
        self.update_selected_file_lineedit()

        # Ensure any pattern / select all logic stays in sync
        if getattr(self.ui, 'select_pattern_checkbox', None) and self.ui.select_pattern_checkbox.isChecked():
            self.apply_pattern_selection(run_controller=False)

    # ------------------------- Selection helper methods -------------------------
    def update_selected_file_lineedit(self):
        """Display selected file(s) name or count in the read-only line-edit."""
        if not hasattr(self.ui, 'selected_file_lineedit'):
            return
        selected_items = self.ui.all_files_list.selectedItems()
        if len(selected_items) == 1:
            self.ui.selected_file_lineedit.setText(selected_items[0].text())
        elif len(selected_items) > 1:
            self.ui.selected_file_lineedit.setText(f"{len(selected_items)} files selected")
        else:
            self.ui.selected_file_lineedit.clear()

    def handle_select_all_checkbox(self, state):
        """Select or deselect all files when the checkbox state changes."""
        if state == Qt.Checked:
            # Select all items
            self.ui.all_files_list.selectAll()
            # Ensure pattern checkbox is unchecked to avoid conflicts
            if hasattr(self.ui, 'select_pattern_checkbox'):
                self.ui.select_pattern_checkbox.setChecked(False)
        self.update_selected_file_lineedit()
        # Re-run controller to update button states
        self.feature_visualization_controller()

    def apply_pattern_selection(self, *_, run_controller: bool = True):
        """Select files matching pattern when the pattern checkbox is active."""
        if not (hasattr(self.ui, 'select_pattern_checkbox') and hasattr(self.ui, 'select_pattern_input')):
            return
        if not self.ui.select_pattern_checkbox.isChecked():
            return
        pattern = self.ui.select_pattern_input.text().strip().lower()
        if not pattern:
            return
        list_widget = self.ui.all_files_list
        list_widget.clearSelection()
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if pattern in item.text().lower():
                item.setSelected(True)
        # Uncheck select all checkbox to avoid ambiguity
        if hasattr(self.ui, 'select_all_checkbox'):
            self.ui.select_all_checkbox.setChecked(False)
        self.update_selected_file_lineedit()
        if run_controller:
            self.feature_visualization_controller()

    def auto_plot_selected_feature(self):
        """Automatically plot the currently selected feature without extra clicks."""
        feature_code = self.get_selected_feature_code()

        if feature_code == 'ROF':
            self.plot_rof_timeseries()
        elif feature_code == 'TP':
            # If at least one file selected
            if self.ui.all_files_list.selectedItems():
                self.show_tp_heatmap()
        else:
            # Determine which visualization mode is active
            static_mode = hasattr(self.ui, 'static_radio') and self.ui.static_radio.isChecked()
            dynamic_mode = hasattr(self.ui, 'dynamic_radio') and self.ui.dynamic_radio.isChecked()

            if dynamic_mode and 'sliding' in self.feature_mode:
                if self.ui.all_files_list.currentItem():
                    self.show_dynamic_line_all()
            elif static_mode:
                if self.ui.all_files_list.selectedItems():
                    self.show_static_box_plot()

    @staticmethod
    def move_file_between_lists(source_list, target_list):
        if source_list.currentItem():
            selected_items = source_list.selectedItems()
            if not selected_items:
                return
            for item in selected_items:
                source_list.takeItem(source_list.row(item))
                target_list.addItem(item.text())

    def move_file_from_all_to_a(self):
        self.move_file_between_lists(self.ui.all_files_list, self.ui.group_a_files_list)

    def move_file_from_all_to_b(self):
        self.move_file_between_lists(self.ui.all_files_list, self.ui.group_b_files_list)

    def move_file_from_a_to_all(self):
        self.move_file_between_lists(self.ui.group_a_files_list, self.ui.all_files_list)

    def move_file_from_b_to_all(self):
        self.move_file_between_lists(self.ui.group_b_files_list, self.ui.all_files_list)

    def reset_groups(self):
        """
        Clears all file lists and resets the plot.
        """
        self.clear_all_lists()
        self.figure.clear()
        self.canvas.draw()
        # Clear figure title
        self.figure.suptitle("")

    def get_plot_parameters(self):
        # Get plot parameters from UI components
        # Check if font sizes should come from menu
        menu_font_sizes = self.get_selected_font_size_from_menu()

        # Use menu font sizes if available, otherwise fall back to UI inputs
        title_size = menu_font_sizes.get('title', 18)
        label_size = menu_font_sizes.get('label', 14)
        tick_size = menu_font_sizes.get('tick', 12)
        legend_size = menu_font_sizes.get('legend', 12)

        # Fallback to UI input fields if menu not available
        if not hasattr(self, 'font_size_action_groups'):
            try:
                title_size = int(self.ui.font_size_input.text()) if hasattr(self.ui, 'font_size_input') else 18
                tick_size = int(self.ui.label_size_input.text()) if hasattr(self.ui, 'label_size_input') else 12
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
            'title': title_size,
            'label': label_size,
            'tick': tick_size,
            'legend': legend_size
        }

        return font_sizes, colormap, font_family, display_options

    def show_static_violin_all(self):
        """
        Displays violin plots of all static features.
        """
        selected_feature = self.get_selected_feature_code()
        # Get the selected files from the all_files_list
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        # Load all features
        all_features_df = self.load_features("averaged")
        # Filter features_df based on selected files
        features_df = all_features_df[all_features_df['Filename'].isin(selected_files)]
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_violin(features_df, selected_feature, font_sizes, colormap, font_family, display_options)
        self._highlight_button('plot_static_violin_plot_button')

    def show_static_box_plot(self):
        """
        Displays box plots of all static features.
        """
        selected_feature = self.get_selected_feature_code()
        # Get the selected files from the all_files_list
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        # Load all features
        all_features_df = self.load_features("averaged")
        # Filter features_df based on selected files
        features_df = all_features_df[all_features_df['Filename'].isin(selected_files)]
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_box(features_df, selected_feature, font_sizes, colormap, font_family, display_options)
        self._highlight_button('plot_static_box_plot_button')

    def show_dynamic_line_all(self):
        """
        Displays line plots of all dynamic features.
        """
        selected_feature = self.get_selected_feature_code()
        selected_file = self.ui.all_files_list.currentItem().text()
        features_df = self.load_features("sliding").query(f'Filename == "{selected_file}"')
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_line(features_df, selected_feature, font_sizes, colormap, font_family, display_options)
        self._highlight_button('plot_all_dynamic_button')

    def show_tp_heatmap(self):
        """
        Displays the heatmap for transition probabilities.
        """
        # Get the selected files from the all_files_list
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        # Load all features
        all_features_df = self.load_features("averaged")
        # Filter features_df based on selected files
        features_df = all_features_df[all_features_df['Filename'].isin(selected_files)]
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_heatmap(features_df, font_sizes, font_family, display_options)
        self._highlight_button('plot_heatmap_button')

        # Set figure title
        self.figure.suptitle("Transition Probability Heatmap",
                             fontsize=font_sizes['title'], fontfamily=font_family)

    def compare_groups(self):
        """
        Compares selected features between two groups.
        """
        selected_feature = self.get_selected_feature_code()
        group_a_name, group_b_name = self.get_group_names()
        features_df = self.load_features("averaged")
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_group_comparison(
            features_df, selected_feature, group_a_name, group_b_name, font_sizes, colormap, font_family,
            display_options
        )

    def load_features(self, mode):
        """
        Loads features from a file and returns them as a DataFrame.
        """
        feature_path = os.path.join(self.extracted_features_path, f"real_{mode}_features{self.export_format}")
        return FeatureIO().import_features(feature_path, self.export_format)

    def clear_all_lists(self):
        """
        Clears all file lists.
        """
        self.ui.all_files_list.clear()
        self.ui.group_a_files_list.clear()
        self.ui.group_b_files_list.clear()
        self.populate_all_files_list()

    def populate_all_files_list(self):
        """
        Populates the all_files_list with EEG file names.
        """
        for eeg_file in self.list_eegs:
            self.ui.all_files_list.addItem(str(eeg_file))

    def set_labels_ticks(self, filter_cols, feature, ax, font_sizes, font_family, display_options):
        """Set x/y labels, tick labels and apply display options"""

        # Helper to convert numerical state indices to letters (1->A, 2->B, ...)
        def _state_to_letter(state_str: str) -> str:
            try:
                state_idx = int(state_str)
                # Ensure 1-based mapping; fall back gracefully
                letter_idx = max(state_idx - 1, 0)
                return chr(ord('A') + letter_idx)
            except (ValueError, TypeError):
                # If conversion fails, just return the original string
                return state_str

        # Determine appropriate x-tick labels
        if feature in ['TP', 'RTF']:
            # Expect column names like "TP_1_2" → "A-B" or "RTF_A_B"
            xticklabels = []
            for col in filter_cols:
                parts = col.split('_')
                if len(parts) >= 3:
                    from_letter = _state_to_letter(parts[-2])
                    to_letter = _state_to_letter(parts[-1])
                    xticklabels.append(f"{from_letter}-{to_letter}")
                else:
                    # Fallback to original logic if format unexpected
                    xticklabels.append(parts[-1])
            ax.set_xlabel("Transition", fontsize=font_sizes['label'], fontfamily=font_family)
        else:
            # Default behaviour for other features
            xticklabels = [col.split('_')[-1] for col in filter_cols]
            ax.set_xlabel("Microstate", fontsize=font_sizes['label'], fontfamily=font_family)

        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(self.comet.feature_list_dictionary[feature], fontsize=font_sizes['label'], fontfamily=font_family)
        ax.tick_params(axis='both', which='major', labelsize=font_sizes['tick'])

        # Set font family for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes['tick'])

        # Apply display options
        if not display_options.get('show_axes', True):
            ax.axis('off')

        if display_options.get('show_grid', False):
            ax.grid(True, alpha=0.3)

        # Set figure title instead of plot_label
        self.figure.suptitle(f"{self.comet.feature_list_dictionary[feature]}",
                             fontsize=font_sizes['title'], fontfamily=font_family)

    def set_labels_ticks_sliding(self, filter_cols, feature, ax, font_sizes, font_family, display_options):
        """Set x/y labels, tick labels and apply display options for sliding window features"""

        # Helper to convert numerical state indices to letters (1->A, 2->B, ...)
        def _state_to_letter(state_str: str) -> str:
            try:
                state_idx = int(state_str)
                # Ensure 1-based mapping; fall back gracefully
                letter_idx = max(state_idx - 1, 0)
                return chr(ord('A') + letter_idx)
            except (ValueError, TypeError):
                # If conversion fails, just return the original string
                return state_str

        # For sliding window features, x-axis represents time in seconds
        ax.set_xlabel("Time (seconds)", fontsize=font_sizes['label'], fontfamily=font_family)
        
        # Set y-axis label based on feature type
        if feature in ['TP', 'RTF']:
            ax.set_ylabel("Transition Probability", fontsize=font_sizes['label'], fontfamily=font_family)
        else:
            ax.set_ylabel(self.comet.feature_list_dictionary[feature], fontsize=font_sizes['label'], fontfamily=font_family)
        
        ax.tick_params(axis='both', which='major', labelsize=font_sizes['tick'])

        # Set font family for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes['tick'])

        # Apply display options
        if not display_options.get('show_axes', True):
            ax.axis('off')

        if display_options.get('show_grid', False):
            ax.grid(True, alpha=0.3)

        # Set figure title
        self.figure.suptitle(f"{self.comet.feature_list_dictionary[feature]} (Sliding Window)",
                             fontsize=font_sizes['title'], fontfamily=font_family)

    def plot_violin(self, features_df, feature, font_sizes, colormap, font_family, display_options):
        """
        Plots a violin plot for the selected static feature.
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=['Filename'], value_vars=filter_cols)
        plot_data.columns = ['Filename', 'Feature', feature]

        self.clear_and_set_fonts(ax, font_family, font_sizes)

        # Use a color palette for the violins based on the number of features
        num_features = len(filter_cols)
        color_palette = sns.color_palette(colormap, num_features)

        sns.violinplot(x='Feature', y=feature, data=plot_data, ax=ax, palette=color_palette)
        sns.swarmplot(x='Feature', y=feature, data=plot_data, ax=ax, color="white", size=10, marker='o')
        self.set_labels_ticks(filter_cols, feature, ax, font_sizes, font_family, display_options)

        # Handle legend display
        legend = ax.get_legend()
        if legend:
            if display_options.get('show_legend', True):
                legend.set_visible(True)
                # Update legend font
                for text in legend.get_texts():
                    text.set_fontsize(font_sizes['legend'])
                    text.set_fontfamily(font_family)
            else:
                legend.remove()

        self.canvas.draw()

    def plot_line(self, features_df, feature, font_sizes, colormap, font_family, display_options):
        """
        Plots a line plot for the selected dynamic feature.
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        filter_cols.sort()

        # Convert window indices to time in seconds for sliding window features
        if 'Window_index' in features_df.columns:
            # Get sampling rate and window size from COMET object
            sampling_rate = getattr(self.comet, 'sample_rate', 250)  # Default to 250 Hz
            sliding_window_size = getattr(self.comet, 'sliding_window_size', 1)  # Default to 1 second
            
            self.clear_and_set_fonts(ax, font_family, font_sizes)
            
            # Plot horizontal lines for each window instead of connected points
            # Get unique features for coloring
            unique_features = [col for col in filter_cols]
            colors = sns.color_palette(n_colors=len(unique_features))
            
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
                        ax.plot([window_start, window_end], [value, value], 
                               color=colors[i], linewidth=2, label=col if window_idx == 0 else "")
                        
                        # Add vertical connectors between windows (optional, for continuity)
                        if window_idx > 0 and not pd.isna(feature_values[window_idx-1]):
                            prev_value = feature_values[window_idx-1]
                            ax.plot([window_start, window_start], [prev_value, value], 
                                   color=colors[i], linewidth=1, alpha=0.7)
            
            # Set x-axis label
            ax.set_xlabel('Time (seconds)', fontsize=font_sizes['label'], fontfamily=font_family)
            self.set_labels_ticks_sliding(filter_cols, feature, ax, font_sizes, font_family, display_options)
            
            # Create legend manually for the sliding window plot
            if display_options.get('show_legend', True):
                # Get handles and labels, removing duplicates
                handles, labels = ax.get_legend_handles_labels()
                # Remove empty labels from the vertical connectors
                filtered_handles = []
                filtered_labels = []
                for h, l in zip(handles, labels):
                    if l and l not in filtered_labels:  # Only add non-empty, unique labels
                        filtered_handles.append(h)
                        filtered_labels.append(l)
                
                if filtered_handles:
                    legend = ax.legend(filtered_handles, filtered_labels)
                    for text in legend.get_texts():
                        text.set_fontsize(font_sizes['legend'])
                        text.set_fontfamily(font_family)
        else:
            # Fallback to original behavior if no Window_index column
            plot_data = pd.melt(features_df.reset_index(), id_vars=['Window_index'], value_vars=filter_cols)
            plot_data.columns = ['Window_index', 'Feature', feature]

            self.clear_and_set_fonts(ax, font_family, font_sizes)
            sns.lineplot(x='Window_index', y=feature, hue='Feature', data=plot_data, ax=ax)
            self.set_labels_ticks(filter_cols, feature, ax, font_sizes, font_family, display_options)
            
            # Handle legend display for non-sliding plots
            legend = ax.get_legend()
            if legend:
                if display_options.get('show_legend', True):
                    legend.set_visible(True)
                    # Update legend font
                    for text in legend.get_texts():
                        text.set_fontsize(font_sizes['legend'])
                        text.set_fontfamily(font_family)
                else:
                    legend.remove()

        self.canvas.draw()

    def plot_heatmap(self, features_df, font_sizes, font_family, display_options):
        """
        Plots a heatmap for the transition probabilities.
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        self.clear_and_set_fonts(ax, font_family, font_sizes)

        # Extract unique states from column names
        states = sorted(set(col.split('_')[1] for col in features_df.columns if col.startswith('TP')))

        # Initialize transition matrix
        transition_matrix = np.zeros((len(states), len(states)))

        # Fill transition matrix
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                if f'TP_{from_state}_{to_state}' in features_df.columns:
                    transition_matrix[i, j] = features_df[f'TP_{from_state}_{to_state}'].mean()

        # Plot heatmap and capture the image object for the colorbar
        im = ax.matshow(transition_matrix, cmap="YlGnBu")

        # Add a colorbar to illustrate the colormap scale
        cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=font_sizes['tick'])
        for label in cbar.ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes['tick'])

        # Annotate values with percentage
        for i in range(len(states)):
            for j in range(len(states)):
                if i != j:
                    value = transition_matrix[i, j]
                    ax.text(j, i, f'{100 * value:.2f}%', ha='center', va='center', color='black',
                            fontsize=font_sizes['label'], fontfamily=font_family)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(states)))
        ax.set_yticks(np.arange(len(states)))
        ax.set_xticklabels(states)
        ax.set_yticklabels(states)
        ax.tick_params(axis='both', which='major', labelsize=font_sizes['tick'])

        # Set font family and size for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_sizes['tick'])

        # Set labels and title
        ax.set_xlabel('To', fontsize=font_sizes['label'], fontfamily=font_family)
        ax.set_ylabel('From', fontsize=font_sizes['label'], fontfamily=font_family)

        # Apply display options
        if not display_options.get('show_axes', True):
            ax.axis('off')

        if display_options.get('show_grid', False):
            ax.grid(True, alpha=0.3)

        # Show the plot
        self.canvas.draw()

    def get_group_names(self):
        """
        Returns the names of Group A and Group B.
        """
        return self.ui.group_a_lineedit.text(), self.ui.group_b_lineedit.text()

    @staticmethod
    def clear_and_set_fonts(ax, font_family, font_sizes):
        """
        Clears the plot and sets the font sizes and family.
        """
        ax.clear()
        # Set default font properties for all text elements
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(font_sizes['title'])
            item.set_fontfamily(font_family)

        # Set font for tick labels
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_sizes['tick'])
            item.set_fontfamily(font_family)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plot_group_comparison(self, features_df, selected_feature, group_a_name, group_b_name, font_sizes,
                              colormap, font_family, display_options):
        """
        Plots a comparison of features between two groups.
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(selected_feature)]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=['Filename'], value_vars=filter_cols)
        plot_data.columns = ['Filename', 'Feature', selected_feature]

        listItems_group_a = [self.ui.group_a_files_list.item(x).text() for x in
                             range(self.ui.group_a_files_list.count())]
        listItems_group_b = [self.ui.group_b_files_list.item(x).text() for x in
                             range(self.ui.group_b_files_list.count())]

        group_a_data = plot_data.query(f'Filename in {listItems_group_a}')
        group_a_data['Group'] = group_a_name

        group_b_data = plot_data.query(f'Filename in {listItems_group_b}')
        group_b_data['Group'] = group_b_name

        comparison_data = pd.concat([group_a_data, group_b_data], axis=0).reset_index()

        self.clear_and_set_fonts(ax, font_family, font_sizes)
        color_palette = sns.color_palette(colormap, 2)
        sns.boxplot(x='Feature', y=selected_feature, hue='Group', data=comparison_data, ax=ax, palette=color_palette)
        sns.swarmplot(
            x='Feature', y=selected_feature, hue='Group', data=comparison_data, ax=ax,
            color="black", size=10, marker='o', dodge=True, legend=False
        )
        self.set_labels_ticks(filter_cols, selected_feature, ax, font_sizes, font_family, display_options)

        # Handle legend display
        legend = ax.get_legend()
        if legend:
            if display_options.get('show_legend', True):
                legend.set_visible(True)
                # Update legend font
                for text in legend.get_texts():
                    text.set_fontsize(font_sizes['legend'])
                    text.set_fontfamily(font_family)
            else:
                legend.remove()

        self.canvas.draw()

    def plot_box(self, features_df, feature, font_sizes, colormap, font_family, display_options):
        """
        Plots a box plot for the selected static feature.
        """
        # Reset figure to avoid residual artifacts from previous plots (e.g., colorbars)
        self.figure.clear()
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=['Filename'], value_vars=filter_cols)
        plot_data.columns = ['Filename', 'Feature', feature]

        self.clear_and_set_fonts(ax, font_family, font_sizes)

        # Use a color palette for the box plots based on the number of features
        num_features = len(filter_cols)
        color_palette = sns.color_palette(colormap, num_features)

        sns.boxplot(x='Feature', y=feature, data=plot_data, ax=ax, palette=color_palette)
        sns.swarmplot(x='Feature', y=feature, data=plot_data, ax=ax, color="black", size=10, marker='o')
        self.set_labels_ticks(filter_cols, feature, ax, font_sizes, font_family, display_options)

        # Handle legend display
        legend = ax.get_legend()
        if legend:
            if display_options.get('show_legend', True):
                legend.set_visible(True)
                # Update legend font
                for text in legend.get_texts():
                    text.set_fontsize(font_sizes['legend'])
                    text.set_fontfamily(font_family)
            else:
                legend.remove()

        self.canvas.draw()

    def plot_rof_timeseries(self):
        """Plot average baseline-corrected ROF time-series with 95 % CI across all subjects."""

        # Determine file path
        if not hasattr(self.comet, 'extracted_features_path'):
            print("[ERROR] COMET extracted_features_path not set")
            return

        rof_file = os.path.join(self.comet.extracted_features_path, f"ROF_timeseries{self.comet.export_format}")
        if not os.path.exists(rof_file):
            print(f"[ERROR] ROF time-series file not found: {rof_file}")
            return

        # Load data
        try:
            if self.comet.export_format == '.csv':
                rof_df = pd.read_csv(rof_file)
            elif self.comet.export_format == '.pkl':
                rof_df = pd.read_pickle(rof_file)
            elif self.comet.export_format == '.hdf':
                rof_df = pd.read_hdf(rof_file, key='rof')
            elif self.comet.export_format == '.json':
                rof_df = pd.read_json(rof_file, orient='records', lines=True)
            else:
                print("[ERROR] Unsupported export format for ROF file")
                return
        except Exception as e:
            print(f"[ERROR] Failed to load ROF file: {e}")
            return

        # Filter by selected files if any
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        if selected_files:
            rof_df = rof_df[rof_df['Filename'].isin(selected_files)]

        # Determine microstate columns
        ms_cols = [c for c in rof_df.columns if c.startswith('ROF_')]
        if not ms_cols:
            print("[ERROR] No ROF_* columns found in data")
            return

        # Filter desired time window
        rof_df = rof_df[(rof_df['Time_ms'] >= -100) & (rof_df['Time_ms'] <= 600)].copy()
        if rof_df.empty:
            print("[ERROR] No data in requested time window")
            return

        # Prepare plotting data: group by Time_ms, compute mean and 95% CI across filenames
        grouped = rof_df.groupby('Time_ms')

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
        ax.set_xlabel('Time (ms)', fontsize=font_sizes['label'], fontfamily=font_family)
        ax.set_ylabel('Relative Occurrence Frequency', fontsize=font_sizes['label'], fontfamily=font_family)
        ax.set_title('Average ROF across subjects', fontsize=font_sizes['title'], fontfamily=font_family)

        # Add vertical line at t=0
        ax.axvline(0, color='red', linewidth=2, zorder=5)
        # Add horizontal line at y=0
        ax.axhline(0, color='black', linewidth=2, linestyle='--', zorder=5)

        # Handle legend display
        legend = ax.legend()
        if legend:
            if display_options.get('show_legend', True):
                for text in legend.get_texts():
                    text.set_fontsize(font_sizes['legend'])
                    text.set_fontfamily(font_family)
            else:
                legend.remove()

        # Grid and axes visibility
        if not display_options.get('show_axes', True):
            ax.axis('off')

        if display_options.get('show_grid', False):
            ax.grid(True, alpha=0.3)

        # Update tick label fonts
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(font_sizes['tick'])
            label.set_fontfamily(font_family)

        self.canvas.draw()

        # Highlight ROF button
        self._highlight_button('plot_rof_button')
