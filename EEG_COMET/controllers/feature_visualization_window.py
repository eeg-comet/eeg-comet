import os
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QActionGroup
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from features_utils.feature_io import FeatureIO
from gui_utils.set_widgets_status import set_widgets_status


class FeatureVisualizationWindow(QMainWindow):
    def __init__(self, context, parent=None, tbx=None):
        super().__init__(parent)

        self.comet = tbx
        self.ui = self.load_ui(context)

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas.setMinimumHeight(100)
        self.ui.Figure_Layout.addWidget(self.canvas)

        self.setup_window()
        self.setup_style_actions()  # Add this line
        self.bind_events()
        self.feature_visualization_controller()

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

    def change_font_family(self, font_name):
        """
        Handle font family change from menu actions
        """
        self.current_font_family = font_name

    def change_display_option(self, option_name, checked):
        """
        Handle display option change from menu actions
        """
        # print(f"Display option '{option_name}' set to: {checked}")

    def change_font_size_category(self, category, font_size):
        """
        Handle font size change for specific category from menu actions
        """

        # Update the UI input fields if they exist
        if category == 'title' and hasattr(self.ui, 'font_size_input'):
            self.ui.font_size_input.setText(str(font_size))
        elif category == 'tick' and hasattr(self.ui, 'label_size_input'):
            self.ui.label_size_input.setText(str(font_size))

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
        self.resize(1000, 800)

    def bind_events(self):
        self.ui.plot_all_static_button.clicked.connect(self.show_static_violin_all)
        self.ui.plot_all_dynamic_button.clicked.connect(self.show_dynamic_line_all)

        self.ui.plot_heatmap_button.clicked.connect(self.show_tp_heatmap)
        self.ui.plot_groups_button.clicked.connect(self.compare_groups)
        self.ui.add_group_a_button.clicked.connect(self.move_file_from_all_to_a)
        self.ui.add_group_b_button.clicked.connect(self.move_file_from_all_to_b)
        self.ui.remove_group_a_button.clicked.connect(self.move_file_from_a_to_all)
        self.ui.remove_group_b_button.clicked.connect(self.move_file_from_b_to_all)
        self.ui.reset_groups_button.clicked.connect(self.reset_groups)

        self.ui.feature_combo.currentTextChanged.connect(self.feature_visualization_controller)
        self.ui.all_files_list.itemSelectionChanged.connect(self.feature_visualization_controller)
        buttons = [
            self.ui.compare_groups_checkbox,
            self.ui.add_group_a_button,
            self.ui.add_group_b_button,
            self.ui.remove_group_a_button,
            self.ui.remove_group_b_button,
            self.ui.reset_groups_button
        ]
        for button in buttons:
            button.clicked.connect(self.feature_visualization_controller)

    def feature_visualization_controller(self):
        """
        Control the behavior of the Feature Visualization window based on user selections.
        """

        self.ui.plot_all_static_button.setEnabled(not len(self.ui.all_files_list.selectedItems()) == 0)
        self.ui.plot_all_dynamic_button.setEnabled(
            len(self.ui.all_files_list.selectedItems()) == 1 and "sliding" in self.feature_mode)

        if self.feature_combo.currentText() == 'TP':
            set_widgets_status(self.ui.plot_heatmap_button, mode='enable')
            set_widgets_status(self.ui.plot_heatmap_button, mode='show')
        else:
            set_widgets_status(self.ui.plot_heatmap_button, mode='disable')
            set_widgets_status(self.ui.plot_heatmap_button, mode='hide')

        compare_groups_widgets = [
            self.ui.add_group_a_button,
            self.ui.group_a_lineedit,
            self.ui.remove_group_a_button,
            self.ui.group_a_files_list,
            self.ui.add_group_b_button,
            self.ui.group_b_lineedit,
            self.ui.remove_group_b_button,
            self.ui.group_b_files_list,
            self.ui.reset_groups_button,
            self.ui.plot_groups_button
        ]
        if self.ui.compare_groups_checkbox.isChecked():
            set_widgets_status(compare_groups_widgets, mode='enable')
            set_widgets_status(compare_groups_widgets, mode='show')
            count_group_a = self.ui.group_a_files_list.count()
            count_group_b = self.ui.group_b_files_list.count()
            self.ui.add_group_a_button.setEnabled(not len(self.ui.all_files_list.selectedItems()) == 0)
            self.ui.add_group_b_button.setEnabled(not len(self.ui.all_files_list.selectedItems()) == 0)
            self.ui.remove_group_a_button.setEnabled(not count_group_a == 0)
            self.ui.remove_group_b_button.setEnabled(not count_group_b == 0)
            self.ui.reset_groups_button.setEnabled(count_group_a > 0 or count_group_b > 0)
            self.ui.plot_groups_button.setEnabled(count_group_a > 0 and count_group_b > 0)
        else:
            set_widgets_status(compare_groups_widgets, mode='disable')
            set_widgets_status(compare_groups_widgets, mode='hide')

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
        selected_feature = self.ui.feature_combo.currentText()
        # Get the selected files from the all_files_list
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        # Load all features
        all_features_df = self.load_features("averaged")
        # Filter features_df based on selected files
        features_df = all_features_df[all_features_df['Filename'].isin(selected_files)]
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_violin(features_df, selected_feature, font_sizes, colormap, font_family, display_options)

    def show_dynamic_line_all(self):
        """
        Displays line plots of all dynamic features.
        """
        selected_feature = self.ui.feature_combo.currentText()
        selected_file = self.ui.all_files_list.currentItem().text()
        features_df = self.load_features("sliding").query(f'Filename == "{selected_file}"')
        font_sizes, colormap, font_family, display_options = self.get_plot_parameters()
        self.plot_line(features_df, selected_feature, font_sizes, colormap, font_family, display_options)

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

        # Set figure title
        self.figure.suptitle("Transition Probability Heatmap",
                             fontsize=font_sizes['title'], fontfamily=font_family)

    def compare_groups(self):
        """
        Compares selected features between two groups.
        """
        selected_feature = self.ui.feature_combo.currentText()
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
        ax.set_xlabel("Microstate", fontsize=font_sizes['label'], fontfamily=font_family)
        xticklabels = [col.split('_')[-1] for col in filter_cols]
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

    def plot_violin(self, features_df, feature, font_sizes, colormap, font_family, display_options):
        """
        Plots a violin plot for the selected static feature.
        """
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
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=['Window_index'], value_vars=filter_cols)
        plot_data.columns = ['Window_index', 'Feature', feature]

        self.clear_and_set_fonts(ax, font_family, font_sizes)
        sns.lineplot(x='Window_index', y=feature, hue='Feature', data=plot_data, ax=ax)
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

    def plot_heatmap(self, features_df, font_sizes, font_family, display_options):
        """
        Plots a heatmap for the transition probabilities.
        """
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

        # Plot heatmap colormap
        ax.matshow(transition_matrix, cmap="YlGnBu")

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

    def plot_group_comparison(self, features_df, selected_feature, group_a_name, group_b_name, font_sizes,
                              colormap, font_family, display_options):
        """
        Plots a comparison of features between two groups.
        """
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
        sns.violinplot(x='Feature', y=selected_feature, hue='Group', data=comparison_data, ax=ax, palette=color_palette)
        sns.swarmplot(
            x='Feature', y=selected_feature, hue='Group', data=comparison_data, ax=ax,
            color="white", size=10, marker='o', dodge=True, legend=False
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
