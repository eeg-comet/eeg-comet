
import os
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from functions.features_utils.feature_io import FeatureIO
from functions.gui_utils.set_widgets_status import set_widgets_status


class FeatureVisualizationDialog(QDialog):
    def __init__(self, context, parent=None, tbx=None):
        super().__init__(parent)

        self.tbx = tbx
        self.ui = self.load_ui(context)

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.ui.Figure_Layout.addWidget(self.canvas)

        self.setup_window()
        self.bind_events()
        self.feature_visualization_controller()

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

        self.ui.figure_settings_checkbox.clicked.connect(self.feature_visualization_controller)
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
            len(self.ui.all_files_list.selectedItems()) == 1 and "dynamic" in self.feature_mode)

        if self.feature_combo.currentText() == 'TP':
            set_widgets_status(self.ui.plot_heatmap_button, mode='enable')
            set_widgets_status(self.ui.plot_heatmap_button, mode='show')
        else:
            set_widgets_status(self.ui.plot_heatmap_button, mode='disable')
            set_widgets_status(self.ui.plot_heatmap_button, mode='hide')

        figure_settings_widgets = [
            self.ui.font_size_label,
            self.ui.font_size_input,
            self.ui.label_size_label,
            self.ui.label_size_input,
            self.ui.colormap_label,
            self.ui.colormap_combobox,
        ]
        if self.ui.figure_settings_checkbox.isChecked():
            set_widgets_status(figure_settings_widgets, mode='show')
        else:
            set_widgets_status(figure_settings_widgets, mode='hide')

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
        self.ui.plot_label.setText("")

    def get_plot_parameters(self):
        # Get plot parameters from UI components
        fontsize = int(self.ui.font_size_input.text())
        labelsize = int(self.ui.label_size_input.text())
        colormap = self.ui.colormap_combobox.currentText()
        return fontsize, labelsize, colormap

    def show_static_violin_all(self):
        """
        Displays violin plots of all static features.
        """
        selected_feature = self.ui.feature_combo.currentText()
        # Get the selected files from the all_files_list
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        # Load all features
        all_features_df = self.load_features("static")
        # Filter features_df based on selected files
        features_df = all_features_df[all_features_df['Filename'].isin(selected_files)]
        fontsize, labelsize, colormap = self.get_plot_parameters()
        self.plot_violin(features_df, selected_feature, fontsize, labelsize, colormap)

    def show_dynamic_line_all(self):
        """
        Displays line plots of all dynamic features.
        """
        selected_feature = self.ui.feature_combo.currentText()
        selected_file = self.ui.all_files_list.currentItem().text()
        features_df = self.load_features("dynamic").query(f'Filename == "{selected_file}"')
        fontsize, labelsize, colormap = self.get_plot_parameters()
        self.plot_line(features_df, selected_feature, fontsize, labelsize, colormap)

    def show_tp_heatmap(self):
        """
        Displays the heatmap for transition probabilities.
        """
        # Get the selected files from the all_files_list
        selected_files = [item.text() for item in self.ui.all_files_list.selectedItems()]
        # Load all features
        all_features_df = self.load_features("static")
        # Filter features_df based on selected files
        features_df = all_features_df[all_features_df['Filename'].isin(selected_files)]
        self.plot_heatmap(features_df)
        self.ui.plot_label.setText("Transition Probability Heatmap")

    def compare_groups(self):
        """
        Compares selected features between two groups.
        """
        selected_feature = self.ui.feature_combo.currentText()
        group_a_name, group_b_name = self.get_group_names()
        features_df = self.load_features("static")
        fontsize, labelsize, colormap = self.get_plot_parameters()
        self.plot_group_comparison(
            features_df, selected_feature, group_a_name, group_b_name, fontsize, labelsize, colormap
        )

    def load_features(self, mode):
        """
        Loads features from a file and returns them as a DataFrame.
        """
        feature_path = os.path.join(self.extracted_features_path, f"{mode}_features{self.export_format}")
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

    def set_labels_ticks(self, filter_cols, feature, ax, fontsize, labelsize):
        ax.set_xlabel("Microstate", fontsize=fontsize)
        xticklabels = [col.split('_')[-1] for col in filter_cols]
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(self.tbx.feature_list_dictionary[feature], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        self.ui.plot_label.setText(f"{self.tbx.feature_list_dictionary[feature]}")

    def plot_violin(self, features_df, feature, fontsize, labelsize, colormap):
        """
        Plots a violin plot for the selected static feature.
        """
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=['Filename'], value_vars=filter_cols)
        plot_data.columns = ['Filename', 'Feature', feature]

        self.clear_and_set_fonts(ax)

        # Use a color palette for the violins based on the number of features
        num_features = len(filter_cols)
        color_palette = sns.color_palette(colormap, num_features)

        sns.violinplot(x='Feature', y=feature, data=plot_data, ax=ax, palette=color_palette)
        sns.swarmplot(x='Feature', y=feature, data=plot_data, ax=ax, color="white", size=10, marker='o')
        self.set_labels_ticks(filter_cols, feature, ax, fontsize, labelsize)
        self.canvas.draw()

    def plot_line(self, features_df, feature, fontsize, labelsize, colormap):
        """
        Plots a line plot for the selected dynamic feature.
        """
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=['Window_index'], value_vars=filter_cols)
        plot_data.columns = ['Window_index', 'Feature', feature]

        self.clear_and_set_fonts(ax)
        sns.lineplot(x='Window_index', y=feature, hue='Feature', data=plot_data, ax=ax)
        self.set_labels_ticks(filter_cols, feature, ax, fontsize, labelsize)
        self.canvas.draw()

    def plot_heatmap(self, features_df, fontsize, labelsize, colormap):
        """
        Plots a heatmap for the transition probabilities.
        """
        ax = self.canvas.figure.gca()
        self.clear_and_set_fonts(ax)

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
                    ax.text(j, i, f'{100 * value:.2f}%', ha='center', va='center', color='black', fontsize=14)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(states)))
        ax.set_yticks(np.arange(len(states)))
        ax.set_xticklabels(states)
        ax.set_yticklabels(states)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)

        # Set labels and title
        ax.set_xlabel('To', fontsize=fontsize)
        ax.set_ylabel('From', fontsize=fontsize)

        # Show the plot
        self.canvas.draw()

    def get_group_names(self):
        """
        Returns the names of Group A and Group B.
        """
        return self.ui.group_a_lineedit.text(), self.ui.group_b_lineedit.text()

    @staticmethod
    def clear_and_set_fonts(ax):
        """
        Clears the plot and sets the font sizes.
        """
        ax.clear()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

    def plot_group_comparison(self, features_df, selected_feature, group_a_name, group_b_name, fontsize, labelsize,
                              colormap):
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

        self.clear_and_set_fonts(ax)
        color_palette = sns.color_palette(colormap, 2)
        sns.violinplot(x='Feature', y=selected_feature, hue='Group', data=comparison_data, ax=ax, palette=color_palette)
        sns.swarmplot(
            x='Feature', y=selected_feature, hue='Group', data=comparison_data, ax=ax,
            color="white", size=10, marker='o', dodge=True, legend=False
        )
        self.set_labels_ticks(filter_cols, selected_feature, ax, fontsize, labelsize)
        self.canvas.draw()
