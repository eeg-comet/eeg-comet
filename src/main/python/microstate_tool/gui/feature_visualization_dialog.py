
import os
import pandas as pd
import seaborn as sns
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from functions.features_utils.feature_io import FeatureIO


class FeatureVisualizationDialog(QDialog):
    def __init__(self, context, parent=None, tbx=None):
        super().__init__(parent)

        self.tbx = tbx
        self.ui = self.load_ui(context)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.ui.Figure_Layout.addWidget(self.canvas)

        self.setup_window()
        self.bind_events()

    def load_ui(self, context):
        basepath = os.path.dirname(__file__)
        return uic.loadUi(context.get_resource("FeatureVisualizationWindow.ui"), self)

    def setup_window(self):
        self.setWindowTitle("Visualization of the extracted features")
        self.resize(1000, 800)

    def bind_events(self):
        self.ui.plot_all_static_button.clicked.connect(self.show_static_violin_all)
        self.ui.plot_all_dynamic_button.clicked.connect(self.show_dynamic_line_all)
        self.ui.plot_groups_button.clicked.connect(self.compare_groups)
        self.ui.add_group_a_button.clicked.connect(self.move_file_from_all_to_a)
        self.ui.add_group_b_button.clicked.connect(self.move_file_from_all_to_b)
        self.ui.remove_group_a_button.clicked.connect(self.move_file_from_a_to_all)
        self.ui.remove_group_b_button.clicked.connect(self.move_file_from_b_to_all)
        self.ui.reset_groups_button.clicked.connect(self.reset_groups)

    def move_file_between_lists(self, source_list, target_list):
        if source_list.currentItem():
            filename = source_list.currentItem().text()
            selected_items = source_list.selectedItems()
            if not selected_items:
                return

            for item in selected_items:
                source_list.takeItem(source_list.row(item))

            target_list.addItem(filename)

    def move_file_from_all_to_a(self):
        self.move_file_between_lists(self.ui.all_files_list, self.ui.group_a_files_list)

    def move_file_from_all_to_b(self):
        self.move_file_between_lists(self.ui.all_files_list, self.ui.group_b_files_list)

    def move_file_from_a_to_all(self):
        self.move_file_between_lists(self.ui.group_a_files_list, self.ui.all_files_list)

    def move_file_from_b_to_all(self):
        self.move_file_between_lists(self.ui.group_b_files_list, self.ui.all_files_list)

    def reset_groups(self):
        """Clears all file lists and resets the plot."""
        self.clear_all_lists()
        self.clear_plot()

    def show_static_violin_all(self):
        """Displays violin plots of all static features."""
        selected_feature = self.ui.feature_combo.currentText()
        selected_mode = self.get_selected_mode()
        features_df = self.load_features(selected_mode)

        self.plot_violin(features_df, selected_feature, selected_mode)
        self.ui.plot_label.setText(f"Static Feature: {selected_feature}")

    def show_dynamic_line_all(self):
        """Displays line plots of all dynamic features."""
        selected_feature = self.ui.feature_combo.currentText()
        selected_file = self.ui.all_files_list.currentItem().text()
        features_df = self.load_features("dynamic").query(f'Filename == "{selected_file}"')

        self.plot_line(features_df, selected_feature)
        self.ui.plot_label.setText(f"Filename: {selected_file}- Dynamic Feature: {selected_feature}")

    def compare_groups(self):
        """Compares selected features between two groups."""
        selected_feature = self.ui.feature_combo.currentText()
        group_a_name, group_b_name = self.get_group_names()
        features_df = self.load_features("static")

        self.plot_group_comparison(features_df, selected_feature, group_a_name, group_b_name)

    def get_selected_mode(self):
        """Returns the selected mode: static or dynamic."""
        return "static" if self.ui.static_mode_radio.isChecked() else "dynamic"

    def load_features(self, mode):
        """Loads features from a file and returns them as a DataFrame."""
        feature_path = os.path.join(self.extracted_features_path, f"{mode}_features{self.export_format}")
        return FeatureIO().import_features(feature_path, self.export_format)

    def clear_all_lists(self):
        """Clears all file lists."""
        self.ui.all_files_list.clear()
        self.ui.group_a_files_list.clear()
        self.ui.group_b_files_list.clear()
        self.populate_all_files_list()

    def clear_plot(self):
        """Clears the current plot."""
        self.canvas.draw()

    def populate_all_files_list(self):
        """Populates the all_files_list with EEG file names."""
        for eeg_file in self.list_eegs:
            self.ui.all_files_list.addItem(str(eeg_file))

    def plot_violin(self, features_df, feature, mode):
        """Plots a violin plot for the selected static feature."""
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=['Filename'], value_vars=filter_cols)
        plot_data.columns = ['Filename', 'Feature', feature]

        self.clear_and_set_fonts(ax)
        sns.violinplot(x='Feature', y=feature, data=plot_data, kind="violin", ax=ax)
        self.canvas.draw()

    def plot_line(self, features_df, feature):
        """Plots a line plot for the selected dynamic feature."""
        ax = self.canvas.figure.gca()
        filter_cols = [col for col in features_df if col.startswith(feature)]
        filter_cols.sort()

        plot_data = pd.melt(features_df.reset_index(), id_vars=['Window_index'], value_vars=filter_cols)
        plot_data.columns = ['Window_index', 'Feature', feature]

        self.clear_and_set_fonts(ax)
        sns.lineplot(x='Window_index', y=feature, hue='Feature', data=plot_data, ax=ax)
        self.canvas.draw()

    def get_group_names(self):
        """Returns the names of Group A and Group B."""
        return self.ui.group_a_lineedit.text(), self.ui.group_b_lineedit.text()

    def clear_and_set_fonts(self, ax):
        """Clears the plot and sets the font sizes."""
        ax.clear()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)

    def plot_group_comparison(self, features_df, selected_feature, group_a_name, group_b_name):
        """Plots a comparison of features between two groups."""
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
        sns.violinplot(x='Feature', y=selected_feature, hue='Group', data=comparison_data, ax=ax)
        self.canvas.draw()
