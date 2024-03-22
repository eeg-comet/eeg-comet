
import os.path
import pickle
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from gui_utils.set_widgets_status import set_widgets_status
from clustering_utils.microstate_visualizer import show_microstate
from features_utils.feature_io import FeatureIO


class CompareStudiesWindow(QDialog):
    def __init__(self, context, parent=None):
        super(CompareStudiesWindow, self).__init__(parent)

        self.study1_loaded = False
        self.study2_loaded = False
        self.setup_ui(context)
        self.connect_ui()
        self.create_figure_and_canvas()
        self.update_ui()

    def setup_ui(self, context):
        """Setup UI components"""
        self.ui = uic.loadUi(context.get_resource("CompareStudiesWindow.ui"), self)
        self.ui.setWindowTitle("Comparison of two EEG-COMET studies")
        # Set window flags to include the maximize button
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

    def update_ui(self):
        """Updates the user interface elements based on the loaded studies and their information."""
        self.ui.feature_combo.clear()
        self.ui.plot_label.clear()
        self.ui.correlations_textedit.clear()
        self.ui.stats_textedit.clear()

        if self.study1_loaded:
            self.ui.study1_name_label.setText(self.comet_tbx_study1.study_name)
            self.feature_list_dictionary = self.comet_tbx_study1.feature_list_dictionary

            study2_widgets = [self.ui.load_study2_button, self.ui.study2_name_label, self.ui.study2_file_list,
                              self.ui.correlations_textedit, self.canvas_microstates_study2]
            if self.ui.compare_study2_radio.isChecked():
                set_widgets_status(study2_widgets, mode='enable')
                set_widgets_status(study2_widgets, mode='show')

                if self.study2_loaded:
                    self.study2_name = self.comet_tbx_study2.study_name
                    self.ui.study2_name_label.setText(self.study2_name)
                else:
                    self.ui.study2_name_label.setText("Study 2")

                if self.study1_loaded and self.study2_loaded:
                    set_widgets_status(self.ui.plot_features_button, mode='enable')
                    common_features = [feat for feat in self.comet_tbx_study1.feature_list
                                       if feat in self.comet_tbx_study2.feature_list]
                    self.ui.feature_combo.clear()
                    self.ui.feature_combo.addItems(list(common_features))
                    self.update_corr_stats()
                else:
                    set_widgets_status(self.ui.plot_features_button, mode='disable')

            else:
                set_widgets_status(study2_widgets, mode='disable')
                set_widgets_status(study2_widgets, mode='hide')

                set_widgets_status(self.ui.plot_features_button, mode='enable')
                features = self.comet_tbx_study1.feature_list
                self.ui.feature_combo.clear()
                self.ui.feature_combo.addItems([i for i in features])
                if self.ui.compare_surrogate_radio.isChecked():
                    self.ui.study2_name_label.setText("Surrogate Study")
                    self.synthetic_type = 'surrogate'
                elif self.ui.compare_random_radio.isChecked():
                    self.ui.study2_name_label.setText("Random Study")
                    self.synthetic_type = 'random'
                self.study2_name = self.synthetic_type

    def update_plot_label(self):
        """Updates the plot label based on the selected feature."""
        if selected_feature := self.ui.feature_combo.currentText():
            self.ui.plot_label.setText(f"{self.feature_list_dictionary[selected_feature]}")

    def connect_ui(self):
        """Connects UI signals to corresponding slots (functions) for event handling."""
        self.ui.load_study1_button.clicked.connect(self.load_study1)
        self.ui.load_study2_button.clicked.connect(self.load_study2)
        compare_widgets = [self.ui.compare_study2_radio, self.ui.compare_surrogate_radio, self.ui.compare_random_radio]
        for item in compare_widgets:
            item.clicked.connect(self.update_ui)
        self.ui.feature_combo.currentTextChanged.connect(self.update_plot_label)
        if self.study2_loaded:
            self.ui.plot_features_button.clicked.connect(self.update_corr_stats)
        self.ui.plot_features_button.clicked.connect(self.plot_features)
        self.ui.plot_features_button.clicked.connect(self.update_feature_stats)

    def create_figure_and_canvas(self):
        """Creates matplotlib figures and canvases for visualization."""
        self.figure_microstates_study1 = Figure(tight_layout=True)
        self.canvas_microstates_study1 = FigureCanvasQTAgg(self.figure_microstates_study1)
        self.canvas_microstates_study1.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas_microstates_study1.setMinimumHeight(100)
        self.Figure_Microstates_Study1_Layout.addWidget(self.canvas_microstates_study1)
        self.figure_microstates_study2 = Figure(tight_layout=True)
        self.canvas_microstates_study2 = FigureCanvasQTAgg(self.figure_microstates_study2)
        self.canvas_microstates_study2.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas_microstates_study2.setMinimumHeight(100)
        self.Figure_Microstates_Study2_Layout.addWidget(self.canvas_microstates_study2)
        self.figure_features = Figure(tight_layout=True)
        self.canvas_features = FigureCanvasQTAgg(self.figure_features)
        self.canvas_features.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas_features.setMinimumHeight(100)
        self.Figure_Features_Layout.addWidget(self.canvas_features)

    @staticmethod
    def update_study_info(tbx, listwidget):
        """Updates the study information in a list widget."""
        for i in range(len(tbx.list_eegs)):
            listwidget.addItem(str(tbx.list_eegs[i]))

    def update_corr_stats(self):
        """Updates the correlation statistics in the UI based on microstate analysis."""
        self.ui.correlations_textedit.clear()
        # Update the correlation coefficients stats
        microlabels, correlation_coefficients, p_values, adjusted_p_values = self.spatial_correlation_analysis()
        correlations_text = "Correlation of Microstates Between Studies:\n"
        for i, val in enumerate(correlation_coefficients):
            correlations_text += f"Microstate {microlabels[i]}: {100 * abs(val):.3f} %\n"
        correlations_text += f"\nP-values:\n"
        for i, val in enumerate(p_values):
            correlations_text += f"Microstate {microlabels[i]}: {val}\n"
        correlations_text += f"\n{self.ui.multiple_test_method_combo.currentText()} Adjusted P-values:\n"
        for i, val in enumerate(adjusted_p_values):
            correlations_text += f"Microstate {microlabels[i]}: {val}\n"
        self.ui.correlations_textedit.appendPlainText(correlations_text)

    def update_feature_stats(self):
        """Updates the feature statistics in the UI based on feature comparison."""
        self.ui.stats_textedit.clear()
        feature_list, t_test_results, p_values, adjusted_p_values = self.feature_comparison_analysis()
        # Display t-test results
        for i, (feat, t_statistic) in enumerate(t_test_results.items()):
            adjusted_p_value = adjusted_p_values[i]
            result_text = f"Feature Name: {feat}:\n"
            result_text += f"T-statistic: {t_statistic}, p-value: {p_values[i]}, " \
                           f"{self.ui.multiple_test_method_combo.currentText()} Adjusted p-value: {adjusted_p_value}\n"
            self.ui.stats_textedit.appendPlainText(result_text)

    def load_study1(self):
        """Loads the first EEG-COMET study."""
        self.study1_path = QFileDialog.getExistingDirectory(
            self, "Select the folder containing an EEG-COMET study.")
        # Check if the eeg_comet_parameters.pkl file exists
        study1_comet_path = os.path.join(self.study1_path, 'eeg_comet_parameters.pkl')
        # Handle the case where loading the study fails
        if not os.path.exists(study1_comet_path):
            QMessageBox.information(self, "Load error",
                                    "The selected folder does not contain a valid study!",
                                    QMessageBox.Ok)
            return
        else:
            # Load the COMET object from the pickle file
            with open(study1_comet_path, 'rb') as input_tbx:
                self.comet_tbx_study1 = pickle.load(input_tbx)
            self.study1_loaded = True
            self.update_study_info(self.comet_tbx_study1, self.ui.study1_file_list)
            self.plot_maps(self.comet_tbx_study1, self.figure_microstates_study1, self.canvas_microstates_study1)
            self.update_ui()

    def load_study2(self):
        """Loads the second EEG-COMET study."""
        self.study2_path = QFileDialog.getExistingDirectory(
            self, "Select the folder containing an EEG-COMET study.")
        # Check if the eeg_comet_parameters.pkl file exists
        study2_comet_path = os.path.join(self.study2_path, 'eeg_comet_parameters.pkl')
        # Handle the case where loading the study fails
        if not os.path.exists(study2_comet_path):
            QMessageBox.information(self, "Load error",
                                    "The selected folder does not contain a valid study!",
                                    QMessageBox.Ok)
            return
        else:
            # Load the COMET object from the pickle file
            with open(study2_comet_path, 'rb') as input_tbx:
                self.comet_tbx_study2 = pickle.load(input_tbx)
            self.study2_loaded = True
            self.update_study_info(self.comet_tbx_study2, self.ui.study2_file_list)
            self.plot_maps(self.comet_tbx_study2, self.figure_microstates_study2, self.canvas_microstates_study2)
            self.update_ui()

    @staticmethod
    def organize_data2plot(common_features_df, selected_feature):
        columns_to_keep = ['Filename', 'Study'] + \
                          [col for col in common_features_df.columns if col.startswith(selected_feature)]
        common_features_df = common_features_df[columns_to_keep]
        feature_list = common_features_df.columns.tolist()
        feature_list = [col for col in feature_list if col not in ['Study', 'Filename']]

        plot_data = pd.melt(common_features_df.reset_index(), id_vars=['Filename', 'Study'], value_vars=feature_list)
        plot_data.columns = ['Filename', 'Study', 'Feature', selected_feature]
        return plot_data, feature_list

    @staticmethod
    def plot_microstates_with_labels(microstate, micro_label, eeg_info, ax):
        """Plots microstates with corresponding labels on a given axis."""

        # Plot the microstate
        show_microstate(microstate, eeg_info, ax)
        # Axis settings
        ax.axis('off')
        ax.text(0.5, -0.2, micro_label.upper(), transform=ax.transAxes,
                fontsize=16, ha='center', va='center')
        ax.text(0, 0, '', transform=ax.transAxes)

    def plot_maps(self, tbx, figure, canvas):
        """Plots microstate maps on the canvas."""
        # Clear existing axes
        for ax in figure.get_axes():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Create subplots for each microstate
        axs = [figure.add_subplot(1, len(tbx.micro_labels), idx + 1) for idx in range(len(tbx.micro_labels))]

        for idx, ax_idx in enumerate(range(len(tbx.micro_labels))):
            # Plot the microstate with the corresponding label
            self.plot_microstates_with_labels(
                tbx.best_maps[ax_idx, :], tbx.micro_labels[idx], tbx.eeg_info, axs[idx]
            )

        # Adjust layout to prevent overlapping text
        figure.tight_layout()

        # Draw the canvas after plotting
        canvas.draw()

    @staticmethod
    def clear_and_set_fonts(ax):
        """Clears the plot and sets the font sizes."""
        ax.clear()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)

    def set_labels_ticks(self, ax, filter_cols, feature):
        """Sets labels and ticks for the plot based on selected features."""
        ax.set_xlabel("Microstate")
        xticklabels = ['_'.join(col.split('_')[1:]) for col in filter_cols]
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(self.feature_list_dictionary[feature])
        self.ui.plot_label.setText(f"{self.feature_list_dictionary[feature]}")

    @staticmethod
    def load_features(tbx, feature_type):
        """Loads features from a file and returns them as a DataFrame."""
        feature_path = os.path.join(
            tbx.extracted_features_path, f"{feature_type}_static_features{tbx.export_format}"
        )
        return FeatureIO().import_features(feature_path, tbx.export_format)

    def get_common_features(self):
        """Retrieves common features between the two loaded studies."""
        features_df_study1 = self.load_features(self.comet_tbx_study1, 'real')
        if self.ui.compare_study2_radio.isChecked():
            features_df_study2 = self.load_features(self.comet_tbx_study2, 'real')
        else:
            features_df_study2 = self.load_features(self.comet_tbx_study1, self.synthetic_type)
        common_columns = [col for col in features_df_study1.columns if col in features_df_study2.columns]
        common_columns.sort()

        # Create DataFrame with 'Study' column and values 'study1' and 'study2'
        study1_df = features_df_study1[common_columns].copy()
        study1_df['Study'] = self.comet_tbx_study1.study_name

        study2_df = features_df_study2[common_columns].copy()
        study2_df['Study'] = self.study2_name

        # Concatenate DataFrames along with the new 'Study' column
        common_features_df = pd.concat([study1_df, study2_df], ignore_index=True)

        return common_features_df, common_columns

    def plot_features(self):
        """Plots a violin plot for the selected static feature."""
        selected_feature = self.ui.feature_combo.currentText()
        common_features_df, common_columns = self.get_common_features()
        plot_data, feature_list = self.organize_data2plot(common_features_df, selected_feature)

        ax = self.canvas_features.figure.gca()
        self.clear_and_set_fonts(ax)
        sns.violinplot(
            x='Feature', y=selected_feature, hue='Study', data=plot_data, ax=ax,
            hue_order=[self.comet_tbx_study1.study_name, self.study2_name]
        )
        sns.swarmplot(
            x='Feature', y=selected_feature, hue='Study', data=plot_data, ax=ax,
            color="white", size=10, marker='o', dodge=True, legend=False
        )
        self.set_labels_ticks(ax, feature_list, selected_feature)
        self.canvas_features.draw()

    def spatial_correlation_analysis(self):
        """Performs spatial correlation analysis between microstates of two studies."""
        # Perform Pearson correlation test
        microstates_study1 = self.comet_tbx_study1.best_maps
        microstates_study2 = self.comet_tbx_study2.best_maps

        microlabels_study1 = self.comet_tbx_study1.micro_labels
        microlabels_study2 = self.comet_tbx_study2.micro_labels
        microlabels = [label for label in microlabels_study1 if label in microlabels_study2]

        min_states = min(microstates_study1.shape[0], microstates_study2.shape[0])

        correlation_coefficients, p_values = [], []

        for m in range(min_states):  # Iterate over each column
            corr_coef, p_val = pearsonr(microstates_study1[m, :], microstates_study2[m, :])
            correlation_coefficients.append(corr_coef)
            p_values.append(p_val)

        # p-value Correction for Multiple Tests
        adjusted_p_values = multipletests(p_values, method=self.ui.multiple_test_method_combo.currentText().lower())[1]

        return microlabels, correlation_coefficients, p_values, adjusted_p_values

    def feature_comparison_analysis(self):
        """Performs feature comparison analysis between two studies using t-tests."""
        common_features_df, common_columns = self.get_common_features()
        selected_feature = self.ui.feature_combo.currentText()
        columns_to_keep = ['Filename', 'Study'] +\
                          [col for col in common_features_df.columns if col.startswith(selected_feature)]
        common_features_df = common_features_df[columns_to_keep]
        # Filter DataFrames based on selected feature and study
        selected_feature_study1_df = common_features_df[common_features_df['Study'] == self.comet_tbx_study1.study_name]
        selected_feature_study1_df = selected_feature_study1_df.sort_values(by='Filename')

        selected_feature_study2_df = common_features_df[common_features_df['Study'] == self.study2_name]
        selected_feature_study2_df = selected_feature_study2_df.sort_values(by='Filename')

        # Perform t-test for each column
        feature_list = selected_feature_study1_df.columns.tolist()
        feature_list = [col for col in feature_list if col not in ['Study', 'Filename']]
        t_test_results = {}
        p_values = []

        for feat in feature_list:
            study1_feature_values = selected_feature_study1_df[feat].tolist()
            study2_feature_values = selected_feature_study2_df[feat].tolist()
            # Check if the t-test should be paired or unpaired
            if self.ui.paired_test_checkbox.isChecked():
                t_statistic, p_value = ttest_rel(study1_feature_values, study2_feature_values)
            else:
                t_statistic, p_value = ttest_ind(study1_feature_values, study2_feature_values)

            p_values.append(p_value)
            t_test_results[feat] = t_statistic

        # p-value Correction for Multiple Tests
        self.ui.multiple_test_method_combo.currentText().lower()
        adjusted_p_values = multipletests(p_values, method=self.ui.multiple_test_method_combo.currentText().lower())[1]

        return feature_list, t_test_results, p_values, adjusted_p_values
