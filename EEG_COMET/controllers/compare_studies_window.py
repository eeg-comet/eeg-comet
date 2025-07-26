import os.path
import numpy as np
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
from comet import COMET


class CompareStudiesWindow(QDialog):
    def __init__(self, context, parent=None):
        """
        Initialize the CompareStudiesWindow.
        """
        super(CompareStudiesWindow, self).__init__(parent)

        # Flags to check if studies are loaded
        self.study1_loaded = False
        self.study2_loaded = False

        # Setup UI components and connections
        self.setup_ui(context)
        self.connect_ui()
        self.create_figure_and_canvas()
        self.update_ui()

    def setup_ui(self, context):
        """Setup UI components."""
        self.ui = uic.loadUi(context.get_resource("CompareStudiesWindow.ui"), self)
        self.ui.setWindowTitle("Comparison of two EEG-COMET studies")
        # Include the maximize button in window flags
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

    def connect_ui(self):
        """
        Connect UI signals to corresponding slots for event handling.
        """
        self.ui.load_study1_button.clicked.connect(self.load_study1)
        self.ui.load_study2_button.clicked.connect(self.load_study2)

        # Connect radio buttons to update UI
        compare_widgets = [
            self.ui.compare_study2_radio,
            self.ui.compare_surrogate_radio,
            self.ui.compare_random_radio
        ]
        for item in compare_widgets:
            item.clicked.connect(self.update_ui)

        # Connect feature selection to plot label update
        self.ui.feature_combo.currentTextChanged.connect(self.update_plot_label)

        # Connect plot features button to statistical analyses and plotting
        self.ui.plot_features_button.clicked.connect(self.update_corr_stats)
        self.ui.plot_features_button.clicked.connect(self.plot_features)
        self.ui.plot_features_button.clicked.connect(self.update_feature_stats)

        # Connect test-retest button to ICC analysis
        self.ui.plot_testretest_button.clicked.connect(self.perform_test_retest_analysis)

    def create_figure_and_canvas(self):
        """Creates matplotlib figures and canvases for visualization."""
        # Microstates Study 1
        self.figure_microstates_study1 = Figure(tight_layout=True)
        self.canvas_microstates_study1 = FigureCanvasQTAgg(self.figure_microstates_study1)
        self.canvas_microstates_study1.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas_microstates_study1.setMinimumHeight(100)
        self.ui.Figure_Microstates_Study1_Layout.addWidget(self.canvas_microstates_study1)

        # Microstates Study 2
        self.figure_microstates_study2 = Figure(tight_layout=True)
        self.canvas_microstates_study2 = FigureCanvasQTAgg(self.figure_microstates_study2)
        self.canvas_microstates_study2.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas_microstates_study2.setMinimumHeight(100)
        self.ui.Figure_Microstates_Study2_Layout.addWidget(self.canvas_microstates_study2)

        # Features Plot
        self.figure_features = Figure(tight_layout=True)
        self.canvas_features = FigureCanvasQTAgg(self.figure_features)
        self.canvas_features.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas_features.setMinimumHeight(100)
        self.ui.Figure_Features_Layout.addWidget(self.canvas_features)

    def update_ui(self):
        """
        Updates the user interface elements based on the loaded studies and their information.
        """
        # Clear previous selections and texts
        self.ui.feature_combo.clear()
        self.ui.plot_label.clear()
        self.ui.correlations_textedit.clear()
        self.ui.stats_textedit.clear()

        if self.study1_loaded:
            # Display Study 1 information
            self.ui.study1_name_label.setText(self.comet_tbx_study1.study_name)
            self.feature_list_dictionary = self.comet_tbx_study1.feature_list_dictionary

            # Widgets related to Study 2
            study2_widgets = [
                self.ui.load_study2_button,
                self.ui.study2_name_label,
                self.ui.study2_file_list,
                self.ui.correlations_textedit,
                self.canvas_microstates_study2
            ]

            if self.ui.compare_study2_radio.isChecked():
                # Enable and show Study 2 widgets
                set_widgets_status(study2_widgets, mode='enable')
                set_widgets_status(study2_widgets, mode='show')

                if self.study2_loaded:
                    self.study2_name = self.comet_tbx_study2.study_name
                    self.ui.study2_name_label.setText(self.study2_name)
                else:
                    self.ui.study2_name_label.setText("Study 2")

                if self.study1_loaded and self.study2_loaded:
                    # Enable plot features button if both studies are loaded
                    set_widgets_status(self.ui.plot_features_button, mode='enable')

                    # Check if both studies have the same number of samples for test-retest
                    if len(self.comet_tbx_study1.list_eegs) == len(self.comet_tbx_study2.list_eegs):
                        set_widgets_status(self.ui.plot_testretest_button, mode='enable')
                    else:
                        set_widgets_status(self.ui.plot_testretest_button, mode='disable')

                    # Populate common features using robust discovery
                    feature_type, feature_mode = self.get_compatible_features()
                    if feature_type and feature_mode:
                        common_features = [
                            feat for feat in self.comet_tbx_study1.feature_list
                            if feat in self.comet_tbx_study2.feature_list
                        ]
                        self.ui.feature_combo.clear()
                        # Use full feature names from dictionary instead of short codes
                        full_feature_names = [
                            self.feature_list_dictionary.get(feat, feat) for feat in common_features
                        ]
                        self.ui.feature_combo.addItems(full_feature_names)
                        self.update_corr_stats()
                    else:
                        self.ui.feature_combo.clear()
                        self.ui.feature_combo.addItem("No compatible features found")
                        set_widgets_status(self.ui.plot_features_button, mode='disable')
                else:
                    set_widgets_status(self.ui.plot_features_button, mode='disable')
                    set_widgets_status(self.ui.plot_testretest_button, mode='disable')

            else:
                # Disable and hide Study 2 widgets
                set_widgets_status(study2_widgets, mode='disable')
                set_widgets_status(study2_widgets, mode='hide')
                set_widgets_status(self.ui.plot_testretest_button, mode='disable')

                # Set Study 2 name based on selected comparison type
                if self.ui.compare_surrogate_radio.isChecked():
                    self.ui.study2_name_label.setText("Surrogate Study")
                    self.synthetic_type = 'surrogate'
                elif self.ui.compare_random_radio.isChecked():
                    self.ui.study2_name_label.setText("Random Study")
                    self.synthetic_type = 'random'
                self.study2_name = self.synthetic_type

                # Check if synthetic features are available
                feature_type, feature_mode = self.get_compatible_features()
                if feature_type and feature_mode:
                    # Enable plot features button and populate all features from Study 1
                    set_widgets_status(self.ui.plot_features_button, mode='enable')
                    features = self.comet_tbx_study1.feature_list
                    self.ui.feature_combo.clear()
                    # Use full feature names from dictionary instead of short codes
                    full_feature_names = [
                        self.feature_list_dictionary.get(feat, feat) for feat in features
                    ]
                    self.ui.feature_combo.addItems(full_feature_names)
                else:
                    # Disable if no compatible synthetic features
                    self.ui.feature_combo.clear()
                    self.ui.feature_combo.addItem("No synthetic features available")
                    set_widgets_status(self.ui.plot_features_button, mode='disable')

    def update_plot_label(self):
        """
        Updates the plot label based on the selected feature.
        """
        selected_feature_full_name = self.ui.feature_combo.currentText()

        # Check if this is a placeholder/error message rather than an actual feature
        if not selected_feature_full_name or selected_feature_full_name in [
            "No compatible features found",
            "No synthetic features available"
        ]:
            self.ui.plot_label.clear()
            return

        # Since combo box now contains full names, just display the selected full name
        self.ui.plot_label.setText(selected_feature_full_name)

    def get_selected_feature_code(self):
        """
        Get the feature short code corresponding to the selected full name in the combo box.

        Returns:
            str: Feature short code (e.g., 'COV', 'OCC') or the full name if not found
        """
        selected_full_name = self.ui.feature_combo.currentText()

        # Create reverse mapping from full names to short codes
        if hasattr(self, 'feature_list_dictionary'):
            reverse_dict = {v: k for k, v in self.feature_list_dictionary.items()}
            return reverse_dict.get(selected_full_name, selected_full_name)

        return selected_full_name

    def load_study1(self):
        """
        Loads the first EEG-COMET study using the new configuration-based approach.
        """
        self.study1_path = QFileDialog.getExistingDirectory(
            self, "Select the folder containing an EEG-COMET study.")

        if not self.study1_path:
            return

        # Check for the new configuration file
        config_path = os.path.join(self.study1_path, 'eeg_comet_config.ini')
        if not os.path.exists(config_path):
            QMessageBox.information(
                self, "Load error",
                "The selected folder does not contain a valid EEG-COMET study!\n"
                "Please select a folder containing 'eeg_comet_config.ini' file.",
                QMessageBox.Ok
            )
            return

        try:
            # Create a new COMET object and load the study
            self.comet_tbx_study1 = COMET(auto_save=False)
            self.comet_tbx_study1.config = self.comet_tbx_study1.load_config(config_path)
            self.comet_tbx_study1.load_config_values()
            self.comet_tbx_study1.reset_directories()

            # Load study data
            self.comet_tbx_study1.load_eeg_info()
            self.comet_tbx_study1.load_maps()
            self.comet_tbx_study1.load_clean()

            self.study1_loaded = True

            # Update UI with loaded study information
            self.update_study_info(self.comet_tbx_study1, self.ui.study1_file_list)
            self.plot_maps(
                self.comet_tbx_study1,
                self.figure_microstates_study1,
                self.canvas_microstates_study1
            )
            self.update_ui()

            # Log completion status for Study 1
            self._log_study_completion_status(self.comet_tbx_study1, "Study 1")

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load study: {str(e)}",
                QMessageBox.Ok
            )

    def load_study2(self):
        """
        Loads the second EEG-COMET study using the new configuration-based approach.
        """
        self.study2_path = QFileDialog.getExistingDirectory(
            self, "Select the folder containing an EEG-COMET study.")

        if not self.study2_path:
            return

        # Check for the new configuration file
        config_path = os.path.join(self.study2_path, 'eeg_comet_config.ini')
        if not os.path.exists(config_path):
            QMessageBox.information(
                self, "Load error",
                "The selected folder does not contain a valid EEG-COMET study!\n"
                "Please select a folder containing 'eeg_comet_config.ini' file.",
                QMessageBox.Ok
            )
            return

        try:
            # Create a new COMET object and load the study
            self.comet_tbx_study2 = COMET(auto_save=False)
            self.comet_tbx_study2.config = self.comet_tbx_study2.load_config(config_path)
            self.comet_tbx_study2.load_config_values()
            self.comet_tbx_study2.reset_directories()

            # Load study data
            self.comet_tbx_study2.load_eeg_info()
            self.comet_tbx_study2.load_maps()
            self.comet_tbx_study2.load_clean()

            self.study2_loaded = True

            # Update UI with loaded study information
            self.update_study_info(self.comet_tbx_study2, self.ui.study2_file_list)
            self.plot_maps(
                self.comet_tbx_study2,
                self.figure_microstates_study2,
                self.canvas_microstates_study2
            )
            self.update_ui()

            # Log completion status for Study 2
            self._log_study_completion_status(self.comet_tbx_study2, "Study 2")

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load study: {str(e)}",
                QMessageBox.Ok
            )

    @staticmethod
    def update_study_info(tbx, listwidget):
        """
        Updates the study information in a list widget.
        """
        for eeg in tbx.list_eegs:
            listwidget.addItem(str(eeg))

    def plot_maps(self, tbx, figure, canvas):
        """
        Plots microstate maps on the canvas.
        """
        # Clear existing axes
        for ax in figure.get_axes():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Create subplots for each microstate
        axs = [
            figure.add_subplot(1, len(tbx.micro_labels), idx + 1)
            for idx in range(len(tbx.micro_labels))
        ]

        # Plot each microstate with its label
        for idx, ax in enumerate(axs):
            self.plot_microstates_with_labels(
                tbx.best_maps[idx, :],
                tbx.micro_labels[idx],
                tbx.eeg_info,
                ax
            )

        # Adjust layout and render
        figure.tight_layout()
        canvas.draw()

    @staticmethod
    def plot_microstates_with_labels(microstate, micro_label, eeg_info, ax):
        """Plots microstates with corresponding labels on a given axis."""
        show_microstate(microstate, eeg_info, ax)
        ax.axis('off')
        ax.text(
            0.5, -0.2, micro_label.upper(),
            transform=ax.transAxes,
            fontsize=16,
            ha='center',
            va='center'
        )
        ax.text(0, 0, '', transform=ax.transAxes)  # Placeholder or additional text

    @staticmethod
    def organize_data2plot(common_features_df, selected_feature):
        """
        Organizes data for plotting based on the selected feature.
        """
        columns_to_keep = ['Filename', 'Study'] + [
            col for col in common_features_df.columns if col.startswith(selected_feature)
        ]
        common_features_df = common_features_df[columns_to_keep]
        feature_list = [
            col for col in common_features_df.columns
            if col not in ['Study', 'Filename']
        ]

        plot_data = pd.melt(
            common_features_df.reset_index(),
            id_vars=['Filename', 'Study'],
            value_vars=feature_list
        )
        plot_data.columns = ['Filename', 'Study', 'Feature', selected_feature]
        return plot_data, feature_list

    @staticmethod
    def clear_and_set_fonts(ax):
        """
        Clears the plot and sets the font sizes.
        """
        ax.clear()
        for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()
        ):
            item.set_fontsize(18)

    def set_labels_ticks(self, ax, filter_cols, feature):
        """
        Sets labels and ticks for the plot based on selected features.
        """
        ax.set_xlabel("Microstate")
        xticklabels = ['_'.join(col.split('_')[1:]) for col in filter_cols]
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(self.feature_list_dictionary[feature])
        self.ui.plot_label.setText(f"{self.feature_list_dictionary[feature]}")

    def discover_available_features(self, tbx):
        """
        Discover what feature files are actually available in a study.

        Args:
            tbx: The COMET toolbox object for the study

        Returns:
            dict: Dictionary with feature types as keys and lists of available modes as values
        """
        available_features = {}

        if not hasattr(tbx, 'extracted_features_path') or not os.path.exists(tbx.extracted_features_path):
            return available_features

        # Check for all possible feature files
        feature_types = ['real', 'surrogate', 'random']
        feature_modes = ['static', 'dynamic', 'averaged', 'sliding']
        export_formats = ['.csv', '.pkl', '.hdf', '.json']

        for feature_type in feature_types:
            available_modes = []
            for feature_mode in feature_modes:
                for export_format in export_formats:
                    feature_filename = f"{feature_type}_{feature_mode}_features{export_format}"
                    feature_path = os.path.join(tbx.extracted_features_path, feature_filename)

                    if os.path.exists(feature_path):
                        if feature_mode not in available_modes:
                            available_modes.append(feature_mode)

            if available_modes:
                available_features[feature_type] = available_modes

        return available_features

    def get_compatible_features(self):
        """
        Get features that are compatible between studies.

        Returns:
            tuple: (feature_type, feature_mode) that can be used for comparison
        """
        if not self.study1_loaded:
            return None, None

        # Discover available features in both studies
        study1_features = self.discover_available_features(self.comet_tbx_study1)

        if self.ui.compare_study2_radio.isChecked():
            if not self.study2_loaded:
                return None, None
            study2_features = self.discover_available_features(self.comet_tbx_study2)

            # Find common feature types and modes
            common_types = set(study1_features.keys()) & set(study2_features.keys())
            if not common_types:
                return None, None

            # Prefer 'real' features if available
            if 'real' in common_types:
                feature_type = 'real'
            else:
                feature_type = list(common_types)[0]

            # Find common modes for the selected type
            common_modes = set(study1_features[feature_type]) & set(study2_features[feature_type])
            if not common_modes:
                return None, None

            # Prefer 'static' mode if available, otherwise use first available
            if 'static' in common_modes:
                feature_mode = 'static'
            else:
                feature_mode = list(common_modes)[0]

        else:
            # Comparing with synthetic data from study 1
            if 'real' not in study1_features:
                return None, None

            feature_type = 'real'
            synthetic_type = self.synthetic_type  # 'surrogate' or 'random'

            # Check if synthetic features exist
            if synthetic_type not in study1_features:
                return None, None

            # Find common modes between real and synthetic
            common_modes = set(study1_features['real']) & set(study1_features[synthetic_type])
            if not common_modes:
                return None, None

            # Prefer 'static' mode if available
            if 'static' in common_modes:
                feature_mode = 'static'
            else:
                feature_mode = list(common_modes)[0]

        return feature_type, feature_mode

    def load_features_safely(self, tbx, feature_type, feature_mode):
        """
        Safely load features with proper error handling.

        Args:
            tbx: The COMET toolbox object
            feature_type: Type of features to load ('real', 'surrogate', 'random')
            feature_mode: Mode of features to load ('static', 'dynamic', etc.)

        Returns:
            pandas.DataFrame or None: The loaded features or None if loading fails
        """
        if not hasattr(tbx, 'extracted_features_path') or not os.path.exists(tbx.extracted_features_path):
            return None

        # Try different export formats
        export_formats = ['.csv', '.pkl', '.hdf', '.json']

        for export_format in export_formats:
            feature_filename = f"{feature_type}_{feature_mode}_features{export_format}"
            feature_path = os.path.join(tbx.extracted_features_path, feature_filename)

            if os.path.exists(feature_path):
                try:
                    return FeatureIO().import_features(feature_path, export_format)
                except Exception as e:
                    print(f"Error loading {feature_path}: {e}")
                    continue

        return None

    @staticmethod
    def load_features(tbx, feature_type):
        """
        Legacy method for backward compatibility.
        Loads features from a file and returns them as a DataFrame.
        """
        feature_path = os.path.join(
            tbx.extracted_features_path,
            f"{feature_type}_static_features{tbx.export_format}"
        )
        return FeatureIO().import_features(feature_path, tbx.export_format)

    def get_common_features(self):
        """
        Retrieves common features between the two loaded studies using robust discovery.
        """
        # Get compatible feature type and mode
        feature_type, feature_mode = self.get_compatible_features()

        if feature_type is None or feature_mode is None:
            # Show helpful error message
            study1_features = self.discover_available_features(self.comet_tbx_study1)
            error_msg = "No compatible features found between studies.\n\n"
            error_msg += f"Study 1 available features: {study1_features}\n"

            if self.ui.compare_study2_radio.isChecked() and self.study2_loaded:
                study2_features = self.discover_available_features(self.comet_tbx_study2)
                error_msg += f"Study 2 available features: {study2_features}\n"

            QMessageBox.warning(
                self, "Feature Loading Error",
                error_msg + "\nPlease ensure both studies have completed feature extraction.",
                QMessageBox.Ok
            )
            return None, None

        # Load features using the robust method
        features_df_study1 = self.load_features_safely(self.comet_tbx_study1, feature_type, feature_mode)

        if features_df_study1 is None:
            QMessageBox.warning(
                self, "Feature Loading Error",
                f"Could not load {feature_type} {feature_mode} features from Study 1.",
                QMessageBox.Ok
            )
            return None, None

        if self.ui.compare_study2_radio.isChecked():
            features_df_study2 = self.load_features_safely(self.comet_tbx_study2, feature_type, feature_mode)
            if features_df_study2 is None:
                QMessageBox.warning(
                    self, "Feature Loading Error",
                    f"Could not load {feature_type} {feature_mode} features from Study 2.",
                    QMessageBox.Ok
                )
                return None, None
        else:
            # Load synthetic features from study 1
            features_df_study2 = self.load_features_safely(self.comet_tbx_study1, self.synthetic_type, feature_mode)
            if features_df_study2 is None:
                QMessageBox.warning(
                    self, "Feature Loading Error",
                    f"Could not load {self.synthetic_type} {feature_mode} features from Study 1.",
                    QMessageBox.Ok
                )
                return None, None

        # Identify common columns
        common_columns = sorted([
            col for col in features_df_study1.columns
            if col in features_df_study2.columns
        ])

        if not common_columns:
            QMessageBox.warning(
                self, "Feature Comparison Error",
                "No common feature columns found between the datasets.",
                QMessageBox.Ok
            )
            return None, None

        # Prepare DataFrames with Study information
        study1_df = features_df_study1[common_columns].copy()
        study1_df['Study'] = self.comet_tbx_study1.study_name

        study2_df = features_df_study2[common_columns].copy()
        study2_df['Study'] = self.study2_name

        # Concatenate DataFrames
        common_features_df = pd.concat([study1_df, study2_df], ignore_index=True)

        return common_features_df, common_columns

    def plot_features(self):
        """
        Plots a violin plot for the selected static feature.
        """
        selected_feature = self.get_selected_feature_code()

        try:
            result = self.get_common_features()
            if result is None or result[0] is None:
                return

            common_features_df, common_columns = result
            plot_data, feature_list = self.organize_data2plot(common_features_df, selected_feature)

            ax = self.canvas_features.figure.gca()
            self.clear_and_set_fonts(ax)

            # Create violin plot
            sns.violinplot(
                x='Feature',
                y=selected_feature,
                hue='Study',
                data=plot_data,
                ax=ax,
                hue_order=[self.comet_tbx_study1.study_name, self.study2_name]
            )

            # Overlay swarm plot for individual data points
            sns.swarmplot(
                x='Feature',
                y=selected_feature,
                hue='Study',
                data=plot_data,
                ax=ax,
                color="white",
                size=10,
                marker='o',
                dodge=True,
                legend=False
            )

            # Set labels and ticks
            self.set_labels_ticks(ax, feature_list, selected_feature)
            self.canvas_features.draw()

        except Exception as e:
            QMessageBox.critical(
                self, "Plotting Error",
                f"Failed to plot features: {str(e)}",
                QMessageBox.Ok
            )

    def update_corr_stats(self):
        """Updates the correlation statistics in the UI based on microstate analysis."""
        self.ui.correlations_textedit.clear()
        # Perform spatial correlation analysis
        microlabels, correlation_coefficients, p_values, adjusted_p_values = self.spatial_correlation_analysis()

        # Check if we have valid results
        if not microlabels or not correlation_coefficients:
            if self.ui.compare_study2_radio.isChecked():
                self.ui.correlations_textedit.appendPlainText(
                    "No microstate maps available for spatial correlation analysis.\n"
                    "Please ensure both studies have completed microstate clustering."
                )
            else:
                self.ui.correlations_textedit.appendPlainText(
                    "Spatial correlation analysis is not available for synthetic data comparisons."
                )
            return

        # Prepare correlation text
        correlations_text = "Correlation of Microstates Between Studies:\n"
        for i, val in enumerate(correlation_coefficients):
            correlations_text += f"Microstate {microlabels[i]}: {100 * abs(val):.3f} %\n"
        correlations_text += f"\nP-values:\n"
        for i, val in enumerate(p_values):
            correlations_text += f"Microstate {microlabels[i]}: {val}\n"
        correlations_text += f"\n{self.ui.multiple_test_method_combo.currentText()} Adjusted P-values:\n"
        for i, val in enumerate(adjusted_p_values):
            correlations_text += f"Microstate {microlabels[i]}: {val}\n"

        # Display correlation statistics
        self.ui.correlations_textedit.appendPlainText(correlations_text)

    def update_feature_stats(self):
        """
        Updates the feature statistics in the UI based on feature comparison.
        """
        self.ui.stats_textedit.clear()
        # Perform feature comparison analysis
        feature_list, t_test_results, p_values, adjusted_p_values = self.feature_comparison_analysis()

        # Check if we have valid results
        if not feature_list:
            self.ui.stats_textedit.appendPlainText("No features available for statistical comparison.")
            return

        # Display t-test results
        for i, feat in enumerate(feature_list):
            t_statistic = t_test_results[feat]
            adjusted_p_value = adjusted_p_values[i]
            result_text = f"Feature Name: {feat}:\n"
            result_text += (
                f"T-statistic: {t_statistic}, "
                f"p-value: {p_values[i]}, "
                f"{self.ui.multiple_test_method_combo.currentText()} Adjusted p-value: {adjusted_p_value}\n"
            )
            self.ui.stats_textedit.appendPlainText(result_text)

    def spatial_correlation_analysis(self):
        """
        Performs spatial correlation analysis between microstates of two studies.
        """
        # Check if both studies have microstate maps
        if not hasattr(self.comet_tbx_study1, 'best_maps') or self.comet_tbx_study1.best_maps is None:
            return [], [], [], []

        if self.ui.compare_study2_radio.isChecked():
            if not hasattr(self.comet_tbx_study2, 'best_maps') or self.comet_tbx_study2.best_maps is None:
                return [], [], [], []

            # Extract microstates from both studies
            microstates_study1 = self.comet_tbx_study1.best_maps
            microstates_study2 = self.comet_tbx_study2.best_maps

            # Get common microstate labels
            microlabels_study1 = self.comet_tbx_study1.micro_labels
            microlabels_study2 = self.comet_tbx_study2.micro_labels
            microlabels = [label for label in microlabels_study1 if label in microlabels_study2]
        else:
            # For synthetic comparisons, we can't do spatial correlation
            return [], [], [], []

        # Determine the number of microstates to compare
        min_states = min(microstates_study1.shape[0], microstates_study2.shape[0])

        if min_states == 0:
            return [], [], [], []

        correlation_coefficients, p_values = [], []

        # Calculate Pearson correlation for each microstate
        for m in range(min_states):
            corr_coef, p_val = pearsonr(microstates_study1[m, :], microstates_study2[m, :])
            correlation_coefficients.append(corr_coef)
            p_values.append(p_val)

        # Adjust p-values for multiple testing
        adjusted_p_values = multipletests(
            p_values,
            method=self.ui.multiple_test_method_combo.currentText().lower()
        )[1]

        return microlabels[:min_states], correlation_coefficients, p_values, adjusted_p_values

    def feature_comparison_analysis(self):
        """
        Performs feature comparison analysis between two studies using t-tests.
        """
        result = self.get_common_features()
        if result is None or result[0] is None:
            return [], {}, [], []

        common_features_df, common_columns = result
        selected_feature = self.get_selected_feature_code()

        # Organize data for the selected feature
        columns_to_keep = ['Filename', 'Study'] + [
            col for col in common_features_df.columns if col.startswith(selected_feature)
        ]
        common_features_df = common_features_df[columns_to_keep]

        # Separate data by study
        selected_feature_study1_df = common_features_df[
            common_features_df['Study'] == self.comet_tbx_study1.study_name
            ].sort_values(by='Filename')

        selected_feature_study2_df = common_features_df[
            common_features_df['Study'] == self.study2_name
            ].sort_values(by='Filename')

        # Perform t-tests for each feature
        feature_list = [
            col for col in selected_feature_study1_df.columns
            if col not in ['Study', 'Filename']
        ]
        t_test_results = {}
        p_values = []

        for feat in feature_list:
            study1_values = selected_feature_study1_df[feat].tolist()
            study2_values = selected_feature_study2_df[feat].tolist()

            if self.ui.paired_test_checkbox.isChecked():
                t_statistic, p_value = ttest_rel(study1_values, study2_values)
            else:
                t_statistic, p_value = ttest_ind(study1_values, study2_values)

            t_test_results[feat] = t_statistic
            p_values.append(p_value)

        # Adjust p-values for multiple testing
        adjusted_p_values = multipletests(
            p_values,
            method=self.ui.multiple_test_method_combo.currentText().lower()
        )[1]

        return feature_list, t_test_results, p_values, adjusted_p_values

    def _log_study_completion_status(self, comet_instance, study_name):
        """Log the completion status of all processing steps for a study"""
        print('\n' + '=' * 60)
        print(f'[INFO] {study_name} Completion Status:')
        print('=' * 60)

        # Check each processing step
        steps_status = []

        # Preprocessing
        if comet_instance.done_preprocessing:
            steps_status.append("✅ Data Preprocessing")
            print(f"[INFO] ✅ Data Preprocessing - COMPLETED")
        else:
            steps_status.append("❌ Data Preprocessing")
            print(f"[INFO] ❌ Data Preprocessing - NOT COMPLETED")

        # Clustering
        if comet_instance.done_clustering:
            steps_status.append("✅ Microstate Clustering")
            print(f"[INFO] ✅ Microstate Clustering - COMPLETED")
            if hasattr(comet_instance, 'best_gev') and comet_instance.best_gev is not None:
                print(f"[INFO]   └─ Best GEV: {100 * comet_instance.best_gev:.3f}%")
            if hasattr(comet_instance, 'number_of_maps') and comet_instance.number_of_maps is not None:
                print(f"[INFO]   └─ Number of Maps: {comet_instance.number_of_maps}")
        else:
            steps_status.append("❌ Microstate Clustering")
            print(f"[INFO] ❌ Microstate Clustering - NOT COMPLETED")

        # Microstate Labeling
        if comet_instance.done_microstate_labeling:
            steps_status.append("✅ Microstate Labeling")
            print(f"[INFO] ✅ Microstate Labeling - COMPLETED")
        else:
            steps_status.append("❌ Microstate Labeling")
            print(f"[INFO] ❌ Microstate Labeling - NOT COMPLETED")

        # Backfitting
        if comet_instance.done_backfitting:
            steps_status.append("✅ Microstate Backfitting")
            print(f"[INFO] ✅ Microstate Backfitting - COMPLETED")
        else:
            steps_status.append("❌ Microstate Backfitting")
            print(f"[INFO] ❌ Microstate Backfitting - NOT COMPLETED")

        # Feature Extraction
        if comet_instance.done_extracting_features:
            steps_status.append("✅ Feature Extraction")
            print(f"[INFO] ✅ Feature Extraction - COMPLETED")
        else:
            steps_status.append("❌ Feature Extraction")
            print(f"[INFO] ❌ Feature Extraction - NOT COMPLETED")

        # Source Localization
        if comet_instance.done_source_localization:
            steps_status.append("✅ Source Localization")
            print(f"[INFO] ✅ Source Localization - COMPLETED")
        else:
            steps_status.append("❌ Source Localization")
            print(f"[INFO] ❌ Source Localization - NOT COMPLETED")

        # Source-Microstate Correlation
        if comet_instance.done_identifying_microstate_sources:
            steps_status.append("✅ Source-Microstate Correlation")
            print(f"[INFO] ✅ Source-Microstate Correlation - COMPLETED")
        else:
            steps_status.append("❌ Source-Microstate Correlation")
            print(f"[INFO] ❌ Source-Microstate Correlation - NOT COMPLETED")

        # Summary
        completed_steps = sum(1 for step in steps_status if step.startswith("✅"))
        total_steps = len(steps_status)

        print('=' * 60)
        print(f"[INFO] {study_name} Summary: {completed_steps}/{total_steps} steps completed")

        if completed_steps == total_steps:
            print(f"[INFO] 🎉 All processing steps completed for {study_name}!")
        elif completed_steps == 0:
            print(f"[INFO] 📋 No processing steps completed yet for {study_name}")
        else:
            print(
                f"[INFO] 📊 {completed_steps} steps completed, {total_steps - completed_steps} remaining for {study_name}")

        print('=' * 60)

    def calculate_icc(self, ratings, icc_type='ICC(2,1)'):
        """
        Calculate Intraclass Correlation Coefficient.

        Args:
            ratings: numpy array of shape (n_subjects, n_raters)
                     where n_raters = 2 for test-retest
            icc_type: Type of ICC to calculate
                     'ICC(2,1)' - Two-way random effects, single measurement, absolute agreement

        Returns:
            icc_value: The ICC value
            ci_low: Lower bound of 95% confidence interval
            ci_high: Upper bound of 95% confidence interval
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
            icc_value = (ms_rows - ms_error) / (ms_rows + (n_raters - 1) * ms_error +
                                                n_raters * (ms_cols - ms_error) / n_subjects)

        # Approximate 95% confidence interval using F-distribution
        from scipy.stats import f
        alpha = 0.05

        # F-statistic
        if ms_error > 0:
            f_stat = ms_rows / ms_error
            # Critical F values
            f_low = f.ppf(alpha / 2, df_rows, df_error)
            f_high = f.ppf(1 - alpha / 2, df_rows, df_error)

            # Confidence intervals
            fl = f_stat / f_high
            fu = f_stat / f_low

            ci_low = (fl - 1) / (fl + n_raters - 1)
            ci_high = (fu - 1) / (fu + n_raters - 1)

            # Ensure bounds are within [0, 1]
            ci_low = max(0, min(1, ci_low))
            ci_high = max(0, min(1, ci_high))
        else:
            ci_low = ci_high = icc_value

        return icc_value, ci_low, ci_high

    def extract_subject_id(self, filename):
        """
        Extract subject ID from filename.
        Handles various naming conventions.
        """
        # Try to extract subject ID from common patterns
        import re

        # Pattern 1: sub-XX format
        match = re.search(r'sub-(\d+)', filename)
        if match:
            return f"sub-{match.group(1)}"

        # Pattern 2: subjectXX or subjXX format
        match = re.search(r'subj(?:ect)?(\d+)', filename, re.IGNORECASE)
        if match:
            return f"subject{match.group(1)}"

        # Pattern 3: SXX format
        match = re.search(r'S(\d+)', filename)
        if match:
            return f"S{match.group(1)}"

        # Pattern 4: Just numbers at the beginning
        match = re.search(r'^(\d+)', filename)
        if match:
            return match.group(1)

        # If no pattern matches, return the filename itself
        return filename

    def perform_test_retest_analysis(self):
        """
        Perform test-retest reliability analysis using ICC for the selected feature only.
        """
        try:
            # Clear previous stats
            self.ui.stats_textedit.clear()

            # Get compatible features
            feature_type, feature_mode = self.get_compatible_features()

            if feature_type is None or feature_mode is None:
                self.ui.stats_textedit.appendPlainText(
                    "No compatible features found for test-retest analysis.\n"
                    "Please ensure both studies have completed feature extraction."
                )
                return

            # Load features from both studies
            features_study1 = self.load_features_safely(
                self.comet_tbx_study1, feature_type, feature_mode
            )
            features_study2 = self.load_features_safely(
                self.comet_tbx_study2, feature_type, feature_mode
            )

            if features_study1 is None or features_study2 is None:
                self.ui.stats_textedit.appendPlainText(
                    "Failed to load features from one or both studies."
                )
                return

            # Extract subject IDs and match between studies
            study1_subjects = {}
            study2_subjects = {}

            for idx, row in features_study1.iterrows():
                filename = row['Filename']
                subject_id = self.extract_subject_id(filename)
                study1_subjects[subject_id] = idx

            for idx, row in features_study2.iterrows():
                filename = row['Filename']
                subject_id = self.extract_subject_id(filename)
                study2_subjects[subject_id] = idx

            # Find common subjects
            common_subjects = set(study1_subjects.keys()) & set(study2_subjects.keys())

            if len(common_subjects) == 0:
                self.ui.stats_textedit.appendPlainText(
                    "No matching subjects found between studies.\n"
                    "Please ensure both studies contain data from the same subjects.\n"
                    f"Study 1 subjects: {list(study1_subjects.keys())[:5]}...\n"
                    f"Study 2 subjects: {list(study2_subjects.keys())[:5]}..."
                )
                return

            # Sort subjects for consistent ordering
            common_subjects = sorted(list(common_subjects))

            self.ui.stats_textedit.appendPlainText(
                f"Test-Retest Reliability Analysis (ICC)\n"
                f"{'=' * 60}\n"
                f"Number of matched subjects: {len(common_subjects)}\n"
                f"Feature type: {feature_type}\n"
                f"Feature mode: {feature_mode}\n"
                f"{'=' * 60}\n"
            )

            # Get selected feature code
            selected_feature = self.get_selected_feature_code()

            # Get feature columns (exclude Filename and Study columns), filter for selected feature
            feature_columns = [
                col for col in features_study1.columns
                if col not in ['Filename', 'Study'] and col.startswith(selected_feature)
            ]

            if not feature_columns:
                self.ui.stats_textedit.appendPlainText(
                    f"No columns found for selected feature: {selected_feature}"
                )
                return

            # Calculate ICC for each sub-feature (if multiple columns)
            icc_results = []

            for feature in feature_columns:
                # Create paired data matrix
                ratings = np.zeros((len(common_subjects), 2))

                for i, subject_id in enumerate(common_subjects):
                    idx1 = study1_subjects[subject_id]
                    idx2 = study2_subjects[subject_id]

                    ratings[i, 0] = features_study1.loc[idx1, feature]
                    ratings[i, 1] = features_study2.loc[idx2, feature]

                # Skip if any NaN values
                if np.any(np.isnan(ratings)):
                    icc_value, ci_low, ci_high = np.nan, np.nan, np.nan
                else:
                    # Calculate ICC
                    icc_value, ci_low, ci_high = self.calculate_icc(ratings)

                # Get full feature name
                feature_parts = feature.split('_')
                if feature_parts[0] in self.feature_list_dictionary:
                    feature_full_name = self.feature_list_dictionary[feature_parts[0]]
                    if len(feature_parts) > 1:
                        feature_full_name += f" ({'_'.join(feature_parts[1:])})"
                else:
                    feature_full_name = feature

                icc_results.append({
                    'Feature': feature,
                    'Feature_Full': feature_full_name,
                    'ICC': icc_value,
                    'CI_Low': ci_low,
                    'CI_High': ci_high
                })

            # Sort results by ICC value (descending)
            icc_results.sort(key=lambda x: x['ICC'] if not np.isnan(x['ICC']) else -1, reverse=True)

            # Display results
            self.ui.stats_textedit.appendPlainText(
                f"{'Feature':<40} {'ICC':>8} {'95% CI':>20}\n"
                f"{'-' * 70}"
            )

            for result in icc_results:
                if not np.isnan(result['ICC']):
                    reliability = self.interpret_icc(result['ICC'])
                    ci_str = f"[{result['CI_Low']:.3f}, {result['CI_High']:.3f}]"
                    self.ui.stats_textedit.appendPlainText(
                        f"{result['Feature_Full']:<40} {result['ICC']:>8.3f} {ci_str:>20} {reliability}"
                    )
                else:
                    self.ui.stats_textedit.appendPlainText(
                        f"{result['Feature_Full']:<40} {'N/A':>8} {'N/A':>20}"
                    )

            # Summary statistics for selected feature
            valid_iccs = [r['ICC'] for r in icc_results if not np.isnan(r['ICC'])]
            if valid_iccs:
                self.ui.stats_textedit.appendPlainText(
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

        except Exception as e:
            QMessageBox.critical(
                self, "Test-Retest Analysis Error",
                f"Failed to perform test-retest analysis: {str(e)}",
                QMessageBox.Ok
            )
            import traceback
            traceback.print_exc()

    def interpret_icc(self, icc_value):
        """
        Interpret ICC value according to Koo & Li (2016) guidelines.
        """
        if icc_value < 0.50:
            return "(Poor)"
        elif icc_value < 0.75:
            return "(Moderate)"
        elif icc_value < 0.90:
            return "(Good)"
        else:
            return "(Excellent)"
