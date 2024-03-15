
import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from functions.data_utils.data_io import DataIO
from functions.backfitting_utils.segmentation_io import SegmentationIO
from functions.gui_utils.set_widgets_status import set_widgets_status


class BackfittingVisualizationDialog(QDialog):
    def __init__(self, context, parent=None):
        super().__init__(parent)

        # Initialize variables
        self.datatype = ""

        # Load UI and initialize UI components
        self._load_ui(context)
        self._initialize_ui()

    def _load_ui(self, context):
        # Load UI from the specified file
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("BackfittingVisualizationWindow.ui"), self)

    def _initialize_ui(self):
        # Set up the main UI components
        self.ui.setWindowTitle("Visualization of the backfitted microstates")
        # Set window flags to include the maximize button
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas.setMinimumHeight(100)
        self.ui.Figure_Layout.addWidget(self.canvas)

        # Connect signals to slots
        self.ui.show_backfitting_button.clicked.connect(self.show_backfitting)
        self.ui.export_backfitting_image_button.clicked.connect(self.export_plot)
        self.ui.figure_settings_checkbox.clicked.connect(self.backfitting_visualization_controller)

        # Initialize UI state
        self.backfitting_visualization_controller()

    def backfitting_visualization_controller(self):
        # Control the visibility and enable/disable state of UI widgets based on conditions
        epoched_data_widgets = [
            self.ui.num_trials_label,
            self.ui.num_trials_spinbox
        ]

        figure_settings_widgets = [
            self.ui.font_size_label,
            self.ui.font_size_input,
            self.ui.label_size_label,
            self.ui.label_size_input,
            self.ui.colormap_label,
            self.ui.colormap_combobox,
        ]

        if self.datatype == 'epoched':
            set_widgets_status(epoched_data_widgets, mode='enable')
            set_widgets_status(epoched_data_widgets, mode='show')

            # Get the list of files in the folder
            selected_file_name = self.ui.eeg_filenames_combobox.currentText()
            files = os.listdir(self.segmentation_path)
            # Count the files that start with the specified prefix
            num_trials = sum(1 for file in files if file.startswith(selected_file_name))
            self.ui.num_trials_spinbox.setRange(0, num_trials - 1)
        else:
            set_widgets_status(epoched_data_widgets, mode='disable')
            set_widgets_status(epoched_data_widgets, mode='hide')

        if self.ui.figure_settings_checkbox.isChecked():
            set_widgets_status(figure_settings_widgets, mode='show')
        else:
            set_widgets_status(figure_settings_widgets, mode='hide')

    def show_backfitting(self):
        # Handle the logic for displaying backfitting data
        selected_file_name = self.ui.eeg_filenames_combobox.currentText()

        # Load EEG data and segmentation data
        eeg_data, eeg_times, segmentation_data = self._load_data_and_segmentation(selected_file_name)

        # Get time range and other parameters
        time_min, time_max = self._get_time_range()
        fontsize, labelsize, colormap = self._get_plot_parameters()

        # Plot the data
        self._plot_data(eeg_times, eeg_data, segmentation_data, time_min, time_max, fontsize, labelsize, colormap)

    def _load_data_and_segmentation(self, selected_file_name):
        # Load EEG data and segmentation data from files
        eeg_dir = os.path.join(self.preprocessed_data_path, f"{selected_file_name}{self.extension}")

        data_io = DataIO()
        eeg = data_io.load_eegs(eeg_dir, self.extension, self.datatype)
        eeg_data = eeg.get_data()
        eeg_times = eeg.times * 1000

        data_to_use = self._get_data_to_use(eeg_data)

        segmentation_io = SegmentationIO()
        if self.datatype == "epoched":
            segmentation_filename = f"{selected_file_name}_{self.ui.num_trials_spinbox.value()}{self.export_format}"
        else:
            segmentation_filename = f"{selected_file_name}{self.export_format}"
        segmentation_path = os.path.join(self.segmentation_path, segmentation_filename)
        segmentation_data = segmentation_io.load_segmentation(segmentation_path, import_format=self.export_format)

        return data_to_use, eeg_times, segmentation_data

    def _get_plot_parameters(self):
        # Get plot parameters from UI components
        fontsize = int(self.ui.font_size_input.text())
        labelsize = int(self.ui.label_size_input.text())
        colormap = self.ui.colormap_combobox.currentText()
        return fontsize, labelsize, colormap

    def _get_data_to_use(self, eeg_data):
        # Get the data to use based on the datatype
        if self.datatype == 'epoched':
            trial = self.ui.num_trials_spinbox.value()
            return np.std(eeg_data[trial, :, :], axis=0)
        return np.std(eeg_data, axis=0)

    def _get_time_range(self):
        # Get the time range from UI components
        time_min = int(self.ui.xlim_min_input.text())
        time_max = int(self.ui.xlim_max_input.text())
        return time_min, time_max

    def _plot_data(self, eeg_times, data_to_use, segmentation_data, time_min, time_max, fontsize, labelsize, colormap):
        # Plot the data using matplotlib
        xmin = np.argmin(np.abs(eeg_times - time_min))
        xmax = np.argmin(np.abs(eeg_times - time_max))

        # Limit the data and times to the desired time range
        times_to_use = eeg_times[xmin:xmax]
        data_to_use = data_to_use[xmin:xmax]
        segmentation_to_use = segmentation_data[xmin:xmax]

        ax = self.canvas.figure.gca()
        ax.clear()

        # Generate the plot
        legend_elements, color_map = self._prepare_legend_and_colors(segmentation_to_use, colormap)
        self._fill_plot(ax, times_to_use, data_to_use, segmentation_to_use, color_map)

        ax.set_xlabel('Time (ms)', fontsize=fontsize)
        ax.set_ylabel('Potential (μV)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.set_xlim([time_min, time_max])
        ax.legend(handles=legend_elements, loc='upper right', fontsize=fontsize)

        self.canvas.draw()

    @staticmethod
    def _prepare_legend_and_colors(segmentation_data, colormap):
        # Prepare legend and colors for the plot
        unique_labels = sorted(set(segmentation_data))
        cm = plt.get_cmap(colormap)
        unique_colors = [cm(1. * i / len(unique_labels)) for i in range(len(unique_labels))]
        color_map = {label: color for label, color in zip(unique_labels, unique_colors)}
        legend_elements = [Patch(facecolor=color, edgecolor='none', label=label) for
                           label, color in zip(unique_labels, unique_colors)]
        return legend_elements, color_map

    @staticmethod
    def _fill_plot(ax, times, data, segmentation, color_map):
        # Fill the plot with data and segmentation information
        prev_index = 0
        for idx, label in enumerate(segmentation[1:], start=1):
            if label != segmentation[idx - 1]:
                segment_color = color_map.get(segmentation[idx - 1], 'grey')
                ax.plot(times[prev_index:idx], data[prev_index:idx], color='black')
                ax.fill_between(times[prev_index:idx], data[prev_index:idx], color=segment_color)
                prev_index = idx

        # Handle the last segment
        segment_color = color_map.get(segmentation[-1], 'grey')
        ax.plot(times[prev_index:], data[prev_index:], color='black')
        ax.fill_between(times[prev_index:], data[prev_index:], color=segment_color)

    def export_plot(self):
        # Export the current plot to a file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Plot", "",
                                                   "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)",
                                                   options=options)
        if file_name:
            self._save_plot(file_name)

    def _save_plot(self, file_name):
        # Save the current plot to the specified file
        extension = os.path.splitext(file_name)[-1].lower()
        if not extension:
            file_name += '.png'
        self.canvas.figure.savefig(file_name)
