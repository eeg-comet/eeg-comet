import mne
import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from data_utils.data_io import DataIO
from backfitting_utils.segmentation_io import SegmentationIO
from gui_utils.set_widgets_status import set_widgets_status


class BackfittingVisualizationWindow(QDialog):
    def __init__(self, context, parent=None):
        """
        Initialize the BackfittingVisualizationWindow.
        """
        super().__init__(parent)

        # Initialize variables
        self.datatype = ""

        # Load UI and initialize UI components
        self._load_ui(context)
        self._initialize_ui()

    def _load_ui(self, context):
        """
        Load the UI layout from the specified .ui file.
        """
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("BackfittingVisualizationWindow.ui"), self)

    def _initialize_ui(self):
        """
        Initialize and set up the main UI components, including the plot canvas and signal-slot connections.
        """
        # Set window properties
        self.ui.setWindowTitle("Visualization of the backfitted microstates")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        # Initialize matplotlib Figure and Canvas
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas.setMinimumHeight(100)
        self.ui.Figure_Layout.addWidget(self.canvas)

        # Connect UI signals to their respective slots
        self.ui.show_backfitting_button.clicked.connect(self.show_backfitting)
        self.ui.export_backfitting_image_button.clicked.connect(self.export_plot)
        self.ui.figure_settings_checkbox.clicked.connect(self.backfitting_visualization_controller)

        # Initialize UI state based on default settings
        self.backfitting_visualization_controller()

    def backfitting_visualization_controller(self):
        """
        Control the visibility and enable/disable state of UI widgets based on the current data type
        and user-selected settings.
        """
        # Widgets related to epoched data
        epoched_data_widgets = [
            self.ui.num_trials_label,
            self.ui.num_trials_spinbox
        ]

        # Widgets related to figure settings
        figure_settings_widgets = [
            self.ui.font_size_label,
            self.ui.font_size_input,
            self.ui.label_size_label,
            self.ui.label_size_input,
            self.ui.colormap_label,
            self.ui.colormap_combobox,
        ]

        # Enable or disable epoched data widgets based on datatype
        if self.datatype == 'epoched':
            set_widgets_status(epoched_data_widgets, mode='enable')
            set_widgets_status(epoched_data_widgets, mode='show')
            selected_file_name = self.ui.eeg_filenames_combobox.currentText()
            segmentation_data = self._load_data_and_segmentation(selected_file_name)[4]
            num_trials = len(segmentation_data)
            self.ui.num_trials_spinbox.setRange(1, num_trials)
        else:
            set_widgets_status(epoched_data_widgets, mode='disable')
            set_widgets_status(epoched_data_widgets, mode='hide')

        # Show or hide figure settings widgets based on checkbox state
        if self.ui.figure_settings_checkbox.isChecked():
            set_widgets_status(figure_settings_widgets, mode='show')
        else:
            set_widgets_status(figure_settings_widgets, mode='hide')

    def _load_data_and_segmentation(self, selected_file_name):
        """
        Load EEG data and corresponding segmentation data from files.
        """
        # Initialize DataIO to load EEG data
        data_io = DataIO()
        eeg_dir, _ = data_io.find_data(
            input_folder=self.preprocessed_data_path,
            extension=self.extension,
            pattern=f"*{selected_file_name}*"
        )
        eeg = data_io.load_eeg(eeg_path=eeg_dir[0], datatype=self.datatype)
        eeg_info = eeg.info
        eeg_data = eeg.get_data()
        eeg_times = eeg.times * 1000  # Convert to milliseconds

        # Process EEG data based on datatype
        gfp_data = self._get_gfp_data(eeg_data)

        # Load segmentation data
        segmentation_io = SegmentationIO()
        segmentation_filename = f"{selected_file_name}{self.export_format}"
        segmentation_path = os.path.join(self.segmentation_path, segmentation_filename)
        segmentation_data = segmentation_io.load_segmentation(segmentation_path, import_format=self.export_format)

        return eeg_data, eeg_info, gfp_data, eeg_times, segmentation_data

    def _get_gfp_data(self, eeg_data):
        """
        Determine the EEG data to use for plotting based on the current datatype.
        """
        if self.datatype == 'epoched':
            trial = self.ui.num_trials_spinbox.value()
            return np.std(eeg_data[trial - 1, :, :], axis=0)
        return np.std(eeg_data, axis=0)

    def _get_time_range(self):
        """
        Retrieve the time range for plotting from the UI inputs.
        """
        time_min = int(self.ui.xlim_min_input.text())
        time_max = int(self.ui.xlim_max_input.text())
        return time_min, time_max

    def _get_plot_parameters(self):
        """
        Retrieve plot customization parameters from the UI inputs.
        """
        fontsize = int(self.ui.font_size_input.text())
        labelsize = int(self.ui.label_size_input.text())
        colormap = self.ui.colormap_combobox.currentText()
        return fontsize, labelsize, colormap

    def show_backfitting(self):
        """
        Handle the logic for displaying backfitting data by loading the selected EEG and segmentation data,
        retrieving plot parameters, and rendering the plot.
        """
        selected_file_name = self.ui.eeg_filenames_combobox.currentText()

        # Load EEG and segmentation data
        eeg_data, eeg_info, gfp_data, eeg_times, segmentation_data = self._load_data_and_segmentation(selected_file_name)

        # Get plot parameters
        time_min, time_max = self._get_time_range()
        if self.datatype == 'epoched':
            trial = self.ui.num_trials_spinbox.value()
            eeg2plot = eeg_data[trial - 1, :, :]
            segmentation2plot = segmentation_data[trial - 1, :]
        else:
            eeg2plot = eeg_data
            segmentation2plot = segmentation_data[0]
        fontsize, labelsize, colormap = self._get_plot_parameters()

        # Plot the data
        self._plot_data(eeg2plot, eeg_info, eeg_times, gfp_data, segmentation2plot, time_min, time_max, fontsize, labelsize, colormap)

    def _plot_data(self, eeg_data, eeg_info, eeg_times, gfp_data, segmentation_data, time_min, time_max, fontsize, labelsize,
                   colormap):
        """
        Plot the EEG data along with segmentation overlays and averaged EEG data where segmentation remains the same.
        Also, plot topographies for each segment directly on the same figure.
        """
        # Determine indices for the specified time range
        xmin = np.argmin(np.abs(eeg_times - time_min))
        xmax = np.argmin(np.abs(eeg_times - time_max))

        # Slice data and time arrays to the desired range
        eeg_to_use = eeg_data[:, xmin:xmax]
        times_to_use = eeg_times[xmin:xmax]
        gfp_to_use = gfp_data[xmin:xmax]
        segmentation_to_use = segmentation_data[xmin:xmax]

        # Clear the previous plot
        ax = self.canvas.figure.gca()
        ax.clear()

        # Prepare legend and color mapping
        legend_elements, color_map = self._prepare_legend_and_colors(segmentation_to_use, colormap)

        # Fill the plot with data and segmentation overlays
        self._fill_plot(ax, times_to_use, gfp_to_use, segmentation_to_use, color_map)

        # Initialize topography plotting for each averaged EEG segment
        avg_eeg_data = []
        prev_label = segmentation_to_use[0]
        prev_index = 0
        for idx, label in enumerate(segmentation_to_use[1:], start=1):
            if label != prev_label:
                # Average the EEG data for the previous segment
                segment_avg = np.mean(eeg_to_use[:, prev_index:idx], axis=1)
                avg_eeg_data.append((times_to_use[prev_index:idx], segment_avg, prev_label))
                prev_label = label
                prev_index = idx

        # Handle the last segment
        segment_avg = np.mean(eeg_to_use[:, prev_index:], axis=1)
        avg_eeg_data.append((times_to_use[prev_index:], segment_avg, prev_label))

        # Customize plot appearance
        ax.set_xlabel('Time (ms)', fontsize=fontsize)
        ax.set_ylabel('Potential (μV)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.set_xlim([time_min, time_max])
        ax.legend(handles=legend_elements, loc='upper right', fontsize=fontsize)

        # Now, plot the topography for each averaged segment using MNE, directly on the same plot
        for i, (times, avg_data, label) in enumerate(avg_eeg_data):
            # Determine the center of the time window for placing the topography
            mid_time = times[len(times) // 2]

            # Create a small inset axis for the topography
            inset_ax = ax.inset_axes(
                [mid_time / (time_max - time_min), 0.8, 0.1, 0.1])  # Adjust inset position and size as needed

            # Plot the topography for the segment
            mne.viz.plot_topomap(avg_data, eeg_info, axes=inset_ax, show=False)

        # Render the plot on the canvas
        self.canvas.draw()

    @staticmethod
    def _prepare_legend_and_colors(segmentation_data, colormap):
        """
        Prepare legend elements and a color map based on segmentation labels.
        """
        unique_labels = sorted(set(segmentation_data))
        cm = plt.get_cmap(colormap)
        unique_colors = [cm(1.0 * i / len(unique_labels)) for i in range(len(unique_labels))]
        color_map = dict(zip(unique_labels, unique_colors))
        legend_elements = [Patch(facecolor=color, edgecolor='none', label=label) for
                           label, color in zip(unique_labels, unique_colors)]
        return legend_elements, color_map

    @staticmethod
    def _fill_plot(ax, times, data, segmentation, color_map):
        """
        Fill the plot with EEG data and color-coded segmentation regions.
        """
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
        """
        Open a file dialog to export the current plot as an image file (PNG, JPG, etc.).
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)",
            options=options
        )
        if file_name:
            self._save_plot(file_name)

    def _save_plot(self, file_name):
        """
        Save the current plot to the specified file.
        """
        extension = os.path.splitext(file_name)[-1].lower()
        if not extension:
            file_name += '.png'
        self.canvas.figure.savefig(file_name)
