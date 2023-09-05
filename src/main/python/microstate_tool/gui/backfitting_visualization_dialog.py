import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QFileDialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from functions.data_utils.data_io import DataIO
from functions.backfitting_utils.segmentation_io import SegmentationIO


class BackfittingVisualizationDialog(QDialog):
    def __init__(self, context, parent=None, tbx=None):
        super().__init__(parent)

        self.tbx = tbx
        self._load_ui(context)
        self._initialize_ui()

    def _load_ui(self, context):
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("BackfittingVisualizationWindow.ui"), self)

    def _initialize_ui(self):
        self.ui.setWindowTitle("Visualization of the localized sources")
        self.ui.show_backfitting_button.clicked.connect(self.show_backfitting)
        self.ui.export_backfitting_image_button.clicked.connect(self.export_plot)
        self.resize(1000, 800)

    def show_backfitting(self):
        selected_file_name = self.ui.eeg_filenames_combobox.currentText()
        eeg_dir = os.path.join(self.preprocessed_data_path, f"{selected_file_name}{self.extension}")

        data_io = DataIO()
        eeg = data_io.load_eegs(eeg_dir, self.extension, self.datatype)
        eeg_data = eeg.get_data()
        eeg_times = eeg.times * 1000

        data_to_use = self._get_data_to_use(eeg_data)

        segmentation_io = SegmentationIO()
        segmentation_path = os.path.join(self.segmentation_path, f"{selected_file_name}{self.export_format}")
        segmentation_data = segmentation_io.load_segmentation(segmentation_path, import_format=self.export_format)

        time_min, time_max = self._get_time_range()
        fontsize = int(self.ui.font_size_input.text())
        labelsize = int(self.ui.label_size_input.text())
        colormap = self.ui.colormap_combobox.currentText()
        self._plot_data(eeg_times, data_to_use, segmentation_data, time_min, time_max, fontsize, labelsize, colormap)

    def _get_data_to_use(self, eeg_data):
        if self.datatype == 'epoched':
            trial = self.ui.num_trials_spinbox.value()
            return np.std(eeg_data[trial, :, :], axis=0)
        return np.std(eeg_data, axis=0)

    def _get_time_range(self):
        time_min = int(self.ui.xlim_min_input.text())
        time_max = int(self.ui.xlim_max_input.text())
        return time_min, time_max

    def _plot_data(self, eeg_times, data_to_use, segmentation_data, time_min, time_max, fontsize, labelsize, colormap):

        xmin = np.argmin(np.abs(eeg_times - time_min))
        xmax = np.argmin(np.abs(eeg_times - time_max))

        # Limit the data and times to the desired time range
        times_to_use = eeg_times[xmin:xmax]
        data_to_use = data_to_use[xmin:xmax]
        segmentation_to_use = segmentation_data[xmin:xmax]

        ax = self.ui.MplWidget.canvas.axes
        ax.clear()

        # Generate the plot
        legend_elements, color_map = self._prepare_legend_and_colors(segmentation_to_use, colormap)
        self._fill_plot(ax, times_to_use, data_to_use, segmentation_to_use, color_map, legend_elements)

        ax.set_xlabel('Time (ms)', fontsize=fontsize)
        ax.set_ylabel('Potential (μV)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.set_xlim([time_min, time_max])
        ax.legend(handles=legend_elements, loc='upper right', fontsize=fontsize)

        self.ui.MplWidget.canvas.draw()

    def _prepare_legend_and_colors(self, segmentation_data, colormap):
        unique_labels = sorted(set(segmentation_data))
        cm = plt.get_cmap(colormap)
        unique_colors = [cm(1. * i / len(unique_labels)) for i in range(len(unique_labels))]
        color_map = {label: color for label, color in zip(unique_labels, unique_colors)}
        legend_elements = [Patch(facecolor=color, edgecolor='none', label=label) for label, color in zip(unique_labels, unique_colors)]
        return legend_elements, color_map

    def _fill_plot(self, ax, times, data, segmentation, color_map, legend_elements):
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
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Plot", "",
                                                   "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)",
                                                   options=options)
        if file_name:
            self._save_plot(file_name)

    def _save_plot(self, file_name):
        extension = os.path.splitext(file_name)[-1].lower()
        if not extension:
            file_name += '.png'
        self.ui.MplWidget.canvas.figure.savefig(file_name)
