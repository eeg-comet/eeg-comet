import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QSizePolicy, QActionGroup, QInputDialog
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from data_utils.data_io import DataIO
from backfitting_utils.segmentation_io import SegmentationIO
from gui_utils.set_widgets_status import set_widgets_status
import logging

# Suppress matplotlib font warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


class BackfittingVisualizationWindow(QMainWindow):
    def __init__(self, context, parent=None):
        """
        Initialize the BackfittingVisualizationWindow.
        """
        super().__init__(parent)

        # Initialize variables
        self.datatype = ""
        self.preprocessed_data_path = ""
        self.segmentation_path = ""
        self.extension = ""
        self.export_format = ""
        self.current_eeg_times = None
        self.current_window_size = 10000  # Default window size in ms
        self.context = context

        # NEW: Add variables for global color mapping
        self.global_color_map = None
        self.all_microstate_labels = None

        # Load UI and initialize UI components
        self._load_ui()
        self._initialize_ui()

    def _load_ui(self):
        """
        Load the UI layout from the specified .ui file.
        """
        self.ui = uic.loadUi(self.context.get_resource("BackfittingVisualizationWindow.ui"), self)

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
        self.ui.eeg_filenames_combobox.currentTextChanged.connect(self.on_filename_changed)
        self.ui.num_trials_spinbox.valueChanged.connect(self.show_backfitting)

        # Connect time range inputs and slider
        self.ui.xlim_min_input.textChanged.connect(self.on_xlim_min_input_changed)
        self.ui.xlim_max_input.textChanged.connect(self.on_xlim_max_input_changed)
        self.ui.xlim_slider.valueChanged.connect(self.on_xlim_slider_changed)

        # Connect window size radio buttons
        self.ui.win500_radio.toggled.connect(self.on_window_size_changed)
        self.ui.win1000_radio.toggled.connect(self.on_window_size_changed)
        self.ui.win5000_radio.toggled.connect(self.on_window_size_changed)
        self.ui.win10000_radio.toggled.connect(self.on_window_size_changed)

        # Connect menu actions
        self.ui.export_backfitting_image_button.triggered.connect(self.export_plot)
        self.ui.export_backfitting_image_button.setShortcut("Ctrl+S")

        # Setup color map action group
        self.cmap_group = QActionGroup(self)
        self.cmap_group.addAction(self.ui.cmap_tab10)
        self.cmap_group.addAction(self.ui.cmap_set1)
        self.cmap_group.addAction(self.ui.cmap_set2)
        self.cmap_group.addAction(self.ui.cmap_dark2)
        self.cmap_group.addAction(self.ui.cmap_accent)
        self.cmap_group.triggered.connect(self.on_colormap_changed)

        # Connect font size action
        self.ui.actionFont_Size.triggered.connect(self.on_font_size_action)

        # Initialize default values
        self.current_colormap = 'tab10'
        self.font_size = 16
        self.label_size = 18

        # Set up preferred font family
        self.font_family = self._get_available_font()

        # Set default window size based on UI (1000ms is checked by default)
        self.current_window_size = self._get_window_size_from_radio()

        # Initialize UI state based on default settings
        self.backfitting_visualization_controller()

    def _get_available_font(self):
        """
        Get the best available font family from the preferred list.
        """
        preferred_fonts = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        for font in preferred_fonts:
            if font in available_fonts or font == 'sans-serif':
                return font

        return 'sans-serif'  # Ultimate fallback

    def _get_window_size_from_radio(self):
        """
        Get the current window size based on selected radio button.
        """
        if self.ui.win500_radio.isChecked():
            return 500
        elif self.ui.win1000_radio.isChecked():
            return 1000
        elif self.ui.win5000_radio.isChecked():
            return 5000
        elif self.ui.win10000_radio.isChecked():
            return 10000
        else:
            return 1000  # Default fallback

    def on_window_size_changed(self):
        """
        Handle window size radio button changes.
        """
        new_window_size = self._get_window_size_from_radio()
        if new_window_size != self.current_window_size:
            self.current_window_size = new_window_size

            # Update slider range and current window position
            if self.current_eeg_times is not None:
                self._update_slider_range()

                # Update the window end time display
                current_start = self.ui.xlim_slider.value()
                new_end = current_start + self.current_window_size

                # Check if new window exceeds data bounds
                time_max = int(self.current_eeg_times[-1])
                if new_end > time_max:
                    # Adjust start position to fit window within bounds
                    new_start = max(self.ui.xlim_slider.minimum(), time_max - self.current_window_size)
                    self.ui.xlim_slider.setValue(new_start)
                    new_end = new_start + self.current_window_size

                self.ui.xlim_min_input.setText(str(self.ui.xlim_slider.value()))
                self.ui.xlim_max_input.setText(str(new_end))

                # Update the plot
                self.show_backfitting()

    def set_data_paths(self, datatype, preprocessed_data_path, segmentation_path, extension, export_format):
        """
        Set the data paths and initialize the filename combobox.
        """
        self.datatype = datatype
        self.preprocessed_data_path = preprocessed_data_path
        self.segmentation_path = segmentation_path
        self.extension = extension
        self.export_format = export_format

        # Populate the filename combobox and auto-select first item
        self._populate_filename_combobox()

        # If there are files, automatically select the first one
        if self.ui.eeg_filenames_combobox.count() > 0:
            self.ui.eeg_filenames_combobox.setCurrentIndex(0)
            self.load_data_and_setup_ui()
            self.show_backfitting()

    def _populate_filename_combobox(self):
        """
        Populate the filename combobox with available EEG files.
        """
        try:
            data_io = DataIO()
            eeg_files, _ = data_io.find_data(
                input_folder=self.preprocessed_data_path,
                extension=self.extension,
                pattern="*"
            )

            # Extract just the filenames without path and extension
            filenames = []
            for file_path in eeg_files:
                filename = os.path.basename(file_path)
                # Remove extension
                if filename.endswith(self.extension):
                    filename = filename[:-len(self.extension)]
                filenames.append(filename)

            # Clear and populate combobox
            self.ui.eeg_filenames_combobox.clear()
            self.ui.eeg_filenames_combobox.addItems(filenames)

        except Exception as e:
            print(f"Error populating filename combobox: {e}")

    def on_filename_changed(self):
        """
        Handle changes to the selected filename.
        """
        # Only load data if paths are properly initialized
        if self.ui.eeg_filenames_combobox.count() > 0 and self._are_paths_initialized():
            self.load_data_and_setup_ui()
            self.show_backfitting()

    def _are_paths_initialized(self):
        """
        Check if all required data paths have been set.
        """
        return all([
            self.datatype,
            self.preprocessed_data_path,
            self.segmentation_path,
            self.extension,
            self.export_format
        ])

    def load_data_and_setup_ui(self):
        """
        Load data for the selected file and set up UI ranges.
        """
        selected_file_name = self.ui.eeg_filenames_combobox.currentText()
        if not selected_file_name:
            return

        # Check if required paths are set
        if not all([self.datatype, self.preprocessed_data_path, self.segmentation_path,
                    self.extension, self.export_format]):
            print("Error: Data paths not properly initialized. Call set_data_paths() first.")
            return

        try:
            _, _, _, eeg_times, segmentation_data = self._load_data_and_segmentation(selected_file_name)
            self.current_eeg_times = eeg_times

            # NEW: Establish global color mapping based on all microstates in the dataset
            self._establish_global_color_mapping(segmentation_data)

            # Set up time range for slider - slider controls the start position
            time_min = int(eeg_times[0])
            time_max = int(eeg_times[-1])

            # For continuous/raw data, ensure no negative values
            if self.datatype == 'raw':
                slider_min = max(0, time_min)
            else:
                # For epoched data, allow negative values (pre-stimulus)
                slider_min = time_min

            # Slider max should account for window size so window doesn't exceed data bounds
            slider_max = time_max - self.current_window_size
            if slider_max < slider_min:
                # If data is shorter than window size, use minimum possible window size
                min_possible_window = min(500, time_max - slider_min)
                if min_possible_window > 0:
                    self.current_window_size = min_possible_window
                    # Update the appropriate radio button
                    if min_possible_window <= 500:
                        self.ui.win500_radio.setChecked(True)
                    else:
                        self.ui.win1000_radio.setChecked(True)
                slider_max = time_max - self.current_window_size

            # Ensure we have a valid range
            if slider_max < slider_min:
                slider_max = slider_min

            self.ui.xlim_slider.setRange(slider_min, slider_max)

            # Set default start position
            default_start = max(0, slider_min) if self.datatype == 'raw' else slider_min
            default_end = default_start + self.current_window_size

            self.ui.xlim_slider.setValue(default_start)
            self.ui.xlim_min_input.setText(str(default_start))
            self.ui.xlim_max_input.setText(str(default_end))

            # Update trial spinbox for epoched data
            if self.datatype == 'epoched':
                num_trials = len(segmentation_data)
                self.ui.num_trials_spinbox.setRange(1, num_trials)

        except Exception as e:
            print(f"Error loading data: {e}")

    def _establish_global_color_mapping(self, segmentation_data):
        """
        Establish a global color mapping based on all microstates in the entire dataset.
        """
        # Get all unique microstate labels from the entire dataset
        if self.datatype == 'epoched':
            # For epoched data, segmentation_data is 2D (trials x timepoints)
            all_labels = set()
            for trial in segmentation_data:
                all_labels.update(trial)
        else:
            # For continuous data, segmentation_data is 1D wrapped in a list
            all_labels = set(segmentation_data[0])

        # Store sorted labels for consistent ordering
        self.all_microstate_labels = sorted(all_labels)

        # Create global color mapping
        cm = plt.get_cmap(self.current_colormap)
        colors = [cm(1.0 * i / len(self.all_microstate_labels)) for i in range(len(self.all_microstate_labels))]
        self.global_color_map = dict(zip(self.all_microstate_labels, colors))

    def on_xlim_min_input_changed(self, text):
        """
        Handle changes to the minimum time input.
        Since inputs are now read-only, this mainly serves as a validator.
        """
        # Input fields are now read-only, so this is mainly for validation
        # The actual window control is through radio buttons and slider
        pass

    def on_xlim_max_input_changed(self, text):
        """
        Handle changes to the maximum time input.
        Since inputs are now read-only, this mainly serves as a validator.
        """
        # Input fields are now read-only, so this is mainly for validation
        # The actual window control is through radio buttons and slider
        pass

    def on_xlim_slider_changed(self, value):
        """
        Handle changes to the slider position (window start).
        """
        if self.current_eeg_times is not None:
            window_start = value
            window_end = window_start + self.current_window_size

            # Update input fields
            self.ui.xlim_min_input.setText(str(window_start))
            self.ui.xlim_max_input.setText(str(window_end))

            # Update the plot
            self.show_backfitting()

    def _update_slider_range(self):
        """
        Update slider range based on current window size and data bounds.
        """
        if self.current_eeg_times is not None:
            time_min = int(self.current_eeg_times[0])
            time_max = int(self.current_eeg_times[-1])

            if self.datatype == 'raw':
                slider_min = max(0, time_min)
            else:
                slider_min = time_min

            slider_max = time_max - self.current_window_size
            if slider_max < slider_min:
                slider_max = slider_min

            self.ui.xlim_slider.setRange(slider_min, slider_max)

    def on_colormap_changed(self, action):
        """
        Handle colormap selection changes.
        """
        colormap_map = {
            self.ui.cmap_tab10: 'tab10',
            self.ui.cmap_set1: 'Set1',
            self.ui.cmap_set2: 'Set2',
            self.ui.cmap_dark2: 'Dark2',
            self.ui.cmap_accent: 'Accent'
        }
        self.current_colormap = colormap_map.get(action, 'tab10')

        # NEW: Update global color mapping with new colormap
        if self.all_microstate_labels is not None:
            cm = plt.get_cmap(self.current_colormap)
            colors = [cm(1.0 * i / len(self.all_microstate_labels)) for i in range(len(self.all_microstate_labels))]
            self.global_color_map = dict(zip(self.all_microstate_labels, colors))

        self.show_backfitting()

    def on_font_size_action(self):
        """
        Handle font size change request.
        """
        font_size, ok = QInputDialog.getInt(self, "Font Size", "Enter font size:",
                                            self.font_size, 8, 24, 1)
        if ok:
            self.font_size = font_size
            self.label_size = max(8, font_size - 2)
            self.show_backfitting()

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

        # Enable or disable epoched data widgets based on datatype
        if self.datatype == 'epoched':
            set_widgets_status(epoched_data_widgets, mode='enable')
            set_widgets_status(epoched_data_widgets, mode='show')
        else:
            set_widgets_status(epoched_data_widgets, mode='disable')
            set_widgets_status(epoched_data_widgets, mode='hide')

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

        # Validate export format before loading
        if not self.export_format:
            raise ValueError("Export format is not set. Please set export_format before loading data.")

        segmentation_data = segmentation_io.load_segmentation(segmentation_path, import_format=self.export_format)

        return eeg_data, eeg_info, gfp_data, eeg_times, segmentation_data

    def _get_gfp_data(self, eeg_data):
        """
        Determine the EEG data to use for plotting based on the current datatype.
        """
        if self.datatype == 'epoched':
            trial = self.ui.num_trials_spinbox.value()
            gfp = np.std(eeg_data[trial - 1, :, :], axis=0)
        else:
            gfp = np.std(eeg_data, axis=0)

        # Scale from V to μV (multiply by 1e6)
        return gfp * 1e6

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
        return self.font_size, self.label_size, self.current_colormap

    def show_backfitting(self):
        """
        Handle the logic for displaying backfitting data by loading the selected EEG and segmentation data,
        retrieving plot parameters, and rendering the plot.
        """
        selected_file_name = self.ui.eeg_filenames_combobox.currentText()
        if not selected_file_name:
            return

        # Check if paths are initialized before proceeding
        if not self._are_paths_initialized():
            print("Warning: Data paths not properly initialized. Call set_data_paths() first.")
            return

        try:
            # Load EEG and segmentation data
            eeg_data, eeg_info, gfp_data, eeg_times, segmentation_data = self._load_data_and_segmentation(
                selected_file_name)

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
            self._plot_data(eeg2plot, eeg_info, eeg_times, gfp_data, segmentation2plot,
                            time_min, time_max, fontsize, labelsize, colormap)
        except Exception as e:
            print(f"Error showing backfitting: {e}")

    def _plot_data(self, eeg_data, eeg_info, eeg_times, gfp_data, segmentation_data,
                   time_min, time_max, fontsize, labelsize, colormap):
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
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Add a dashed horizontal line at y=0 if self.datatype is 'epoched'
        if self.datatype == 'epoched':
            ax.axvline(0, color='red', linestyle='--')

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

        # Customize plot appearance with available font
        ax.set_xlabel('Time (ms)', fontsize=fontsize, fontfamily=self.font_family)
        ax.set_ylabel('Global Field Power (μV)', fontsize=fontsize, fontfamily=self.font_family)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.set_xlim([time_min, time_max])

        # Set font for legend
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=fontsize)
        legend.set_title(None)
        for text in legend.get_texts():
            text.set_fontfamily(self.font_family)

        # Set font for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(self.font_family)

        # Render the plot on the canvas
        self.canvas.draw()

    def _prepare_legend_and_colors(self, segmentation_data, colormap):
        """
        Prepare legend elements and use the global color map.
        """
        # NEW: Use global color mapping instead of window-specific mapping
        if self.global_color_map is None:
            # Fallback to old behavior if global mapping not established
            unique_labels = sorted(set(segmentation_data))
            cm = plt.get_cmap(colormap)
            unique_colors = [cm(1.0 * i / len(unique_labels)) for i in range(len(unique_labels))]
            color_map = dict(zip(unique_labels, unique_colors))
        else:
            color_map = self.global_color_map
            unique_labels = self.all_microstate_labels

        # Create legend elements for all microstates (not just those in current window)
        legend_elements = [
            Patch(facecolor=color_map[label], edgecolor='none', label=f"Microstate {label}")
            for label in unique_labels
        ]

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
                ax.fill_between(times[prev_index:idx], data[prev_index:idx], color=segment_color, alpha=0.5)
                prev_index = idx

        # Handle the last segment
        segment_color = color_map.get(segmentation[-1], 'grey')
        ax.plot(times[prev_index:], data[prev_index:], color='black')
        ax.fill_between(times[prev_index:], data[prev_index:], color=segment_color, alpha=0.5)

    def export_plot(self):
        """
        Open a file dialog to export the current plot in vector or raster format.
        """
        # Get current filename for default save name
        current_filename = self.ui.eeg_filenames_combobox.currentText()
        default_name = f"{current_filename}_backfitting.pdf" if current_filename else "backfitting_plot.pdf"

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            default_name,
            "PDF Files (*.pdf);;PNG Files (*.png);;JPG Files (*.jpg);;SVG Files (*.svg);;All Files (*)",
            options=options
        )
        if file_name:
            self._save_plot(file_name)

    def _save_plot(self, file_name):
        """
        Save the current plot to the specified file with title and format-appropriate settings.
        """
        # Get file extension
        extension = os.path.splitext(file_name)[-1].lower()

        # Ensure file has an extension, defaulting to PDF for vector format
        if not extension:
            file_name += '.pdf'
            extension = '.pdf'

        # Add title to the figure
        current_filename = self.ui.eeg_filenames_combobox.currentText()
        if current_filename:
            self.figure.suptitle(current_filename, fontsize=self.font_size + 2,
                                 fontweight='bold', fontfamily=self.font_family)

        # Save with format-appropriate settings
        if extension in ['.pdf', '.svg']:
            # Vector formats - high quality with clean settings
            self.canvas.figure.savefig(file_name, format=extension[1:],
                                       dpi=300, bbox_inches='tight',
                                       facecolor='white', edgecolor='none')
        else:
            # Raster formats - high DPI for quality
            self.canvas.figure.savefig(file_name, dpi=300, bbox_inches='tight')

        # Remove title after saving to avoid cluttering the display
        self.figure.suptitle('')
        self.canvas.draw()
