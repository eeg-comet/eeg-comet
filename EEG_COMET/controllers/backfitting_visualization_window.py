"""Backfitting visualization window for displaying segmentation overlays and topomaps."""

import logging
import os.path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QActionGroup, QInputDialog, QMainWindow

from backfitting_utils.segmentation_io import SegmentationIO
from data_utils.data_io import DataIO
from gui_utils.export_utils import get_save_file_path, save_matplotlib_figure
from gui_utils.responsive import apply_window_minimum, expand_canvas
from gui_utils.set_widgets_status import set_widgets_status

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class BackfittingVisualizationWindow(QMainWindow):
    """Interactive window to visualize backfitting results.

    Displays Global Field Power (GFP) time-series with color-coded microstate
    segmentation overlays and provides export utilities. Supports both
    continuous (raw) and epoched data and maintains a global microstate color
    mapping across the dataset for legend consistency.

    Attributes:
      datatype (str): Data type, either "raw" or "epoched".
      preprocessed_data_path (str): Folder containing preprocessed EEG files.
      segmentation_path (str): Folder containing segmentation label files.
      extension (str): EEG file extension (for discovery).
      export_format (str): Segmentation export format/extension.
      current_eeg_times (np.ndarray | None): Time vector (ms) of the loaded EEG.
      current_window_size (int): Current view window size in milliseconds.
      context: Resource/context provider for resolving UI resources.
      global_color_map (dict[int, tuple] | None): Global label-to-color mapping.
      all_microstate_labels (list[int] | None): Sorted list of labels in dataset.
      current_colormap (str): Name of the active matplotlib colormap.
      font_size (int): Base font size used for labels/legend.
      label_size (int): Tick label font size.
      font_family (str): Preferred font family.
    """

    def __init__(self, context, parent=None):
        """Construct the window and initialize UI state.

        Args:
          context: Resource/context provider used to resolve UI files and settings.
          parent: Optional parent widget.
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

        # Initialize UI-related instance attributes
        self.current_colormap = "tab10"
        self.font_size = 16
        self.label_size = 18
        self.font_family = "sans-serif"

        # Load UI and initialize UI components
        self._load_ui()
        self._initialize_ui()

    def _load_ui(self):
        """Load the Qt Designer `.ui` layout for this window.

        Returns:
          None
        """
        self.ui = uic.loadUi(self.context.get_resource("BackfittingVisualizationWindow.ui"), self)

    def _initialize_ui(self):
        """Create the canvas, wire connections, and sync initial UI state.

        Sets up the matplotlib canvas, connects signals, chooses a font
        family, initializes the default window size, and shows the initial
        instruction message.

        Returns:
          None
        """
        self.ui.setWindowTitle("Visualization of the backfitted microstates")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        apply_window_minimum(self, "tool")
        if hasattr(self.ui, "backfitting_visualization_label"):
            self.ui.backfitting_visualization_label.setProperty("role", "banner")

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        expand_canvas(self.canvas, minimum=(360, 240))
        self.ui.Figure_Layout.addWidget(self.canvas)

        # Connect UI signals and menu actions
        self._setup_connections()

        # Set up preferred font family
        self.font_family = self._get_available_font()

        # Set default window size based on UI (1000ms is checked by default)
        self.current_window_size = self._get_window_size_from_radio()

        # Show initial instruction message
        self._show_initial_message()

        # Initialize UI state based on default settings
        self.backfitting_visualization_controller()

        # Set up checkable font styling
        self.setup_checkable_font_styling()

    def update_widget_font_weight(self, checked_or_widget=None):
        """Update font weight of a radio button or checkbox based on its checked state.

        Args:
            checked_or_widget: Either a boolean (from toggled signal) or a widget object.
                              If boolean or None, uses self.sender() to get the widget.

        Makes the widget bold when checked, normal when unchecked.
        """
        if checked_or_widget is None or isinstance(checked_or_widget, bool):
            widget = self.sender()
        else:
            widget = checked_or_widget

        if widget is None:
            return

        if widget.isChecked():
            widget.setStyleSheet("font-weight: bold;")
        else:
            widget.setStyleSheet("")

    def setup_checkable_font_styling(self):
        """Set up font weight styling for all radio buttons and checkboxes.

        Connects toggled signal to update font weight and initializes current states.
        """
        checkable_widgets = [
            "win500_radio",
            "win1000_radio",
            "win5000_radio",
            "win10000_radio",
        ]

        for widget_name in checkable_widgets:
            widget = getattr(self.ui, widget_name, None)
            if widget is not None:
                widget.toggled.connect(self.update_widget_font_weight)
                self.update_widget_font_weight(widget)

    def _setup_connections(self):
        """Connect UI widgets and actions to their handlers.

        Wires core selectors, time-range controls, window-size radio buttons,
        export action, colormap action group, and font-size action.

        Returns:
          None
        """
        # Core selectors
        self.ui.eeg_filenames_combobox.currentTextChanged.connect(self.on_filename_changed)
        self.ui.num_trials_spinbox.valueChanged.connect(self.show_backfitting)

        # Time range controls
        self.ui.xlim_min_input.textChanged.connect(self.on_xlim_min_input_changed)
        self.ui.xlim_max_input.textChanged.connect(self.on_xlim_max_input_changed)
        self.ui.xlim_slider.valueChanged.connect(self.on_xlim_slider_changed)

        # Window size radios
        self.ui.win500_radio.toggled.connect(self.on_window_size_changed)
        self.ui.win1000_radio.toggled.connect(self.on_window_size_changed)
        self.ui.win5000_radio.toggled.connect(self.on_window_size_changed)
        self.ui.win10000_radio.toggled.connect(self.on_window_size_changed)

        # Export
        self.ui.export_backfitting_image_button.triggered.connect(self.export_plot)
        self.ui.export_backfitting_image_button.setShortcut("Ctrl+S")

        # Colormap action group
        self.cmap_group = QActionGroup(self)
        self.cmap_group.addAction(self.ui.cmap_tab10)
        self.cmap_group.addAction(self.ui.cmap_set1)
        self.cmap_group.addAction(self.ui.cmap_set2)
        self.cmap_group.addAction(self.ui.cmap_dark2)
        self.cmap_group.addAction(self.ui.cmap_accent)
        self.cmap_group.triggered.connect(self.on_colormap_changed)

        # Font size action
        self.ui.actionFont_Size.triggered.connect(self.on_font_size_action)

    @staticmethod
    def _get_available_font():
        """Return the first available font from a preferred list.

        Returns:
          str: Chosen font family name; falls back to "sans-serif".
        """
        preferred_fonts = ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        for font in preferred_fonts:
            if font in available_fonts or font == "sans-serif":
                return font

        return "sans-serif"  # Ultimate fallback

    def _get_window_size_from_radio(self):
        """Return the currently selected window size in milliseconds.

        Returns:
          int: Window size in milliseconds.
        """
        if self.ui.win500_radio.isChecked():
            return 500
        if self.ui.win1000_radio.isChecked():
            return 1000
        if self.ui.win5000_radio.isChecked():
            return 5000
        if self.ui.win10000_radio.isChecked():
            return 10000
        return 1000  # Default fallback

    def _show_initial_message(self):
        """Display an initial message instructing the user to select a file.

        Returns:
          None
        """
        # Clear any existing plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No EEG file selected.\n"
            "Please choose a file from the dropdown menu above to visualize the backfitted microstate maps over time.",
            ha="center",
            va="center",
            fontsize=self.font_size + 2,
            fontfamily=self.font_family,
            fontweight="bold",
            wrap=True,
        )
        ax.axis("off")
        self.canvas.draw()

    def on_window_size_changed(self):
        """React to changes in window size radio buttons and update the view.

        Returns:
          None
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
                    new_start = max(
                        self.ui.xlim_slider.minimum(), time_max - self.current_window_size
                    )
                    self.ui.xlim_slider.setValue(new_start)
                    new_end = new_start + self.current_window_size

                self.ui.xlim_min_input.setText(str(self.ui.xlim_slider.value()))
                self.ui.xlim_max_input.setText(str(new_end))

                # Update the plot
                self.show_backfitting()

    def set_data_paths(
        self, datatype, preprocessed_data_path, segmentation_path, extension, export_format
    ):
        """Configure data locations and initialize filename selection.

        Args:
          datatype (str): "raw" for continuous data or "epoched" for trials.
          preprocessed_data_path (str): Folder containing preprocessed EEG files.
          segmentation_path (str): Folder containing segmentation label files.
          extension (str): EEG file extension (e.g., ".fif").
          export_format (str): Segmentation export format (e.g., ".csv", ".pkl").

        Returns:
          None
        """
        self.datatype = datatype
        self.preprocessed_data_path = preprocessed_data_path
        self.segmentation_path = segmentation_path
        self.extension = extension
        self.export_format = export_format

        # Refresh UI visibility based on datatype
        self.backfitting_visualization_controller()

        # Populate the filename combobox and auto-select first item
        self._populate_filename_combobox()

        # If there are files, automatically select the first one
        if self.ui.eeg_filenames_combobox.count() > 0:
            self.ui.eeg_filenames_combobox.setCurrentIndex(0)
            self.load_data_and_setup_ui()
            self.show_backfitting()

    def _populate_filename_combobox(self):
        """Populate the filename combobox with available EEG basenames.

        Scans the `preprocessed_data_path` for files matching `extension` and
        lists basenames (without extension) for selection.

        Returns:
          None
        """
        try:
            data_io = DataIO()
            eeg_files, _ = data_io.find_data(
                input_folder=self.preprocessed_data_path, extension=self.extension, pattern="*"
            )

            # Extract just the filenames without path and extension
            filenames = []
            for file_path in eeg_files:
                filename = os.path.basename(file_path)
                # Remove extension
                if filename.endswith(self.extension):
                    filename = filename[: -len(self.extension)]
                filenames.append(filename)

            # Clear and populate combobox
            self.ui.eeg_filenames_combobox.clear()
            self.ui.eeg_filenames_combobox.addItems(filenames)

        except Exception as e:
            print(f"Error populating filename combobox: {e}")

    def on_filename_changed(self):
        """Handle changes to the selected filename by reloading and plotting.

        Returns:
          None
        """
        # Only load data if paths are properly initialized
        if self.ui.eeg_filenames_combobox.count() > 0 and self._are_paths_initialized():
            self.load_data_and_setup_ui()
            self.show_backfitting()

    def _are_paths_initialized(self):
        """Return True if all required configuration paths are set.

        Returns:
          bool: True if all paths are configured; otherwise False.
        """
        return all(
            [
                self.datatype,
                self.preprocessed_data_path,
                self.segmentation_path,
                self.extension,
                self.export_format,
            ]
        )

    def load_data_and_setup_ui(self):
        """Load data for the selected file and configure UI ranges.

        Loads EEG timing and segmentation to set the slider range and trial
        spinbox (for epoched data). Also establishes a global microstate color
        mapping across the dataset to keep legend colors stable.

        Returns:
          None
        """
        selected_file_name = self.ui.eeg_filenames_combobox.currentText()
        if not selected_file_name:
            return

        # Check if required paths are set
        if not all(
            [
                self.datatype,
                self.preprocessed_data_path,
                self.segmentation_path,
                self.extension,
                self.export_format,
            ]
        ):
            print("Error: Data paths not properly initialized. Call set_data_paths() first.")
            return

        try:
            _, _, _, eeg_times, segmentation_data = self._load_data_and_segmentation(
                selected_file_name
            )
            self.current_eeg_times = eeg_times

            # NEW: Establish global color mapping based on all microstates in the dataset
            self._establish_global_color_mapping(segmentation_data)

            # Set up time range for slider - slider controls the start position
            time_min = int(eeg_times[0])
            time_max = int(eeg_times[-1])

            # For continuous/raw data, ensure no negative values
            slider_min = max(0, time_min) if self.datatype == "raw" else time_min

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
            default_start = max(0, slider_min) if self.datatype == "raw" else slider_min
            default_end = default_start + self.current_window_size

            self.ui.xlim_slider.setValue(default_start)
            self.ui.xlim_min_input.setText(str(default_start))
            self.ui.xlim_max_input.setText(str(default_end))

            # Update trial spinbox for epoched data
            if self.datatype == "epoched":
                num_trials = len(segmentation_data)
                self.ui.num_trials_spinbox.setRange(1, num_trials)

        except Exception as e:
            print(f"Error loading data: {e}")

    def _establish_global_color_mapping(self, segmentation_data):
        """Compute a global microstate color mapping for the dataset.

        Args:
          segmentation_data (list[np.ndarray] | np.ndarray): Labels per time
            point for all trials (epoched) or a single array (continuous), used
            to enumerate all labels present in the study.

        Returns:
          None
        """
        # Get all unique microstate labels from the entire dataset
        if self.datatype == "epoched":
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
        colors = [
            cm(1.0 * i / len(self.all_microstate_labels))
            for i in range(len(self.all_microstate_labels))
        ]
        self.global_color_map = dict(zip(self.all_microstate_labels, colors))

    def on_xlim_min_input_changed(self, text):
        """No-op validator for minimum time input (read-only in UI).

        Args:
          text (str): Current text value (ignored).

        Returns:
          None
        """
        # Input fields are now read-only, so this is mainly for validation
        # The actual window control is through radio buttons and slider
        pass

    def on_xlim_max_input_changed(self, text):
        """No-op validator for maximum time input (read-only in UI).

        Args:
          text (str): Current text value (ignored).

        Returns:
          None
        """
        # Input fields are now read-only, so this is mainly for validation
        # The actual window control is through radio buttons and slider
        pass

    def on_xlim_slider_changed(self, value):
        """Update start/end fields and replot when the window slider moves.

        Args:
          value (int): New window start (ms).

        Returns:
          None
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
        """Recompute slider range from data bounds and current window size.

        Returns:
          None
        """
        if self.current_eeg_times is not None:
            time_min = int(self.current_eeg_times[0])
            time_max = int(self.current_eeg_times[-1])

            slider_min = max(0, time_min) if self.datatype == "raw" else time_min

            slider_max = time_max - self.current_window_size
            if slider_max < slider_min:
                slider_max = slider_min

            self.ui.xlim_slider.setRange(slider_min, slider_max)

    def on_colormap_changed(self, action):
        """Apply a new colormap and refresh the global color mapping.

        Args:
          action: Triggered QAction representing the chosen colormap.

        Returns:
          None
        """
        colormap_map = {
            self.ui.cmap_tab10: "tab10",
            self.ui.cmap_set1: "Set1",
            self.ui.cmap_set2: "Set2",
            self.ui.cmap_dark2: "Dark2",
            self.ui.cmap_accent: "Accent",
        }
        self.current_colormap = colormap_map.get(action, "tab10")

        # NEW: Update global color mapping with new colormap
        if self.all_microstate_labels is not None:
            cm = plt.get_cmap(self.current_colormap)
            colors = [
                cm(1.0 * i / len(self.all_microstate_labels))
                for i in range(len(self.all_microstate_labels))
            ]
            self.global_color_map = dict(zip(self.all_microstate_labels, colors))

        self.show_backfitting()

    def on_font_size_action(self):
        """Prompt for a new font size and refresh the plot if confirmed.

        Returns:
          None
        """
        font_size, ok = QInputDialog.getInt(
            self, "Font Size", "Enter font size:", self.font_size, 8, 24, 1
        )
        if ok:
            self.font_size = font_size
            self.label_size = max(8, font_size - 2)
            self.show_backfitting()

    def backfitting_visualization_controller(self):
        """Toggle epoched-only widgets based on current `datatype`.

        Returns:
          None
        """
        # Widgets related to epoched data
        epoched_data_widgets = [self.ui.num_trials_label, self.ui.num_trials_spinbox]

        # Enable or disable epoched data widgets based on datatype
        if self.datatype == "epoched":
            set_widgets_status(epoched_data_widgets, mode="enable")
            set_widgets_status(epoched_data_widgets, mode="show")
        else:
            set_widgets_status(epoched_data_widgets, mode="disable")
            set_widgets_status(epoched_data_widgets, mode="hide")

    def _load_data_and_segmentation(self, selected_file_name):
        """Load EEG and corresponding segmentation arrays for a file.

        Args:
          selected_file_name (str): Basename (without extension) of the EEG file to load.

        Returns:
          tuple: (eeg_data, eeg_info, eeg_times_ms, segmentation_data) with EEG data,
          MNE `info`, time vector in ms, and segmentation labels.

        Raises:
          ValueError: If `export_format` is not set.
        """
        # Initialize DataIO to load EEG data
        data_io = DataIO()
        eeg_dir, _ = data_io.find_data(
            input_folder=self.preprocessed_data_path,
            extension=self.extension,
            pattern=f"*{selected_file_name}*",
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
            raise ValueError(
                "Export format is not set. Please set export_format before loading data."
            )

        segmentation_data = segmentation_io.load_segmentation(
            segmentation_path, import_format=self.export_format
        )

        return eeg_data, eeg_info, gfp_data, eeg_times, segmentation_data

    def _get_gfp_data(self, eeg_data):
        """Compute GFP (µV) from EEG data according to current `datatype`.

        For epoched data, uses the trial selected by the spinbox.

        Args:
          eeg_data (np.ndarray): Continuous (n_channels, n_times) or epoched
            (n_trials, n_channels, n_times) EEG array.

        Returns:
          np.ndarray: GFP in microvolts for each time point of the selected data.
        """
        if self.datatype == "epoched":
            trial = self.ui.num_trials_spinbox.value()
            gfp = np.std(eeg_data[trial - 1, :, :], axis=0)
        else:
            gfp = np.std(eeg_data, axis=0)

        # Scale from V to μV (multiply by 1e6)
        return gfp * 1e6

    def _get_time_range(self):
        """Return the current [min, max] time window in milliseconds.

        Returns:
          tuple[int, int]: (time_min_ms, time_max_ms).
        """
        time_min = int(self.ui.xlim_min_input.text())
        time_max = int(self.ui.xlim_max_input.text())
        return time_min, time_max

    def _get_plot_parameters(self):
        """Return current plot styling parameters.

        Returns:
          tuple[int, int, str]: (font_size, tick_label_size, colormap_name).
        """
        return self.font_size, self.label_size, self.current_colormap

    def show_backfitting(self):
        """Load data for the selected file and render the backfitting view.

        Returns:
          None
        """
        selected_file_name = self.ui.eeg_filenames_combobox.currentText()
        if not selected_file_name:
            self._show_initial_message()
            return

        # Check if paths are initialized before proceeding
        if not self._are_paths_initialized():
            self._show_initial_message()
            print("Warning: Data paths not properly initialized. Call set_data_paths() first.")
            return

        try:
            # Load EEG and segmentation data
            eeg_data, eeg_info, gfp_data, eeg_times, segmentation_data = (
                self._load_data_and_segmentation(selected_file_name)
            )

            # Get plot parameters
            time_min, time_max = self._get_time_range()
            if self.datatype == "epoched":
                trial = self.ui.num_trials_spinbox.value()
                eeg2plot = eeg_data[trial - 1, :, :]
                segmentation2plot = segmentation_data[trial - 1, :]
            else:
                eeg2plot = eeg_data
                segmentation2plot = segmentation_data[0]
            fontsize, labelsize, colormap = self._get_plot_parameters()

            # Plot the data
            self._plot_data(
                eeg2plot,
                eeg_times,
                gfp_data,
                segmentation2plot,
                time_min,
                time_max,
                fontsize,
                labelsize,
                colormap,
            )
        except Exception as e:
            print(f"Error showing backfitting: {e}")
            self._show_initial_message()

    def _plot_data(
        self,
        eeg_data,
        eeg_times,
        gfp_data,
        segmentation_data,
        time_min,
        time_max,
        fontsize,
        labelsize,
        colormap,
    ):
        """Plot GFP with segmentation overlays for a selected time window.

        Slices signals to the given [time_min, time_max] bounds, fills the area
        under the GFP curve with colors determined by the segmentation labels,
        and styles axes/legend using the current font settings.

        Args:
          eeg_data (np.ndarray): EEG data (n_channels, n_times) for the view.
          eeg_times (np.ndarray): Time vector in milliseconds (n_times,).
          gfp_data (np.ndarray): Global Field Power in µV (n_times,).
          segmentation_data (np.ndarray): Microstate labels per sample (n_times,).
          time_min (int): Lower bound of the window in milliseconds.
          time_max (int): Upper bound of the window in milliseconds.
          fontsize (int): Font size for labels/legend.
          labelsize (int): Tick label font size.
          colormap (str): Matplotlib colormap name if no global map exists.

        Returns:
          None
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
        if self.datatype == "epoched":
            ax.axvline(0, color="red", linestyle="--")

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
        ax.set_xlabel("Time (ms)", fontsize=fontsize, fontfamily=self.font_family)
        ax.set_ylabel("Global Field Power (μV)", fontsize=fontsize, fontfamily=self.font_family)
        ax.tick_params(axis="both", which="major", labelsize=labelsize)
        ax.set_xlim([time_min, time_max])

        # Set font for legend
        legend = ax.legend(handles=legend_elements, loc="upper right", fontsize=fontsize)
        legend.set_title(None)
        for text in legend.get_texts():
            text.set_fontfamily(self.font_family)

        # Set font for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(self.font_family)

        # Render the plot on the canvas
        self.canvas.draw()

    def _prepare_legend_and_colors(self, segmentation_data, colormap):
        """Return legend handles and the color map for microstate labels.

        Uses the global color map when available; otherwise derives colors from
        the provided colormap limited to labels present in the current window.

        Args:
          segmentation_data (np.ndarray): Labels in the current window (n_times,).
          colormap (str): Matplotlib colormap name used when no global map exists.

        Returns:
          tuple[list[matplotlib.patches.Patch], dict[int, tuple]]: Legend elements
          and label-to-color mapping.
        """
        # Prefer the shared global color mapping; build a window-local map
        # only when no global mapping has been established yet.
        if self.global_color_map is None:
            unique_labels = sorted(set(segmentation_data))
            cm = plt.get_cmap(colormap)
            unique_colors = [cm(1.0 * i / len(unique_labels)) for i in range(len(unique_labels))]
            color_map = dict(zip(unique_labels, unique_colors))
        else:
            color_map = self.global_color_map
            unique_labels = self.all_microstate_labels

        # Create legend elements for all microstates (not just those in current window)
        legend_elements = [
            Patch(facecolor=color_map[label], edgecolor="none", label=f"Microstate {label}")
            for label in unique_labels
        ]

        return legend_elements, color_map

    @staticmethod
    def _fill_plot(ax, times, data, segmentation, color_map):
        """Plot GFP and fill segments with label-specific colors.

        Args:
          ax (matplotlib.axes.Axes): Target axes.
          times (np.ndarray): Time vector in milliseconds.
          data (np.ndarray): GFP values (µV) aligned to `times`.
          segmentation (np.ndarray): Integer label per time point.
          color_map (dict): Mapping from label to RGBA color.

        Returns:
          None
        """
        prev_index = 0
        for idx, label in enumerate(segmentation[1:], start=1):
            if label != segmentation[idx - 1]:
                segment_color = color_map.get(segmentation[idx - 1], "grey")
                ax.plot(times[prev_index:idx], data[prev_index:idx], color="black")
                ax.fill_between(
                    times[prev_index:idx], data[prev_index:idx], color=segment_color, alpha=0.5
                )
                prev_index = idx

        # Handle the last segment
        segment_color = color_map.get(segmentation[-1], "grey")
        ax.plot(times[prev_index:], data[prev_index:], color="black")
        ax.fill_between(times[prev_index:], data[prev_index:], color=segment_color, alpha=0.5)

    def export_plot(self):
        """Open a save dialog and export the current plot as vector or raster.

        Returns:
          None
        """
        # Get current filename for default save name
        current_filename = self.ui.eeg_filenames_combobox.currentText()
        default_name = (
            f"{current_filename}_backfitting.pdf" if current_filename else "backfitting_plot.pdf"
        )

        file_name = get_save_file_path(self, default_name, "Export Plot")
        if file_name:
            title_text = current_filename if current_filename else None
            save_matplotlib_figure(
                figure=self.figure,
                canvas=self.canvas,
                file_name=file_name,
                title_text=title_text,
                title_fontsize=self.font_size + 2,
                font_family=self.font_family,
            )

    def _save_plot(self, file_name):
        """Save the current plot to ``file_name`` using the standard helper.

        Args:
          file_name (str): Target output path.

        Returns:
          None
        """
        current_filename = self.ui.eeg_filenames_combobox.currentText()
        title_text = current_filename if current_filename else None
        save_matplotlib_figure(
            figure=self.figure,
            canvas=self.canvas,
            file_name=file_name,
            title_text=title_text,
            title_fontsize=self.font_size + 2,
            font_family=self.font_family,
        )
