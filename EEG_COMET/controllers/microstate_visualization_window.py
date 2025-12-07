"""Microstate visualization window for interactively viewing maps and labels."""

import contextlib
import os.path
import logging

import mne
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QActionGroup,
    QFileDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from clustering_utils.microstate_io import MicrostateIO
from clustering_utils.microstate_visualizer import show_microstate
from gui_utils.terminal_logger import get_logger


class MicrostateVisualizationWindow(QMainWindow):
    """Window to explore microstate maps and their ordering.

    Provides interactive controls to relabel and reorder microstates, toggle
    plotting options (sensors, contours, colour-bar), and export figures.

    Attributes:
      main_window: Optional reference to the main window for syncing state.
      comet: COMET toolbox instance with EEG info, maps, and settings.
      current_order_labels (list[str]): Current microstate label ordering.
      current_order_axs_labels (list[str]): Axis label ordering (mirrors labels).
      current_order_maps (np.ndarray): Current map ordering (n_maps x n_channels).
      microstate_polarities (list[int]): Per-map polarity flags (1 or -1).
    """
    def __init__(self, context, parent=None, main_window=None, tbx=None):
        """Initialize the microstate visualization window and UI components.

        Args:
          context: Resource/context provider used to resolve UI resources.
          parent: Optional parent widget.
          main_window: Optional main window to notify about state changes.
          tbx: COMET toolbox instance providing EEG info and microstate data.
        """
        super().__init__(parent)
        
        # Suppress MNE warnings for electrode positions in this window
        mne.set_log_level('ERROR')
        
        # Add logging filter for this specific window
        class LocalMNEWarningFilter(logging.Filter):
            def filter(self, record):
                if hasattr(record, 'getMessage'):
                    message = record.getMessage()
                    if "Did not find any electrode locations" in message:
                        return False
                    if "digitization points do not correspond" in message:
                        return False
                return True
        
        self._mne_filter = LocalMNEWarningFilter()
        logging.getLogger().addFilter(self._mne_filter)
        
        self.main_window = main_window
        self.comet = tbx
        self.current_order_labels = self.comet.micro_labels
        self.current_order_axs_labels = self.comet.micro_labels

        # Enable maximize button
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        # Fix: Check if best_maps exists before copying
        if self.comet.best_maps is not None:
            self.current_order_maps = self.comet.best_maps.copy()
        else:
            # Initialize with an empty array of appropriate shape
            n_maps = self.comet.number_of_maps
            n_channels = len(self.comet.eeg_info["ch_names"])
            self.current_order_maps = np.zeros((n_maps, n_channels))

        # Initialize polarity for each microstate (1 = normal, -1 = reversed)
        self.microstate_polarities = [1] * self.comet.number_of_maps

        # Initialize label confidences (label -> confidence %)
        self.label_confidences = {}

        self.setup_ui(context)
        self.setup_menu_actions()
        self.create_label_widgets(self.comet.micro_labels)
        self.connect_ui()
        self.create_figure_and_canvas()

    def closeEvent(self, event):
        """Handle window close event and cleanup logging filter."""
        # Remove the logging filter when window is closed
        if hasattr(self, '_mne_filter'):
            logging.getLogger().removeFilter(self._mne_filter)
        super().closeEvent(event)

    def setup_ui(self, context):
        """Set up UI components.

        Args:
          context: Resource/context provider used to resolve UI resources.
        """
        self.ui = uic.loadUi(context.get_resource("MicrostateVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the identified microstates")
        # Set window flags to include the maximize button
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

    def setup_menu_actions(self):
        """Set up menu actions and action groups."""
        # Colormap actions are mutually exclusive
        self.colormap_action_group = QActionGroup(self)
        self.colormap_action_group.addAction(self.ui.cmap_rdbur)
        self.colormap_action_group.addAction(self.ui.cmap_coolwarm)
        self.colormap_action_group.addAction(self.ui.cmap_bwr)
        self.colormap_action_group.addAction(self.ui.cmap_seismic)

        # Set the default colormap (RdBu_r) as checked
        self.ui.cmap_rdbur.setChecked(True)

        # Add colour-bar visibility toggle
        self.ui.show_sensors.setCheckable(True)
        self.ui.show_contours.setCheckable(True)

        if hasattr(self.ui, "show_colorbar"):
            self.ui.show_colorbar.setCheckable(True)
            self.ui.show_colorbar.setChecked(True)

        # Set default states
        self.ui.show_sensors.setChecked(False)
        self.ui.show_contours.setChecked(True)

    def connect_ui(self):
        """Connect UI signals to slots."""
        for _i, label_widget in enumerate(self.micro_label_widgets):
            # Note: sync_labels_with_widgets() is already called inside update_all_label_texts()
            label_widget.textChanged.connect(self.update_all_label_texts)
        self.ui.export_microstates_image_button.triggered.connect(self.export_microstates_image)
        self.ui.export_microstates_image_button.setShortcut("Ctrl+S")
        self.ui.reorder_microstates_button.clicked.connect(self.reorder_microstates)
        self.ui.auto_labeling_button.clicked.connect(self.auto_micro_label)
        self.ui.reset_labels_button.clicked.connect(self.reset_labeling)
        self.ui.done_labeling_button.clicked.connect(self.close_window)

        # Connect menu actions
        self.connect_menu_actions()

    def connect_menu_actions(self):
        """Connect menu actions to their respective slots."""
        # Connect colormap actions
        self.ui.cmap_rdbur.triggered.connect(lambda: self.on_colormap_action_triggered("RdBu_r"))
        self.ui.cmap_coolwarm.triggered.connect(
            lambda: self.on_colormap_action_triggered("coolwarm")
        )
        self.ui.cmap_bwr.triggered.connect(lambda: self.on_colormap_action_triggered("bwr"))
        self.ui.cmap_seismic.triggered.connect(lambda: self.on_colormap_action_triggered("seismic"))

        # Connect view actions
        self.ui.show_sensors.triggered.connect(self.on_show_sensors_action_triggered)
        self.ui.show_contours.triggered.connect(self.on_show_contours_action_triggered)

        # Connect colour-bar visibility action
        if hasattr(self.ui, "show_colorbar"):
            self.ui.show_colorbar.triggered.connect(self.on_show_colorbar_action_triggered)

    def on_colormap_action_triggered(self, colormap_name):
        """Handle colormap action selection.

        Args:
          colormap_name (str): Name of the selected matplotlib colormap.
        """
        self.plot_maps()

    def on_show_sensors_action_triggered(self):
        """Handle show sensors action."""
        self.plot_maps()

    def on_show_contours_action_triggered(self):
        """Handle show contours action."""
        self.plot_maps()

    def on_show_colorbar_action_triggered(self):
        """Handle show color-bar toggle action."""
        # Re-plot to update colour-bar visibility
        self.plot_maps()

    # Settings helper
    def get_current_settings(self):
        """Collect current visualization settings selected in the UI.

        Returns:
          dict: Dictionary with keys 'cmap', 'sensors', 'contours', 'colorbar'.
        """
        # Choose current colormap
        if self.ui.cmap_rdbur.isChecked():
            cmap = "RdBu_r"
        elif self.ui.cmap_coolwarm.isChecked():
            cmap = "coolwarm"
        elif self.ui.cmap_bwr.isChecked():
            cmap = "bwr"
        elif self.ui.cmap_seismic.isChecked():
            cmap = "seismic"
        else:
            cmap = "RdBu_r"

        sensors = self.ui.show_sensors.isChecked()
        contours = 6 if self.ui.show_contours.isChecked() else 0
        colorbar = self.ui.show_colorbar.isChecked() if hasattr(self.ui, "show_colorbar") else True

        return {
            "cmap": cmap,
            "sensors": sensors,
            "contours": contours,
            "colorbar": colorbar,
        }

    def create_figure_and_canvas(self):
        """Create matplotlib figure and canvas."""
        # Use constrained_layout for automatic centring
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas.setMinimumHeight(100)
        self.Microstate_Layout.addWidget(self.canvas)

    def create_label_widgets(self, micro_labels):
        """Create label inputs and polarity toggles for each microstate.

        Args:
          micro_labels (list[str] | None): Initial labels to populate; if None or
            empty, fields start blank and editable.
        """
        self.micro_label_widgets = []  # Editable label widgets
        self.polarity_radio_buttons = []

        for i in range(self.comet.number_of_maps):
            # Create a container widget for each label and its radio button
            container_widget = QWidget()
            container_layout = QVBoxLayout(container_widget)
            container_layout.setContentsMargins(2, 2, 2, 2)
            container_layout.setSpacing(2)

            # Create label widget
            label_widget = QLineEdit(self)
            self.set_label_widget_attributes(label_widget, micro_labels, i)

            # Create radio button for polarity
            polarity_radio = QRadioButton("Reverse Polarity", container_widget)
            polarity_radio.setFont(QtGui.QFont("Calibri", 14))
            polarity_radio.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            polarity_radio.setChecked(False)
            polarity_radio.toggled.connect(
                lambda checked, idx=i: self.on_polarity_radio_toggled(idx, checked)
            )

            # Add widgets to container
            container_layout.addWidget(label_widget)
            container_layout.addWidget(polarity_radio)

            # Store references
            setattr(self, f"micro_label_{i}", label_widget)
            self.micro_label_widgets.append(label_widget)
            self.polarity_radio_buttons.append(polarity_radio)

            # Add container to layout
            self.Microstates_Labels_Layout.addWidget(container_widget)

    def on_polarity_radio_toggled(self, microstate_idx, checked):
        """Handle polarity radio button toggle.

        Args:
          microstate_idx (int): Index of the toggled microstate.
          checked (bool): True when reversed polarity is selected.
        """
        # Update polarity list
        self.microstate_polarities[microstate_idx] = -1 if checked else 1

        # Update the visualization
        self.plot_maps()

    @staticmethod
    def set_label_widget_attributes(widget, micro_labels, index):
        """Configure QLineEdit widget attributes.

        Args:
          widget (QLineEdit): Label input widget to configure.
          micro_labels (list[str] | None): Existing labels for initialization.
          index (int): Index used to select initial label text.
        """
        widget.setAlignment(QtCore.Qt.AlignCenter)
        widget.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[a-z-A-Z]")))
        widget.setFont(QtGui.QFont("Calibri", 15, QtGui.QFont.Bold))
        widget.setMaxLength(1)

        if micro_labels:
            sorted_indices = sorted(range(len(micro_labels)), key=lambda k: micro_labels[k])
            sorted_labels = [micro_labels[i] for i in sorted_indices]
            widget.setText(sorted_labels[index])
            widget.setDisabled(True)

    def _log_electrode_warning(self, message):
        """Log electrode positioning warnings using the proper EEG-COMET logger format."""
        if hasattr(self.comet, 'LogWindow') and self.comet.LogWindow:
            # Use the proper logger format for warnings
            logger = get_logger(self.comet.LogWindow)
            logger.warning("VISUALIZATION", "Electrode positions not found. Applying standard montage for microstate visualization.")

    def plot_microstates_with_labels(self, microstate, micro_label, ax, polarity=1, confidence=None):
        """Plot a microstate topomap and its label on the provided axis.

        Returns the image handle so a shared colour-bar can be created later.

        Args:
          microstate (np.ndarray): Topographic map values for one microstate.
          micro_label (str): Label character to display below the map.
          ax (matplotlib.axes.Axes): Axis to draw into.
          polarity (int, optional): Polarity multiplier (1 or -1). Defaults to 1.
          confidence (float, optional): Classification confidence (0-100%). Defaults to None.

        Returns:
          matplotlib.image.AxesImage: Handle used for colour-bar creation.
        """
        settings = self.get_current_settings()

        # Plot and get image handle for colour-bar
        im = show_microstate(
            microstate,
            self.comet.eeg_info,
            ax,
            polarity=polarity,
            sensors=settings["sensors"],
            contours=settings["contours"],
            cmap=settings["cmap"],
            log_callback=self._log_electrode_warning,
        )

        # Label placement
        ax.axis("off")

        # Build label text with optional confidence
        if confidence is not None and confidence > 0:
            label_text = f"{micro_label.upper()}\n({confidence:.1f}%)"
        else:
            label_text = micro_label.upper()

        ax.text(
            0.5,
            -0.2,
            label_text,
            transform=ax.transAxes,
            fontsize=20,
            ha="center",
            va="center",
        )
        ax.text(0, 0, "", transform=ax.transAxes)

        return im

    def plot_maps(self):
        """Plot microstate maps on canvas."""
        # Ensure MNE logging is suppressed for this entire plotting session
        mne.set_log_level('ERROR')
        
        micro_labels_texts = [
            getattr(self, f"micro_label_{i}").text() for i in range(self.comet.number_of_maps)
        ]

        # Fallback placeholders if labels missing
        if not any(micro_labels_texts):
            micro_labels_texts = [f"M{i + 1}" for i in range(len(micro_labels_texts))]

        # Keep widget and map ordering consistent; reordering handled elsewhere

        # Clear existing axes
        for ax in self.figure.get_axes():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

        # Create subplots – one per microstate
        self.axs = [
            self.figure.add_subplot(1, len(micro_labels_texts), idx + 1)
            for idx in range(len(micro_labels_texts))
        ]

        im = None  # Image handle for colour-bar
        for idx in range(len(micro_labels_texts)):
            # Get confidence for this label if available
            label = micro_labels_texts[idx].upper()
            confidence = self.label_confidences.get(label)

            # Plot each microstate
            im = self.plot_microstates_with_labels(
                self.current_order_maps[idx, :],
                micro_labels_texts[idx].upper(),
                self.axs[idx],
                polarity=self.microstate_polarities[idx],
                confidence=confidence,
            )

        # With constrained_layout, no manual layout adjustment needed

        # Colour-bar visibility
        if self.ui.show_colorbar.isChecked() and im is not None:
            # Remove previous colour-bar
            if hasattr(self, "cbar") and self.cbar:
                with contextlib.suppress(Exception):
                    self.cbar.remove()

            # Create colour-bar at top
            self.cbar = self.figure.colorbar(
                im,
                ax=self.axs,
                orientation="horizontal",
                location="top",
                pad=0.04,
                fraction=0.05,
            )

            # Add title/label and increase font sizes
            self.cbar.set_label("Amplitude (µV)", fontsize=20, labelpad=6)

            # Enlarge tick font size
            self.cbar.ax.tick_params(labelsize=18)

        else:
            # Remove colour-bar if toggled off
            if hasattr(self, "cbar") and self.cbar:
                with contextlib.suppress(Exception):
                    self.cbar.remove()
                self.cbar = None

        # Sync labels after plotting
        self.sync_labels_with_widgets()

        # Draw the canvas after plotting
        self.canvas.draw()

    def update_all_label_texts(self):
        """Update the text on the corresponding image in real-time."""
        if hasattr(self, "axs"):
            for idx, ax in enumerate(self.axs):
                # Retrieve the label widget for this microstate
                label_widget = self.micro_label_widgets[idx]
                label_text = label_widget.text()

                # Update the text on the corresponding axis
                if ax.texts:  # Check if there are text objects
                    ax_text = ax.texts[0]
                    ax_text.set_text(label_text.upper())

            # Sync labels
            self.sync_labels_with_widgets()
            self.canvas.draw()

    def export_microstates_image(self):
        """Export the current microstate figure to an image file.

        Prompts the user for a destination path and format; saves with suitable
        settings for vector (PDF/SVG) and raster (PNG/JPG) formats.
        """
        # Default filename
        study_name = getattr(self.comet, "study_name", "microstates")
        default_filename = f"{study_name}_microstates.pdf"

        # Get default directory from comet object
        default_dir = getattr(self.comet, "save_dir", "")

        # Combine directory and filename for full default path
        if default_dir:
            default_path = os.path.join(default_dir, default_filename)
        else:
            default_path = default_filename

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Choose a location and filename to save the image",
            default_path,  # Use the full default path (directory + filename)
            "PDF Files (*.pdf);;PNG Files (*.png);;JPG Files (*.jpg);;SVG Files (*.svg);;All Files (*)",
            options=options,
        )

        if file_name:
            extension = os.path.splitext(file_name)[-1].lower()

            # Ensure that the file has an extension
            if not extension:
                file_name += ".pdf"

            # Save with appropriate settings for vector formats
            if extension in [".pdf", ".svg"]:
                # Vector formats
                self.figure.savefig(
                    file_name, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
                )
            else:
                # Raster formats
                self.figure.savefig(file_name, dpi=300, bbox_inches="tight")

    def auto_micro_label(self):
        """Automatically label microstates and update the visualization."""
        self.comet.run_microstate_labeling()
        if self.comet.done_microstate_labeling:
            # Restore maps to default order so labels align correctly
            if self.comet.best_maps is not None:
                self.current_order_maps = self.comet.best_maps.copy()

            # Reset polarities when auto-labeling
            self.microstate_polarities = [1] * self.comet.number_of_maps

            # Store label confidences from the labeling results
            if self.comet.label_confidences is not None:
                self.label_confidences = self.comet.label_confidences.copy()

            for i, label_widget in enumerate(self.micro_label_widgets):
                label_widget.setText(self.comet.micro_labels[i])
                label_widget.setDisabled(True)
            self.sync_labels_with_widgets()  # Sync after setting widget texts
            self.reorder_microstates()
            if self.main_window and hasattr(self.main_window, "mainwindow_controller"):
                self.main_window.mainwindow_controller()

    def reorder_microstates(self):
        """Reorder maps, labels, and polarities based on label text order."""
        # Store current ordering
        current_order_labels = [label_widget.text() for label_widget in self.micro_label_widgets]

        if hasattr(self, "axs"):
            [ax.texts[0].get_text() for ax in self.axs if ax.texts]
        else:
            pass

        # Sort microstate labels and their corresponding widgets
        sorted_indices = sorted(
            range(len(current_order_labels)), key=lambda k: current_order_labels[k]
        )
        sorted_micro_labels = [current_order_labels[i] for i in sorted_indices]

        # Update the order of maps and polarities
        sorted_maps = self.current_order_maps[sorted_indices]
        sorted_polarities = [self.microstate_polarities[i] for i in sorted_indices]

        # Clear existing axes
        for ax in self.figure.get_axes():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

        # Create subplots for each microstate
        self.axs = [
            self.figure.add_subplot(1, len(sorted_indices), idx + 1)
            for idx in range(len(sorted_indices))
        ]

        im = None
        for i in range(len(sorted_indices)):
            # Get confidence for this label if available
            label = sorted_micro_labels[i].upper()
            confidence = self.label_confidences.get(label)

            # Plot each microstate
            im = self.plot_microstates_with_labels(
                sorted_maps[i],
                sorted_micro_labels[i].upper(),
                self.axs[i],
                polarity=sorted_polarities[i],
                confidence=confidence,
            )

        # Update label widget text and radio button states
        for i, label_widget in enumerate(self.micro_label_widgets):
            label_widget.setText(sorted_micro_labels[i])
            # Update radio button state based on sorted polarity
            self.polarity_radio_buttons[i].setChecked(sorted_polarities[i] == -1)

        # No manual layout with constrained_layout

        # Colour-bar visibility
        if self.ui.show_colorbar.isChecked() and im is not None:
            if hasattr(self, "cbar") and self.cbar:
                with contextlib.suppress(Exception):
                    self.cbar.remove()

            self.cbar = self.figure.colorbar(
                im,
                ax=self.axs,
                orientation="horizontal",
                location="top",
                pad=0.04,
                fraction=0.05,
            )

            try:
                self.cbar.set_label("µV", fontsize=18, labelpad=6)
            except Exception:
                self.cbar.ax.set_title("µV", fontsize=14, pad=6)

            self.cbar.ax.tick_params(labelsize=16)
        else:
            if hasattr(self, "cbar") and self.cbar:
                with contextlib.suppress(Exception):
                    self.cbar.remove()
                self.cbar = None

        # Draw the canvas after plotting
        self.canvas.draw()

        # Update UI widgets with sorted labels
        self.current_order_labels = sorted_micro_labels
        self.current_order_axs_labels = sorted_micro_labels
        self.current_order_maps = sorted_maps
        self.microstate_polarities = sorted_polarities

    def reset_labeling(self):
        """Reset labels, axes labels, maps, and polarities to defaults."""
        self.current_order_labels.clear()
        self.current_order_axs_labels.clear()

        # Fix: Check if best_maps exists before copying
        if self.comet.best_maps is not None:
            self.current_order_maps = self.comet.best_maps.copy()

        # Reset all polarities to normal (1)
        self.microstate_polarities = [1] * self.comet.number_of_maps

        # Clear label confidences
        self.label_confidences = {}

        # Reflect the empty default order in the Qt window
        for _i, label_widget in enumerate(self.micro_label_widgets):
            label_widget.setText("")
            label_widget.setEnabled(True)

        if hasattr(self, "axs"):
            for ax in self.axs:
                ax.text(
                    0.5, -0.2, "", transform=ax.transAxes, fontsize=20, ha="center", va="center"
                )

        self.sync_labels_with_widgets()
        self.reorder_microstates()
        # Reset processing flags
        processing_flags = [
            "done_microstate_labeling",
            "done_backfitting",
            "done_extracting_features",
            "done_source_localization",
            "done_identifying_microstate_sources",
        ]
        # Reset relevant processing flags on the main window
        if self.main_window and hasattr(self.main_window, "processing_flags"):
            for flag in processing_flags:
                setattr(self.main_window.processing_flags, flag, False)

            # Sync flags to COMET so the rest of the app reflects the change
            if hasattr(self.main_window, "_sync_flags_to_comet"):
                self.main_window._sync_flags_to_comet()
        if self.main_window and hasattr(self.main_window, "mainwindow_controller"):
            self.main_window.mainwindow_controller()

    def sync_labels_with_widgets(self):
        """Ensure current_order_labels stays in sync with widget state."""
        self.current_order_labels = [
            label_widget.text() for label_widget in self.micro_label_widgets
        ]

    def close_window(self):
        """Validate labels, persist maps/labels, export, and close the window."""
        # Check if all label widgets have values
        all_labels_filled = all(label_widget.text() for label_widget in self.micro_label_widgets)
        if all_labels_filled:
            # Ensure labels are synchronized with current widget state
            widget_labels = [label_widget.text() for label_widget in self.micro_label_widgets]

            # Update the current order labels to match widget state
            self.current_order_labels = widget_labels

            # Save to comet object
            self.comet.micro_labels = self.current_order_labels.copy()
            self.comet.best_maps = self.current_order_maps.copy()

            # Export with synchronized labels and maps
            MicrostateIO().export_microstates(
                self.current_order_maps,
                self.comet.eeg_info,
                self.comet.microstate_maps_path,
                self.current_order_labels,
            )
            self.comet.done_microstate_labeling = True
            # Update MainWindow's log if necessary
            if self.main_window and hasattr(self.main_window, "mainwindow_controller"):
                self.main_window.mainwindow_controller()
            self.close()
        else:
            # Show message
            QMessageBox.information(
                self,
                "Manual Labeling",
                "Please add a label to each microstate and click Apply Label Changes when done.",
                QMessageBox.Ok,
            )
