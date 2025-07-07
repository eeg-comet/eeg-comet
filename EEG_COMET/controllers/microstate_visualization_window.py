import os.path
import numpy as np
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QLineEdit, QMessageBox, QFileDialog, QSizePolicy, QActionGroup, QRadioButton, \
    QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from clustering_utils.microstate_io import MicrostateIO
from clustering_utils.microstate_visualizer import show_microstate


class MicrostateVisualizationWindow(QMainWindow):
    def __init__(self, context, parent=None, main_window=None, tbx=None):
        super(MicrostateVisualizationWindow, self).__init__(parent)
        self.main_window = main_window
        self.comet = tbx
        self.current_order_labels = self.comet.micro_labels
        self.current_order_axs_labels = self.comet.micro_labels

        # Fix: Check if best_maps exists before copying
        if self.comet.best_maps is not None:
            self.current_order_maps = self.comet.best_maps.copy()
        else:
            # Initialize with an empty array of appropriate shape
            n_maps = self.comet.number_of_maps
            n_channels = len(self.comet.eeg_info['ch_names'])
            self.current_order_maps = np.zeros((n_maps, n_channels))

        # Initialize polarity for each microstate (1 = normal, -1 = reversed)
        self.microstate_polarities = [1] * self.comet.number_of_maps

        self.setup_ui(context)
        self.setup_menu_actions()
        self.create_label_widgets(self.comet.micro_labels)
        self.connect_ui()
        self.create_figure_and_canvas()

    def setup_ui(self, context):
        """Setup UI components"""
        self.ui = uic.loadUi(context.get_resource("MicrostateVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the identified microstates")
        # Set window flags to include the maximize button
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

    def setup_menu_actions(self):
        """Setup menu actions and action groups"""
        # Create action group for colormap actions (mutually exclusive)
        self.colormap_action_group = QActionGroup(self)
        self.colormap_action_group.addAction(self.ui.cmap_rdbur)
        self.colormap_action_group.addAction(self.ui.cmap_coolwarm)
        self.colormap_action_group.addAction(self.ui.cmap_bwr)
        self.colormap_action_group.addAction(self.ui.cmap_seismic)

        # Set the default colormap (RdBu_r) as checked
        self.ui.cmap_rdbur.setChecked(True)

        # Make show_sensors and show_contours checkable
        self.ui.show_sensors.setCheckable(True)
        self.ui.show_contours.setCheckable(True)

        # Set default states
        self.ui.show_sensors.setChecked(False)
        self.ui.show_contours.setChecked(True)

    def connect_ui(self):
        """Connect UI signals to slots"""
        for i, label_widget in enumerate(self.micro_label_widgets):
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
        """Connect menu actions to their respective slots"""
        # Connect colormap actions
        self.ui.cmap_rdbur.triggered.connect(lambda: self.on_colormap_action_triggered('RdBu_r'))
        self.ui.cmap_coolwarm.triggered.connect(lambda: self.on_colormap_action_triggered('coolwarm'))
        self.ui.cmap_bwr.triggered.connect(lambda: self.on_colormap_action_triggered('bwr'))
        self.ui.cmap_seismic.triggered.connect(lambda: self.on_colormap_action_triggered('seismic'))

        # Connect view actions
        self.ui.show_sensors.triggered.connect(self.on_show_sensors_action_triggered)
        self.ui.show_contours.triggered.connect(self.on_show_contours_action_triggered)

    def get_current_settings(self):
        """Get current visualization settings from QActions"""
        # Get colormap from checked action
        if self.ui.cmap_rdbur.isChecked():
            cmap = 'RdBu_r'
        elif self.ui.cmap_coolwarm.isChecked():
            cmap = 'coolwarm'
        elif self.ui.cmap_bwr.isChecked():
            cmap = 'bwr'
        elif self.ui.cmap_seismic.isChecked():
            cmap = 'seismic'
        else:
            cmap = 'RdBu_r'  # default

        # Get other settings from actions
        sensors = self.ui.show_sensors.isChecked()
        contours = 6 if self.ui.show_contours.isChecked() else 0

        return {
            'cmap': cmap,
            'sensors': sensors,
            'contours': contours
        }

    def on_colormap_action_triggered(self, colormap_name):
        """Handle colormap action selection"""
        self.plot_maps()

    def on_show_sensors_action_triggered(self):
        """Handle show sensors action"""
        self.plot_maps()

    def on_show_contours_action_triggered(self):
        """Handle show contours action"""
        self.plot_maps()

    def create_figure_and_canvas(self):
        """Create matplotlib figure and canvas"""
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas.setMinimumHeight(100)
        self.Microstate_Layout.addWidget(self.canvas)

    def create_label_widgets(self, micro_labels):
        """Create QLineEdit widgets for microstate labels with polarity radio buttons"""
        self.micro_label_widgets = []  # Store QLineEdit widgets as an attribute
        self.polarity_radio_buttons = []  # Store radio buttons for polarity

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
            polarity_radio.setChecked(False)  # Default to normal polarity
            polarity_radio.toggled.connect(lambda checked, idx=i: self.on_polarity_radio_toggled(idx, checked))

            # Add widgets to container
            container_layout.addWidget(label_widget)
            container_layout.addWidget(polarity_radio)

            # Store references
            setattr(self, f"micro_label_{i}", label_widget)  # Set attribute with unique name
            self.micro_label_widgets.append(label_widget)  # Append to the list
            self.polarity_radio_buttons.append(polarity_radio)

            # Add container to layout
            self.Microstates_Labels_Layout.addWidget(container_widget)

    def on_polarity_radio_toggled(self, microstate_idx, checked):
        """Handle polarity radio button toggle"""
        # Update polarity: checked = reversed (-1), unchecked = normal (1)
        self.microstate_polarities[microstate_idx] = -1 if checked else 1

        # Update the visualization
        self.plot_maps()

    @staticmethod
    def set_label_widget_attributes(widget, micro_labels, index):
        """Configure QLineEdit widget attributes"""
        widget.setAlignment(QtCore.Qt.AlignCenter)
        widget.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[a-z-A-Z]")))
        widget.setFont(QtGui.QFont("Calibri", 15, QtGui.QFont.Bold))
        widget.setMaxLength(1)

        if micro_labels:
            sorted_indices = sorted(range(len(micro_labels)), key=lambda k: micro_labels[k])
            sorted_labels = [micro_labels[i] for i in sorted_indices]
            widget.setText(sorted_labels[index])
            widget.setDisabled(True)

    def plot_microstates_with_labels(self, microstate, micro_label, ax, polarity=1):
        """
        Plot the microstate with the corresponding label on a given axis.
        """
        settings = self.get_current_settings()

        # Use the provided polarity (individual per microstate) instead of global setting
        # Plot the microstate
        show_microstate(microstate, self.comet.eeg_info, ax,
                        polarity=polarity,
                        sensors=settings['sensors'],
                        contours=settings['contours'],
                        cmap=settings['cmap'])
        # Axis settings
        ax.axis('off')
        ax.text(0.5, -0.2, micro_label.upper(), transform=ax.transAxes,
                fontsize=16, ha='center', va='center')
        ax.text(0, 0, '', transform=ax.transAxes)

    def plot_maps(self):
        """Plot microstate maps on canvas"""
        micro_labels_texts = [getattr(self, f"micro_label_{i}").text() for i in range(self.comet.number_of_maps)]

        # If no labels are provided yet, fall back to generic placeholders (M1, M2, …)
        if not any(micro_labels_texts):
            micro_labels_texts = [f"M{i + 1}" for i in range(len(micro_labels_texts))]

        # ------------------------------------------------------------------
        #  KEEP ORDER CONSISTENT
        # ------------------------------------------------------------------
        # Do *not* reorder anything here; the widgets, the internal map order
        # (self.current_order_maps) and the polarity list must remain in the
        # exact same index order so that figures and labels stay aligned. Any
        # deliberate re-ordering is handled exclusively by `reorder_microstates`.

        # Clear existing axes
        for ax in self.figure.get_axes():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Create subplots – one per microstate
        self.axs = [self.figure.add_subplot(1, len(micro_labels_texts), idx + 1)
                    for idx in range(len(micro_labels_texts))]

        for idx in range(len(micro_labels_texts)):
            # Plot the microstate with its corresponding label and polarity
            self.plot_microstates_with_labels(
                self.current_order_maps[idx, :],
                micro_labels_texts[idx].upper(),
                self.axs[idx],
                polarity=self.microstate_polarities[idx]
            )

        # Adjust layout to prevent overlapping text
        self.figure.tight_layout()

        # Sync labels after plotting
        self.sync_labels_with_widgets()

        # Draw the canvas after plotting
        self.canvas.draw()

    def update_all_label_texts(self):
        """Update the text on the corresponding image in real-time"""
        if hasattr(self, 'axs'):
            for idx, ax in enumerate(self.axs):
                # Retrieve the label widget for this microstate
                label_widget = self.micro_label_widgets[idx]
                label_text = label_widget.text()

                # Update the text on the corresponding axis
                if ax.texts:  # Check if there are text objects
                    ax_text = ax.texts[0]  # Assuming there's only one text object on each axis
                    ax_text.set_text(label_text.upper())

            # Keep labels synchronized with widget state
            self.sync_labels_with_widgets()
            self.canvas.draw()

    def export_microstates_image(self):
        """Export microstate images to file"""
        # Get study name for default filename
        study_name = getattr(self.comet, 'study_name', 'microstates')
        default_filename = f"{study_name}_microstates.pdf"

        # Get default directory from comet object
        default_dir = getattr(self.comet, 'save_dir', '')

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
            options=options
        )

        if file_name:
            extension = os.path.splitext(file_name)[-1].lower()

            # Ensure that the file has an extension
            if not extension:
                file_name += '.pdf'  # Default to PDF for vector format

            # Save with appropriate settings for vector formats
            if extension in ['.pdf', '.svg']:
                # Vector formats - save with high DPI and vector-friendly settings
                self.figure.savefig(file_name, dpi=300, bbox_inches='tight',
                                    facecolor='white', edgecolor='none')
            else:
                # Raster formats
                self.figure.savefig(file_name, dpi=300, bbox_inches='tight')

    def auto_micro_label(self):
        """Automatically label microstates and update the visualization."""
        self.comet.run_microstate_labeling()
        if self.comet.done_microstate_labeling:
            for i, label_widget in enumerate(self.micro_label_widgets):
                label_widget.setText(self.comet.micro_labels[i])
                label_widget.setDisabled(True)
            self.sync_labels_with_widgets()  # Sync after setting widget texts
            self.reorder_microstates()
            if self.main_window:
                self.main_window.mainwindow_controller()

    def reorder_microstates(self):
        """Update the window to reflect changes"""
        # Store the current order of micro_label_widgets texts and axs current image labels
        current_order_labels = [label_widget.text() for label_widget in self.micro_label_widgets]

        if hasattr(self, 'axs'):
            current_order_axs_labels = [ax.texts[0].get_text() for ax in self.axs if ax.texts]
        else:
            current_order_axs_labels = current_order_labels

        # Sort microstate labels and their corresponding widgets
        sorted_indices = sorted(range(len(current_order_labels)), key=lambda k: current_order_labels[k])
        sorted_micro_labels = [current_order_labels[i] for i in sorted_indices]

        # Update the order of maps and polarities
        sorted_maps = self.current_order_maps[sorted_indices]
        sorted_polarities = [self.microstate_polarities[i] for i in sorted_indices]

        # Clear existing axes
        for ax in self.figure.get_axes():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Create subplots for each microstate
        self.axs = [self.figure.add_subplot(1, len(sorted_indices), idx + 1) for idx in range(len(sorted_indices))]

        # FIX: Use range(len(sorted_indices)) instead of sorted_indices
        for i in range(len(sorted_indices)):
            # Plot the microstate with the corresponding label and polarity
            self.plot_microstates_with_labels(
                sorted_maps[i],
                sorted_micro_labels[i].upper(),
                self.axs[i],
                polarity=sorted_polarities[i]
            )

        # Update label widget text and radio button states
        for i, label_widget in enumerate(self.micro_label_widgets):
            label_widget.setText(sorted_micro_labels[i])
            # Update radio button state based on sorted polarity
            self.polarity_radio_buttons[i].setChecked(sorted_polarities[i] == -1)

        # Adjust layout to prevent overlapping text
        self.figure.tight_layout()

        # Draw the canvas after plotting
        self.canvas.draw()

        # FIX: Update with the SORTED order, not the original order
        self.current_order_labels = sorted_micro_labels
        self.current_order_axs_labels = sorted_micro_labels  # Should match sorted labels
        self.current_order_maps = sorted_maps
        self.microstate_polarities = sorted_polarities  # Update polarities order

    def reset_labeling(self):
        """Store the default order of labels, axs labels, and maps"""
        self.current_order_labels.clear()  # Clear previous default order
        self.current_order_axs_labels.clear()  # Clear previous default order

        # Fix: Check if best_maps exists before copying
        if self.comet.best_maps is not None:
            self.current_order_maps = self.comet.best_maps.copy()  # Store default order of maps

        # Reset all polarities to normal (1)
        self.microstate_polarities = [1] * self.comet.number_of_maps

        # Reflect the empty default order in the Qt window
        for i, label_widget in enumerate(self.micro_label_widgets):
            label_widget.setText("")  # Set labels to empty
            label_widget.setEnabled(True)  # Enable editing
            # Reset radio button to unchecked (normal polarity)
            self.polarity_radio_buttons[i].setChecked(False)

        if hasattr(self, 'axs'):
            for ax in self.axs:
                ax.text(0.5, -0.2, "", transform=ax.transAxes, fontsize=16, ha='center',
                        va='center')  # Set axs labels to empty

        self.sync_labels_with_widgets()  # Sync after clearing widget texts
        self.reorder_microstates()
        # Reset next steps processing flags to False
        processing_flags = [
            'done_microstate_labeling', 'done_backfitting', 'done_extracting_features',
            'done_source_localization', 'done_identifying_microstate_sources'
        ]
        self.main_window.reset_processing_flags(processing_flags)
        self.main_window.mainwindow_controller()

    def sync_labels_with_widgets(self):
        """Ensure current_order_labels stays in sync with widget state"""
        self.current_order_labels = [label_widget.text() for label_widget in self.micro_label_widgets]

    def close_window(self):
        """Close the window"""
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
                self.current_order_maps, self.comet.eeg_info, self.comet.microstate_maps_path, self.current_order_labels
            )
            self.comet.done_microstate_labeling = True
            # Update MainWindow's log if necessary
            self.main_window.mainwindow_controller()
            self.close()
        else:
            # Show message
            QMessageBox.information(self, "Manual Labeling",
                                    "Please add a label to each microstate and click Apply Label Changes when done.",
                                    QMessageBox.Ok)
