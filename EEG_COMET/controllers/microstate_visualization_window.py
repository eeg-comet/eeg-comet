import os.path
import numpy as np
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QLineEdit, QMessageBox, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from gui_utils.set_widgets_status import set_widgets_status
from clustering_utils.microstate_clusterer import MicrostateClusterer
from clustering_utils.microstate_visualizer import show_microstate


class MicrostateVisualizationWindow(QDialog):
    def __init__(self, context, parent=None, main_window=None, tbx=None):
        super(MicrostateVisualizationWindow, self).__init__(parent)
        self.main_window = main_window
        self.tbx = tbx
        self.current_order_labels = self.tbx.micro_labels
        self.current_order_axs_labels = self.tbx.micro_labels

        # Fix: Check if best_maps exists before copying
        if self.tbx.best_maps is not None:
            self.current_order_maps = self.tbx.best_maps.copy()
        else:
            # Initialize with an empty array of appropriate shape
            n_maps = self.tbx.number_of_maps
            n_channels = len(self.tbx.eeg_info['ch_names'])
            self.current_order_maps = np.zeros((n_maps, n_channels))
            print("Warning: best_maps is None, initialized empty maps array")

        self.setup_ui(context)
        self.create_label_widgets(self.tbx.micro_labels)
        self.connect_ui()
        self.create_figure_and_canvas()

    def setup_ui(self, context):
        """Setup UI components"""
        self.ui = uic.loadUi(context.get_resource("MicrostateVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the identified microstates")
        self.microstates_combobox.addItems(
            [str(i) for i in range(self.current_order_maps.shape[0])])
        # Set window flags to include the maximize button
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

    def connect_ui(self):
        """Connect UI signals to slots"""
        self.ui.settings_checkbox.stateChanged.connect(self.toggle_settings_widgets)
        self.ui.microstates_combobox.currentIndexChanged.connect(self.update_label_colors)
        for i, label_widget in enumerate(self.micro_label_widgets):
            label_widget.textChanged.connect(lambda: self.update_all_label_texts())
        self.ui.num_contours_spinbox.valueChanged.connect(self.update_maps)
        self.ui.colormap_combobox.activated.connect(self.update_maps)
        self.ui.reverse_polarity_checkbox.clicked.connect(self.update_maps)
        self.ui.show_sensors_checkbox.clicked.connect(self.update_maps)
        self.ui.apply_all_button.clicked.connect(self.plot_maps)
        self.ui.export_microstates_image_button.clicked.connect(self.export_microstates_image)
        self.ui.reorder_microstates_button.clicked.connect(self.reorder_microstates)
        self.ui.auto_labeling_button.clicked.connect(self.auto_micro_label)
        self.ui.reset_labels_button.clicked.connect(self.reset_labeling)
        self.ui.done_labeling_button.clicked.connect(self.close_window)

    def create_figure_and_canvas(self):
        """Create matplotlib figure and canvas"""
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.canvas.setMinimumHeight(100)
        self.Microstate_Layout.addWidget(self.canvas)

    def create_label_widgets(self, micro_labels):
        """Create QLineEdit widgets for microstate labels"""
        self.micro_label_widgets = []  # Store QLineEdit widgets as an attribute
        for i in range(self.tbx.number_of_maps):
            label_widget = QLineEdit(self)
            self.set_label_widget_attributes(label_widget, micro_labels, i)
            setattr(self, f"micro_label_{i}", label_widget)  # Set attribute with unique name
            self.micro_label_widgets.append(label_widget)  # Append to the list
            self.Microstates_Labels_Layout.addWidget(label_widget)

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

    def toggle_settings_widgets(self, state):
        """Toggle the visibility of settings widgets based on checkbox state"""
        setting_widgets = [
            self.ui.microstate_index_label, self.ui.microstates_combobox,
            self.ui.num_contours_label, self.ui.num_contours_spinbox,
            self.ui.colormap_label, self.ui.colormap_combobox,
            self.ui.other_options_label, self.ui.reverse_polarity_checkbox,
            self.ui.show_sensors_checkbox, self.ui.apply_all_button,
            self.ui.reorder_microstates_button
        ]
        mode = 'show' if state == QtCore.Qt.Checked else 'hide'
        set_widgets_status(setting_widgets, mode)

    def plot_microstates_with_labels(self, microstate, micro_label, ax):
        """
        Plot the microstate with the corresponding label on a given axis.
        """
        polarity = -1 if self.ui.reverse_polarity_checkbox.isChecked() else 1
        sensors = self.ui.show_sensors_checkbox.isChecked()
        contours = int(self.ui.num_contours_spinbox.value())
        cmap = self.ui.colormap_combobox.currentText()

        # Plot the microstate
        show_microstate(microstate, self.tbx.eeg_info, ax,
                        polarity=polarity, sensors=sensors, contours=contours, cmap=cmap)
        # Axis settings
        ax.axis('off')
        ax.text(0.5, -0.2, micro_label.upper(), transform=ax.transAxes,
                fontsize=16, ha='center', va='center')
        ax.text(0, 0, '', transform=ax.transAxes)

    def plot_maps(self):
        """Plot microstate maps on canvas"""
        micro_labels_texts = [getattr(self, f"micro_label_{i}").text() for i in range(self.tbx.number_of_maps)]

        # Check if any labels are filled
        if any(micro_labels_texts):
            # Sort the microstate labels and indices
            sorted_indices = sorted(range(len(micro_labels_texts)), key=lambda k: micro_labels_texts[k])
            sorted_labels = [micro_labels_texts[i].upper() for i in sorted_indices]
        else:
            # If no labels provided, assign default labels "M1", "M2", ...
            sorted_labels = [f"M{i + 1}" for i in range(len(micro_labels_texts))]

        # Clear existing axes
        for ax in self.figure.get_axes():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Create subplots for each microstate
        self.axs = [self.figure.add_subplot(1, len(micro_labels_texts), idx + 1) for idx in
                    range(len(micro_labels_texts))]

        for idx, ax_idx in enumerate(range(len(micro_labels_texts))):
            # Plot the microstate with the corresponding label
            self.plot_microstates_with_labels(
                self.current_order_maps[ax_idx, :], sorted_labels[idx], self.axs[idx]
            )

        # Adjust layout to prevent overlapping text
        self.figure.tight_layout()

        # Draw the canvas after plotting
        self.canvas.draw()

    def update_all_label_texts(self):
        """Update the text on the corresponding image in real-time"""
        for idx, ax in enumerate(self.axs):
            # Retrieve the label widget for this microstate
            label_widget = self.micro_label_widgets[idx]
            label_text = label_widget.text()

            # Update the text on the corresponding axis
            ax_text = ax.texts[0]  # Assuming there's only one text object on each axis
            ax_text.set_text(label_text.upper())

        self.canvas.draw()

    def update_maps(self):
        """Update specific microstate map based on user inputs"""
        # Convert the selected label to an integer index
        idx = int(self.ui.microstates_combobox.currentText())

        # Store the current order of micro_label_widgets texts and axs current image labels
        self.current_order_axs_labels = [ax.texts[0].get_text() for ax in self.axs]

        # Plot the microstate with the corresponding label
        self.plot_microstates_with_labels(
            self.current_order_maps[idx, :], self.current_order_axs_labels[idx].upper(), self.axs[idx]
        )

        # Redraw the canvas
        self.canvas.draw()

    def update_label_colors(self, index):
        """Update the colors of microstate labels based on the selected index"""
        for i, label_widget in enumerate(self.micro_label_widgets):
            if i == index:
                # Set background color for the selected index
                label_widget.setStyleSheet("background-color: yellow")  # Change to your desired color
            else:
                # Reset background color for other indices
                label_widget.setStyleSheet("")  # Reset stylesheet if not the current value

    def export_microstates_image(self):
        """Export microstate images to file"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Choose a location and filename to save the image", "",
                                                   "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)",
                                                   options=options)

        if file_name:
            extension = os.path.splitext(file_name)[-1].lower()

            # Ensure that the file has an extension
            if not extension:
                file_name += '.png'  # Default to PNG if no extension specified

            self.figure.savefig(file_name)

    def auto_micro_label(self):
        """Automatically label microstates and update the visualization."""
        self.tbx.do_labeling()
        if self.tbx.done_labeling_microstates:
            for i, label_widget in enumerate(self.micro_label_widgets):
                label_widget.setText(self.tbx.micro_labels[i])
                label_widget.setDisabled(True)
            self.reorder_microstates()
            if self.main_window:
                self.tbx.save_tbx()
                self.main_window.mainwindow_controller()

    def reorder_microstates(self):
        """Update the window to reflect changes"""
        # Store the current order of micro_label_widgets texts and axs current image labels
        current_order_labels = [label_widget.text() for label_widget in self.micro_label_widgets]
        current_order_axs_labels = [ax.texts[0].get_text() for ax in self.axs]

        # Sort microstate labels and their corresponding widgets
        sorted_indices = sorted(range(len(current_order_labels)), key=lambda k: current_order_labels[k])
        sorted_micro_labels = [current_order_labels[i] for i in sorted_indices]

        # Update the order of maps
        sorted_maps = self.current_order_maps[sorted_indices]

        # Clear existing axes (Note: Not necessary when using add_subplot)
        for ax in self.figure.get_axes():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Create subplots for each microstate
        self.axs = [self.figure.add_subplot(1, len(sorted_indices), idx + 1) for idx in range(len(sorted_indices))]

        for i in sorted_indices:
            # Plot the microstate with the corresponding label
            self.plot_microstates_with_labels(
                sorted_maps[i], sorted_micro_labels[i].upper(), self.axs[i]
            )

        # Update label widget text
        for i, label_widget in enumerate(self.micro_label_widgets):
            label_widget.setText(sorted_micro_labels[i])

        # Adjust layout to prevent overlapping text
        self.figure.tight_layout()

        # Draw the canvas after plotting
        self.canvas.draw()

        # Update the current order of labels and axs labels
        self.current_order_labels = current_order_labels
        self.current_order_axs_labels = current_order_axs_labels
        self.current_order_maps = sorted_maps

    def reset_labeling(self):
        """Store the default order of labels, axs labels, and maps"""
        self.current_order_labels.clear()  # Clear previous default order
        self.current_order_axs_labels.clear()  # Clear previous default order

        # Fix: Check if best_maps exists before copying
        if self.tbx.best_maps is not None:
            self.current_order_maps = self.tbx.best_maps.copy()  # Store default order of maps
        else:
            # Keep the current maps as they are, or reinitialize if needed
            print("Warning: best_maps is None, keeping current maps")

        # Reflect the empty default order in the Qt window
        for label_widget in self.micro_label_widgets:
            label_widget.setText("")  # Set labels to empty
            label_widget.setEnabled(True)  # Enable editing
        for ax in self.axs:
            ax.text(0.5, -0.2, "", transform=ax.transAxes, fontsize=16, ha='center',
                    va='center')  # Set axs labels to empty

        self.reorder_microstates()
        # Reset next steps processing flags to False
        processing_flags = [
            'done_labeling_microstates', 'done_backfitting', 'done_extracting_features',
            'done_source_localization', 'done_source_microstate_correlation'
        ]
        self.main_window.reset_processing_flags(processing_flags)
        self.main_window.mainwindow_controller()

    def close_window(self):
        """Close the window"""
        # Check if all label widgets have values
        all_labels_filled = all(label_widget.text() for label_widget in self.micro_label_widgets)
        if all_labels_filled:
            self.tbx.update_microstates_order(self.current_order_labels, self.current_order_maps)
            MicrostateClusterer().microstates2csv(
                self.current_order_maps, self.tbx.eeg_info, self.tbx.microstate_maps_path, self.current_order_labels)
            self.tbx.done_labeling_microstates = True
            # Update MainWindow's log if necessary
            self.main_window.mainwindow_controller()
            self.close()
        else:
            # Show message
            QMessageBox.information(self, "Manual Labeling",
                                    "Please add a label to each microstate and click Apply Label Changes when done.",
                                    QMessageBox.Ok)
