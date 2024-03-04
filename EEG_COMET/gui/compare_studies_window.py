
import os.path
import pickle
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from functions.gui_utils.set_widgets_status import set_widgets_status
from functions.clustering_utils.microstate_visualizer import show_microstate


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
        self.ui.setWindowTitle("Visualization of the identified microstates")
        # Set window flags to include the maximize button
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

    def update_ui(self):
        """Connect UI signals to slots"""
        if self.study1_loaded:
            self.ui.study1_name_label.setText(self.comet_tbx_study1.study_name)
        if self.study2_loaded:
            self.ui.study2_name_label.setText(self.comet_tbx_study2.study_name)
        if self.study1_loaded and self.study2_loaded:
            set_widgets_status(self.ui.plot_features_button, mode='enable')
            mutual_features = set(self.comet_tbx_study1.feature_list) & set(self.comet_tbx_study2.feature_list)
            self.ui.feature_combo.addItems([i for i in mutual_features])
            selected_feature = self.ui.feature_combo.currentText()
            self.ui.plot_label.setText(f"{self.comet_tbx_study1.feature_list_dictionary[selected_feature]}")
        else:
            set_widgets_status(self.ui.plot_features_button, mode='disable')
            self.ui.feature_combo.clear()

    def connect_ui(self):
        """Connect UI signals to slots"""
        self.ui.load_study1_button.clicked.connect(self.load_study1)
        self.ui.load_study2_button.clicked.connect(self.load_study2)
        self.ui.feature_combo.currentTextChanged.connect(self.update_ui)

    def create_figure_and_canvas(self):
        """Create matplotlib figure and canvas"""
        self.figure_microstates_study1 = Figure(tight_layout=True)
        self.canvas_microstates_study1 = FigureCanvasQTAgg(self.figure_microstates_study1)
        self.canvas_microstates_study1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Figure_Microstates_Study1_Layout.addWidget(self.canvas_microstates_study1)
        self.figure_microstates_study2 = Figure(tight_layout=True)
        self.canvas_microstates_study2 = FigureCanvasQTAgg(self.figure_microstates_study2)
        self.canvas_microstates_study2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Figure_Microstates_Study2_Layout.addWidget(self.canvas_microstates_study2)
        self.figure_features = Figure(tight_layout=True)
        self.canvas_features = FigureCanvasQTAgg(self.figure_features)
        self.canvas_features.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Figure_Features_Layout.addWidget(self.canvas_features)

    @staticmethod
    def update_study_info(tbx, listwidget):
        for i in range(len(tbx.list_eegs)):
            listwidget.addItem(str(tbx.list_eegs[i]))

    def load_study1(self):
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

    def plot_microstates_with_labels(self, microstate, micro_label, eeg_info, ax):
        """
        Plot the microstate with the corresponding label on a given axis.
        """

        # Plot the microstate
        show_microstate(microstate, eeg_info, ax)
        # Axis settings
        ax.axis('off')
        ax.text(0.5, -0.2, micro_label.upper(), transform=ax.transAxes,
                fontsize=16, ha='center', va='center')
        ax.text(0, 0, '', transform=ax.transAxes)

    def plot_maps(self, tbx, figure, canvas):
        """Plot microstate maps on canvas"""
        # Clear existing axes
        for ax in figure.get_axes():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Create subplots for each microstate
        axs = [figure.add_subplot(1, len(tbx.micro_labels), idx + 1) for idx in
                    range(len(tbx.micro_labels))
               ]

        for idx, ax_idx in enumerate(range(len(tbx.micro_labels))):
            # Plot the microstate with the corresponding label
            self.plot_microstates_with_labels(
                tbx.best_maps[ax_idx, :], tbx.micro_labels[idx], tbx.eeg_info, axs[idx]
            )

        # Adjust layout to prevent overlapping text
        figure.tight_layout()

        # Draw the canvas after plotting
        canvas.draw()

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
