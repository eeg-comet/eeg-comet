
import os.path
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import (QDialog, QLabel, QPushButton, QLineEdit,
                             QLCDNumber, QMessageBox, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QSizePolicy)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from functions.clustering_utils.microstate_visualizer import show_microstate

class MicrostateVisualizationDialog(QDialog):
    def __init__(self, context, parent=None, main_window=None, tbx=None):
        super(MicrostateVisualizationDialog, self).__init__(parent)
        self.setupUI(context)
        self.init_attributes(main_window, tbx)
        self.connectUI()

    def setupUI(self, context):
        """Setup UI components"""
        self.ui = uic.loadUi(context.get_resource("MicrostateVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the identified microstates")

    def init_attributes(self, main_window, tbx):
        """Initialize dialog attributes"""
        self.main_window = main_window
        self.tbx = tbx

    def connectUI(self):
        """Connect UI signals to slots"""
        self.ui.num_contours_spinbox.valueChanged.connect(self.update_maps)
        self.ui.colormap_combobox.activated.connect(self.update_maps)
        self.ui.reverse_polarity_checkbox.clicked.connect(self.update_maps)
        self.ui.show_sensors_checkbox.clicked.connect(self.update_maps)
        self.ui.apply_all_button.clicked.connect(self.plot_maps)
        self.ui.export_microstates_image_button.clicked.connect(self.export_microstates_image)
        self.ui.manual_labeling_button.clicked.connect(self.manual_micro_label)
        self.ui.auto_labeling_button.clicked.connect(self.auto_micro_label)

    def set_layout(self, micro_labels=None):
        """Set layout for microstate visualization"""
        self.create_figure_and_canvas()
        self.create_label_widgets(micro_labels)

    def create_figure_and_canvas(self):
        """Create matplotlib figure and canvas"""
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.Microstate_Layout.addWidget(self.canvas)

    def create_label_widgets(self, micro_labels):
        """Create QLineEdit widgets for microstate labels"""
        for i in range(self.n_states):
            label_widget = QLineEdit(self)
            self.set_label_widget_attributes(label_widget, micro_labels, i)
            setattr(self, f"micro_labels{i}", label_widget)
            self.Microstates_Labels_Layout.addWidget(label_widget)

    def set_label_widget_attributes(self, widget, micro_labels, index):
        """Configure QLineEdit widget attributes"""
        widget.setAlignment(QtCore.Qt.AlignCenter)
        widget.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[a-z-A-Z]")))
        widget.setFont(QtGui.QFont("Calibri", 15, QtGui.QFont.Bold))
        widget.setMaxLength(1)
        if micro_labels:
            widget.setText(micro_labels[index])
            widget.setDisabled(True)

    def plot_maps(self):
        """Plot microstate maps on canvas"""
        self.axs = [self.figure.add_subplot(1, self.microstate_maps.shape[0], idx + 1)
                    for idx in range(self.n_states)]
        for idx, ax in enumerate(self.axs):
            ax.clear()
            polarity = -1 if self.ui.reverse_polarity_checkbox.isChecked() else 1
            sensors = self.ui.show_sensors_checkbox.isChecked()
            contours = int(self.ui.num_contours_spinbox.value())
            cmap = self.ui.colormap_combobox.currentText()

            show_microstate(self.microstate_maps[idx, :], self.eeg_info, ax,
                            polarity=polarity, sensors=sensors, contours=contours, cmap=cmap)
        self.canvas.draw()

    def update_maps(self):
        """Update specific microstate map based on user inputs"""
        idx = int(self.ui.microstates_combobox.currentText())
        ax = self.axs[idx]
        ax.clear()

        polarity = -1 if self.ui.reverse_polarity_checkbox.isChecked() else 1
        sensors = self.ui.show_sensors_checkbox.isChecked()
        contours = int(self.ui.num_contours_spinbox.value())
        cmap = self.ui.colormap_combobox.currentText()

        show_microstate(self.microstate_maps[idx, :], self.eeg_info, ax,
                        polarity=polarity, sensors=sensors, contours=contours, cmap=cmap)
        self.canvas.draw()


    def export_microstates_image(self):
        """Export microstate images to file"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                   "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)",
                                                   options=options)

        if file_name:
            extension = os.path.splitext(file_name)[-1].lower()

            # Ensure that the file has an extension
            if not extension:
                file_name += '.png'  # Default to PNG if no extension specified

            self.figure.savefig(file_name)

    def manual_micro_label(self):
        """Manually label the microstates"""
        self.micro_labels = []
        for i in range(self.n_states):
            microlabel_attr = getattr(self, "micro_labels{}".format(i))
            self.micro_labels.append(microlabel_attr.text())
        self.done_labeling = True
        for i in range(len(self.micro_labels)):
            if self.micro_labels[i] == '':
                self.done_labeling = False
        if self.done_labeling:
            self.tbx.micro_labels = self.micro_labels
            if self.main_window:
                self.tbx.done_labeling_microstates = True
                self.tbx.save_tbx()
                self.main_window.mainwindow_controller()
            self.close()
        else:
            QMessageBox.information(self, "Labeling Error",
                                    "Please add a label to each microstate",
                                    QMessageBox.Ok)

    def auto_micro_label(self):
        # TODO: UNDER DEVELOPMENT
        # print("under development ...")
        self.tbx.do_labeling()
        if self.tbx.done_labeling_microstates:
            # self.tbx.micro_labels = self.micro_labels
            str_micro_labels = ','.join(map(str, self.tbx.micro_labels))
            if self.main_window:
                self.tbx.done_labeling_microstates = True
                self.tbx.save_tbx()
                # currently commented (mainWindow's log needs to be updated)
                # self.main_window.ui.step0_log_textbrowser.insertPlainText(
                #     f"\nMicrostate labels: {str_micro_labels}\n")
                self.main_window.mainwindow_controller()
            self.close()
        else:
            QMessageBox.information(self, "Labeling Error",
                                    "Please add a label to each microstate",
                                    QMessageBox.Ok)

