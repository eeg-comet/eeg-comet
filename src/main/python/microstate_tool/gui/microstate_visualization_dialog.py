
import os.path
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QLineEdit, QLCDNumber, QMessageBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSizePolicy
from configparser import ConfigParser
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from functions.clustering_utils.microstate_visualizer import show_microstate
from functions.gui_utils.config_io import load_config, save_config

class MicrostateVisualizationDialog(QDialog):
    def __init__(self, context, parent=None, main_window=None, tbx=None):
        super(MicrostateVisualizationDialog, self).__init__(parent)

        self.main_window = main_window
        self.tbx = tbx

        self.ui = uic.loadUi(context.get_resource("MicrostateVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the identified microstates")

        self.ui.num_contours_spinbox.valueChanged.connect(self.update_maps)
        self.ui.colormap_combobox.activated.connect(self.update_maps)
        self.ui.reverse_polarity_checkbox.clicked.connect(self.update_maps)
        self.ui.show_sensors_checkbox.clicked.connect(self.update_maps)

        self.ui.apply_all_button.clicked.connect(self.plot_maps)

        self.ui.export_microstates_image_button.clicked.connect(self.export_microstates_image)
        self.ui.manual_labeling_button.clicked.connect(self.manual_micro_label)
        self.ui.auto_labeling_button.clicked.connect(self.auto_micro_label)

    def set_layout(self, micro_labels=None):
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Preferred)
        self.Microstate_Layout.addWidget(self.canvas)

        for i in range(self.n_states):
            exec(f'self.micro_labels{i} = QLineEdit(self)')
            microlabel_attr = getattr(self, "micro_labels{}".format(i))
            self.Microstates_Labels_Layout.addWidget(microlabel_attr)
            microlabel_attr.setAlignment(QtCore.Qt.AlignCenter)
            regex = QtCore.QRegExp("[a-z-A-Z]")
            validator = QtGui.QRegExpValidator(regex, microlabel_attr)
            microlabel_attr.setValidator(validator)
            font = QtGui.QFont("Calibri", 15, QtGui.QFont.Bold)
            microlabel_attr.setFont(font)
            microlabel_attr.setMaxLength(1)
            if micro_labels:
                exec(f'self.micro_labels{i}.setText(micro_labels[{i}])')
                exec(f'self.micro_labels{i}.setDisabled(True)')

    def plot_maps(self):
        self.axs = []
        for idx in range(self.n_states):
            ax = self.figure.add_subplot(1, self.microstate_maps.shape[0], idx + 1)
            self.axs.append(ax)
            ax.clear()
            if self.ui.reverse_polarity_checkbox.isChecked():
                polarity = -1
            else:
                polarity = 1
            if self.ui.show_sensors_checkbox.isChecked():
                sensors = True
            else:
                sensors = False
            show_microstate(self.microstate_maps[idx, :], self.eeg_info, ax,
                            polarity=polarity,
                            sensors=sensors,
                            contours=int(self.ui.num_contours_spinbox.value()),
                            cmap=self.ui.colormap_combobox.currentText())
            self.canvas.draw()

    def update_maps(self):
        idx = int(self.ui.microstates_combobox.currentText())
        ax = self.axs[idx]
        ax.clear()
        if self.ui.reverse_polarity_checkbox.isChecked():
            polarity = -1
        else:
            polarity = 1
        if self.ui.show_sensors_checkbox.isChecked():
            sensors = True
        else:
            sensors = False
        show_microstate(self.microstate_maps[idx, :], self.eeg_info, ax,
                        polarity=polarity,
                        sensors=sensors,
                        contours=int(self.ui.num_contours_spinbox.value()),
                        cmap=self.ui.colormap_combobox.currentText())
        self.canvas.draw()



    def export_microstates_image(self):
        # Save the current figure to the specified image path
        self.figure.savefig(self.microstates_image_path, bbox_inches='tight', dpi=200)
        # Show a message box to inform the user about the successful image save
        QMessageBox.information(self,
                                "Image Saved",
                                f"The image was saved to the study folder as '{self.microstates_image_path}' successfully.",
                                QMessageBox.Ok)
    def manual_micro_label(self):
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
            str_micro_labels = ','.join(map(str, self.micro_labels))
            if self.main_window:
                self.tbx.done_labeling_microstates = True
                self.tbx.save_tbx()
                self.main_window.ui.step0_log_textbrowser.insertPlainText(
                    f"\nMicrostate labels: {str_micro_labels}\n")
                self.main_window.mainwindow_controller()
            self.close()
        else:
            QMessageBox.information(self, "Labeling Error",
                                    "Please add a label to each microstate",
                                    QMessageBox.Ok)

    def auto_micro_label(self):
        # UNDER DEVELOPMENT
        print("under development ...")

