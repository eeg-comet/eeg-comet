
import os.path
from functions.clustering_utils.elbow_function import get_elbow

from PyQt5 import uic
from PyQt5.QtWidgets import QDialog

class NumberMapsDialog(QDialog):
    def __init__(self, context, parent=None):
        super(NumberMapsDialog, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("NumberMapsWindow.ui"), self)
        self.ui.setWindowTitle("Exploring the number of microstate maps")

        self.ui.plot_k_button.clicked.connect(self.plot_k)

    def plot_k(self):
        self.k_min = int(self.ui.k_min_maps_input.text())
        self.k_max = int(self.ui.k_max_maps_input.text())

        ax_res = self.ui.MplWidget_residual.canvas.axes
        ax_res.clear()
        ax_gev = self.ui.MplWidget_gev.canvas.axes
        ax_gev.clear()
        ax_sil = self.ui.MplWidget_silhouette.canvas.axes
        ax_sil.clear()

        get_elbow(self.preprocessed_data_path, self.extension, self.datatype,
                  self.use_percentages, self.min_distance_size, self.number_of_repeats,
                  self.k_min, self.k_max, ax1=ax_res, ax2=ax_gev, ax3=ax_sil)

        self.ui.MplWidget_residual.canvas.draw()
        self.ui.MplWidget_gev.canvas.draw()
        self.ui.MplWidget_silhouette.canvas.draw()
