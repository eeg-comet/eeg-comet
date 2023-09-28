
import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog
from functions.clustering_utils.microstate_clusterer import ClusterOptimizer
from functions.data_utils.extract_peaks_maps import generate_maps_and_peaks


class ElbowVisualizationDialog(QDialog):
    def __init__(self, context, parent=None):
        super(ElbowVisualizationDialog, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("ElbowVisualizationWindow.ui"), self)
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

        maps2use, peaks2use = generate_maps_and_peaks(self.preprocessed_data_path, self.extension, self.datatype,
                                                      self.use_percentages, self.min_distance_size)
        elbow_optimizer = ClusterOptimizer(maps2use, self.min_distance_size, self.number_of_repeats,
                                         self.k_min, self.k_max, self.preprocessed_data_path,
                                         self.extension, self.datatype, self.clustering_tolerance, self.max_iterations)
        elbow_optimizer.find_elbow_with_plot(ax1=ax_res, ax2=ax_gev, ax3=ax_sil)

        self.ui.MplWidget_residual.canvas.draw()
        self.ui.MplWidget_gev.canvas.draw()
        self.ui.MplWidget_silhouette.canvas.draw()
