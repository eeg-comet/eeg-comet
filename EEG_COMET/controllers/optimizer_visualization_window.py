
import os.path
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from data_utils.data_initializer import DataInitializer
from clustering_utils.clusterer_optimizer import ClustererOptimizer


class OptimizerVisualizationWindow(QDialog):
    def __init__(self, context, parent=None):
        super(OptimizerVisualizationWindow, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("OptimizerVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Exploring the number of microstate maps")

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.ui.Figure_Layout.addWidget(self.canvas)

        self.ui.optimizer_combobox.activated.connect(self.optimizer_visualization_controller)
        self.ui.optimizer_button.clicked.connect(self.plot_optimizer)

    def optimizer_visualization_controller(self):
        optimizer_method = self.ui.optimizer_combobox.currentText()
        stop_conditions = {
            'Gap Statistic': 'Random datasets:',
            'Cross Validation': 'Folds:',
            'Elbow - Global Explained Variance': 'Threshold (%):',
            'Elbow - Residual Variance': 'Threshold (%):',
            'Silhouette Method': '',
            'Calinski-Harabasz Method': '',
            'Davies-Bouldin Method': ''
        }
        self.ui.optimizer_stop_condition_label.setText(stop_conditions.get(optimizer_method, ''))

    def show_optimizer(self, mode, threshold=5):
        if not hasattr(self, f'optimizer_{mode}_done'):
            self.maps2use, self.peaks2use = DataInitializer().generate_maps_and_peaks(
                self.preprocessed_data_path,
                self.extension,
                self.datatype,
                self.use_percentages,
                self.min_distance_size
            )
            self.clusterer_optimizer = ClustererOptimizer(
                self.maps2use,
                self.min_distance_size,
                self.number_of_repeats,
                int(self.ui.optimizer_min_input.text()),
                int(self.ui.optimizer_max_input.text()),
                self.preprocessed_data_path,
                self.extension,
                self.datatype,
                self.clustering_tolerance,
                self.max_iterations
            )

            optimal_k, k_values, target_values = \
                self.clusterer_optimizer.find_optimal_k(mode, threshold)

            # Store the results for the current mode
            self.optimizer_results[mode] = {
                'optimal_k': optimal_k,
                'k_values': k_values,
                'target_values': target_values
            }
            # TODO: save optimizer_results to comet_tbx
            setattr(self, f'optimizer_{mode}_done', True)
        else:
            self.plot_optimizer_chart(mode)

    def plot_optimizer_chart(self, mode):
        self.canvas.figure.clear()
        ax = self.canvas.figure.gca()
        ax.plot(self.optimizer_results[mode]['k_values'], self.optimizer_results[mode]['target_values'],
                marker='o', linestyle='-', color='b', linewidth=2)
        font_size = 14
        ax.set_xlabel('Number of Clusters (K)', fontsize=font_size)
        ax.set_ylabel(self.get_metric_text(mode), fontsize=font_size)
        ax.set_title(f'{self.get_metric_text(mode)} Method', fontsize=font_size)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=font_size)
        ax.axvline(x=self.optimizer_results[mode]['optimal_k'], color='r', linestyle='--', label='Optimal K')
        self.canvas.draw()

    def get_metric_text(self, mode):
        metrics = {'gev': 'Elbow - Global Explained Variance',
                   'res': 'Elbow - Residual Variance',
                   'cv': 'Cross Validation',
                   'gs': 'Gap Statistic',
                   'ch': 'Calinski-Harabasz',
                   'db': 'Davies-Bouldin'}
        return metrics.get(mode, '')

    def plot_optimizer(self):
        optimizer_method = self.ui.optimizer_combobox.currentText()
        threshold = float(self.ui.optimizer_stopping_threshold_input.text())
        if optimizer_method == 'Gap Statistic':
            self.show_optimizer('gs', threshold)
        elif optimizer_method == 'Cross Validation':
            self.show_optimizer('cv', threshold)
        elif optimizer_method == 'Elbow - Global Explained Variance':
            self.show_optimizer('gev', threshold)
        elif optimizer_method == 'Elbow - Residual Variance':
            self.show_optimizer('res', threshold)
        elif optimizer_method == 'Silhouette Method':
            self.show_optimizer('sil', threshold)
        elif optimizer_method == 'Calinski-Harabasz Method':
            self.show_optimizer('ch', threshold)
        elif optimizer_method == 'Davies-Bouldin Method':
            self.show_optimizer('db', threshold)
