from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QSizePolicy, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from data_utils.data_initializer import DataInitializer
from clustering_utils.clusterer_optimizer import ClustererOptimizer, OptimizationResult


class OptimizerWorker(QThread):
    """Worker thread for running optimization methods"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(dict)  # results
    error = pyqtSignal(str)

    def __init__(self, optimizer, method, parameter_value=None):
        super().__init__()
        self.optimizer = optimizer
        self.method = method
        self.parameter_value = parameter_value
        self.results = {}

    def run(self):
        """Run the optimization method"""
        try:
            # Set progress callback
            self.optimizer.progress_callback = lambda c, t, m: self.progress.emit(c, t, m)

            # Run the optimization
            optimal_k, k_values, scores = self.optimizer.find_optimal_k(
                self.method, self.parameter_value
            )

            # Get the result
            result = self.optimizer.results[self.method]

            self.results = {
                'method': self.method,
                'result': result,
                'optimal_k': optimal_k,
                'k_values': k_values,
                'scores': scores
            }

            self.finished.emit(self.results)

        except Exception as e:
            self.error.emit(str(e))


class OptimizerVisualizationWindow(QDialog):
    """
    Window for manual inspection of clustering optimization results.
    Supports interactive visualization of different optimization methods.
    """

    def __init__(self, context, comet_instance, parent=None):
        super(OptimizerVisualizationWindow, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        # Store references
        self.context = context
        self.comet = comet_instance

        # Load the UI
        self.ui = uic.loadUi(context.get_resource("OptimizerVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Exploring the number of microstate maps")

        # Set up the matplotlib figure
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.ui.Figure_Layout.addWidget(self.canvas)

        # Connect signals
        self.ui.optimizer_combobox.activated.connect(self.optimizer_visualization_controller)
        self.ui.optimizer_button.clicked.connect(self.plot_optimizer)

        # Initialize
        self.optimizer = None
        self.results_cache = {}
        self.worker = None

        # Get parameters from COMET instance
        self._load_comet_parameters()

    def _load_comet_parameters(self):
        """Load parameters from COMET instance"""
        # These should be available from the COMET instance
        self.preprocessed_data_path = self.comet.preprocessed_data_path
        self.extension = self.comet.extension
        self.datatype = self.comet.datatype
        self.use_percentages = self.comet.use_percentages
        self.min_distance_size = getattr(self.comet, 'min_distance_size', None)
        self.number_of_repeats = 1  # Single repeat as requested
        self.clustering_tolerance = self.comet.clustering_tolerance
        self.max_iterations = self.comet.max_iterations

    def optimizer_visualization_controller(self):
        """Update UI based on selected optimization method"""
        optimizer_method = self.ui.optimizer_combobox.currentText()

        # Define parameter labels for each method
        param_labels = {
            'Gap Statistic': 'Reference datasets:',
            'Cross Validation': 'Number of folds:',
            'Elbow - Global Explained Variance': 'Threshold (%):',
            'Elbow - Residual Variance': 'Threshold (%):',
            'Silhouette Method': '',
            'Calinski-Harabasz Method': '',
            'Davies-Bouldin Method': ''
        }

        label = param_labels.get(optimizer_method, '')
        self.ui.optimizer_stop_condition_label.setText(label)

        # Show/hide parameter input based on method
        self.ui.optimizer_stopping_threshold_input.setVisible(label != '')

        # Set default values
        if optimizer_method == 'Gap Statistic':
            self.ui.optimizer_stopping_threshold_input.setText('10')
        elif optimizer_method == 'Cross Validation':
            self.ui.optimizer_stopping_threshold_input.setText('5')

    def _get_method_code(self, method_name):
        """Convert method name to internal code"""
        method_map = {
            'Elbow - Global Explained Variance': 'gev',
            'Elbow - Residual Variance': 'res',
            'Silhouette Method': 'sil',
            'Calinski-Harabasz Method': 'ch',
            'Davies-Bouldin Method': 'db',
            'Cross Validation': 'cv',
            'Gap Statistic': 'gs'
        }
        return method_map.get(method_name, 'gev')

    def plot_optimizer(self):
        """Run optimization and plot results"""
        # Disable button during computation
        self.ui.optimizer_button.setEnabled(False)
        self.ui.optimizer_progressbar.setValue(0)

        # Get method and parameters
        method_name = self.ui.optimizer_combobox.currentText()
        method_code = self._get_method_code(method_name)

        # Get parameter value if applicable
        parameter_value = None
        if self.ui.optimizer_stopping_threshold_input.isVisible():
            try:
                parameter_value = float(self.ui.optimizer_stopping_threshold_input.text())
            except ValueError:
                parameter_value = None

        # Check if we already have results for this method
        cache_key = f"{method_code}_{parameter_value}"
        if cache_key in self.results_cache:
            self._display_results(self.results_cache[cache_key])
            self.ui.optimizer_button.setEnabled(True)
            return

        # Initialize optimizer if needed
        if self.optimizer is None:
            self._initialize_optimizer()

        # Create and start worker thread
        self.worker = OptimizerWorker(self.optimizer, method_code, parameter_value)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_optimization_finished)
        self.worker.error.connect(self._on_optimization_error)
        self.worker.start()

    def _initialize_optimizer(self):
        """Initialize the optimizer with data"""
        # Generate maps and peaks
        self.maps2use, self.peaks2use = DataInitializer().generate_maps_and_peaks(
            self.preprocessed_data_path,
            self.extension,
            self.datatype,
            self.use_percentages,
            self.min_distance_size
        )

        # Create optimizer
        kmin = int(self.ui.optimizer_min_input.text())
        kmax = int(self.ui.optimizer_max_input.text())

        self.optimizer = ClustererOptimizer(
            maps2use=self.maps2use,
            min_dist=self.min_distance_size,
            n_inits=self.number_of_repeats,
            kmin=kmin,
            kmax=kmax,
            preprocessed_data_path=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            tolerance=self.clustering_tolerance,
            max_iter=self.max_iterations
        )

    def _update_progress(self, current, total, message):
        """Update progress bar"""
        progress = int((current / total) * 100)
        self.ui.optimizer_progressbar.setValue(progress)
        QApplication.processEvents()  # Keep UI responsive

    def _on_optimization_finished(self, results):
        """Handle optimization completion"""
        # Cache results
        method_code = results['method']
        parameter_value = self.worker.parameter_value
        cache_key = f"{method_code}_{parameter_value}"
        self.results_cache[cache_key] = results

        # Display results
        self._display_results(results)

        # Re-enable button
        self.ui.optimizer_button.setEnabled(True)
        self.ui.optimizer_progressbar.setValue(100)

    def _on_optimization_error(self, error_msg):
        """Handle optimization error"""
        print(f"Optimization error: {error_msg}")
        self.ui.optimizer_button.setEnabled(True)
        self.ui.optimizer_progressbar.setValue(0)

    def _display_results(self, results):
        """Display optimization results"""
        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        result = results['result']

        # Plot based on method type
        if result.higher_is_better:
            marker_color = 'green'
            line_color = 'darkgreen'
        else:
            marker_color = 'red'
            line_color = 'darkred'

        # Main plot
        ax.plot(result.k_values, result.scores, 'o-',
                color=line_color, linewidth=2, markersize=8,
                markerfacecolor=marker_color, markeredgewidth=2,
                markeredgecolor='white')

        # Highlight optimal k
        ax.axvline(x=result.optimal_k, color='red', linestyle='--',
                   linewidth=2, label=f'Optimal k = {result.optimal_k}')

        # Styling
        ax.set_xlabel('Number of Clusters (k)', fontsize=14)
        ax.set_ylabel(self._get_ylabel(result.method_name), fontsize=14)
        ax.set_title(result.method_name, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # Set integer ticks for k values
        ax.set_xticks(result.k_values)

        # Adjust y-axis for better visualization
        y_margin = 0.1 * (max(result.scores) - min(result.scores))
        ax.set_ylim(min(result.scores) - y_margin, max(result.scores) + y_margin)

        # Refresh canvas
        self.canvas.draw()

    def _get_ylabel(self, method_name):
        """Get appropriate y-axis label for method"""
        labels = {
            'Elbow - Global Explained Variance': 'Global Explained Variance',
            'Elbow - Residual Variance': 'Residual Variance',
            'Silhouette Method': 'Silhouette Score',
            'Calinski-Harabasz Method': 'Calinski-Harabasz Index',
            'Davies-Bouldin Method': 'Davies-Bouldin Index',
            'Cross Validation': 'Cross-Validation Score (GEV)',
            'Gap Statistic': 'Gap Score'
        }
        return labels.get(method_name, 'Score')

    def get_all_results_summary(self):
        """Get summary of all computed results"""
        if not self.results_cache:
            return None

        summary = {}
        for cache_key, results in self.results_cache.items():
            method = results['result'].method_name
            optimal_k = results['result'].optimal_k
            summary[method] = optimal_k

        return summary
