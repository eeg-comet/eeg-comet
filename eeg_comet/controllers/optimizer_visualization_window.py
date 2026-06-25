"""Optimizer visualization GUI using modified K-means clustering outputs.

Provides windows to run optimization metrics over microstate clustering
results, visualize method curves, and export figures. Uses a polarity-
independent modified K-means and consolidated metrics implementations.
"""

import time
import traceback
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QActionGroup,
    QApplication,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
)

from eeg_comet.clustering_utils.clusterer_optimizer import ClustererOptimizer
from eeg_comet.clustering_utils.microstate_clusterer import MicrostateClusterer
from eeg_comet.data_utils.data_initializer import DataInitializer
from eeg_comet.gui_utils.export_utils import get_save_file_path, save_matplotlib_figure
from eeg_comet.gui_utils.responsive import apply_window_minimum, expand_canvas
from eeg_comet.gui_utils.terminal_logger import get_logger

 

# ============================================================================
# Worker Thread Classes
# ============================================================================


class OptimizedOptimizerWorker(QThread):
    """Optimized worker thread for computing metrics across K values.

    Args:
      optimizer: Instance providing batch metric computation APIs.
      methods_to_run (list[str]): Method codes to compute (e.g., ["gev", "db"]).
      parameters (dict | None): Optional per-method parameters (e.g., thresholds).

    Signals:
      progress (pyqtSignal): Emits (current:int, total:int, message:str).
      finished (pyqtSignal): Emits results dict when all methods complete.
      error (pyqtSignal): Emits error message when an exception occurs.
    """

    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(dict)  # results
    error = pyqtSignal(str)

    def __init__(self, optimizer, methods_to_run, parameters=None):
        """Create worker with optimizer, methods list, and optional parameters.

        Args:
          optimizer: Optimizer instance used to compute metrics.
          methods_to_run (list[str]): Method codes to compute.
          parameters (dict | None): Optional per-method parameters.
        """
        super().__init__()
        self.optimizer = optimizer
        self.methods_to_run = methods_to_run
        self.parameters = parameters or {}
        self.results = {}

    def run(self):
        """Execute the requested optimization methods.

        Delegates to the underlying optimizer batch API to avoid redundant
        recomputation across methods.
        """
        try:
            # Stream per-k progress from the optimizer to the GUI as a percentage
            # so the progress bar advances during the run instead of jumping to
            # 100% only at the end.
            def _forward_progress(current, total, message):
                pct = int(current / total * 100) if total else 0
                self.progress.emit(min(pct, 99), 100, message)

            self.optimizer.progress_callback = _forward_progress

            # Use the optimised batch computation to minimise repeated work
            batch_results = self.optimizer.compute_methods_batch(
                self.methods_to_run, self.parameters
            )

            # Convert to the expected window cache structure
            for method, result_obj in batch_results.items():
                self.results[method] = {
                    "method": method,
                    "result": result_obj,
                    "optimal_k": result_obj.optimal_k,
                    "original_optimal_k": result_obj.optimal_k,
                    "k_values": result_obj.k_values,
                    "scores": result_obj.scores,
                    "threshold": self.parameters.get(method, None),
                }

            self.progress.emit(100, 100, "All optimisations complete")
            self.finished.emit(self.results)

        except Exception as e:
            error_msg = f"Error in optimization: {str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            self.error.emit(error_msg)


# ============================================================================
# Extended Optimizer Class - Uses ClustererOptimizer metrics
# ============================================================================


class OptimizedMicrostateClustererOptimizer(ClustererOptimizer):
    """Optimizer using modified K-means for microstate clustering.

    All metric computations are inherited from ClustererOptimizer (single source
    of truth), with polarity-invariant behavior.
    """

    def __init__(
        self,
        maps2use,
        min_dist=None,
        n_inits=10,
        kmin=2,
        kmax=10,
        preprocessed_data_path=None,
        extension=None,
        datatype=None,
        tolerance=1e-6,
        max_iter=500,
        batch_size=None,
        logger=None,
    ):
        """Initialize the optimizer with data and hyperparameters.

        Args:
          maps2use (np.ndarray): Input maps matrix.
          min_dist (int | None): Minimum distance between GFP peaks.
          n_inits (int): Number of random initializations.
          kmin (int): Minimum K.
          kmax (int): Maximum K.
          preprocessed_data_path (str | None): Path to preprocessed data.
          extension (str | None): Data extension.
          datatype (str | None): Data type (raw/epoched).
          tolerance (float): Convergence tolerance.
          max_iter (int): Maximum iterations.
          batch_size (int | None): Batch size for modified K-means.
          logger: Logger instance.
        """
        super().__init__(
            maps2use,
            min_dist,
            n_inits,
            kmin,
            kmax,
            preprocessed_data_path,
            extension,
            datatype,
            tolerance,
            max_iter,
            batch_size,
            progress_callback=None,
            logger=logger,
        )

        # Ensure we have a logger instance
        if logger is None:
            logger = get_logger()
        self.logger = logger

        # Prepare data in the correct format for modified K-means (n_channels, n_samples)
        if self.maps2use.shape[0] > self.maps2use.shape[1]:
            self.eeg_data = self.maps2use.T
            self.logger.processing_info(
                "CLUSTERING", "Data matrix transposed to channels × samples"
            )
        else:
            self.eeg_data = self.maps2use

        # Validate data size and provide memory warnings
        n_channels, n_samples = self.eeg_data.shape

        self.batch_size = batch_size

        if self.logger is None:
            self.logger = get_logger()
        self.logger.processing_info(
            "CLUSTERING", f"Visualization optimizer ready (k={kmin}-{kmax})"
        )

    def _perform_single_clustering(self, k):
        """Perform modified K-means clustering for a single K value.

        Args:
          k (int): Number of clusters.

        Returns:
          dict: Clustering result with maps, labels, and metrics.
        """
        self.logger.processing_info("CLUSTERING", f"Clustering k={k} (modified K-means)")
        start_time = time.time()

        # Initialize the MicrostateClusterer for this K
        clusterer = MicrostateClusterer(
            n_states=k,
            batch_size=self.batch_size,
            n_inits=1,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
        )

        n_channels, n_samples = self.eeg_data.shape

        # Run multiple initializations and keep the best result
        best_maps = None
        best_residual = np.inf
        best_labels = None
        best_gev = 0.0

        self.logger.processing_info("CLUSTERING", f"k={k}: running {self.n_inits} initializations")

        for init_attempt in range(self.n_inits):
            try:
                # Random initialization of maps
                initial_maps = np.random.randn(k, n_channels)
                for i in range(k):
                    initial_maps[i] /= np.linalg.norm(initial_maps[i])

                # Run modified K-means
                maps, residual = clusterer.modified_kmeans(
                    self.eeg_data, initial_maps, verbose=False
                )

                # Calculate GEV for this result using the full dataset
                full_dataset = self._get_full_dataset()
                gev = clusterer.compute_gev(full_dataset, maps)

                # Keep the best result based on GEV
                if gev > best_gev:
                    best_gev = gev
                    best_residual = residual
                    best_maps = maps.copy()

                    # Calculate final segmentation
                    activation = best_maps.dot(self.eeg_data)
                    best_labels = np.argmax(np.abs(activation), axis=0)

            except Exception as e:
                self.logger.warning(
                    "CLUSTERING", f"Init {init_attempt+1} failed for k={k}: {str(e)}"
                )
                continue

        if best_maps is None:
            raise RuntimeError(f"All {self.n_inits} initializations failed for k={k}")

        # Create comprehensive clustering result
        clustering_result = {
            "k": k,
            "labels": best_labels,
            "centers": best_maps,
            "data": self.maps2use,
            "eeg_data": self.eeg_data,
            "residual": best_residual,
            "n_iter": None,
        }

        # Compute all metrics efficiently using CONSOLIDATED METHODS from ClustererOptimizer
        self._compute_all_metrics(clustering_result, k, best_labels, best_gev, best_residual)

        elapsed = time.time() - start_time
        print(f"        Completed in {elapsed:.2f}s - GEV: {clustering_result['gev']:.4f}")

        # Show cluster distribution
        unique_labels, counts = np.unique(best_labels, return_counts=True)
        len(best_labels)

        return clustering_result

    def _compute_all_metrics(self, clustering_result, k, best_labels, best_gev, best_residual):
        """Compute all metrics for a clustering result.

        Uses consolidated methods from ClustererOptimizer (single source of truth).

        Args:
          clustering_result (dict): Result container to write metrics into.
          k (int): Number of clusters.
          best_labels (np.ndarray): Segmentation labels.
          best_gev (float): Global explained variance of the best solution.
          best_residual (float): Residual variance of the best solution.
        """
        # Store maps for metric computation
        maps = clustering_result.get("centers")  # shape: (k, n_channels)
        
        # 1. Global Explained Variance (PRIMARY metric for microstates)
        clustering_result["gev"] = best_gev

        # 2. Residual Variance
        clustering_result["residual_variance"] = best_residual

        # 3. Davies-Bouldin Index - USING CONSOLIDATED METHOD
        if k >= 2:
            try:
                db_score = self.compute_custom_davies_bouldin(
                    data=self.eeg_data, labels=best_labels, maps=maps
                )
                clustering_result["davies_bouldin"] = db_score
            except Exception as e:
                print(f"        Warning: Davies-Bouldin computation failed for k={k}: {str(e)}")
                clustering_result["davies_bouldin"] = np.nan
        else:
            clustering_result["davies_bouldin"] = np.nan

        # 4. Cross-Validation Score - USING CONSOLIDATED METHOD
        try:
            cv_score = self._compute_cross_validation_criterion_vectorized(
                self.eeg_data, maps, best_labels
            )
            clustering_result["cv_score"] = cv_score
        except Exception as e:
            print(f"        Warning: CV computation failed for k={k}: {str(e)}")
            clustering_result["cv_score"] = np.nan

        # 5. Silhouette Score - USING CONSOLIDATED METHOD
        if k >= 2:
            try:
                sil_score = self.silhouette_coefficient_correlation(
                    data=self.eeg_data, labels=best_labels
                )
                clustering_result["silhouette_score"] = sil_score
            except Exception as e:
                print(f"        Warning: Silhouette computation failed for k={k}: {str(e)}")
                clustering_result["silhouette_score"] = np.nan
        else:
            clustering_result["silhouette_score"] = -1.0

        # 6. Dunn Index - USING CONSOLIDATED METHOD
        if k >= 2:
            try:
                dunn_score = self.compute_dunn_index(
                    data=self.eeg_data, labels=best_labels, maps=maps
                )
                clustering_result["dunn_index"] = dunn_score
            except Exception as e:
                print(f"        Warning: Dunn Index computation failed for k={k}: {str(e)}")
                clustering_result["dunn_index"] = np.nan
        else:
            clustering_result["dunn_index"] = 0.0

        # 7. Calinski-Harabasz Index - USING CONSOLIDATED METHOD
        if k >= 2:
            try:
                ch_score = self.compute_calinski_harabasz_index(
                    data=self.eeg_data, labels=best_labels, maps=maps
                )
                clustering_result["calinski_harabasz_index"] = ch_score
            except Exception as e:
                print(f"        Warning: Calinski-Harabasz computation failed for k={k}: {str(e)}")
                clustering_result["calinski_harabasz_index"] = np.nan
        else:
            clustering_result["calinski_harabasz_index"] = 0.0

        # 8. Gap Statistic - USING CONSOLIDATED METHOD
        try:
            gap_score, gap_std = self.compute_gap_statistic(
                data=self.eeg_data, labels=best_labels, maps=maps
            )
            clustering_result["gap_statistic"] = gap_score
            clustering_result["gap_std"] = gap_std
        except Exception as e:
            print(f"        Warning: Gap Statistic computation failed for k={k}: {str(e)}")
            clustering_result["gap_statistic"] = np.nan
            clustering_result["gap_std"] = np.nan

        # 9. AIC - USING CONSOLIDATED METHOD
        try:
            aic_score = self.compute_information_criteria(
                data=self.eeg_data, labels=best_labels, maps=maps, criterion='AIC'
            )
            clustering_result["aic_score"] = aic_score
        except Exception as e:
            print(f"        Warning: AIC computation failed for k={k}: {str(e)}")
            clustering_result["aic_score"] = np.nan

        # 10. BIC - USING CONSOLIDATED METHOD
        try:
            bic_score = self.compute_information_criteria(
                data=self.eeg_data, labels=best_labels, maps=maps, criterion='BIC'
            )
            clustering_result["bic_score"] = bic_score
        except Exception as e:
            print(f"        Warning: BIC computation failed for k={k}: {str(e)}")
            clustering_result["bic_score"] = np.nan

        # 11. KL criterion components (for consistency)
        try:
            W_q = self._compute_W_q(self.eeg_data, best_labels, maps)
            n_channels = self.eeg_data.shape[0]
            M_q = W_q * (k ** (2.0 / n_channels))
            clustering_result["W_q"] = W_q
            clustering_result["M_q"] = M_q
            clustering_result["kl_score"] = np.nan  # Will be computed when adjacent k values are available
        except Exception as e:
            print(f"        Warning: KL components computation failed for k={k}: {str(e)}")
            clustering_result["W_q"] = np.nan
            clustering_result["M_q"] = np.nan
            clustering_result["kl_score"] = np.nan

    def get_microstate_maps(self, k):
        """Get the final microstate maps for a specific k value.

        Args:
          k (int): Number of clusters.

        Returns:
          np.ndarray: Microstate maps (k × n_channels).
        """
        result = self._perform_single_clustering(k)
        return result["centers"]

    def compute_microstate_metrics(self, k, verbose=True):
        """Compute comprehensive microstate-specific metrics for a given k.

        Args:
          k (int): Number of clusters.
          verbose (bool): If True, prints a summary to stdout.

        Returns:
          dict: Dictionary of computed metrics and outputs.
        """
        result = self._perform_single_clustering(k)

        metrics = {
            "k": k,
            "gev": result["gev"],
            "residual_variance": result["residual_variance"],
            "davies_bouldin": result["davies_bouldin"],
            "cv_score": result["cv_score"],
            "silhouette_score": result["silhouette_score"],
            "dunn_index": result["dunn_index"],
            "calinski_harabasz_index": result["calinski_harabasz_index"],
            "gap_statistic": result["gap_statistic"],
            "aic_score": result["aic_score"],
            "bic_score": result["bic_score"],
            "microstate_maps": result["centers"],
            "segmentation": result["labels"],
        }

        if verbose:
            print(f"\nMicrostate Metrics for k={k}:")
            print(f"  Global Explained Variance: {metrics['gev']:.4f}")
            print(f"  Residual Variance: {metrics['residual_variance']:.6f}")
            print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
            print(f"  Cross-Validation Score: {metrics['cv_score']:.4f}")
            print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"  Dunn Index: {metrics['dunn_index']:.4f}")
            print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.2f}")
            print(f"  Gap Statistic: {metrics['gap_statistic']:.4f}")
            print(f"  AIC Score: {metrics['aic_score']:.4f}")
            print(f"  BIC Score: {metrics['bic_score']:.4f}")
            print("  All metrics computed using polarity-invariant spatial correlation")

        return metrics


# ============================================================================
# Main Window Class
# ============================================================================


class OptimizerVisualizationWindow(QMainWindow):
    """Window to inspect microstate clustering optimization results.

    Uses a polarity-invariant modified K-means algorithm and consolidated
    metrics from ClustererOptimizer to compute GEV, DB, CV, and KL criteria.

    Attributes:
      context: Resource provider used to load UI assets.
      comet: COMET toolbox instance for parameters and persistence.
      optimizer: Optimizer instance (initialized on demand).
      results_cache (dict[str, Any]): Cached results keyed by method code.
    """

    def __init__(self, context, comet_instance, parent=None):
        """Initialize the window with persistence and UI setup.

        Args:
          context: Resource/context provider used to resolve UI resources.
          comet_instance: COMET toolbox instance.
          parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        # Store references
        self.context = context
        self.comet = comet_instance

        # Initialize data structures
        self._init_data_structures()

        # Setup UI
        self._setup_ui()

        # Setup font system
        self._setup_font_system()

        # Initialize connections
        self._setup_connections()

        # Load initial state
        self._initialize_state()

    def closeEvent(self, event):
        """Disconnect signals and wait for the worker before closing.

        We can't interrupt ``compute_methods_batch`` mid-call, but we can stop
        its emissions from reaching slots on this (about-to-be-deleted) window.
        """
        # Ask the optimizer to stop so the background batch exits at the next
        # k boundary rather than running to completion after the window closes.
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "stop"):
            try:
                optimizer.stop()
            except Exception:
                pass

        worker = getattr(self, "worker", None)
        if worker is not None:
            for sig_name in ("progress", "finished", "error"):
                sig = getattr(worker, sig_name, None)
                if sig is not None:
                    try:
                        sig.disconnect()
                    except (TypeError, RuntimeError):
                        pass
            try:
                if worker.isRunning():
                    worker.wait(2000)
            except RuntimeError:
                pass
            self.worker = None
        super().closeEvent(event)

    # ========================================================================
    # Initialization Methods
    # ========================================================================

    def _init_data_structures(self):
        """Initialize core data structures."""
        self.optimizer = None
        self.results_cache = {}
        self.worker = None
        self.all_methods_complete = False
        self.last_used_parameters = {}
        self.current_font_family = "Arial"
        self.current_font_size = "Large"
        # Logger
        self.logger = get_logger()

    def _setup_ui(self):
        """Set up the UI components."""
        self.ui = uic.loadUi(self.context.get_resource("OptimizerVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Exploring the number of microstate maps (Modified K-means)")
        apply_window_minimum(self, "tool")
        if hasattr(self.ui, "optimizer_top_label"):
            self.ui.optimizer_top_label.setProperty("role", "banner")

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        expand_canvas(self.canvas, minimum=(360, 280))
        self.ui.Figure_Layout.addWidget(self.canvas)

    def _setup_font_system(self):
        """Set up font menu action groups."""
        # Font family action group
        self.font_family_group = QActionGroup(self)
        self.font_family_group.addAction(self.ui.actionArial)
        self.font_family_group.addAction(self.ui.actionCalibri)
        self.font_family_group.addAction(self.ui.actionTimes_New_Roman)

        # Font size action group
        self.font_size_group = QActionGroup(self)
        self.font_size_group.addAction(self.ui.actionSmall)
        self.font_size_group.addAction(self.ui.actionMedium)
        self.font_size_group.addAction(self.ui.actionLarge)
        self.font_size_group.addAction(self.ui.actionX_Large)

    def _setup_connections(self):
        """Set up all signal-slot connections."""
        # Main controls
        self.ui.optimizer_combobox.activated.connect(self.optimizer_visualization_controller)
        self.ui.optimizer_button.clicked.connect(self.run_all_analyses)
        self.ui.visualize_button.clicked.connect(self.visualize_selected_method)
        self.ui.export_figure_image_button.triggered.connect(self.export_plot)

        # Threshold input
        self.ui.optimizer_stopping_threshold_input.returnPressed.connect(self._on_threshold_changed)

        # Font actions
        self.ui.actionArial.triggered.connect(lambda: self._set_font_family("Arial"))
        self.ui.actionCalibri.triggered.connect(lambda: self._set_font_family("Calibri"))
        self.ui.actionTimes_New_Roman.triggered.connect(
            lambda: self._set_font_family("Times New Roman")
        )

        self.ui.actionSmall.triggered.connect(lambda: self._set_font_size("Small"))
        self.ui.actionMedium.triggered.connect(lambda: self._set_font_size("Medium"))
        self.ui.actionLarge.triggered.connect(lambda: self._set_font_size("Large"))
        self.ui.actionX_Large.triggered.connect(lambda: self._set_font_size("X-Large"))

    def _initialize_state(self):
        """Initialize window state, loading parameters and previous results."""
        # Get parameters from COMET instance
        self._load_comet_parameters()

        # Try to load previous optimization results
        self._load_previous_results()

        # Set initial UI state based on whether we have previous results
        if self.results_cache:
            self._setup_with_results()
        else:
            self._setup_without_results()

        # Final UI state verification
        self.verify_and_fix_ui_state()

    def _setup_with_results(self):
        """Set up UI when previous results are available."""
        # Enable visualization UI
        self._enable_visualization_ui()

        # Update button text
        self.ui.optimizer_button.setText("Re-run All Analyses")

        # Set status message
        method_count = len(self.results_cache)
        k_range = self._get_k_range_text()
        status_msg = f"Loaded {method_count} optimization method(s){k_range} - Ready to visualize!"
        self.ui.statusbar.showMessage(status_msg)

        print(f"Loaded previous optimization results for {len(self.results_cache)} methods")
        print("Visualization UI enabled - ready to display results")

    def _setup_without_results(self):
        """Set up UI when no previous results are available."""
        # Disable visualization initially
        self._disable_visualization_ui()

        # Add status message
        self.ui.statusbar.showMessage(
            "Ready - Using polarity-independent modified K-means algorithm"
        )

    def _get_k_range_text(self) -> str:
        """Return K range text from results."""
        if not self.results_cache:
            return ""

        first_result = next(iter(self.results_cache.values()))
        k_values = first_result.get("k_values", [])
        if k_values:
            return f" (K: {min(k_values)}-{max(k_values)})"
        return ""

    # ========================================================================
    # Parameter Loading Methods
    # ========================================================================

    def _load_comet_parameters(self):
        """Load parameters from COMET instance."""
        self.preprocessed_data_path = self.comet.preprocessed_data_path
        self.extension = self.comet.extension
        self.datatype = self.comet.datatype
        self.use_percentages = self.comet.use_percentages
        self.min_distance_size = getattr(self.comet, "min_distance_size", None)
        self.number_of_repeats = getattr(self.comet, "number_of_repeats", 10)
        self.clustering_tolerance = self.comet.clustering_tolerance
        self.max_iterations = self.comet.max_iterations
        self.batch_size = getattr(self.comet, "batch_size", None)

    # ========================================================================
    # Results Loading and Saving Methods
    # ========================================================================

    def _load_previous_results(self):
        """Load previous optimization results and update the GUI."""
        try:
            previous_results = self.comet.load_optimization_results()
            if previous_results:
                self.results_cache = previous_results
                print(
                    f"Successfully loaded previous optimization results for {len(previous_results)} methods"
                )

                # Log the loaded results
                self._log_loaded_results(previous_results)

                # Update GUI to reflect loaded results
                self._update_gui_for_loaded_results()

            else:
                self.results_cache = {}
        except Exception as e:
            print(f"Error loading previous optimization results: {str(e)}")
            self.results_cache = {}

    def _log_loaded_results(self, results: dict[str, Any]):
        """Log loaded optimization results.

        Args:
          results (dict[str, Any]): Cached results keyed by method code.
        """
        print("Loaded optimization results:")
        for method_code, method_results in results.items():
            method_name = self._get_method_name(method_code)
            optimal_k = method_results["optimal_k"]
            k_range = f"{min(method_results['k_values'])}-{max(method_results['k_values'])}"
            threshold_info = (
                f" (threshold: {method_results['threshold']}%)" if method_results.get("threshold") else ""
            )
            print(f"  {method_name}: k={optimal_k}, range={k_range}{threshold_info}")

    def _update_gui_for_loaded_results(self):
        """Update GUI elements when previous results are loaded."""
        if not self.results_cache:
            print("No results to update GUI with")
            return

        print("Updating GUI for loaded optimization results...")

        # 1. Enable all visualization UI elements
        self._enable_visualization_ui()

        # 2. Populate K range from loaded results
        self._update_k_range_from_results()

        # 3. Populate method combo box with available methods
        self._populate_method_combo_box()

        # 4. Set the default method and update parameters
        self._set_default_method_selection()

        # 5. Update last used parameters from loaded results
        self._restore_method_parameters()

        # 6. Update the visualization controller for the current method
        self.optimizer_visualization_controller()

        # 7. Automatically display the first available method
        self._auto_display_first_method()

        print("GUI successfully updated for loaded results")

    def _update_k_range_from_results(self):
        """Update K range input fields based on loaded results."""
        if not self.results_cache:
            return

        # Get K range from the first available result
        first_result = next(iter(self.results_cache.values()))
        k_values = first_result.get("k_values", [])

        if k_values:
            kmin = min(k_values)
            kmax = max(k_values)

            # Update the UI input fields
            self.ui.optimizer_min_input.setText(str(kmin))
            self.ui.optimizer_max_input.setText(str(kmax))

            print(f"Updated K range to: {kmin} - {kmax}")

    def _populate_method_combo_box(self):
        """Populate the method combo box with available methods from loaded results."""
        if not self.results_cache:
            return

        # Get available methods from loaded results
        available_methods = []
        method_display_names = self._get_method_display_names()

        for method_code in self.results_cache:
            if method_code in method_display_names:
                available_methods.append(method_display_names[method_code])

        # Clear and populate combo box
        self.ui.optimizer_combobox.clear()
        if available_methods:
            self.ui.optimizer_combobox.addItems(available_methods)
            print(f"Populated combo box with {len(available_methods)} methods: {available_methods}")

    def _set_default_method_selection(self):
        """Set the default method selection in the combo box."""
        if not self.results_cache:
            return

        # Try to select GEV as default, or the first available method
        preferred_order = ["gev", "db", "cv", "kl", "sil", "dunn", "ch", "gap", "aic", "bic"]

        for method_code in preferred_order:
            if method_code in self.results_cache:
                method_name = self._get_method_name(method_code)

                # Find and select this method in the combo box
                for i in range(self.ui.optimizer_combobox.count()):
                    if self.ui.optimizer_combobox.itemText(i) == method_name:
                        self.ui.optimizer_combobox.setCurrentIndex(i)
                        print(f"Set default method selection to: {method_name}")
                        break
                break

    def _restore_method_parameters(self):
        """Restore method parameters from loaded results."""
        if not self.results_cache:
            return

        # Restore last used parameters for all methods
        for method_code, results in self.results_cache.items():
            threshold = results.get("threshold")
            if threshold is not None:
                self.last_used_parameters[method_code] = threshold

        print(f"Restored parameters for {len(self.last_used_parameters)} methods")

    def _auto_display_first_method(self):
        """Automatically display the first available method."""
        if not self.results_cache or self.ui.optimizer_combobox.count() == 0:
            return

        try:
            # Trigger the controller to update the UI for the selected method
            self.optimizer_visualization_controller()

            # Auto-visualize the first method
            if self.ui.visualize_button.isEnabled():
                print("Auto-displaying first method visualization...")
                self.visualize_selected_method()

        except Exception as e:
            print(f"Error auto-displaying first method: {str(e)}")

    def _save_results_to_comet(self):
        """Save current optimization results to COMET configuration."""
        try:
            if self.results_cache and self.comet:
                self.comet.save_optimization_results(self.results_cache)

                # Save the updated configuration
                if self.comet.auto_save:
                    self.comet.save_config()
                    print("Optimization results saved to configuration file")
            else:
                print("No results to save or COMET instance not available")
        except Exception as e:
            print(f"Error saving optimization results: {str(e)}")

    # ========================================================================
    # UI State Management Methods
    # ========================================================================

    def _enable_visualization_ui(self):
        """Enable visualization UI elements when results are available."""
        self.ui.visualize_button.setEnabled(True)
        self.ui.optimizer_combobox.setEnabled(True)
        self.ui.optimizer_method_label.setEnabled(True)
        self.ui.optimizer_stop_condition_label.setEnabled(True)
        self.ui.optimizer_stopping_threshold_input.setEnabled(True)
        self.all_methods_complete = True

    def _disable_visualization_ui(self):
        """Disable visualization UI elements when no results are available."""
        self.ui.visualize_button.setEnabled(False)
        self.ui.optimizer_combobox.setEnabled(False)
        self.ui.optimizer_method_label.setEnabled(False)
        self.ui.optimizer_stop_condition_label.setEnabled(False)
        self.ui.optimizer_stopping_threshold_input.setEnabled(False)
        self.all_methods_complete = False

    def verify_and_fix_ui_state(self):
        """Verify and fix the UI state based on available results."""
        if self.results_cache and len(self.results_cache) > 0:
            # We have results, ensure UI is enabled
            if not self.ui.visualize_button.isEnabled():
                print("Fixing UI state: Enabling visualization controls")
                self._enable_visualization_ui()

            # Ensure combo box is populated
            if self.ui.optimizer_combobox.count() == 0:
                print("Fixing UI state: Populating method combo box")
                self._populate_method_combo_box()
                self._set_default_method_selection()

            # Update status
            method_count = len(self.results_cache)
            self.ui.statusbar.showMessage(
                f"{method_count} optimization results available - Ready to visualize!"
            )

        else:
            # No results, ensure UI is disabled
            if self.ui.visualize_button.isEnabled():
                print("Fixing UI state: Disabling visualization controls")
                self._disable_visualization_ui()

            self.ui.statusbar.showMessage("No optimization results available - Run analysis first")

    # ========================================================================
    # UI Controller Methods
    # ========================================================================

    def optimizer_visualization_controller(self):
        """Update UI based on the selected optimization method.

        Supports both freshly computed results and results loaded from disk.
        """
        optimizer_method = self.ui.optimizer_combobox.currentText()

        if not optimizer_method:
            return

        # Define parameter labels for each method
        param_labels = {
            "Cross Validation Criterion": "",
            "Global Explained Variance Criterion": "Threshold (%):",
            "Davies-Bouldin Criterion": "",
            "Krzanowski-Lai Criterion": "",
        }

        label = param_labels.get(optimizer_method, "")
        self.ui.optimizer_stop_condition_label.setText(label)

        # Show/hide parameter input based on method
        self.ui.optimizer_stopping_threshold_input.setVisible(label != "")

        # Set values based on loaded results or defaults
        method_code = self._get_method_code(optimizer_method)

        # First, try to get value from loaded results
        if method_code in self.results_cache:
            cached_result = self.results_cache[method_code]
            threshold = cached_result.get("threshold")

            if threshold is not None:
                self.ui.optimizer_stopping_threshold_input.setText(str(threshold))
                print(f"Set threshold from loaded results: {threshold}")
            else:
                # Set default values if no threshold in loaded results
                self._set_default_threshold(optimizer_method)

        # Second, try last used parameters
        elif method_code in self.last_used_parameters:
            # Restore last used value
            self.ui.optimizer_stopping_threshold_input.setText(
                str(self.last_used_parameters[method_code])
            )
            print(f"Set threshold from last used: {self.last_used_parameters[method_code]}")

        else:
            # Set default values
            self._set_default_threshold(optimizer_method)

    def _set_default_threshold(self, optimizer_method: str):
        """Set default threshold values for threshold-based methods."""
        if optimizer_method in ["Global Explained Variance Criterion", "Elbow - Residual Variance"]:
            self.ui.optimizer_stopping_threshold_input.setText("5")
        else:
            # No threshold needed
            self.ui.optimizer_stopping_threshold_input.clear()

    def _on_threshold_changed(self):
        """Handle threshold change by automatically updating visualization."""
        if self.all_methods_complete and self.ui.visualize_button.isEnabled():
            # Only update if we have results and the current method uses thresholds
            method_code = self._get_method_code(self.ui.optimizer_combobox.currentText())
            if method_code in ["gev"]:
                self.visualize_selected_method()

    # ========================================================================
    # Font Management Methods
    # ========================================================================

    def _set_font_family(self, family: str):
        """Update font family.

        Args:
          family (str): Font family name.
        """
        self.current_font_family = family
        if hasattr(self, "current_plot_method"):
            self.visualize_selected_method()

    def _set_font_size(self, size: str):
        """Update font size.

        Args:
          size (str): One of {"Small","Medium","Large","X-Large"}.
        """
        self.current_font_size = size
        if hasattr(self, "current_plot_method"):
            self.visualize_selected_method()

    def _get_font_size(self) -> int:
        """Get numerical font size based on current setting.

        Returns:
          int: Point size for labels/ticks.
        """
        sizes = {"Small": 10, "Medium": 12, "Large": 14, "X-Large": 16}
        return sizes.get(self.current_font_size, 14)

    # ========================================================================
    # Analysis Methods
    # ========================================================================

    def run_all_analyses(self):
        """Run optimization for all available methods using modified K-means."""
        # Ask user if they want to overwrite existing results
        if self.results_cache:
            reply = QMessageBox.question(
                self,
                "Overwrite Previous Results?",
                "Previous optimization results exist. Do you want to re-run the analysis and overwrite them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.No:
                return

        # Reset button text
        self.ui.optimizer_button.setText("Run All Analyses")

        # Disable button during computation
        self.ui.optimizer_button.setEnabled(False)
        self.ui.optimizer_progressbar.setValue(0)

        # Clear previous results
        self.results_cache.clear()
        self.all_methods_complete = False

        # Initialize optimizer if needed
        if self.optimizer is None:
            self._initialize_optimizer()

        # Verify K range
        kmin = int(self.ui.optimizer_min_input.text())
        kmax = int(self.ui.optimizer_max_input.text())

        if kmin >= kmax:
            print(f"Error: Invalid K range. Min ({kmin}) must be less than Max ({kmax})")
            self.ui.optimizer_button.setEnabled(True)
            self.ui.statusbar.showMessage("Error: Invalid K range")
            return

        self.logger.processing_info(
            "CLUSTERING",
            f"Running visualization analyses (modified K-means), k range {kmin}-{kmax}, total {kmax-kmin+1}",
        )

        # Get all methods to run
        all_methods = ["gev", "db", "cv", "kl", "sil", "dunn", "ch", "gap", "aic", "bic"]

        # Get parameters from UI for methods that need them
        parameters = self._get_method_parameters()

        if parameters is None:
            return  # Error occurred in parameter validation

        # Create and start optimized worker thread
        self.worker = OptimizedOptimizerWorker(self.optimizer, all_methods, parameters)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_all_analyses_finished)
        self.worker.error.connect(self._on_optimization_error)
        self.worker.start()

    def _initialize_optimizer(self):
        """Initialize the optimizer with data using modified K-means."""
        # Generate maps and peaks
        self.maps2use, self.peaks2use = DataInitializer().generate_maps_and_peaks(
            self.preprocessed_data_path,
            self.extension,
            self.datatype,
            self.use_percentages,
            self.min_distance_size,
        )

        # Create optimized optimizer
        kmin = int(self.ui.optimizer_min_input.text())
        kmax = int(self.ui.optimizer_max_input.text())

        # Validate K range
        if kmin < 2:
            kmin = 2
            self.ui.optimizer_min_input.setText("2")
            print("Adjusted kmin to 2 (minimum allowed)")

        if kmax <= kmin:
            kmax = kmin + 5
            self.ui.optimizer_max_input.setText(str(kmax))
            print(f"Adjusted kmax to {kmax} (must be greater than kmin)")

        # Use the microstate-specific optimizer with modified K-means
        self.optimizer = OptimizedMicrostateClustererOptimizer(
            maps2use=self.maps2use,
            min_dist=self.min_distance_size,
            n_inits=self.number_of_repeats,
            kmin=kmin,
            kmax=kmax,
            preprocessed_data_path=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            tolerance=self.clustering_tolerance,
            max_iter=self.max_iterations,
            batch_size=self.batch_size,
        )

    def _get_method_parameters(self) -> Optional[dict[str, float]]:
        """Get parameters from UI for methods that need them.

        Returns:
          dict[str, float] | None: Parameters or None if invalid.
        """
        parameters = {}

        # Read threshold values from UI for all methods
        try:
            threshold_value = float(self.ui.optimizer_stopping_threshold_input.text())
            if threshold_value <= 0:
                print("Error: Threshold must be positive")
                self.ui.statusbar.showMessage("Error: Threshold must be positive")
                self.ui.optimizer_button.setEnabled(True)
                return None
        except ValueError:
            print("Error: Invalid threshold value")
            self.ui.statusbar.showMessage("Error: Invalid threshold value")
            self.ui.optimizer_button.setEnabled(True)
            return None

        # Apply threshold to methods that use it
        parameters["gev"] = threshold_value

        # Store the parameters used
        self.last_used_parameters = parameters.copy()

        return parameters

    def _update_progress(self, current: int, total: int, message: str):
        """Update progress bar.

        Args:
          current (int): Current progress value.
          total (int): Total progress range upper bound.
          message (str): Status message to show.
        """
        self.ui.optimizer_progressbar.setValue(current)
        self.ui.statusbar.showMessage(message)
        QApplication.processEvents()  # Keep UI responsive

    def _on_all_analyses_finished(self, all_results: dict[str, Any]):
        """Handle completion of all analyses and save results.

        Args:
          all_results (dict[str, Any]): Results keyed by method code.
        """
        self.all_methods_complete = True
        self.results_cache = all_results

        # CRITICAL: Save results to COMET configuration
        self._save_results_to_comet()

        # Enable visualization controls
        self._enable_visualization_ui()

        # Re-enable run button and update text
        self.ui.optimizer_button.setEnabled(True)
        self.ui.optimizer_button.setText("Re-run All Analyses")
        self.ui.optimizer_progressbar.setValue(100)
        self.ui.statusbar.showMessage(
            "All modified K-means optimization methods completed and saved!"
        )

        # Update visualization controller for first method
        self.optimizer_visualization_controller()

        print("\n" + "=" * 60)
        print("ALL MICROSTATE OPTIMIZATION METHODS COMPLETED AND SAVED!")
        print("Using modified K-means (polarity-independent)")
        print("Using consolidated metrics from ClustererOptimizer (single source of truth)")
        print("=" * 60)

        # Display detailed results
        self._display_optimization_summary(all_results)

    def _display_optimization_summary(self, results: dict[str, Any]):
        """Display summary of optimization results.

        Args:
          results (dict[str, Any]): Results keyed by method code.
        """
        print("\n[CLUSTERING] Optimization results summary:")

        for method_code, method_results in results.items():
            method_name = self._get_method_name(method_code)
            optimal_k = method_results["optimal_k"]
            threshold = method_results.get("threshold", None)

            threshold_info = (
                f" (threshold: {threshold}%)"
                if threshold is not None and method_code in ["gev"]
                else ""
            )
            print(f"[CLUSTERING] {method_name}: Optimal k = {optimal_k}{threshold_info}")

        print("[CLUSTERING] All results computed using modified K-means algorithm")
        print("[CLUSTERING] Results saved and available for visualization")

    def _on_optimization_error(self, error_msg: str):
        """Handle optimization error.

        Args:
          error_msg (str): Error message.
        """
        print(f"[ERROR] Optimization error: {error_msg}")
        self.ui.optimizer_button.setEnabled(True)
        self.ui.optimizer_progressbar.setValue(0)
        self.ui.statusbar.showMessage(f"Error: {error_msg}")

    # ========================================================================
    # Visualization Methods
    # ========================================================================

    def visualize_selected_method(self):
        """Visualize the currently selected optimization method.

        Supports loaded results and threshold updates.
        """
        if not self.all_methods_complete and not self.results_cache:
            print("Cannot visualize: analysis not complete and no loaded results")
            return

        method_name = self.ui.optimizer_combobox.currentText()
        if not method_name:
            print("Cannot visualize: no method selected")
            return

        method_code = self._get_method_code(method_name)

        if method_code in self.results_cache:
            self.current_plot_method = method_code

            # Check if this is a threshold-based method and if threshold has changed
            if method_code in ["gev"]:
                self._handle_threshold_visualization(method_code)
            else:
                self.ui.statusbar.showMessage(f"Displaying {method_name} (Modified K-means)")

            self._display_results(self.results_cache[method_code])
        else:
            print(f"Cannot visualize: method {method_code} not in results cache")
            self.ui.statusbar.showMessage(f"No results available for {method_name}")

    def _handle_threshold_visualization(self, method_code: str):
        """Handle visualization for threshold-based methods.

        Args:
          method_code (str): Method code (e.g., 'gev').
        """
        try:
            current_threshold = float(self.ui.optimizer_stopping_threshold_input.text())
            if current_threshold <= 0:
                self.ui.statusbar.showMessage("Error: Threshold must be positive")
                return

            cached_threshold = self.results_cache[method_code].get("threshold", None)

            if cached_threshold is not None and current_threshold != cached_threshold:
                # Recalculate optimal K with new threshold
                self._recalculate_optimal_k_with_new_threshold(method_code, current_threshold)
                # Save the updated results
                self._save_results_to_comet()
                self.ui.statusbar.showMessage(
                    f"Optimal K recalculated with threshold {current_threshold}%"
                )
            else:
                method_name = self._get_method_name(method_code)
                self.ui.statusbar.showMessage(f"Displaying {method_name} (Modified K-means)")
        except ValueError:
            self.ui.statusbar.showMessage("Error: Invalid threshold value")
            return

    def _recalculate_optimal_k_with_new_threshold(self, method_code: str, new_threshold: float):
        """Recalculate optimal K using a new threshold without re-running analysis.

        Args:
          method_code (str): Method code (e.g., 'gev').
          new_threshold (float): New threshold percentage.
        """
        print(
            f"\nRecalculating optimal K for {self._get_method_name(method_code)} with new threshold: {new_threshold}%"
        )

        # Get the existing results
        cached_results = self.results_cache[method_code]
        k_values = cached_results["k_values"]
        scores = cached_results["scores"]

        # Filter out NaN values for optimal K finding
        valid_scores = [(k, s) for k, s in zip(k_values, scores) if not np.isnan(s)]

        if valid_scores:
            valid_k_values, valid_scores_list = zip(*valid_scores)

            # Use the same elbow detection algorithm with the new threshold
            # Delegate elbow detection to the optimiser implementation to avoid
            # duplicate local logic.
            optimal_k = self.optimizer._find_elbow_point(
                list(valid_k_values),
                list(valid_scores_list),
                threshold=new_threshold,
                higher_is_better=(method_code == "gev"),
            )
        else:
            optimal_k = k_values[0] if k_values else 2

        # Update the cached results with new optimal K and threshold
        cached_results["optimal_k"] = optimal_k
        cached_results["threshold"] = new_threshold
        cached_results["result"].optimal_k = optimal_k

        # Update the last used parameters
        self.last_used_parameters[method_code] = new_threshold

        print(
            f"New optimal K: {optimal_k} (was: {cached_results.get('original_optimal_k', 'unknown')})"
        )

    @staticmethod
    def _find_elbow_point_for_visualization(
        k_values: list[int],
        scores: list[float],
        threshold: float,
        higher_is_better: bool = True,
    ) -> int:
        """Find elbow point using threshold for percentage change (visualization updates).

        Args:
          k_values (list[int]): Candidate K values.
          scores (list[float]): Metric scores.
          threshold (float): Percentage change threshold.
          higher_is_better (bool): If True, higher scores are better.

        Returns:
          int: Selected K value.
        """
        if len(scores) < 2:
            return k_values[0]

        print(f"  Finding elbow point with threshold {threshold}%:")

        for i in range(1, len(scores)):
            # Calculate percentage change
            if scores[i - 1] != 0:
                change_percent = abs((scores[i] - scores[i - 1]) / scores[i - 1]) * 100
            else:
                change_percent = 100.0

            print(f"    K={k_values[i - 1]} to K={k_values[i]}: {change_percent:.2f}% change")

            # Check if improvement is below threshold
            if change_percent < threshold:
                print(
                    f"    Elbow detected at K={k_values[i]} (change {change_percent:.2f}% < threshold {threshold}%)"
                )
                return k_values[i]

        print(f"    No elbow found with threshold {threshold}%, returning last K={k_values[-1]}")
        return k_values[-1]

    def _display_results(self, results: dict[str, Any]):
        """Display optimization results.

        Args:
          results (dict[str, Any]): Result structure for a method code.
        """
        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        result = results["result"]
        threshold = results.get("threshold")
        method_code = results["method"]

        # Get font settings early
        font_size = self._get_font_size()
        font_family = self.current_font_family

        # Plot based on method type
        if result.higher_is_better:
            marker_color = "green"
            line_color = "darkgreen"
        else:
            marker_color = "red"
            line_color = "darkred"

        # Main plot
        ax.plot(
            result.k_values,
            result.scores,
            "o-",
            color=line_color,
            linewidth=2,
            markersize=8,
            markerfacecolor=marker_color,
            markeredgewidth=2,
            markeredgecolor="white",
        )

        # For elbow methods, show the threshold region and percentage changes
        if method_code in ["gev"] and threshold is not None:
            self._add_elbow_annotations(ax, result, threshold)

        # Highlight optimal k
        optimal_label = f"Optimal k = {result.optimal_k}"
        if threshold is not None and method_code in ["gev"]:
            optimal_label += f" (threshold: {threshold}%)"

        ax.axvline(
            x=result.optimal_k, color="red", linestyle="--", linewidth=2, label=optimal_label
        )

        # Styling with custom font
        ax.set_xlabel("Number of Clusters (k)", fontsize=font_size, fontfamily=font_family)
        ax.set_ylabel(
            self._get_ylabel(result.method_name), fontsize=font_size, fontfamily=font_family
        )

        # Update title to show algorithm and parameters
        title = self._create_plot_title(result, results, threshold, method_code)
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold", fontfamily=font_family)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=font_size - 2, prop={"family": font_family})

        # Set integer ticks for k values
        ax.set_xticks(result.k_values)

        # Update tick label font
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontsize(font_size - 2)

        # Adjust y-axis for better visualization
        self._adjust_y_axis(ax, result.scores)

        # Refresh canvas
        self.canvas.draw()

    def _add_elbow_annotations(self, ax, result: Any, threshold: float):
        """Add elbow method annotations to the plot.

        Args:
          ax (matplotlib.axes.Axes): Axis to annotate.
          result: Result object with k_values and scores.
          threshold (float): Percentage threshold for elbow detection.
        """
        # Find where the elbow detection happens
        valid_indices = [i for i, score in enumerate(result.scores) if not np.isnan(score)]
        if len(valid_indices) > 1:
            elbow_found = False
            for i in range(1, len(valid_indices)):
                idx = valid_indices[i]
                prev_idx = valid_indices[i - 1]

                if result.scores[prev_idx] != 0:
                    change_percent = (
                        abs(
                            (result.scores[idx] - result.scores[prev_idx]) / result.scores[prev_idx]
                        )
                        * 100
                    )

                    # Add percentage change annotations
                    mid_k = (result.k_values[prev_idx] + result.k_values[idx]) / 2
                    mid_y = (result.scores[prev_idx] + result.scores[idx]) / 2

                    # Color based on whether it's above or below threshold
                    if change_percent < threshold:
                        color = "red"
                        weight = "bold"
                        if not elbow_found:
                            # Highlight the region where change is below threshold
                            ax.axvspan(
                                result.k_values[prev_idx],
                                result.k_values[idx],
                                alpha=0.2,
                                color="yellow",
                                label=f"Below threshold ({threshold}%)",
                            )
                            elbow_found = True
                    else:
                        color = "black"
                        weight = "normal"

                    # Add text annotation
                    ax.annotate(
                        f"{change_percent:.1f}%",
                        xy=(mid_k, mid_y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=self._get_font_size() - 4,
                        color=color,
                        weight=weight,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8
                        ),
                    )

    @staticmethod
    def _create_plot_title(
        result: Any, results: dict[str, Any], threshold: Optional[float], method_code: str
    ) -> str:
        """Create the plot title.

        Args:
          result: Result object with method name, k_values, optimal_k.
          results (dict[str, Any]): Cached entry for the current method.
          threshold (float | None): Threshold used (if any).
          method_code (str): Method code (e.g., 'gev').

        Returns:
          str: Title text.
        """
        title_parts = [result.method_name]

        # Add specific notes for custom implementations
        if method_code in ["db"]:
            title_parts[0] += " (Consolidated Implementation)"
        else:
            title_parts[0] += " (Modified K-means)"

        title_parts.append(f"K range: {result.k_values[0]} to {result.k_values[-1]}")

        if threshold is not None and method_code in ["gev"]:
            title_parts.append(f"Threshold: {threshold}%")

            # Show if threshold was changed after analysis
            original_optimal = results.get("original_optimal_k")
            if original_optimal and original_optimal != result.optimal_k:
                title_parts.append(f"(Original optimal K was {original_optimal})")

        return "\n".join(title_parts)

    @staticmethod
    def _adjust_y_axis(ax, scores: list[float]):
        """Adjust y-axis for better visualization.

        Args:
          ax (matplotlib.axes.Axes): Axis to adjust.
          scores (list[float]): Score values.
        """
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            y_margin = 0.1 * (max(valid_scores) - min(valid_scores))
            ax.set_ylim(min(valid_scores) - y_margin, max(valid_scores) + y_margin)

    # ========================================================================
    # Export Methods
    # ========================================================================

    def export_plot(self):
        """Open a file dialog to export the current plot in vector or raster format."""
        # Get current method for default save name
        current_method = self.ui.optimizer_combobox.currentText()
        safe_method_name = current_method.replace(" - ", "_").replace(" ", "_").lower()
        default_name = f"microstate_optimization_{safe_method_name}.pdf"

        file_name = get_save_file_path(self, default_name, "Export Microstate Optimization Plot")
        if file_name:
            # Use the method name as an optional title
            title_text = current_method if current_method else None
            save_matplotlib_figure(
                figure=self.figure,
                canvas=self.canvas,
                file_name=file_name,
                title_text=title_text,
                title_fontsize=self._get_font_size() + 2,
                font_family=self.current_font_family,
            )

    def _save_plot(self, file_name: str):
        """Save the current plot to ``file_name`` via the export utility.

        Args:
          file_name (str): Destination path.
        """
        title_text = self.ui.optimizer_combobox.currentText() or None
        save_matplotlib_figure(
            figure=self.figure,
            canvas=self.canvas,
            file_name=file_name,
            title_text=title_text,
            title_fontsize=self._get_font_size() + 2,
            font_family=self.current_font_family,
        )

    # ========================================================================
    # Public Interface Methods
    # ========================================================================

    def get_all_results_summary(self) -> Optional[dict[str, Any]]:
        """Get summary of all computed results using modified K-means.

        Returns:
          dict[str, Any] | None: Summary dictionary or None if no results.
        """
        if not self.results_cache:
            return None

        summary = {
            "algorithm": "Modified K-means (polarity-independent)",
            "metrics": "Consolidated implementations from ClustererOptimizer",
            "results": {},
            "has_saved_results": True,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        for method_code, results in self.results_cache.items():
            method_name = self._get_method_name(method_code)
            optimal_k = results["result"].optimal_k
            threshold = results.get("threshold", None)

            if threshold is not None and method_code in ["gev"]:
                summary["results"][method_name] = {"optimal_k": optimal_k, "threshold": threshold}
            else:
                summary["results"][method_name] = optimal_k

        return summary

    def get_microstate_maps_for_optimal_k(self, method: str = "gev") -> Optional[np.ndarray]:
        """Get microstate maps for the optimal K determined by a method.

        Args:
          method (str): Method code (default 'gev').

        Returns:
          np.ndarray | None: Microstate maps or None on error/missing data.
        """
        if method in self.results_cache and self.optimizer:
            optimal_k = self.results_cache[method]["optimal_k"]
            try:
                return self.optimizer.get_microstate_maps(optimal_k)
            except Exception as e:
                print(f"Error getting microstate maps for k={optimal_k}: {str(e)}")
                return None
        return None

    def clear_saved_results(self):
        """Clear saved optimization results from both memory and configuration."""
        try:
            # Clear from memory
            self.results_cache.clear()
            self.all_methods_complete = False

            # Clear from COMET configuration
            if self.comet:
                if "optimization_results" in self.comet.config:
                    self.comet.config.remove_section("optimization_results")

                # Save the updated configuration
                if self.comet.auto_save:
                    self.comet.save_config()

            # Reset UI state
            self._disable_visualization_ui()
            self.ui.optimizer_button.setText("Run All Analyses")

            # Clear the plot
            self.figure.clear()
            self.canvas.draw()

            self.ui.statusbar.showMessage("Saved optimization results cleared")
            print("Cleared all saved optimization results")

        except Exception as e:
            print(f"Error clearing saved results: {str(e)}")
            self.ui.statusbar.showMessage(f"Error clearing results: {str(e)}")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @staticmethod
    def _get_method_code(method_name: str) -> str:
        """Convert method name to internal code.

        Args:
          method_name (str): Human-readable method name.

        Returns:
          str: Method code.
        """
        method_map = {
            "Global Explained Variance Criterion": "gev",
            "Davies-Bouldin Criterion": "db",
            "Cross Validation Criterion": "cv",
            "Krzanowski-Lai Criterion": "kl",
            "Silhouette Coefficient": "sil",
            "Dunn Index": "dunn",
            "Calinski-Harabasz Index": "ch",
            "Gap Statistic": "gap",
            "Akaike Information Criterion": "aic",
            "Bayesian Information Criterion": "bic",
        }
        return method_map.get(method_name, "gev")

    @staticmethod
    def _get_method_name(method_code: str) -> str:
        """Convert method code to display name.

        Args:
          method_code (str): Internal method code.

        Returns:
          str: Human-readable method name.
        """
        name_map = {
            "gev": "Global Explained Variance Criterion",
            "db": "Davies-Bouldin Criterion",
            "cv": "Cross Validation Criterion",
            "kl": "Krzanowski-Lai Criterion",
            "sil": "Silhouette Coefficient",
            "dunn": "Dunn Index",
            "ch": "Calinski-Harabasz Index",
            "gap": "Gap Statistic",
            "aic": "Akaike Information Criterion",
            "bic": "Bayesian Information Criterion",
        }
        return name_map.get(method_code, method_code)

    @staticmethod
    def _get_method_display_names() -> dict[str, str]:
        """Get all method display names.

        Returns:
          dict[str, str]: Mapping from code to display name.
        """
        return {
            "gev": "Global Explained Variance Criterion",
            "db": "Davies-Bouldin Criterion",
            "cv": "Cross Validation Criterion",
            "kl": "Krzanowski-Lai Criterion",
            "sil": "Silhouette Coefficient",
            "dunn": "Dunn Index",
            "ch": "Calinski-Harabasz Index",
            "gap": "Gap Statistic",
            "aic": "Akaike Information Criterion",
            "bic": "Bayesian Information Criterion",
        }

    @staticmethod
    def _get_ylabel(method_name: str) -> str:
        """Get appropriate y-axis label for a method name.

        Args:
          method_name (str): Human-readable method name.

        Returns:
          str: Y-axis label text.
        """
        labels = {
            "Global Explained Variance Criterion": "Global Explained Variance",
            "Davies-Bouldin Criterion": "Davies-Bouldin Index",
            "Cross Validation Criterion": "Cross-Validation Score",
            "Krzanowski-Lai Criterion": "Krzanowski-Lai Score",
            "Silhouette Coefficient": "Silhouette Score",
            "Dunn Index": "Dunn Index",
            "Calinski-Harabasz Index": "Calinski-Harabasz Index",
            "Gap Statistic": "Gap Statistic",
            "Akaike Information Criterion": "AIC Score",
            "Bayesian Information Criterion": "BIC Score",
        }
        return labels.get(method_name, "Score")
