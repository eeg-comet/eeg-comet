"""Optimization utilities for selecting microstate cluster counts (K).

This module implements comprehensive statistical validation techniques for 
objectively determining the optimal number of microstate clusters. All methods
use spatial correlation as the primary similarity metric to ensure polarity-
invariant topographic relationships are preserved.

Available validation methods:
- Cross-Validation (CV): Balances explanatory power against parsimony
- Global Explained Variance (GEV): Identifies elbow point for efficiency  
- Silhouette Analysis: Measures clustering consistency
- Dunn Index: Quantifies cluster compactness and separation
- Davies-Bouldin Index: Assesses cluster distinctiveness
- Calinski-Harabasz Index: Evaluates variance ratios
- Gap Statistic: Compares to random distributions
- Information Criteria (AIC/BIC): Model selection principles
- Krzanowski-Lai Criterion: Evaluates relative improvement

Example usage:
    optimizer = ClustererOptimizer(maps2use, kmin=2, kmax=10)
    
    # Single method
    result = optimizer.compute_silhouette()
    
    # Multiple methods in batch
    methods = ['gev', 'db', 'cv', 'kl', 'sil', 'dunn', 'ch', 'gap', 'aic', 'bic']
    results = optimizer.compute_methods_batch(methods)
    
    # Majority vote across all methods
    optimal_k = optimizer.find_optimal_k_majority_vote()
"""

import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
from scipy.stats import pearsonr

from clustering_utils.microstate_clusterer import MicrostateClusterer
from data_utils.data_initializer import DataInitializer

warnings.filterwarnings("ignore")


@dataclass(init=False)
class OptimizationResult:
    """Container for optimization results.
    
    Attributes:
        k_values: List of k values evaluated.
        scores: List of scores for each k value.
        optimal_k: Optimal k value selected by the method.
        method_name: Name of the optimization method.
        higher_is_better: Whether higher scores indicate better clustering.
    """

    k_values: list[int]
    scores: list[float]
    optimal_k: int
    method_name: str
    higher_is_better: bool = True

    def __init__(
        self,
        k_values: list[int],
        scores: list[float],
        optimal_k: int,
        method_name: str,
        higher_is_better: bool = True,
    ) -> None:
        """Initialize OptimizationResult.
        
        Args:
            k_values: List of k values evaluated.
            scores: List of scores for each k value.
            optimal_k: Optimal k value selected by the method.
            method_name: Name of the optimization method.
            higher_is_better: Whether higher scores indicate better clustering.
        """
        self.k_values = k_values
        self.scores = scores
        self.optimal_k = optimal_k
        self.method_name = method_name
        self.higher_is_better = higher_is_better


class ClustererOptimizer:
    """Optimized clusterer for finding the optimal number of microstate clusters.
    
    This is the single source of truth for all clustering metrics implementations.
    All methods use spatial correlation as the similarity metric to ensure
    polarity-invariant topographic relationships are preserved.
    
    **Reproducibility Note:**
    Results are deterministic by default (random_seed=42). To get different results:
    - Use set_random_seed(new_seed) to change the seed
    - Use clear_cache() to force recomputation with the same seed
    """

    def __init__(
        self,
        maps2use: np.ndarray,
        min_dist: Optional[int] = None,
        n_inits: int = 1,
        kmin: int = 2,
        kmax: int = 10,
        preprocessed_data_path: str = None,
        extension: str = ".set",
        datatype: str = "raw",
        tolerance: float = 1e-6,
        max_iter: int = 500,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        logger=None,
        clustering_results_path: str = None,
    ):
        """Initialize the clusterer optimizer.

        Args:
            maps2use: Data to cluster (n_channels x n_timepoints).
            min_dist: Minimum distance for peak detection.
            n_inits: Number of clustering initializations (set to 1 for single repeat).
            kmin: Minimum number of clusters.
            kmax: Maximum number of clusters.
            preprocessed_data_path: Path to preprocessed data.
            extension: File extension.
            datatype: Data type.
            tolerance: Convergence tolerance.
            max_iter: Maximum iterations.
            batch_size: Batch size for clustering.
            progress_callback: Callback for progress updates.
            logger: Logger instance for consistent formatting.
            clustering_results_path: Path to save clustering results and plots.
        """
        # Validate and ensure correct data format
        if maps2use.ndim != 2:
            raise ValueError(f"maps2use must be 2D array, got shape {maps2use.shape}")

        # Ensure data is in (n_channels, n_timepoints) format
        if maps2use.shape[0] > maps2use.shape[1]:
            print(
                "Warning: Data appears to be in (n_timepoints, n_channels) format. "
                "Transposing to (n_channels, n_timepoints)"
            )
            maps2use = maps2use.T

        self.maps2use = maps2use
        self.min_dist = min_dist
        self.n_inits = n_inits
        self.kmin = kmin
        self.kmax = kmax
        
        # Store full dataset for GEV calculation (loaded on demand)
        self._full_dataset = None

        # Validate k range
        if kmin < 2:
            print(f"Warning: kmin {kmin} is less than 2, setting to 2")
            self.kmin = 2
        if kmax <= kmin:
            print(
                f"Warning: kmax {kmax} is not greater than kmin {kmin}, "
                f"setting kmax to {kmin + 1}"
            )
            self.kmax = kmin + 1

        # Ensure kmax doesn't exceed the number of samples
        max_possible_k = min(self.maps2use.shape[1], 20)  # Limit to 20 for practical reasons
        if self.kmax > max_possible_k:
            print(
                f"Warning: kmax {self.kmax} exceeds maximum possible k {max_possible_k}, "
                f"setting to {max_possible_k}"
            )
            self.kmax = max_possible_k

        self.k_range = list(range(self.kmin, self.kmax + 1))
        self.preprocessed_data_path = preprocessed_data_path
        self.extension = extension
        self.datatype = datatype
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.logger = logger
        self.clustering_results_path = clustering_results_path

        # Results storage
        self.results: dict[str, OptimizationResult] = {}
        self._clustering_cache: dict[int, dict] = {}

        # Majority vote results storage (populated in find_optimal_k_majority_vote)
        self.majority_vote_results: Union[dict[str, Any], None] = None

        # Stop functionality
        self._stopped = False
        
        # Random seed for reproducibility (can be changed by user)
        self.random_seed = 42

    def stop(self):
        """Stop the optimization process."""
        self._stopped = True
    
    def clear_cache(self):
        """Clear the clustering cache to force re-computation of all k values."""
        self._clustering_cache.clear()
        self._log_message("Clustering cache cleared - next optimization will recompute all k values")
    
    def set_random_seed(self, seed: int):
        """Set the random seed for reproducible results.
        
        Args:
            seed: Random seed value. Use different seeds to get different results.
        """
        self.random_seed = seed
        self.clear_cache()  # Clear cache since seed change affects results
        self._log_message(f"Random seed set to {seed} - cache cleared for fresh results")
    
    def _get_full_dataset(self):
        """Load the full dataset for GEV calculation (lazy loading).
        
        Returns:
            np.ndarray: Full dataset for GEV calculation.
        """
        if self._full_dataset is None and self.preprocessed_data_path:
            try:
                # Load entire dataset (100% of data)
                self._full_dataset, _ = DataInitializer.generate_maps_and_peaks(
                    preprocessed_folder=self.preprocessed_data_path,
                    extension=self.extension,
                    datatype=self.datatype,
                    use_percentages=100,  # Use entire dataset
                    min_dist=None,  # Not relevant for full dataset
                )
            except Exception as e:
                print(f"Warning: Could not load full dataset for GEV calculation: {e}")
                # Fallback to clustering subset
                self._full_dataset = self.maps2use
        elif self._full_dataset is None:
            # Fallback to clustering subset if no path available
            self._full_dataset = self.maps2use
        return self._full_dataset

    def is_stopped(self):
        """Check if the optimization process has been stopped.
        
        Returns:
            bool: True if stopped, False otherwise.
        """
        return self._stopped

    def _check_stop(self):
        """Check if the process should stop and raise exception if so.
        
        Raises:
            RuntimeError: If optimization process stopped by user.
        """
        if self._stopped:
            raise RuntimeError("Optimization process stopped by user")

    def _update_progress(self, current: int, total: int, message: str):
        """Update progress if callback is provided.
        
        Args:
            current: Current progress value.
            total: Total progress value.
            message: Progress message.
        """
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def _filter_valid_scores(
        self, scores: list[float], exclude_inf: bool = False
    ) -> list[tuple[int, float]]:
        """Pair k values with scores and filter out invalid entries.

        Args:
            scores: Scores aligned with `self.k_range`.
            exclude_inf: If True, also exclude np.inf.

        Returns:
            list[tuple[int, float]]: Valid (k, score) pairs.
        """
        pairs = list(zip(self.k_range, scores))
        if exclude_inf:
            return [(k, s) for k, s in pairs if not np.isnan(s) and s != np.inf]
        return [(k, s) for k, s in pairs if not np.isnan(s)]

    def _compute_M_q(self, k: int, result: Optional[dict] = None) -> tuple[float, float]:
        """Compute W_q and M_q for a given k.

        Args:
            k: Number of clusters.
            result: Optional cached clustering result.

        Returns:
            tuple[float, float]: (W_q, M_q) values.
        """
        if result is None:
            result = self._get_clustering_result(k)
        maps = result["maps"]
        segmentation = result["segmentation"]
        W_q = self._compute_W_q(self.maps2use, segmentation, maps)
        n_channels = self.maps2use.shape[0]
        M_q = W_q * (k ** (2.0 / n_channels))
        return W_q, M_q

    def _compute_kl_scores_from_M_values(self, M_values: dict[int, float]) -> list[float]:
        """Compute KL scores given M values across k.

        Args:
            M_values: Mapping from k to M_q value.

        Returns:
            list[float]: KL scores aligned with `self.k_range`.
        """
        kl_scores: list[float] = []
        for i, k in enumerate(self.k_range):
            if i == 0 or i == len(self.k_range) - 1:
                kl_scores.append(np.nan)
                continue
            k_prev = self.k_range[i - 1]
            k_next = self.k_range[i + 1]
            M_prev = M_values.get(k_prev, np.nan)
            M_curr = M_values.get(k, np.nan)
            M_next = M_values.get(k_next, np.nan)
            if np.isnan(M_prev) or np.isnan(M_curr) or np.isnan(M_next):
                kl_scores.append(np.nan)
                continue
            d_prev = M_prev - M_curr
            d_curr = M_curr - M_next
            if d_prev < 0 or d_prev < d_curr:
                kl_scores.append(0.0)
            elif M_prev > 0:
                kl_scores.append(d_prev / M_prev)
            else:
                kl_scores.append(0.0)
        return kl_scores

    def _log_message(self, message: str, level: str = "info"):
        """Log a message with consistent formatting.

        Args:
            message: Message to log.
            level: Log level ('info', 'warning', 'error', 'success').
        """
        # Use the logger instance if available for consistent emoji formatting
        if self.logger is not None:
            if level == "error":
                self.logger.error("CLUSTERING", message)
            elif level == "warning":
                self.logger.warning("CLUSTERING", message)
            elif level == "success":
                self.logger.processing_success("CLUSTERING", message)
            else:
                self.logger.processing_info("CLUSTERING", message)
        else:
            # Fallback to terminal output with consistent formatting
            if level == "error":
                print(f"[ERROR] {message}")
            elif level == "warning":
                print(f"[WARNING] {message}")
            elif level == "success":
                print(f"[SUCCESS] {message}")
            else:
                print(f"[INFO] {message}")

    def _get_clustering_result(self, k: int, compute_all_metrics: bool = True) -> dict:
        """Get clustering result for k clusters, using cache if available.

        Args:
            k: Number of clusters.

        Returns:
            dict: Contains all metrics computed like optimizer window.
        """
        if k in self._clustering_cache:
            return self._clustering_cache[k]

        # Clustering timing start
        start_time = time.time()

        # Initialize the MicrostateClusterer for this K (same as main clustering workflow)
        clusterer = MicrostateClusterer(
            n_states=k,
            batch_size=self.batch_size,
            n_inits=1,  # We handle repetitions manually
            max_iter=self.max_iter,
            tolerance=self.tolerance,
        )

        # For optimization, we need speed over perfect clustering quality
        # Use single initialization with proper method but avoid expensive repeated operations
        best_maps = None
        best_residual = np.inf
        best_labels = None

        try:
            # Check if process should stop
            self._check_stop()
            
            # Set reproducible random seed based on k value for consistent results
            np.random.seed(self.random_seed + k)  # Different seed per k, but reproducible
            
            # Initialize cluster centers using proper method (single initialization for speed)
            initial_maps = DataInitializer.initialize_cluster_centers(
                maps2use=self.maps2use, 
                n_states=k, 
                initializer="K-Means++"  # Use K-Means++ for better initialization
            )

            # Run modified K-means (single run for optimization speed)
            maps, residual = clusterer.modified_kmeans(
                data=self.maps2use,
                initial_maps=initial_maps,
                verbose=False
            )

            # Store results
            best_maps = maps.copy()
            best_residual = residual
            
            # Calculate segmentation for the result
            activation = best_maps.dot(self.maps2use)
            best_labels = np.argmax(np.abs(activation), axis=0)

        except Exception as err:
            # Import traceback for detailed error logging
            import traceback
            
            self._log_message(
                f"❌ Clustering failed for k={k}: {str(err)}", level="error"
            )
            self._log_message(
                f"Full traceback for k={k}: {traceback.format_exc()}", level="error"
            )
            raise RuntimeError(f"Clustering failed for k={k}: {str(err)}") from err

        # Ensure we have valid results
        if best_maps is None:
            raise RuntimeError(f"No valid clustering results obtained for k={k}")

        # Calculate GEV for this result using GFP peaks (optimization dataset)
        # All optimization metrics should use GFP peaks for speed and consistency
        best_gev = clusterer.compute_gev(self.maps2use, best_maps)

        # Cache the result with basic metrics first
        self._clustering_cache[k] = {
            "maps": best_maps.copy(),
            "segmentation": best_labels,
            "gev": best_gev,
            "residual": best_residual,
            "computation_time": time.time() - start_time,
        }

        # Compute all additional metrics only if requested (for ensemble optimization)
        if compute_all_metrics:
            self._compute_all_metrics_exact(
                self._clustering_cache[k], k, best_labels, best_gev, best_residual
            )

        return self._clustering_cache[k]

    def _compute_all_metrics_exact(
        self, clustering_result, k, best_labels, best_gev, best_residual
    ):
        """Compute all metrics for a clustering result.

        Ensures consistency between auto-k selection and optimizer visualization.

        Args:
            clustering_result: Cache entry to update with computed metrics.
            k: Number of clusters.
            best_labels: Segmentation labels for each sample.
            best_gev: Global explained variance score.
            best_residual: Residual error from clustering.
        """
        # Store basic metrics
        clustering_result["gev"] = best_gev
        clustering_result["residual"] = best_residual

        # Compute Davies-Bouldin score
        if k >= 2:
            try:
                db_score = self.compute_custom_davies_bouldin(
                    data=self.maps2use, labels=best_labels, maps=clustering_result["maps"]
                )
                clustering_result["davies_bouldin"] = db_score
                self._log_message(f"k={k}: Davies-Bouldin = {db_score:.4f}")
            except Exception as e:
                self._log_message(
                    f"Error computing Davies-Bouldin for k={k}: {str(e)}", level="error"
                )
                clustering_result["davies_bouldin"] = np.nan
        else:
            clustering_result["davies_bouldin"] = np.nan

        # Compute Cross-Validation score
        try:
            cv_score = self._compute_cross_validation_criterion_vectorized(
                self.maps2use, clustering_result["maps"], best_labels
            )
            clustering_result["cv_score"] = cv_score
            self._log_message(f"k={k}: Cross-Validation = {cv_score:.4f}")
        except Exception as e:
            self._log_message(
                f"Error computing Cross-Validation for k={k}: {str(e)}", level="error"
            )
            clustering_result["cv_score"] = np.nan

        # Compute KL score
        try:
            # For KL criterion, we need to compute W_q and M_q
            W_q = self._compute_W_q(self.maps2use, best_labels, clustering_result["maps"])
            n_samples = self.maps2use.shape[1]
            M_q = W_q * (k ** (2.0 / n_samples))
            clustering_result["W_q"] = W_q
            clustering_result["M_q"] = M_q

            # KL score will be computed later when we have M values for adjacent k values
            clustering_result["kl_score"] = np.nan  # Will be updated in KL method
        except Exception as e:
            self._log_message(f"Error computing KL components for k={k}: {str(e)}", level="error")
            clustering_result["W_q"] = np.nan
            clustering_result["M_q"] = np.nan
            clustering_result["kl_score"] = np.nan

        # Compute Silhouette score
        if k >= 2:
            try:
                sil_score = self.silhouette_coefficient_correlation(
                    data=self.maps2use, labels=best_labels
                )
                clustering_result["silhouette_score"] = sil_score
                self._log_message(f"k={k}: Silhouette = {sil_score:.4f}")
            except Exception as e:
                self._log_message(
                    f"Error computing Silhouette for k={k}: {str(e)}", level="error"
                )
                clustering_result["silhouette_score"] = np.nan
        else:
            clustering_result["silhouette_score"] = -1.0  # Worst possible score for k=1

        # Compute Dunn Index
        if k >= 2:
            try:
                dunn_score = self.compute_dunn_index(
                    data=self.maps2use, labels=best_labels, maps=clustering_result["maps"]
                )
                clustering_result["dunn_index"] = dunn_score
                self._log_message(f"k={k}: Dunn Index = {dunn_score:.4f}")
            except Exception as e:
                self._log_message(
                    f"Error computing Dunn Index for k={k}: {str(e)}", level="error"
                )
                clustering_result["dunn_index"] = np.nan
        else:
            clustering_result["dunn_index"] = 0.0

        # Compute Calinski-Harabasz Index
        if k >= 2:
            try:
                ch_score = self.compute_calinski_harabasz_index(
                    data=self.maps2use, labels=best_labels, maps=clustering_result["maps"]
                )
                clustering_result["calinski_harabasz_index"] = ch_score
                self._log_message(f"k={k}: Calinski-Harabasz = {ch_score:.4f}")
            except Exception as e:
                self._log_message(
                    f"Error computing Calinski-Harabasz for k={k}: {str(e)}", level="error"
                )
                clustering_result["calinski_harabasz_index"] = np.nan
        else:
            clustering_result["calinski_harabasz_index"] = 0.0

        # Compute Gap Statistic
        try:
            gap_score, gap_std = self.compute_gap_statistic(
                data=self.maps2use, labels=best_labels, maps=clustering_result["maps"]
            )
            clustering_result["gap_statistic"] = gap_score
            clustering_result["gap_std"] = gap_std
            self._log_message(f"k={k}: Gap Statistic = {gap_score:.4f}")
        except Exception as e:
            self._log_message(
                f"Error computing Gap Statistic for k={k}: {str(e)}", level="error"
            )
            clustering_result["gap_statistic"] = np.nan
            clustering_result["gap_std"] = np.nan

        # Compute AIC
        try:
            aic_score = self.compute_information_criteria(
                data=self.maps2use, labels=best_labels, maps=clustering_result["maps"], criterion='AIC'
            )
            clustering_result["aic_score"] = aic_score
            self._log_message(f"k={k}: AIC = {aic_score:.4f}")
        except Exception as e:
            self._log_message(
                f"Error computing AIC for k={k}: {str(e)}", level="error"
            )
            clustering_result["aic_score"] = np.nan

        # Compute BIC
        try:
            bic_score = self.compute_information_criteria(
                data=self.maps2use, labels=best_labels, maps=clustering_result["maps"], criterion='BIC'
            )
            clustering_result["bic_score"] = bic_score
            self._log_message(f"k={k}: BIC = {bic_score:.4f}")
        except Exception as e:
            self._log_message(
                f"Error computing BIC for k={k}: {str(e)}", level="error"
            )
            clustering_result["bic_score"] = np.nan

    # ============================================================================
    # PLOT GENERATION METHODS
    # ============================================================================
    
    def _generate_gev_plot(self, k_values: list[int], gev_scores: list[float]):
        """Generate and save GEV optimization plot.
        
        Args:
            k_values: List of k values evaluated.
            gev_scores: List of GEV scores for each k.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Filter out NaN values
            valid_pairs = [(k, score) for k, score in zip(k_values, gev_scores) if not np.isnan(score)]
            if not valid_pairs:
                self._log_message("No valid GEV scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Global Explained Variance (GEV)', fontsize=12)
            plt.title('GEV Optimization Results', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            # Add score annotations on points
            for k, score in valid_pairs:
                plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            # Find and highlight the elbow point
            if len(valid_pairs) > 1:
                elbow_k = self._find_elbow_point(k_values, gev_scores, higher_is_better=True)
                elbow_score = None
                for k, score in valid_pairs:
                    if k == elbow_k:
                        elbow_score = score
                        break
                if elbow_score is not None:
                    plt.plot(elbow_k, elbow_score, 'gs', markersize=12, markerfacecolor='lightgreen', 
                            markeredgecolor='green', markeredgewidth=2, label=f'Elbow Point (k={elbow_k})')
                    plt.legend()
            
            # Save the plot
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                # Fallback to current directory
                output_dir = '.'
                
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'global_explained_variance_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("GEV optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping plot generation", level="warning")
        except Exception as e:
            self._log_message(f"Error generating GEV plot: {str(e)}", level="error")

    def _generate_aic_plot(self, k_values: list[int], aic_scores: list[float]):
        """Generate and save AIC optimization plot.
        
        Args:
            k_values: List of k values evaluated.
            aic_scores: List of AIC scores for each k.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Filter out NaN values
            valid_pairs = [(k, score) for k, score in zip(k_values, aic_scores) if not np.isnan(score)]
            if not valid_pairs:
                self._log_message("No valid AIC scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Akaike Information Criterion (AIC)', fontsize=12)
            plt.title('AIC Optimization Results (Lower is Better)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            # Add score annotations on points
            for k, score in valid_pairs:
                plt.annotate(f'{score:.0f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            # Find and highlight the elbow point
            if len(valid_pairs) > 1:
                elbow_k = self._find_elbow_point(k_values, aic_scores, higher_is_better=False)
                elbow_score = None
                for k, score in valid_pairs:
                    if k == elbow_k:
                        elbow_score = score
                        break
                if elbow_score is not None:
                    plt.plot(elbow_k, elbow_score, 'gs', markersize=12, markerfacecolor='lightgreen', 
                            markeredgecolor='green', markeredgewidth=2, label=f'Elbow Point (k={elbow_k})')
                    plt.legend()
            
            # Save the plot
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                # Fallback to current directory
                output_dir = '.'
                
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'akaike_information_criterion_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("AIC optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping AIC plot generation", level="warning")
        except Exception as e:
            self._log_message(f"Error generating AIC plot: {str(e)}", level="error")

    def _generate_bic_plot(self, k_values: list[int], bic_scores: list[float]):
        """Generate and save BIC optimization plot.
        
        Args:
            k_values: List of k values evaluated.
            bic_scores: List of BIC scores for each k.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Filter out NaN values
            valid_pairs = [(k, score) for k, score in zip(k_values, bic_scores) if not np.isnan(score)]
            if not valid_pairs:
                self._log_message("No valid BIC scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'mo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Bayesian Information Criterion (BIC)', fontsize=12)
            plt.title('BIC Optimization Results (Lower is Better)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            # Add score annotations on points
            for k, score in valid_pairs:
                plt.annotate(f'{score:.0f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            # Find and highlight the elbow point
            if len(valid_pairs) > 1:
                elbow_k = self._find_elbow_point(k_values, bic_scores, higher_is_better=False)
                elbow_score = None
                for k, score in valid_pairs:
                    if k == elbow_k:
                        elbow_score = score
                        break
                if elbow_score is not None:
                    plt.plot(elbow_k, elbow_score, 'gs', markersize=12, markerfacecolor='lightgreen', 
                            markeredgecolor='green', markeredgewidth=2, label=f'Elbow Point (k={elbow_k})')
                    plt.legend()
            
            # Save the plot
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                # Fallback to current directory
                output_dir = '.'
                
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'bayesian_information_criterion_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("BIC optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping BIC plot generation", level="warning")
        except Exception as e:
            self._log_message(f"Error generating BIC plot: {str(e)}", level="error")

    def _generate_davies_bouldin_plot(self, k_values: list[int], db_scores: list[float]):
        """Generate and save Davies-Bouldin optimization plot."""
        try:
            import matplotlib.pyplot as plt
            
            valid_pairs = [(k, score) for k, score in zip(k_values, db_scores) if not np.isnan(score) and score != np.inf]
            if not valid_pairs:
                self._log_message("No valid Davies-Bouldin scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'co-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Davies-Bouldin Index', fontsize=12)
            plt.title('Davies-Bouldin Index Results (Lower is Better)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            for k, score in valid_pairs:
                plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'davies_bouldin_index_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("Davies-Bouldin optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping Davies-Bouldin plot", level="warning")
        except Exception as e:
            self._log_message(f"Error generating Davies-Bouldin plot: {str(e)}", level="error")

    def _generate_cross_validation_plot(self, k_values: list[int], cv_scores: list[float]):
        """Generate and save Cross-Validation optimization plot."""
        try:
            import matplotlib.pyplot as plt
            
            valid_pairs = [(k, score) for k, score in zip(k_values, cv_scores) if not np.isnan(score)]
            if not valid_pairs:
                self._log_message("No valid CV scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Cross-Validation Score', fontsize=12)
            plt.title('Cross-Validation Results (Lower is Better)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            for k, score in valid_pairs:
                plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            # Find and highlight the elbow point
            if len(valid_pairs) > 1:
                elbow_k = self._find_elbow_point(k_values, cv_scores, higher_is_better=False)
                elbow_score = None
                for k, score in valid_pairs:
                    if k == elbow_k:
                        elbow_score = score
                        break
                if elbow_score is not None:
                    plt.plot(elbow_k, elbow_score, 'gs', markersize=12, markerfacecolor='lightgreen', 
                            markeredgecolor='green', markeredgewidth=2, label=f'Elbow Point (k={elbow_k})')
                    plt.legend()
            
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'cross_validation_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("Cross-Validation optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping Cross-Validation plot", level="warning")
        except Exception as e:
            self._log_message(f"Error generating Cross-Validation plot: {str(e)}", level="error")

    def _generate_krzanowski_lai_plot(self, k_values: list[int], kl_scores: list[float]):
        """Generate and save Krzanowski-Lai optimization plot."""
        try:
            import matplotlib.pyplot as plt
            
            valid_pairs = [(k, score) for k, score in zip(k_values, kl_scores) if not np.isnan(score) and score > 0]
            if not valid_pairs:
                self._log_message("No valid KL scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'yo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Krzanowski-Lai Criterion', fontsize=12)
            plt.title('Krzanowski-Lai Criterion Results (Higher is Better)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            for k, score in valid_pairs:
                plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'krzanowski_lai_criterion_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("Krzanowski-Lai optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping Krzanowski-Lai plot", level="warning")
        except Exception as e:
            self._log_message(f"Error generating Krzanowski-Lai plot: {str(e)}", level="error")

    def _generate_silhouette_plot(self, k_values: list[int], sil_scores: list[float]):
        """Generate and save Silhouette optimization plot."""
        try:
            import matplotlib.pyplot as plt
            
            valid_pairs = [(k, score) for k, score in zip(k_values, sil_scores) if not np.isnan(score)]
            if not valid_pairs:
                self._log_message("No valid Silhouette scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'o-', color='purple', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Silhouette Coefficient', fontsize=12)
            plt.title('Silhouette Coefficient Results (Higher is Better)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            for k, score in valid_pairs:
                plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'silhouette_coefficient_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("Silhouette optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping Silhouette plot", level="warning")
        except Exception as e:
            self._log_message(f"Error generating Silhouette plot: {str(e)}", level="error")

    def _generate_dunn_index_plot(self, k_values: list[int], dunn_scores: list[float]):
        """Generate and save Dunn Index optimization plot."""
        try:
            import matplotlib.pyplot as plt
            
            valid_pairs = [(k, score) for k, score in zip(k_values, dunn_scores) if not np.isnan(score)]
            if not valid_pairs:
                self._log_message("No valid Dunn Index scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'o-', color='orange', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Dunn Index', fontsize=12)
            plt.title('Dunn Index Results (Higher is Better)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            for k, score in valid_pairs:
                plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'dunn_index_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("Dunn Index optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping Dunn Index plot", level="warning")
        except Exception as e:
            self._log_message(f"Error generating Dunn Index plot: {str(e)}", level="error")

    def _generate_calinski_harabasz_plot(self, k_values: list[int], ch_scores: list[float]):
        """Generate and save Calinski-Harabasz optimization plot."""
        try:
            import matplotlib.pyplot as plt
            
            valid_pairs = [(k, score) for k, score in zip(k_values, ch_scores) if not np.isnan(score)]
            if not valid_pairs:
                self._log_message("No valid Calinski-Harabasz scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'o-', color='brown', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Calinski-Harabasz Index', fontsize=12)
            plt.title('Calinski-Harabasz Index Results (Higher is Better)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            for k, score in valid_pairs:
                plt.annotate(f'{score:.0f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            # Find and highlight the elbow point
            if len(valid_pairs) > 1:
                elbow_k = self._find_elbow_point(k_values, ch_scores, higher_is_better=True)
                elbow_score = None
                for k, score in valid_pairs:
                    if k == elbow_k:
                        elbow_score = score
                        break
                if elbow_score is not None:
                    plt.plot(elbow_k, elbow_score, 'gs', markersize=12, markerfacecolor='lightgreen', 
                            markeredgecolor='green', markeredgewidth=2, label=f'Elbow Point (k={elbow_k})')
                    plt.legend()
            
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'calinski_harabasz_index_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("Calinski-Harabasz optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping Calinski-Harabasz plot", level="warning")
        except Exception as e:
            self._log_message(f"Error generating Calinski-Harabasz plot: {str(e)}", level="error")

    def _generate_gap_statistic_plot(self, k_values: list[int], gap_scores: list[float]):
        """Generate and save Gap Statistic optimization plot."""
        try:
            import matplotlib.pyplot as plt
            
            valid_pairs = [(k, score) for k, score in zip(k_values, gap_scores) if not np.isnan(score)]
            if not valid_pairs:
                self._log_message("No valid Gap Statistic scores to plot", level="warning")
                return
            
            valid_k, valid_scores = zip(*valid_pairs)
            
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k, valid_scores, 'o-', color='pink', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (K)', fontsize=12)
            plt.ylabel('Gap Statistic', fontsize=12)
            plt.title('Gap Statistic Results (Higher is Better)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(valid_k)
            
            for k, score in valid_pairs:
                plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            if self.clustering_results_path:
                output_dir = self.clustering_results_path
            else:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, 'gap_statistic_optimization_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_message("Gap Statistic optimization plot saved successfully!", level="success")
            
        except ImportError:
            self._log_message("Matplotlib not available - skipping Gap Statistic plot", level="warning")
        except Exception as e:
            self._log_message(f"Error generating Gap Statistic plot: {str(e)}", level="error")

    # ============================================================================
    # UTILITY METHODS FOR SPATIAL CORRELATION
    # ============================================================================
    
    @staticmethod
    def _spatial_correlation(X, Y):
        """Calculate spatial correlation between topographic maps.
        
        Args:
            X: First set of maps (n_maps, n_channels) or (n_channels,).
            Y: Second set of maps (n_maps, n_channels) or (n_channels,).
            
        Returns:
            np.ndarray: Correlation matrix of shape (X.shape[0], Y.shape[0]).
        """
        # Handle both 1D and 2D inputs
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        # Calculate correlations
        correlations = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                corr_coeff = np.corrcoef(X[i], Y[j])[0, 1]
                correlations[i, j] = np.abs(corr_coeff) if not np.isnan(corr_coeff) else 0.0
        
        return correlations

    # ============================================================================
    # MAIN METRIC COMPUTATION METHODS - SINGLE SOURCE OF TRUTH
    # ============================================================================

    @staticmethod
    def compute_custom_davies_bouldin(
        data: np.ndarray, labels: np.ndarray, maps: np.ndarray
    ) -> float:
        """Compute Davies-Bouldin Index for polarity-invariant microstate clustering.

        The Davies-Bouldin Index is defined as:
        DBI = (1/K) * Σ(i=1 to K) max_j≠i[(S_i + S_j) / d_ij]

        Where:
        - K is the number of clusters
        - S_i is the average within-cluster distance for cluster i
        - d_ij is the distance between cluster centers i and j

        Lower values indicate better clustering (minimum value is 0).
        This implementation uses correlation-based distances for polarity invariance.

        Args:
            data: Data matrix of shape (n_channels, n_samples).
            labels: Cluster labels for each sample.
            maps: Cluster centers of shape (n_clusters, n_channels).

        Returns:
            float: Davies-Bouldin Index (lower is better).
        """
        # data.shape is (n_channels, n_samples); explicit sizes not needed below
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Handle edge case: only one cluster
        if n_clusters == 1:
            return 0.0

        # Normalize data and maps for correlation computation
        data_norm = data / (np.linalg.norm(data, axis=0, keepdims=True) + 1e-10)
        maps_norm = maps / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-10)

        # Step 1: Calculate within-cluster scatter (S_i) for each cluster
        within_cluster_scatter = np.zeros(n_clusters)

        for i, label in enumerate(unique_labels):
            cluster_mask = labels == label
            n_points = np.sum(cluster_mask)

            if n_points > 0:
                # Get normalized cluster data
                cluster_data = data_norm[:, cluster_mask]
                cluster_center = maps_norm[i : i + 1]

                # Compute scatter as average correlation distance from center
                # Correlation: center @ data
                correlations = np.abs(np.dot(cluster_center, cluster_data)).flatten()
                distances = 1 - correlations

                # S_i = average distance within cluster
                within_cluster_scatter[i] = np.mean(distances)
            else:
                within_cluster_scatter[i] = 0.0

        # Step 2: Calculate between-cluster distances (d_ij) using correlation
        between_cluster_dist = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # Correlation-based distance between normalized cluster centers
                correlation = np.abs(np.dot(maps_norm[i], maps_norm[j]))
                distance = 1 - correlation
                between_cluster_dist[i, j] = distance
                between_cluster_dist[j, i] = distance  # Symmetric matrix

        # Step 3: For each cluster, find the maximum similarity ratio
        db_scores = np.zeros(n_clusters)

        for i in range(n_clusters):
            max_ratio = 0.0

            for j in range(n_clusters):
                if i != j:
                    # Calculate similarity ratio: (S_i + S_j) / d_ij
                    if between_cluster_dist[i, j] > 1e-10:
                        ratio = (
                            within_cluster_scatter[i] + within_cluster_scatter[j]
                        ) / between_cluster_dist[i, j]
                        max_ratio = max(max_ratio, ratio)
                    else:
                        # If centers are nearly identical (shouldn't happen in practice)
                        # Assign a high penalty
                        max_ratio = max(max_ratio, 10.0)

            db_scores[i] = max_ratio

        # Step 4: Return average of maximum ratios
        return np.mean(db_scores)

    @staticmethod
    def _compute_cross_validation_criterion_vectorized(data, maps, segmentation):
        """Vectorized implementation of the Cross-Validation (CV) criterion.

        Implements the exact mathematical formula:
        CV = σ̂²_μ ⋅ ((n - 1) / (n - 1 - q))²
        where σ̂²_μ = [ Σ_{t=1}^{tmax} (||u(t)||² - (T_t ⋅ u(t))²) ] / [ tmax ⋅ (n - 1) ].

        Args:
            data: EEG data (n_channels, n_timepoints); this is u(t).
            maps: Template maps (n_clusters, n_channels); these are T_k.
            segmentation: Cluster assignments per timepoint; selects T_t.

        Returns:
            float: Cross-validation criterion score.
        """
        n_channels, n_timepoints = data.shape  # n = n_channels, tmax = n_timepoints
        n_clusters = maps.shape[0]  # q = n_clusters

        # Validate segmentation indices before indexing
        if np.any(segmentation >= n_clusters) or np.any(segmentation < 0):
            # Clip indices to valid range
            segmentation = np.clip(segmentation, 0, n_clusters - 1)

        # Compute ||u(t)||² for all timepoints
        u_norms_squared = np.sum(data**2, axis=0)  # ||u(t)||² for each t

        # Get the assigned template T_t for each timepoint t
        # T_t is the template map corresponding to the cluster assigned to timepoint t
        assigned_templates = maps[segmentation]  # Shape: (n_timepoints, n_channels)

        # Compute (T_t ⋅ u(t))² for each timepoint
        # This is the squared dot product between the assigned template and the data
        template_data_products = np.sum(assigned_templates * data.T, axis=1)  # T_t ⋅ u(t)
        template_data_products_squared = template_data_products**2  # (T_t ⋅ u(t))²

        # Compute σ̂²_μ = [ Σ_{t=1}^{tmax} (||u(t)||² - (T_t ⋅ u(t))²) ] / [ tmax ⋅ (n - 1) ]
        residuals = u_norms_squared - template_data_products_squared  # (||u(t)||² - (T_t ⋅ u(t))²)
        sigma_mu_squared = np.sum(residuals) / (n_timepoints * (n_channels - 1))

        # Compute CV = σ̂²_μ ⋅ ((n - 1) / (n - 1 - q))²
        if n_channels - 1 - n_clusters <= 0:
            cv_score = 0.0
        else:
            factor = ((n_channels - 1) / (n_channels - 1 - n_clusters)) ** 2
            cv_score = sigma_mu_squared * factor

        return cv_score

    @staticmethod
    def silhouette_coefficient_correlation(data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette coefficient using absolute correlation coefficient as similarity measure.
        
        The silhouette coefficient is a measure of how similar an object is to its own 
        cluster compared to other clusters. It ranges from -1 to 1, where higher values 
        indicate better clustering. This implementation uses correlation-based distances 
        for polarity-invariant microstate clustering.
        
        The silhouette coefficient for a point i is defined as:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
        
        Where:
        - a(i): mean distance from point i to all other points in the same cluster
        - b(i): minimum mean distance from point i to points in any other cluster
        - Distance is computed as 1 - |correlation coefficient|
        
        Args:
            data: The input data where each column is a time point/sample 
                with shape (n_channels, n_samples).
            labels: Cluster labels for each data point with shape (n_samples,).
        
        Returns:
            float: Average silhouette coefficient across all data points. Higher values 
                indicate better clustering quality.
        
        Raises:
            ValueError: If data and labels have mismatched dimensions.
        """
        data = np.array(data)
        labels = np.array(labels)
        
        # Ensure data is in correct format (n_channels, n_samples)
        if data.shape[1] != labels.shape[0]:
            raise ValueError(
                f"Data samples ({data.shape[1]}) must match labels length ({labels.shape[0]})"
            )
        
        n_samples = data.shape[1]
        
        # Check if we have valid clustering
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters <= 1:
            return 0.0  # Silhouette coefficient is undefined for single cluster
        
        # Pre-compute correlation matrix for efficiency (O(n²) but vectorized)
        # This is much faster than individual pearsonr calls
        correlation_matrix = ClustererOptimizer._spatial_correlation(data.T, data.T)
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        silhouette_scores = []
        
        for i in range(n_samples):
            current_label = labels[i]
            
            # Calculate a(i): average distance within same cluster
            same_cluster_mask = (labels == current_label) & (np.arange(n_samples) != i)
            same_cluster_indices = np.where(same_cluster_mask)[0]
            
            if len(same_cluster_indices) == 0:
                # If point is alone in cluster, a(i) = 0
                a_i = 0.0
            else:
                # Use pre-computed distances
                a_i = np.mean(distance_matrix[i, same_cluster_indices])
            
            # Calculate b(i): minimum average distance to other clusters
            b_i = np.inf
            
            for other_label in unique_labels:
                if other_label == current_label:
                    continue
                    
                other_cluster_mask = (labels == other_label)
                other_cluster_indices = np.where(other_cluster_mask)[0]
                
                if len(other_cluster_indices) > 0:
                    # Use pre-computed distances
                    avg_distance_to_cluster = np.mean(distance_matrix[i, other_cluster_indices])
                    b_i = min(b_i, avg_distance_to_cluster)
            
            # Calculate silhouette coefficient for point i
            if max(a_i, b_i) == 0:
                s_i = 0.0  # Handle division by zero
            else:
                s_i = (b_i - a_i) / max(a_i, b_i)
            
            silhouette_scores.append(s_i)
        
        return np.mean(silhouette_scores)

    @staticmethod
    def compute_dunn_index(data: np.ndarray, labels: np.ndarray, maps: np.ndarray) -> float:
        """Compute Dunn Index using spatial correlation distances.
        
        The Dunn Index is the ratio of minimum inter-cluster to maximum intra-cluster distance.
        Higher values indicate better clustering (compact clusters with clear separation).
        
        Args:
            data: Data matrix of shape (n_channels, n_samples).
            labels: Cluster labels for each sample.
            maps: Cluster centers of shape (n_clusters, n_channels).
            
        Returns:
            float: Dunn Index (higher is better).
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return 0.0
        
        # Calculate minimum inter-cluster distance (between actual data points)
        min_inter = np.inf
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # Get data points from each cluster
                cluster_i_mask = labels == unique_labels[i]
                cluster_j_mask = labels == unique_labels[j]
                
                cluster_i_data = data[:, cluster_i_mask]
                cluster_j_data = data[:, cluster_j_mask]
                
                # Calculate minimum distance between any two points from different clusters
                if cluster_i_data.shape[1] > 0 and cluster_j_data.shape[1] > 0:
                    # Use correlation-based distance
                    correlations = ClustererOptimizer._spatial_correlation(
                        cluster_i_data.T, cluster_j_data.T
                    )
                    distances = 1 - correlations
                    min_inter = min(min_inter, np.min(distances))
        
        # Calculate maximum intra-cluster distance
        max_intra = 0.0
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > 1:
                cluster_data = data[:, cluster_mask]
                
                # For small clusters, compute full correlation matrix
                if cluster_size <= 50:  # Increased threshold for better accuracy
                    correlations = ClustererOptimizer._spatial_correlation(cluster_data.T, cluster_data.T)
                    np.fill_diagonal(correlations, 1.0)
                    distances = 1 - correlations
                    max_intra = max(max_intra, np.max(distances))
                else:
                    # For large clusters, use sampling but with more samples
                    n_samples = min(50, cluster_size)  # Increased sample size
                    sample_indices = np.random.choice(cluster_size, n_samples, replace=False)
                    sample_data = cluster_data[:, sample_indices]
                    
                    correlations = ClustererOptimizer._spatial_correlation(sample_data.T, sample_data.T)
                    np.fill_diagonal(correlations, 1.0)
                    distances = 1 - correlations
                    max_intra = max(max_intra, np.max(distances))
        
        # Dunn index
        return min_inter / max_intra if max_intra > 0 else 0.0

    @staticmethod
    def compute_calinski_harabasz_index(data: np.ndarray, labels: np.ndarray, maps: np.ndarray) -> float:
        """Compute Calinski-Harabasz Index using spatial correlation.
        
        The Calinski-Harabasz Index is the ratio of between-cluster to within-cluster variance.
        Higher values indicate better clustering.
        
        Args:
            data: Data matrix of shape (n_channels, n_samples).
            labels: Cluster labels for each sample.
            maps: Cluster centers of shape (n_clusters, n_channels).
            
        Returns:
            float: Calinski-Harabasz Index (higher is better).
        """
        n_samples = data.shape[1]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters == 1:
            return 0.0
        
        # Global centroid
        global_centroid = np.mean(data, axis=1)
        
        # Between-cluster variance
        between_var = 0.0
        for i, label in enumerate(unique_labels):
            n_k = np.sum(labels == label)
            if n_k > 0:
                corr = ClustererOptimizer._spatial_correlation(
                    maps[i].reshape(1, -1), global_centroid.reshape(1, -1)
                )[0, 0]
                dist = 1 - corr
                between_var += n_k * (dist ** 2)
        
        # Within-cluster variance
        within_var = 0.0
        for i, label in enumerate(unique_labels):
            cluster_mask = labels == label
            cluster_data = data[:, cluster_mask]
            if cluster_data.shape[1] > 0:
                correlations = ClustererOptimizer._spatial_correlation(
                    cluster_data.T, maps[i].reshape(1, -1)
                )
                distances = 1 - correlations.flatten()
                within_var += np.sum(distances ** 2)
        
        # CH index
        if within_var > 0:
            ch_index = (between_var / (n_clusters - 1)) / (within_var / (n_samples - n_clusters))
        else:
            ch_index = 0.0
        
        return ch_index

    @staticmethod
    def compute_gap_statistic(data: np.ndarray, labels: np.ndarray, maps: np.ndarray, n_refs: int = 5) -> tuple[float, float]:
        """Compute Gap Statistic using spatial correlation.
        
        The Gap Statistic compares clustering quality to random reference distribution.
        Maximum gap indicates optimal number of clusters.
        
        Args:
            data: Data matrix of shape (n_channels, n_samples).
            labels: Cluster labels for each sample.
            maps: Cluster centers of shape (n_clusters, n_channels).
            n_refs: Number of reference datasets to generate.
            
        Returns:
            tuple[float, float]: (gap_statistic, gap_std).
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Calculate within-cluster dispersion for actual data
        W_k = 0.0
        for i, label in enumerate(unique_labels):
            cluster_mask = labels == label
            cluster_data = data[:, cluster_mask]
            if cluster_data.shape[1] > 1:
                correlations = ClustererOptimizer._spatial_correlation(cluster_data.T, cluster_data.T)
                distances = 1 - correlations
                # Remove diagonal (self-correlations)
                np.fill_diagonal(distances, 0.0)
                W_k += np.sum(distances) / (2 * cluster_data.shape[1])
        
        # Generate reference datasets and calculate expected dispersion
        W_k_refs = []
        for _ in range(n_refs):
            # Create random reference data preserving topographic structure
            ref_data = np.random.randn(*data.shape)
            # Normalize to maintain similar properties to EEG data
            norms = np.linalg.norm(ref_data, axis=0, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            ref_data = ref_data / norms
            
            # Use simplified clustering for reference data (much faster)
            try:
                # Generate random initial maps
                ref_initial_maps = np.random.randn(n_clusters, data.shape[0])
                ref_initial_maps = ref_initial_maps / np.linalg.norm(ref_initial_maps, axis=1, keepdims=True)
                
                # Simple k-means without full MicrostateClusterer overhead
                # Just do a few iterations for reference data
                ref_maps = ref_initial_maps.copy()
                for _ in range(10):  # Reduced iterations
                    # Calculate segmentation
                    ref_activation = ref_maps.dot(ref_data)
                    ref_labels = np.argmax(np.abs(ref_activation), axis=0)
                    
                    # Update cluster centers
                    for k in range(n_clusters):
                        cluster_mask = ref_labels == k
                        if np.sum(cluster_mask) > 0:
                            ref_maps[k] = np.mean(ref_data[:, cluster_mask], axis=1)
                            ref_maps[k] = ref_maps[k] / np.linalg.norm(ref_maps[k])
                
                # Calculate final segmentation
                ref_activation = ref_maps.dot(ref_data)
                ref_labels = np.argmax(np.abs(ref_activation), axis=0)
                
                W_k_ref = 0.0
                for k in range(n_clusters):
                    cluster_mask = ref_labels == k
                    if np.sum(cluster_mask) > 1:
                        cluster_data = ref_data[:, cluster_mask]
                        # Use sampling for large clusters
                        if cluster_data.shape[1] > 20:
                            sample_indices = np.random.choice(cluster_data.shape[1], 20, replace=False)
                            cluster_data = cluster_data[:, sample_indices]
                        
                        correlations = ClustererOptimizer._spatial_correlation(cluster_data.T, cluster_data.T)
                        distances = 1 - correlations
                        np.fill_diagonal(distances, 0.0)
                        W_k_ref += np.sum(distances) / (2 * cluster_data.shape[1])
                W_k_refs.append(W_k_ref)
            except Exception:
                # If clustering fails, use a default value
                W_k_refs.append(W_k)
        
        # Calculate gap statistic
        if W_k > 0 and W_k_refs:
            gap = np.mean(np.log(W_k_refs)) - np.log(W_k)
            gap_std = np.std(np.log(W_k_refs))
        else:
            gap = 0.0
            gap_std = 0.0
        
        return gap, gap_std

    @staticmethod
    def compute_information_criteria(data: np.ndarray, labels: np.ndarray, maps: np.ndarray, criterion: str = 'AIC') -> float:
        """Compute Information Criteria (AIC/BIC) using spatial correlation.
        
        Information Criteria balance model fit with complexity.
        Lower values indicate better models.
        
        Args:
            data: Data matrix of shape (n_channels, n_samples).
            labels: Cluster labels for each sample.
            maps: Cluster centers of shape (n_clusters, n_channels).
            criterion: Either 'AIC' or 'BIC'.
            
        Returns:
            float: Information criterion value (lower is better).
        """
        n_samples = data.shape[1]
        n_features = data.shape[0]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_params = n_clusters * n_features  # Parameters for centroids
        
        # Calculate residual variance using spatial correlation
        residual_var = 0.0
        for i, label in enumerate(unique_labels):
            cluster_mask = labels == label
            cluster_data = data[:, cluster_mask]
            if cluster_data.shape[1] > 0:
                correlations = ClustererOptimizer._spatial_correlation(
                    cluster_data.T, maps[i].reshape(1, -1)
                )
                distances = 1 - correlations.flatten()
                residual_var += np.sum(distances ** 2)
        
        # Approximate log-likelihood
        if residual_var > 0:
            log_likelihood = -n_samples * np.log(residual_var / n_samples)
        else:
            log_likelihood = 0.0
        
        # Calculate criterion
        if criterion.upper() == 'AIC':
            ic_value = -2 * log_likelihood + 2 * n_params
        elif criterion.upper() == 'BIC':
            ic_value = -2 * log_likelihood + n_params * np.log(n_samples)
        else:
            raise ValueError("Criterion must be 'AIC' or 'BIC'")
        
        return ic_value

    # ============================================================================
    # OPTIMIZATION METHODS - USING CONSOLIDATED METRICS
    # ============================================================================

    def compute_elbow_gev(self) -> OptimizationResult:
        """Compute elbow method using Global Explained Variance.

        Returns:
            OptimizationResult: Result including scores and selected k.
        """
        scores = []
        total_steps = len(self.k_range)

        for i, k in enumerate(self.k_range):
            try:
                # Check if process should stop
                self._check_stop()

                self._log_message(f"Processing k={k} ({i+1}/{total_steps})")
                self._update_progress(i + 1, total_steps, f"Computing GEV for k={k}")
                
                result = self._get_clustering_result(k, compute_all_metrics=False)
                gev_score = result["gev"]
                scores.append(gev_score)
                
                self._log_message(f"k={k}: GEV = {gev_score:.4f}")

            except Exception as e:
                error_msg = f"Failed to compute GEV for k={k}: {str(e)}"
                self._log_message(error_msg, level="error")
                # Add NaN score and continue to next k
                scores.append(np.nan)
                continue

        # Generate and save GEV plot
        self._generate_gev_plot(self.k_range, scores)

        # Filter out NaN scores for optimal k selection
        valid_scores = [(self.k_range[i], score) for i, score in enumerate(scores) if not np.isnan(score)]
        if not valid_scores:
            self._log_message("No valid GEV scores found, using default k=5", level="warning")
            optimal_k = 5
        else:
            optimal_k = self._intelligent_k_selection(
                self.k_range, scores, higher_is_better=True
            )

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Global Explained Variance Criterion",
            higher_is_better=True,
        )

    def compute_davies_bouldin(self) -> OptimizationResult:
        """Compute Davies-Bouldin index using polarity-invariant implementation.

        Returns:
            OptimizationResult: Result including scores and selected k.
        """
        scores = []
        total_steps = len(self.k_range)

        for i, k in enumerate(self.k_range):
            # Check if process should stop
            self._check_stop()

            self._log_message(f"Processing k={k} ({i+1}/{total_steps})")
            self._update_progress(i + 1, total_steps, f"Computing Davies-Bouldin for k={k}")

            if k < 2:  # DB requires at least 2 clusters
                scores.append(np.inf)
                continue

            result = self._get_clustering_result(k, compute_all_metrics=False)

            # Use custom polarity-invariant Davies-Bouldin score
            score = self.compute_custom_davies_bouldin(
                data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
            )
            scores.append(score)
            
            self._log_message(f"k={k}: Davies-Bouldin = {score:.4f}")

        # Generate and save Davies-Bouldin plot
        self._generate_davies_bouldin_plot(self.k_range, scores)

        # Find optimal k using intelligent selection (lower Davies-Bouldin is better)
        valid_scores = self._filter_valid_scores(scores, exclude_inf=True)
        if not valid_scores:
            optimal_k = self.kmin
        else:
            optimal_k = self._intelligent_k_selection(
                self.k_range, scores, higher_is_better=False
            )

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Davies-Bouldin Criterion",
            higher_is_better=False,
        )

    def compute_cross_validation(self) -> OptimizationResult:
        """Compute the classical microstate cross-validation (CV) criterion.

        Based on Pascual-Marqui et al. (1995) without data splitting. Lower values
        indicate better clustering.

        Returns:
            OptimizationResult: Result including scores and selected k.
        """
        scores = []
        total_steps = len(self.k_range)

        for idx, k in enumerate(self.k_range):
            # Check stop flag
            self._check_stop()
            self._log_message(f"Processing k={k} ({idx+1}/{total_steps})")
            self._update_progress(idx + 1, total_steps, f"Computing CV for k={k}")

            # Get clustering result and compute CV score directly
            result = self._get_clustering_result(k, compute_all_metrics=False)
            cv_score = self._compute_cross_validation_criterion_vectorized(
                data=self.maps2use, maps=result["maps"], segmentation=result["segmentation"]
            )
            scores.append(cv_score)
            
            self._log_message(f"k={k}: Cross-Validation = {cv_score:.4f}")

        # Generate and save Cross-Validation plot
        self._generate_cross_validation_plot(self.k_range, scores)

        # Find optimal k using elbow method (lower CV is better)
        valid_scores = self._filter_valid_scores(scores)
        if not valid_scores:
            self._log_message("No valid CV scores found, using default k=5", level="warning")
            optimal_k = 5
        else:
            optimal_k = self._intelligent_k_selection(
                self.k_range, scores, higher_is_better=False
            )

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Cross-Validation Criterion",
            higher_is_better=False,
        )

    def compute_krzanowski_lai(self) -> OptimizationResult:
        """Compute Krzanowski-Lai criterion for finding optimal number of clusters.

        The KL criterion is defined as:
        KL_q = (d_{q-1} - d_q) / M_{q-1}
        where:
        - d_q = M_q - M_{q+1}
        - M_q = W_q * q^(2/n)
        - W_q = sum_{r=1}^q (1/(2*n_r)) * D_r
        - D_r = sum_{u,v in cluster r} ||u - v||^2

        Returns:
            OptimizationResult: Result including KL scores and selected k.
        """
        self._log_message("Starting Krzanowski-Lai criterion computation")

        # Need at least 3 k values for KL criterion (q-1, q, q+1)
        if len(self.k_range) < 3:
            self._log_message(
                "Warning: Need at least 3 k values for KL criterion, using default k=5",
                level="warning",
            )
            return OptimizationResult(
                k_values=self.k_range.copy(),
                scores=[np.nan] * len(self.k_range),
                optimal_k=5,
                method_name="Krzanowski-Lai Method",
                higher_is_better=True,
            )

        # Compute M_q values for all k
        M_values = {}
        total_steps = len(self.k_range)

        for i, k in enumerate(self.k_range):
            # Check if process should stop
            self._check_stop()

            self._update_progress(i + 1, total_steps, f"Computing KL criterion for k={k}")
            self._log_message(f"Computing KL criterion for k={k} ({i+1}/{total_steps})")

            try:
                # Compute W_q and M_q using shared helper
                W_q, M_q = self._compute_M_q(k)
                M_values[k] = M_q

            except Exception as e:
                self._log_message(
                    f"Error computing KL criterion for k={k}: {str(e)}", level="error"
                )
                M_values[k] = np.nan

        # Compute KL scores
        kl_scores = self._compute_kl_scores_from_M_values(M_values)
        
        # Log the actual KL scores used for optimization
        for i, k in enumerate(self.k_range):
            kl_score = kl_scores[i]
            if not np.isnan(kl_score):
                self._log_message(f"k={k}: KL = {kl_score:.4f}")
            else:
                self._log_message(f"k={k}: KL = N/A (boundary)")

        # Find optimal k (higher KL score is better)
        # Generate and save Krzanowski-Lai plot
        self._generate_krzanowski_lai_plot(self.k_range, kl_scores)

        valid_scores = [
            (k_val, score)
            for k_val, score in zip(self.k_range, kl_scores)
            if not np.isnan(score) and score > 0
        ]
        if valid_scores:
            optimal_k = self._intelligent_k_selection(
                self.k_range, kl_scores, higher_is_better=True
            )
        else:
            optimal_k = self.kmin
            self._log_message("No valid KL scores found, using default k", level="warning")

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=kl_scores,
            optimal_k=optimal_k,
            method_name="Krzanowski-Lai Criterion",
            higher_is_better=True,
        )

    def compute_silhouette(self) -> OptimizationResult:
        """Compute silhouette coefficient using correlation-based distances.
        
        The silhouette coefficient measures how similar a data point is to its own 
        cluster compared to other clusters. Higher values indicate better clustering.
        This implementation uses correlation-based distances for polarity-invariant 
        microstate clustering.
        
        Returns:
            OptimizationResult: Result including scores and selected k.
        """
        scores = []
        total_steps = len(self.k_range)
        
        for i, k in enumerate(self.k_range):
            # Check if process should stop
            self._check_stop()
            
            self._log_message(f"Processing k={k} ({i+1}/{total_steps})")
            self._update_progress(i + 1, total_steps, f"Computing Silhouette for k={k}")
            
            if k < 2:  # Silhouette requires at least 2 clusters
                scores.append(-1.0)  # Worst possible silhouette score
                continue
            
            result = self._get_clustering_result(k, compute_all_metrics=False)
            
            # Use correlation-based silhouette score
            score = self.silhouette_coefficient_correlation(
                data=self.maps2use, labels=result["segmentation"]
            )
            scores.append(score)
            
            self._log_message(f"k={k}: Silhouette = {score:.4f}")
        
        # Generate and save Silhouette plot
        self._generate_silhouette_plot(self.k_range, scores)
        
        # Find optimal k using intelligent selection (higher silhouette score is better)
        valid_scores = self._filter_valid_scores(scores)
        if not valid_scores:
            optimal_k = self.kmin
        else:
            optimal_k = self._intelligent_k_selection(
                self.k_range, scores, higher_is_better=True
            )
        
        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Silhouette Coefficient",
            higher_is_better=True,
        )

    def compute_dunn_index_optimization(self) -> OptimizationResult:
        """Compute Dunn Index for all k values.
        
        Returns:
            OptimizationResult: Result including scores and selected k.
        """
        scores = []
        total_steps = len(self.k_range)
        
        for i, k in enumerate(self.k_range):
            self._check_stop()
            self._log_message(f"Processing k={k} ({i+1}/{total_steps})")
            self._update_progress(i + 1, total_steps, f"Computing Dunn Index for k={k}")
            
            if k < 2:
                scores.append(0.0)
                continue
            
            result = self._get_clustering_result(k, compute_all_metrics=False)
            
            score = self.compute_dunn_index(
                data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
            )
            scores.append(score)
            
            self._log_message(f"k={k}: Dunn Index = {score:.4f}")
        
        # Generate and save Dunn Index plot
        self._generate_dunn_index_plot(self.k_range, scores)
        
        # Find optimal k using elbow method (Dunn index can be monotonic)
        valid_scores = self._filter_valid_scores(scores)
        if not valid_scores:
            self._log_message("No valid Dunn Index scores found, using default k=5", level="warning")
            optimal_k = 5
        else:
            optimal_k = self._intelligent_k_selection(
                self.k_range, scores, higher_is_better=True
            )
        
        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Dunn Index",
            higher_is_better=True,
        )

    def compute_calinski_harabasz_optimization(self) -> OptimizationResult:
        """Compute Calinski-Harabasz Index for all k values.
        
        Returns:
            OptimizationResult: Result including scores and selected k.
        """
        scores = []
        total_steps = len(self.k_range)
        
        for i, k in enumerate(self.k_range):
            self._check_stop()
            self._log_message(f"Processing k={k} ({i+1}/{total_steps})")
            self._update_progress(i + 1, total_steps, f"Computing Calinski-Harabasz for k={k}")
            
            if k < 2:
                scores.append(0.0)
                continue
            
            result = self._get_clustering_result(k, compute_all_metrics=False)
            
            score = self.compute_calinski_harabasz_index(
                data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
            )
            scores.append(score)
            
            self._log_message(f"k={k}: Calinski-Harabasz = {score:.4f}")
        
        # Generate and save Calinski-Harabasz plot
        self._generate_calinski_harabasz_plot(self.k_range, scores)
        
        # Find optimal k using elbow method (higher CH index is better)
        valid_scores = self._filter_valid_scores(scores)
        if not valid_scores:
            self._log_message("No valid Calinski-Harabasz scores found, using default k=5", level="warning")
            optimal_k = 5
        else:
            optimal_k = self._intelligent_k_selection(
                self.k_range, scores, higher_is_better=True
            )
        
        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Calinski-Harabasz Index",
            higher_is_better=True,
        )

    def compute_gap_statistic_optimization(self) -> OptimizationResult:
        """Compute Gap Statistic for all k values.
        
        Returns:
            OptimizationResult: Result including scores and selected k.
        """
        scores = []
        total_steps = len(self.k_range)
        
        for i, k in enumerate(self.k_range):
            self._check_stop()
            self._update_progress(i + 1, total_steps, f"Computing Gap Statistic for k={k}")
            
            result = self._get_clustering_result(k, compute_all_metrics=False)
            
            gap_score, _ = self.compute_gap_statistic(
                data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
            )
            scores.append(gap_score)
        
        # Generate and save Gap Statistic plot
        self._generate_gap_statistic_plot(self.k_range, scores)
        
        # Find optimal k using intelligent selection (maximum gap is better)
        valid_scores = self._filter_valid_scores(scores)
        if not valid_scores:
            optimal_k = self.kmin
        else:
            optimal_k = self._intelligent_k_selection(
                self.k_range, scores, higher_is_better=True
            )
        
        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Gap Statistic",
            higher_is_better=True,
        )

    def compute_aic_optimization(self) -> OptimizationResult:
        """Compute AIC for all k values using elbow method.
        
        Returns:
            OptimizationResult: Result including scores and selected k.
        """
        scores = []
        total_steps = len(self.k_range)
        
        for i, k in enumerate(self.k_range):
            try:
                self._check_stop()
                self._log_message(f"Processing k={k} ({i+1}/{total_steps})")
                self._update_progress(i + 1, total_steps, f"Computing AIC for k={k}")
                
                result = self._get_clustering_result(k, compute_all_metrics=False)
                
                aic_score = self.compute_information_criteria(
                    data=self.maps2use, labels=result["segmentation"], maps=result["maps"], criterion='AIC'
                )
                scores.append(aic_score)
                
                self._log_message(f"k={k}: AIC = {aic_score:.4f}")
                
            except Exception as e:
                error_msg = f"Failed to compute AIC for k={k}: {str(e)}"
                self._log_message(error_msg, level="error")
                # Add NaN score and continue to next k
                scores.append(np.nan)
                continue
        
        # Generate and save AIC plot
        self._generate_aic_plot(self.k_range, scores)
        
        # Filter out NaN scores for optimal k selection
        valid_scores = [(self.k_range[i], score) for i, score in enumerate(scores) if not np.isnan(score)]
        if not valid_scores:
            self._log_message("No valid AIC scores found, using default k=5", level="warning")
            optimal_k = 5
        else:
            optimal_k = self._intelligent_k_selection(
                self.k_range, scores, higher_is_better=False
            )
        
        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Akaike Information Criterion",
            higher_is_better=False,
        )

    def compute_bic_optimization(self) -> OptimizationResult:
        """Compute BIC for all k values using elbow method.
        
        Returns:
            OptimizationResult: Result including scores and selected k.
        """
        scores = []
        total_steps = len(self.k_range)
        
        for i, k in enumerate(self.k_range):
            try:
                self._check_stop()
                self._log_message(f"Processing k={k} ({i+1}/{total_steps})")
                self._update_progress(i + 1, total_steps, f"Computing BIC for k={k}")
                
                result = self._get_clustering_result(k, compute_all_metrics=False)
                
                bic_score = self.compute_information_criteria(
                    data=self.maps2use, labels=result["segmentation"], maps=result["maps"], criterion='BIC'
                )
                scores.append(bic_score)
                
                self._log_message(f"k={k}: BIC = {bic_score:.4f}")
                
            except Exception as e:
                error_msg = f"Failed to compute BIC for k={k}: {str(e)}"
                self._log_message(error_msg, level="error")
                # Add NaN score and continue to next k
                scores.append(np.nan)
                continue
        
        # Generate and save BIC plot
        self._generate_bic_plot(self.k_range, scores)
        
        # Filter out NaN scores for optimal k selection
        valid_scores = [(self.k_range[i], score) for i, score in enumerate(scores) if not np.isnan(score)]
        if not valid_scores:
            self._log_message("No valid BIC scores found, using default k=5", level="warning")
            optimal_k = 5
        else:
            optimal_k = self._intelligent_k_selection(
                self.k_range, scores, higher_is_better=False
            )
        
        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Bayesian Information Criterion",
            higher_is_better=False,
        )

    def compute_methods_batch(
        self, methods: list[str], parameters: Optional[dict[str, float]] = None
    ) -> dict[str, OptimizationResult]:
        """Compute multiple optimisation methods in a single pass over k.

        Avoids recomputing clustering for each method separately while preserving
        exact metric definitions.

        Args:
            methods: Method codes to compute. Supported: ['gev', 'db', 'cv', 'kl', 'sil', 'dunn', 'ch', 'gap', 'aic', 'bic'].
            parameters: Optional parameters per method (e.g., {'gev': 5.0}).

        Returns:
            dict[str, OptimizationResult]: Mapping from method code to result.
        """
        parameters = parameters or {}

        # Storage for per-k scores
        gev_scores: list[float] = []
        db_scores: list[float] = []
        cv_scores: list[float] = []
        sil_scores: list[float] = []
        dunn_scores: list[float] = []
        ch_scores: list[float] = []
        gap_scores: list[float] = []
        aic_scores: list[float] = []
        bic_scores: list[float] = []

        # For KL we need M_q values across adjacent k
        M_values: dict[int, float] = {}

        total_steps = len(self.k_range)
        for step_index, k in enumerate(self.k_range):
            self._check_stop()
            self._update_progress(step_index + 1, total_steps, f"Computing metrics for k={k}")

            try:
                result = self._get_clustering_result(k)
            except Exception as e:
                self._log_message(f"Failed to compute clustering for k={k}: {str(e)}", level="error")
                # Fill with NaN values for all metrics for this k
                gev_scores.append(np.nan)
                db_scores.append(np.nan)
                cv_scores.append(np.nan)
                sil_scores.append(np.nan)
                if 'dunn' in methods:
                    dunn_scores.append(np.nan)
                if 'ch' in methods:
                    ch_scores.append(np.nan)
                if 'gap' in methods:
                    gap_scores.append(np.nan)
                if 'aic' in methods:
                    aic_scores.append(np.nan)
                if 'bic' in methods:
                    bic_scores.append(np.nan)
                M_values[k] = np.nan
                continue  # Skip to next k value

            # Extract scores from cached result
            gev_scores.append(result.get("gev", np.nan))

            if k < 2:
                db_scores.append(np.inf)
            else:
                try:
                    db_val = self.compute_custom_davies_bouldin(
                        data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
                    )
                except Exception:
                    db_val = np.nan
                db_scores.append(db_val)

            cv_scores.append(result.get("cv_score", np.nan))

            # Compute silhouette score
            if k < 2:
                sil_scores.append(-1.0)  # Worst possible silhouette score
            else:
                try:
                    sil_val = self.silhouette_coefficient_correlation(
                        data=self.maps2use, labels=result["segmentation"]
                    )
                except Exception:
                    sil_val = np.nan
                sil_scores.append(sil_val)

            # Compute new metrics if requested
            if 'dunn' in methods:
                if k < 2:
                    dunn_scores.append(0.0)
                else:
                    try:
                        dunn_val = self.compute_dunn_index(
                            data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
                        )
                    except Exception:
                        dunn_val = np.nan
                    dunn_scores.append(dunn_val)
            
            if 'ch' in methods:
                if k < 2:
                    ch_scores.append(0.0)
                else:
                    try:
                        ch_val = self.compute_calinski_harabasz_index(
                            data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
                        )
                    except Exception:
                        ch_val = np.nan
                    ch_scores.append(ch_val)
            
            if 'gap' in methods:
                try:
                    gap_val, _ = self.compute_gap_statistic(
                        data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
                    )
                except Exception:
                    gap_val = np.nan
                gap_scores.append(gap_val)
            
            if 'aic' in methods:
                try:
                    aic_val = self.compute_information_criteria(
                        data=self.maps2use, labels=result["segmentation"], maps=result["maps"], criterion='AIC'
                    )
                except Exception:
                    aic_val = np.nan
                aic_scores.append(aic_val)
            
            if 'bic' in methods:
                try:
                    bic_val = self.compute_information_criteria(
                        data=self.maps2use, labels=result["segmentation"], maps=result["maps"], criterion='BIC'
                    )
                except Exception:
                    bic_val = np.nan
                bic_scores.append(bic_val)

            # Prepare M_q for KL using the same formulation as in compute_krzanowski_lai
            try:
                maps = result["maps"]
                segmentation = result["segmentation"]
                W_q = self._compute_W_q(self.maps2use, segmentation, maps)
                n_channels = self.maps2use.shape[0]
                M_q = W_q * (k ** (2.0 / n_channels))
                M_values[k] = M_q
            except Exception:
                M_values[k] = np.nan

        # Build KL scores from M_values
        kl_scores: list[float] = self._compute_kl_scores_from_M_values(M_values)

        # Helper to construct OptimizationResult per metric
        def build_result(method: str) -> OptimizationResult:
            if method == "gev":
                valid_scores = self._filter_valid_scores(gev_scores)
                if not valid_scores:
                    self._log_message("No valid GEV scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, gev_scores, higher_is_better=True
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=gev_scores,
                    optimal_k=optimal_k,
                    method_name="Global Explained Variance Criterion",
                    higher_is_better=True,
                )
            if method == "db":
                valid_scores = [
                    (k_val, score)
                    for k_val, score in zip(self.k_range, db_scores)
                    if score != np.inf and not np.isnan(score)
                ]
                if not valid_scores:
                    self._log_message("No valid Davies-Bouldin scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, db_scores, higher_is_better=False
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=db_scores,
                    optimal_k=optimal_k,
                    method_name="Davies-Bouldin Criterion",
                    higher_is_better=False,
                )
            if method == "cv":
                valid_scores = self._filter_valid_scores(cv_scores)
                if not valid_scores:
                    self._log_message("No valid Cross-Validation scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, cv_scores, higher_is_better=False
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=cv_scores,
                    optimal_k=optimal_k,
                    method_name="Cross-Validation Criterion",
                    higher_is_better=False,
                )
            if method == "kl":
                valid_scores = [
                    (k_val, score)
                    for k_val, score in zip(self.k_range, kl_scores)
                    if not np.isnan(score) and score > 0
                ]
                if not valid_scores:
                    self._log_message("No valid Krzanowski-Lai scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, kl_scores, higher_is_better=True
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=kl_scores,
                    optimal_k=optimal_k,
                    method_name="Krzanowski-Lai Criterion",
                    higher_is_better=True,
                )
            if method == "sil":
                valid_scores = self._filter_valid_scores(sil_scores)
                if not valid_scores:
                    self._log_message("No valid Silhouette scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, sil_scores, higher_is_better=True
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=sil_scores,
                    optimal_k=optimal_k,
                    method_name="Silhouette Coefficient",
                    higher_is_better=True,
                )
            if method == "dunn":
                valid_scores = self._filter_valid_scores(dunn_scores)
                if not valid_scores:
                    self._log_message("No valid Dunn Index scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, dunn_scores, higher_is_better=True
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=dunn_scores,
                    optimal_k=optimal_k,
                    method_name="Dunn Index",
                    higher_is_better=True,
                )
            if method == "ch":
                valid_scores = self._filter_valid_scores(ch_scores)
                if not valid_scores:
                    self._log_message("No valid Calinski-Harabasz scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, ch_scores, higher_is_better=True
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=ch_scores,
                    optimal_k=optimal_k,
                    method_name="Calinski-Harabasz Index",
                    higher_is_better=True,
                )
            if method == "gap":
                valid_scores = self._filter_valid_scores(gap_scores)
                if not valid_scores:
                    self._log_message("No valid Gap Statistic scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, gap_scores, higher_is_better=True
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=gap_scores,
                    optimal_k=optimal_k,
                    method_name="Gap Statistic",
                    higher_is_better=True,
                )
            if method == "aic":
                valid_scores = self._filter_valid_scores(aic_scores)
                if not valid_scores:
                    self._log_message("No valid AIC scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, aic_scores, higher_is_better=False
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=aic_scores,
                    optimal_k=optimal_k,
                    method_name="Akaike Information Criterion",
                    higher_is_better=False,
                )
            if method == "bic":
                valid_scores = self._filter_valid_scores(bic_scores)
                if not valid_scores:
                    self._log_message("No valid BIC scores found in ensemble", level="warning")
                    optimal_k = self.kmin
                else:
                    optimal_k = self._intelligent_k_selection(
                        self.k_range, bic_scores, higher_is_better=False
                    )
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=bic_scores,
                    optimal_k=optimal_k,
                    method_name="Bayesian Information Criterion",
                    higher_is_better=False,
                )
            # Fallback (should not occur with validated inputs)
            return OptimizationResult(
                k_values=self.k_range.copy(),
                scores=[np.nan] * len(self.k_range),
                optimal_k=self.kmin,
                method_name=method,
                higher_is_better=True,
            )

        results: dict[str, OptimizationResult] = {}
        for m in methods:
            results[m] = build_result(m)

        return results

    @staticmethod
    def _compute_W_q(data: np.ndarray, segmentation: np.ndarray, maps: np.ndarray) -> float:
        """Compute W_q (measure of dispersion) for KL criterion.

        W_q = sum_{r=1}^q (1/(2*n_r)) * D_r where D_r = sum_{u,v in cluster r} distance(u, v)^2.
        Uses correlation-based distance to respect polarity invariance of microstates.

        Args:
            data: Data matrix (n_channels, n_samples).
            segmentation: Cluster labels for each sample.
            maps: Cluster centers (n_clusters, n_channels).

        Returns:
            float: W_q value.
        """
        n_clusters = maps.shape[0]
        W_q = 0.0

        for r in range(n_clusters):
            # Samples in cluster r
            cluster_mask = segmentation == r
            n_r = int(np.sum(cluster_mask))
            if n_r <= 1:
                continue

            cluster_data = data[:, cluster_mask]

            # Normalize columns (timepoints) for correlation computation
            norms = np.linalg.norm(cluster_data, axis=0, keepdims=True) + 1e-10
            cluster_data_norm = cluster_data / norms

            # Pairwise absolute correlations between all samples in cluster r
            corr = np.abs(cluster_data_norm.T @ cluster_data_norm)  # (n_r, n_r)
            # Convert to distances and square
            dist_sq = (1.0 - corr) ** 2

            # Sum over upper triangle (i < j) to avoid double counting and exclude diagonal
            D_r = np.sum(np.triu(dist_sq, k=1))

            # Accumulate contribution
            W_q += (1.0 / (2.0 * n_r)) * D_r

        return W_q

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def find_optimal_k(
        self, optimizer_mode: str, parameter_value: float = None
    ) -> tuple[int, list[int], list[float]]:
        """Find optimal k using specified optimization method.

        Args:
            optimizer_mode: Optimization method ('gev', 'db', 'cv', 'kl', 'majority_vote').
            parameter_value: Parameter value for the method (e.g., n_folds for CV).

        Returns:
            tuple[int, list[int], list[float]]: (optimal_k, k_values, scores).
        """
        # Method mapping
        method_mapping = {
            "gev": ("compute_elbow_gev", None),
            "db": ("compute_davies_bouldin", None),
            "cv": ("compute_cross_validation", None),
            "kl": ("compute_krzanowski_lai", None),
            "sil": ("compute_silhouette", None),
            "dunn": ("compute_dunn_index_optimization", None),
            "ch": ("compute_calinski_harabasz_optimization", None),
            "gap": ("compute_gap_statistic_optimization", None),
            "aic": ("compute_aic_optimization", None),
            "bic": ("compute_bic_optimization", None),
            "majority_vote": ("find_optimal_k_majority_vote", None),
        }

        if optimizer_mode not in method_mapping:
            raise ValueError(
                f"Unknown optimizer mode: {optimizer_mode}. Available: {list(method_mapping.keys())}"
            )

        method_name, param_name = method_mapping[optimizer_mode]
        method = getattr(self, method_name)

        # Call method with parameter if applicable
        if param_name and parameter_value is not None:
            kwargs = {param_name: int(parameter_value)}
            self._log_message(f"Computing {optimizer_mode.upper()} optimization...")
            result = method(**kwargs)
        else:
            # Get friendly method name for single method optimization
            method_names = {
                "gev": "Global Explained Variance",
                "db": "Davies-Bouldin Index", 
                "cv": "Cross-Validation",
                "kl": "Krzanowski-Lai Criterion",
                "sil": "Silhouette Coefficient",
                "dunn": "Dunn Index",
                "ch": "Calinski-Harabasz Index",
                "gap": "Gap Statistic",
                "aic": "Akaike Information Criterion",
                "bic": "Bayesian Information Criterion"
            }
            friendly_name = method_names.get(optimizer_mode, optimizer_mode)
            self._log_message(f"Computing {friendly_name} optimization...")
            result = method()

        # Store result
        self.results[optimizer_mode] = result

        return result.optimal_k, result.k_values, result.scores

    def find_optimal_k_majority_vote(self, methods: list[str] = None) -> int:
        """Find optimal k using majority vote across multiple methods.

        Computes all metrics for each k value, then uses majority vote.

        Args:
            methods: Methods to use. If None, uses all available methods.

        Returns:
            int: Optimal k based on majority vote.
        """
        if methods is None:
            methods = ["gev", "db", "cv", "kl", "sil", "dunn", "ch", "gap", "aic", "bic"]

        self._log_message(
            f"Computing ensemble optimization (k={self.kmin}-{self.kmax}, {len(methods)} methods)..."
        )

        # Store results for each k value
        k_results = {}
        metric_votes = {metric: {"optimal_k": None, "scores": []} for metric in methods}

        # For each k value, compute clustering once and calculate all metrics
        total_steps = len(self.k_range)
        for step_idx, k in enumerate(self.k_range):
            # Check if process should stop
            self._check_stop()

            current_step = step_idx + 1
            self._update_progress(current_step, total_steps, f"Computing all metrics for k={k}")

            # Only log progress every few steps to reduce verbosity
            if current_step == 1 or current_step == total_steps or current_step % 3 == 0:
                self._log_message(f"Processing k={k} ({current_step}/{total_steps})")

            try:
                # Perform clustering once for this k
                clustering_result = self._get_clustering_result(k, compute_all_metrics=True)

                # Store all metrics for this k
                k_results[k] = clustering_result

                # Extract scores for each metric (use exact keys from optimizer window)
                metric_key_map = {
                    "gev": "gev",
                    "db": "davies_bouldin",
                    "cv": "cv_score",
                    "kl": "kl_score",
                    "sil": "silhouette_score",
                    "dunn": "dunn_index",
                    "ch": "calinski_harabasz_index",
                    "gap": "gap_statistic",
                    "aic": "aic_score",
                    "bic": "bic_score",
                }

                for metric in methods:
                    key = metric_key_map.get(metric, metric)
                    if key in clustering_result:
                        metric_votes[metric]["scores"].append(clustering_result[key])
                    else:
                        metric_votes[metric]["scores"].append(np.nan)

                # Skip detailed metric logging for each k to reduce verbosity
                # Detailed results will be shown in final summary

            except Exception as e:
                self._log_message(f"Error computing metrics for k={k}: {str(e)}", level="error")
                # Add NaN values for this k
                for metric in methods:
                    metric_votes[metric]["scores"].append(np.nan)

        # Special handling for KL scores - need to compute from M_q values
        if "kl" in methods:
            try:
                self._log_message("Computing KL scores from M_q values...")
                M_values = {}
                for k in self.k_range:
                    if k in k_results and "M_q" in k_results[k]:
                        M_values[k] = k_results[k]["M_q"]
                    else:
                        M_values[k] = np.nan
                
                kl_scores = self._compute_kl_scores_from_M_values(M_values)
                
                # Update the kl scores in metric_votes
                for i, score in enumerate(kl_scores):
                    metric_votes["kl"]["scores"][i] = score
                    
                self._log_message("KL scores computed successfully")
            except Exception as e:
                self._log_message(f"Error computing KL scores: {str(e)}", level="error")

        # Find optimal k for each metric
        k_votes = {k: 0 for k in self.k_range}

        for metric in methods:
            try:
                scores = metric_votes[metric]["scores"]
                # Filter out NaN values
                valid_scores = [(i, s) for i, s in enumerate(scores) if not np.isnan(s)]

                if valid_scores:
                    valid_indices, valid_scores_list = zip(*valid_scores)
                    valid_k_values = [self.k_range[i] for i in valid_indices]

                    # Find optimal k for this metric
                    optimal_k = self._find_optimal_k_for_metric(
                        metric, valid_k_values, valid_scores_list
                    )

                    if optimal_k in k_votes:
                        k_votes[optimal_k] += 1
                        metric_votes[metric]["optimal_k"] = optimal_k
                        self._log_message(f"{metric.upper()} method voted for k={optimal_k}")
                    else:
                        self._log_message(
                            f"Warning: {metric.upper()} optimal k {optimal_k} not in range {self.k_range}",
                            level="warning",
                        )
                else:
                    self._log_message(
                        f"No valid scores for {metric.upper()} method", level="warning"
                    )

            except Exception as e:
                self._log_message(
                    f"Error finding optimal k for {metric.upper()} method: {str(e)}", level="error"
                )
                continue

        # Find k with most votes
        if not any(k_votes.values()):
            self._log_message("No valid votes found, using default k=5", level="warning")
            return 5

        optimal_k = max(k_votes, key=k_votes.get)
        vote_count = k_votes[optimal_k]
        total_metrics = len([m for m in methods if metric_votes[m]["optimal_k"] is not None])

        self._log_message(
            f"Majority vote result: k={optimal_k} with {vote_count}/{total_metrics} votes"
        )
        vote_distribution = ", ".join(
            [f"k={k}: {votes}" for k, votes in k_votes.items() if votes > 0]
        )
        self._log_message(f"Vote distribution: {vote_distribution}")

        # Store results for later access
        self.majority_vote_results = {
            "k_results": k_results,
            "metric_votes": metric_votes,
            "k_votes": k_votes,
            "optimal_k": optimal_k,
        }

        return optimal_k

    def _find_optimal_k_for_metric(
        self, metric: str, k_values: list[int], scores: list[float]
    ) -> int:
        """Find optimal k for a specific metric.

        Args:
            metric: Metric name ('gev', 'db', 'cv', 'kl').
            k_values: List of k values.
            scores: Scores for each k value.

        Returns:
            int: Optimal k value.
        """
        if not scores or not k_values:
            return k_values[0] if k_values else 2

        # Handle methods based on optimization direction
        if metric == "db":
            # Lower is better - find minimum
            min_idx = np.argmin(scores)
            return k_values[min_idx]
        if metric == "gev":
            # Global Explained Variance - use elbow method
            return self._find_elbow_point(k_values, scores, higher_is_better=True)
        if metric == "cv":
            # Cross validation - typically lower is better
            min_idx = np.argmin(scores)
            return k_values[min_idx]
        if metric == "kl":
            # Krzanowski-Lai - higher is better
            max_idx = np.argmax(scores)
            return k_values[max_idx]
        if metric == "sil":
            # Silhouette coefficient - higher is better
            max_idx = np.argmax(scores)
            return k_values[max_idx]
        if metric == "dunn":
            # Dunn Index - higher is better
            max_idx = np.argmax(scores)
            return k_values[max_idx]
        if metric == "ch":
            # Calinski-Harabasz Index - higher is better
            max_idx = np.argmax(scores)
            return k_values[max_idx]
        if metric == "gap":
            # Gap Statistic - higher is better
            max_idx = np.argmax(scores)
            return k_values[max_idx]
        if metric == "aic":
            # AIC - lower is better
            min_idx = np.argmin(scores)
            return k_values[min_idx]
        if metric == "bic":
            # BIC - lower is better
            min_idx = np.argmin(scores)
            return k_values[min_idx]
        # Default to first k value
        return k_values[0]

    def get_all_results(self) -> dict[str, OptimizationResult]:
        """Get all computed results.

        Returns:
            dict[str, OptimizationResult]: Mapping from method code to result.
        """
        return self.results.copy()

    @staticmethod
    def _normalize_data(data: np.ndarray) -> np.ndarray:
        """Normalize data for clustering metrics.

        Args:
            data: Input data (n_channels, n_samples).

        Returns:
            np.ndarray: Normalized data.
        """
        return (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-10)

    @staticmethod
    def _compute_wss(data: np.ndarray, maps: np.ndarray, segmentation: np.ndarray) -> float:
        """Compute within-cluster sum of squares using correlation distance.

        Args:
            data: Data matrix (n_channels, n_samples).
            maps: Cluster centers (n_clusters, n_channels).
            segmentation: Cluster labels for each sample.

        Returns:
            float: Within-cluster sum of squares.
        """
        # Normalize data and maps
        data_norm = data / (np.linalg.norm(data, axis=0, keepdims=True) + 1e-10)
        maps_norm = maps / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-10)

        wss = 0
        for k in range(maps.shape[0]):
            cluster_mask = segmentation == k
            if np.sum(cluster_mask) > 0:
                cluster_data = data_norm[:, cluster_mask]
                center = maps_norm[k : k + 1]

                # Correlation distances (1 - |correlation|)
                correlations = np.abs(np.dot(center, cluster_data)).flatten()
                distances = 1 - correlations
                wss += np.sum(distances**2)
        return wss

    @staticmethod
    def _is_monotonic(scores: list[float], increasing: bool = True) -> bool:
        """Check if scores are monotonically increasing or decreasing.
        
        Args:
            scores: List of scores to check.
            increasing: True for monotonic increasing, False for decreasing.
            
        Returns:
            bool: True if monotonic within tolerance.
        """
        if len(scores) < 3:
            return True  # Too few points to determine pattern
        
        valid_scores = [s for s in scores if not np.isnan(s)]
        if len(valid_scores) < 3:
            return True
        
        if increasing:
            # Check if generally increasing with tolerance for small fluctuations
            differences = np.diff(valid_scores)
            increasing_count = np.sum(differences > 0)
            return increasing_count >= len(differences) * 0.7  # 70% threshold
        else:
            # Check if generally decreasing
            differences = np.diff(valid_scores)
            decreasing_count = np.sum(differences < 0)
            return decreasing_count >= len(differences) * 0.7  # 70% threshold

    @staticmethod
    def _find_local_optima(
        k_values: list[int], 
        scores: list[float], 
        higher_is_better: bool = True
    ) -> int:
        """Find the most significant local optimum (maximum or minimum).
        
        Args:
            k_values: List of k values.
            scores: List of corresponding scores.
            higher_is_better: True for local maxima, False for local minima.
            
        Returns:
            int: k value at the most significant local optimum.
        """
        valid_pairs = [(k, score) for k, score in zip(k_values, scores) if not np.isnan(score)]
        if len(valid_pairs) < 3:
            # Not enough points for local optima, return best value
            if higher_is_better:
                return max(valid_pairs, key=lambda x: x[1])[0]
            else:
                return min(valid_pairs, key=lambda x: x[1])[0]
        
        valid_k, valid_scores = zip(*valid_pairs)
        valid_scores = list(valid_scores)
        
        # Find local optima (peaks or valleys)
        optima_indices = []
        for i in range(1, len(valid_scores) - 1):
            if higher_is_better:
                # Look for local maxima
                if valid_scores[i] > valid_scores[i-1] and valid_scores[i] > valid_scores[i+1]:
                    optima_indices.append(i)
            else:
                # Look for local minima
                if valid_scores[i] < valid_scores[i-1] and valid_scores[i] < valid_scores[i+1]:
                    optima_indices.append(i)
        
        if not optima_indices:
            # No local optima found, return global optimum
            if higher_is_better:
                return max(valid_pairs, key=lambda x: x[1])[0]
            else:
                return min(valid_pairs, key=lambda x: x[1])[0]
        
        # Return the most extreme local optimum
        if higher_is_better:
            best_idx = max(optima_indices, key=lambda i: valid_scores[i])
        else:
            best_idx = min(optima_indices, key=lambda i: valid_scores[i])
        
        return valid_k[best_idx]

    @staticmethod
    def _has_global_optimum(
        k_values: list[int], 
        scores: list[float], 
        higher_is_better: bool = True
    ) -> tuple[bool, int]:
        """Check if there's a global optimum not at the endpoints.
        
        Args:
            k_values: List of k values.
            scores: List of corresponding scores.
            higher_is_better: True for global max, False for global min.
            
        Returns:
            tuple: (has_global_optimum, optimal_k)
        """
        valid_pairs = [(k, score) for k, score in zip(k_values, scores) if not np.isnan(score)]
        if len(valid_pairs) < 3:
            return False, None
        
        valid_k, valid_scores = zip(*valid_pairs)
        
        # Find global optimum
        if higher_is_better:
            global_optimum_idx = np.argmax(valid_scores)
            global_optimum_value = max(valid_scores)
        else:
            global_optimum_idx = np.argmin(valid_scores)
            global_optimum_value = min(valid_scores)
        
        # Check if global optimum is not at endpoints (avoid boundary effects)
        is_not_endpoint = 0 < global_optimum_idx < len(valid_scores) - 1
        
        if is_not_endpoint:
            return True, valid_k[global_optimum_idx]
        else:
            # Global optimum is at endpoint, check if it's significantly better
            if higher_is_better:
                # For max: check if endpoint is much better than second-best
                sorted_scores = sorted(valid_scores, reverse=True)
                if len(sorted_scores) >= 2:
                    improvement = (sorted_scores[0] - sorted_scores[1]) / abs(sorted_scores[1])
                    if improvement > 0.05:  # 5% improvement threshold
                        return True, valid_k[global_optimum_idx]
            else:
                # For min: check if endpoint is much better than second-best
                sorted_scores = sorted(valid_scores)
                if len(sorted_scores) >= 2:
                    improvement = (sorted_scores[1] - sorted_scores[0]) / abs(sorted_scores[1])
                    if improvement > 0.05:  # 5% improvement threshold
                        return True, valid_k[global_optimum_idx]
        
        return False, None

    @staticmethod
    def _intelligent_k_selection(
        k_values: list[int],
        scores: list[float],
        higher_is_better: bool = True
    ) -> int:
        """Intelligently select optimal k using global optima, local optima, or elbow method.
        
        Strategy:
        - If scores are monotonic: use elbow method
        - If non-monotonic with global optimum: choose global optimum
        - If non-monotonic without clear global optimum: choose local optimum
        
        Args:
            k_values: List of k values.
            scores: List of corresponding scores.
            higher_is_better: True for metrics where higher is better.
            
        Returns:
            int: Optimal k value.
        """
        # Check if monotonic
        is_monotonic = ClustererOptimizer._is_monotonic(scores, increasing=higher_is_better)
        
        if is_monotonic:
            # Use elbow method for monotonic data
            return ClustererOptimizer._find_elbow_point(k_values, scores, higher_is_better=higher_is_better)
        else:
            # Data is non-monotonic, check for global optimum first
            has_global_opt, global_k = ClustererOptimizer._has_global_optimum(k_values, scores, higher_is_better)
            
            if has_global_opt:
                # Use global optimum for non-monotonic data
                return global_k
            else:
                # Fall back to local optima for non-monotonic data
                return ClustererOptimizer._find_local_optima(k_values, scores, higher_is_better=higher_is_better)

    @staticmethod
    def _find_elbow_point(
        x_values: list[int],
        y_values: list[float],
        threshold: float = None,
        higher_is_better: bool = True,
    ) -> int:
        """Find elbow point using automatic threshold detection (Kneedle algorithm).

        Args:
            x_values: K values evaluated.
            y_values: Corresponding metric scores.
            threshold: Deprecated - kept for backward compatibility.
            higher_is_better: Direction of optimization for context.

        Returns:
            int: Selected elbow k value.
        """
        if len(y_values) < 3:
            return x_values[0] if x_values else 2
            
        # Filter out NaN values
        valid_pairs = [(x, y) for x, y in zip(x_values, y_values) if not np.isnan(y)]
        if len(valid_pairs) < 3:
            return x_values[0] if x_values else 2
            
        x_vals, y_vals = zip(*valid_pairs)
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        
        # Normalize data to [0, 1] range for consistent processing
        x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min()) if x_vals.max() != x_vals.min() else np.zeros_like(x_vals)
        y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min()) if y_vals.max() != y_vals.min() else np.zeros_like(y_vals)
        
        # Adjust for optimization direction
        if not higher_is_better:
            y_norm = 1 - y_norm
        
        # Kneedle algorithm: find maximum distance from line connecting start to end
        start_point = np.array([x_norm[0], y_norm[0]])
        end_point = np.array([x_norm[-1], y_norm[-1]])
        
        # Calculate distances from each point to the line
        distances = []
        for i in range(len(x_norm)):
            point = np.array([x_norm[i], y_norm[i]])
            # Distance from point to line
            if np.linalg.norm(end_point - start_point) > 0:
                distance = np.abs(np.cross(end_point - start_point, start_point - point)) / np.linalg.norm(end_point - start_point)
            else:
                distance = 0
            distances.append(distance)
        
        # Find point with maximum distance (elbow point)
        if distances:
            elbow_idx = np.argmax(distances)
            return x_vals[elbow_idx]
        
        # Fallback to midpoint if no clear elbow
        return x_vals[len(x_vals) // 2]