"""Optimization utilities for selecting microstate cluster counts (K)."""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from clustering_utils.microstate_clusterer import MicrostateClusterer

warnings.filterwarnings("ignore")


@dataclass(init=False)
class OptimizationResult:
    """Container for optimization results."""

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
        self.k_values = k_values
        self.scores = scores
        self.optimal_k = optimal_k
        self.method_name = method_name
        self.higher_is_better = higher_is_better


class ClustererOptimizer:
    """Optimized clusterer for finding the optimal number of microstate clusters.
    This is the SINGLE SOURCE OF TRUTH for all clustering metrics implementations.
    """

    def __init__(
        self,
        maps2use: np.ndarray,
        min_dist: Optional[int] = None,
        n_inits: int = 1,  # Single repeat per k for auto-k selection
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
    ):
        """Initialize the clusterer optimizer.

        Args:
            maps2use (np.ndarray): Data to cluster (n_channels x n_timepoints).
            min_dist (int | None): Minimum distance for peak detection.
            n_inits (int): Number of clustering initializations (set to 1 for single repeat).
            kmin (int): Minimum number of clusters.
            kmax (int): Maximum number of clusters.
            preprocessed_data_path (str | None): Path to preprocessed data.
            extension (str): File extension.
            datatype (str): Data type.
            tolerance (float): Convergence tolerance.
            max_iter (int): Maximum iterations.
            batch_size (int | None): Batch size for clustering.
            progress_callback (Callable[[int, int, str], None] | None): Callback for progress updates.
            logger (object | None): Logger instance for consistent formatting.
        """
        # Validate and ensure correct data format
        if maps2use.ndim != 2:
            raise ValueError(f"maps2use must be 2D array, got shape {maps2use.shape}")

        # Ensure data is in (n_channels, n_timepoints) format
        if maps2use.shape[0] > maps2use.shape[1]:
            print(
                "Warning: Data appears to be in (n_timepoints, n_channels) format. Transposing to (n_channels, n_timepoints)"
            )
            maps2use = maps2use.T

        self.maps2use = maps2use
        self.min_dist = min_dist
        self.n_inits = n_inits
        self.kmin = kmin
        self.kmax = kmax

        # Validate k range
        if kmin < 2:
            print(f"Warning: kmin {kmin} is less than 2, setting to 2")
            self.kmin = 2
        if kmax <= kmin:
            print(
                f"Warning: kmax {kmax} is not greater than kmin {kmin}, setting kmax to {kmin + 1}"
            )
            self.kmax = kmin + 1

        # Ensure kmax doesn't exceed the number of samples
        max_possible_k = min(self.maps2use.shape[1], 20)  # Limit to 20 for practical reasons
        if self.kmax > max_possible_k:
            print(
                f"Warning: kmax {self.kmax} exceeds maximum possible k {max_possible_k}, setting to {max_possible_k}"
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

        # Results storage
        self.results: dict[str, OptimizationResult] = {}
        self._clustering_cache: dict[int, dict] = {}

        # Majority vote results storage (populated in find_optimal_k_majority_vote)
        self.majority_vote_results: dict[str, Any] | None = None

        # Stop functionality
        self._stopped = False

    def stop(self):
        """Stop the optimization process."""
        self._stopped = True
        # Log stop request using the new logger format if available
        if hasattr(self, "progress_callback") and self.progress_callback:
            # Try to get the logger from the progress callback context
            from contextlib import suppress

            with suppress(BaseException):
                # In GUI mode, the callback usually has access to a logger via its closure
                pass
        self._log_message("Optimization process stopped by user", level="warning")

    def is_stopped(self):
        """Check if the optimization process has been stopped."""
        return self._stopped

    def _check_stop(self):
        """Check if the process should stop and raise exception if so."""
        if self._stopped:
            raise RuntimeError("Optimization process stopped by user")

    def _update_progress(self, current: int, total: int, message: str):
        """Update progress if callback is provided."""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def _filter_valid_scores(self, scores: list[float], exclude_inf: bool = False) -> list[tuple[int, float]]:
        """Pair k values with scores and filter out invalid entries.

        Args:
            scores (list[float]): Scores aligned with `self.k_range`.
            exclude_inf (bool): If True, also exclude float("inf").

        Returns:
            list[tuple[int, float]]: Valid (k, score) pairs.
        """
        pairs = list(zip(self.k_range, scores))
        if exclude_inf:
            return [(k, s) for k, s in pairs if not np.isnan(s) and s != float("inf")]
        return [(k, s) for k, s in pairs if not np.isnan(s)]

    def _compute_M_q(self, k: int, result: Optional[dict] = None) -> tuple[float, float]:
        """Compute W_q and M_q for a given k.

        Args:
            k (int): Number of clusters.
            result (dict | None): Optional cached clustering result.

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
            M_values (dict[int, float]): Mapping from k to M_q value.

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
            message (str): Message to log.
            level (str): Log level ('info', 'warning', 'error').
        """
        # Use the logger instance if available for consistent emoji formatting
        if self.logger is not None:
            if level == "error":
                self.logger.error("CLUSTERING", message)
            elif level == "warning":
                self.logger.warning("CLUSTERING", message)
            else:
                self.logger.processing_info("CLUSTERING", message)
        else:
            # Fallback to terminal output with consistent formatting
            if level == "error":
                print(f"[ERROR] {message}")
            elif level == "warning":
                print(f"[WARNING] {message}")
            else:
                print(f"[INFO] {message}")

        # If progress callback is available, it might be connected to a log window
        # The progress callback can handle additional logging if needed

    def _get_clustering_result(self, k: int) -> dict:
        """Get clustering result for k clusters, using cache if available.

        Args:
            k (int): Number of clusters.

        Returns:
            dict: Contains all metrics computed like optimizer window.
        """
        if k in self._clustering_cache:
            return self._clustering_cache[k]

        # Clustering timing start
        start_time = time.time()

        # Initialize the MicrostateClusterer for this K (same as optimizer window)
        clusterer = MicrostateClusterer(
            n_states=k,
            batch_size=self.batch_size,
            n_inits=1,  # Force single repeat for auto-k selection
            max_iter=self.max_iter,
            tolerance=self.tolerance,
        )

        n_channels = self.maps2use.shape[0]

        def _generate_normalized_random_maps(num_clusters: int, num_channels: int) -> np.ndarray:
            """Generate normalized random maps (rows unit-norm).

            Args:
                num_clusters (int): Number of clusters/maps to generate.
                num_channels (int): Number of channels per map.

            Returns:
                np.ndarray: Array of shape (num_clusters, num_channels).
            """
            random_maps = np.random.randn(num_clusters, num_channels)
            row_norms = np.linalg.norm(random_maps, axis=1, keepdims=True) + 1e-12
            return random_maps / row_norms

        # Run single initialization (same as optimizer window but with n_inits=1)
        try:
            # Random initialization of maps
            initial_maps = _generate_normalized_random_maps(k, n_channels)

            # Run modified K-means
            maps, residual = clusterer.modified_kmeans(self.maps2use, initial_maps, verbose=False)

            # Calculate GEV for this result
            gev = clusterer.compute_gev(self.maps2use, maps)

            # Calculate final segmentation
            activation = maps.dot(self.maps2use)
            labels = np.argmax(np.abs(activation), axis=0)

        except Exception as err:
            self._log_message(
                f"Warning: Initialization failed for k={k}: {err}", level="warning"
            )
            raise RuntimeError(f"Clustering failed for k={k}") from err

        # Cache the result
        self._clustering_cache[k] = {
            "maps": maps.copy(),
            "segmentation": labels,
            "gev": gev,
            "residual": residual,
            "computation_time": time.time() - start_time,
        }

        # Compute additional metrics for this clustering result
        self._compute_all_metrics_exact(self._clustering_cache[k], k, labels, gev, residual)

        return self._clustering_cache[k]

    def _compute_all_metrics_exact(
        self, clustering_result, k, best_labels, best_gev, best_residual
    ):
        """Compute all metrics for a clustering result.

        Ensures consistency between auto-k selection and optimizer visualization.

        Args:
            clustering_result (dict): Cache entry to update with computed metrics.
            k (int): Number of clusters.
            best_labels (np.ndarray): Segmentation labels for each sample.
            best_gev (float): Global explained variance score.
            best_residual (float): Residual error from clustering.
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
            data (np.ndarray): Data matrix of shape (n_channels, n_samples).
            labels (np.ndarray): Cluster labels for each sample.
            maps (np.ndarray): Cluster centers of shape (n_clusters, n_channels).

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
            data (np.ndarray): EEG data (n_channels, n_timepoints); this is u(t).
            maps (np.ndarray): Template maps (n_clusters, n_channels); these are T_k.
            segmentation (np.ndarray): Cluster assignments per timepoint; selects T_t.

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
            # Check if process should stop
            self._check_stop()

            self._update_progress(i + 1, total_steps, f"Computing GEV for k={k}")
            result = self._get_clustering_result(k)
            scores.append(result["gev"])

        optimal_k = self._find_elbow_point(
            self.k_range, scores, threshold=5.0, higher_is_better=True
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

            self._update_progress(i + 1, total_steps, f"Computing Davies-Bouldin for k={k}")

            if k < 2:  # DB requires at least 2 clusters
                scores.append(float("inf"))
                continue

            result = self._get_clustering_result(k)

            # Use custom polarity-invariant Davies-Bouldin score
            score = self.compute_custom_davies_bouldin(
                data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
            )
            scores.append(score)

        # Find optimal k (excluding k=1)
        valid_scores = self._filter_valid_scores(scores, exclude_inf=True)
        optimal_k = min(valid_scores, key=lambda x: x[1])[0] if valid_scores else self.kmin

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
            self._update_progress(idx + 1, total_steps, f"Computing CV for k={k}")

            # Retrieve (or compute) clustering result – this already contains the CV score
            result = self._get_clustering_result(k)
            scores.append(result.get("cv_score", np.nan))

        # Optimal k = argmin CV
        valid_scores = self._filter_valid_scores(scores)
        optimal_k = min(valid_scores, key=lambda x: x[1])[0] if valid_scores else self.kmin

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
                "Warning: Need at least 3 k values for KL criterion, using default k=4",
                level="warning",
            )
            return OptimizationResult(
                k_values=self.k_range.copy(),
                scores=[np.nan] * len(self.k_range),
                optimal_k=4,
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

                self._log_message(f"k={k}: W_q={W_q:.6f}, M_q={M_q:.6f}")

            except Exception as e:
                self._log_message(
                    f"Error computing KL criterion for k={k}: {str(e)}", level="error"
                )
                M_values[k] = np.nan

        # Compute KL scores
        kl_scores = self._compute_kl_scores_from_M_values(M_values)

        # Find optimal k (higher KL score is better)
        valid_scores = [
            (k_val, score)
            for k_val, score in zip(self.k_range, kl_scores)
            if not np.isnan(score) and score > 0
        ]
        if valid_scores:
            optimal_k = max(valid_scores, key=lambda x: x[1])[0]
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

    def compute_methods_batch(
        self, methods: list[str], parameters: Optional[dict[str, float]] = None
    ) -> dict[str, OptimizationResult]:
        """Compute multiple optimisation methods in a single pass over k.

        Avoids recomputing clustering for each method separately while preserving
        exact metric definitions.

        Args:
            methods (list[str]): Method codes to compute. Supported: ['gev', 'db', 'cv', 'kl'].
            parameters (dict[str, float] | None): Optional parameters per method (e.g., {'gev': 5.0}).

        Returns:
            dict[str, OptimizationResult]: Mapping from method code to result.
        """
        parameters = parameters or {}

        # Storage for per-k scores
        gev_scores: list[float] = []
        db_scores: list[float] = []
        cv_scores: list[float] = []

        # For KL we need M_q values across adjacent k
        M_values: dict[int, float] = {}

        total_steps = len(self.k_range)
        for step_index, k in enumerate(self.k_range):
            self._check_stop()
            self._update_progress(step_index + 1, total_steps, f"Computing metrics for k={k}")

            result = self._get_clustering_result(k)

            # Extract scores from cached result
            gev_scores.append(result.get("gev", np.nan))

            if k < 2:
                db_scores.append(float("inf"))
            else:
                try:
                    db_val = self.compute_custom_davies_bouldin(
                        data=self.maps2use, labels=result["segmentation"], maps=result["maps"]
                    )
                except Exception:
                    db_val = np.nan
                db_scores.append(db_val)

            cv_scores.append(result.get("cv_score", np.nan))

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
                threshold = float(parameters.get("gev", 5.0))
                optimal_k = self._find_elbow_point(
                    self.k_range, gev_scores, threshold, higher_is_better=True
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
                    if score != float("inf") and not np.isnan(score)
                ]
                optimal_k = min(valid_scores, key=lambda x: x[1])[0] if valid_scores else self.kmin
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=db_scores,
                    optimal_k=optimal_k,
                    method_name="Davies-Bouldin Criterion",
                    higher_is_better=False,
                )
            if method == "cv":
                valid_scores = self._filter_valid_scores(cv_scores)
                optimal_k = min(valid_scores, key=lambda x: x[1])[0] if valid_scores else self.kmin
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
                optimal_k = max(valid_scores, key=lambda x: x[1])[0] if valid_scores else self.kmin
                return OptimizationResult(
                    k_values=self.k_range.copy(),
                    scores=kl_scores,
                    optimal_k=optimal_k,
                    method_name="Krzanowski-Lai Criterion",
                    higher_is_better=True,
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
            data (np.ndarray): Data matrix (n_channels, n_samples).
            segmentation (np.ndarray): Cluster labels for each sample.
            maps (np.ndarray): Cluster centers (n_clusters, n_channels).

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
            optimizer_mode (str): Optimization method ('gev', 'db', 'cv', 'kl', 'majority_vote').
            parameter_value (float | None): Parameter value for the method (e.g., n_folds for CV).

        Returns:
            tuple[int, list[int], list[float]]: (optimal_k, k_values, scores).
        """
        # Method mapping
        method_mapping = {
            "gev": ("compute_elbow_gev", None),
            "db": ("compute_davies_bouldin", None),
            "cv": ("compute_cross_validation", None),
            "kl": ("compute_krzanowski_lai", None),
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
            self._log_message(f"Calling {method_name} with parameters")
            result = method(**kwargs)
        else:
            self._log_message(f"Calling {method_name}")
            result = method()

        # Store result
        self.results[optimizer_mode] = result

        return result.optimal_k, result.k_values, result.scores

    def find_optimal_k_majority_vote(self, methods: list[str] = None) -> int:
        """Find optimal k using majority vote across multiple methods.

        Computes all metrics for each k value, then uses majority vote.

        Args:
            methods (list[str] | None): Methods to use. If None, uses all available methods.

        Returns:
            int: Optimal k based on majority vote.
        """
        if methods is None:
            methods = ["gev", "db", "cv", "kl"]

        self._log_message(
            f"Starting majority vote optimization for k range {self.kmin} to {self.kmax}"
        )
        self._log_message(f"Using optimization methods: {', '.join(methods)}")

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

            self._log_message(f"Computing all metrics for k={k} ({current_step}/{total_steps})")

            try:
                # Perform clustering once for this k
                clustering_result = self._get_clustering_result(k)

                # Store all metrics for this k
                k_results[k] = clustering_result

                # Extract scores for each metric (use exact keys from optimizer window)
                metric_key_map = {
                    "gev": "gev",
                    "db": "davies_bouldin",
                    "cv": "cv_score",
                    "kl": "kl_score",
                }

                for metric in methods:
                    key = metric_key_map.get(metric, metric)
                    if key in clustering_result:
                        metric_votes[metric]["scores"].append(clustering_result[key])
                    else:
                        metric_votes[metric]["scores"].append(np.nan)

                # Safe formatting for debug output (use exact keys from optimizer window)
                gev_val = clustering_result.get("gev", "N/A")
                db_val = clustering_result.get("davies_bouldin", "N/A")
                cv_val = clustering_result.get("cv_score", "N/A")
                kl_val = clustering_result.get("kl_score", "N/A")

                gev_str = f"{gev_val:.4f}" if isinstance(gev_val, (int, float)) else str(gev_val)
                db_str = f"{db_val:.4f}" if isinstance(db_val, (int, float)) else str(db_val)
                cv_str = f"{cv_val:.4f}" if isinstance(cv_val, (int, float)) else str(cv_val)
                kl_str = f"{kl_val:.4f}" if isinstance(kl_val, (int, float)) else str(kl_val)

                self._log_message(
                    f"k={k} metrics - GEV: {gev_str}, Davies-Bouldin: {db_str}, CV: {cv_str}, KL: {kl_str}"
                )

            except Exception as e:
                self._log_message(f"Error computing metrics for k={k}: {str(e)}", level="error")
                # Add NaN values for this k
                for metric in methods:
                    metric_votes[metric]["scores"].append(np.nan)

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
            self._log_message("No valid votes found, using default k=4", level="warning")
            return 4

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
            metric (str): Metric name ('gev', 'db', 'cv', 'kl').
            k_values (list[int]): List of k values.
            scores (list[float]): Scores for each k value.

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
            threshold = 5.0  # Default threshold
            return self._find_elbow_point(k_values, scores, threshold, higher_is_better=True)
        if metric == "cv":
            # Cross validation - typically lower is better
            min_idx = np.argmin(scores)
            return k_values[min_idx]
        if metric == "kl":
            # Krzanowski-Lai - higher is better
            max_idx = np.argmax(scores)
            return k_values[max_idx]
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
            data (np.ndarray): Input data (n_channels, n_samples).

        Returns:
            np.ndarray: Normalized data.
        """
        return (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-10)

    @staticmethod
    def _compute_wss(data: np.ndarray, maps: np.ndarray, segmentation: np.ndarray) -> float:
        """Compute within-cluster sum of squares using correlation distance.

        Args:
            data (np.ndarray): Data matrix (n_channels, n_samples).
            maps (np.ndarray): Cluster centers (n_clusters, n_channels).
            segmentation (np.ndarray): Cluster labels for each sample.

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
    def _find_elbow_point(
        x_values: list[int],
        y_values: list[float],
        threshold: float = 5.0,
        higher_is_better: bool = True,
    ) -> int:
        """Find elbow point using threshold method for percentage change.

        Args:
            x_values (list[int]): K values evaluated.
            y_values (list[float]): Corresponding metric scores.
            threshold (float): Percentage improvement threshold.
            higher_is_better (bool): Direction of optimization for context.

        Returns:
            int: Selected elbow k value.
        """
        if len(y_values) < 2:
            return x_values[0]

        # Adjust series based on optimization direction so that "improvement" is positive
        series = y_values if higher_is_better else [-v for v in y_values]

        # Vectorized percentage change computation
        s = np.asarray(series, dtype=float)
        prev = s[:-1]
        curr = s[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            change_percent = np.where(prev != 0, np.abs((curr - prev) / prev) * 100.0, 100.0)
        matches = np.where(change_percent < threshold)[0]
        if matches.size > 0:
            return x_values[int(matches[0]) + 1]

        # If no elbow found, return the last K
        return x_values[-1]
