import time
import numpy as np
from typing import Dict, List, Optional, Any

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QMainWindow, QSizePolicy, QApplication, QFileDialog,
    QActionGroup, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from data_utils.data_initializer import DataInitializer
from clustering_utils.clusterer_optimizer import ClustererOptimizer, OptimizationResult
from clustering_utils.microstate_clusterer import MicrostateClusterer


# ============================================================================
# Custom Microstate Clustering Metrics
# ============================================================================

class MicrostateClusteringMetrics:
    """
    A collection of clustering quality metrics specifically designed for
    polarity-invariant microstate analysis.
    """

    @staticmethod
    def compute_custom_silhouette(data: np.ndarray, labels: np.ndarray,
                                  maps: Optional[np.ndarray] = None) -> float:
        """
        Compute silhouette score for polarity-invariant microstate clustering.
        Optimized version that avoids creating large pairwise distance matrices.

        The silhouette coefficient for a sample is (b - a) / max(a, b), where:
        - a is the mean distance between a sample and all other points in the same cluster
        - b is the mean distance between a sample and all points in the nearest cluster

        This implementation uses correlation-based distances: d = 1 - |correlation|
        to handle polarity invariance in EEG microstates.

        Parameters
        ----------
        data : np.ndarray
            Data matrix of shape (n_channels, n_samples) or (n_samples, n_channels)
        labels : np.ndarray
            Cluster labels for each sample
        maps : np.ndarray, optional
            Cluster centers (n_clusters, n_channels) - not used in this implementation
            but kept for API consistency

        Returns
        -------
        float
            Average silhouette score across all samples (-1 to 1, higher is better)
        """
        # Ensure data is in (n_samples, n_channels) format for easier computation
        if data.shape[0] < data.shape[1]:
            data = data.T

        n_samples, n_channels = data.shape
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Handle edge case: only one cluster
        if n_clusters == 1:
            return 0.0

        # Check for very large datasets and warn/optimize accordingly
        if n_samples > 50000:
            print(f"Warning: Very large dataset ({n_samples} samples). Using aggressive sampling for performance.")
            return MicrostateClusteringMetrics._compute_sampled_silhouette(
                data, labels, unique_labels, sample_size=2000
            )
        elif n_samples > 10000:
            print(f"Warning: Large dataset ({n_samples} samples). Using sampling for performance.")
            return MicrostateClusteringMetrics._compute_sampled_silhouette(
                data, labels, unique_labels, sample_size=5000
            )
        elif n_samples > 1000:
            return MicrostateClusteringMetrics._compute_batch_silhouette(
                data, labels, unique_labels
            )
        else:
            return MicrostateClusteringMetrics._compute_efficient_silhouette(
                data, labels, unique_labels
            )

    @staticmethod
    def _compute_sampled_silhouette(data: np.ndarray, labels: np.ndarray, 
                                    unique_labels: np.ndarray, sample_size: int = 5000) -> float:
        """
        Compute silhouette score using sampling for very large datasets.
        """
        # Normalize data for correlation computation
        data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-10)
        n_samples = data_norm.shape[0]
        
        # Stratified sampling to ensure each cluster is represented
        sample_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > 0:
                # Sample proportionally from each cluster
                cluster_sample_size = max(1, int(sample_size * len(label_indices) / n_samples))
                cluster_sample_size = min(cluster_sample_size, len(label_indices))
                sampled = np.random.choice(label_indices, size=cluster_sample_size, replace=False)
                sample_indices.extend(sampled)
        
        sample_indices = np.array(sample_indices)
        
        # Compute silhouette on sampled data
        sampled_data = data_norm[sample_indices]
        sampled_labels = labels[sample_indices]
        
        return MicrostateClusteringMetrics._compute_efficient_silhouette(sampled_data, sampled_labels, unique_labels)

    @staticmethod
    def _compute_batch_silhouette(data: np.ndarray, labels: np.ndarray, 
                                  unique_labels: np.ndarray, batch_size: int = 1000) -> float:
        """
        Compute silhouette score using batch processing to manage memory.
        """
        # Normalize data for correlation computation
        data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-10)
        n_samples = data_norm.shape[0]
        silhouette_scores = np.zeros(n_samples)
        
        # Process data in batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = slice(start_idx, end_idx)
            
            batch_data = data_norm[batch_indices]
            batch_labels = labels[batch_indices]
            
            # Compute distances for this batch against all data
            batch_scores = MicrostateClusteringMetrics._compute_batch_scores(
                batch_data, batch_labels, data_norm, labels, unique_labels, start_idx
            )
            
            silhouette_scores[batch_indices] = batch_scores
        
        return np.mean(silhouette_scores)

    @staticmethod
    def _compute_batch_scores(batch_data: np.ndarray, batch_labels: np.ndarray,
                              all_data: np.ndarray, all_labels: np.ndarray,
                              unique_labels: np.ndarray, start_idx: int) -> np.ndarray:
        """
        Compute silhouette scores for a batch of samples.
        """
        batch_size = batch_data.shape[0]
        batch_scores = np.zeros(batch_size)
        
        for i in range(batch_size):
            global_i = start_idx + i
            current_label = batch_labels[i]
            sample = batch_data[i:i+1]  # Keep as 2D array
            
            # Calculate a(i): average distance to points in same cluster
            same_cluster_mask = (all_labels == current_label) & (np.arange(len(all_labels)) != global_i)
            n_same = np.sum(same_cluster_mask)
            
            if n_same > 0:
                same_cluster_data = all_data[same_cluster_mask]
                correlations = np.abs(np.dot(sample, same_cluster_data.T)).flatten()
                distances = 1 - correlations
                a_i = np.mean(distances)
            else:
                a_i = 0.0
            
            # Calculate b(i): minimum average distance to points in other clusters
            b_i = np.inf
            
            for other_label in unique_labels:
                if other_label != current_label:
                    other_cluster_mask = all_labels == other_label
                    n_other = np.sum(other_cluster_mask)
                    
                    if n_other > 0:
                        other_cluster_data = all_data[other_cluster_mask]
                        correlations = np.abs(np.dot(sample, other_cluster_data.T)).flatten()
                        distances = 1 - correlations
                        mean_dist = np.mean(distances)
                        b_i = min(b_i, mean_dist)
            
            # Calculate silhouette coefficient for this sample
            if b_i == np.inf:
                s_i = 0.0
            else:
                max_val = max(a_i, b_i)
                if max_val > 0:
                    s_i = (b_i - a_i) / max_val
                else:
                    s_i = 0.0
            
            batch_scores[i] = s_i
        
        return batch_scores

    @staticmethod
    def _compute_efficient_silhouette(data_norm: np.ndarray, labels: np.ndarray, 
                                      unique_labels: np.ndarray) -> float:
        """
        Compute silhouette score efficiently for small to moderate datasets.
        """
        n_samples = data_norm.shape[0]
        silhouette_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            current_label = labels[i]
            sample = data_norm[i:i+1]  # Keep as 2D array
            
            # Calculate a(i): average distance to points in same cluster
            same_cluster_mask = (labels == current_label) & (np.arange(n_samples) != i)
            n_same = np.sum(same_cluster_mask)
            
            if n_same > 0:
                same_cluster_data = data_norm[same_cluster_mask]
                # Compute correlations in batches if needed
                if n_same > 5000:
                    correlations = []
                    batch_size = 1000
                    for start in range(0, n_same, batch_size):
                        end = min(start + batch_size, n_same)
                        batch_corr = np.abs(np.dot(sample, same_cluster_data[start:end].T)).flatten()
                        correlations.extend(batch_corr)
                    correlations = np.array(correlations)
                else:
                    correlations = np.abs(np.dot(sample, same_cluster_data.T)).flatten()
                
                distances = 1 - correlations
                a_i = np.mean(distances)
            else:
                a_i = 0.0
            
            # Calculate b(i): minimum average distance to points in other clusters
            b_i = np.inf
            
            for other_label in unique_labels:
                if other_label != current_label:
                    other_cluster_mask = labels == other_label
                    n_other = np.sum(other_cluster_mask)
                    
                    if n_other > 0:
                        other_cluster_data = data_norm[other_cluster_mask]
                        # Compute correlations in batches if needed
                        if n_other > 5000:
                            correlations = []
                            batch_size = 1000
                            for start in range(0, n_other, batch_size):
                                end = min(start + batch_size, n_other)
                                batch_corr = np.abs(np.dot(sample, other_cluster_data[start:end].T)).flatten()
                                correlations.extend(batch_corr)
                            correlations = np.array(correlations)
                        else:
                            correlations = np.abs(np.dot(sample, other_cluster_data.T)).flatten()
                        
                        distances = 1 - correlations
                        mean_dist = np.mean(distances)
                        b_i = min(b_i, mean_dist)
            
            # Calculate silhouette coefficient for this sample
            if b_i == np.inf:
                s_i = 0.0
            else:
                max_val = max(a_i, b_i)
                if max_val > 0:
                    s_i = (b_i - a_i) / max_val
                else:
                    s_i = 0.0
            
            silhouette_scores[i] = s_i
        
        return np.mean(silhouette_scores)

    @staticmethod
    def compute_custom_davies_bouldin(data: np.ndarray, labels: np.ndarray,
                                      maps: np.ndarray) -> float:
        """
        Compute Davies-Bouldin Index for polarity-invariant microstate clustering.

        The Davies-Bouldin Index is defined as:
        DBI = (1/K) * Σ(i=1 to K) max_j≠i[(S_i + S_j) / d_ij]

        Where:
        - K is the number of clusters
        - S_i is the average within-cluster distance for cluster i
        - d_ij is the distance between cluster centers i and j

        Lower values indicate better clustering (minimum value is 0).
        This implementation uses correlation-based distances for polarity invariance.

        Parameters
        ----------
        data : np.ndarray
            Data matrix of shape (n_channels, n_samples)
        labels : np.ndarray
            Cluster labels for each sample
        maps : np.ndarray
            Cluster centers of shape (n_clusters, n_channels)

        Returns
        -------
        float
            Davies-Bouldin Index (lower is better)
        """
        n_channels, n_samples = data.shape
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
                cluster_center = maps_norm[i:i + 1]

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
                        ratio = (within_cluster_scatter[i] + within_cluster_scatter[j]) / between_cluster_dist[i, j]
                        max_ratio = max(max_ratio, ratio)
                    else:
                        # If centers are nearly identical (shouldn't happen in practice)
                        # Assign a high penalty
                        max_ratio = max(max_ratio, 10.0)

            db_scores[i] = max_ratio

        # Step 4: Return average of maximum ratios
        return np.mean(db_scores)


# ============================================================================
# Worker Thread Classes
# ============================================================================

class OptimizedOptimizerWorker(QThread):
    """Optimized worker thread that computes all metrics for each K using modified K-means"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(dict)  # results
    error = pyqtSignal(str)

    def __init__(self, optimizer, methods_to_run, parameters=None):
        super().__init__()
        self.optimizer = optimizer
        self.methods_to_run = methods_to_run
        self.parameters = parameters or {}
        self.results = {}

    def run(self):
        """Run optimization by computing all metrics for each K value using modified K-means"""
        try:
            # Initialize results storage
            k_values = list(range(self.optimizer.kmin, self.optimizer.kmax + 1))

            # Validate k_values generation
            if not k_values:
                raise ValueError(f"No K values generated! kmin={self.optimizer.kmin}, kmax={self.optimizer.kmax}")

            if len(k_values) != (self.optimizer.kmax - self.optimizer.kmin + 1):
                raise ValueError(
                    f"K values mismatch! Expected {self.optimizer.kmax - self.optimizer.kmin + 1} values, got {len(k_values)}")

            metrics_data = {method: [] for method in self.methods_to_run}

            print(f"\n{'=' * 60}")
            print(f"MICROSTATE CLUSTERING ANALYSIS (Modified K-means)")
            print(f"{'=' * 60}")
            print(f"K range: {self.optimizer.kmin} to {self.optimizer.kmax}")
            print(f"Methods: {', '.join(self.methods_to_run)}")
            print(f"Using polarity-independent modified K-means algorithm")
            print(f"Parameters:")
            for method, param in self.parameters.items():
                print(f"  {method}: {param}")
            print(
                f"Total computations: {len(k_values)} × {len(self.methods_to_run)} = {len(k_values) * len(self.methods_to_run)}")
            print(f"{'=' * 60}")

            # For each K value, compute clustering once and calculate all metrics
            total_steps = len(k_values)

            for idx, k in enumerate(k_values):
                # Update progress
                current_step = idx + 1
                progress_percent = (current_step / total_steps) * 100
                self.progress.emit(
                    int(progress_percent),
                    100,
                    f"Computing all metrics for k={k} ({current_step}/{total_steps}) using modified K-means"
                )

                print(f"\nProcessing k={k} ({current_step}/{total_steps}) with modified K-means...")

                # Perform clustering once for this K using modified K-means
                try:
                    clustering_result = self.optimizer._perform_single_clustering(k)

                    # Compute all metrics on this clustering result
                    for method in self.methods_to_run:
                        try:
                            score = self._compute_metric(method, clustering_result, k)
                            metrics_data[method].append(score)
                            print(f"  {method}: {score:.4f}")
                        except Exception as e:
                            print(f"  Error computing {method} for k={k}: {str(e)}")
                            metrics_data[method].append(np.nan)

                except Exception as e:
                    print(f"  Error in modified K-means clustering for k={k}: {str(e)}")
                    for method in self.methods_to_run:
                        metrics_data[method].append(np.nan)

            print(f"\n{'=' * 60}")
            print("CREATING RESULTS...")
            print(f"{'=' * 60}")

            # Process results for each method
            for method in self.methods_to_run:
                scores = metrics_data[method]

                # Filter out NaN values for optimal K finding
                valid_scores = [(k, s) for k, s in zip(k_values, scores) if not np.isnan(s)]

                if valid_scores:
                    valid_k_values, valid_scores_list = zip(*valid_scores)
                    optimal_k = self._find_optimal_k(method, list(valid_k_values), list(valid_scores_list))
                else:
                    optimal_k = k_values[0]

                # Create result object with all scores (including NaN)
                result = OptimizationResult(
                    method_name=self._get_method_display_name(method),
                    k_values=k_values,
                    scores=scores,
                    optimal_k=optimal_k,
                    higher_is_better=self._is_higher_better(method)
                )

                self.results[method] = {
                    'method': method,
                    'result': result,
                    'optimal_k': optimal_k,
                    'original_optimal_k': optimal_k,
                    'k_values': k_values,
                    'scores': scores,
                    'threshold': self.parameters.get(method, None)
                }

                threshold_info = f" (threshold: {self.parameters.get(method)}%)" if method in ['gev', 'res'] else ""
                print(f"{self._get_method_display_name(method)}: Optimal k = {optimal_k}{threshold_info}")

            # Final progress update
            self.progress.emit(100, 100, "All modified K-means computations complete")
            self.finished.emit(self.results)

        except Exception as e:
            import traceback
            error_msg = f"Error in optimization: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)

    def _compute_metric(self, method, clustering_result, k):
        """Compute a specific metric on the clustering result"""
        metric_map = {
            'gev': 'gev',
            'res': 'residual_variance',
            'sil': 'silhouette',
            'ch': 'calinski_harabasz',
            'db': 'davies_bouldin',
            'cv': 'cv_score',
            'gs': 'gap_statistic'
        }

        metric_key = metric_map.get(method)
        if metric_key and metric_key in clustering_result:
            return clustering_result[metric_key]
        else:
            raise ValueError(f"Metric {method} not found in clustering result")

    def _find_optimal_k(self, method, k_values, scores):
        """Find optimal K based on method and scores"""
        if not scores or not k_values:
            return k_values[0] if k_values else 2

        # Handle methods based on optimization direction
        if method == 'sil' or method == 'ch' or method == 'gs':
            # Higher is better - find maximum
            max_idx = np.argmax(scores)
            return k_values[max_idx]
        elif method == 'db':
            # Lower is better - find minimum
            min_idx = np.argmin(scores)
            return k_values[min_idx]
        elif method == 'gev':
            # Global Explained Variance - use elbow method
            threshold = self.parameters.get(method, 5.0)
            return self._find_elbow_point(k_values, scores, threshold, higher_is_better=True)
        elif method == 'res':
            # Residual Variance - use elbow method
            threshold = self.parameters.get(method, 5.0)
            return self._find_elbow_point(k_values, scores, threshold, higher_is_better=False)
        elif method == 'cv':
            # Cross validation - typically higher is better
            max_idx = np.argmax(scores)
            return k_values[max_idx]
        else:
            # Default to first k value
            return k_values[0]

    def _find_elbow_point(self, k_values, scores, threshold, higher_is_better=True):
        """Find elbow point using threshold for percentage change"""
        if len(scores) < 2:
            return k_values[0]

        print(f"\n  Finding elbow point with threshold {threshold}%:")

        for i in range(1, len(scores)):
            # Calculate percentage change
            if scores[i - 1] != 0:
                change_percent = abs((scores[i] - scores[i - 1]) / scores[i - 1]) * 100
            else:
                change_percent = 100.0

            print(f"    K={k_values[i - 1]} to K={k_values[i]}: {change_percent:.2f}% change")

            # Check if improvement is below threshold
            if change_percent < threshold:
                print(f"    Elbow detected at K={k_values[i]} (change {change_percent:.2f}% < threshold {threshold}%)")
                return k_values[i]

        # If no elbow found, return the last K
        print(f"    No elbow found with threshold {threshold}%, returning last K={k_values[-1]}")
        return k_values[-1]

    def _is_higher_better(self, method):
        """Determine if higher scores are better for the method"""
        return method in ['gev', 'sil', 'ch', 'gs', 'cv']

    def _get_method_display_name(self, method):
        """Get display name for method"""
        name_map = {
            'gev': 'Elbow - Global Explained Variance',
            'res': 'Elbow - Residual Variance',
            'sil': 'Silhouette Method',
            'ch': 'Calinski-Harabasz Method',
            'db': 'Davies-Bouldin Method',
            'cv': 'Cross Validation',
            'gs': 'Gap Statistic'
        }
        return name_map.get(method, method)


# ============================================================================
# Extended Optimizer Class
# ============================================================================

class OptimizedMicrostateClustererOptimizer(ClustererOptimizer):
    """Extended optimizer that uses modified K-means for microstate clustering"""

    def __init__(self, maps2use, min_dist=None, n_inits=10, kmin=2, kmax=10,
                 preprocessed_data_path=None, extension=None, datatype=None,
                 tolerance=1e-6, max_iter=500, batch_size=None):
        super().__init__(maps2use, min_dist, n_inits, kmin, kmax,
                         preprocessed_data_path, extension, datatype, tolerance, max_iter)

        # Prepare data in the correct format for modified K-means (n_channels, n_samples)
        if self.maps2use.shape[0] > self.maps2use.shape[1]:
            self.eeg_data = self.maps2use.T
            print(f"Transposed data from {self.maps2use.shape} to {self.eeg_data.shape}")
        else:
            self.eeg_data = self.maps2use
            print(f"Data already in correct format: {self.eeg_data.shape}")

        # Validate data size and provide memory warnings
        n_channels, n_samples = self.eeg_data.shape
        self._validate_data_size(n_channels, n_samples)

        self.batch_size = batch_size

        # Initialize metrics calculator
        self.metrics_calculator = MicrostateClusteringMetrics()

        print(f"Initialized MicrostateClustererOptimizer:")
        print(f"  EEG data shape: {self.eeg_data.shape} (channels × samples)")
        print(f"  K range: {kmin} to {kmax}")
        print(f"  Using modified K-means (polarity-independent)")
        print(f"  Memory usage strategy: {self._get_memory_strategy(n_samples)}")

    def _validate_data_size(self, n_channels: int, n_samples: int):
        """
        Validate data size and provide warnings about potential memory issues.
        """
        # Calculate approximate memory requirements
        memory_gb = self._estimate_memory_usage(n_samples)
        
        print(f"\nData validation:")
        print(f"  Channels: {n_channels}")
        print(f"  Samples: {n_samples:,}")
        print(f"  Estimated peak memory usage: {memory_gb:.2f} GB")
        
        # Provide warnings based on data size
        if n_samples > 100000:
            print(f"\n⚠️  WARNING: Very large dataset detected!")
            print(f"  - {n_samples:,} samples may cause memory issues")
            print(f"  - Estimated memory needed: {memory_gb:.2f} GB")
            print(f"  - Using aggressive sampling and batch processing")
            print(f"  - Consider using fewer samples or more RAM")
            
        elif n_samples > 50000:
            print(f"\n⚠️  WARNING: Large dataset detected!")
            print(f"  - {n_samples:,} samples will use sampling for efficiency")
            print(f"  - Estimated memory needed: {memory_gb:.2f} GB")
            print(f"  - Processing will be slower but memory-safe")
            
        elif n_samples > 10000:
            print(f"\nℹ️  INFO: Medium dataset detected")
            print(f"  - {n_samples:,} samples will use batch processing")
            print(f"  - Estimated memory needed: {memory_gb:.2f} GB")
            print(f"  - Processing optimized for memory efficiency")
            
        else:
            print(f"\nℹ️  INFO: Small dataset - optimal for full computation")
            print(f"  - {n_samples:,} samples can be processed efficiently")

        # Check if this might be raw data instead of GFP peaks
        if n_samples > 50000:
            print(f"\n🔍 DIAGNOSTIC: Very high sample count detected")
            print(f"  - Are you using raw EEG timepoints instead of GFP peaks?")
            print(f"  - Microstate analysis typically uses GFP peaks (~hundreds to thousands)")
            print(f"  - Consider using only GFP peak timepoints for better results")

    def _estimate_memory_usage(self, n_samples: int) -> float:
        """
        Estimate peak memory usage in GB for the clustering process.
        """
        # Main memory consumers:
        # 1. Distance matrices for silhouette (avoided with optimization)
        # 2. Correlation matrices in batch processing
        # 3. Data copies and intermediate arrays
        
        # Conservative estimate based on optimized algorithms
        if n_samples > 50000:
            # Using sampling approach
            estimated_mb = (n_samples * 64 * 8) / (1024**2)  # Basic data storage
            estimated_mb += 100  # Overhead for sampling and processing
        elif n_samples > 10000:
            # Using batch processing
            estimated_mb = (n_samples * 128 * 8) / (1024**2)  # Data + batch overhead
        else:
            # Full computation but optimized
            estimated_mb = (n_samples * 256 * 8) / (1024**2)  # Data + correlation matrices
        
        return estimated_mb / 1024  # Convert to GB

    def _get_memory_strategy(self, n_samples: int) -> str:
        """Get the memory strategy description based on sample count."""
        if n_samples > 50000:
            return "Aggressive sampling (memory-critical)"
        elif n_samples > 10000:
            return "Stratified sampling (memory-efficient)"
        elif n_samples > 1000:
            return "Batch processing (balanced)"
        else:
            return "Full computation (optimal)"

    def _perform_single_clustering(self, k):
        """Perform modified K-means clustering for a single K value"""
        print(f"    Performing modified K-means clustering for k={k}")
        start_time = time.time()

        # Initialize the MicrostateClusterer for this K
        clusterer = MicrostateClusterer(
            n_states=k,
            batch_size=self.batch_size,
            n_inits=1,
            max_iter=self.max_iter,
            tolerance=self.tolerance
        )

        n_channels, n_samples = self.eeg_data.shape

        # Run multiple initializations and keep the best result
        best_maps = None
        best_residual = np.inf
        best_labels = None
        best_gev = 0.0

        print(f"      Running {self.n_inits} initializations...")

        for init_attempt in range(self.n_inits):
            try:
                # Random initialization of maps
                initial_maps = np.random.randn(k, n_channels)
                for i in range(k):
                    initial_maps[i] /= np.linalg.norm(initial_maps[i])

                # Run modified K-means
                maps, residual = clusterer.modified_kmeans(
                    self.eeg_data,
                    initial_maps,
                    verbose=False
                )

                # Calculate GEV for this result
                gev = clusterer.compute_gev(self.eeg_data, maps)

                # Keep the best result based on GEV
                if gev > best_gev:
                    best_gev = gev
                    best_residual = residual
                    best_maps = maps.copy()

                    # Calculate final segmentation
                    activation = best_maps.dot(self.eeg_data)
                    best_labels = np.argmax(np.abs(activation), axis=0)

            except Exception as e:
                print(f"        Warning: Initialization {init_attempt + 1} failed: {str(e)}")
                continue

        if best_maps is None:
            raise RuntimeError(f"All {self.n_inits} initializations failed for k={k}")

        # Create comprehensive clustering result
        clustering_result = {
            'k': k,
            'labels': best_labels,
            'centers': best_maps,
            'data': self.maps2use,
            'eeg_data': self.eeg_data,
            'residual': best_residual,
            'n_iter': None,
        }

        # Compute all metrics efficiently
        self._compute_all_metrics(clustering_result, k, best_labels, best_gev, best_residual)

        elapsed = time.time() - start_time
        print(f"        Completed in {elapsed:.2f}s - GEV: {clustering_result['gev']:.4f}")

        # Show cluster distribution
        unique_labels, counts = np.unique(best_labels, return_counts=True)
        total_samples = len(best_labels)
        print(f"        Cluster distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / total_samples) * 100
            print(f"          State {label}: {count} samples ({percentage:.1f}%)")

        return clustering_result

    def _compute_all_metrics(self, clustering_result, k, best_labels, best_gev, best_residual):
        """Compute all metrics for the clustering result"""
        # 1. Global Explained Variance (PRIMARY metric for microstates)
        clustering_result['gev'] = best_gev

        # 2. Residual Variance
        clustering_result['residual_variance'] = best_residual

        # 3. Prepare data matrix for metrics: shape (n_samples, n_features)
        #    self.eeg_data is (n_channels × n_samples), so transpose to (n_samples × n_channels)
        sample_features = self.eeg_data.T

        # 4. Silhouette Score (if k > 1) - CUSTOM IMPLEMENTATION
        if k > 1:
            clustering_result['silhouette'] = self.metrics_calculator.compute_custom_silhouette(
                sample_features, best_labels
            )
        else:
            clustering_result['silhouette'] = 0.0

        # 5. Calinski-Harabasz Index
        try:
            # Use custom polarity-invariant CH implementation from parent class
            maps = clustering_result.get('centers')  # shape: (k, n_channels)

            # Ensure that required data are present
            if maps is not None and maps.shape[0] == k:
                ch_score = self._compute_custom_calinski_harabasz(
                    data=self.eeg_data,  # (n_channels × n_samples)
                    maps=maps,  # (k × n_channels)
                    segmentation=best_labels,  # (n_samples,)
                    k=k
                )
            else:
                raise ValueError("Cluster centers missing or have unexpected shape for CH computation")

            clustering_result['calinski_harabasz'] = ch_score
        except Exception as e:
            print(f"        Warning: CH computation failed for k={k}: {str(e)}")
            clustering_result['calinski_harabasz'] = 0.0

        # 6. Davies-Bouldin Index (if k > 1) - CUSTOM IMPLEMENTATION
        if k > 1:
            clustering_result['davies_bouldin'] = self.metrics_calculator.compute_custom_davies_bouldin(
                self.eeg_data,  # (n_channels × n_samples)
                best_labels,  # (n_samples,)
                clustering_result.get('centers')  # (k × n_channels)
            )
        else:
            clustering_result['davies_bouldin'] = 0.0

        # 7. Cross Validation Score (use GEV as proxy for microstates)
        clustering_result['cv_score'] = best_gev

        # 8. Gap Statistic (microstate-adapted version)
        clustering_result['gap_statistic'] = self._calculate_gap_statistic_microstate(
            clustering_result, k
        )

    def _compute_silhouette_score(self, data, labels):
        """Compute custom silhouette score for polarity-invariant clustering"""
        return self.metrics_calculator.compute_custom_silhouette(data, labels)

    def _compute_davies_bouldin_score(self, data, labels):
        """Compute custom Davies-Bouldin score for polarity-invariant clustering"""
        # This method should not be called anymore, but kept for compatibility
        # The actual computation is done in _compute_all_metrics
        return 0.0

    def _calculate_gap_statistic_microstate(self, clustering_result, k):
        """Calculate gap statistic adapted for microstate analysis"""
        gev = clustering_result['gev']
        random_gev_estimate = max(0.1, 1.0 / k)

        if gev > 0 and random_gev_estimate > 0:
            gap = np.log(gev) - np.log(random_gev_estimate)
        else:
            gap = 0.0

        return max(0.0, gap)

    def get_microstate_maps(self, k):
        """Get the final microstate maps for a specific k value"""
        result = self._perform_single_clustering(k)
        return result['centers']

    def compute_microstate_metrics(self, k, verbose=True):
        """Compute comprehensive microstate-specific metrics for a given k"""
        result = self._perform_single_clustering(k)

        metrics = {
            'k': k,
            'gev': result['gev'],
            'residual_variance': result['residual_variance'],
            'silhouette': result['silhouette'],
            'calinski_harabasz': result['calinski_harabasz'],
            'davies_bouldin': result['davies_bouldin'],
            'microstate_maps': result['centers'],
            'segmentation': result['labels']
        }

        if verbose:
            print(f"\nMicrostate Metrics for k={k}:")
            print(f"  Global Explained Variance: {metrics['gev']:.4f}")
            print(f"  Residual Variance: {metrics['residual_variance']:.6f}")
            print(f"  Silhouette Score: {metrics['silhouette']:.4f} (custom polarity-invariant)")
            print(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.2f} (custom polarity-invariant)")
            print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f} (custom polarity-invariant)")

        return metrics


# ============================================================================
# Main Window Class
# ============================================================================

class OptimizerVisualizationWindow(QMainWindow):
    """
    Window for manual inspection of microstate clustering optimization results.
    Uses modified K-means algorithm that ignores polarity shifts with persistent results.
    """

    def __init__(self, context, comet_instance, parent=None):
        """Initialize the OptimizerVisualizationWindow with enhanced persistence support."""
        super(OptimizerVisualizationWindow, self).__init__(parent)
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

    # ========================================================================
    # Initialization Methods
    # ========================================================================

    def _init_data_structures(self):
        """Initialize core data structures"""
        self.optimizer = None
        self.results_cache = {}
        self.worker = None
        self.all_methods_complete = False
        self.last_used_parameters = {}
        self.current_font_family = 'Arial'
        self.current_font_size = 'Large'

    def _setup_ui(self):
        """Setup the UI components"""
        # Load the UI
        self.ui = uic.loadUi(self.context.get_resource("OptimizerVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Exploring the number of microstate maps (Modified K-means)")

        # Set up the matplotlib figure
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.ui.Figure_Layout.addWidget(self.canvas)

    def _setup_font_system(self):
        """Setup font menu action groups"""
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
        """Setup all signal-slot connections"""
        # Main controls
        self.ui.optimizer_combobox.activated.connect(self.optimizer_visualization_controller)
        self.ui.optimizer_button.clicked.connect(self.run_all_analyses)
        self.ui.visualize_button.clicked.connect(self.visualize_selected_method)
        self.ui.export_figure_image_button.triggered.connect(self.export_plot)

        # Threshold input
        self.ui.optimizer_stopping_threshold_input.returnPressed.connect(self._on_threshold_changed)

        # Font actions
        self.ui.actionArial.triggered.connect(lambda: self._set_font_family('Arial'))
        self.ui.actionCalibri.triggered.connect(lambda: self._set_font_family('Calibri'))
        self.ui.actionTimes_New_Roman.triggered.connect(lambda: self._set_font_family('Times New Roman'))

        self.ui.actionSmall.triggered.connect(lambda: self._set_font_size('Small'))
        self.ui.actionMedium.triggered.connect(lambda: self._set_font_size('Medium'))
        self.ui.actionLarge.triggered.connect(lambda: self._set_font_size('Large'))
        self.ui.actionX_Large.triggered.connect(lambda: self._set_font_size('X-Large'))

    def _initialize_state(self):
        """Initialize the window state, loading parameters and previous results"""
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
        """Setup UI when we have previous results"""
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
        """Setup UI when we don't have previous results"""
        # Disable visualization initially
        self._disable_visualization_ui()

        # Add status message
        self.ui.statusbar.showMessage("Ready - Using polarity-independent modified K-means algorithm")

    def _get_k_range_text(self) -> str:
        """Get K range text from results"""
        if not self.results_cache:
            return ""

        first_result = next(iter(self.results_cache.values()))
        k_values = first_result.get('k_values', [])
        if k_values:
            return f" (K: {min(k_values)}-{max(k_values)})"
        return ""

    # ========================================================================
    # Parameter Loading Methods
    # ========================================================================

    def _load_comet_parameters(self):
        """Load parameters from COMET instance"""
        self.preprocessed_data_path = self.comet.preprocessed_data_path
        self.extension = self.comet.extension
        self.datatype = self.comet.datatype
        self.use_percentages = self.comet.use_percentages
        self.min_distance_size = getattr(self.comet, 'min_distance_size', None)
        self.number_of_repeats = getattr(self.comet, 'number_of_repeats', 10)
        self.clustering_tolerance = self.comet.clustering_tolerance
        self.max_iterations = self.comet.max_iterations
        self.batch_size = getattr(self.comet, 'batch_size', None)

    # ========================================================================
    # Results Loading and Saving Methods
    # ========================================================================

    def _load_previous_results(self):
        """Load previous optimization results from COMET configuration and update GUI."""
        try:
            previous_results = self.comet.load_optimization_results()
            if previous_results:
                self.results_cache = previous_results
                print(f"Successfully loaded previous optimization results for {len(previous_results)} methods")

                # Log the loaded results
                self._log_loaded_results(previous_results)

                # Update GUI to reflect loaded results
                self._update_gui_for_loaded_results()

            else:
                self.results_cache = {}
        except Exception as e:
            print(f"Error loading previous optimization results: {str(e)}")
            self.results_cache = {}

    def _log_loaded_results(self, results: Dict[str, Any]):
        """Log loaded optimization results"""
        print("Loaded optimization results:")
        for method_code, results in results.items():
            method_name = self._get_method_name(method_code)
            optimal_k = results['optimal_k']
            k_range = f"{min(results['k_values'])}-{max(results['k_values'])}"
            threshold_info = f" (threshold: {results['threshold']}%)" if results.get('threshold') else ""
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
        k_values = first_result.get('k_values', [])

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

        for method_code in self.results_cache.keys():
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
        preferred_order = ['gev', 'res', 'sil', 'ch', 'db', 'cv', 'gs']

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
            threshold = results.get('threshold')
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
            self.ui.statusbar.showMessage(f"{method_count} optimization results available - Ready to visualize!")

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
        """Update UI based on selected optimization method - enhanced for loaded results."""
        optimizer_method = self.ui.optimizer_combobox.currentText()

        if not optimizer_method:
            return

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

        # Set values based on loaded results or defaults
        method_code = self._get_method_code(optimizer_method)

        # First, try to get value from loaded results
        if method_code in self.results_cache:
            cached_result = self.results_cache[method_code]
            threshold = cached_result.get('threshold')

            if threshold is not None:
                self.ui.optimizer_stopping_threshold_input.setText(str(threshold))
                print(f"Set threshold from loaded results: {threshold}")
            else:
                # Set default values if no threshold in loaded results
                self._set_default_threshold(optimizer_method)

        # Second, try last used parameters
        elif method_code in self.last_used_parameters:
            # Restore last used value
            self.ui.optimizer_stopping_threshold_input.setText(str(self.last_used_parameters[method_code]))
            print(f"Set threshold from last used: {self.last_used_parameters[method_code]}")

        else:
            # Set default values
            self._set_default_threshold(optimizer_method)

    def _set_default_threshold(self, optimizer_method: str):
        """Set default threshold values for methods."""
        if optimizer_method == 'Gap Statistic':
            self.ui.optimizer_stopping_threshold_input.setText('10')
        elif optimizer_method == 'Cross Validation':
            self.ui.optimizer_stopping_threshold_input.setText('5')
        elif optimizer_method in ['Elbow - Global Explained Variance', 'Elbow - Residual Variance']:
            self.ui.optimizer_stopping_threshold_input.setText('5')

    def _on_threshold_changed(self):
        """Handle threshold change by automatically updating visualization"""
        if self.all_methods_complete and self.ui.visualize_button.isEnabled():
            # Only update if we have results and the current method uses thresholds
            method_code = self._get_method_code(self.ui.optimizer_combobox.currentText())
            if method_code in ['gev', 'res']:
                self.visualize_selected_method()

    # ========================================================================
    # Font Management Methods
    # ========================================================================

    def _set_font_family(self, family: str):
        """Update font family"""
        self.current_font_family = family
        if hasattr(self, 'current_plot_method'):
            self.visualize_selected_method()

    def _set_font_size(self, size: str):
        """Update font size"""
        self.current_font_size = size
        if hasattr(self, 'current_plot_method'):
            self.visualize_selected_method()

    def _get_font_size(self) -> int:
        """Get numerical font size based on current setting"""
        sizes = {
            'Small': 10,
            'Medium': 12,
            'Large': 14,
            'X-Large': 16
        }
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
                'Overwrite Previous Results?',
                'Previous optimization results exist. Do you want to re-run the analysis and overwrite them?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
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

        print(f"\nStarting microstate optimization with modified K-means")
        print(f"K range: {kmin} to {kmax}")
        print(f"Total K values to process: {kmax - kmin + 1}")

        # Get all methods to run
        all_methods = ['gev', 'res', 'sil', 'ch', 'db', 'cv', 'gs']

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
        """Initialize the optimizer with data using modified K-means"""
        # Generate maps and peaks
        self.maps2use, self.peaks2use = DataInitializer().generate_maps_and_peaks(
            self.preprocessed_data_path,
            self.extension,
            self.datatype,
            self.use_percentages,
            self.min_distance_size
        )

        print(f"Generated data for optimization:")
        print(f"  Maps shape: {self.maps2use.shape}")
        print(f"  Peaks count: {len(self.peaks2use) if self.peaks2use is not None else 'None'}")

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
            batch_size=self.batch_size
        )

        print(f"Microstate optimizer initialized with K range: {kmin} to {kmax}")
        print("Using modified K-means algorithm (polarity-independent)")
        print("Using custom Silhouette and Davies-Bouldin implementations")

    def _get_method_parameters(self) -> Optional[Dict[str, float]]:
        """Get parameters from UI for methods that need them"""
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
        parameters['gev'] = threshold_value
        parameters['res'] = threshold_value
        parameters['cv'] = int(threshold_value) if threshold_value.is_integer() else 5
        parameters['gs'] = int(threshold_value) if threshold_value.is_integer() else 10

        # Store the parameters used
        self.last_used_parameters = parameters.copy()

        print(f"Using parameters: {parameters}")
        print("Algorithm: Modified K-means (polarity-independent)")

        return parameters

    def _update_progress(self, current: int, total: int, message: str):
        """Update progress bar"""
        self.ui.optimizer_progressbar.setValue(current)
        self.ui.statusbar.showMessage(message)
        QApplication.processEvents()  # Keep UI responsive

    def _on_all_analyses_finished(self, all_results: Dict[str, Any]):
        """Handle completion of all analyses and save results."""
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
        self.ui.statusbar.showMessage("All modified K-means optimization methods completed and saved!")

        # Update visualization controller for first method
        self.optimizer_visualization_controller()

        print("\n" + "=" * 60)
        print("ALL MICROSTATE OPTIMIZATION METHODS COMPLETED AND SAVED!")
        print("Using modified K-means (polarity-independent)")
        print("Using custom metrics for Silhouette and Davies-Bouldin")
        print("=" * 60)

        # Display detailed results
        self._display_optimization_summary(all_results)

    def _display_optimization_summary(self, results: Dict[str, Any]):
        """Display summary of optimization results"""
        for method_code, results in results.items():
            method_name = self._get_method_name(method_code)
            optimal_k = results['optimal_k']
            k_values = results['k_values']
            scores = results['scores']
            threshold = results.get('threshold', None)

            print(f"\n{method_name}:")
            print(f"  Optimal k = {optimal_k}")
            if threshold is not None and method_code in ['gev', 'res']:
                print(f"  Threshold used: {threshold}%")
            print(f"  K values tested: {k_values[0]} to {k_values[-1]} ({len(k_values)} values)")

            # Filter out NaN scores for min/max calculation
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                print(f"  Score range: {min(valid_scores):.4f} to {max(valid_scores):.4f}")
            else:
                print(f"  Score range: All scores invalid")

            # Show scores for each K
            print("  Scores by K:")
            for k, score in zip(k_values, scores):
                marker = " <-- OPTIMAL" if k == optimal_k else ""
                score_str = f"{score:.4f}" if not np.isnan(score) else "NaN"
                print(f"    k={k}: {score_str}{marker}")

        print("\n" + "=" * 60)
        print("NOTE: All results computed using modified K-means algorithm")
        print("This ensures polarity-independent microstate identification")
        print("Silhouette and Davies-Bouldin use custom polarity-invariant implementations")
        print("Results have been saved and will be available next time you load this study")

    def _on_optimization_error(self, error_msg: str):
        """Handle optimization error"""
        print(f"Modified K-means optimization error: {error_msg}")
        self.ui.optimizer_button.setEnabled(True)
        self.ui.optimizer_progressbar.setValue(0)
        self.ui.statusbar.showMessage(f"Error: {error_msg}")

    # ========================================================================
    # Visualization Methods
    # ========================================================================

    def visualize_selected_method(self):
        """Visualize the currently selected optimization method - enhanced for loaded results."""
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
            if method_code in ['gev', 'res']:
                self._handle_threshold_visualization(method_code)
            else:
                self.ui.statusbar.showMessage(f"Displaying {method_name} (Modified K-means)")

            self._display_results(self.results_cache[method_code])
        else:
            print(f"Cannot visualize: method {method_code} not in results cache")
            self.ui.statusbar.showMessage(f"No results available for {method_name}")

    def _handle_threshold_visualization(self, method_code: str):
        """Handle visualization for threshold-based methods"""
        try:
            current_threshold = float(self.ui.optimizer_stopping_threshold_input.text())
            if current_threshold <= 0:
                self.ui.statusbar.showMessage("Error: Threshold must be positive")
                return

            cached_threshold = self.results_cache[method_code].get('threshold', None)

            if cached_threshold is not None and current_threshold != cached_threshold:
                # Recalculate optimal K with new threshold
                self._recalculate_optimal_k_with_new_threshold(method_code, current_threshold)
                # Save the updated results
                self._save_results_to_comet()
                self.ui.statusbar.showMessage(f"Optimal K recalculated with threshold {current_threshold}%")
            else:
                method_name = self._get_method_name(method_code)
                self.ui.statusbar.showMessage(f"Displaying {method_name} (Modified K-means)")
        except ValueError:
            self.ui.statusbar.showMessage("Error: Invalid threshold value")
            return

    def _recalculate_optimal_k_with_new_threshold(self, method_code: str, new_threshold: float):
        """Recalculate optimal K for a method with a new threshold without re-running the analysis"""
        print(
            f"\nRecalculating optimal K for {self._get_method_name(method_code)} with new threshold: {new_threshold}%")

        # Get the existing results
        cached_results = self.results_cache[method_code]
        k_values = cached_results['k_values']
        scores = cached_results['scores']

        # Filter out NaN values for optimal K finding
        valid_scores = [(k, s) for k, s in zip(k_values, scores) if not np.isnan(s)]

        if valid_scores:
            valid_k_values, valid_scores_list = zip(*valid_scores)

            # Use the same elbow detection algorithm with the new threshold
            optimal_k = self._find_elbow_point_for_visualization(
                list(valid_k_values),
                list(valid_scores_list),
                new_threshold,
                higher_is_better=(method_code == 'gev')
            )
        else:
            optimal_k = k_values[0] if k_values else 2

        # Update the cached results with new optimal K and threshold
        cached_results['optimal_k'] = optimal_k
        cached_results['threshold'] = new_threshold
        cached_results['result'].optimal_k = optimal_k

        # Update the last used parameters
        self.last_used_parameters[method_code] = new_threshold

        print(f"New optimal K: {optimal_k} (was: {cached_results.get('original_optimal_k', 'unknown')})")

    def _find_elbow_point_for_visualization(self, k_values: List[int], scores: List[float],
                                            threshold: float, higher_is_better: bool = True) -> int:
        """Find elbow point using threshold for percentage change (for visualization updates)"""
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
                print(f"    Elbow detected at K={k_values[i]} (change {change_percent:.2f}% < threshold {threshold}%)")
                return k_values[i]

        print(f"    No elbow found with threshold {threshold}%, returning last K={k_values[-1]}")
        return k_values[-1]

    def _display_results(self, results: Dict[str, Any]):
        """Display optimization results"""
        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        result = results['result']
        threshold = results.get('threshold', None)
        method_code = results['method']

        # Get font settings early
        font_size = self._get_font_size()
        font_family = self.current_font_family

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

        # For elbow methods, show the threshold region and percentage changes
        if method_code in ['gev', 'res'] and threshold is not None:
            self._add_elbow_annotations(ax, result, threshold)

        # Highlight optimal k
        optimal_label = f'Optimal k = {result.optimal_k}'
        if threshold is not None and method_code in ['gev', 'res']:
            optimal_label += f' (threshold: {threshold}%)'

        ax.axvline(x=result.optimal_k, color='red', linestyle='--',
                   linewidth=2, label=optimal_label)

        # Styling with custom font
        ax.set_xlabel('Number of Clusters (k)', fontsize=font_size, fontfamily=font_family)
        ax.set_ylabel(self._get_ylabel(result.method_name), fontsize=font_size, fontfamily=font_family)

        # Update title to show algorithm and parameters
        title = self._create_plot_title(result, results, threshold, method_code)
        ax.set_title(title, fontsize=font_size + 2, fontweight='bold', fontfamily=font_family)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=font_size - 2, prop={'family': font_family})

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
        """Add elbow method annotations to the plot"""
        # Find where the elbow detection happens
        valid_indices = [i for i, score in enumerate(result.scores) if not np.isnan(score)]
        if len(valid_indices) > 1:
            elbow_found = False
            for i in range(1, len(valid_indices)):
                idx = valid_indices[i]
                prev_idx = valid_indices[i - 1]

                if result.scores[prev_idx] != 0:
                    change_percent = abs(
                        (result.scores[idx] - result.scores[prev_idx]) / result.scores[prev_idx]) * 100

                    # Add percentage change annotations
                    mid_k = (result.k_values[prev_idx] + result.k_values[idx]) / 2
                    mid_y = (result.scores[prev_idx] + result.scores[idx]) / 2

                    # Color based on whether it's above or below threshold
                    if change_percent < threshold:
                        color = 'red'
                        weight = 'bold'
                        if not elbow_found:
                            # Highlight the region where change is below threshold
                            ax.axvspan(result.k_values[prev_idx], result.k_values[idx],
                                       alpha=0.2, color='yellow',
                                       label=f'Below threshold ({threshold}%)')
                            elbow_found = True
                    else:
                        color = 'black'
                        weight = 'normal'

                    # Add text annotation
                    ax.annotate(f'{change_percent:.1f}%',
                                xy=(mid_k, mid_y),
                                xytext=(0, 10),
                                textcoords='offset points',
                                ha='center',
                                fontsize=self._get_font_size() - 4,
                                color=color,
                                weight=weight,
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='white',
                                          edgecolor=color,
                                          alpha=0.8))

    def _create_plot_title(self, result: Any, results: Dict[str, Any],
                           threshold: Optional[float], method_code: str) -> str:
        """Create the plot title"""
        title_parts = [result.method_name]

        # Add specific notes for custom implementations
        if method_code in ['sil', 'db']:
            title_parts[0] += " (Custom Polarity-Invariant)"
        else:
            title_parts[0] += " (Modified K-means)"

        title_parts.append(f"K range: {result.k_values[0]} to {result.k_values[-1]}")

        if threshold is not None and method_code in ['gev', 'res']:
            title_parts.append(f"Threshold: {threshold}%")

            # Show if threshold was changed after analysis
            original_optimal = results.get('original_optimal_k', None)
            if original_optimal and original_optimal != result.optimal_k:
                title_parts.append(f"(Original optimal K was {original_optimal})")

        return '\n'.join(title_parts)

    def _adjust_y_axis(self, ax, scores: List[float]):
        """Adjust y-axis for better visualization"""
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
        safe_method_name = current_method.replace(' - ', '_').replace(' ', '_').lower()
        default_name = f"microstate_optimization_{safe_method_name}.pdf"

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Microstate Optimization Plot",
            default_name,
            "PDF Files (*.pdf);;PNG Files (*.png);;JPG Files (*.jpg);;SVG Files (*.svg);;All Files (*)",
            options=options
        )
        if file_name:
            self._save_plot(file_name)

    def _save_plot(self, file_name: str):
        """Save the current plot to file"""
        try:
            # Get file extension
            extension = file_name.split('.')[-1].lower()

            # Set appropriate DPI for raster formats
            dpi = 300 if extension in ['png', 'jpg', 'jpeg'] else None

            # Save the figure
            self.figure.savefig(file_name, dpi=dpi, bbox_inches='tight')
            print(f"Microstate optimization plot exported successfully to: {file_name}")
            self.ui.statusbar.showMessage(f"Plot exported to: {file_name}")

        except Exception as e:
            error_msg = f"Error exporting plot: {str(e)}"
            print(error_msg)
            self.ui.statusbar.showMessage(error_msg)

    # ========================================================================
    # Public Interface Methods
    # ========================================================================

    def get_all_results_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of all computed results using modified K-means."""
        if not self.results_cache:
            return None

        summary = {
            'algorithm': 'Modified K-means (polarity-independent)',
            'metrics': 'Custom implementations for Silhouette and Davies-Bouldin',
            'results': {},
            'has_saved_results': True,
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        for method_code, results in self.results_cache.items():
            method_name = self._get_method_name(method_code)
            optimal_k = results['result'].optimal_k
            threshold = results.get('threshold', None)

            if threshold is not None and method_code in ['gev', 'res']:
                summary['results'][method_name] = {'optimal_k': optimal_k, 'threshold': threshold}
            else:
                summary['results'][method_name] = optimal_k

        return summary

    def get_microstate_maps_for_optimal_k(self, method: str = 'gev') -> Optional[np.ndarray]:
        """Get the microstate maps for the optimal K determined by the specified method"""
        if method in self.results_cache and self.optimizer:
            optimal_k = self.results_cache[method]['optimal_k']
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

    def _get_method_code(self, method_name: str) -> str:
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

    def _get_method_name(self, method_code: str) -> str:
        """Convert method code to display name"""
        name_map = {
            'gev': 'Elbow - Global Explained Variance',
            'res': 'Elbow - Residual Variance',
            'sil': 'Silhouette Method',
            'ch': 'Calinski-Harabasz Method',
            'db': 'Davies-Bouldin Method',
            'cv': 'Cross Validation',
            'gs': 'Gap Statistic'
        }
        return name_map.get(method_code, method_code)

    def _get_method_display_names(self) -> Dict[str, str]:
        """Get all method display names"""
        return {
            'gev': 'Elbow - Global Explained Variance',
            'res': 'Elbow - Residual Variance',
            'sil': 'Silhouette Method',
            'ch': 'Calinski-Harabasz Method',
            'db': 'Davies-Bouldin Method',
            'cv': 'Cross Validation',
            'gs': 'Gap Statistic'
        }

    def _get_ylabel(self, method_name: str) -> str:
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
