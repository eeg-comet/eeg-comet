import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances
import warnings

warnings.filterwarnings('ignore')

from clustering_utils.microstate_clusterer import MicrostateClusterer
from data_utils.data_initializer import DataInitializer


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    k_values: List[int]
    scores: List[float]
    optimal_k: int
    method_name: str
    higher_is_better: bool = True


class ClustererOptimizer:
    """
    Optimized clusterer for finding the optimal number of microstate clusters.

    This implementation provides:
    - Clean separation of concerns
    - Progress callback support for GUI updates
    - Both automatic (majority vote) and manual inspection modes
    - Thread-safe operation for worker threads
    """

    def __init__(self,
                 maps2use: np.ndarray,
                 min_dist: Optional[int] = None,
                 n_inits: int = 1,  # Single repeat per k
                 kmin: int = 2,
                 kmax: int = 10,
                 preprocessed_data_path: str = None,
                 extension: str = '.set',
                 datatype: str = 'raw',
                 tolerance: float = 1e-6,
                 max_iter: int = 500,
                 progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Initialize the clusterer optimizer.

        Parameters
        ----------
        maps2use : np.ndarray
            Data to cluster (n_channels x n_timepoints)
        min_dist : int, optional
            Minimum distance for peak detection
        n_inits : int
            Number of clustering initializations (set to 1 for single repeat)
        kmin : int
            Minimum number of clusters
        kmax : int
            Maximum number of clusters
        preprocessed_data_path : str
            Path to preprocessed data
        extension : str
            File extension
        datatype : str
            Data type
        tolerance : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
        progress_callback : callable, optional
            Callback for progress updates: callback(current, total, message)
        """
        self.maps2use = maps2use
        self.min_dist = min_dist
        self.n_inits = n_inits
        self.kmin = kmin
        self.kmax = kmax
        self.k_range = list(range(kmin, kmax + 1))
        self.preprocessed_data_path = preprocessed_data_path
        self.extension = extension
        self.datatype = datatype
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.progress_callback = progress_callback

        # Results storage
        self.results: Dict[str, OptimizationResult] = {}
        self._clustering_cache: Dict[int, dict] = {}

    def _update_progress(self, current: int, total: int, message: str):
        """Update progress if callback is provided"""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def _get_clustering_result(self, k: int) -> dict:
        """
        Get clustering result for k clusters, using cache if available.

        Returns
        -------
        dict
            Contains 'maps', 'gev', 'residual', 'segmentation'
        """
        if k in self._clustering_cache:
            return self._clustering_cache[k]

        # Initialize clusterer
        clusterer = MicrostateClusterer(
            n_states=k,
            n_inits=self.n_inits,
            max_iter=self.max_iter,
            tolerance=self.tolerance
        )

        # Initialize cluster centers
        data_initializer = DataInitializer()
        initial_maps = data_initializer.initialize_cluster_centers(
            maps2use=self.maps2use,
            n_states=k,
            initializer="Random"
        )

        # Perform clustering (modified k-means)
        maps, residual = clusterer.modified_kmeans(
            data=self.maps2use,
            initial_maps=initial_maps
        )

        # Compute GEV
        gev = clusterer.compute_gev(data=self.maps2use, maps=maps)

        # Get segmentation
        activation = maps.dot(self.maps2use)
        segmentation = np.argmax(np.abs(activation), axis=0)

        result = {
            'maps': maps,
            'gev': gev,
            'residual': residual,
            'segmentation': segmentation,
            'activation': activation
        }

        self._clustering_cache[k] = result
        return result

    def compute_elbow_gev(self) -> OptimizationResult:
        """Compute elbow method using Global Explained Variance"""
        scores = []
        total_steps = len(self.k_range)

        for i, k in enumerate(self.k_range):
            self._update_progress(i + 1, total_steps, f"Computing GEV for k={k}")
            result = self._get_clustering_result(k)
            scores.append(result['gev'])

        optimal_k = self._find_elbow_point(self.k_range, scores, maximize=True)

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Elbow - Global Explained Variance",
            higher_is_better=True
        )

    def compute_elbow_residual(self) -> OptimizationResult:
        """Compute elbow method using Residual Variance"""
        scores = []
        total_steps = len(self.k_range)

        for i, k in enumerate(self.k_range):
            self._update_progress(i + 1, total_steps, f"Computing Residual for k={k}")
            result = self._get_clustering_result(k)
            scores.append(result['residual'])

        optimal_k = self._find_elbow_point(self.k_range, scores, maximize=False)

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Elbow - Residual Variance",
            higher_is_better=False
        )

    def compute_silhouette(self) -> OptimizationResult:
        """Compute silhouette analysis"""
        scores = []
        total_steps = len(self.k_range)

        for i, k in enumerate(self.k_range):
            self._update_progress(i + 1, total_steps, f"Computing Silhouette for k={k}")

            if k < 2:  # Silhouette requires at least 2 clusters
                scores.append(-1)
                continue

            result = self._get_clustering_result(k)

            # Use cosine distance for EEG topographies
            distances = cosine_distances(self.maps2use.T)
            score = silhouette_score(distances, result['segmentation'], metric='precomputed')
            scores.append(score)

        # Find optimal k (excluding k=1)
        valid_scores = [(k, s) for k, s in zip(self.k_range, scores) if s != -1]
        optimal_k = max(valid_scores, key=lambda x: x[1])[0] if valid_scores else self.kmin

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Silhouette Method",
            higher_is_better=True
        )

    def compute_calinski_harabasz(self) -> OptimizationResult:
        """Compute Calinski-Harabasz index"""
        scores = []
        total_steps = len(self.k_range)

        for i, k in enumerate(self.k_range):
            self._update_progress(i + 1, total_steps, f"Computing Calinski-Harabasz for k={k}")

            if k < 2:  # CH requires at least 2 clusters
                scores.append(0)
                continue

            result = self._get_clustering_result(k)

            # Normalize data for CH index
            data_normalized = self._normalize_data(self.maps2use.T)
            score = calinski_harabasz_score(data_normalized, result['segmentation'])
            scores.append(score)

        # Find optimal k (excluding k=1)
        valid_scores = [(k, s) for k, s in zip(self.k_range, scores) if k >= 2]
        optimal_k = max(valid_scores, key=lambda x: x[1])[0] if valid_scores else self.kmin

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Calinski-Harabasz Method",
            higher_is_better=True
        )

    def compute_davies_bouldin(self) -> OptimizationResult:
        """Compute Davies-Bouldin index"""
        scores = []
        total_steps = len(self.k_range)

        for i, k in enumerate(self.k_range):
            self._update_progress(i + 1, total_steps, f"Computing Davies-Bouldin for k={k}")

            if k < 2:  # DB requires at least 2 clusters
                scores.append(float('inf'))
                continue

            result = self._get_clustering_result(k)

            # Normalize data for DB index
            data_normalized = self._normalize_data(self.maps2use.T)
            score = davies_bouldin_score(data_normalized, result['segmentation'])
            scores.append(score)

        # Find optimal k (excluding k=1)
        valid_scores = [(k, s) for k, s in zip(self.k_range, scores) if s != float('inf')]
        optimal_k = min(valid_scores, key=lambda x: x[1])[0] if valid_scores else self.kmin

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Davies-Bouldin Method",
            higher_is_better=False
        )

    def compute_cross_validation(self, n_folds: int = 5) -> OptimizationResult:
        """Compute cross-validation scores"""
        scores = []
        total_steps = len(self.k_range) * n_folds
        current_step = 0

        for k in self.k_range:
            fold_scores = []
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.maps2use.T)):
                current_step += 1
                self._update_progress(current_step, total_steps,
                                      f"Cross-validation k={k}, fold {fold_idx + 1}/{n_folds}")

                train_data = self.maps2use[:, train_idx]
                test_data = self.maps2use[:, test_idx]

                # Initialize and run clustering on training data
                clusterer = MicrostateClusterer(
                    n_states=k,
                    n_inits=1,
                    max_iter=self.max_iter,
                    tolerance=self.tolerance
                )

                data_initializer = DataInitializer()
                initial_maps = data_initializer.initialize_cluster_centers(
                    maps2use=train_data,
                    n_states=k,
                    initializer="Random"
                )

                maps, _ = clusterer.modified_kmeans(
                    data=train_data,
                    initial_maps=initial_maps
                )

                # Evaluate on test data
                gev_test = clusterer.compute_gev(data=test_data, maps=maps)
                fold_scores.append(gev_test)

            scores.append(np.mean(fold_scores))

        optimal_k = self.k_range[np.argmax(scores)]

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=scores,
            optimal_k=optimal_k,
            method_name="Cross Validation",
            higher_is_better=True
        )

    def compute_gap_statistic(self, n_refs: int = 10) -> OptimizationResult:
        """Compute gap statistic"""
        wss_actual = []
        wss_random_mean = []
        wss_random_std = []

        total_steps = len(self.k_range) * (1 + n_refs)
        current_step = 0

        # Compute WSS for actual data
        for k in self.k_range:
            current_step += 1
            self._update_progress(current_step, total_steps, f"Computing WSS for actual data, k={k}")

            result = self._get_clustering_result(k)
            wss = self._compute_wss(self.maps2use, result['maps'], result['segmentation'])
            wss_actual.append(wss)

        # Compute WSS for random data
        for k in self.k_range:
            ref_wss = []

            for ref_idx in range(n_refs):
                current_step += 1
                self._update_progress(current_step, total_steps,
                                      f"Computing WSS for random data, k={k}, ref {ref_idx + 1}/{n_refs}")

                # Generate random data
                random_data = np.random.uniform(
                    low=self.maps2use.min(axis=1, keepdims=True),
                    high=self.maps2use.max(axis=1, keepdims=True),
                    size=self.maps2use.shape
                )

                # Cluster random data
                clusterer = MicrostateClusterer(
                    n_states=k,
                    n_inits=1,
                    max_iter=self.max_iter,
                    tolerance=self.tolerance
                )

                data_initializer = DataInitializer()
                initial_maps = data_initializer.initialize_cluster_centers(
                    maps2use=random_data,
                    n_states=k,
                    initializer="Random"
                )

                maps, _ = clusterer.modified_kmeans(
                    data=random_data,
                    initial_maps=initial_maps
                )

                activation = maps.dot(random_data)
                segmentation = np.argmax(np.abs(activation), axis=0)
                wss = self._compute_wss(random_data, maps, segmentation)
                ref_wss.append(wss)

            wss_random_mean.append(np.mean(ref_wss))
            wss_random_std.append(np.std(ref_wss))

        # Compute gap scores
        gap_scores = []
        for i in range(len(self.k_range)):
            gap = np.log(wss_random_mean[i]) - np.log(wss_actual[i])
            gap_scores.append(gap)

        # Find optimal k using gap statistic criterion
        optimal_k = self._find_optimal_gap(self.k_range, gap_scores, wss_random_std)

        return OptimizationResult(
            k_values=self.k_range.copy(),
            scores=gap_scores,
            optimal_k=optimal_k,
            method_name="Gap Statistic",
            higher_is_better=True
        )

    def find_optimal_k(self, optimizer_mode: str, parameter_value: float = None) -> Tuple[int, List[int], List[float]]:
        """
        Find optimal k using specified method.

        Parameters
        ----------
        optimizer_mode : str
            One of: 'gev', 'res', 'sil', 'ch', 'db', 'cv', 'gs'
        parameter_value : float, optional
            Parameter for the method (e.g., threshold, n_folds, n_refs)

        Returns
        -------
        optimal_k : int
            Optimal number of clusters
        k_values : list
            Range of k values tested
        target_values : list
            Scores for each k
        """
        method_map = {
            'gev': ('compute_elbow_gev', None),
            'res': ('compute_elbow_residual', None),
            'sil': ('compute_silhouette', None),
            'ch': ('compute_calinski_harabasz', None),
            'db': ('compute_davies_bouldin', None),
            'cv': ('compute_cross_validation', 'n_folds'),
            'gs': ('compute_gap_statistic', 'n_refs')
        }

        if optimizer_mode not in method_map:
            raise ValueError(f"Unknown optimizer mode: {optimizer_mode}")

        method_name, param_name = method_map[optimizer_mode]
        method = getattr(self, method_name)

        # Call method with parameter if applicable
        if param_name and parameter_value is not None:
            kwargs = {param_name: int(parameter_value)}
            result = method(**kwargs)
        else:
            result = method()

        # Store result
        self.results[optimizer_mode] = result

        return result.optimal_k, result.k_values, result.scores

    def find_optimal_k_majority_vote(self, methods: List[str] = None) -> int:
        """
        Find optimal k using majority vote across multiple methods.

        Parameters
        ----------
        methods : list, optional
            List of methods to use. If None, uses all available methods.

        Returns
        -------
        int
            Optimal k based on majority vote
        """
        if methods is None:
            methods = ['gev', 'res', 'sil', 'ch', 'db', 'cv', 'gs']

        # Progress tracking
        total_methods = len(methods)
        current_method = 0

        # Compute all methods
        optimal_ks = []
        for method in methods:
            current_method += 1
            self._update_progress(current_method, total_methods, f"Running {method.upper()} method...")

            # Default parameters
            params = {'cv': 5, 'gs': 10}
            param_value = params.get(method)

            optimal_k, _, _ = self.find_optimal_k(method, param_value)
            optimal_ks.append(optimal_k)

        # Find most common k
        from collections import Counter
        k_counts = Counter(optimal_ks)
        majority_k = k_counts.most_common(1)[0][0]

        # Store majority vote result
        self.results['majority_vote'] = OptimizationResult(
            k_values=self.k_range.copy(),
            scores=optimal_ks,
            optimal_k=majority_k,
            method_name="Majority Vote",
            higher_is_better=True
        )

        return majority_k

    def get_all_results(self) -> Dict[str, OptimizationResult]:
        """Get all computed results"""
        return self.results.copy()

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data for clustering metrics"""
        return (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-10)

    def _compute_wss(self, data: np.ndarray, maps: np.ndarray, segmentation: np.ndarray) -> float:
        """Compute within-cluster sum of squares"""
        wss = 0
        for k in range(maps.shape[0]):
            cluster_mask = segmentation == k
            if np.sum(cluster_mask) > 0:
                cluster_data = data[:, cluster_mask]
                center = maps[k:k + 1]
                distances = cosine_distances(cluster_data.T, center.T)
                wss += np.sum(distances ** 2)
        return wss

    def _find_elbow_point(self, x_values: List[int], y_values: List[float], maximize: bool) -> int:
        """Find elbow point using perpendicular distance method"""
        if not maximize:
            y_values = [-y for y in y_values]

        # Normalize data
        x_norm = np.array(x_values, dtype=float)
        y_norm = np.array(y_values, dtype=float)

        # Handle edge cases
        if len(x_norm) < 3:
            return x_values[0] if maximize else x_values[-1]

        x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min() + 1e-10)
        y_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min() + 1e-10)

        # Find point with maximum perpendicular distance to line
        line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
        line_length = np.linalg.norm(line_vec)

        if line_length < 1e-10:
            return x_values[len(x_values) // 2]

        line_vec_norm = line_vec / line_length

        max_dist = 0
        elbow_idx = 0

        for i in range(1, len(x_norm) - 1):
            point_vec = np.array([x_norm[i] - x_norm[0], y_norm[i] - y_norm[0]])
            dist = np.abs(np.cross(point_vec, line_vec_norm))

            if dist > max_dist:
                max_dist = dist
                elbow_idx = i

        return x_values[elbow_idx]

    def _find_optimal_gap(self, k_values: List[int], gap_scores: List[float],
                          gap_stds: List[float]) -> int:
        """Find optimal k using gap statistic criterion"""
        # Use the standard gap statistic criterion
        for i in range(len(gap_scores) - 1):
            if gap_scores[i] >= gap_scores[i + 1] - gap_stds[i + 1]:
                return k_values[i]
        return k_values[-1]
