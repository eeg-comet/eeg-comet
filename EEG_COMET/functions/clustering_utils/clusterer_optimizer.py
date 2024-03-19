
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances
from gui.logging_window import LogWindow
from functions.data_utils.data_initializer import DataInitializer
from functions.clustering_utils.microstate_clusterer import MicrostateClusterer


# TODO: move the log to COMET / add figure settings
class ClustererOptimizer:
    def __init__(self, maps2use, min_dist, n_inits, kmin, kmax, preprocessed_data_path, extension, datatype,
                 tolerance=None, max_iter=None):
        """
        Initialize the ClustererOptimizer object.

        Args:
            maps2use: The maps to use for clustering.
            min_dist: The minimum distance between microstate maps.
            n_inits: The number of initializations for the clustering algorithm.
            kmin: The minimum number of microstate maps to consider.
            kmax: The maximum number of microstate maps to consider.
            preprocessed_data_path: The path to the preprocessed data.
            extension: The file extension of the preprocessed data.
            datatype: The data type of the preprocessed data.
            tolerance: The tolerance for convergence of the clustering algorithm (optional).
            max_iter: The maximum number of iterations for the clustering algorithm (optional).
        """

        self.maps2use = maps2use
        self.min_dist = min_dist
        self.tolerance = tolerance
        self.n_inits = n_inits
        self.kmin = kmin
        self.kmax = kmax
        self.max_iter = max_iter
        self.preprocessed_data_path = preprocessed_data_path
        self.extension = extension
        self.datatype = datatype
        self.microstate_clusterer = MicrostateClusterer()
        self.data, _ = DataInitializer().generate_maps_and_peaks(
            self.preprocessed_data_path,
            self.extension,
            self.datatype,
            use_percentages=100
        )

    def find_elbow_with_plot(self, ax):
        """
        Find the elbow point with plots to visualize the clustering metrics.

        Args:
            ax: The matplotlib axis to plot the metrics on.
        """

        target_values = self.find_optimal_k_elbow('all')
        df = pd.DataFrame({
            'K': list(range(self.kmin, self.kmax + 1)),
            'Values': target_values
        })
        self._plot(df, ax)

    def find_optimal_k(self, optimizer_mode='cv', parameter_value=5):
        """
        Find the elbow point without generating plots.

        Args:
            optimizer_mode: The mode of the optimizer (optional).
            parameter_value: The value of the parameter for the optimizer (optional).

        Returns:
            The optimal number of microstate maps.
        Raises:
            ValueError: If the optimizer mode is invalid.
        """

        if optimizer_mode == 'gs':
            return self.find_optimal_k_gap_statistic(int(parameter_value))
        elif optimizer_mode == 'cv':
            return self.find_optimal_k_cross_validation(int(parameter_value), int(parameter_value))
        elif optimizer_mode in ['gev', 'res']:
            return self.find_optimal_k_elbow(optimizer_mode, int(parameter_value))
        elif optimizer_mode == 'sil':
            return self.find_optimal_k_using_silhouette()
        elif optimizer_mode == 'ch':
            return self.find_optimal_k_using_calinski_harabasz()
        elif optimizer_mode == 'db':
            return self.find_optimal_k_using_davies_bouldin()
        else:
            raise ValueError("Invalid optimizer_mode.")

    def export_segmentation(self, n_clusters, data):
        """
        Export the microstate segmentation.

        Args:
            n_clusters: The number of microstate clusters.
            data: The data to be segmented.

        Returns:
            The microstate maps and the segmentation.
        """
        modified_kmeans_results = self.microstate_clusterer.run_modified_kmeans(
            preprocessed_data_path=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            maps2use=self.maps2use,
            n_states=n_clusters,
            n_inits=self.n_inits,
            initializer="Random",
            max_iter=self.max_iter,
            thresh=self.tolerance,
            verbose=False
        )
        maps = modified_kmeans_results['best']['maps']
        segmentation = np.argmax(np.abs(maps.dot(data)), axis=0)
        return maps, segmentation

    def _calculate_ssd_ignoring_polarity(self, n_clusters, data):
        """
        Calculate the sum of squared distances (SSD) ignoring polarity.

        Args:
            n_clusters: The number of microstate clusters.
            data: The data to calculate the SSD.

        Returns:
            The SSD value.
        """

        maps, segmentation = self.export_segmentation(n_clusters, data)
        ssd = 0
        for cluster_idx in range(n_clusters):
            cluster_points = data[:, segmentation == cluster_idx]
            cluster_center = maps[cluster_idx]
            ssd += np.sum(np.sum((np.abs(cluster_points) - np.abs(cluster_center).reshape(-1, 1)) ** 2, axis=0))
        return ssd

    def _calculate_silhouette_score(self, data, n_clusters):
        """
        Find the optimal number of clusters using the silhouette method.

        Returns:
            The optimal number of microstate maps, the range of k values, and the target silhouette values.
        """

        # Run your modified K-means clustering algorithm
        maps, segmentation = self.export_segmentation(n_clusters, data)
        n_samples = data.shape[1]
        silhouette_values = np.zeros(n_samples)
        for i in range(n_samples):
            a_i = 0
            b_i = float('inf')
            cluster_i = segmentation[i]
            data_i = data[:, i]
            for j in range(n_samples):
                if i == j:
                    continue
                cluster_j = segmentation[j]
                if cluster_i == cluster_j:
                    data_j = data[:, j]
                    a_i += np.dot(data_i, data_j) / (np.linalg.norm(data_i) * np.linalg.norm(data_j))
                else:
                    similarity = np.dot(data_i, maps[cluster_j, :]) /\
                                     (np.linalg.norm(data_i) * np.linalg.norm(maps[cluster_j, :]))
                    if similarity < b_i:
                        b_i = similarity
            a_i /= (segmentation == cluster_i).sum() - 1
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
        return np.mean(silhouette_values)

    def find_optimal_k_using_silhouette(self):
        """
        Find the optimal number of clusters using the silhouette method.

        Returns:
            The optimal number of microstate maps, the range of k values, and the target silhouette values.
        """

        print("Identifying the optimal number of clusters using the silhouette method")
        self.k_values_silhouette = range(self.kmin, self.kmax + 1)
        self.target_silhouette = []
        # Create an instance of the progress dialog
        progress_dialog = LogWindow()
        progress_dialog.set_window_title("Finding Optimal K ...")
        progress_dialog.set_label_text("Silhouette Method")
        progress_dialog.show()
        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(self.k_values_silhouette), ncols=100, position=0, leave=True)

        for index, k in enumerate(self.k_values_silhouette):
            sil_value = self._calculate_silhouette_score(k, self.data)
            self.target_silhouette.append(sil_value)

            progress_dialog.set_line_edit_text(f"Silhouette Value for K={k}: {sil_value}")
            progress_dialog.update_progress(index + 1, len(self.k_values_silhouette))
            progress_bar.update(1)

        progress_dialog.close()
        progress_bar.close()
        self.optimal_k_using_silhouette = np.argmax(self.target_silhouette) + self.kmin
        print(f"\nOptimal clusters: {self.optimal_k_using_silhouette}")
        return self.optimal_k_using_silhouette, self.k_values_silhouette, self.target_silhouette

    def _calculate_calinski_harabasz_score(self, n_clusters, data):
        """
        Calculate the Calinski-Harabasz score.

        Args:
            n_clusters: The number of microstate clusters.
            data: The data to calculate the score.

        Returns:
            The Calinski-Harabasz score.
        """

        maps, segmentation = self.export_segmentation(n_clusters, data)
        activation = maps.dot(data)
        n_samples = data.shape[1]
        extra_disp, intra_disp = 0.0, 0.0
        mean_data = np.mean(data, axis=1)
        for cluster_idx in range(n_clusters):
            idx = (segmentation == cluster_idx)
            cluster_points = data[:, idx]
            activation_points = activation[cluster_idx, idx]
            cluster_center = np.dot(cluster_points, activation_points)
            cluster_center /= np.linalg.norm(cluster_center)
            extra_disp += cluster_points.shape[1] * np.sum((np.abs(cluster_center) - np.abs(mean_data)) ** 2)
            intra_disp += np.sum((np.abs(cluster_points) - np.abs(cluster_center)[:, np.newaxis]) ** 2)
        return (
            1.0
            if intra_disp == 0.0
            else (extra_disp * (n_samples - n_clusters) / (intra_disp * (n_clusters - 1.0)))
        )

    def find_optimal_k_using_calinski_harabasz(self):
        """
        Find the optimal number of clusters using the Calinski-Harabasz method.

        Returns:
            The optimal number of microstate maps, the range of k values, and the target Calinski-Harabasz values.
        """

        print("Identifying the optimal number of clusters using the Calinski-Harabasz method")
        self.k_values_calinski_harabasz = range(self.kmin, self.kmax + 1)
        self.target_calinski_harabasz = []
        # Create an instance of the progress dialog
        progress_dialog = LogWindow()
        progress_dialog.set_window_title("Finding Optimal K ...")
        progress_dialog.set_label_text("Calinski-Harabasz Method")
        progress_dialog.show()
        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(self.k_values_calinski_harabasz), ncols=100, position=0, leave=True)

        for index, k in enumerate(self.k_values_calinski_harabasz):
            ch_value = self._calculate_calinski_harabasz_score(k, self.data)
            self.target_calinski_harabasz.append(ch_value)

            progress_dialog.set_line_edit_text(f"Calinski-Harabasz score for K={k}: {ch_value}")
            progress_dialog.update_progress(index + 1, len(self.k_values_calinski_harabasz))
            progress_bar.update(1)

        progress_dialog.close()
        progress_bar.close()
        self.optimal_k_using_calinski_harabasz = np.argmax(self.target_calinski_harabasz) + self.kmin
        print(f"\nOptimal clusters: {self.optimal_k_using_calinski_harabasz}")
        return self.optimal_k_using_calinski_harabasz, self.k_values_calinski_harabasz, self.target_calinski_harabasz

    def _calculate_davies_bouldin_score(self, n_clusters, data):
        """
        Calculate the Davies-Bouldin score.

        Args:
            n_clusters: The number of microstate clusters.
            data: The data to calculate the score.

        Returns:
            The Davies-Bouldin score.
        """

        maps, segmentation = self.export_segmentation(n_clusters, data)
        activation = maps.dot(data)
        intra_dists = np.zeros(n_clusters)
        centroids = np.zeros((n_clusters, data.shape[0]), dtype=float)
        for cluster_idx in range(n_clusters):
            idx = (segmentation == cluster_idx)
            cluster_points = data[:, idx]
            activation_points = activation[cluster_idx, idx]
            cluster_center = np.dot(cluster_points, activation_points)
            cluster_center /= np.linalg.norm(cluster_center)
            centroids[cluster_idx, :] = cluster_center
            intra_dists[cluster_idx] = np.average(np.abs(pairwise_distances(
                cluster_points.T, [cluster_center], metric='cosine')))
        centroid_distances = np.abs(pairwise_distances(centroids, metric='cosine'))
        if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
            return 0.0
        centroid_distances[centroid_distances == 0] = np.inf
        combined_intra_dists = intra_dists[:, None] + intra_dists
        scores = np.max(combined_intra_dists / centroid_distances, axis=1)
        return np.mean(scores)

    def find_optimal_k_using_davies_bouldin(self):
        """
        Find the optimal number of clusters using the Davies-Bouldin method.

        Returns:
            The optimal number of microstate maps, the range of k values, and the target Davies-Bouldin values.
        """

        print("Identifying the optimal number of clusters using the Davies-Bouldin method")
        self.k_values_davies_bouldin = range(self.kmin, self.kmax + 1)
        self.target_davies_bouldin = []
        # Create an instance of the progress dialog
        progress_dialog = LogWindow()
        progress_dialog.set_window_title("Finding Optimal K ...")
        progress_dialog.set_label_text("Davies_Bouldin Method")
        progress_dialog.show()
        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(self.k_values_davies_bouldin), ncols=100, position=0, leave=True)

        for index, k in enumerate(self.k_values_davies_bouldin):
            db_value = self._calculate_davies_bouldin_score(k, self.data)
            self.target_davies_bouldin.append(db_value)

            progress_dialog.set_line_edit_text(f"Davies-Bouldin score for K={k}: {db_value}")
            progress_dialog.update_progress(index + 1, len(self.k_values_davies_bouldin))
            progress_bar.update(1)

        progress_dialog.close()
        progress_bar.close()
        self.optimal_k_using_davies_bouldin = np.argmin(self.target_davies_bouldin) + self.kmin
        print(f"\nOptimal clusters: {self.optimal_k_using_davies_bouldin}")
        return self.optimal_k_using_davies_bouldin, self.k_values_davies_bouldin, self.target_davies_bouldin

    def find_optimal_k_elbow(self, metric, threshold=5):
        """
        Find the optimal number of clusters using the elbow method.

        Args:
            metric: The metric to use for the elbow method.
            threshold: The threshold for reduction in elbow values (optional).

        Returns:
            The optimal number of microstate maps, the range of k values, and the target elbow values.
        """

        print(
            f"\nIdentifying the optimal number of clusters using the elbow method with %{threshold} threshold")

        # Number of clusters to consider
        self.k_values_elbow = range(self.kmin, self.kmax + 1)

        # Initialize SSD
        self.target_elbow = []

        # Create an instance of the progress dialog
        progress_dialog = LogWindow()
        progress_dialog.set_window_title("Finding Optimal K ...")
        progress_dialog.set_label_text(f"Elbow Method with %{threshold} Threshold")
        progress_dialog.show()
        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(self.k_values_elbow), ncols=100, position=0, leave=True)

        for index, k in enumerate(self.k_values_elbow):
            # Run your modified K-means clustering algorithm
            modified_kmeans_results = self.microstate_clusterer.run_modified_kmeans(
                preprocessed_data_path=self.preprocessed_data_path,
                extension=self.extension,
                datatype=self.datatype,
                maps2use=self.maps2use,
                n_states=k,
                n_inits=self.n_inits,
                initializer="Random",
                max_iter=self.max_iter,
                thresh=self.tolerance,
                verbose=False
            )
            gev_values, res_values = [], []
            for init_result in modified_kmeans_results.values():
                gev_values.append(init_result['gev'])
                res_values.append(init_result['residual'])

            if metric == 'gev':
                metric_log = 'Global Explained Variance'
                metric_value = np.mean(gev_values)
                self.target_elbow.append(np.mean(gev_values))
            elif metric == 'res':
                metric_log = 'Residual'
                metric_value = np.mean(res_values)
                self.target_elbow.append(np.mean(res_values))

            progress_dialog.set_line_edit_text(f"{metric_log} for K={k}: {metric_value:.3f}")
            progress_dialog.update_progress(index + 1, len(self.k_values_elbow))
            progress_bar.update(1)

        progress_dialog.close()
        progress_bar.close()

        target_values = [e / k for e, k in zip(self.target_elbow, self.k_values_elbow)]

        # Calculate the reduction in elbow values
        reductions = [target_values[i] - target_values[i - 1] for i in range(1, len(target_values))]

        # Find the index where the reduction is not significant
        self.optimal_k_elbow = None

        for i in range(1, len(reductions)):
            if reductions[i] <= threshold / 100:
                self.optimal_k_elbow = self.k_values_elbow[i] + 1
                break

        # If no optimal k is found, choose the last k value
        if self.optimal_k_elbow is None:
            self.optimal_k_elbow = self.k_values_elbow[-1]

        print(f"\nOptimal clusters: {self.optimal_k_elbow}")
        return self.optimal_k_elbow, self.k_values_elbow, self.target_elbow

    def find_optimal_k_gap_statistic(self, n_random_datasets=5):
        """
        Find the optimal number of clusters using the gap statistic method.

        Args:
            n_random_datasets: The number of random datasets to generate (optional).

        Returns:
            The optimal number of microstate maps, the range of k values, and the target gap statistic values.
        """

        print(
            f"\nIdentifying the optimal number of clusters using the gap statistic method"
            f"with {n_random_datasets}-random datasets")

        # Number of clusters to consider
        self.k_values_gap_statistic = range(self.kmin, self.kmax + 1)

        # Initialize arrays to store SSD values
        ssd_real = np.zeros(len(self.k_values_gap_statistic))
        ssd_random = np.zeros((len(self.k_values_gap_statistic), n_random_datasets))

        # Create an instance of the progress dialog
        progress_dialog = LogWindow()
        progress_dialog.set_window_title("Finding Optimal K ...")
        progress_dialog.set_label_text(f"Gap Statistic Method with {n_random_datasets}-Random Datasets")
        progress_dialog.show()
        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(self.k_values_gap_statistic) * (1 + n_random_datasets),
                            ncols=100, position=0, leave=True)

        for i, k in enumerate(self.k_values_gap_statistic):
            ssd_real[i] = self._calculate_ssd_ignoring_polarity(self.data, k)
            progress_dialog.set_line_edit_text(
                f"Sum of Squared Distances for Real Dataset with K={k}: {ssd_real[i]}")
            progress_dialog.update_progress(i + 1, len(self.k_values_gap_statistic))
            progress_bar.update(1)

        for j in range(n_random_datasets):
            random_data = np.random.rand(*self.data.shape)  # Generate random data with the same shape as your data
            for i, k in enumerate(self.k_values_gap_statistic):
                ssd_random[i, j] = self._calculate_ssd_ignoring_polarity(random_data, k)
                progress_dialog.set_line_edit_text(
                    f"Sum of Squared Distances for Random Dataset {j} and K={k}: {ssd_random[i, j]}")
                progress_dialog.update_progress(i + 1, len(self.k_values_gap_statistic) * (1 + n_random_datasets))
                progress_bar.update(1)

        # Calculate the expected SSD for random data
        ssd_random_mean = ssd_random.mean(axis=1)

        # Calculate the gap statistic
        self.target_gap_statistic = np.log(ssd_random_mean) - np.log(ssd_real)

        # Find the optimal number of clusters (the maximum point of the gap statistic)
        self.optimal_k_gap_statistic = np.argmax(self.target_gap_statistic) + 1

        # Close the progress bar
        progress_dialog.close()
        progress_bar.close()

        print(f"\nOptimal clusters: {self.optimal_k_gap_statistic}")
        self.optimal_k_gap_statistic_done = True
        return self.optimal_k_gap_statistic, self.k_values_gap_statistic, self.target_gap_statistic

    def find_optimal_k_cross_validation(self, n_splits=5, threshold=5):
        """
        Find the optimal number of microstates using cross-validation.

        Args:
            n_splits: The number of splits for cross-validation (optional).
            threshold: The threshold for reduction in cross-validation scores (optional).

        Returns:
            The optimal number of microstate maps, the range of k values, and the target cross-validation scores.
        """

        print(f"\nIdentifying the optimal number of microstates using {n_splits}-fold cross-validation method")
        # Number of clusters to consider
        self.k_values_cross_validation = range(self.kmin, self.kmax + 1)
        # Initialize arrays to store cross-validation scores
        cv_scores = []

        # Calculate the total number of updates
        total_updates = len(self.k_values_cross_validation) * n_splits

        # Create an instance of the progress dialog
        progress_dialog = LogWindow()
        progress_dialog.set_window_title("Finding Optimal K ...")
        progress_dialog.set_label_text(f"{n_splits}-Fold Cross-Validation Method")
        progress_dialog.show()
        # Create a tqdm progress bar
        combined_progress = tqdm(total=total_updates, desc="Progress", ncols=100, position=0, leave=True)

        for k in self.k_values_cross_validation:
            wcss = 0
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(self.data.T):
                train_data, test_data = self.data[:, train_index], self.data[:, test_index]
                # Run the modified K-means clustering algorithm
                modified_kmeans_results = self.microstate_clusterer.run_modified_kmeans(
                    preprocessed_data_path=self.preprocessed_data_path,
                    extension=self.extension,
                    datatype=self.datatype,
                    maps2use=train_data,
                    n_states=k,
                    n_inits=1,
                    initializer="Random",
                    max_iter=self.max_iter,
                    thresh=self.tolerance,
                    verbose=False
                )
                maps = modified_kmeans_results['best']['maps']
                segmentation = np.argmax(np.abs(maps.dot(test_data)), axis=0)
                # Calculate WCSS for each cluster
                for cluster_idx in range(k):
                    cluster_points = test_data[:, segmentation == cluster_idx]
                    cluster_center = maps[cluster_idx]
                    # Calculate the sum of squares of distances within the cluster
                    wcss += np.sum(np.sum((cluster_points - cluster_center.reshape(-1, 1)) ** 2, axis=0))
                # Update the combined progress bar
                combined_progress.update(1)
            # Calculate the average WCSS over all folds
            avg_wcss = wcss / n_splits
            cv_scores.append(avg_wcss)

        combined_progress.close()

        self.target_cross_validation = cv_scores
        target_values = [e / k for e, k in zip(cv_scores, self.k_values_cross_validation)]

        # Calculate the reduction in elbow values
        reductions = [target_values[i] - target_values[i - 1] for i in range(1, len(target_values))]

        # Find the index where the reduction is not significant
        self.optimal_k_cross_validation = None

        for i in range(1, len(reductions)):
            if reductions[i] <= threshold / 100:
                self.optimal_k_cross_validation = self.k_values_cross_validation[i] + 1
                break

        # If no optimal k is found, choose the last k value
        if self.optimal_k_cross_validation is None:
            self.optimal_k_cross_validation = self.k_values_cross_validation[-1]

        print(f"\nOptimal clusters: {self.optimal_k_cross_validation}")
        return self.optimal_k_cross_validation, self.k_values_cross_validation, self.target_cross_validation

    @staticmethod
    def _plot(df, ax1, ax2, ax3):
        """
        Generate line plots for the calculated metrics to visualize the elbow point.

        Args:
            df: The dataframe containing the metrics.
            ax1: The matplotlib axis for the residual plot.
            ax2: The matplotlib axis for the global explained variance plot.
            ax3: The matplotlib axis for the silhouette score plot.
        """

        # Visualize the result
        res_fig = sns.lineplot(data=df, x="K", y="Residual",
                               linewidth=5, marker="o", markersize=16, dashes=False, ax=ax1)
        res_fig.set_xlabel("Number of Microstate Maps", size=16)
        res_fig.set_ylabel("Residual", size=16)

        xlabel_format = '{:,.0f}'
        ticks_loc = ax1.get_xticks().tolist()
        ax1.set_xticks(ax1.get_xticks().tolist())
        ax1.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], size=14)
        ylabel_format = '{:,.2f}'
        ticks_loc = ax1.get_yticks().tolist()
        ax1.set_yticks(ax1.get_yticks().tolist())
        ax1.set_yticklabels([ylabel_format.format(x) for x in ticks_loc], size=14)

        gev_fig = sns.lineplot(data=df, x="K", y="Global Explained Variance",
                               linewidth=5, style=None, marker="o", markersize=16, dashes=False, ax=ax2)
        gev_fig.set_xlabel("Number of Microstate Maps", size=16)
        gev_fig.set_ylabel("Global Explained Variance", size=16)

        xlabel_format = '{:,.0f}'
        ticks_loc = ax2.get_xticks().tolist()
        ax2.set_xticks(ax2.get_xticks().tolist())
        ax2.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], size=14)
        ylabel_format = '{:,.2f}'
        ticks_loc = ax2.get_yticks().tolist()
        ax2.set_yticks(ax2.get_yticks().tolist())
        ax2.set_yticklabels([ylabel_format.format(x) for x in ticks_loc], size=14)

        silhouette_fig = sns.lineplot(data=df, x="K", y="Silhouette Score",
                                      linewidth=5, style=None, marker="o", markersize=16, dashes=False, ax=ax3)
        silhouette_fig.set_xlabel("Number of Microstate Maps", size=16)
        silhouette_fig.set_ylabel("Silhouette Score", size=16)

        xlabel_format = '{:,.0f}'
        ticks_loc = ax3.get_xticks().tolist()
        ax3.set_xticks(ax3.get_xticks().tolist())
        ax3.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], size=14)
        ylabel_format = '{:,.2f}'
        ticks_loc = ax3.get_yticks().tolist()
        ax3.set_yticks(ax3.get_yticks().tolist())
        ax3.set_yticklabels([ylabel_format.format(x) for x in ticks_loc], size=14)
