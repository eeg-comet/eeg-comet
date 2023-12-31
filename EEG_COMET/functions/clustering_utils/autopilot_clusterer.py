
import os
import numpy as np
from sklearn.model_selection import ParameterGrid

from functions.data_utils.data_io import DataIO
from functions.data_utils.data_initializer import DataInitializer
from functions.clustering_utils.microstate_clusterer import MicrostateClusterer
from functions.clustering_utils.microstate_labeler import MicrostateLabeler
from functions.backfitting_utils.microstate_backfitter import MicrostateBackfitter
from functions.backfitting_utils.segmentation_io import SegmentationIO


class AutopilotClusterer:

    def __init__(self, save_dir, study_name, extension, datatype):
        self.preprocessed_data_path = os.path.join(save_dir, f"{study_name}_preprocessed_data")
        self.segmentation_path = os.path.join(save_dir, f"{study_name}_segmentation")
        #self.extracted_features_path = os.path.join(save_dir, f"{study_name}_extracted_features")
        self.microstate_maps_path = os.path.join(save_dir, 'microstate_maps.csv')
        eeg_info_path = os.path.join(save_dir, "eeg_info.pkl")
        self.eeg_info = DataIO().load_eeg_info(eeg_info_path)
        self.sample_rate = self.eeg_info['sfreq']

        self.study_name = study_name
        self.extension = extension
        self.datatype = datatype
        self.export_format = '.csv'

        # Load EEG
        self.eeg_data, _ = DataInitializer().generate_maps_and_peaks(
            self.preprocessed_data_path, extension, datatype, use_percentages=100)

        self.microstate_clusterer = MicrostateClusterer(n_inits=1, max_iter=500, tolerance=1e-6)

        # Define the parameters to search
        param_grid = {
            'data2use': ['Random', 'Peaks'],
            'number_maps': np.arange(2, 11),
            'initializer': ['Random', 'K-Means++'],
            'backfit_maps': ['all'],
            'backfit_smoothing_methods': ['replace_half', 'replace_high', 'smooth'],
            'backfit_smoothing_win': np.arange(int(self.sample_rate * 0.05)),
        }

        # Generate all combinations of parameters
        self.param_combinations = list(ParameterGrid(param_grid))


        self.best_maps = None
        self.best_gev = 0
        self.best_residual = None
        self.best_segmentation = None
        self.best_clustering_score = 0

    """
    def clustering_score(self, microstate_maps, segmentation):
        # Compute the Calinski-Harabasz score
        #gfp = np.std(self.eeg_data, axis=0)  # Global Field Power
        # Normalize maps
        microstate_maps = microstate_maps / np.linalg.norm(microstate_maps, axis=1, keepdims=True)

        num_maps = microstate_maps.shape[0]
        # Create a mapping between unique labels and integers
        label_to_int = {label: i for i, label in enumerate(np.unique(segmentation))}
        # Convert the segmentation array to integers using the mapping
        segmentation_int = np.array([label_to_int[label] for label in segmentation])

        selected_maps = microstate_maps[segmentation_int, :]
        n_samples = self.eeg_data.shape[1]
        microstate_labels = np.unique(segmentation)

        extra_disp, intra_disp = 0.0, 0.0
        mean_data = np.mean(self.eeg_data, axis=1)
        for cluster_idx in microstate_labels:
            cluster_points = self.eeg_data[:, segmentation == cluster_idx]
            cluster_center = np.mean(cluster_points, axis=1)

            extra_disp += cluster_points.shape[1] * np.sum((np.abs(cluster_center) - np.abs(mean_data)) ** 2)
            intra_disp += np.sum((np.abs(cluster_points) - np.abs(cluster_center)[:, np.newaxis]) ** 2)

        return (
            1.0
            if intra_disp == 0.0
            else extra_disp * (n_samples - num_maps) / (intra_disp * (num_maps - 1.0))
        )
    """

    def clustering_score(self, microstate_maps, segmentation):
        microstate_maps = microstate_maps / np.linalg.norm(microstate_maps, axis=1, keepdims=True)

        num_maps = microstate_maps.shape[0]
        label_to_int = {label: i for i, label in enumerate(np.unique(segmentation))}
        segmentation_int = np.array([label_to_int[label] for label in segmentation])

        selected_maps = microstate_maps[segmentation_int, :]
        n_samples = self.eeg_data.shape[1]
        microstate_labels = np.unique(segmentation)

        activation = microstate_maps.dot(self.eeg_data)

        extra_disp, intra_disp, penalty_term = 0.0, 0.0, 0.0
        mean_data = np.mean(self.eeg_data, axis=1)

        for cluster_idx, cluster_label in enumerate(microstate_labels):
            idx = (segmentation == cluster_label)
            cluster_points = self.eeg_data[:, idx]
            activation_points = activation[cluster_idx, idx]
            cluster_center = np.dot(cluster_points, activation_points)
            cluster_center /= np.linalg.norm(cluster_center)

            cluster_size_penalty = 1 / (1 + np.exp(-cluster_points.shape[1]))  # Example penalty term for small clusters

            extra_disp += cluster_points.shape[1] * np.sum((np.abs(cluster_center) - np.abs(mean_data)) ** 2)
            intra_disp += np.sum((np.abs(cluster_points) - np.abs(cluster_center)[:, np.newaxis]) ** 2)
            penalty_term += cluster_size_penalty

        return (
            1.0
            if intra_disp == 0.0
            else (extra_disp * (n_samples - num_maps) / (intra_disp * (num_maps - 1.0))) * penalty_term
        )

    def run_iteration(self, use_percentages, n_states, initializer, backfit_to, filter_segments_method,
                      filter_segments_smoothing_win):

        maps2use, peaks2use = DataInitializer().generate_maps_and_peaks(
            self.preprocessed_data_path,
            self.extension,
            self.datatype,
            use_percentages
        )

        modified_kmeans_results = self.microstate_clusterer.run_modified_kmeans(
            self.preprocessed_data_path,
            self.extension,
            self.datatype,
            maps2use=maps2use,
            n_states=n_states,
            n_inits=1,
            initializer=initializer,
        )

        iteration_maps = modified_kmeans_results['best']['maps']
        iteration_maps = np.array(iteration_maps)
        iteration_gev = modified_kmeans_results['best']['gev']
        iteration_residual = modified_kmeans_results['best']['residual']

        microstate_labeler = MicrostateLabeler(iteration_maps, self.eeg_info, self.microstate_maps_path)
        micro_labels = microstate_labeler.do_labeling()

        if not os.path.exists(self.segmentation_path):
            os.makedirs(self.segmentation_path)

        #if backfit_to == 'peaks':
        #    filter_segments = False
        #    filter_segments_method = None
        #    filter_segments_smoothing_win = 0
        #else:
        filter_segments = True
        identify_short_window = False
        epsilon = 1E-6
        microstate_backfitter = MicrostateBackfitter(
            self.study_name,
            self.preprocessed_data_path,
            iteration_maps,
            backfit_to,
            filter_segments,
            filter_segments_method,
            identify_short_window,
            filter_segments_smoothing_win,
            micro_labels,
            self.segmentation_path,
            self.extension,
            self.datatype,
            self.sample_rate,
            [epsilon, filter_segments_smoothing_win, 2 * filter_segments_smoothing_win],
            self.export_format,
        )
        microstate_backfitter.perform_segmentation()

        segmentation_list_path, segmentation_list_filename = DataIO().find_data(
            self.segmentation_path,
            self.export_format
        )
        iteration_segmentation = []
        for s in range(len(segmentation_list_path)):
            segmentation_path = segmentation_list_path[s]
            file_segmentation = SegmentationIO().load_segmentation(
                segmentation_path, import_format='.csv')
            iteration_segmentation = np.concatenate((iteration_segmentation, file_segmentation))

        iteration_clustering_score = self.clustering_score(iteration_maps, iteration_segmentation)

        print(use_percentages, n_states, initializer, backfit_to,
                      filter_segments_method, filter_segments_smoothing_win)
        print(f"\niteration_clustering_score: {iteration_clustering_score}")
        return iteration_maps, iteration_gev, iteration_residual, iteration_segmentation, iteration_clustering_score

    def run_autopilot(self):
        best_clustering_score = 0  # Initialize with a low score
        best_maps = None
        best_gev = None
        best_residual = None
        best_segmentation = None

        for params in self.param_combinations:
            use_percentages = 50 if params['data2use'] == 'Random' else None

            iteration_maps, iteration_gev, iteration_residual, iteration_segmentation, iteration_clustering_score = \
                self.run_iteration(
                    use_percentages,
                    params['number_maps'],
                    params['initializer'],
                    params['backfit_maps'],
                    params['backfit_smoothing_methods'],
                    params['backfit_smoothing_win']
                )

            if iteration_clustering_score > best_clustering_score:
                best_clustering_score = iteration_clustering_score
                best_maps = iteration_maps
                best_gev = iteration_gev
                best_residual = iteration_residual
                best_segmentation = iteration_segmentation

            print(f"Best SCORE: {best_clustering_score}, Best GEV: {best_gev}")
