
import os
import mne
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from configparser import ConfigParser

from controllers.logging_window import LogWindow
from data_utils.data_io import DataIO
from data_utils.data_preprocessor import DataPreprocessor
from data_utils.data_initializer import DataInitializer
from clustering_utils.clusterer_optimizer import ClustererOptimizer
from clustering_utils.microstate_clusterer import MicrostateClusterer
from clustering_utils.microstate_labeler import MicrostateLabeler
from backfitting_utils.microstate_backfitter import MicrostateBackfitter
from backfitting_utils.segmentation_io import SegmentationIO
from features_utils.feature_helper import FeatureHelper
from features_utils.feature_extractor import FeatureExtractor
from features_utils.feature_io import FeatureIO
from sourcelocalization_utils.source_localizer import SourceLocalizer
from clustering_utils.autopilot_clusterer import AutopilotClusterer


class COMET:
    """
    The COMET class represents an instance of the EEG-COMET application.
    It provides methods to load configuration settings, perform various processes
    such as preprocessing, clustering, labeling, backfitting, feature extraction,
    source localization, and source-microstate correlation.
    If auto_save is True, the COMET will automatically save itself after each process
    """

    def __init__(self, config: dict = None, auto_save: bool = True):

        # Define all instance variables
        # TODO: Set Default Values
        self.LogWindow = None
        self.log_text = []
        self.config = {}

        config_path = "./default_config.ini"
        self.config = self.load_config(config_path)

        # if config:
        #     self.load_config(config)

        # Define global directories based on the study information
        self.save_dir = os.path.join(self.output_folder, self.study_name)
        self.tbx_object_path = os.path.join(self.save_dir, "eeg_comet_parameters.pkl")
        self.config_path = os.path.join(self.save_dir, f"{self.study_name}_config.ini")
        self.preprocessed_data_path = os.path.join(
            self.save_dir, f"{self.study_name}_preprocessed_data")
        self.eeg_info_path = os.path.join(self.save_dir, "eeg_info.pkl")
        self.microstate_maps_path = os.path.join(self.save_dir, "microstate_maps.csv")
        self.extracted_features_path = os.path.join(
            self.save_dir, f"{self.study_name}_extracted_features")
        self.segmentation_path = os.path.join(
            self.save_dir, f"{self.study_name}_segmentation")
        self.localized_sources_path = os.path.join(
            self.save_dir, f"{self.study_name}_localized_sources")
        self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
        self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")

        # Initialize boolean flags indicating if the step has been completed
        self.done_preprocessing: bool = False
        self.done_clustering: bool = False
        self.done_labeling_microstates: bool = False
        self.done_backfitting: bool = False
        self.done_extracting_features: bool = False
        self.done_source_localization: bool = False
        self.done_source_microstate_correlation: bool = False
        self.auto_save = auto_save

    def load_config(self, config_path):
        """
        Load configuration settings from a dictionary.
        """
        config = ConfigParser()
        config.read(config_path)

        # Input/Output Configs
        io_config = config["io_config"]
        self.study_name = io_config.get("study_name", "")
        self.input_folder = io_config.get("input_folder", "")
        self.channel_location_dir = io_config.get("channel_location_dir", "")
        self.extension = io_config.get("extension", ".auto")
        self.pattern_content = io_config.get("pattern", "*")
        self.datatype = io_config.get("datatype", "raw")
        self.output_folder = io_config.get("output_folder", "")

        # Preprocessing Configs
        preprocessing_config = config["preprocessing_config"]
        self.filter_data = preprocessing_config.getboolean("filter_data", True)
        if self.filter_data:
            self.filter_method = preprocessing_config.get("filter_method", "fir")
            self.lowcut_freq = preprocessing_config.getint("lowcut_freq", 2)
            self.highcut_freq = preprocessing_config.getint("highcut_freq", 20)
        self.downsample_data = preprocessing_config.getboolean("downsample_data", True)
        if self.downsample_data:
            self.sample_rate = preprocessing_config.getint("sample_rate", 250)
        self.remove_channels = preprocessing_config.getboolean("remove_channels", False)
        if self.remove_channels:
            self.chan2rm = preprocessing_config.get("ch2rm", "")

        # Clustering Configs
        clustering_config = config["clustering_config"]
        self.smoothing_gfp = clustering_config.getboolean("smoothing_gfp", False)
        if self.smoothing_gfp:
            self.smoothing_distance = clustering_config.getint("smoothing_distance", 10)
        number_of_maps = clustering_config.get("number_of_maps", 4)
        self.number_of_maps = number_of_maps if number_of_maps == "auto" else int(number_of_maps)
        self.choose_number_of_maps = "Auto" if number_of_maps == "auto" else "User"
        if self.number_of_maps == "auto":
            self.stopping_mode = clustering_config.get("stopping_mode", "gev")
            self.stopping_parameter = clustering_config.getfloat("stopping_parameter", 10)
            self.kmin = clustering_config.getint("kmin", 2)
            self.kmax = clustering_config.getint("kmax", 10)
        self.initializer = clustering_config.get("initializer", "Random")
        self.clustering_method = clustering_config.get("clustering_method", "Modified K-Means Clustering")
        self.max_iterations = clustering_config.getint("max_iterations", 500)
        self.clustering_tolerance = clustering_config.getfloat("clustering_tolerance", 1e-6)
        need_options = ["X-Means Clustering", "Agglomerative Hierarchical Clustering",
                        "K-Means Clustering", "PCA + K-Means Clustering",
                        "Autoencoder + K-Means Clustering", ]
        self.clustering_option = clustering_config.get("clustering_option",
                                                       "") if self.clustering_method in need_options else ""
        self.number_of_repeats = clustering_config.getint("number_of_repeats", 5)
        self.use_percentages = clustering_config.getint("use_percentages", "")

        # Backfitting Configs
        backfitting_config = config["backfitting_config"]
        self.backfit_to = backfitting_config.get("backfit_to", "")
        self.identify_short_window = backfitting_config.getboolean("identify_short_window", False)
        self.filter_segments = backfitting_config.getboolean("filter_segments", False)
        if self.filter_segments:
            self.remove_segments_less_than = backfitting_config.getint("remove_segments_less_than", 20)
            self.filter_segments_option = backfitting_config.get("filter_segments_option", "smooth")
        self.epsilon = backfitting_config.getfloat("epsilon", 1e-6)
        self.b = backfitting_config.getint("b", 3)
        self.lamb = backfitting_config.getint("lamb", 5)

        # Feature Extraction Configs
        features_config = config["features_config"]
        self.export_format = features_config.get("export_format", ".csv")
        self.feature_list = [x.strip() for x in features_config.get("feature_list", "").split(",")]
        self.feature_mode = [x.strip() for x in features_config.get("feature_mode", "").split(",")]
        self.feature_types = [x.strip() for x in features_config.get("feature_types", "").split(",")]
        self.window_size = features_config.getint("window_size", 1) if "OCC" in self.feature_list else ""

        # Source Localization Configs
        source_config = config["source_config"]
        self.inverse_method = source_config.get("inverse_method", "dSPM")
        self.nperm = source_config.getint("nperm", 2000)
        self.spacing = source_config.get("spacing", "ico3")
        self.source_localization_method = source_config.get("source_localization_method", "tess")
        self.anatomy_subjects_dir = source_config.get("anatomy_subjects_dir", "")

        return config

    def load_tbx(self, tbx_object_path):
        # Load the TBX object from the file
        with open(tbx_object_path, 'rb') as input_file:
            loaded_object = pickle.load(input_file)
        # Restore the attributes of the current object from the loaded object
        for attr, value in loaded_object.__dict__.items():
            # Exclude loading LogWindow
            if attr == 'LogWindow' and value is None:
                continue
            setattr(self, attr, value)

    def initialize_log_window(self):
        self.LogWindow = LogWindow()
        self.LogWindow.append_log("Welcome to EEG-COMET!")

    def update_microstates_order(self, current_order_labels, current_order_maps):
        self.micro_labels = current_order_labels
        self.best_maps = current_order_maps

    def load_raw(self):
        """
        Locate EEG file paths.
        """
        self.pattern = '*' if self.load_all_files else '*' + self.pattern_content + '*'
        self.list_eegs_path, self.list_eegs = DataIO().find_data(
            input_folder=self.input_folder, extension=self.extension, pattern=self.pattern
        )

    def do_preprocessing(self):
        print('\nPreprocessing ...')

        # Define global directories based on the study information
        self.tbx_object_path = os.path.join(self.save_dir, "eeg_comet_parameters.pkl")
        self.config_path = os.path.join(self.save_dir, f"{self.study_name}_config.ini")
        self.preprocessed_data_path = os.path.join(
            self.save_dir, f"{self.study_name}_preprocessed_data")
        self.eeg_info_path = os.path.join(self.save_dir, "eeg_info.pkl")
        self.microstate_maps_path = os.path.join(self.save_dir, "microstate_maps.csv")
        self.extracted_features_path = os.path.join(
            self.save_dir, f"{self.study_name}_extracted_features")
        self.segmentation_path = os.path.join(
            self.save_dir, f"{self.study_name}_segmentation")
        self.localized_sources_path = os.path.join(
            self.save_dir, f"{self.study_name}_localized_sources")
        self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
        self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")

        # Create preprocessed data directory if it doesn't exist
        if not os.path.exists(self.preprocessed_data_path):
            os.makedirs(self.preprocessed_data_path)

        # Initialize variables
        length_all_data = []
        data_io = DataIO()
        self.LogWindow.setup_progress_dialog(
            window_title="Preprocessing ...",
            label_text="Preprocessing data ...",
            max_value=len(self.list_eegs)
        )

        # Log the preprocessing progress
        self.LogWindow.append_log(
            f"Study Created\n"
            f"✓ Study Name: {self.study_name}\n"
            f"✓ Input Path: {self.input_folder}\n"
            f"✓ Found {len(self.list_eegs_path)} {self.datatype} EEG data with {self.extension} extension."
        )

        self.LogWindow.append_log(
            f"EEG Preprocessing Settings:\n"
            f"* Channels to Remove: {self.chan2rm}\n"
            f"* Bandpass Filter: {self.filter_method.upper()} Method ({self.lowcut_freq}Hz and {self.highcut_freq}Hz)\n"
            f"* Downsampling Rate: {self.sample_rate}Hz", log_type='settings'
        )

        # Create an instance of the DataPreprocessor class
        preprocessor = DataPreprocessor()

        # Iterate through EEG files
        for eeg_idx, (eeg_path, eeg_name) in tqdm(enumerate(zip(self.list_eegs_path, self.list_eegs)),
                                                  total=len(self.list_eegs_path)):
            self.LogWindow.update_progress(value=eeg_idx, text=f"{eeg_name}")

            # Preprocess EEG data
            eeg, preprocessed_data, length_data, eeg_info, channels2remove = preprocessor.preprocess_eegs(
                eeg_path=eeg_path,
                list_eegs=self.list_eegs_path,
                datatype=self.datatype,
                channel_location_dir=self.channel_location_dir,
                filter_bool=self.filter_data,
                filtermethod=self.filter_method,
                lowcut=self.lowcut_freq,
                highcut=self.highcut_freq,
                downsample_bool=self.downsample_data,
                sampling_rate=self.sample_rate,
                chan2rm=self.chan2rm
            )

            # Save EEG info
            self.eeg_info = eeg.info
            if not self.sample_rate:
                self.sample_rate = int(self.eeg_info['sfreq'])
            self.channels2remove = channels2remove
            data_io.save_eeg_info(eeg_info_path=self.eeg_info_path, eeg_info=eeg_info)

            # Save EEG object
            ch_names, ch_location = list(self.eeg_info['ch_names']), self.eeg_info['chs']
            n_chan = len(ch_names)
            self.n_chan = n_chan
            self.ch_names = ch_names
            # name = os.path.basename(eeg_path)
            name = os.path.splitext(eeg_name)[0]
            save_path = os.path.join(self.preprocessed_data_path, name)
            self.length_all_data = np.append(length_all_data, int(length_data))
            data_io.export_eegs(eeg=eeg, save_path=save_path, extension=self.extension, datatype=self.datatype)

            # Log the preprocessing progress

            self.LogWindow.append_log(
                f"EEG Preprocessed [{eeg_idx + 1}/{len(self.list_eegs_path)}]\n"
                f"✓ Data: {eeg_name} - Length: {int(length_data / self.sample_rate)} sec"
            )
            # Check if the process should be stopped
            if not self.LogWindow.running:
                break

        # Set preprocessing flag
        self.done_preprocessing = True
        self.LogWindow.process_finished("✓ All EEG data have been successfully preprocessed!")

        # Optionally save the preprocessed data
        if self.auto_save:
            self.save_config()
            self.save_tbx()

    def do_autopilot(self):
        # TODO: not completed!
        autopilot_clusterer = AutopilotClusterer(
            self.save_dir, self.study_name, self.extension, self.datatype
        )
        autopilot_clusterer.run_autopilot()

        # Set parameters
        self.clustering_method = 'Modified K-Means Clustering'
        self.number_of_repeats = 1
        use_percentages = [60, 80, 100]
        min_distance_size = [0, 10, 20, 40]

    def do_clustering(self):
        print('\nClustering ...')

        # Calculate minimum distance size if smoothing GFP is enabled
        if self.smoothing_gfp:
            self.min_distance_size = int(int(self.smoothing_distance) / (1000 / int(self.sample_rate)))
        else:
            self.min_distance_size = []

        # Check if clustering method is supported
        available_methods = [
            'Modified K-Means Clustering',
            'K-Means Clustering',
            'PCA + K-Means Clustering',
            'Autoencoder + K-Means Clustering',
            'X-Means Clustering',
            'Agglomerative Hierarchical Clustering',
        ]
        assert self.clustering_method in available_methods, "Clustering method not supported"

        # Initialize microstate clusterer
        self.LogWindow.setup_progress_dialog(
            window_title="Clustering ...",
            label_text=f"Clustering data into {self.number_of_maps} microstates ...",
            max_value=self.number_of_repeats
        )
        data_initializer = DataInitializer()
        microstate_clusterer = MicrostateClusterer(
            n_inits=self.number_of_repeats,
            max_iter=self.max_iterations,
            tolerance=self.clustering_tolerance
        )

        # Generate maps and peaks automatically if number_of_maps is set to 'auto'
        if self.number_of_maps == 'auto':
            data_initializer = DataInitializer()
            self.maps2use, self.peaks2use = data_initializer.generate_maps_and_peaks(
                preprocessed_folder=self.preprocessed_data_path,
                extension=self.extension,
                datatype=self.datatype,
                use_percentages=self.use_percentages,
                min_dist=self.min_distance_size
            )
            clusterer_optimizer = ClustererOptimizer(
                maps2use=self.maps2use,
                min_dist=self.min_distance_size,
                n_inits=self.number_of_repeats,
                kmin=self.kmin,
                kmax=self.kmax,
                preprocessed_data_path=self.preprocessed_data_path,
                extension=self.extension,
                datatype=self.datatype,
                tolerance=self.clustering_tolerance,
                max_iter=self.max_iterations
            )

            # Find optimal number of maps
            self.optimal_k, self.k_values, self.target_values = clusterer_optimizer.find_optimal_k(
                optimizer_mode=self.stopping_mode, parameter_value=self.stopping_parameter
            )
            self.number_of_maps = self.optimal_k
            print(f'Result: n_states = {self.number_of_maps}')

        # Perform clustering
        self.maps2use, self.peaks2use = data_initializer.generate_maps_and_peaks(
            preprocessed_folder=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            use_percentages=self.use_percentages,
            min_dist=self.min_distance_size
        )

        if self.clustering_method == 'PCA + K-Means Clustering':
            # Extract features using PCA
            self.maps2use = microstate_clusterer.extract_features_with_pca(
                eeg_data=np.transpose(self.maps2use), pca_components=self.n_pca
            )

        if self.clustering_method == 'Autoencoder + K-Means Clustering':
            # Extract features using Autoencoder
            self.maps2use, autoencoder = microstate_clusterer.extract_features_with_autoencoder(
                eeg_data=np.transpose(self.maps2use), encoding_dim=10
            )

        all_data, _ = data_initializer.generate_maps_and_peaks(
            preprocessed_folder=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            use_percentages=100
        )

        if self.choose_number_of_maps == "auto":
            k_log = 'will be automatically determined.'
        else:
            k_log = 'is user-predefined.'

        if self.use_percentages is not None:
            cluster_data_log = f"{self.use_percentages}% randomly selected time-points of the data."
        else:
            cluster_data_log = 'the local peaks of the global field power.'

        self.LogWindow.append_log(
            f"Clustering Settings:\n"
            f"* Clustering algorithm: {self.clustering_method}\n"
            f"* The number of maps to extract {k_log}\n"
            f"* Clustering will be performed on {cluster_data_log}", log_type='settings'
        )

        self.best_residual, self.best_maps = None, None
        self.best_gev, self.best_confidence = 0, 0
        if self.clustering_method == 'Modified K-Means Clustering':
            modified_kmeans_results = {}
            for init in range(self.number_of_repeats):
                self.LogWindow.update_progress(
                    value=init,
                    text=f"Clustering [{init + 1}/{self.number_of_repeats}] - "
                    f"Global Explained Variance: {100 * self.best_gev:.3f}%"
                )
                initial_maps = data_initializer.initialize_cluster_centers(
                    maps2use=self.maps2use, n_states=self.number_of_maps, initializer=self.initializer
                )

                maps_init, residual_init = microstate_clusterer.modified_kmeans(
                    data=self.maps2use,
                    initial_maps=initial_maps,
                    n_states=self.number_of_maps,
                    max_iter=self.max_iterations,
                    thresh=self.clustering_tolerance
                )
                gev_init = microstate_clusterer.compute_gev(data=all_data, maps=maps_init)
                # Store the results for this initialization in the dictionary
                modified_kmeans_results[init] = {
                    'maps': maps_init,
                    'gev': gev_init,
                    'residual': residual_init
                }

                print(f'Found {self.number_of_maps} Microstate Maps')
                print(f'GEV: {gev_init}')
                self.LogWindow.append_log(
                    f"Data Clustered [{init + 1}/{self.number_of_repeats}]\n"
                    f"✓ Global Explained Variance: {100 * gev_init}%"
                )

                # Update the best results if current gev is higher
                if gev_init > self.best_gev:
                    self.best_residual, self.best_gev, self.best_maps = residual_init, gev_init, maps_init

                # Check if the process should be stopped
                if not self.LogWindow.running:
                    break

            modified_kmeans_results['best'] = {
                'maps': self.best_maps,
                'gev': self.best_gev,
                'residual': self.best_residual
            }
            # self.best_maps = modified_kmeans_results['best']['maps']
            # self.best_gev = modified_kmeans_results['best']['gev']
            # self.best_residual = modified_kmeans_results['best']['residual']

        else:
            initial_maps = data_initializer.initialize_cluster_centers(
                maps2use=self.maps2use, n_states=self.number_of_maps, initializer=self.initializer
            )
            clustering_instance = microstate_clusterer.get_clustering_instance(
                maps2use=self.maps2use,
                initial_maps=initial_maps,
                method=self.clustering_method,
                n_states=self.number_of_maps,
                clustering_option=self.clustering_option
            )

            for init in range(self.number_of_repeats):
                self.LogWindow.update_progress(
                    value=init,
                    text=f"Clustering [{init + 1}/{self.number_of_repeats}] - "
                    f"Best Global Explained Variance: {100 * self.best_gev:.3f}%"
                )

                if self.clustering_method in ['K-Means Clustering', 'X-Means Clustering']:
                    clustering_instance.process()
                    residual_init = clustering_instance.get_total_wce()
                    maps_init = clustering_instance.get_centers()
                elif self.clustering_method in ['PCA + K-Means Clustering', 'Autoencoder + K-Means Clustering']:
                    clustering_instance.process()
                    cluster_labels = clustering_instance.get_clusters()
                    # Flatten the cluster labels
                    cluster_labels_flat = np.zeros(len(self.maps2use))
                    for cluster_id, cluster in enumerate(cluster_labels):
                        cluster_labels_flat[cluster] = cluster_id
                    # Find original centroids
                    maps_init = microstate_clusterer.find_original_centroids(
                        self.maps2use, cluster_labels_flat, self.number_of_maps)
                    # Calculate residuals
                    residual_init = 0  # self.calculate_residuals(maps2use, autoencoder)

                maps_init = np.array(maps_init)
                gev_init = microstate_clusterer.compute_gev(data=self.maps2use, maps=maps_init)

                microstate_labeler = MicrostateLabeler(
                    microstate_maps=maps_init, eeg_info=self.eeg_info, microstate_maps_path=self.microstate_maps_path
                )
                micro_labels, labels_overall_confidence = microstate_labeler.do_labeling()

                # if gev_r > best_gev:
                if labels_overall_confidence > self.best_confidence:
                    self.best_gev = gev_init
                    self.best_maps = maps_init
                    self.best_residual = residual_init

                # Check if the process should be stopped
                if not self.LogWindow.running:
                    break

        # Save Best Maps
        microstate_clusterer.microstates2csv(
            microstate_maps=self.best_maps, eeg_info=self.eeg_info, microstate_maps_path=self.microstate_maps_path)

        print(f'Global Explained Variance: {self.best_gev}')
        # Set clustering flag
        self.done_clustering = True
        self.LogWindow.process_finished(
            f"✓ The data has been successfully clustered into {self.number_of_maps} microstates."
            f"\nBest Global Explained Variance Achieved: {100 * self.best_gev:.3f}%"
        )

        # Optionally save the clustered data
        if self.auto_save:
            self.save_config()
            self.save_tbx()

    def do_labeling(self):
        # Initialize microstate labeler
        self.microstate_labeler = MicrostateLabeler(
            microstate_maps=self.best_maps, eeg_info=self.eeg_info, microstate_maps_path=self.microstate_maps_path
        )

        # Perform labeling
        self.LogWindow.setup_progress_dialog(
            window_title="Labeling ...",
            label_text="Labeling microstates ...",
            max_value=1
        )
        self.micro_labels, self.labels_overall_confidence = self.microstate_labeler.do_labeling()

        # Set labeling flag
        self.done_labeling_microstates = True
        self.LogWindow.process_finished("✓ Microstates have been successfully labeled!")

    def do_backfitting(self):
        print('\nBackfitting ...')

        # Create segmentation directory if it doesn't exist
        self.segmentation_path = os.path.join(self.save_dir, f"{self.study_name}_segmentation")
        if not os.path.exists(self.segmentation_path):
            os.makedirs(self.segmentation_path)

        # Create an instance of the SegmentationIO class
        data_io = DataIO()
        segmentation_io = SegmentationIO()
        microstate_backfitter = MicrostateBackfitter(
            study_name=self.study_name,
            preprocessed_data_path=self.preprocessed_data_path,
            microstate_maps=self.best_maps,
            backfit_to=self.backfit_to,
            filter_segments=self.filter_segments,
            filter_segments_option=self.filter_segments_option,
            identify_short_window=self.identify_short_window,
            micro_labels=self.micro_labels,
            segmentation_path=self.segmentation_path,
            extension=self.extension,
            datatype=self.datatype,
            sample_rate=self.sample_rate,
            smoothing_parameters=[self.epsilon, self.b, self.lamb],
            export_format=self.export_format,
        )

        # Perform segmentation
        if self.identify_short_window:
            # Create an instance of the progress dialog
            self.LogWindow.setup_progress_dialog(
                window_title="Backfitting ...",
                label_text="Identifying the optimal length of the smoothing window ...",
                max_value=len(self.list_eegs_path)
            )

            rm_max_len = 50
            len_win2rm_list = list(range(0, rm_max_len, int(1000 / self.sample_rate)))
            similarity_scores = np.empty((len(self.list_eegs_path), len(len_win2rm_list)))

            # Iterate through EEG files
            for eeg_idx, (eeg_path, eeg_name) in enumerate(zip(self.list_eegs_path, self.list_eegs)):
                self.LogWindow.update_progress(value=eeg_idx, text=f"{eeg_name}")

                eeg = data_io.load_eegs(eeg_path=eeg_path, datatype=self.datatype)

                # Compute similarity scores for different segment removal lengths
                for idx_win2rm, len_win2rm in enumerate(len_win2rm_list):
                    similarity_scores[eeg_idx, idx_win2rm] = microstate_backfitter.get_similarity_score(
                        eeg=eeg, rm_max_len=len_win2rm
                    )

                if not self.LogWindow.running:  # Check if the process should be stopped
                    break

            # Identify optimal length filter based on similarity scores
            remove_segments_less_than = microstate_backfitter.identify_optimal_length_filter(
                similarity_scores=similarity_scores
            )

        else:
            if self.filter_segments:
                remove_segments_less_than = self.remove_segments_less_than
            else:
                remove_segments_less_than = 0

        remove_segments_less_than_ms = remove_segments_less_than * (1000 / self.sample_rate)

        if self.backfit_to == 'peaks':
            backfit_to_text = "Backfitting microstates to the local peaks of the global field power."
        else:
            backfit_to_text = "Backfitting microstates to all time points."

        if self.filter_segments_option == 'remove':
            filter_segments_option_text = f"Removing segments with less than " \
                                          f"{remove_segments_less_than_ms}ms in duration."
        elif self.filter_segments_option == 'replace_high':
            filter_segments_option_text = f"Replacing segments with less than {remove_segments_less_than_ms}ms" \
                f"by the nearby microstate with higher occurrence."
        elif self.filter_segments_option == 'replace_half':
            filter_segments_option_text = f"Replacing segments with less than {remove_segments_less_than_ms}ms" \
                f"by half by the previous and half by the next dominant microstate."
        elif self.filter_segments_option == 'smooth':
            filter_segments_option_text = f"Smoothing segments with window size {remove_segments_less_than_ms}ms" \
                                          f" and lambda {self.lamb}."
        else:
            filter_segments_option_text = "No filtering applied to short segments."

        # Create an instance of the progress dialog
        self.LogWindow.setup_progress_dialog(
            window_title="Backfitting ...",
            label_text="Backfitting microstates to data ...",
            max_value=len(self.list_eegs_path)
        )

        # Log the backfitting progress
        self.LogWindow.append_log(
            f"Microstates Backfitting Settings:\n"
            f"* {backfit_to_text}\n"
            f"* {filter_segments_option_text}", log_type='settings'
        )

        # Iterate through EEG files
        for eeg_idx, (eeg_path, eeg_name) in tqdm(enumerate(zip(self.list_eegs_path, self.list_eegs)),
                                                  total=len(self.list_eegs_path)):
            self.LogWindow.update_progress(value=eeg_idx, text=f"{eeg_name}")

            eeg = data_io.load_eegs(eeg_path=eeg_path, datatype=self.datatype)

            labeled_segmentation, trial_filename, trial_times, segmentation_fit = microstate_backfitter.\
                perform_segmentation(eeg=eeg, eeg_name=eeg_name, remove_segments_less_than=remove_segments_less_than)

            segmentation_io.export_segmentation(
                output_folder=self.segmentation_path,
                filename=trial_filename,
                segmentation_array=labeled_segmentation,
                time_array=trial_times,
                export_format=self.export_format
            )

            # Log the backfitting progress
            self.LogWindow.append_log(
                f"Microstates Backfitted [{eeg_idx + 1}/{len(self.list_eegs_path)}]\n"
                f"✓ Data: {eeg_name}"
            )

            if not self.LogWindow.running:  # Check if the process should be stopped
                break

        # Set backfitting flag
        self.done_backfitting = True
        self.LogWindow.process_finished("✓ Microstates have been successfully backfitted to the data!")

        # Optionally save the results
        if self.auto_save:
            self.save_config()
            self.save_tbx()

    def extract_features(self):
        print('\nExtracting Features ...')

        # Create directory for extracted features if it doesn't exist
        self.feature_list_dictionary = {
            "OCC": "Frequency of Occurrence (Hz)", "DUR": "Mean Microstate Duration (ms)",
            "COV": "Microstate Coverage (%)", "GEV": "Microstate Global Explained Variance (%)",
            "TP": "Transition Probability", "SE": "Sequence Entropy", "LZC": "Sequence Lempel-Ziv Complexity",
            "ER": "Sequence Entropy Representation"
        }
        self.extracted_features_path = os.path.join(self.save_dir, f"{self.study_name}_extracted_features")
        if not os.path.exists(self.extracted_features_path):
            os.makedirs(self.extracted_features_path)

        # Find segmentation files
        segmentation_list_path, segmentation_list_filename = DataIO().find_data(
            input_folder=self.segmentation_path,
            extension=self.export_format
        )

        # Create an instance of the progress dialog
        self.LogWindow.setup_progress_dialog(
            window_title="Extracting Features ...",
            label_text="Extracting features for data ...",
            max_value=len(segmentation_list_path)
        )

        self.LogWindow.append_log(
            f"Feature Extraction Settings:\n"
            f"* Features to Extract: {self.feature_list}\n"
            f"* Feature Type: {self.feature_mode}", log_type='settings'
        )

        for feature_type in self.feature_types:
            static_features_dfs = pd.DataFrame()
            dynamic_features_dfs = pd.DataFrame()
            for segmentation_idx, (segmentation_path, segmentation_name) in tqdm(
                    enumerate(zip(segmentation_list_path, segmentation_list_filename)),
                    total=len(segmentation_list_path)):
                self.LogWindow.update_progress(value=segmentation_idx, text=f"{segmentation_name}")

                # Load segmentation array
                segmentation_array = SegmentationIO().load_segmentation(
                    segmentation_path=segmentation_path, import_format='.csv'
                )
                if feature_type == 'real':
                    input_sequence = segmentation_array
                else:
                    input_sequence = FeatureHelper().generate_synthetic_sequence(
                        input_sequence=segmentation_array, method=feature_type
                    )

                # Extract static features
                if 'static' in self.feature_mode:
                    feature_extractor = FeatureExtractor(
                        input_sequence=input_sequence,
                        sampling_rate=self.sample_rate,
                        window_size=self.window_size,
                        feature_mode='static'
                    )

                    if 'GEV' in self.feature_list:
                        data_io = DataIO()
                        eeg_path, _ = data_io.find_data(
                            input_folder=self.preprocessed_data_path,
                            extension=self.extension,
                            pattern=f"*{segmentation_name}*"
                        )
                        eeg = data_io.load_eegs(eeg_path=eeg_path[0], datatype=self.datatype)
                        if self.datatype == 'epoched':
                            underscore_index = segmentation_name.rfind('_')
                            trial_number = segmentation_name[underscore_index + 1:]
                            trial_data = np.squeeze(eeg[int(trial_number)].get_data())
                        else:
                            eeg_data = data_io.get_eeg_data(eeg=eeg, datatype=self.datatype)

                        output_features = feature_extractor.extract_microstate_features(
                            filename=segmentation_name,
                            feature_list=self.feature_list,
                            eeg_data=trial_data if self.datatype == 'epoched' else eeg_data,
                            microstate_maps=self.best_maps,
                            microstate_labels=self.micro_labels,
                            word_size=self.word_size
                        )
                    else:
                        output_features = feature_extractor.extract_microstate_features(
                            filename=segmentation_name, feature_list=self.feature_list, word_size=self.word_size
                        )

                    if segmentation_idx == 0:
                        static_features_dfs = output_features
                    else:
                        static_features_dfs = pd.concat([static_features_dfs, output_features], ignore_index=True)

                # Extract dynamic features
                if 'dynamic' in self.feature_mode:
                    feature_extractor = FeatureExtractor(
                        input_sequence=input_sequence,
                        sampling_rate=self.sample_rate,
                        window_size=self.window_size,
                        feature_mode='dynamic'
                    )

                    if 'GEV' in self.feature_list:
                        data_io = DataIO()
                        eeg_path = os.path.join(self.preprocessed_data_path, f"{segmentation_name}{self.extension}")
                        eeg = data_io.load_eegs(eeg_path=eeg_path, datatype=self.datatype)
                        eeg_data = data_io.get_eeg_data(eeg=eeg, datatype=self.datatype)
                        output_features = feature_extractor.extract_microstate_features(
                            filename=segmentation_name,
                            feature_list=self.feature_list,
                            eeg_data=eeg_data,
                            microstate_maps=self.best_maps,
                            microstate_labels=self.micro_labels
                        )
                    else:
                        output_features = feature_extractor.extract_microstate_features(
                            filename=segmentation_name, feature_list=self.feature_list
                        )

                    if segmentation_idx == 0:
                        dynamic_features_dfs = output_features
                    else:
                        dynamic_features_dfs = pd.concat([dynamic_features_dfs, output_features], ignore_index=True)

                # Log the feature extraction progress
                self.LogWindow.append_log(
                    f"Features Extracted [{segmentation_idx + 1}/{len(self.list_eegs_path)}]\n"
                    f"✓ Data: {segmentation_name}"
                )

                if not self.LogWindow.running:  # Check if the process should be stopped
                    break

            # Export extracted features
            feature_io = FeatureIO()
            if 'static' in self.feature_mode:
                feature_io.export_features(
                    features_df=static_features_dfs,
                    feature_type=feature_type,
                    feature_mode='static',
                    output_folder=self.extracted_features_path,
                    export_format=self.export_format
                )
            if 'dynamic' in self.feature_mode:
                feature_io.export_features(
                    features_df=dynamic_features_dfs,
                    feature_type=feature_type,
                    feature_mode='dynamic',
                    output_folder=self.extracted_features_path,
                    export_format=self.export_format
                )

        # Set extracting features flag
        self.done_extracting_features = True
        self.LogWindow.process_finished("✓ All features have been successfully extracted!")

        # Optionally save the results
        if self.auto_save:
            self.save_config()
            self.save_tbx()

    def source_localize_microstates(self):
        print("\nCalculating Source Time Series ...")

        # Determine the subjects directory based on the anatomy choice
        if self.use_anatomy == "fsaverage":
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.anatomy_subjects_dir = os.path.dirname(fs_dir)
        elif self.use_anatomy == "individual":
            self.anatomy_subjects_dir = self.individual_subjects_dir

        # Initialize the source localizer
        source_localizer = SourceLocalizer(
            subjects_dir=self.anatomy_subjects_dir,
            localized_sources_path=self.localized_sources_path,
            preprocessed_data_path=self.preprocessed_data_path,
            segmentation_path=self.segmentation_path,
            use_anatomy=self.use_anatomy,
            extension=self.extension,
            datatype=self.datatype,
            spacing=self.spacing,
            inverse_method=self.inverse_method,
            microstate_maps=self.best_maps,
            nperm=self.nperm
        )
        # Perform source localization
        source_localizer.run_source_localization()

        # Set source localization flag
        self.done_source_localization = True

        # Optionally save the results
        if self.auto_save:
            self.save_config()
            self.save_tbx()

    def source_microstate_correlation(self, method='tess'):
        print("\nCorrelating sources and microstates ...")

        # Check if anatomy subjects directory is available; if not, fetch the fsaverage directory
        try:
            print(self.anatomy_subjects_dir)
        except AttributeError:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.anatomy_subjects_dir = os.path.dirname(fs_dir)

        # Initialize the source localizer
        source_localizer = SourceLocalizer(
            subjects_dir=self.anatomy_subjects_dir,
            localized_sources_path=self.localized_sources_path,
            preprocessed_data_path=self.preprocessed_data_path,
            segmentation_path=self.segmentation_path,
            use_anatomy=self.use_anatomy,
            extension=self.extension,
            datatype=self.datatype,
            spacing=self.spacing,
            inverse_method=self.inverse_method,
            microstate_maps=self.best_maps,
            nperm=self.nperm
        )

        # Identify microstates sources using the specified method
        source_localizer.identify_microstates_sources(source_method=method)

        # Set source-microstate correlation flag
        self.done_source_microstate_correlation = True

        # Optionally save the results
        if self.auto_save:
            self.save_config()
            self.save_tbx()

    # 	TODO: Need to expand this function to include the following:
    # 	1. Load the source localized time series
    # 	2. Load the microstates
    # 	3. Run TESS and Averaging based on the user input
    # 	4. Save the results

    def save_config(self):
        """
        Update and save the current configuration settings based on the COMET object attributes.
        """

        # Update the config dictionary with the current attribute values
        self.config["io_config"]["study_name"] = self.study_name
        self.config["io_config"]["input_folder"] = self.input_folder
        self.config["io_config"]["channel_location_dir"] = self.channel_location_dir
        self.config["io_config"]["extension"] = self.extension
        self.config["io_config"]["pattern_content"] = self.pattern_content
        self.config["io_config"]["datatype"] = self.datatype
        self.config["io_config"]["output_folder"] = self.output_folder

        self.config["preprocessing_config"]["filter_data"] = str(self.filter_data)
        self.config["preprocessing_config"]["filter_method"] = self.filter_method
        self.config["preprocessing_config"]["lowcut_freq"] = str(self.lowcut_freq)
        self.config["preprocessing_config"]["highcut_freq"] = str(self.highcut_freq)
        self.config["preprocessing_config"]["downsample_data"] = str(self.downsample_data)
        self.config["preprocessing_config"]["sample_rate"] = str(self.sample_rate)
        self.config["preprocessing_config"]["remove_channels"] = str(self.remove_channels)
        self.config["preprocessing_config"]["chan2rm"] = str(self.chan2rm)

        self.config["clustering_config"]["smoothing_gfp"] = str(self.smoothing_gfp)
        self.config["clustering_config"]["smoothing_distance"] = str(self.smoothing_distance)
        self.config["clustering_config"]["number_of_maps"] = str(self.number_of_maps)
        if self.number_of_maps == "auto":
            self.config["clustering_config"]["kmin"] = str(self.kmin)
            self.config["clustering_config"]["kmax"] = str(self.kmax)
            self.config["clustering_config"]["stopping_mode"] = self.stopping_mode
            self.config["clustering_config"]["stopping_parameter"] = str(self.stopping_parameter)
        self.config["clustering_config"]["use_percentages"] = str(self.use_percentages)
        self.config["clustering_config"]["initializer"] = self.initializer
        self.config["clustering_config"]["clustering_method"] = self.clustering_method
        self.config["clustering_config"]["max_iterations"] = str(self.max_iterations)
        self.config["clustering_config"]["clustering_tolerance"] = str(self.clustering_tolerance)
        self.config["clustering_config"]["clustering_option"] = self.clustering_option
        self.config["clustering_config"]["number_of_repeats"] = str(self.number_of_repeats)

        self.config["backfitting_config"]["backfit_to"] = self.backfit_to
        self.config["backfitting_config"]["identify_short_window"] = str(self.identify_short_window)
        self.config["backfitting_config"]["filter_segments"] = str(self.filter_segments)
        self.config["backfitting_config"]["remove_segments_less_than"] = str(self.remove_segments_less_than)
        self.config["backfitting_config"]["filter_segments_option"] = self.filter_segments_option
        self.config["backfitting_config"]["epsilon"] = str(self.epsilon)
        self.config["backfitting_config"]["b"] = str(self.b)
        self.config["backfitting_config"]["lamb"] = str(self.lamb)

        self.config["features_config"]["export_format"] = self.export_format
        self.config["features_config"]["feature_list"] = ', '.join(self.feature_list)
        self.config["features_config"]["feature_mode"] = ', '.join(self.feature_mode)
        self.config["features_config"]["feature_types"] = ', '.join(self.feature_types)
        self.config["features_config"]["window_size"] = str(self.window_size)

        self.config["source_config"]["inverse_method"] = self.inverse_method
        self.config["source_config"]["nperm"] = str(self.nperm)
        self.config["source_config"]["spacing"] = self.spacing
        self.config["source_config"]["source_localization_method"] = self.source_localization_method
        self.config["source_config"]["anatomy_subjects_dir"] = self.anatomy_subjects_dir

        with open(self.config_path, 'w+') as configfile:
            self.config.write(configfile)

    def save_tbx(self):
        """
        Save current EEG-COMET parameters for future use.
        """
        # Exclude LogWindow from pickling
        self.log_text = self.LogWindow.ui.log_text_area.toPlainText()
        log_window = self.LogWindow
        self.LogWindow = None

        # Serialize and save the TBX object
        with open(self.tbx_object_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        # Restore LogWindow after pickling
        self.LogWindow = log_window
