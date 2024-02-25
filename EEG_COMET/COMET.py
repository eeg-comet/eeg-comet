
import os
import mne
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from functions.data_utils.data_io import DataIO
from functions.data_utils.data_preprocessor import DataPreprocessor
from functions.data_utils.data_initializer import DataInitializer
from functions.features_utils.feature_extractor import FeatureExtractor
from functions.features_utils.feature_io import FeatureIO
from functions.backfitting_utils.segmentation_io import SegmentationIO
from functions.backfitting_utils.microstate_backfitter import MicrostateBackfitter
from functions.clustering_utils.autopilot_clusterer import AutopilotClusterer
from functions.clustering_utils.microstate_clusterer import MicrostateClusterer
from functions.clustering_utils.clusterer_optimizer import ClustererOptimizer
from functions.clustering_utils.microstate_labeler import MicrostateLabeler
from functions.sourcelocalization_utils.source_localizer import SourceLocalizer

from gui.progress_dialog import ProgressDialog


class COMET:
    """
    The COMET class represents an instance of the EEG-COMET application.
    It provides methods to load configuration settings, perform various processes
    such as preprocessing, clustering, labeling, backfitting, feature extraction,
    source localization, and source-microstate correlation.
    If auto_save is True, the COMET will automatically save itself after each process
    """
    def __init__(self, config: dict = None, auto_save: bool = True):

        if config:
            self.load_config(config)
        self.log_text = []  # Initialize log_text as an empty list
        self.done_preprocessing: bool = False
        self.done_clustering: bool = False
        self.done_labeling_microstates: bool = False
        self.done_backfitting: bool = False
        self.done_extracting_features: bool = False
        self.done_source_localization: bool = False
        self.done_source_microstate_correlation: bool = False
        self.auto_save = auto_save

    def load_config(self, config):
        """
        Load configuration settings from a dictionary.
        """
        # Base configs
        self.config: dict = config
        self.study_name: str = config['base']['study_name']
        self.input_folder: str = config['base']['input_folder']
        self.channel_location_dir: str = config['base']['channel_location_dir']
        self.output_folder: str = config['base']['output_folder']
        # self.log_text: str = config['base']['log_text']

        self.save_dir: str = os.path.join(self.output_folder, self.study_name)
        assert self.study_name != "", "study name cannot be empty"
        self.preprocessed_data_path: str = os.path.join(self.save_dir, f"{self.study_name}_preprocessed_data")
        self.eeg_info_path: str = os.path.join(self.save_dir, "eeg_info.pkl")

        # Load new study configs
        self.load_all_files: bool = config.getboolean('load_new_study', 'load_all_files')
        if self.load_all_files:
            self.pattern_content: str = config.get('load_new_study', 'pattern_content', fallback='')
        self.extension: str = config['load_new_study']['extension']
        self.datatype: str = config['load_new_study']['datatype']
        self.filter_data: bool = config.getboolean('load_new_study', 'filter_data')
        self.filter_method: str = config['load_new_study']['filter_method'] if self.filter_data else ''
        self.lowcut_freq: int = config.getint('load_new_study', 'lowcut_freq') if self.filter_data else ''
        self.highcut_freq: int = config.getint('load_new_study', 'highcut_freq') if self.filter_data else ''
        self.downsample_data: bool = config.getboolean('load_new_study', 'downsample_data')
        self.sample_rate: int = config.getint('load_new_study', 'sample_rate') if self.downsample_data else ''
        self.remove_channels: bool = config.getboolean('load_new_study', 'remove_channels')
        self.ch2rm: str = config['load_new_study']['ch2rm'] if self.remove_channels else 'missing'

        # Clustering configs
        self.smoothing_gfp: bool = config.getboolean('do_clustering', 'smoothing_gfp')
        self.smoothing_distance: int = config.getint('do_clustering',
                                                     'smoothing_distance') if self.smoothing_gfp else ''
        number_of_maps: str = config['do_clustering']['number_of_maps']
        self.number_of_maps = number_of_maps if number_of_maps == 'auto' else int(number_of_maps)
        self.choose_number_of_maps: str = 'Auto' if number_of_maps == 'auto' else 'User'
        self.stopping_mode: str = config['do_clustering']['stopping_mode'] if self.number_of_maps == 'auto' else ''
        self.stopping_parameter: float = config.getfloat('do_clustering',
                                                         'stopping_parameter') if self.number_of_maps == 'auto' else ''
        self.kmin: int = config.getint('do_clustering', 'kmin') if self.number_of_maps == 'auto' else ''
        self.kmax: int = config.getint('do_clustering', 'kmax') if self.number_of_maps == 'auto' else ''
        self.initializer: str = config['do_clustering']['initializer']
        self.clustering_method: str = config['do_clustering']['clustering_method']
        self.max_iterations: int = config.getint('do_clustering', 'max_iterations')
        self.clustering_tolerance: float = config.getfloat('do_clustering', 'clustering_tolerance')
        need_options = ['X-Means Clustering', 'Agglomerative Hierarchical Clustering',
                                   'K-Means Clustering', 'PCA + K-Means Clustering',
                                   'Autoencoder + K-Means Clustering', ]
        self.clustering_option: str = config['do_clustering'][
            'clustering_option'] if self.clustering_method in need_options else ''
        self.number_of_repeats: int = config.getint('do_clustering', 'number_of_repeats')
        self.microstate_maps_path: str = os.path.join(self.save_dir, 'microstate_maps.csv')
        self.use_percentages: int = config.getint('do_clustering', 'use_percentages')

        # Backfitting configs
        self.backfit_to: str = config['do_backfitting']['backfit_to']
        self.identify_short_window: bool = config.getboolean('do_backfitting', 'identify_short_window')
        self.filter_segments: bool = config.getboolean('do_backfitting', 'filter_segments')
        self.remove_segments_less_than: int = config.getint('do_backfitting',
                                                            'remove_segments_less_than') if self.filter_segments else ''
        self.filter_segments_option: str = config['do_backfitting'][
            'filter_segments_option'] if self.filter_segments else ''
        self.epsilon: float = config.getfloat('do_backfitting', 'epsilon')
        self.b: int = config.getint('do_backfitting', 'b')
        self.lamb: int = config.getint('do_backfitting', 'lamb')
        self.micro_labels = [chr(i) for i in range(ord('A'), ord('A') + self.number_of_maps)]

        # Feature extraction configs
        self.extracted_features_path: str = os.path.join(self.save_dir, f"{self.study_name}_extracted_features")
        self.segmentation_path: str = os.path.join(self.save_dir, f"{self.study_name}_segmentation")
        self.export_format: str = config['extract_features']['export_format']
        self.feature_list = [x for x in config['extract_features']['feature_list'].split(',')]
        self.save_transitions_bool: bool = True if 'TP' in self.feature_list else False
        self.window_size: int = config.getint('extract_features', 'window_size') if 'OCC' in self.feature_list else ''

        # Source localization configs
        self.localized_sources_path: str = os.path.join(self.save_dir, f"{self.study_name}_localized_sources")
        self.inverse_method: str = config['source_localize_microstates']['inverse_method']
        self.nperm: int = config.getint('source_localize_microstates', 'nperm')
        self.spacing: str = config['source_localize_microstates']['spacing']
        self.source_localization_method: str = config['source_localize_microstates']['source_localization_method']
        self.anatomy_subjects_dir: str = config['source_localize_microstates']['anatomy_subjects_dir']

    def append_log(self, log, log_type='info'):
        """
        Appends a log entry with the current date and time to the log_text list.
        """
        current_date = datetime.now().strftime("%d/%m/%y")
        current_time = datetime.now().strftime("%I:%M %p")
        separator = "******************************************************"
        if log_type == 'settings':
            current_log_text = f"\n{separator}\n{log}\n{separator}\n"
        else:
            current_log_text = f"[{current_date} {current_time}]: {log}\n"
        self.log_text.append(current_log_text)

    @staticmethod
    def setup_progress_dialog(window_title, label_text, max_value):
        """
        Sets up a progress dialog with the specified window title, label text, and maximum value.
        """
        # Initialize the progress dialog
        progress_dialog = ProgressDialog()
        progress_dialog.set_window_title(window_title)
        progress_dialog.set_label_text(label_text)
        progress_dialog.show()
        progress_dialog.start_process(max_value)
        progress_dialog.update_progress(0)
        # Create a tqdm progress bar
        progress_bar = tqdm(total=max_value, ncols=100, position=0, leave=True)
        return progress_dialog, progress_bar

    @staticmethod
    def update_progress(progress_dialog, progress_bar, value, description):
        """
        Updates the progress of the progress dialog and progress bar with the specified value and description.
        """
        progress_dialog.set_line_edit_text(description)
        progress_dialog.update_progress(value)
        progress_bar.set_description(description)
        progress_bar.update(1)

    def load_raw(self):
        """
        Locate EEG file paths.
        """
        if self.load_all_files:
            self.pattern = '*'
        else:
            self.pattern = '*' + self.pattern_content + '*'

        self.list_eegs_path, self.list_eegs = DataIO().find_data(self.input_folder, self.extension, self.pattern)

    # self.list_eegs = [os.path.basename(x).split('.')[0] for x in self.list_eegs_path]
    # assert self.list_eegs_path, 'eeg list is empty'

    def do_preprocessing(self):
        print('\nPreprocessing ...')

        # Create preprocessed data directory if it doesn't exist
        if not os.path.exists(self.preprocessed_data_path):
            os.makedirs(self.preprocessed_data_path)

        # Initialize variables
        length_all_data = []
        data_io = DataIO()
        progress_dialog, progress_bar = self.setup_progress_dialog(
            window_title="Preprocessing ...",
            label_text="Preprocessing data: ",
            max_value=len(self.list_eegs)
        )

        # Log the preprocessing progress
        self.append_log(
            f"Study Created\n"
            f"✓ Study Name: {self.study_name}\n"
            f"✓ Input Path: {self.input_folder}\n"
            f"✓ Found {len(self.list_eegs_path)} {self.datatype} EEG data with {self.extension} extension."
        )

        self.append_log(
            f"EEG Preprocessing Settings:\n"
            f"* Channels to Remove: {self.ch2rm}\n"
            f"* Bandpass Filter: {self.filter_method.upper()} Method ({self.lowcut_freq}Hz and {self.highcut_freq}Hz)\n"
            f"* Downsampling Rate: {self.sample_rate}Hz", log_type='settings'
        )

        # Create an instance of the DataPreprocessor class
        preprocessor = DataPreprocessor()

        # Iterate through EEG files
        for eeg_idx, (eeg_path, eeg_name) in enumerate(zip(self.list_eegs_path, self.list_eegs)):
            self.update_progress(progress_dialog, progress_bar, eeg_idx, f"{eeg_name}")

            # Preprocess EEG data
            eeg, preprocessed_data, length_data, eeg_info, channels2remove = preprocessor.preprocess_eegs(
                eeg_path,
                self.list_eegs_path,
                self.extension,
                self.datatype,
                self.channel_location_dir,
                self.filter_data,
                self.filter_method,
                self.lowcut_freq,
                self.highcut_freq,
                self.downsample_data,
                self.sample_rate,
                self.ch2rm
            )

            # Save EEG info
            self.eeg_info = eeg.info
            if not self.sample_rate:
                self.sample_rate = int(self.eeg_info['sfreq'])
            self.channels2remove = channels2remove
            data_io.save_eeg_info(self.eeg_info_path, eeg_info)

            # Save EEG object
            ch_names, ch_location = list(self.eeg_info['ch_names']), self.eeg_info['chs']
            n_chan = len(ch_names)
            self.n_chan = n_chan
            self.ch_names = ch_names
            # name = os.path.basename(eeg_path)
            name = os.path.splitext(eeg_name)[0]
            save_path = os.path.join(self.preprocessed_data_path, name)
            self.length_all_data = np.append(length_all_data, int(length_data))
            data_io.export_eegs(eeg, save_path, self.extension, self.datatype)

            # Log the preprocessing progress

            self.append_log(
                f"EEG Preprocessed [{eeg_idx + 1}/{len(self.list_eegs_path)}]\n"
                f"✓ Data: {eeg_name} - Length: {int(length_data / self.sample_rate)} sec"
            )
            # Check if the process should be stopped
            if not progress_dialog.running:
                break

        # Close progress dialogs
        progress_dialog.close()
        progress_bar.close()

        # Set preprocessing flag
        self.done_preprocessing = True

        # Optionally save the preprocessed data
        if self.auto_save:
            self.save_tbx()

    def do_autopilot(self):

        autopilot_clusterer = AutopilotClusterer(
            self.save_dir, self.study_name, self.extension, self.datatype
        )
        autopilot_clusterer.run_autopilot()

    def do_clustering(self):
        print('\nClustering ...')

        # Calculate minimum distance size if smoothing GFP is enabled
        if self.smoothing_gfp:
            self.min_distance_size = int(self.smoothing_distance / (1000 / self.sample_rate))
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
        progress_dialog, progress_bar = self.setup_progress_dialog(
            window_title="Clustering ...",
            label_text=f"Performing Clustering: Generating {self.number_of_maps} Microstate Maps.",
            max_value=self.number_of_repeats
        )
        data_initializer = DataInitializer()
        microstate_clusterer = MicrostateClusterer(
            self.number_of_repeats,
            self.max_iterations,
            self.clustering_tolerance
        )

        # Generate maps and peaks automatically if number_of_maps is set to 'auto'
        if self.number_of_maps == 'auto':
            data_initializer = DataInitializer()
            self.maps2use, self.peaks2use = data_initializer.generate_maps_and_peaks(
                self.preprocessed_data_path,
                self.extension,
                self.datatype,
                self.use_percentages,
                self.min_distance_size
            )
            clusterer_optimizer = ClustererOptimizer(
                self.maps2use,
                self.min_distance_size,
                self.number_of_repeats,
                self.kmin,
                self.kmax,
                self.preprocessed_data_path,
                self.extension,
                self.datatype,
                self.clustering_tolerance,
                self.max_iterations
            )

            # Find optimal number of maps
            self.optimal_k, self.k_values, self.target_values = clusterer_optimizer.find_optimal_k(
                self.stopping_mode, self.stopping_parameter)
            self.number_of_maps = self.optimal_k
            print(f'Result: n_states = {self.number_of_maps}')

        # Perform clustering
        self.maps2use, self.peaks2use = data_initializer.generate_maps_and_peaks(
            self.preprocessed_data_path,
            self.extension,
            self.datatype,
            self.use_percentages,
            self.min_distance_size
        )

        if self.clustering_method == 'PCA + K-Means Clustering':
            # Extract features using PCA
            self.maps2use = microstate_clusterer.extract_features_with_pca(
                np.transpose(self.maps2use), pca_components=self.n_pca)

        if self.clustering_method == 'Autoencoder + K-Means Clustering':
            # Extract features using Autoencoder
            self.maps2use, autoencoder = microstate_clusterer.extract_features_with_autoencoder(np.transpose(
                self.maps2use), encoding_dim=10)

        self.initial_maps = data_initializer.initialize_cluster_centers(
            self.maps2use, self.number_of_maps, self.initializer
        )

        all_data, _ = data_initializer.generate_maps_and_peaks(
            self.preprocessed_data_path, self.extension, self.datatype, use_percentages=100
        )

        if self.choose_number_of_maps == "auto":
            k_log = 'will be automatically determined.'
        else:
            k_log = 'is user-predefined.'

        if self.use_percentages is not None:
            cluster_data_log = f"{self.use_percentages}% randomly selected time-points of the data."
        else:
            cluster_data_log = 'the local peaks of the global field power.'

        self.append_log(
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
                self.update_progress(
                    progress_dialog, progress_bar, init,
                    f"Clustering [{init + 1}/{self.number_of_repeats}] - "
                    f"Global Explained Variance: {100 * self.best_gev:.3f}%"
                )

                maps_init, residual_init = microstate_clusterer.modified_kmeans(
                    data=self.maps2use,
                    initial_maps=self.initial_maps,
                    n_states=self.number_of_maps,
                    max_iter=self.max_iterations,
                    thresh=self.clustering_tolerance
                )
                gev_init = microstate_clusterer.compute_gev(all_data, maps_init)
                # Store the results for this initialization in the dictionary
                modified_kmeans_results[init] = {
                    'maps': maps_init,
                    'gev': gev_init,
                    'residual': residual_init
                }

                print(f'Found {self.number_of_maps} Microstate Maps')
                print(f'GEV: {gev_init}')
                self.append_log(
                    f"Data Clustered [{init + 1}/{self.number_of_repeats}]\n"
                    f"✓ Global Explained Variance: {100 * gev_init}%"
                )

                # Update the best results if current gev is higher
                if gev_init > self.best_gev:
                    self.best_residual, self.best_gev, self.best_maps = residual_init, gev_init, maps_init

                # Check if the process should be stopped
                if not progress_dialog.running:
                    break

            # Close progress dialogs
            progress_dialog.close()
            progress_bar.close()

            modified_kmeans_results['best'] = {
                'maps': self.best_maps,
                'gev': self.best_gev,
                'residual': self.best_residual
            }
            # self.best_maps = modified_kmeans_results['best']['maps']
            # self.best_gev = modified_kmeans_results['best']['gev']
            # self.best_residual = modified_kmeans_results['best']['residual']

        else:

            clustering_instance = microstate_clusterer.get_clustering_instance(
                self.maps2use,
                self.initial_maps,
                self.clustering_method,
                self.number_of_maps,
                self.clustering_option
            )

            progress_dialog.set_label_text(f"Clustering {int(self.number_of_maps)} Microstate Maps")
            progress_dialog.update_progress(0)
            for init in range(self.number_of_repeats):
                self.update_progress(
                    progress_dialog, progress_bar, init,
                    f"Clustering [{init + 1}/{self.number_of_repeats}] - "
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
                gev_init = microstate_clusterer.compute_gev(self.maps2use, maps_init)

                microstate_labeler = MicrostateLabeler(maps_init, self.eeg_info, self.microstate_maps_path)
                micro_labels, labels_overall_confidence = microstate_labeler.do_labeling()

                # if gev_r > best_gev:
                if labels_overall_confidence > self.best_confidence:
                    self.best_gev = gev_init
                    self.best_maps = maps_init
                    self.best_residual = residual_init

                # Check if the process should be stopped
                if not progress_dialog.running:
                    break

                # Close progress dialogs
            progress_dialog.close()
            progress_bar.close()

        # Save Best Maps
        microstate_clusterer.microstates2csv(self.best_maps, self.eeg_info, self.microstate_maps_path)

        print(f'Global Explained Variance: {self.best_gev}')
        # Log the clustering progress
        self.append_log(
            f"Clustering Done\n"
            f"✓ The Best Global Explained Variance: {100 * self.best_gev:.3f}%"
        )

        # Set clustering flag
        self.done_clustering = True

        # Optionally save the clustered data
        if self.auto_save:
            self.save_tbx()

    def do_labeling(self):
        # Initialize microstate labeler
        self.microstate_labeler = MicrostateLabeler(self.best_maps, self.eeg_info, self.microstate_maps_path)

        # Perform labeling
        self.micro_labels, self.labels_overall_confidence = self.microstate_labeler.do_labeling()

        # Set labeling flag
        self.append_log("✓ Microstates have been successfully labeled.")
        self.done_labeling_microstates = True

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
            remove_segments_less_than=self.remove_segments_less_than,
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
            progress_dialog, progress_bar = self.setup_progress_dialog(
                window_title="Backfitting ...",
                label_text="Backfitting microstates to data: ",
                max_value=len(self.list_eegs_path)
            )

            rm_max_len = 50
            len_win2rm_list = list(range(0, rm_max_len, int(1000 / self.sample_rate)))
            similarity_scores = np.empty((len(self.list_eegs_path), len(len_win2rm_list)))

            # Iterate through EEG files
            for eeg_idx, (eeg_path, eeg_name) in enumerate(zip(self.list_eegs_path, self.list_eegs)):
                self.update_progress(progress_dialog, progress_bar, eeg_idx, f"{eeg_name}")

                eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)

                # Compute similarity scores for different segment removal lengths
                for idx_win2rm, len_win2rm in enumerate(len_win2rm_list):
                    similarity_scores[eeg_idx, idx_win2rm] = microstate_backfitter.get_similarity_score(eeg, len_win2rm)

                if not progress_dialog.running:  # Check if the process should be stopped
                    break

            progress_dialog.close()
            progress_bar.close()

            # Identify optimal length filter based on similarity scores
            remove_segments_less_than = microstate_backfitter.identify_optimal_length_filter(similarity_scores)

        else:
            remove_segments_less_than = self.remove_segments_less_than

        remove_segments_less_than_ms = remove_segments_less_than * (1000 / self.sample_rate)
        print("Optimal length to remove:", remove_segments_less_than_ms)

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
            filter_segments_option_text = ""

        # Create an instance of the progress dialog
        progress_dialog, progress_bar = self.setup_progress_dialog(
            window_title="Backfitting ...",
            label_text="Backfitting microstates to data: ",
            max_value=len(self.list_eegs_path)
        )

        # Log the backfitting progress
        self.append_log(
            f"Microstates Backfitting Settings:\n"
            f"* {backfit_to_text}\n"
            f"* {filter_segments_option_text}", log_type='settings'
        )

        # Iterate through EEG files
        for eeg_idx, (eeg_path, eeg_name) in enumerate(zip(self.list_eegs_path, self.list_eegs)):
            self.update_progress(progress_dialog, progress_bar, eeg_idx, f"{eeg_name}")

            eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)

            labeled_segmentation, trial_filename, trial_times, segmentation_fit = microstate_backfitter.perform_segmentation(
                eeg, eeg_name, remove_segments_less_than)
            segmentation_io.export_segmentation(
                self.segmentation_path, trial_filename, labeled_segmentation, trial_times, self.export_format
            )

            # Log the backfitting progress
            self.append_log(
                f"Microstates Backfitted [{eeg_idx + 1}/{len(self.list_eegs_path)}]\n"
                f"✓ Data: {eeg_name}"
            )

            if not progress_dialog.running:  # Check if the process should be stopped
                break

        progress_dialog.close()
        progress_bar.close()

        # Set backfitting flag
        self.done_backfitting = True

        # Optionally save the results
        if self.auto_save:
            self.save_tbx()

    def extract_features(self):
        print('\nExtracting Features ...')

        # Create directory for extracted features if it doesn't exist
        self.extracted_features_path = os.path.join(self.save_dir, f"{self.study_name}_extracted_features")
        if not os.path.exists(self.extracted_features_path):
            os.makedirs(self.extracted_features_path)

        # Find segmentation files
        segmentation_list_path, segmentation_list_filename = DataIO().find_data(
            self.segmentation_path,
            self.export_format
        )

        # Create an instance of the progress dialog
        progress_dialog, progress_bar = self.setup_progress_dialog(
            window_title="Extracting Features ...",
            label_text="Extracting features for data: ",
            max_value=len(segmentation_list_path)
        )

        self.append_log(
            f"Feature Extraction Settings:\n"
            f"* Features to Extract: {self.feature_list}\n"
            f"* Feature Type: {self.feature_mode}", log_type='settings'
        )
        for segmentation_idx, (segmentation_path, segmentation_name) in enumerate(zip(segmentation_list_path, segmentation_list_filename)):
            progress_bar.set_description(f"{segmentation_name}")

            self.update_progress(progress_dialog, progress_bar, segmentation_idx, f"{segmentation_name}")

            # Load segmentation array
            segmentation_array = SegmentationIO().load_segmentation(segmentation_path, import_format='.csv')

            # Extract static features
            if 'static' in self.feature_mode:
                feature_extractor = FeatureExtractor(
                    segmentation_array,
                    self.sample_rate,
                    self.window_size,
                    mode='static'
                )

                if 'GEV' in self.feature_list:
                    data_io = DataIO()
                    if self.datatype == 'epoched':
                        underscore_index = segmentation_name.rfind('_')
                        eeg_filename = segmentation_name[:underscore_index]
                        trial_number = segmentation_name[underscore_index + 1:]
                        eeg_path = os.path.join(self.preprocessed_data_path, f"{eeg_filename}{self.extension}")
                        eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
                        trial_data = np.squeeze(eeg[int(trial_number)].get_data())
                    else:
                        eeg_path = os.path.join(self.preprocessed_data_path, f"{segmentation_name}{self.extension}")
                        eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
                        eeg_data = data_io.get_eeg_data(eeg, self.datatype)

                    output_features = feature_extractor.extract_microstate_features(
                        segmentation_name,
                        self.feature_list,
                        trial_data if self.datatype == 'epoched' else eeg_data,
                        self.best_maps,
                        self.micro_labels
                    )
                else:
                    output_features = feature_extractor.extract_microstate_features(
                        segmentation_name, self.feature_list
                    )

                if segmentation_idx == 0:
                    static_features_dfs = output_features
                else:
                    static_features_dfs = pd.concat([static_features_dfs, output_features], ignore_index=True)

            # Extract dynamic features
            if 'dynamic' in self.feature_mode:
                feature_extractor = FeatureExtractor(
                    segmentation_array,
                    self.sample_rate,
                    self.window_size,
                    mode='dynamic'
                )

                if 'GEV' in self.feature_list:
                    data_io = DataIO()
                    eeg_path = os.path.join(self.preprocessed_data_path, f"{segmentation_name}{self.extension}")
                    eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
                    eeg_data = data_io.get_eeg_data(eeg, self.datatype)
                    output_features = feature_extractor.extract_microstate_features(
                        segmentation_name,
                        self.feature_list,
                        eeg_data,
                        self.best_maps,
                        self.micro_labels
                    )
                else:
                    output_features = feature_extractor.extract_microstate_features(
                        segmentation_name, self.feature_list
                    )

                if segmentation_idx == 0:
                    dynamic_features_dfs = output_features
                else:
                    dynamic_features_dfs = pd.concat([dynamic_features_dfs, output_features], ignore_index=True)

            # Log the feature extraction progress
            self.append_log(
                f"Features Extracted [{segmentation_idx + 1}/{len(self.list_eegs_path)}]\n"
                f"✓ Data: {segmentation_name}"
            )

            if not progress_dialog.running:  # Check if the process should be stopped
                break

            progress_dialog.update_progress(segmentation_idx + 1)
            progress_bar.update(1)

        # Close progress dialogs
        progress_dialog.close()
        progress_bar.close()

        # Export extracted features
        feature_io = FeatureIO()
        if 'static' in self.feature_mode:
            feature_io.export_features(
                static_features_dfs,
                'static',
                self.extracted_features_path,
                self.export_format
            )
        if 'dynamic' in self.feature_mode:
            feature_io.export_features(
                dynamic_features_dfs,
                'dynamic',
                self.extracted_features_path,
                self.export_format
            )

        # Set extracting features flag
        self.done_extracting_features = True

        # Optionally save the results
        if self.auto_save:
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
        source_localizer.identify_microstates_sources(method)

        # Set source-microstate correlation flag
        self.done_source_microstate_correlation = True

        # Optionally save the results
        if self.auto_save:
            self.save_tbx()

    # 	TODO: Need to expand this function to include the following:
    # 	1. Load the source localized time series
    # 	2. Load the microstates
    # 	3. Run TESS and Averaging based on the user input
    # 	4. Save the results

    def save_tbx(self):
        """
        Save current EEG-COMET parameters for future use.
        """

        # Define the path for saving the TBX object
        self.tbx_object_path = os.path.join(self.save_dir, 'eeg_comet_parameters.pkl')

        # Serialize and save the TBX object
        with open(self.tbx_object_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
