import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import mne

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
        self.log_text: str = config['base']['log_text']

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
        progress_dialog = ProgressDialog()
        progress_dialog.set_window_title("Preprocessing ...")
        progress_dialog.set_label_text("Preprocessing data: ")
        progress_dialog.show()
        progress_dialog.start_process(len(self.list_eegs))
        progress_dialog.update_progress(0)

        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(self.list_eegs), ncols=100, position=0, leave=True)

        # Iterate through EEG files
        for filename_idx, filename in enumerate(self.list_eegs_path):
            eeg_name = self.list_eegs[self.list_eegs_path.index(filename)]
            progress_dialog.set_line_edit_text(f"{eeg_name}")
            progress_dialog.update_progress(filename_idx)
            progress_bar.set_description(f"Preprocessing file: {eeg_name}")

            # Preprocess EEG data
            preprocessor = DataPreprocessor()
            eeg, preprocessed_data, length_data, eeg_info, channels2remove = preprocessor.preprocess_eegs(
                filename,
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
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            save_path = os.path.join(self.preprocessed_data_path, name)
            self.length_all_data = np.append(length_all_data, int(length_data))
            data_io.export_eegs(eeg, save_path, self.extension, self.datatype)

            progress_dialog.update_progress(filename_idx + 1)
            progress_bar.update(1)

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
            self.clusterer_optimizer = ClustererOptimizer(
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
            self.optimal_k, self.k_values, self.target_values = self.clusterer_optimizer.find_optimal_k(
                self.stopping_mode, self.stopping_parameter)
            self.number_of_maps = self.optimal_k
            print(f'Result: n_states = {self.number_of_maps}')

        # Perform clustering
        self.best_maps, self.gev, _ = microstate_clusterer.clustering_func(
            self.preprocessed_data_path,
            self.extension,
            self.datatype,
            self.n_pca,
            self.clustering_method,
            self.number_of_maps,
            self.initializer,
            self.use_percentages,
            self.min_distance_size,
            self.clustering_option,
            self.eeg_info,
            self.microstate_maps_path
        )
        print(f'Global Explained Variance: {self.gev}')

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
        self.done_labeling_microstates = True

    def do_backfitting(self):
        print('\nBackfitting ...')

        # Create segmentation directory if it doesn't exist
        self.segmentation_path = os.path.join(self.save_dir, f"{self.study_name}_segmentation")
        if not os.path.exists(self.segmentation_path):
            os.makedirs(self.segmentation_path)

        # Initialize backfitter instance
        backfitter_instance = MicrostateBackfitter(
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
        backfitter_instance.perform_segmentation()

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

        # Initialize progress dialog
        progress_dialog = ProgressDialog()
        progress_dialog.set_window_title("Extracting Features ...")
        progress_dialog.set_label_text("Extracting features: ")
        progress_dialog.show()
        progress_dialog.start_process(len(segmentation_list_path))
        progress_dialog.update_progress(0)

        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(segmentation_list_path), ncols=100, position=0, leave=True)

        for s in range(len(segmentation_list_path)):
            progress_dialog.set_line_edit_text(f"{filename}")
            progress_dialog.update_progress(s)

            segmentation_path = segmentation_list_path[s]
            filename = segmentation_list_filename[s]
            progress_bar.set_description(f"{filename}")

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
                        underscore_index = filename.rfind('_')
                        eeg_filename = filename[:underscore_index]
                        trial_number = filename[underscore_index + 1:]
                        eeg_path = os.path.join(self.preprocessed_data_path, f"{eeg_filename}{self.extension}")
                        eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
                        trial_data = np.squeeze(eeg[int(trial_number)].get_data())
                    else:
                        eeg_path = os.path.join(self.preprocessed_data_path, f"{filename}{self.extension}")
                        eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
                        eeg_data = data_io.get_eeg_data(eeg, self.datatype)

                    output_features = feature_extractor.extract_microstate_features(
                        filename,
                        self.feature_list,
                        trial_data if self.datatype == 'epoched' else eeg_data,
                        self.best_maps,
                        self.micro_labels
                    )
                else:
                    output_features = feature_extractor.extract_microstate_features(filename, self.feature_list)

                if s == 0:
                    static_features_dfs = output_features
                else:
                    static_features_dfs = pd.concat([static_features_dfs, output_features], ignore_index=True)

                if not progress_dialog.running:  # Check if the process should be stopped
                    break

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
                    eeg_path = os.path.join(self.preprocessed_data_path, f"{filename}{self.extension}")
                    eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
                    eeg_data = data_io.get_eeg_data(eeg, self.datatype)
                    output_features = feature_extractor.extract_microstate_features(
                        filename,
                        self.feature_list,
                        eeg_data,
                        self.best_maps,
                        self.micro_labels
                    )
                else:
                    output_features = feature_extractor.extract_microstate_features(filename, self.feature_list)

                if s == 0:
                    dynamic_features_dfs = output_features
                else:
                    dynamic_features_dfs = pd.concat([dynamic_features_dfs, output_features], ignore_index=True)

                if not progress_dialog.running:  # Check if the process should be stopped
                    break

            progress_dialog.update_progress(s + 1)
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

        print("\nEEG-COMET parameters saved successfully.")
