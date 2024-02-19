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
    def __init__(self, config=None, auto_save=True):
        """
        config: should be a ConfigParser, use it to load configs
        auto_save: if True, the COMET will automatically save itself after each process
        """
        if config:
            self.load_config(config)
        self.done_preprocessing = False
        self.done_clustering = False
        self.done_labeling_microstates = False
        self.done_backfitting = False
        self.done_extracting_features = False
        self.done_source_localization = False
        self.done_source_microstate_correlation = False
        self.auto_save = auto_save

    def load_config(self, config):
        # Base configs
        self.config = config
        self.study_name = config['base']['study_name']
        self.input_folder = config['base']['input_folder']
        self.channel_location_dir = config['base']['channel_location_dir']
        self.output_folder = config['base']['output_folder']
        self.log_text = config['base']['log_text']

        self.save_dir = os.path.join(self.output_folder, self.study_name)
        assert self.study_name != "", "study name cannot be empty"
        self.preprocessed_data_path = os.path.join(self.save_dir, f"{self.study_name}_preprocessed_data")
        self.eeg_info_path = os.path.join(self.save_dir, "eeg_info.pkl")

        # Load new study configs
        self.load_all_files = config.getboolean('load_new_study', 'load_all_files')
        if self.load_all_files:
            self.pattern_content = config.get('load_new_study', 'pattern_content', fallback='')
        self.extension = config['load_new_study']['extension']
        self.datatype = config['load_new_study']['datatype']
        self.filter_data = config.getboolean('load_new_study', 'filter_data')
        self.filter_method = config['load_new_study']['filter_method'] if self.filter_data else ''
        self.lowcut_freq = config.getint('load_new_study', 'lowcut_freq') if self.filter_data else ''
        self.highcut_freq = config.getint('load_new_study', 'highcut_freq') if self.filter_data else ''
        self.downsample_data = config.getboolean('load_new_study', 'downsample_data')
        self.sample_rate = config.getint('load_new_study', 'sample_rate') if self.downsample_data else ''
        self.remove_channels = config.getboolean('load_new_study', 'remove_channels')
        self.ch2rm = config['load_new_study']['ch2rm'] if self.remove_channels else 'missing'

        # Clustering configs
        self.smoothing_gfp = config.getboolean('do_clustering', 'smoothing_gfp')
        self.smoothing_distance = config.getint('do_clustering', 'smoothing_distance') if self.smoothing_gfp else ''
        number_of_maps = config['do_clustering']['number_of_maps']
        self.number_of_maps = number_of_maps if number_of_maps == 'auto' else int(number_of_maps)
        self.choose_number_of_maps = 'Auto' if number_of_maps == 'auto' else 'User'
        self.stopping_mode = config['do_clustering']['stopping_mode'] if self.number_of_maps == 'auto' else ''
        self.stopping_parameter = config.getfloat('do_clustering',
                                                  'stopping_parameter') if self.number_of_maps == 'auto' else ''
        self.kmin = config.getint('do_clustering', 'kmin') if self.number_of_maps == 'auto' else ''
        self.kmax = config.getint('do_clustering', 'kmax') if self.number_of_maps == 'auto' else ''
        self.initializer = config['do_clustering']['initializer']
        self.clustering_method = config['do_clustering']['clustering_method']
        self.max_iterations = config.getint('do_clustering', 'max_iterations')
        self.clustering_tolerance = config.getfloat('do_clustering', 'clustering_tolerance')
        need_options = ['X-Means Clustering', 'Agglomerative Hierarchical Clustering', 'K-Means Clustering',
                        'PCA + K-Means Clustering',
                        'Autoencoder + K-Means Clustering', ]
        self.clustering_option = config['do_clustering'][
            'clustering_option'] if self.clustering_method in need_options else ''
        self.number_of_repeats = config.getint('do_clustering', 'number_of_repeats')
        self.microstate_maps_path = os.path.join(self.save_dir, 'microstate_maps.csv')
        self.use_percentages = config.getint('do_clustering', 'use_percentages')

        # Backfitting configs
        self.backfit_to = config['do_backfitting']['backfit_to']
        self.identify_short_window = config.getboolean('do_backfitting', 'identify_short_window')
        self.filter_segments = config.getboolean('do_backfitting', 'filter_segments')
        self.remove_segments_less_than = config.getint('do_backfitting',
                                                       'remove_segments_less_than') if self.filter_segments else ''
        self.filter_segments_option = config['do_backfitting']['filter_segments_option'] if self.filter_segments else ''
        self.epsilon = config.getfloat('do_backfitting', 'epsilon')
        self.b = config.getint('do_backfitting', 'b')
        self.lamb = config.getint('do_backfitting', 'lamb')
        self.micro_labels = [chr(i) for i in range(ord('A'), ord('A') + self.choose_number_of_maps)]

        # Feature extraction configs
        self.extracted_features_path = os.path.join(self.save_dir, f"{self.study_name}_extracted_features")
        self.segmentation_path = os.path.join(self.save_dir, f"{self.study_name}_segmentation")
        self.export_format = config['extract_features']['export_format']
        self.feature_list = [x for x in config['extract_features']['feature_list'].split(',')]
        self.save_transitions_bool = True if 'TP' in self.feature_list else False
        self.window_size = config.getint('extract_features', 'window_size') if 'OCC' in self.feature_list else ''

        # Source localization configs
        self.localized_sources_path = os.path.join(self.save_dir, f"{self.study_name}_localized_sources")
        self.inverse_method = config['source_localize_microstates']['inverse_method']
        self.nperm = config.getint('source_localize_microstates', 'nperm')
        self.spacing = config['source_localize_microstates']['spacing']
        self.source_localization_method = config['source_localize_microstates']['source_localization_method']
        self.anatomy_subjects_dir = config['source_localize_microstates']['anatomy_subjects_dir']

    def load_raw(self):
        '''
        get all avaliable eeg file path
        '''
        if self.load_all_files:
            self.pattern = '*'
        else:
            self.pattern = '*' + self.pattern_content + '*'

        self.list_eegs_path, self.list_eegs = DataIO().find_data(self.input_folder, self.extension, self.pattern)

    # self.list_eegs = [os.path.basename(x).split('.')[0] for x in self.list_eegs_path]
    # assert self.list_eegs_path, 'eeg list is empty'

    def do_preprocessing(self):

        if not os.path.exists(self.preprocessed_data_path):
            os.makedirs(self.preprocessed_data_path)

        length_all_data = []
        data_io = DataIO()

        # Create an instance of the progress dialog
        progress_dialog = ProgressDialog()
        progress_dialog.set_window_title("Preprocessing ...")
        progress_dialog.set_label_text("Preprocessing data: ")
        progress_dialog.show()
        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(self.list_eegs), ncols=100, position=0, leave=True)

        for filename_idx, filename in enumerate(self.list_eegs_path):
            progress_dialog.set_line_edit_text(f"{self.list_eegs[self.list_eegs_path.index(filename)]}")
            progress_bar.set_description(f"Preprocessing file: {self.list_eegs[self.list_eegs_path.index(filename)]}")
            # Create an instance of the DataPreprocessor class
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
            progress_dialog.update_progress(filename_idx + 1, len(self.list_eegs))
            progress_bar.update(1)

            self.eeg_info = eeg.info
            # Save EEG info
            if not self.sample_rate:
                self.sample_rate = int(self.eeg_info['sfreq'])
            self.channels2remove = channels2remove

            data_io.save_eeg_info(self.eeg_info_path, eeg_info)

            ch_names, ch_location = list(self.eeg_info['ch_names']), self.eeg_info['chs']
            n_chan = len(ch_names)
            self.n_chan = n_chan
            self.ch_names = ch_names
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            save_path = os.path.join(self.preprocessed_data_path, name)

            self.length_all_data = np.append(length_all_data, int(length_data))

            # save EEG object
            data_io.export_eegs(eeg, save_path, self.extension, self.datatype)

        progress_dialog.close()
        progress_bar.close()

        self.done_preprocessing = True
        if self.auto_save:
            self.save_tbx()

    def do_autopilot(self):

        autopilot_clusterer = AutopilotClusterer(self.save_dir, self.study_name, self.extension, self.datatype)
        autopilot_clusterer.run_autopilot()

    def do_clustering(self):
        print('\nClustering ...')

        if self.smoothing_gfp:
            self.min_distance_size = int(self.smoothing_distance / (1000 / self.sample_rate))
        else:
            self.min_distance_size = []

        avaliable_methods = [
            'Modified K-Means Clustering',
            'K-Means Clustering',
            'PCA + K-Means Clustering',
            'Autoencoder + K-Means Clustering',
            'X-Means Clustering',
            'Agglomerative Hierarchical Clustering',
            ]
        assert self.clustering_method in avaliable_methods, "clustering_method not supported"

        microstate_clusterer = MicrostateClusterer(self.number_of_repeats,
                                                   self.max_iterations,
                                                   self.clustering_tolerance
                                                   )

        if self.number_of_maps == 'auto':
            self.maps2use, self.peaks2use = DataInitializer().generate_maps_and_peaks(
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

            self.optimal_k, self.k_values, self.target_values = self.clusterer_optimizer.find_optimal_k(
                self.stopping_mode, self.stopping_parameter)
            self.number_of_maps = self.optimal_k
            print(f'result: n_states = {self.number_of_maps}')

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
        self.done_clustering = True
        # return best_maps
        if self.auto_save:
            self.save_tbx()

    def do_labeling(self):

        self.microstate_labeler = MicrostateLabeler(self.best_maps, self.eeg_info, self.microstate_maps_path)
        self.micro_labels = self.microstate_labeler.do_labeling()
        self.done_labeling_microstates = True

    def do_backfitting(self):
        print('\nBackfitting ...')

        self.segmentation_path = os.path.join(self.save_dir, f"{self.study_name}_segmentation")
        if not os.path.exists(self.segmentation_path):
            os.makedirs(self.segmentation_path)

        backfitter_instance = MicrostateBackfitter(
            self.study_name,
            self.preprocessed_data_path,
            self.best_maps,
            self.backfit_to,
            self.filter_segments,
            self.filter_segments_option,
            self.identify_short_window,
            self.remove_segments_less_than,
            self.micro_labels,
            self.segmentation_path,
            self.extension,
            self.datatype,
            self.sample_rate,
            [self.epsilon, self.b, self.lamb],
            self.export_format,
        )
        backfitter_instance.perform_segmentation()

        self.done_backfitting = True
        if self.auto_save:
            self.save_tbx()

    def extract_features(self):
        print('\nExtracting Features ...')

        self.extracted_features_path = os.path.join(self.save_dir, f"{self.study_name}_extracted_features")
        if not os.path.exists(self.extracted_features_path):
            os.makedirs(self.extracted_features_path)

        segmentation_list_path, segmentation_list_filename = DataIO().find_data(
            self.segmentation_path,
            self.export_format
        )

        # Create an instance of the progress dialog
        progress_dialog = ProgressDialog()
        progress_dialog.set_window_title("Extracting Features ...")
        progress_dialog.show()

        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(segmentation_list_path), ncols=100, position=0, leave=True)

        for s in range(len(segmentation_list_path)):

            segmentation_path = segmentation_list_path[s]
            filename = segmentation_list_filename[s]
            progress_bar.set_description(f"Extracting features: {filename}")

            segmentation_array = SegmentationIO().load_segmentation(segmentation_path, import_format='.csv')

            if 'static' in self.feature_mode:
                feature_extractor = FeatureExtractor(
                    segmentation_array,
                    self.sample_rate,
                    self.window_size,
                    mode='static')
                if 'GEV' in self.feature_list:
                    data_io = DataIO()
                    if self.datatype == 'epoched':
                        underscore_index = filename.rfind('_')
                        eeg_filename = filename[:underscore_index]
                        trial_number = filename[underscore_index+1:]
                        eeg_path = os.path.join(self.preprocessed_data_path, f"{eeg_filename}{self.extension}")
                        eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
                        trial_data = np.squeeze(eeg[int(trial_number)].get_data())
                        output_features = feature_extractor.extract_microstate_features(
                            filename,
                            self.feature_list,
                            trial_data,
                            self.best_maps,
                            self.micro_labels
                        )

                    else:
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
                    static_features_dfs = output_features
                else:
                    static_features_dfs = pd.concat([static_features_dfs, output_features], ignore_index=True)

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
                        filename, self.feature_list,
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

            progress_dialog.set_label_text("Extracting features: ")
            progress_dialog.set_line_edit_text(f"{filename}")
            progress_dialog.update_progress(s + 1, len(segmentation_list_path))
            progress_bar.update(1)

        progress_dialog.close()
        progress_bar.close()

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

        self.done_extracting_features = True
        if self.auto_save:
            self.save_tbx()

    def source_localize_microstates(self):
        print("Calculating Source Time Series ...")

        if self.use_anatomy == "fsaverage":
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.anatomy_subjects_dir = os.path.dirname(fs_dir)
        elif self.use_anatomy == "individual":
            self.anatomy_subjects_dir = self.individual_subjects_dir

        source_localizer = SourceLocalizer(
            self.anatomy_subjects_dir,
            self.localized_sources_path,
            self.preprocessed_data_path,
            self.segmentation_path,
            self.use_anatomy,
            self.extension,
            self.datatype,
            self.spacing,
            self.inverse_method,
            self.best_maps,
            self.nperm
        )
        source_localizer.run_source_localization()

        self.done_source_localization = True
        if self.auto_save:
            self.save_tbx()

    def source_microstate_correlation(self, method='tess'):
        print("Correlating sources and microstates ...")

        try:
            print(self.anatomy_subjects_dir)
        except AttributeError:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.anatomy_subjects_dir = os.path.dirname(fs_dir)

        source_localizer = SourceLocalizer(
            self.anatomy_subjects_dir,
            self.localized_sources_path,
            self.preprocessed_data_path,
            self.segmentation_path,
            self.use_anatomy,
            self.extension,
            self.datatype,
            self.spacing,
            self.inverse_method,
            self.best_maps,
            self.nperm
        )

        source_localizer.identify_microstates_sources(method)

        self.done_source_microstate_correlation = True
        if self.auto_save:
            self.save_tbx()

    # 	TODO: Need to expand this function to include the following:
    # 	1. Load the source localized time series
    # 	2. Load the microstates
    # 	3. Run TESS and Averaging based on the user input
    # 	4. Save the results

    def save_tbx(self):
        self.tbx_object_path = os.path.join(self.save_dir, 'comet_tbx_object.pkl')
        with open(self.tbx_object_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
