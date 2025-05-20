import os
import mne
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
from clustering_utils.microstate_io import MicrostateIO
from backfitting_utils.microstate_backfitter import MicrostateBackfitter
from backfitting_utils.segmentation_io import SegmentationIO
from features_utils.feature_helper import FeatureHelper
from features_utils.feature_extractor import FeatureExtractor
from features_utils.feature_io import FeatureIO
from sourcelocalization_utils.source_localizer import SourceLocalizer


class COMET:
    """
    The COMET class represents an instance of the EEG-COMET application.
    It provides methods to load configuration settings, perform various processes
    such as preprocessing, clustering, labeling, backfitting, feature extraction,
    source localization, and source-microstate correlation.

    This version uses optimized data storage strategies, saving only critical
    parameters and metadata in memory while storing large datasets on disk.

    It also eliminates dependency on external configuration files, allowing
    direct parameter configuration or using optional config files.
    """

    # Class-level shared storage for feature extraction results
    _shared_feature_results = {}

    def __init__(self, config=None, config_path=None, study_name=None, input_folder=None, output_folder=None,
                 auto_save=True):
        """
        Initialize COMET with flexible configuration options.

        Parameters:
        -----------
        config : dict or ConfigParser, optional
            Direct configuration object
        config_path : str, optional
            Path to a configuration file
        study_name : str, optional
            Name of the study (overrides config value)
        input_folder : str, optional
            Path to input data (overrides config value)
        output_folder : str, optional
            Path to output folder (overrides config value)
        auto_save : bool, default=True
            Whether to automatically save after processing steps
        """
        # Define all instance variables
        self.LogWindow = None
        self.log_text = []

        # Initialize utility objects
        self.comet_data_io = DataIO()
        self.comet_preprocessor = DataPreprocessor()
        self.comet_data_initializer = DataInitializer()
        self.comet_microstate_io = MicrostateIO()
        self.comet_segmentation_io = SegmentationIO()
        self.comet_feature_io = FeatureIO()

        # Load or create configuration
        if config is not None:
            # Use provided config
            self.config = config
        elif config_path is not None:
            # Load from specified path
            self.config = self.load_config(config_path)
        else:
            # Create default configuration
            self.config = self.create_default_config()

        # Override config values with directly provided parameters
        if study_name:
            self.config["io_config"]["study_name"] = study_name
        if input_folder:
            self.config["io_config"]["input_folder"] = input_folder
        if output_folder:
            self.config["io_config"]["output_folder"] = output_folder

        # Load configuration values into instance variables
        self.load_config_values()

        # Set up directories
        self.setup_directories()

        # Initialize flags and data holders
        self.done_preprocessing = False
        self.done_clustering = False
        self.done_microstate_labeling = False
        self.done_backfitting = False
        self.done_extracting_features = False
        self.done_source_localization = False
        self.done_identifying_microstate_sources = False
        self.auto_save = auto_save

        @property
        def study_name(self):
            return self._study_name

        @study_name.setter
        def study_name(self, value):
            self._study_name = value
            self.reset_directories()

        @property
        def output_folder(self):
            return self._output_folder

        @output_folder.setter
        def output_folder(self, value):
            self._output_folder = value
            self.reset_directories()

        @property
        def input_folder(self):
            return self._input_folder

        @input_folder.setter
        def input_folder(self, value):
            self._input_folder = value

    def create_default_config(self):
        """
        Create a default configuration without loading from a file.
        These defaults are intended to be overridden by user inputs.
        """
        config = ConfigParser()

        # Create default sections
        config.add_section("io_config")
        config.add_section("preprocessing_config")
        config.add_section("clustering_config")
        config.add_section("backfitting_config")
        config.add_section("features_config")
        config.add_section("source_config")

        # Set default values for IO section - these must be updated from UI
        config["io_config"]["study_name"] = "my_study"
        config["io_config"]["input_folder"] = ""
        config["io_config"]["montage"] = ""
        config["io_config"]["extension"] = ".set"
        config["io_config"]["pattern"] = "*"
        config["io_config"]["datatype"] = "raw"
        config["io_config"]["output_folder"] = ""

        # Set default preprocessing values
        config["preprocessing_config"]["temporal_filter_data"] = "True"
        config["preprocessing_config"]["filter_method"] = "fir"
        config["preprocessing_config"]["lowcut_freq"] = "2"
        config["preprocessing_config"]["highcut_freq"] = "20"
        config["preprocessing_config"]["downsample_data"] = "True"
        config["preprocessing_config"]["sample_rate"] = "250"
        config["preprocessing_config"]["spatial_filter_data"] = "False"
        config["preprocessing_config"]["auto_clean_data"] = "False"
        config["preprocessing_config"]["remove_channels"] = "False"
        config["preprocessing_config"]["ch2rm"] = ""
        config["preprocessing_config"]["prep_data"] = "False"

        # Set default clustering values
        config["clustering_config"]["smoothing_gfp"] = "False"
        config["clustering_config"]["smoothing_distance"] = "10"
        config["clustering_config"]["number_of_maps"] = "4"
        config["clustering_config"]["kmin"] = "2"
        config["clustering_config"]["kmax"] = "10"
        config["clustering_config"]["stopping_mode"] = "gev"
        config["clustering_config"]["stopping_parameter"] = "10"
        config["clustering_config"]["use_percentages"] = "100"
        config["clustering_config"]["initializer"] = "Random"
        config["clustering_config"]["clustering_method"] = "Modified K-Means Clustering (Pascual-Marqui et al. 1995)"
        config["clustering_config"]["max_iterations"] = "500"
        config["clustering_config"]["clustering_tolerance"] = "1e-6"
        config["clustering_config"]["similarity_metric"] = "Spatial Correlation"
        config["clustering_config"]["number_of_repeats"] = "5"

        # Set default backfitting values
        config["backfitting_config"]["backfit_to"] = "all"
        config["backfitting_config"]["identify_short_window"] = "False"
        config["backfitting_config"]["filter_segments"] = "False"
        config["backfitting_config"]["filter_segments_less_than"] = "20"
        config["backfitting_config"]["filter_segments_option"] = "smooth"
        config["backfitting_config"]["epsilon"] = "1e-6"
        config["backfitting_config"]["b"] = "3"
        config["backfitting_config"]["lamb"] = "5"

        # Set default feature extraction values
        config["features_config"]["export_format"] = ".csv"
        config["features_config"]["feature_list"] = "OCC, DUR, COV"
        config["features_config"]["feature_mode"] = "averaged"
        config["features_config"]["feature_types"] = "real"
        config["features_config"]["sliding_window_size"] = "1"
        config["features_config"]["pre_window_size"] = "1"
        config["features_config"]["post_window_size"] = "1"

        # Set default source localization values
        config["source_config"]["use_anatomy"] = "fsaverage"
        config["source_config"]["inverse_method"] = "dSPM"
        config["source_config"]["nperm"] = "2000"
        config["source_config"]["spacing"] = "ico3"
        config["source_config"]["source_localization_method"] = "tess"
        config["source_config"]["anatomy_subjects_dir"] = ""

        return config

    def reset_directories(self):
        """
        Reset and recreate directory structure when critical parameters change.
        Call this whenever study_name or output_folder are changed.
        """
        # Set up the save_dir based on output_folder and study_name
        if self.output_folder and self.study_name:
            self.save_dir = os.path.join(self.output_folder, self.study_name)

            # Update all paths relative to save_dir
            self.config_path = os.path.join(self.save_dir, "eeg_comet_config.ini")
            self.preprocessed_data_path = os.path.join(self.save_dir, f"{self.study_name}_preprocessed_data")
            self.eeg_info_path = os.path.join(self.save_dir, "eeg_info.fif")
            self.microstate_maps_path = os.path.join(self.save_dir, "microstate_maps.csv")
            self.extracted_features_path = os.path.join(self.save_dir, f"{self.study_name}_extracted_features")
            self.segmentation_path = os.path.join(self.save_dir, f"{self.study_name}_segmentation")
            self.localized_sources_path = os.path.join(self.save_dir, f"{self.study_name}_localized_sources")
            self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
            self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")

    def load_config(self, config_path):
        """
        Load configuration from a file path.
        """
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found. Using default configuration.")
            return self.create_default_config()

        config = ConfigParser()
        try:
            config.read(config_path)
            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
            return self.create_default_config()

    def load_config_values(self):
        """
        Load configuration values from self.config into instance variables.
        """
        # Input/Output Configs
        io_config = self.config["io_config"]
        self.study_name = io_config.get("study_name", "my_study")
        self.input_folder = io_config.get("input_folder", "")
        self.montage = io_config.get("montage", "")
        self.extension = io_config.get("extension", ".auto")
        self.pattern_content = io_config.get("pattern", "*")
        self.datatype = io_config.get("datatype", "raw")
        self.output_folder = io_config.get("output_folder", "")

        # Preprocessing Configs
        preprocessing_config = self.config["preprocessing_config"]
        self.temporal_filter_data = preprocessing_config.getboolean("temporal_filter_data", True)
        if self.temporal_filter_data:
            self.filter_method = preprocessing_config.get("filter_method", "fir")
            self.lowcut_freq = preprocessing_config.getint("lowcut_freq", 2)
            self.highcut_freq = preprocessing_config.getint("highcut_freq", 20)
        else:
            self.filter_method = "fir"
            self.lowcut_freq = 2
            self.highcut_freq = 20

        self.downsample_data = preprocessing_config.getboolean("downsample_data", True)
        if self.downsample_data:
            self.sample_rate = preprocessing_config.getint("sample_rate", 250)
        else:
            self.sample_rate = 250

        self.spatial_filter_data = preprocessing_config.getboolean("spatial_filter_data", False)
        self.auto_clean_data = preprocessing_config.getboolean("auto_clean_data", False)
        self.remove_channels = preprocessing_config.getboolean("remove_channels", False)
        if self.remove_channels:
            self.chan2rm = preprocessing_config.get("ch2rm", "")
        else:
            self.chan2rm = ""
        self.prep_data = preprocessing_config.getboolean("prep_data", False)

        # Clustering Configs
        clustering_config = self.config["clustering_config"]
        self.smoothing_gfp = clustering_config.getboolean("smoothing_gfp", False)
        if self.smoothing_gfp:
            self.smoothing_distance = clustering_config.getint("smoothing_distance", 10)
        else:
            self.smoothing_distance = 10

        number_of_maps = clustering_config.get("number_of_maps", "4")
        self.number_of_maps = number_of_maps if number_of_maps == "auto" else int(number_of_maps)
        self.choose_number_of_maps = "Auto" if number_of_maps == "auto" else "User"

        if self.number_of_maps == "auto":
            self.stopping_mode = clustering_config.get("stopping_mode", "gev")
            self.stopping_parameter = clustering_config.getfloat("stopping_parameter", 10)
            self.kmin = clustering_config.getint("kmin", 2)
            self.kmax = clustering_config.getint("kmax", 10)
        else:
            self.stopping_mode = "gev"
            self.stopping_parameter = 10
            self.kmin = 2
            self.kmax = 10

        self.initializer = clustering_config.get("initializer", "Random")
        self.clustering_method = clustering_config.get(
            "clustering_method", "Modified K-Means Clustering (Pascual-Marqui et al. 1995)"
        )
        self.max_iterations = clustering_config.getint("max_iterations", 500)
        self.clustering_tolerance = clustering_config.getfloat("clustering_tolerance", 1e-6)
        # self.batch_size
        self.similarity_metric = clustering_config.get(
            "similarity_metric",
            "Spatial Correlation"
        ) if not self.clustering_method == "Modified K-Means Clustering (Pascual-Marqui et al. 1995)" else ""
        self.number_of_repeats = clustering_config.getint("number_of_repeats", 5)

        # Handle use_percentages - this could be an int or empty string
        try:
            self.use_percentages = clustering_config.getint("use_percentages", 100)
        except ValueError:
            self.use_percentages = 100

        # Backfitting Configs
        backfitting_config = self.config["backfitting_config"]
        self.backfit_to = backfitting_config.get("backfit_to", "all")
        self.identify_short_window = backfitting_config.getboolean("identify_short_window", False)
        self.filter_segments = backfitting_config.getboolean("filter_segments", False)

        if self.filter_segments:
            self.filter_segments_less_than = backfitting_config.getint("filter_segments_less_than", 20)
            self.filter_segments_option = backfitting_config.get("filter_segments_option", "smooth")
        else:
            self.filter_segments_less_than = 20
            self.filter_segments_option = "smooth"

        self.epsilon = backfitting_config.getfloat("epsilon", 1e-6)
        self.b = backfitting_config.getint("b", 3)
        self.lamb = backfitting_config.getint("lamb", 5)

        # Feature Extraction Configs
        features_config = self.config["features_config"]
        self.export_format = features_config.get("export_format", ".csv")

        feature_list_str = features_config.get("feature_list", "OCC, DUR, COV")
        self.feature_list = [x.strip() for x in feature_list_str.split(",")]

        feature_mode_str = features_config.get("feature_mode", "averaged")
        self.feature_mode = [x.strip() for x in feature_mode_str.split(",")]

        feature_types_str = features_config.get("feature_types", "real")
        self.feature_types = [x.strip() for x in feature_types_str.split(",")]

        if "OCC" in self.feature_list:
            try:
                self.sliding_window_size = features_config.getint("sliding_window_size", 1)
            except ValueError:
                self.sliding_window_size = 1
        else:
            self.sliding_window_size = 1

        try:
            self.pre_window_size = features_config.getint("pre_window_size", 1)
        except ValueError:
            self.pre_window_size = 1

        try:
            self.post_window_size = features_config.getint("post_window_size", 1)
        except ValueError:
            self.post_window_size = 1

        # Create directory for extracted features
        self.feature_list_dictionary = {
            "OCC": "Frequency of Occurrence (Hz)", "DUR": "Mean Microstate Duration (ms)",
            "COV": "Microstate Coverage (%)", "GEV": "Microstate Global Explained Variance (%)",
            "TP": "Transition Probability", "SE": "Sequence Entropy", "LZC": "Sequence Lempel-Ziv Complexity",
            "ER": "Sequence Entropy Representation", "ROF": "Relative Occurrence Frequency",
            "RTF": "Relative Transition Frequency"
        }

        # Source Localization Configs
        source_config = self.config["source_config"]
        self.use_anatomy = source_config.get("use_anatomy", "fsaverage")
        self.inverse_method = source_config.get("inverse_method", "dSPM")
        self.nperm = source_config.getint("nperm", 2000)
        self.spacing = source_config.get("spacing", "ico3")
        self.source_localization_method = source_config.get("source_localization_method", "tess")
        self.anatomy_subjects_dir = source_config.get("anatomy_subjects_dir", "")

        # State Flags
        # Check if the section exists to handle loading from older config files
        if "state_flags" in self.config:
            state_flags = self.config["state_flags"]
            self.done_preprocessing = state_flags.getboolean("done_preprocessing", False)
            self.done_clustering = state_flags.getboolean("done_clustering", False)
            self.done_microstate_labeling = state_flags.getboolean("done_microstate_labeling", False)
            self.done_backfitting = state_flags.getboolean("done_backfitting", False)
            self.done_extracting_features = state_flags.getboolean("done_extracting_features", False)
            self.done_source_localization = state_flags.getboolean("done_source_localization", False)
            self.done_identifying_microstate_sources = state_flags.getboolean("done_identifying_microstate_sources",
                                                                             False)
        else:
            # Initialize state flags to False if the section doesn't exist
            self.done_preprocessing = False
            self.done_clustering = False
            self.done_microstate_labeling = False
            self.done_backfitting = False
            self.done_extracting_features = False
            self.done_source_localization = False
            self.done_identifying_microstate_sources = False

    def ensure_directory(self, path):
        """
        Ensures a directory exists and is writable.

        Parameters:
        -----------
        path : str
            Directory path to check/create

        Returns:
        --------
        bool
            True if directory exists and is writable, False otherwise
        """
        if not path:
            return False

        try:
            os.makedirs(path, exist_ok=True)
            # Test if we can write to this directory
            test_file = os.path.join(path, ".test_write_access")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except (PermissionError, OSError) as e:
            print(f"Warning: Cannot access directory {path}: {str(e)}")
            return False

    def setup_directories(self):
        """
        Set up all required directories based on user configuration.
        """
        # Check if output_folder and study_name are set before proceeding
        if not self.output_folder:
            self.output_folder = os.path.expanduser("~/EEG-COMET_Data")

        if not self.study_name:
            self.study_name = "my_study"
            print(f"No study name specified, using: {self.study_name}")

        # Create output folder if it doesn't exist
        if not self.ensure_directory(self.output_folder):
            old_output = self.output_folder
            self.output_folder = os.path.expanduser("~/EEG-COMET_Data")
            print(f"Cannot access {old_output}, falling back to: {self.output_folder}")
            self.ensure_directory(self.output_folder)

        # Setup study directory
        self.save_dir = os.path.join(self.output_folder, self.study_name)
        self.ensure_directory(self.save_dir)

        # Define all paths relative to save_dir
        self.config_path = os.path.join(self.save_dir, "eeg_comet_config.ini")
        self.preprocessed_data_path = os.path.join(self.save_dir, f"{self.study_name}_preprocessed_data")
        self.eeg_info_path = os.path.join(self.save_dir, "eeg_info.fif")
        self.microstate_maps_path = os.path.join(self.save_dir, "microstate_maps.csv")
        self.extracted_features_path = os.path.join(self.save_dir, f"{self.study_name}_extracted_features")
        self.segmentation_path = os.path.join(self.save_dir, f"{self.study_name}_segmentation")
        self.localized_sources_path = os.path.join(self.save_dir, f"{self.study_name}_localized_sources")
        self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
        self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")

    def initialize_log_window(self):
        """
        Initialize the logging window
        """
        self.LogWindow = LogWindow()
        self.LogWindow.append_log("Welcome to EEG-COMET!")

    def load_raw(self):
        """
        Locate EEG file paths.
        """
        # Check if load_all_files attribute exists, set it to True if not
        if not hasattr(self, 'load_all_files'):
            self.load_all_files = True

        self.pattern = '*' if self.load_all_files else '*' + self.pattern_content + '*'
        self.list_eegs_path, self.list_eegs = self.comet_data_io.find_data(
            input_folder=self.input_folder, extension=self.extension, pattern=self.pattern
        )

    def load_clean(self):
        """
        Locate EEG file paths that were automatically cleaned.
        """
        self.list_eegs_path, self.list_eegs = self.comet_data_io.find_data(
            input_folder=self.preprocessed_data_path, extension=self.extension, pattern='*'
        )

    def load_maps(self):
        """
        Load microstate maps from the specified path.
        """
        if not os.path.isfile(self.microstate_maps_path):
            error_msg = f"Microstate maps file not found: {self.microstate_maps_path}"
            raise FileNotFoundError(error_msg)
        try:
            self.best_maps, self.micro_labels = self.comet_microstate_io.load_microstates(self.microstate_maps_path)
            return True
        except Exception as e:
            error_msg = f"Error loading microstate maps: {str(e)}"
            raise ValueError(error_msg)

    def check_chan2rm(self):
        """
        Check the consistency of EEG channels across all data
        """
        consistent_channels, missing_channels = self.comet_data_io.check_chan2rm(
            self.list_eegs_path,
            datatype=self.datatype,
            montage=self.montage
        )
        self.chan2rm = list(set(self.chan2rm).union(set(missing_channels)))
        self.ch_names = consistent_channels

    def save_eeg_info(self, eeg_info_path):
        """Save EEG information to a binary file using pickle.

        Args:
            eeg_info_path (str): Path where the EEG information will be saved
            eeg_info (object): EEG information object to be serialized
        """
        # Create a basic Info object
        eeg_info = mne.create_info(
            ch_names=self.ch_names,
            ch_types=['eeg'] * len(self.ch_names),
            sfreq=self.sample_rate
        )
        eeg_info['description'] = self.study_name

        # Set montage
        montage = self.comet_data_io.load_montage(self.montage)
        eeg_info.set_montage(montage)

        mne.io.write_info(eeg_info_path, eeg_info)

    def load_eeg_info(self):
        """
        Load EEG info from file instead of keeping it in memory
        """
        self.eeg_info = mne.io.read_info(self.eeg_info_path)

    def autoclean_rseeg(self, eeg_path, eeg_name):
        """
        Auto-clean a raw EEG file and save the result
        """
        # Load the EEG
        eeg = self.comet_data_io.load_eeg(
            eeg_path=eeg_path,
            datatype=self.datatype,
            montage=self.montage,
            chan2rm=self.chan2rm
        )

        # Clean Resting-State EEG data
        eeg = self.comet_preprocessor.auto_clean_raw_eeg(eeg)

        name = os.path.splitext(eeg_name)[0]
        save_path = os.path.join(self.preprocessed_data_path, name)
        self.comet_data_io.export_eegs(eeg=eeg, save_path=save_path, extension=self.extension, datatype=self.datatype)

    def preprocess_eeg(self, eeg_path, eeg_name):
        """
        Preprocess an EEG file and save the result
        """
        # Load the EEG
        eeg = self.comet_data_io.load_eeg(
            eeg_path=eeg_path,
            datatype=self.datatype,
            montage=self.montage,
            chan2rm=self.chan2rm
        )

        # If desired, do pre-processing
        if self.prep_data:
            eeg = self.comet_preprocessor.identify_bad_channels(eeg)
            eeg.interpolate_bads(reset_bads=False)

        # Preprocess the EEG data
        eeg = self.comet_preprocessor.preprocess_eeg(
            eeg=eeg,
            filter_bool=self.temporal_filter_data,
            filtermethod=self.filter_method,
            lowcut=self.lowcut_freq,
            highcut=self.highcut_freq,
            downsample_bool=self.downsample_data,
            sampling_rate=self.sample_rate,
            spatial_smooth_bool=self.spatial_filter_data
        )

        # Extract and save EEG info
        eeg_data = self.comet_data_io.get_eeg_data(eeg=eeg, datatype=self.datatype)
        length_data = eeg_data.shape[1]

        if not self.sample_rate:
            self.sample_rate = int(self.eeg_info['sfreq'])

        # Prepare saving path and export EEG
        name = os.path.splitext(eeg_name)[0]
        save_path = os.path.join(self.preprocessed_data_path, name)
        self.comet_data_io.export_eegs(eeg=eeg, save_path=save_path, extension=self.extension, datatype=self.datatype)

        # (Optional) Append a log for each file processed.
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"EEG Preprocessed: {eeg_name} - Length: {int(length_data / self.sample_rate)} sec"
            )

    def get_preprocessed_eeg(self, subject_name):
        """
        Load a preprocessed EEG file on demand instead of keeping it in memory
        """
        eeg_path = os.path.join(self.preprocessed_data_path, f"{subject_name}{self.extension}")
        return self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)

    def compute_gev_all_data(self):
        """
        Compute Global Explained Variance for all data
        """
        all_data, _ = self.comet_data_initializer.generate_maps_and_peaks(
            preprocessed_folder=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            use_percentages=100
        )
        return self.comet_microstate_clusterer.compute_gev(data=all_data, maps=self.best_maps)

    def cluster_eeg_microstates(self, init):
        """
        Perform modified K-means clustering iteration
        """
        print(f"Running clustering iteration {init + 1}/{self.number_of_repeats}")

        initial_maps = self.comet_data_initializer.initialize_cluster_centers(
            maps2use=self.maps2use, n_states=self.number_of_maps, initializer=self.initializer
        )

        if self.clustering_method == "Modified K-Means Clustering (Pascual-Marqui et al. 1995)":
            maps_init, residual_init = self.comet_microstate_clusterer.modified_kmeans(
                data=self.maps2use,
                initial_maps=initial_maps
            )
        elif self.clustering_method == "Modified K-Means Clustering with Spatial Similarity":
            maps_init, residual_init = self.comet_microstate_clusterer.modified_kmeans_similarity(
                data=self.maps2use,
                initial_maps=initial_maps,
                metric=self.similarity_metric
            )
        elif self.clustering_method == "Topographic Atomize and Agglomerate Hierarchical Clustering":
            maps_init, residual_init = self.comet_microstate_clusterer.taahc(
                data=self.maps2use,
                metric=self.similarity_metric
            )

        gev_init = self.comet_microstate_clusterer.compute_gev(data=self.maps2use, maps=maps_init)
        print(f'Found {self.number_of_maps} Microstate Maps')
        print(f'GEV: {gev_init}')
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Data Clustered [{init + 1}/{self.number_of_repeats}]\n"
                f"✓ Global Explained Variance: {100 * gev_init}%"
            )

        # Update the best results if current gev is higher
        if self.best_maps is None or gev_init > self.best_gev:
            self.best_residual = residual_init.copy() if hasattr(residual_init, 'copy') else residual_init
            self.best_gev = gev_init
            self.best_maps = maps_init.copy() if hasattr(maps_init, 'copy') else maps_init

            # Save the best maps immediately
            self.comet_microstate_io.export_microstates(self.best_maps, self.eeg_info, self.microstate_maps_path)

    def backfit_eeg(self, eeg_path, eeg_name):
        """
        Backfit microstate maps to an EEG file
        """
        eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)
        time_array = eeg.times * 1000

        labeled_segmentation, segmentation_fit = self.comet_microstate_backfitter. \
            perform_segmentation(
            eeg=eeg, filter_segments_less_than=int(self.filter_segments_less_than_ms / (1000 / self.sample_rate))
        )

        self.comet_segmentation_io.export_segmentation(
            output_folder=self.segmentation_path,
            filename=eeg_name,
            segmentation_array=labeled_segmentation,
            time_array=time_array,
            export_format=self.export_format
        )

        # Log the backfitting progress
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Microstates Backfitted: {eeg_name}"
            )

    def run_preprocessing(self):
        """
        Preprocess all EEG files in the input folder
        """
        print('\nPreprocessing ...')

        # Reset directories based on current parameters
        self.reset_directories()

        # Ensure directories exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.preprocessed_data_path, exist_ok=True)

        # Iterate through EEG files
        self.load_raw()

        # Check if any files were found
        if not hasattr(self, 'list_eegs_path') or len(self.list_eegs_path) == 0:
            print(f"Error: No EEG files found in {self.input_folder} with extension {self.extension}")
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log(
                    f"Error: No EEG files found in {self.input_folder} with extension {self.extension}"
                )
            return

        # Log the preprocessing progress
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Study Created\n"
                f"✓ Study Name: {self.study_name}\n"
                f"✓ Input Path: {self.input_folder}\n"
                f"✓ Output Path: {self.save_dir}\n"
                f"✓ Found {len(self.list_eegs_path)} {self.datatype} EEG data with {self.extension} extension."
            )

            self.LogWindow.append_log(
                f"EEG Preprocessing Settings:\n"
                f"* Channels to Remove: {self.chan2rm}\n"
                f"* Bandpass Filter: {self.filter_method.upper()} Method ({self.lowcut_freq}Hz and {self.highcut_freq}Hz)\n"
                f"* Downsampling Rate: {self.sample_rate}Hz\n",
                log_type='settings'
            )

        # Build a list of EEG files
        self.zipped_eeg_files = list(zip(self.list_eegs_path, self.list_eegs))

        # Check channel consistency
        self.check_chan2rm()

        if self.auto_clean_data:
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log(f"Auto-cleaning EEG data...")
                self.LogWindow.setup_progress_dialog(
                    window_title="Preprocessing ...",
                    label_text="Cleaning file ...",
                    tasks=self.zipped_eeg_files,
                    processing_func=self.autoclean_rseeg
                )
            else:
                print("Auto-cleaning EEG data...")
                for eeg_path, eeg_name in self.zipped_eeg_files:
                    self.autoclean_rseeg(eeg_path, eeg_name)

            self.load_clean()

        # Start preprocessing
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(f"Preprocessing {len(self.zipped_eeg_files)} EEG files...")
            self.LogWindow.setup_progress_dialog(
                window_title="Preprocessing ...",
                label_text="Preprocessing file ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.preprocess_eeg
            )
        else:
            print(f"Preprocessing {len(self.zipped_eeg_files)} EEG files...")
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Preprocessing"):
                self.preprocess_eeg(eeg_path, eeg_name)

        # Set preprocessing flag
        self.save_eeg_info(self.eeg_info_path)
        self.done_preprocessing = True

        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(f"Preprocessing completed. Data saved to {self.preprocessed_data_path}")
        else:
            print(f"Preprocessing completed. Data saved to {self.preprocessed_data_path}")

        # Save configuration and parameters
        if self.auto_save:
            self.save_config()

    def run_clustering(self):
        """
        Perform clustering on preprocessed EEG data
        """
        print('\nClustering ...')

        self.load_eeg_info()

        # Calculate minimum distance size if smoothing GFP is enabled
        if self.smoothing_gfp:
            self.min_distance_size = int(int(self.smoothing_distance) / (1000 / int(self.sample_rate)))
        else:
            self.min_distance_size = []

        # Check if clustering method is supported
        available_methods = [
            'Modified K-Means Clustering (Pascual-Marqui et al. 1995)',
            'Modified K-Means Clustering with Spatial Similarity',
            'Topographic Atomize and Agglomerate Hierarchical Clustering',
        ]
        if self.clustering_method not in available_methods:
            error_msg = f"Clustering method '{self.clustering_method}' not supported. Available methods: {', '.join(available_methods)}"
            print(error_msg)
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg)
            return

        # Initialize microstate clusterer
        self.comet_microstate_clusterer = MicrostateClusterer(
            n_states=self.number_of_maps,
            batch_size=self.batch_size,
            n_inits=self.number_of_repeats,
            max_iter=self.max_iterations,
            tolerance=self.clustering_tolerance
        )

        # Generate maps and peaks automatically if number_of_maps is set to 'auto'
        if self.number_of_maps == 'auto':
            self.maps2use, self.peaks2use = self.comet_data_initializer.generate_maps_and_peaks(
                preprocessed_folder=self.preprocessed_data_path,
                extension=self.extension,
                datatype=self.datatype,
                use_percentages=self.use_percentages,
                min_dist=self.min_distance_size
            )
            self.comet_clusterer_optimizer = ClustererOptimizer(
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
            self.optimal_k, self.k_values, self.target_values = self.comet_clusterer_optimizer.find_optimal_k(
                optimizer_mode=self.stopping_mode, parameter_value=self.stopping_parameter
            )
            self.number_of_maps = self.optimal_k
            print(f'Result: n_states = {self.number_of_maps}')

        # Perform clustering
        self.maps2use, self.peaks2use = self.comet_data_initializer.generate_maps_and_peaks(
            preprocessed_folder=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            use_percentages=self.use_percentages,
            min_dist=self.min_distance_size
        )

        if self.choose_number_of_maps == "auto":
            k_log = 'will be automatically determined.'
        else:
            k_log = 'is user-predefined.'

        if self.use_percentages is not None:
            cluster_data_log = f"{self.use_percentages}% randomly selected time-points of the data."
        else:
            cluster_data_log = 'the local peaks of the global field power.'

        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Clustering Settings:\n"
                f"* Clustering algorithm: {self.clustering_method}\n"
                f"* The number of maps to extract {k_log}\n"
                f"* Clustering will be performed on {cluster_data_log}", log_type='settings'
            )

        # Reset best values
        self.best_residual, self.best_maps = None, None
        self.best_gev = 0

        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.setup_progress_dialog(
                window_title="Clustering ...",
                label_text="Clustering ...",
                tasks=list(range(self.number_of_repeats)),
                processing_func=self.cluster_eeg_microstates
            )
        else:
            print(f"Running {self.number_of_repeats} clustering iterations...")
            for init in tqdm(range(self.number_of_repeats), desc="Clustering"):
                self.cluster_eeg_microstates(init)

        if self.best_maps is not None:
            self.best_gev = self.compute_gev_all_data()
            print(f'Global Explained Variance: {self.best_gev}')

            # Always set clustering flag to True if we have maps
            self.done_clustering = True

            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.process_finished(
                    f"✓ The data has been successfully clustered into {self.number_of_maps} microstates."
                    f"\nBest Global Explained Variance Achieved: {100 * self.best_gev:.3f}%"
                )
            else:
                print(f"✓ The data has been successfully clustered into {self.number_of_maps} microstates.")
                print(f"Best Global Explained Variance Achieved: {100 * self.best_gev:.3f}%")

            # Save updated configuration and parameters
            if self.auto_save:
                self.save_config()

    def run_microstate_labeling(self):
        """
        Label microstate maps
        """
        # Check if best maps are available
        if self.best_maps is None:
            print("Error: No microstate maps available for labeling.")
            return

        # Initialize microstate labeler
        self.comet_microstate_labeler = MicrostateLabeler(
            microstate_maps=self.best_maps, eeg_info=self.eeg_info, microstate_maps_path=self.microstate_maps_path
        )

        # Perform labeling
        # if hasattr(self, 'LogWindow') and self.LogWindow is not None:
        #     self.LogWindow.setup_progress_dialog(
        #         window_title="Labeling ...",
        #         label_text="Labeling microstates ...",
        #         max_value=1
        #     )

        print("Labeling microstates...")
        micro_labels, labels_overall_confidence = self.comet_microstate_labeler.do_labeling()

        # Save updated microstate maps with labels
        self.comet_microstate_io.export_microstates(
            self.best_maps, self.eeg_info, self.microstate_maps_path, headers=micro_labels
        )

        # Update and save labels
        self.load_maps()
        self.labels_overall_confidence = labels_overall_confidence

        # Set labeling flag
        self.done_microstate_labeling = True

        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.process_finished("✓ Microstates have been successfully labeled!")
        else:
            print("✓ Microstates have been successfully labeled!")

        # Save parameters
        if self.auto_save:
            self.save_config()

    def run_backfitting(self):
        """
        Backfit microstate maps to all EEG files
        """
        print('\nBackfitting ...')

        # Check if best maps are available
        if self.best_maps is None:
            print("Error: No microstate maps available for backfitting.")
            return

        # Create segmentation directory
        os.makedirs(self.segmentation_path, exist_ok=True)

        # Create an instance of the microstate backfitter
        self.comet_microstate_backfitter = MicrostateBackfitter(
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

        # Identify optimal window size if requested
        if self.identify_short_window:
            # Create progress dialog
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.setup_progress_dialog(
                    window_title="Backfitting ...",
                    label_text="Identifying the optimal length of the smoothing window ...",
                    max_value=len(self.list_eegs_path)
                )
            else:
                print("Identifying the optimal length of the smoothing window...")

            rm_max_len = 50
            len_win2rm_list = list(range(0, rm_max_len, int(1000 / self.sample_rate)))
            similarity_scores = np.empty((len(self.list_eegs_path), len(len_win2rm_list)))

            # Iterate through EEG files
            for eeg_idx, (eeg_path, eeg_name) in enumerate(zip(self.list_eegs_path, self.list_eegs)):
                if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                    self.LogWindow.update_progress(value=eeg_idx, text=f"{eeg_name}")
                else:
                    print(f"Processing {eeg_name} ({eeg_idx + 1}/{len(self.list_eegs_path)})")

                eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)

                # Compute similarity scores for different segment removal lengths
                for idx_win2rm, len_win2rm in enumerate(len_win2rm_list):
                    similarity_scores[eeg_idx, idx_win2rm] = self.comet_microstate_backfitter.get_similarity_score(
                        eeg=eeg, rm_max_len=len_win2rm
                    )

            # Identify optimal length filter
            self.filter_segments_less_than_ms = self.comet_microstate_backfitter.identify_optimal_length_filter(
                similarity_scores=similarity_scores
            )
            print(f"Optimal filter length: {self.filter_segments_less_than_ms} ms")
        else:
            if self.filter_segments:
                self.filter_segments_less_than_ms = self.filter_segments_less_than
            else:
                self.filter_segments_less_than_ms = 0

        # Log backfitting settings
        if self.backfit_to == 'peaks':
            backfit_to_text = "Backfitting microstates to the local peaks of the global field power."
        else:
            backfit_to_text = "Backfitting microstates to all time points."

        if self.filter_segments_option == 'remove':
            filter_segments_option_text = f"Removing segments with less than " \
                                          f"{self.filter_segments_less_than_ms}ms in duration."
        elif self.filter_segments_option == 'replace_high':
            filter_segments_option_text = f"Replacing segments with less than {self.filter_segments_less_than_ms}ms" \
                                          f"by the nearby microstate with higher occurrence."
        elif self.filter_segments_option == 'replace_half':
            filter_segments_option_text = f"Replacing segments with less than {self.filter_segments_less_than_ms}ms" \
                                          f"by half by the previous and half by the next dominant microstate."
        elif self.filter_segments_option == 'smooth':
            filter_segments_option_text = f"Smoothing segments with window size {self.filter_segments_less_than_ms}ms" \
                                          f" and lambda {self.lamb}."
        else:
            filter_segments_option_text = "No filtering applied to short segments."

        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Microstates Backfitting Settings:\n"
                f"* {backfit_to_text}\n"
                f"* {filter_segments_option_text}", log_type='settings'
            )
        else:
            print(f"Microstates Backfitting Settings:")
            print(f"* {backfit_to_text}")
            print(f"* {filter_segments_option_text}")

        self.load_clean()
        self.zipped_eeg_files = list(zip(self.list_eegs_path, self.list_eegs))

        # Perform backfitting on all files
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.setup_progress_dialog(
                window_title="Backfitting ...",
                label_text="Backfitting microstates to data ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.backfit_eeg
            )
        else:
            print(f"Backfitting microstates to {len(self.zipped_eeg_files)} EEG files...")
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Backfitting"):
                self.backfit_eeg(eeg_path, eeg_name)

        # Set backfitting flag
        self.done_backfitting = True

        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.process_finished("✓ Microstates have been successfully backfitted to the data!")
        else:
            print("✓ Microstates have been successfully backfitted to the data!")

        # Save parameters
        if self.auto_save:
            self.save_config()

    def run_feature_extraction(self):
        """
        Extract features from segmentation data
        """
        print('\nExtracting Features ...')

        # Check if backfitting has been done
        if not self.done_backfitting:
            print("Error: Backfitting must be completed before extracting features.")
            return

        os.makedirs(self.extracted_features_path, exist_ok=True)

        # Find segmentation files
        self.segmentation_list_path, self.segmentation_list_filename = self.comet_data_io.find_data(
            input_folder=self.segmentation_path,
            extension=self.export_format
        )

        # Make sure word_size is set if not already
        if not hasattr(self, 'word_size'):
            self.word_size = 3
            print(f"Setting default word size to {self.word_size}")

        # Initialize class-level shared storage for feature extraction
        COMET._shared_feature_results = {
            'averaged': {feature_type: None for feature_type in self.feature_types},
            'sliding': {feature_type: None for feature_type in self.feature_types}
        }

        # Initialize dictionaries to store results (these won't be used by the thread)
        self.averaged_features_dfs = {feature_type: None for feature_type in self.feature_types}
        self.sliding_features_dfs = {feature_type: None for feature_type in self.feature_types}

        # Create tasks for worker thread
        segmentation_tasks = [
            (idx, path, name) for idx, (path, name) in
            enumerate(zip(self.segmentation_list_path, self.segmentation_list_filename))
        ]

        # Log feature extraction settings
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Feature Extraction Settings:\n"
                f"* Features to Extract: {self.feature_list}\n"
                f"* Feature Type: {self.feature_mode}", log_type='settings'
            )

            # Use worker thread via setup_progress_dialog
            self.LogWindow.setup_progress_dialog(
                window_title="Extracting Features ...",
                label_text="Extracting features for data ...",
                tasks=segmentation_tasks,
                processing_func=self.extract_features_for_file_threadsafe
            )

            # Connect to the finished signal to collect results after thread completes
            self.LogWindow.worker_thread.finished.connect(self.collect_feature_extraction_results)
        else:
            # Process in main thread if no GUI
            print(f"Feature Extraction Settings:")
            print(f"* Features to Extract: {self.feature_list}")
            print(f"* Feature Type: {self.feature_mode}")

            for idx, path, name in tqdm(segmentation_tasks, desc="Extracting Features"):
                self.extract_features_for_file_threadsafe(idx, path, name)

            # Collect results and export in non-GUI mode
            self.collect_feature_extraction_results()

    def extract_features_for_file_threadsafe(self, segmentation_idx, segmentation_path, segmentation_name):
        """
        Thread-safe version of extract_features_for_file that uses class-level shared storage
        """
        # Load segmentation array
        segmentation_array = self.comet_segmentation_io.load_segmentation(
            segmentation_path=segmentation_path, import_format=self.export_format
        )

        # Process each feature type
        for feature_type in self.feature_types:
            # Process feature extraction based on type
            if feature_type == 'real':
                input_sequence = segmentation_array.flatten().tolist()
            else:
                input_sequence = FeatureHelper().generate_synthetic_sequence(
                    input_sequence=segmentation_array, method=feature_type
                )

            # Extract Averaged Features
            if 'averaged' in self.feature_mode:
                params = {
                    "input_sequence": input_sequence,
                    "sampling_rate": self.sample_rate,
                    "sliding_window_size": self.sliding_window_size,
                    "feature_mode": "averaged"
                }
                self.comet_feature_extractor = FeatureExtractor(**params)

                if 'GEV' in self.feature_list:
                    eeg_path, _ = self.comet_data_io.find_data(
                        input_folder=self.preprocessed_data_path,
                        extension=self.extension,
                        pattern=f"*{segmentation_name}*"
                    )
                    eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path[0], datatype=self.datatype)
                    eeg_data = self.comet_data_io.get_eeg_data(eeg=eeg, datatype=self.datatype)

                    output_features = self.comet_feature_extractor.extract_microstate_features(
                        filename=segmentation_name,
                        feature_list=self.feature_list,
                        eeg_data=eeg_data,
                        microstate_maps=self.best_maps,
                        microstate_labels=self.micro_labels,
                        word_size=self.word_size
                    )
                else:
                    output_features = self.comet_feature_extractor.extract_microstate_features(
                        filename=segmentation_name,
                        feature_list=self.feature_list,
                        word_size=self.word_size
                    )

                # Update shared storage instead of instance variables
                if segmentation_idx == 0:
                    COMET._shared_feature_results['averaged'][feature_type] = output_features
                else:
                    COMET._shared_feature_results['averaged'][feature_type] = pd.concat(
                        [COMET._shared_feature_results['averaged'][feature_type], output_features],
                        ignore_index=True
                    )

            # Extract Sliding Features
            if 'sliding' in self.feature_mode:
                params = {
                    "input_sequence": input_sequence,
                    "sampling_rate": self.sample_rate,
                    "sliding_window_size": self.sliding_window_size,
                    "feature_mode": "sliding"
                }
                if hasattr(self, "pre_window_size") and self.pre_window_size is not None:
                    params["pre_window_size"] = self.pre_window_size
                if hasattr(self, "post_window_size") and self.post_window_size is not None:
                    params["post_window_size"] = self.post_window_size

                self.comet_feature_extractor = FeatureExtractor(**params)

                if 'GEV' in self.feature_list:
                    eeg_path = os.path.join(self.preprocessed_data_path, f"{segmentation_name}{self.extension}")
                    eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)
                    eeg_data = self.comet_data_io.get_eeg_data(eeg=eeg, datatype=self.datatype)
                    output_features = self.comet_feature_extractor.extract_microstate_features(
                        filename=segmentation_name,
                        feature_list=self.feature_list,
                        eeg_data=eeg_data,
                        microstate_maps=self.best_maps,
                        microstate_labels=self.micro_labels
                    )
                else:
                    output_features = self.comet_feature_extractor.extract_microstate_features(
                        filename=segmentation_name, feature_list=self.feature_list
                    )

                # Update shared storage instead of instance variables
                if segmentation_idx == 0:
                    COMET._shared_feature_results['sliding'][feature_type] = output_features
                else:
                    COMET._shared_feature_results['sliding'][feature_type] = pd.concat(
                        [COMET._shared_feature_results['sliding'][feature_type], output_features],
                        ignore_index=True
                    )

        # Log the feature extraction progress
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Features Extracted [{segmentation_idx + 1}/{len(self.segmentation_list_path)}]\n"
                f"✓ Data: {segmentation_name}"
            )

    def collect_feature_extraction_results(self, message=None):
        """
        Collect feature extraction results from shared storage and export them
        This is called when the worker thread finishes
        """
        # Transfer results from shared storage to instance variables
        self.averaged_features_dfs = COMET._shared_feature_results.get('averaged', {})
        self.sliding_features_dfs = COMET._shared_feature_results.get('sliding', {})

        # Export extracted features
        for feature_type in self.feature_types:
            if 'averaged' in self.feature_mode and self.averaged_features_dfs.get(feature_type) is not None:
                self.comet_feature_io.export_features(
                    features_df=self.averaged_features_dfs[feature_type],
                    feature_type=feature_type,
                    feature_mode='averaged',
                    output_folder=self.extracted_features_path,
                    export_format=self.export_format
                )
                print(f"Exported averaged features for feature type: {feature_type}")

            if 'sliding' in self.feature_mode and self.sliding_features_dfs.get(feature_type) is not None:
                self.comet_feature_io.export_features(
                    features_df=self.sliding_features_dfs[feature_type],
                    feature_type=feature_type,
                    feature_mode='sliding',
                    output_folder=self.extracted_features_path,
                    export_format=self.export_format
                )
                print(f"Exported sliding features for feature type: {feature_type}")

        # Set feature extraction flag
        self.done_extracting_features = True

        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.process_finished("✓ All features have been successfully extracted!")
        else:
            print("✓ All features have been successfully extracted!")

        # Save parameters
        if self.auto_save:
            self.save_config()

        # Clear the shared storage to free memory
        COMET._shared_feature_results = {}

    def get_segmentation(self, subject_name):
        """
        Load a segmentation file on demand
        """
        seg_path = os.path.join(self.segmentation_path, f"{subject_name}{self.export_format}")
        return self.comet_segmentation_io.load_segmentation(
            segmentation_path=seg_path,
            import_format=self.export_format
        )

    def source_localize_file(self, eeg_path, eeg_name):
        """
        Process source localization for a single file (worker-friendly version).

        Parameters:
        -----------
        eeg_path : str
            Path to the EEG file
        eeg_name : str
            Name of the EEG file
        """
        success = self.comet_source_localizer.localize_single_file(eeg_path, eeg_name)

        # Log the source localization progress
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            if success:
                self.LogWindow.append_log(f"Source Localized: {eeg_name}")
            else:
                self.LogWindow.append_log(f"Error Source Localizing: {eeg_name}")

    def source_identify_file(self, eeg_path, eeg_name):
        """
        Process microstate source identification for a single file (worker-friendly version).

        Parameters:
        -----------
        eeg_path : str
            Path to the EEG file
        eeg_name : str
            Name of the EEG file
        """
        success = self.comet_source_localizer.identify_sources_single_file(
            eeg_path, eeg_name, self.source_localization_method
        )

        # Log the source identification progress
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            if success:
                self.LogWindow.append_log(f"Microstate Sources Identified: {eeg_name}")
            else:
                self.LogWindow.append_log(f"Error Identifying Microstate Sources: {eeg_name}")

    def run_source_localization(self):
        """
        Perform source localization for microstates
        """
        print("\nCalculating Source Time Series ...")

        # Check if backfitting has been done
        if not self.done_backfitting:
            print("Error: Backfitting must be completed before source localization.")
            return

        # Determine the subjects directory
        if self.use_anatomy == "individual":
            if hasattr(self, 'individual_subjects_dir'):
                self.anatomy_subjects_dir = self.individual_subjects_dir
            else:
                print("Error: individual_subjects_dir not set for individual anatomy")
                return
        else:  # use_anatomy == "fsaverage"
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.anatomy_subjects_dir = os.path.dirname(fs_dir)

        # Create directories for source localization results
        os.makedirs(self.localized_sources_path, exist_ok=True)
        stc_path = os.path.join(self.localized_sources_path, "stc")
        os.makedirs(stc_path, exist_ok=True)

        # Initialize the source localizer
        self.comet_source_localizer = SourceLocalizer(
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

        # Make sure the stc_path is set correctly in the source localizer
        self.comet_source_localizer.stc_path = stc_path

        # Load EEG files to process
        list_eeg_path, list_eeg_name = self.comet_data_io.find_data(
            self.preprocessed_data_path, extension=self.extension, pattern='*'
        )
        self.zipped_eeg_files = list(zip(list_eeg_path, list_eeg_name))

        # Log source localization settings
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Source Localization Settings:\n"
                f"* Anatomy: {self.use_anatomy}\n"
                f"* Inverse Method: {self.inverse_method}\n"
                f"* Spacing: {self.spacing}", log_type='settings'
            )

        # Perform source localization on all files
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.setup_progress_dialog(
                window_title="Source Localization ...",
                label_text="Localizing sources ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.source_localize_file
            )
        else:
            print(f"Source Localization Settings:")
            print(f"* Anatomy: {self.use_anatomy}")
            print(f"* Inverse Method: {self.inverse_method}")
            print(f"* Spacing: {self.spacing}")

            print(f"Localizing sources for {len(self.zipped_eeg_files)} EEG files...")
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Source Localization"):
                self.source_localize_file(eeg_path, eeg_name)

        # Set source localization flag
        self.done_source_localization = True

        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.process_finished("✓ Source localization completed")
        else:
            print("✓ Source localization completed")

        # Save parameters
        if self.auto_save:
            self.save_config()

    def run_identifying_microstate_sources(self, method='tess'):
        """
        Correlate sources and microstates
        """
        print("\nCorrelating sources and microstates ...")

        # Check if source localization has been done
        if not self.done_source_localization:
            print("Error: Source localization must be completed before correlation.")
            return

        # Check if anatomy subjects directory is available
        try:
            print(f"Using anatomy directory: {self.anatomy_subjects_dir}")
        except AttributeError:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.anatomy_subjects_dir = os.path.dirname(fs_dir)
            print(f"Using default anatomy directory: {self.anatomy_subjects_dir}")

        # Set the source localization method
        self.source_localization_method = method

        # Make sure the necessary paths are set correctly in the source localizer
        if not hasattr(self, 'comet_source_localizer') or self.comet_source_localizer is None:
            # Initialize the source localizer if it doesn't exist
            self.comet_source_localizer = SourceLocalizer(
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

        # Ensure directories are created and paths are set
        if method == 'tess':
            self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
            os.makedirs(self.tess_path, exist_ok=True)
            self.comet_source_localizer.tess_path = self.tess_path
        elif method == 'avg':
            self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")
            os.makedirs(self.avg_sources_path, exist_ok=True)
            self.comet_source_localizer.avg_sources_path = self.avg_sources_path

        # Make sure stc_path is set
        stc_path = os.path.join(self.localized_sources_path, "stc")
        self.comet_source_localizer.stc_path = stc_path

        # Load EEG files to process
        list_eeg_path, list_eeg_name = self.comet_data_io.find_data(
            self.preprocessed_data_path, extension=self.extension, pattern='*'
        )
        self.zipped_eeg_files = list(zip(list_eeg_path, list_eeg_name))

        # Log source identification settings
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Source-Microstate Correlation Settings:\n"
                f"* Method: {method}\n"
                f"* Number of permutations: {self.nperm}", log_type='settings'
            )

        # Perform source identification on all files
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.setup_progress_dialog(
                window_title="Source Identification ...",
                label_text=f"Identifying microstate sources using {method} method ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.source_identify_file
            )
        else:
            print(f"Source-Microstate Correlation Settings:")
            print(f"* Method: {method}")
            print(f"* Number of permutations: {self.nperm}")

            print(f"Identifying microstate sources for {len(self.zipped_eeg_files)} EEG files...")
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Source Identification"):
                self.source_identify_file(eeg_path, eeg_name)

        # Set source-microstate correlation flag
        self.done_identifying_microstate_sources = True

        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.process_finished(f"✓ Source-microstate correlation completed using {method} method")
        else:
            print(f"✓ Source-microstate correlation completed using {method} method")

        # Save parameters
        if self.auto_save:
            self.save_config()

    def save_config(self):
        """
        Update and save the current configuration settings and program state to save_dir
        """
        # Update the config dictionary with current attributes
        self.config["io_config"]["study_name"] = self.study_name
        self.config["io_config"]["input_folder"] = self.input_folder
        self.config["io_config"]["montage"] = self.montage
        self.config["io_config"]["extension"] = self.extension
        self.config["io_config"]["pattern_content"] = self.pattern_content
        self.config["io_config"]["datatype"] = self.datatype
        self.config["io_config"]["output_folder"] = self.output_folder

        self.config["preprocessing_config"]["temporal_filter_data"] = str(self.temporal_filter_data)
        self.config["preprocessing_config"]["filter_method"] = self.filter_method
        self.config["preprocessing_config"]["lowcut_freq"] = str(self.lowcut_freq)
        self.config["preprocessing_config"]["highcut_freq"] = str(self.highcut_freq)
        self.config["preprocessing_config"]["downsample_data"] = str(self.downsample_data)
        self.config["preprocessing_config"]["spatial_filter_data"] = str(self.spatial_filter_data)
        self.config["preprocessing_config"]["auto_clean_data"] = str(self.auto_clean_data)
        self.config["preprocessing_config"]["sample_rate"] = str(self.sample_rate)
        self.config["preprocessing_config"]["remove_channels"] = str(self.remove_channels)
        self.config["preprocessing_config"]["ch2rm"] = str(self.chan2rm)
        self.config["preprocessing_config"]["prep_data"] = str(self.prep_data)

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
        self.config["clustering_config"]["similarity_metric"] = self.similarity_metric
        self.config["clustering_config"]["number_of_repeats"] = str(self.number_of_repeats)

        self.config["backfitting_config"]["backfit_to"] = self.backfit_to
        self.config["backfitting_config"]["identify_short_window"] = str(self.identify_short_window)
        self.config["backfitting_config"]["filter_segments"] = str(self.filter_segments)
        self.config["backfitting_config"]["filter_segments_less_than"] = str(self.filter_segments_less_than)
        self.config["backfitting_config"]["filter_segments_option"] = self.filter_segments_option
        self.config["backfitting_config"]["epsilon"] = str(self.epsilon)
        self.config["backfitting_config"]["b"] = str(self.b)
        self.config["backfitting_config"]["lamb"] = str(self.lamb)

        self.config["features_config"]["export_format"] = self.export_format
        self.config["features_config"]["feature_list"] = ', '.join(self.feature_list)
        self.config["features_config"]["feature_mode"] = ', '.join(self.feature_mode)
        self.config["features_config"]["feature_types"] = ', '.join(self.feature_types)
        self.config["features_config"]["sliding_window_size"] = str(self.sliding_window_size)
        self.config["features_config"]["pre_window_size"] = str(self.pre_window_size)
        self.config["features_config"]["post_window_size"] = str(self.post_window_size)

        self.config["source_config"]["inverse_method"] = self.inverse_method
        self.config["source_config"]["nperm"] = str(self.nperm)
        self.config["source_config"]["spacing"] = self.spacing
        self.config["source_config"]["source_localization_method"] = self.source_localization_method
        self.config["source_config"]["anatomy_subjects_dir"] = self.anatomy_subjects_dir

        # Add a new section for state flags
        if "state_flags" not in self.config:
            self.config.add_section("state_flags")

        # Save the program state
        self.config["state_flags"]["done_preprocessing"] = str(self.done_preprocessing)
        self.config["state_flags"]["done_clustering"] = str(self.done_clustering)
        self.config["state_flags"]["done_microstate_labeling"] = str(self.done_microstate_labeling)
        self.config["state_flags"]["done_backfitting"] = str(self.done_backfitting)
        self.config["state_flags"]["done_extracting_features"] = str(self.done_extracting_features)
        self.config["state_flags"]["done_source_localization"] = str(self.done_source_localization)
        self.config["state_flags"]["done_identifying_microstate_sources"] = str(self.done_identifying_microstate_sources)

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        try:
            # Write config to file in save_dir
            with open(self.config_path, 'w+') as configfile:
                self.config.write(configfile)
        except Exception as e:
            print(f"Error saving configuration: {e}")
            print("Check directory permissions.")

    # def cleanup_intermediate_files(self, keep_preprocessed=True, keep_important=True):
    #     """
    #     Clean up intermediate files to save disk space.
    #
    #     Parameters:
    #     -----------
    #     keep_preprocessed : bool
    #         If True, keep preprocessed EEG files
    #     keep_important : bool
    #         If True, keep final results (maps, features, segmentation)
    #     """
    #     import shutil
    #
    #     # Always keep these if keep_important is True
    #     important_paths = []
    #     if keep_important:
    #         important_paths = [
    #             self.microstate_maps_path,
    #             self.extracted_features_path,
    #             self.segmentation_path,
    #             self.localized_sources_path
    #         ]
    #
    #     # Add preprocessed data if requested
    #     if keep_preprocessed:
    #         important_paths.append(self.preprocessed_data_path)
    #
    #     # Temp paths that could be deleted to save space
    #     temp_dirs = []
    #
    #     # Only delete directories not in important_paths
    #     for path in temp_dirs:
    #         if path not in important_paths and os.path.exists(path):
    #             if os.path.isdir(path):
    #                 shutil.rmtree(path)
    #             else:
    #                 os.remove(path)
    #             print(f"Removed {path}")
    #
    #     print("Cleanup completed")