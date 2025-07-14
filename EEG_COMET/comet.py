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
from features_utils.feature_extractor import FeatureExtractor, FeatureExtractionCoordinator
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
        Initialize COMET instance with optimized memory management.
        """
        # Initialize callback for clustering completion
        self.clustering_completed_callback = None
        
        # Initialize basic attributes
        self.auto_save = auto_save
        self.config = None
        self.config_path = config_path
        self.log_text = ""  # Initialize as string instead of list

        # Define all instance variables
        self.LogWindow = None
        self.study_name = study_name
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Initialize utility objects
        self.comet_data_io = DataIO()
        self.comet_preprocessor = DataPreprocessor()
        self.comet_data_initializer = DataInitializer()
        self.comet_microstate_io = MicrostateIO()
        self.comet_segmentation_io = SegmentationIO()
        self.comet_feature_io = FeatureIO()
        self.comet_feature_helper = FeatureHelper()
        self.comet_feature_extractor = FeatureExtractionCoordinator()

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
        
        # Load logs from separate file after directories are set up
        self.log_text = self.load_logs_from_file()

        # Initialize flags and data holders
        self.done_preprocessing = False
        self.done_clustering = False
        self.done_microstate_labeling = False
        self.done_backfitting = False
        self.done_extracting_features = False
        self.done_source_localization = False
        self.done_identifying_microstate_sources = False

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
        config["source_config"]["bem_solver"] = "mne"
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
        self.log_file_path = os.path.join(self.save_dir, "eeg_comet_log.txt")
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
            print(f"[WARNING] Config file {config_path} not found. Using default configuration.")
            return self.create_default_config()

        config = ConfigParser()
        try:
            # Use UTF-8 encoding to properly handle Unicode characters
            config.read(config_path, encoding='utf-8')
            return config
        except Exception as e:
            print(f"[WARNING] Failed to load configuration: {e}")
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
            "TP": "Transition Probability", "ER": "Entropy Rate", "LZC": "Sequence Lempel-Ziv Complexity",
            "HE": "Hurst Exponent", "ERR": "Sequence Entropy Representation",
            "ROF": "Relative Occurrence Frequency", "RTF": "Relative Transition Frequency"
        }

        # Source Localization Configs
        source_config = self.config["source_config"]
        self.use_anatomy = source_config.get("use_anatomy", "fsaverage")
        self.bem_solver = source_config.get("bem_solver", "mne")
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

        # Load metadata if available
        if "metadata" in self.config:
            metadata = self.config["metadata"]
            self.config_last_saved = metadata.get("last_saved", "Unknown")
        else:
            self.config_last_saved = "Unknown"

        # Initialize log_text - will be loaded after directories are set up
        self.log_text = ""

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
        self.log_file_path = os.path.join(self.save_dir, "eeg_comet_log.txt")
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
        self.LogWindow = LogWindow(comet_instance=self)
        
        # Add default session messages (will be replaced if logs are restored)
        self.LogWindow.append_log("EEG-COMET Session Started", log_type='section')
        self.LogWindow.append_log("Welcome to EEG-COMET (EEG Comprehensive Microstate Extraction Toolbox)", log_type='info')
        self.LogWindow.append_log("Ready to process EEG microstate data.", log_type='info')

    def restore_logs_if_available(self):
        """
        Restore saved logs if available, called after config is loaded
        """
        # Load logs from file if not already loaded
        if not hasattr(self, 'log_text') or not self.log_text:
            self.log_text = self.load_logs_from_file()
        
        if hasattr(self, 'log_text') and self.log_text and hasattr(self, 'LogWindow') and self.LogWindow:
            self.LogWindow.set_log_content(self.log_text)
            return True
        return False

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
        self.load_maps()

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
        eeg_info.set_montage(montage, match_case=False, on_missing='warn')

        mne.io.write_info(eeg_info_path, eeg_info)

    def load_eeg_info(self):
        """
        Load EEG info from file instead of keeping it in memory
        """
        self.eeg_info = mne.io.read_info(self.eeg_info_path)

    def preprocess_eeg(self, eeg_path, eeg_name):
        """
        Preprocess a single EEG file
        """
        # Load EEG data
        eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)
        
        # Log file processing start
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            file_info = f"{eeg_name} | Channels: {len(eeg.ch_names)} | Duration: {eeg.times[-1]:.1f}s | Sampling Rate: {eeg.info['sfreq']}Hz"
            self.LogWindow.append_log(f"Processing {file_info}", log_type='file')

        # Apply montage if specified
        if hasattr(self, 'montage') and self.montage:
            montage_obj = self.comet_data_io.load_montage(self.montage)
            eeg.set_montage(montage_obj, match_case=False, on_missing='warn')

        # Remove channels if specified
        if hasattr(self, 'chan2rm') and self.chan2rm:
            channels_to_remove = [ch.strip() for ch in self.chan2rm.split(',') if ch.strip()]
            if channels_to_remove:
                eeg.drop_channels(channels_to_remove, on_missing='ignore')

        # Apply automatic bad channel detection and interpolation if requested
        if hasattr(self, 'prep_data') and self.prep_data:
            eeg = self.comet_preprocessor.identify_bad_channels(eeg)
            eeg.interpolate_bads(reset_bads=False)

        # Apply preprocessing steps
        preprocessed_eeg = self.comet_preprocessor.preprocess_eeg(
            eeg=eeg, 
            filter_bool=self.temporal_filter_data,
            filtermethod=self.filter_method,
            lowcut=self.lowcut_freq,
            highcut=self.highcut_freq,
            downsample_bool=self.downsample_data,
            sampling_rate=self.sample_rate,
            spatial_smooth_bool=self.spatial_filter_data
        )

        # Save preprocessed data
        name = os.path.splitext(eeg_name)[0]
        save_path = os.path.join(self.preprocessed_data_path, name)
        self.comet_data_io.export_eegs(eeg=preprocessed_eeg, save_path=save_path, extension=self.extension, datatype=self.datatype)
        
        # Log successful processing
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(f"Successfully preprocessed {eeg_name}", log_type='success')

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
        Perform modified K-means clustering iteration with enhanced TAAHC progress tracking
        """
        is_taahc = (self.clustering_method == "Topographic Atomize and Agglomerate Hierarchical Clustering")

        # Initialize cluster centers (not needed for TAAHC but kept for consistency)
        if not is_taahc:
            initial_maps = self.comet_data_initializer.initialize_cluster_centers(
                maps2use=self.maps2use, n_states=self.number_of_maps, initializer=self.initializer
            )
        else:
            self.number_of_repeats = 1

        # Perform clustering based on selected method
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
            # For TAAHC, create a comprehensive progress callback
            def taahc_progress_callback(current_step, total_steps, message):
                if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                    if hasattr(self.LogWindow, 'worker_thread') and self.LogWindow.worker_thread:
                        # Calculate overall progress including the current clustering iteration
                        overall_current = (init * total_steps) + current_step
                        overall_total = self.number_of_repeats * total_steps

                        # Create detailed progress message
                        detailed_message = f"TAAHC Iteration {init + 1}/{self.number_of_repeats}: {message}"

                        # Emit progress signal from worker thread
                        self.LogWindow.worker_thread.progress_updated.emit(
                            overall_current,
                            detailed_message
                        )

            # Run TAAHC with progress tracking using the original function signature
            maps_init, residual_init = self.comet_microstate_clusterer.taahc(
                data=self.maps2use,
                metric=self.similarity_metric,
                verbose=True,
                progress_callback=taahc_progress_callback
            )

        # Compute Global Explained Variance
        gev_init = self.comet_microstate_clusterer.compute_gev(data=self.maps2use, maps=maps_init)

        # Log iteration results
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            if is_taahc:
                log_message = (
                    f"TAAHC Iteration [{init + 1}/{self.number_of_repeats}] Completed\n"
                    f"✓ Hierarchical clustering processed {self.maps2use.shape[1]} timepoints\n"
                    f"✓ Global Explained Variance: {100 * gev_init:.3f}%"
                )
            else:
                log_message = (
                    f"Data Clustered [{init + 1}/{self.number_of_repeats}]\n"
                    f"✓ Global Explained Variance: {100 * gev_init:.3f}%"
                )

            self.LogWindow.append_log(log_message)

        # Update the best results if current gev is higher
        if self.best_maps is None or gev_init > self.best_gev:
            self.best_residual = residual_init.copy() if hasattr(residual_init, 'copy') else residual_init
            self.best_gev = gev_init
            self.best_maps = maps_init.copy() if hasattr(maps_init, 'copy') else maps_init

            # Save the best maps immediately
            self.comet_microstate_io.export_microstates(self.best_maps, self.eeg_info, self.microstate_maps_path)

            # Additional logging for best result updates
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log(
                    f"✓ New best result found in iteration {init + 1} - GEV: {100 * gev_init:.3f}%")

    def backfit_eeg(self, eeg_path, eeg_name):
        """
        Backfit microstate maps to an EEG file
        """
        eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)
        time_array = eeg.times * 1000

        # Log file processing start
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            file_info = f"{eeg_name} | Duration: {eeg.times[-1]:.1f}s | {len(eeg.ch_names)} channels"
            self.LogWindow.append_log(f"Backfitting {file_info}", log_type='file')

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
            segments_count = len(labeled_segmentation)
            self.LogWindow.append_log(f"Successfully backfitted {eeg_name} ({segments_count} segments identified)", log_type='success')

    def run_preprocessing(self):
        """
        Preprocess all EEG files in the input folder
        """
        print('\n[INFO] Starting data preprocessing...')

        # Reset directories based on current parameters
        self.reset_directories()

        # Ensure directories exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.preprocessed_data_path, exist_ok=True)

        # Iterate through EEG files
        self.load_raw()

        # Check if any files were found
        if not hasattr(self, 'list_eegs_path') or len(self.list_eegs_path) == 0:
            error_msg = f"[ERROR] No EEG files found in {self.input_folder} with extension {self.extension}"
            print(error_msg)
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type='error')
            return

        # Log the preprocessing start
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log("Data Preprocessing", log_type='section')
            
            # Study information
            study_info = (
                f"Study Name: {self.study_name}\n"
                f"Input Directory: {self.input_folder}\n"
                f"Output Directory: {self.save_dir}\n"
                f"Files Found: {len(self.list_eegs_path)} {self.datatype} EEG files with {self.extension} extension"
            )
            self.LogWindow.append_log(study_info, log_type='info')

            # Preprocessing settings
            filter_info = f"Bandpass Filter: {self.filter_method.upper()} method ({self.lowcut_freq}-{self.highcut_freq} Hz)" if self.temporal_filter_data else "Bandpass Filter: Disabled"
            spatial_info = f"Spatial Filtering: {'Enabled' if self.spatial_filter_data else 'Disabled'}"
            downsample_info = f"Downsampling: {self.sample_rate} Hz" if self.downsample_data else "Downsampling: Disabled"
            channels_info = f"Channels to Remove: {self.chan2rm}" if self.chan2rm else "Channels to Remove: None"
            
            settings_text = f"{filter_info}\n{spatial_info}\n{downsample_info}\n{channels_info}"
            self.LogWindow.append_log(settings_text, log_type='settings')

        # Build a list of EEG files
        self.zipped_eeg_files = list(zip(self.list_eegs_path, self.list_eegs))

        # Check channel consistency
        self.check_chan2rm()

        # Start preprocessing
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(f"Starting preprocessing of {len(self.zipped_eeg_files)} files...", log_type='process')
            self.LogWindow.setup_progress_dialog(
                window_title="Preprocessing ...",
                label_text="Preprocessing file ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.preprocess_eeg
            )
            # Set callback to run when preprocessing worker thread finishes
            self.LogWindow.process_finished_callback = self._on_preprocessing_finished
        else:
            print(f"[INFO] Processing {len(self.zipped_eeg_files)} EEG files...")
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Preprocessing"):
                self.preprocess_eeg(eeg_path, eeg_name)
            # For non-GUI mode, call completion directly
            self._on_preprocessing_finished()

    def run_clustering(self):
        """
        Perform clustering on preprocessed EEG data with automatic or manual k selection
        Enhanced with proper TAAHC progress tracking and batch processing support
        """
        print('\n[INFO] Starting microstate clustering...')

        self.load_eeg_info()

        # Calculate minimum distance size if smoothing GFP is enabled
        if self.smoothing_gfp:
            self.min_distance_size = int(int(self.smoothing_distance) / (1000 / int(self.sample_rate)))
        else:
            self.min_distance_size = None

        # Check if clustering method is supported
        available_methods = [
            'Modified K-Means Clustering (Pascual-Marqui et al. 1995)',
            'Modified K-Means Clustering with Spatial Similarity',
            'Topographic Atomize and Agglomerate Hierarchical Clustering',
        ]
        if self.clustering_method not in available_methods:
            error_msg = f"[ERROR] Clustering method '{self.clustering_method}' not supported. Available methods: {', '.join(available_methods)}"
            print(error_msg)
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg)
            return

        # Generate maps and peaks for clustering
        self.maps2use, self.peaks2use = self.comet_data_initializer.generate_maps_and_peaks(
            preprocessed_folder=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            use_percentages=self.use_percentages,
            min_dist=self.min_distance_size
        )

        # Special handling for TAAHC method
        is_taahc = (self.clustering_method == "Topographic Atomize and Agglomerate Hierarchical Clustering")

        if is_taahc:
            # TAAHC requires special batch processing setup
            if not hasattr(self, 'batch_size') or self.batch_size is None:
                self.batch_size = 10000  # Default large batch for TAAHC
                print(f"[INFO] TAAHC: Setting batch size to {self.batch_size}")

            # TAAHC works better with larger datasets, so adjust use_percentages if too small
            if self.use_percentages and self.use_percentages < 20:
                print(f"[INFO] TAAHC: Increasing data percentage from {self.use_percentages}% to 50% for better clustering")
                self.use_percentages = 50
                # Regenerate maps with higher percentage
                self.maps2use, self.peaks2use = self.comet_data_initializer.generate_maps_and_peaks(
                    preprocessed_folder=self.preprocessed_data_path,
                    extension=self.extension,
                    datatype=self.datatype,
                    use_percentages=self.use_percentages,
                    min_dist=self.min_distance_size
                )

        # Handle automatic number of maps selection
        if self.number_of_maps == 'auto':
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log("Determining optimal number of clusters...")

                # Create wrapper task for worker thread
                optimization_task = [('auto_optimization',)]

                # Use setup_progress_dialog for worker thread
                self.LogWindow.setup_progress_dialog(
                    window_title="Finding Optimal Number of Clusters...",
                    label_text="Running optimization methods...",
                    tasks=optimization_task,
                    processing_func=self._run_automatic_optimization_worker
                )

                # Wait for the worker to finish before continuing
                if hasattr(self.LogWindow, 'worker_thread') and self.LogWindow.worker_thread:
                    self.LogWindow.worker_thread.wait()

            else:
                # Non-GUI mode - run directly
                self._run_automatic_optimization_direct()

        # After determining number_of_maps, perform actual clustering
        if self.number_of_maps and self.number_of_maps != 'auto':
            # Initialize microstate clusterer with special settings for TAAHC
            if is_taahc:
                # TAAHC typically needs fewer repetitions since it's deterministic
                clustering_repeats = min(self.number_of_repeats, 3)
                if clustering_repeats != self.number_of_repeats:
                    print(
                        f"[INFO] TAAHC: Reducing repetitions from {self.number_of_repeats} to {clustering_repeats}")
            else:
                clustering_repeats = self.number_of_repeats

            self.comet_microstate_clusterer = MicrostateClusterer(
                n_states=self.number_of_maps,
                batch_size=self.batch_size,
                n_inits=clustering_repeats,
                max_iter=self.max_iterations,
                tolerance=self.clustering_tolerance
            )

            # Log clustering settings with TAAHC-specific information
            if self.choose_number_of_maps == "auto":
                k_log = f'automatically determined to be {self.number_of_maps}'
            else:
                k_log = f'set to {self.number_of_maps} (user-defined)'

            if self.use_percentages is not None:
                cluster_data_log = f"{self.use_percentages}% randomly selected time-points"
            else:
                cluster_data_log = 'local peaks of the global field power'

            # Enhanced clustering settings
            additional_settings = ""
            if self.clustering_method == "Topographic Atomize and Agglomerate Hierarchical Clustering":
                additional_settings = f"\nBatch Size: {self.batch_size if self.batch_size else 'Full dataset'}"

            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log("Microstate Clustering", log_type='section')
                
                clustering_info = (
                    f"Algorithm: {self.clustering_method}\n"
                    f"Number of Maps: {k_log}\n"
                    f"Data Selection: {cluster_data_log}\n"
                    f"Repetitions: {self.number_of_repeats}\n"
                    f"Max Iterations: {self.max_iterations}\n"
                    f"Convergence Tolerance: {self.clustering_tolerance}"
                    f"{additional_settings}"
                )
                self.LogWindow.append_log(clustering_info, log_type='settings')

            # Reset best values
            self.best_residual, self.best_maps = None, None
            self.best_gev = 0

            # Perform clustering iterations with enhanced progress for TAAHC
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                if is_taahc:
                    # Special progress setup for TAAHC
                    window_title = "TAAHC Clustering ..."
                    label_text = "Performing Topographic Atomize and Agglomerate Hierarchical Clustering ..."

                    # Estimate total progress steps for TAAHC
                    n_timepoints = self.maps2use.shape[1]
                    estimated_peaks = min(n_timepoints // 10, 1000)
                    iterations_needed = max(estimated_peaks - self.number_of_maps, 0)
                    estimated_steps_per_iteration = 10 + iterations_needed * 5 + 20
                    total_estimated_steps = clustering_repeats * estimated_steps_per_iteration

                    # Set up progress dialog with dynamic range
                    self.LogWindow.setup_progress_dialog(
                        window_title=window_title,
                        label_text=label_text,
                        tasks=list(range(clustering_repeats)),
                        processing_func=self.cluster_eeg_microstates
                    )

                    # Update progress bar range for TAAHC
                    self.LogWindow.ui.progress_bar.setRange(0, total_estimated_steps)

                else:
                    # Standard clustering progress
                    self.LogWindow.setup_progress_dialog(
                        window_title="Clustering ...",
                        label_text="Clustering ...",
                        tasks=list(range(clustering_repeats)),
                        processing_func=self.cluster_eeg_microstates
                    )

                # NEW LINE: Ensure we finalize clustering once the worker is done
                self.LogWindow.process_finished_callback = self._on_clustering_iterations_finished
            else:
                print(f"[INFO] Running {clustering_repeats} clustering iterations...")
                from tqdm import tqdm
                for init in tqdm(range(clustering_repeats), desc="Clustering"):
                    self.cluster_eeg_microstates(init)

            # Compute final GEV and save results
            if self.best_maps is not None:
                self.best_gev = self.compute_gev_all_data()
                print(f'[INFO] Global Explained Variance: {100 * self.best_gev:.3f}%')

                # Always set clustering flag to True if we have maps
                self.done_clustering = True

                # Log completion with method-specific information
                completion_message = f"✓ Clustering completed successfully: {self.number_of_maps} microstates (GEV: {100 * self.best_gev:.3f}%)"

                # Only print immediately if NOT using GUI (to avoid duplicate with callback)
                if not (hasattr(self, 'LogWindow') and self.LogWindow is not None):
                    print(completion_message)
                
                if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                    self.LogWindow.process_finished(completion_message)

                # Save updated configuration and parameters
                if self.auto_save:
                    self.save_config()
                    
                # Ensure logs are saved
                self._save_logs()
                
                # Call clustering completion callback if set
                if self.clustering_completed_callback is not None:
                    self.clustering_completed_callback()

            else:
                error_msg = "[ERROR] Clustering failed - no microstate maps were generated"
                if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                    self.LogWindow.append_log(error_msg)
                else:
                    print(error_msg)

    def _run_automatic_optimization_worker(self, task_name):
        """
        Run automatic optimization using majority vote in worker thread.
        This method is designed to be called by the LogWindow's worker thread.
        """

        # Create optimizer with progress callback
        def progress_callback(current, total, message):
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                # Emit progress signal from worker thread
                if hasattr(self.LogWindow, 'worker_thread') and self.LogWindow.worker_thread:
                    self.LogWindow.worker_thread.progress_updated.emit(
                        int((current / total) * 100),
                        message
                    )

        # Initialize optimizer
        self.comet_clusterer_optimizer = ClustererOptimizer(
            maps2use=self.maps2use,
            min_dist=self.min_distance_size,
            n_inits=1,  # Single repeat per k as requested
            kmin=self.kmin,
            kmax=self.kmax,
            preprocessed_data_path=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            tolerance=self.clustering_tolerance,
            max_iter=self.max_iterations,
            progress_callback=progress_callback
        )

        # Run automatic optimization
        self._run_automatic_optimization_core()

    def _run_automatic_optimization_direct(self):
        """
        Run automatic optimization directly (non-GUI mode).
        """
        # Initialize optimizer without progress callback
        self.comet_clusterer_optimizer = ClustererOptimizer(
            maps2use=self.maps2use,
            min_dist=self.min_distance_size,
            n_inits=1,  # Single repeat per k as requested
            kmin=self.kmin,
            kmax=self.kmax,
            preprocessed_data_path=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            tolerance=self.clustering_tolerance,
            max_iter=self.max_iterations
        )

        # Run automatic optimization
        self._run_automatic_optimization_core()

    def _run_automatic_optimization_core(self):
        """
        Core automatic optimization logic using majority vote.
        """
        # Define methods to use for majority vote
        methods_to_use = ['gev', 'res', 'sil', 'ch', 'db']

        # Conditionally include cross-validation if we have enough samples
        if self.maps2use.shape[1] >= 50:  # Need enough samples for meaningful CV
            methods_to_use.append('cv')

        # Skip gap statistic by default as it's computationally expensive
        # Can be enabled if needed
        # methods_to_use.append('gs')

        # Run majority vote optimization
        optimal_k = self.comet_clusterer_optimizer.find_optimal_k_majority_vote(methods_to_use)

        # Store results
        self.number_of_maps = optimal_k
        self.optimization_results = self.comet_clusterer_optimizer.get_all_results()

        # Log results
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(f"\n✓ Optimal number of clusters determined: {optimal_k}")

            # Log individual method results
            self.LogWindow.append_log("\nIndividual optimization method results:")

            method_names = {
                'gev': 'Elbow - Global Explained Variance',
                'res': 'Elbow - Residual Variance',
                'sil': 'Silhouette Method',
                'ch': 'Calinski-Harabasz Method',
                'db': 'Davies-Bouldin Method',
                'cv': 'Cross-Validation',
                'gs': 'Gap Statistic'
            }

            for method in methods_to_use:
                if method in self.optimization_results:
                    k = self.optimization_results[method].optimal_k
                    name = method_names.get(method, method.upper())
                    self.LogWindow.append_log(f"  - {name}: k = {k}")

            # Log majority vote summary
            if 'majority_vote' in self.optimization_results:
                votes = self.optimization_results['majority_vote'].scores
                self.LogWindow.append_log(f"\nMajority vote: {votes}")

        else:
            print(f"[INFO] Optimal number of clusters determined: {optimal_k}")

    def run_microstate_labeling(self):
        """
        Label microstate maps
        """
        # Check if best maps are available
        if self.best_maps is None:
            print("[ERROR] No microstate maps available for labeling")
            return

        # Initialize microstate labeler
        self.comet_microstate_labeler = MicrostateLabeler(
            microstate_maps=self.best_maps, eeg_info=self.eeg_info, microstate_maps_path=self.microstate_maps_path
        )

        # Perform labeling
        print("[INFO] Starting microstate labeling...")
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

        # Always print completion to terminal for important steps
        print("✓ Microstate labeling completed successfully")
        
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.process_finished("✓ Microstates have been successfully labeled!")

        # Save parameters
        if self.auto_save:
            self.save_config()

    def run_backfitting(self):
        """
        Backfit microstate maps to all EEG files
        """
        print('\n[INFO] Starting microstate backfitting...')

        # Check if best maps are available
        if self.best_maps is None:
            print("[ERROR] No microstate maps available for backfitting")
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
                print("[INFO] Identifying optimal smoothing window length...")

            rm_max_len = 50
            len_win2rm_list = list(range(0, rm_max_len, int(1000 / self.sample_rate)))
            similarity_scores = np.empty((len(self.list_eegs_path), len(len_win2rm_list)))

            # Iterate through EEG files
            for eeg_idx, (eeg_path, eeg_name) in enumerate(zip(self.list_eegs_path, self.list_eegs)):
                if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                    self.LogWindow.update_progress(value=eeg_idx, text=f"{eeg_name}")
                # Removed individual file processing print for cleaner output

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
            print(f"[INFO] Optimal filter length determined: {self.filter_segments_less_than_ms} ms")
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
            # Set callback to run when backfitting worker thread finishes
            self.LogWindow.process_finished_callback = self._on_backfitting_finished
        else:
            print(f"[INFO] Processing {len(self.zipped_eeg_files)} EEG files...")
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Backfitting"):
                self.backfit_eeg(eeg_path, eeg_name)
            # For non-GUI mode, call completion directly
            self._on_backfitting_finished()

    def run_feature_extraction(self):
        """
        Extract features from all segmentation files
        """
        print('\n[INFO] Starting feature extraction...')

        # Create features output directory
        os.makedirs(self.extracted_features_path, exist_ok=True)

        # Check if segmentation files exist
        self.segmentation_list_path, self.segmentation_list = self.comet_data_io.find_data(
            input_folder=self.segmentation_path, extension=self.export_format, pattern='*'
        )

        if not self.segmentation_list_path:
            error_msg = f"[ERROR] No segmentation files found in {self.segmentation_path}"
            print(error_msg)
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type='error')
            return

        # Log feature extraction start
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log("Feature Extraction", log_type='section')
            
            # Feature extraction settings
            feature_info = (
                f"Features to Extract: {', '.join(self.feature_list)}\n"
                f"Feature Modes: {', '.join(self.feature_mode)}\n"
                f"Feature Types: {', '.join(self.feature_types)}\n"
                f"Segmentation Files: {len(self.segmentation_list_path)}"
            )
            
            if 'sliding' in self.feature_mode:
                feature_info += f"\nSliding Window Size: {self.sliding_window_size}"
                if hasattr(self, 'pre_window_size'):
                    feature_info += f"\nPre-Window Size: {self.pre_window_size}"
                if hasattr(self, 'post_window_size'):
                    feature_info += f"\nPost-Window Size: {self.post_window_size}"
            
            self.LogWindow.append_log(feature_info, log_type='settings')

        # Initialize shared storage for thread-safe feature extraction
        COMET._shared_feature_results = {}

        # Create list of tasks for threading
        tasks = [(idx, path, name) for idx, (path, name) in enumerate(zip(self.segmentation_list_path, self.segmentation_list))]

        # Start feature extraction
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            # Use the logging window with a progress dialog
            self.LogWindow.append_log(
                f"Starting feature extraction from {len(tasks)} segmentation files...", log_type='process'
            )
            self.LogWindow.setup_progress_dialog(
                window_title="Extracting Features ...",
                label_text="Extracting features ...",
                tasks=tasks,
                processing_func=self.extract_features_for_file_threadsafe
            )
            # Defer result collection until the worker thread finishes
            self.LogWindow.process_finished_callback = self.collect_feature_extraction_results
        else:
            # Fallback: run sequentially in the main thread
            print(f"[INFO] Processing {len(tasks)} segmentation files...")
            for task in tqdm(tasks, desc="Feature Extraction"):
                self.extract_features_for_file_threadsafe(*task)
            # Collect results immediately when done
            self.collect_feature_extraction_results()

    def extract_features_for_file_threadsafe(self, segmentation_idx, segmentation_path, segmentation_name):
        """
        Extract features for a single file (thread-safe version).
        
        Parameters:
        -----------
        segmentation_idx : int
            Index of the segmentation file
        segmentation_path : str
            Path to the segmentation file
        segmentation_name : str
            Name of the segmentation file
        """
        try:
            # Load segmentation
            segmentation_array = self.comet_segmentation_io.load_segmentation(
                segmentation_path=segmentation_path,
                import_format=self.export_format
            )

            # Convert to expected format for feature extraction
            if segmentation_array is not None:
                # Check if we have epoched data with sliding features enabled
                is_epoched_sliding = (self.datatype == 'epoched' and 'sliding' in self.feature_mode)
                
                if is_epoched_sliding and len(segmentation_array.shape) == 2:
                    # For epoched sliding features, preserve trial structure
                    # Use first trial for time array creation, but keep all trials for processing
                    labels = segmentation_array[0, :].tolist()  # For time array creation
                    # Store original segmentation array for feature extraction
                    original_segmentation_array = segmentation_array
                else:
                    # Standard processing - flatten if needed
                    if len(segmentation_array.shape) == 2:
                        labels = segmentation_array[0, :].tolist()  # Use first trial
                    else:
                        labels = segmentation_array.tolist()
                    original_segmentation_array = None
                
                # Create time array (assuming consistent sampling)
                num_samples = len(labels)
                if hasattr(self, 'sample_rate') and self.sample_rate:
                    time_step = 1000 / self.sample_rate  # Convert to ms
                    
                    # For epoched data, create time array centered around TMS (t=0)
                    if self.datatype == 'epoched':
                        # Assuming typical TMS epoch: -1000ms to +1000ms around TMS
                        start_time = -1000  # Start at -1000ms
                        time = [start_time + i * time_step for i in range(num_samples)]
                    else:
                        time = [i * time_step for i in range(num_samples)]
                else:
                    time = list(range(num_samples))  # Default time points
                
                # Load corresponding EEG data
                eeg_name = os.path.splitext(segmentation_name)[0]
                # Find the actual EEG file with the correct extension
                eeg_file = None
                preprocessed_files = os.listdir(self.preprocessed_data_path)
                for file in preprocessed_files:
                    if file.startswith(eeg_name + '.') or file == eeg_name:
                        eeg_file = os.path.join(self.preprocessed_data_path, file)
                        break
                
                if eeg_file is None:
                    raise FileNotFoundError(f"Could not find EEG file for {eeg_name} in {self.preprocessed_data_path}")
                
                # Load the EEG data
                eeg = self.comet_data_io.load_eeg(eeg_file, self.datatype)
                
                # For epoched sliding features, preserve 3D structure (trials, channels, timepoints)
                # Otherwise use standard flattened structure
                if is_epoched_sliding:
                    eeg_data = eeg.get_data()  # Keep original 3D structure for epoched data
                else:
                    eeg_data = self.comet_data_io.get_eeg_data(eeg, self.datatype)  # Standard processing
                
                # Create segmentation dictionary in expected format
                segmentation = {
                    'labels': labels,
                    'time': time,
                    'filename': segmentation_name,
                    'eeg_data': eeg_data,
                    'microstate_maps': self.best_maps,
                    'microstate_labels': self.micro_labels
                }
                
                # Add original segmentation data for epoched sliding processing
                if original_segmentation_array is not None:
                    segmentation['original_segmentation_array'] = original_segmentation_array
            else:
                # Create empty segmentation if loading failed
                segmentation = {
                    'labels': [],
                    'time': [],
                    'filename': segmentation_name,
                    'eeg_data': None,
                    'microstate_maps': None,
                    'microstate_labels': None
                }

            # Log file processing start
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                segments_count = len(segmentation['labels']) if 'labels' in segmentation else 'unknown'
                duration = segmentation['time'][-1] / 1000 if 'time' in segmentation else 'unknown'
                file_info = f"{segmentation_name} | Duration: {duration:.1f}s | Segments: {segments_count}"
                self.LogWindow.append_log(f"Extracting features from {file_info}", log_type='file')

            # Extract features
            extracted_features = self.comet_feature_extractor.extract_features(
                segmentation=segmentation,
                feature_list=self.feature_list,
                feature_mode=self.feature_mode,
                feature_types=self.feature_types,
                sliding_window_size=self.sliding_window_size,
                pre_window_size=self.pre_window_size,
                post_window_size=self.post_window_size
            )

            # Store in shared storage with thread-safe access
            COMET._shared_feature_results[segmentation_idx] = {
                'segmentation_name': segmentation_name,
                'extracted_features': extracted_features
            }

            # Log successful feature extraction
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                feature_count = sum(len(mode_data) for mode_data in extracted_features.values()) if extracted_features else 0
                self.LogWindow.append_log(f"Successfully extracted features from {segmentation_name} (modes: {feature_count})", log_type='success')

        except Exception as e:
            error_msg = f"Error extracting features from {segmentation_name}: {str(e)}"
            print(error_msg)
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type='error')

    def collect_feature_extraction_results(self, message=None):
        """
        Collect feature extraction results from shared storage and export them
        This is called when the worker thread finishes
        """
        # Organize results by mode and type
        organized_results = {}
        
        # Initialize the structure for each mode and type
        for mode in self.feature_mode:
            organized_results[mode] = {}
            for feature_type in self.feature_types:
                organized_results[mode][feature_type] = []
        
        # Collect results from each file
        for file_idx, file_data in COMET._shared_feature_results.items():
            if isinstance(file_data, dict) and 'extracted_features' in file_data:
                extracted_features = file_data['extracted_features']
                
                # Organize by mode and type
                for mode in self.feature_mode:
                    if mode in extracted_features:
                        for feature_type in self.feature_types:
                            if feature_type in extracted_features[mode]:
                                organized_results[mode][feature_type].extend(
                                    extracted_features[mode][feature_type]
                                )
        
        # Combine DataFrames for each mode and type
        for mode in self.feature_mode:
            for feature_type in self.feature_types:
                if organized_results[mode][feature_type]:
                    # Combine all DataFrames for this mode and type
                    combined_df = pd.concat(organized_results[mode][feature_type], ignore_index=True)
                    
                    # Create output directory if it doesn't exist
                    os.makedirs(self.extracted_features_path, exist_ok=True)
                    
                    # Export the combined features
                    try:
                        self.comet_feature_io.export_features(
                            features_df=combined_df,
                            feature_type=feature_type,
                            feature_mode=mode,
                            output_folder=self.extracted_features_path,
                            export_format=self.export_format
                        )
                        
                        # Log successful export
                        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                            self.LogWindow.append_log(f"Exported {mode} features for {feature_type} ({len(combined_df)} records)", log_type='success')
                            
                    except Exception as export_error:
                        error_msg = f"[ERROR] Failed to export {mode} features for {feature_type}: {export_error}"
                        print(error_msg)
                        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                            self.LogWindow.append_log(error_msg, log_type='error')

        # Set feature extraction flag
        self.done_extracting_features = True

        # Always print completion to terminal for important steps
        print("✓ Feature extraction completed successfully")
        
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            # Clear the callback to prevent infinite recursion
            self.LogWindow.process_finished_callback = None
            self.LogWindow.process_finished("✓ All features have been successfully extracted!")

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
        print("\n[INFO] Starting source localization...")

        # Check if backfitting has been done
        if not self.done_backfitting:
            print("[ERROR] Backfitting must be completed before source localization")
            return

        # Determine the subjects directory
        if self.use_anatomy == "individual":
            if hasattr(self, 'individual_subjects_dir'):
                self.anatomy_subjects_dir = self.individual_subjects_dir
            else:
                print("[ERROR] individual_subjects_dir not set for individual anatomy")
                return
        else:  # use_anatomy == "fsaverage"
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
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
            bem_solver=self.bem_solver,
            inverse_method=self.inverse_method,
            spacing=self.spacing,
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
                f"* Boundary Element Method: {self.bem_solver}\n"
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
            # Set callback to run when source localization worker thread finishes
            self.LogWindow.process_finished_callback = self._on_source_localization_finished
        else:
            print(f"[INFO] Processing {len(self.zipped_eeg_files)} EEG files...")
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Source Localization"):
                self.source_localize_file(eeg_path, eeg_name)
            # For non-GUI mode, call completion directly
            self._on_source_localization_finished()

    def run_identifying_microstate_sources(self):
        """
        Correlate sources and microstates
        """
        print("\n[INFO] Starting source-microstate correlation...")

        # Check if source localization has been done
        if not self.done_source_localization:
            print("[ERROR] Source localization must be completed before correlation")
            return

        # Check if anatomy subjects directory is available
        try:
            self.anatomy_subjects_dir
        except AttributeError:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
            self.anatomy_subjects_dir = os.path.dirname(fs_dir)

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
                bem_solver=self.bem_solver,
                inverse_method=self.inverse_method,
                spacing=self.spacing,
                microstate_maps=self.best_maps,
                nperm=self.nperm
            )

        # Ensure directories are created and paths are set
        if self.source_localization_method == 'tess':
            self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
            os.makedirs(self.tess_path, exist_ok=True)
            self.comet_source_localizer.tess_path = self.tess_path
        elif self.source_localization_method == 'avg':
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
                f"* Method: {self.source_localization_method}\n"
                f"* Number of permutations: {self.nperm}", log_type='settings'
            )

        # Perform source identification on all files
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.setup_progress_dialog(
                window_title="Source Identification ...",
                label_text=f"Identifying microstate sources using {self.source_localization_method} method ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.source_identify_file
            )
            # Set callback to run when source identification worker thread finishes
            self.LogWindow.process_finished_callback = self._on_source_identification_finished
        else:
            print(f"[INFO] Processing {len(self.zipped_eeg_files)} EEG files...")
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Source Identification"):
                self.source_identify_file(eeg_path, eeg_name)
            # For non-GUI mode, call completion directly
            self._on_source_identification_finished()

    def save_config(self):
        """
        Update and save the current configuration settings and program state to save_dir.
        This version preserves existing optimization results unless explicitly overwritten.
        """
        # Update the config dictionary with current attributes
        self.config["io_config"]["study_name"] = self.study_name
        self.config["io_config"]["input_folder"] = self.input_folder
        self.config["io_config"]["montage"] = str(self.montage) if self.montage is not None else ""
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

        self.config["source_config"]["bem_solver"] = self.bem_solver
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
        self.config["state_flags"]["done_identifying_microstate_sources"] = str(
            self.done_identifying_microstate_sources)

        # Note: optimization_results section is preserved if it exists
        # It should only be modified through save_optimization_results() method

        # Add timestamp for when config was last saved
        if "metadata" not in self.config:
            self.config.add_section("metadata")

        import time
        self.config["metadata"]["last_saved"] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        try:
            # Write config to file in save_dir
            with open(self.config_path, 'w+', encoding='utf-8') as configfile:
                self.config.write(configfile)
        except Exception as e:
            print(f"[WARNING] Failed to save configuration: {e}")

    def _save_logs(self):
        """Save current log content to the log_text attribute and to the log file."""
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.log_text = self.LogWindow.get_log_content()
        
        # Save logs to the separate log file
        self.save_logs_to_file()

    def save_logs_to_file(self):
        """Save current log content to the separate log file."""
        if hasattr(self, 'log_file_path') and self.log_text:
            try:
                with open(self.log_file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text)
            except Exception as e:
                print(f"[WARNING] Failed to save logs: {e}")

    def load_logs_from_file(self):
        """Load log content from the separate log file."""
        if hasattr(self, 'log_file_path') and os.path.exists(self.log_file_path):
            try:
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"[WARNING] Failed to load logs: {e}")
                return ""
        return ""

    def load_optimization_results(self):
        """Load optimization results from configuration if available"""
        if hasattr(self, 'optimization_results') and self.optimization_results:
            return self.optimization_results
        return None

    def save_optimization_results(self, results):
        """Save optimization results to configuration"""
        self.optimization_results = results

    def _on_clustering_iterations_finished(self, *args, **kwargs):
        """Finalize clustering once the worker thread has completed all iterations."""
        # Verify that clustering produced maps
        if self.best_maps is None:
            error_msg = "Clustering failed - no microstate maps were generated"
            if hasattr(self, 'LogWindow') and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type='error')
            else:
                print(error_msg)
            return

        # Compute final GEV across all data
        self.best_gev = self.compute_gev_all_data()

        # Mark clustering as completed
        self.done_clustering = True

        # Compose completion message
        completion_message = f"✓ Clustering completed successfully: {self.number_of_maps} microstates (GEV: {100 * self.best_gev:.3f}%)"

        # Always print completion to terminal for important steps
        print(completion_message)
        
        # Log or print completion
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            self.LogWindow.append_log(completion_message, log_type='success')
            # Clear the callback to prevent it from being triggered by other processes
            self.LogWindow.process_finished_callback = None

        # Persist configuration and logs
        if self.auto_save:
            self.save_config()
        self._save_logs()

        # Notify any registered callbacks (e.g., GUI updates)
        if self.clustering_completed_callback is not None:
            self.clustering_completed_callback()

    def _on_preprocessing_finished(self, message=None):
        """Handle preprocessing completion when worker thread finishes."""
        # Set preprocessing flag
        self.save_eeg_info(self.eeg_info_path)
        self.done_preprocessing = True

        # Always print completion to terminal for important steps
        print("✓ Preprocessing completed successfully")
        
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            # Clear the callback to prevent it from being triggered by other processes
            self.LogWindow.process_finished_callback = None
            self.LogWindow.append_log(f"Preprocessing completed successfully! Data saved to: {self.preprocessed_data_path}", log_type='success')

        # Save configuration and parameters
        if self.auto_save:
            self.save_config()
            
        # Ensure logs are saved
        self._save_logs()

    def _on_backfitting_finished(self, message=None):
        """Handle backfitting completion when worker thread finishes."""
        # Set backfitting flag
        self.done_backfitting = True

        # Always print completion to terminal for important steps
        print("✓ Backfitting completed successfully")
        
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            # Clear the callback to prevent it from being triggered by other processes
            self.LogWindow.process_finished_callback = None
            self.LogWindow.process_finished("✓ Microstates have been successfully backfitted to the data!")

        # Save parameters
        if self.auto_save:
            self.save_config()
            
        # Ensure logs are saved
        self._save_logs()

    def _on_source_localization_finished(self, message=None):
        """Handle source localization completion when worker thread finishes."""
        # Set source localization flag
        self.done_source_localization = True

        # Always print completion to terminal for important steps
        print("✓ Source localization completed successfully")
        
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            # Clear the callback to prevent it from being triggered by other processes
            self.LogWindow.process_finished_callback = None
            self.LogWindow.process_finished("✓ Source localization completed")

        # Save parameters
        if self.auto_save:
            self.save_config()
            
        # Ensure logs are saved
        self._save_logs()

    def _on_source_identification_finished(self, message=None):
        """Handle source identification completion when worker thread finishes."""
        # Set source-microstate correlation flag
        self.done_identifying_microstate_sources = True

        # Always print completion to terminal for important steps
        print("✓ Source-microstate correlation completed successfully")
        
        if hasattr(self, 'LogWindow') and self.LogWindow is not None:
            # Clear the callback to prevent it from being triggered by other processes
            self.LogWindow.process_finished_callback = None
            self.LogWindow.process_finished(f"✓ Source-microstate correlation completed using {self.source_localization_method} method")

        # Save parameters
        if self.auto_save:
            self.save_config()
            
        # Ensure logs are saved
        self._save_logs()
