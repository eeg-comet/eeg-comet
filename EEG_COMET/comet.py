"""Core COMET workflow for EEG-COMET.

Provides the `COMET` class implementing preprocessing, clustering, labeling,
backfitting, feature extraction, source localization, and correlation routines.
"""

import contextlib
import glob
import os
from configparser import ConfigParser

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from backfitting_utils.microstate_backfitter import MicrostateBackfitter
from backfitting_utils.segmentation_io import SegmentationIO
from clustering_utils.clusterer_optimizer import ClustererOptimizer
from clustering_utils.microstate_clusterer import MicrostateClusterer
from clustering_utils.microstate_io import MicrostateIO
from clustering_utils.microstate_labeler import MicrostateLabeler
from controllers.logging_window import LogWindow
from data_utils.data_initializer import DataInitializer
from data_utils.data_io import DataIO
from data_utils.data_preprocessor import DataPreprocessor
from features_utils.feature_extractor import FeatureExtractionCoordinator, FeatureExtractor
from features_utils.feature_helper import FeatureHelper
from features_utils.feature_io import FeatureIO
from gui_utils.terminal_logger import get_logger
from sourcelocalization_utils.source_localizer import SourceLocalizer


class COMET:
    """EEG-COMET core orchestrator for preprocessing, clustering, and analysis."""
    @property
    def output_folder(self):
        """Output folder root for this study."""
        return getattr(self, "_output_folder", None)

    @output_folder.setter
    def output_folder(self, folder_path):
        self._output_folder = folder_path
        # When output folder changes, rebuild directories
        if hasattr(self, "reset_directories"):
            self.reset_directories()

    @property
    def input_folder(self):
        """Input folder containing raw EEG files."""
        return getattr(self, "_input_folder", None)

    @input_folder.setter
    def input_folder(self, folder_path):
        self._input_folder = folder_path

    """The COMET class represents an instance of the EEG-COMET application.
    It provides methods to load configuration settings, perform various processes
    such as preprocessing, clustering, labeling, backfitting, feature extraction,
    source localization, and source-microstate correlation.
    """

    # Class-level shared storage for feature extraction results
    _shared_feature_results = {}

    def __init__(
        self,
        config=None,
        config_path=None,
        study_name=None,
        input_folder=None,
        output_folder=None,
        auto_save=True,
        context=None,
    ):
        """Initialize COMET instance with optimized memory management."""
        # Initialize logger
        self.logger = get_logger()

        # Initialize callbacks for process completion
        self.clustering_completed_callback = None
        self.backfitting_completed_callback = None
        self.feature_extraction_completed_callback = None
        self.source_localization_completed_callback = None
        self.source_microstate_correlation_completed_callback = None
        # Callback that gets invoked when the full preprocessing pipeline has finished
        # (e.g. to let the GUI update its widget state automatically once data are ready).
        self.preprocessing_completed_callback = None

        # Initialize basic attributes
        self.auto_save = auto_save
        self.config = None
        self.config_path = config_path
        self.log_text = ""  # Initialize as string instead of list

        # Define all instance variables
        self.LogWindow = None
        self.study_name = study_name
        self._input_folder = input_folder
        self._output_folder = output_folder
        self.context = context  # Store context reference

        # Predefine paths to avoid "defined outside __init__" warnings
        self.save_dir = None
        self.log_file_path = None
        self.preprocessed_data_path = None
        self.eeg_info_path = None
        self.microstate_maps_path = None
        self.extracted_features_path = None
        self.segmentation_path = None
        self.localized_sources_path = None
        self.tess_path = None
        self.avg_sources_path = None

        # Predefine configuration-driven attributes
        # IO
        self.montage = ""
        self.extension = ".auto"
        self.pattern_content = "*"
        self.datatype = "raw"

        # Preprocessing
        self.temporal_filter_data = True
        self.filter_method = "fir"
        self.lowcut_freq = 2
        self.highcut_freq = 20
        self.downsample_data = True
        self.sample_rate = 250
        self.spatial_filter_data = False
        self.auto_clean_data = False
        self.remove_channels = False
        self.chan2rm = ""
        self.prep_data = False

        # Clustering
        self.smoothing_gfp = False
        self.smoothing_distance = 10
        self.number_of_maps = 4
        self.choose_number_of_maps = "User"
        self.kmin = 2
        self.kmax = 10
        self.stopping_mode = "gev"
        self.stopping_parameter = 10
        self.initializer = "Random"
        self.clustering_method = "Modified K-Means Clustering (Pascual-Marqui et al. 1995)"
        self.max_iterations = 500
        self.clustering_tolerance = 1e-6
        self.similarity_metric = ""
        self.number_of_repeats = 5
        self.use_percentages = 100

        # Backfitting
        self.backfit_to = "all"
        self.identify_short_window = False
        self.filter_segments = False
        self.filter_segments_less_than = 20
        self.filter_segments_option = "smooth"
        self.epsilon = 1e-6
        self.b = 3
        self.lamb = 5
        self.filter_segments_less_than_ms = 0

        # Features
        self.export_format = ".csv"
        self.feature_list = ["OCC", "DUR", "COV"]
        self.feature_mode = ["averaged"]
        self.feature_types = ["real"]
        self.sliding_window_size = 1
        self.event_based_sliding = False
        self.selected_events = []
        self.event_matching_mode = "partial"  # "exact", "case_insensitive", or "partial"
        self.pre_window_size = 1
        self.post_window_size = 1
        self.feature_list_dictionary = {}

        # Source Localization
        self.use_anatomy = "fsaverage"
        self.bem_solver = "mne"
        self.inverse_method = "dSPM"
        self.nperm = 2000
        self.spacing = "ico3"
        self.source_localization_method = "tess"
        self.anatomy_subjects_dir = ""

        # Runtime/derived attributes used across methods
        self.config_last_saved = "Unknown"
        self.events_per_file = {}
        self.common_events = []
        self.labels_overall_confidence = None
        self.maps2use = None
        self.min_distance_size = None
        self.clustering_results = []
        self.optimization_results = None
        self.zipped_eeg_files = []
        self.segmentation_list_path = []
        self.segmentation_list = []
        self.comet_microstate_backfitter = None
        self.comet_clusterer_optimizer = None
        self.comet_source_localizer = None
        self.comet_microstate_labeler = None

        # Additional attributes to satisfy static analyzers
        self.load_all_files = False
        self.pattern = "*"
        self.list_eegs_path = []
        self.list_eegs = []
        self.micro_labels = []
        self.ch_names = []
        self.eeg_info = None

        # Initialize utility objects
        self.comet_data_io = DataIO()
        self.comet_preprocessor = DataPreprocessor()
        self.comet_data_initializer = DataInitializer()
        self.comet_microstate_io = MicrostateIO()
        self.comet_segmentation_io = SegmentationIO()
        self.comet_feature_io = FeatureIO()
        self.comet_feature_helper = FeatureHelper()
        self.comet_feature_extractor = FeatureExtractionCoordinator()
        self.comet_microstate_clusterer = None  # Will be initialized during clustering

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

        # Initialize clustering-related attributes
        self.batch_size = None
        self.best_maps = None
        self.best_gev = 0.0
        self.best_residual = np.inf

        # Properties are defined at the class level (see below)

    @staticmethod
    def create_default_config():
        """Create a default configuration without loading from a file."""
        config = ConfigParser()

        # Create default sections
        config.add_section("io_config")
        config.add_section("preprocessing_config")
        config.add_section("clustering_config")
        config.add_section("backfitting_config")
        config.add_section("features_config")
        config.add_section("source_config")

        # Set default values for IO section
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
        config["clustering_config"][
            "clustering_method"
        ] = "Modified K-Means Clustering (Pascual-Marqui et al. 1995)"
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
        """Reset and recreate directory structure when critical parameters change.
        Call this whenever study_name or output_folder are changed.
        """
        # Set up the save_dir based on output_folder and study_name
        if self.output_folder and self.study_name:
            self.save_dir = os.path.join(self.output_folder, self.study_name)
            # Update all paths relative to save_dir
            self.config_path = os.path.join(self.save_dir, "eeg_comet_config.ini")
            self.log_file_path = os.path.join(self.save_dir, "eeg_comet_log.txt")
            self.preprocessed_data_path = os.path.join(
                self.save_dir, f"{self.study_name}_preprocessed_data"
            )
            self.eeg_info_path = os.path.join(self.save_dir, "eeg_info.fif")
            self.microstate_maps_path = os.path.join(self.save_dir, "microstate_maps.csv")
            self.extracted_features_path = os.path.join(
                self.save_dir, f"{self.study_name}_extracted_features"
            )
            self.segmentation_path = os.path.join(self.save_dir, f"{self.study_name}_segmentation")
            self.localized_sources_path = os.path.join(
                self.save_dir, f"{self.study_name}_localized_sources"
            )
            self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
            self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")
        else:
            # Clear paths when essential parameters are missing
            self.save_dir = None
            self.config_path = None
            self.log_file_path = None
            self.preprocessed_data_path = None
            self.eeg_info_path = None
            self.microstate_maps_path = None
            self.extracted_features_path = None
            self.segmentation_path = None
            self.localized_sources_path = None
            self.tess_path = None
            self.avg_sources_path = None

    def load_config(self, config_path):
        """Load configuration from a file path."""
        if not os.path.exists(config_path):
            self.logger.warning(
                "CONFIGURATION",
                f"Config file {config_path} not found. Using default configuration.",
            )
            return self.create_default_config()

        config = ConfigParser()
        try:
            # Use UTF-8 encoding to properly handle Unicode characters
            config.read(config_path, encoding="utf-8")
            return config
        except Exception as e:
            self.logger.warning("CONFIGURATION", f"Failed to load configuration: {e}")
            return self.create_default_config()

    def load_config_values(self):
        """Load configuration values from self.config into instance variables."""
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
            self.kmin = clustering_config.getint("kmin", 2)
            self.kmax = clustering_config.getint("kmax", 10)
            self.stopping_mode = clustering_config.get("stopping_mode", "majority_vote")
            self.stopping_parameter = clustering_config.getint("stopping_parameter", 10)
        else:
            self.kmin = 2
            self.kmax = 10
            self.stopping_mode = "majority_vote"
            self.stopping_parameter = 10

        self.initializer = clustering_config.get("initializer", "Random")
        self.clustering_method = clustering_config.get(
            "clustering_method", "Modified K-Means Clustering (Pascual-Marqui et al. 1995)"
        )
        self.max_iterations = clustering_config.getint("max_iterations", 500)
        self.clustering_tolerance = clustering_config.getfloat("clustering_tolerance", 1e-6)
        self.similarity_metric = (
            clustering_config.get("similarity_metric", "Spatial Correlation")
            if self.clustering_method != "Modified K-Means Clustering (Pascual-Marqui et al. 1995)"
            else ""
        )
        self.number_of_repeats = clustering_config.getint("number_of_repeats", 5)

        # Handle use_percentages
        try:
            self.use_percentages = clustering_config.getint("use_percentages", 100)
        except (ValueError, KeyError):
            self.use_percentages = 100

        # Ensure use_percentages is never None
        if self.use_percentages is None:
            self.use_percentages = 100

        # --- NEW: Load common events if present ---
        if "events_config" in self.config:
            common_events_str = self.config["events_config"].get("common_events", "")
            self.common_events = [e.strip() for e in common_events_str.split(",") if e.strip()]
        else:
            self.common_events = []
        # --- END NEW ---

        # Backfitting Configs
        backfitting_config = self.config["backfitting_config"]
        self.backfit_to = backfitting_config.get("backfit_to", "all")
        self.identify_short_window = backfitting_config.getboolean("identify_short_window", False)
        self.filter_segments = backfitting_config.getboolean("filter_segments", False)

        if self.filter_segments:
            self.filter_segments_less_than = backfitting_config.getint(
                "filter_segments_less_than", 20
            )
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
        
        # Load event-based sliding settings
        self.event_based_sliding = features_config.getboolean("event_based_sliding", False)
        selected_events_str = features_config.get("selected_events", "")
        self.selected_events = [e.strip() for e in selected_events_str.split(",") if e.strip()]
        self.event_matching_mode = features_config.get("event_matching_mode", "partial")

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
            "OCC": "Frequency of Occurrence (Hz)",
            "DUR": "Mean Microstate Duration (ms)",
            "COV": "Microstate Coverage (%)",
            "GEV": "Microstate Global Explained Variance (%)",
            "TP": "Transition Probability",
            "ER": "Entropy Rate",
            "LZC": "Sequence Lempel-Ziv Complexity",
            "HE": "Hurst Exponent",
            "ERR": "Sequence Entropy Representation",
            "ROF": "Relative Occurrence Frequency",
            "RTF": "Relative Transition Frequency",
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
        if "state_flags" in self.config:
            state_flags = self.config["state_flags"]
            self.done_preprocessing = state_flags.getboolean("done_preprocessing", False)
            self.done_clustering = state_flags.getboolean("done_clustering", False)
            self.done_microstate_labeling = state_flags.getboolean(
                "done_microstate_labeling", False
            )
            self.done_backfitting = state_flags.getboolean("done_backfitting", False)
            self.done_extracting_features = state_flags.getboolean(
                "done_extracting_features", False
            )
            self.done_source_localization = state_flags.getboolean(
                "done_source_localization", False
            )
            self.done_identifying_microstate_sources = state_flags.getboolean(
                "done_identifying_microstate_sources", False
            )
        else:
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

        # Initialize log_text
        self.log_text = ""

    def ensure_directory(self, path):
        """Ensure a directory exists and is writable.

        Args:
            path (str): Directory path to check/create.

        Returns:
            bool: True if directory exists and is writable, False otherwise.
        """
        if not path:
            return False

        try:
            os.makedirs(path, exist_ok=True)
            # Test if we can write to this directory
            test_file = os.path.join(path, ".test_write_access")
            with open(test_file, "w", encoding="utf-8") as temp_handle:
                temp_handle.write("test")
            os.remove(test_file)
            return True
        except (PermissionError, OSError) as exc:
            self.logger.warning("DATA_IO", f"Cannot access directory {path}: {exc}")
            return False

    def setup_directories(self):
        """Set up all required directories based on user configuration."""
        # Check if output_folder and study_name are set before proceeding
        if not self.output_folder:
            self.output_folder = os.path.expanduser("~/EEG-COMET_Data")

        if not self.study_name:
            self.study_name = "my_study"
            self.logger.warning(
                "CONFIGURATION", f"No study name specified, using: {self.study_name}"
            )

        # Create output folder if it doesn't exist
        if not self.ensure_directory(self.output_folder):
            old_output = self.output_folder
            self.output_folder = os.path.expanduser("~/EEG-COMET_Data")
            self.logger.warning(
                "DATA_IO", f"Cannot access {old_output}, falling back to: {self.output_folder}"
            )
            self.ensure_directory(self.output_folder)

        # Setup study directory
        self.save_dir = os.path.join(self.output_folder, self.study_name)
        self.ensure_directory(self.save_dir)

        # Define all paths relative to save_dir
        self.config_path = os.path.join(self.save_dir, "eeg_comet_config.ini")
        self.log_file_path = os.path.join(self.save_dir, "eeg_comet_log.txt")
        self.preprocessed_data_path = os.path.join(
            self.save_dir, f"{self.study_name}_preprocessed_data"
        )
        self.eeg_info_path = os.path.join(self.save_dir, "eeg_info.fif")
        self.microstate_maps_path = os.path.join(self.save_dir, "microstate_maps.csv")
        self.extracted_features_path = os.path.join(
            self.save_dir, f"{self.study_name}_extracted_features"
        )
        self.segmentation_path = os.path.join(self.save_dir, f"{self.study_name}_segmentation")
        self.localized_sources_path = os.path.join(
            self.save_dir, f"{self.study_name}_localized_sources"
        )
        self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
        self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")

    def initialize_log_window(self):
        """Initialize the logging window."""
        self.LogWindow = LogWindow(comet_instance=self)

        # Set the log window for the logger
        self.logger.log_window = self.LogWindow

        # Add default session messages (will be replaced if logs are restored)
        self.LogWindow.append_log("EEG-COMET Session Started", log_type="section")
        self.LogWindow.append_log(
            "Welcome to EEG-COMET (EEG Comprehensive Microstate Extraction Toolbox)",
            log_type="info",
        )
        self.LogWindow.append_log("Ready to process EEG microstate data.", log_type="info")

    def restore_logs_if_available(self):
        """Restore saved logs if available, called after config is loaded."""
        # Load logs from file if not already loaded
        if not hasattr(self, "log_text") or not self.log_text:
            self.log_text = self.load_logs_from_file()

        if (
            hasattr(self, "log_text")
            and self.log_text
            and hasattr(self, "LogWindow")
            and self.LogWindow
        ):
            self.LogWindow.set_log_content(self.log_text)
            return True
        return False

    def load_raw(self):
        """Locate EEG file paths."""
        # Check if load_all_files attribute exists, set it to True if not
        if not hasattr(self, "load_all_files"):
            self.load_all_files = True

        self.pattern = "*" if self.load_all_files else "*" + self.pattern_content + "*"
        self.list_eegs_path, self.list_eegs = self.comet_data_io.find_data(
            input_folder=self.input_folder, extension=self.extension, pattern=self.pattern
        )

    def load_clean(self):
        """Locate EEG file paths that were automatically cleaned."""
        self.list_eegs_path, self.list_eegs = self.comet_data_io.find_data(
            input_folder=self.preprocessed_data_path, extension=self.extension, pattern="*"
        )
        self.load_maps()

    def load_maps(self):
        """Load microstate maps from the specified path."""
        if not os.path.isfile(self.microstate_maps_path):
            raise FileNotFoundError(
                f"Microstate maps file not found: {self.microstate_maps_path}"
            )
        try:
            self.best_maps, self.micro_labels = self.comet_microstate_io.load_microstates(
                self.microstate_maps_path
            )
            return True
        except Exception as err:
            raise ValueError(f"Error loading microstate maps: {err}") from err

    def check_chan2rm(self):
        """Check the consistency of EEG channels across all data and prepare for interpolation."""
        consistent_channels, missing_channels_per_file = self.comet_data_io.check_channel_consistency_per_file(
            self.list_eegs_path, datatype=self.datatype, montage=self.montage
        )
        
        # Store missing channels per file for later interpolation
        self.missing_channels_per_file = missing_channels_per_file
        
        # Keep track of channels that should be removed (user-specified)
        if isinstance(self.chan2rm, str):
            # Convert string to list if it contains comma-separated values
            if self.chan2rm.strip():
                self.chan2rm = [ch.strip() for ch in self.chan2rm.split(",") if ch.strip()]
            else:
                self.chan2rm = []
        elif not isinstance(self.chan2rm, list):
            self.chan2rm = []
        
        # Store consistent channels (channels present in all files)
        self.ch_names = consistent_channels

    def get_missing_channels_stats(self):
        """Get statistics about missing channels across all files."""
        if not hasattr(self, 'missing_channels_per_file'):
            return 0, 0, []
        
        total_missing = 0
        files_with_missing = 0
        all_missing_channels = set()
        
        for file_path, missing_channels in self.missing_channels_per_file.items():
            if missing_channels:
                files_with_missing += 1
                total_missing += len(missing_channels)
                all_missing_channels.update(missing_channels)
        
        return total_missing, files_with_missing, sorted(all_missing_channels)

    def interpolate_missing_channels_for_file(self, eeg, eeg_path):
        """Interpolate missing channels for a specific file using the DataPreprocessor."""
        if not hasattr(self, 'missing_channels_per_file') or eeg_path not in self.missing_channels_per_file:
            return eeg
        
        missing_channels = self.missing_channels_per_file[eeg_path]
        if not missing_channels:
            return eeg
        
        # Use the DataPreprocessor's interpolation method with user's selected montage
        try:
            # Determine the montage to use for interpolation
            if hasattr(self, "montage") and self.montage:
                # User has selected a specific montage
                montage_obj = self.comet_data_io.load_montage(self.montage)
                interpolated_eeg = self.comet_preprocessor.interpolate_missing_channels(
                    raw=eeg,
                    missing_channels=missing_channels,
                    custom_montage=montage_obj,  # Use user's selected montage
                    verbose=False
                )
            else:
                # Fallback to standard montage if no user selection
                interpolated_eeg = self.comet_preprocessor.interpolate_missing_channels(
                    raw=eeg,
                    missing_channels=missing_channels,
                    montage_name='standard_1020',
                    verbose=False
                )
            
            # Log the interpolation
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                montage_name = self.montage if hasattr(self, "montage") and self.montage else "standard_1020"
                self.LogWindow.append_log(
                    f"Interpolated {len(missing_channels)} missing channels using montage '{montage_name}': {', '.join(missing_channels)}",
                    log_type="info"
                )
            
            return interpolated_eeg
        except Exception as e:
            # Log the error but continue with original data
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(
                    f"Warning: Failed to interpolate missing channels: {e}",
                    log_type="warning"
                )
            return eeg

    def validate_channel_count_consistency(self):
        """Validate that the channel count in EEG info matches the actual data after preprocessing."""
        try:
            # Load the first preprocessed file to get the actual channel count
            data_io = DataIO()
            all_preprocessed_paths, _ = data_io.find_data(self.preprocessed_data_path, self.extension)
            
            if all_preprocessed_paths:
                # Load the first file to get the actual channel count after preprocessing
                first_eeg = data_io.load_eeg(all_preprocessed_paths[0], self.datatype)
                actual_ch_count = len(first_eeg.ch_names)
                stored_ch_count = len(self.ch_names)
                
                if actual_ch_count != stored_ch_count:
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            f"Channel count mismatch detected: stored={stored_ch_count}, actual={actual_ch_count}. Updating...",
                            log_type="warning"
                        )
                    # Update the channel names to match the actual data
                    self.ch_names = first_eeg.ch_names
                    return True
                    
        except Exception as e:
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(
                    f"Warning: Failed to validate channel count consistency: {e}",
                    log_type="warning"
                )
        return False

    def update_channel_names_after_preprocessing(self):
        """Update channel names to reflect the actual channels after preprocessing and interpolation."""
        try:
            # Load the first preprocessed file to get the actual channel names
            data_io = DataIO()
            all_preprocessed_paths, _ = data_io.find_data(self.preprocessed_data_path, self.extension)
            
            if all_preprocessed_paths:
                # Load the first file to get the actual channel names after preprocessing
                first_eeg = data_io.load_eeg(all_preprocessed_paths[0], self.datatype)
                actual_ch_names = first_eeg.ch_names
                
                # Update the channel names to reflect the actual channels after interpolation
                self.ch_names = actual_ch_names
                
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(
                        f"Updated channel names after preprocessing: {len(self.ch_names)} channels",
                        log_type="info"
                    )
                    
        except Exception as e:
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(
                    f"Warning: Failed to update channel names after preprocessing: {e}",
                    log_type="warning"
                )

    def save_eeg_info(self, eeg_info_path):
        """Save EEG information to a binary file using pickle.

        Args:
            eeg_info_path (str): Path where the EEG information will be saved
        """
        # Validate and update channel names to reflect the actual channels after preprocessing
        self.validate_channel_count_consistency()
        
        # Create a basic Info object
        eeg_info = mne.create_info(
            ch_names=self.ch_names, ch_types=["eeg"] * len(self.ch_names), sfreq=self.sample_rate
        )
        eeg_info["description"] = self.study_name

        # Set montage
        montage = self.comet_data_io.load_montage(self.montage)
        eeg_info.set_montage(montage, match_case=False, on_missing="warn")

        mne.io.write_info(eeg_info_path, eeg_info)

    def load_eeg_info(self):
        """Load EEG info from file instead of keeping it in memory."""
        self.eeg_info = mne.io.read_info(self.eeg_info_path)

    def preprocess_eeg(self, eeg_path, eeg_name, worker=None):
        """Preprocess a single EEG file. If the optional worker argument is supplied and its
        stopped flag is set, the function returns immediately so that the thread can
        terminate quickly when the user presses the STOP button.
        """
        # Early-exit if the user requested cancellation
        if worker is not None and getattr(worker, "stopped", False):
            return

        # Load EEG data
        eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)

        # Log file processing start
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            file_info = f"{eeg_name} | Channels: {len(eeg.ch_names)} | Duration: {eeg.times[-1]:.1f}s | Sampling Rate: {eeg.info['sfreq']}Hz"
            self.LogWindow.append_log(f"Processing {file_info}", log_type="file")

        # Apply montage if specified
        if hasattr(self, "montage") and self.montage:
            montage_obj = self.comet_data_io.load_montage(self.montage)
            eeg.set_montage(montage_obj, match_case=False, on_missing="warn")

        # Interpolate missing channels if needed
        eeg = self.interpolate_missing_channels_for_file(eeg, eeg_path)
        
        # Remove user-specified channels if any
        if hasattr(self, "chan2rm") and self.chan2rm:
            if isinstance(self.chan2rm, list):
                channels_to_remove = [ch.strip() for ch in self.chan2rm if ch.strip()]
            else:
                channels_to_remove = []
            
            if channels_to_remove:
                eeg.drop_channels(channels_to_remove, on_missing="ignore")

        # --- NEW: Extract event information and store for later use ---
        try:
            events, event_id = mne.events_from_annotations(eeg)
        except Exception:
            events, event_id = None, {}
        if events is not None and len(events) > 0:
            # Build a dictionary {event_name: [onset_times_in_sec, ...]}
            event_info = {}
            sfreq = eeg.info["sfreq"]
            for desc, code in event_id.items():
                onsets = (events[events[:, 2] == code][:, 0] / sfreq).tolist()
                # round to milliseconds precision to keep file size small
                onsets = [round(t, 3) for t in onsets]
                event_info[desc] = onsets
            # Initialize container on first use
            if not hasattr(self, "events_per_file"):
                self.events_per_file = {}
            self.events_per_file[eeg_name] = event_info
            
            # Log extracted events for debugging
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                event_names = sorted(event_info.keys())
                self.LogWindow.append_log(
                    f"Extracted events from {eeg_name}: {', '.join(event_names)}",
                    log_type="info"
                )
        else:
            # No events found - also log this
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(
                    f"No events found in {eeg_name} annotations",
                    log_type="warning"
                )
        # --- END NEW ---

        # Apply automatic bad channel detection and interpolation if requested
        if hasattr(self, "prep_data") and self.prep_data:
            eeg = self.comet_preprocessor.identify_bad_channels(eeg)
            eeg.interpolate_bads(reset_bads=False)

        # Apply preprocessing steps (event selection happens inside preprocessor after processing)
        preprocessed_eeg = self.comet_preprocessor.preprocess_eeg(
            eeg=eeg,
            filter_bool=self.temporal_filter_data,
            filtermethod=self.filter_method,
            lowcut=self.lowcut_freq,
            highcut=self.highcut_freq,
            downsample_bool=self.downsample_data,
            sampling_rate=self.sample_rate,
            spatial_smooth_bool=self.spatial_filter_data,
            select_events_only=getattr(self, "select_events_only", False),
            selected_event_label=getattr(self, "selected_event_label", None),
            datatype=self.datatype,
        )

        # Save preprocessed data
        name = os.path.splitext(eeg_name)[0]
        save_path = os.path.join(self.preprocessed_data_path, name)
        self.comet_data_io.export_eegs(
            eeg=preprocessed_eeg,
            save_path=save_path,
            extension=self.extension,
            datatype=self.datatype,
        )

        # Log successful processing
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.append_log(f"Successfully preprocessed {eeg_name}", log_type="success")

    def get_preprocessed_eeg(self, subject_name):
        """Load a preprocessed EEG file on demand instead of keeping it in memory."""
        # Handle .auto extension by looking for actual saved file
        if self.extension == ".auto":
            # Try common extensions in order of preference
            possible_extensions = [".set", ".vhdr", ".edf"]
            for ext in possible_extensions:
                eeg_path = os.path.join(self.preprocessed_data_path, f"{subject_name}{ext}")
                if os.path.exists(eeg_path):
                    return self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)
            
            # If no file found with standard extensions, try to find any matching file
            pattern = f"{subject_name}.*"
            files = glob.glob(os.path.join(self.preprocessed_data_path, pattern))
            if files:
                # Use the first matching file
                return self.comet_data_io.load_eeg(eeg_path=files[0], datatype=self.datatype)
            else:
                raise FileNotFoundError(
                    f"No preprocessed EEG file found for {subject_name} in {self.preprocessed_data_path}"
                )
        else:
            # Use the specified extension
            eeg_path = os.path.join(self.preprocessed_data_path, f"{subject_name}{self.extension}")
            return self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)

    def compute_gev_all_data(self):
        """Compute Global Explained Variance for all data."""
        all_data, _ = self.comet_data_initializer.generate_maps_and_peaks(
            preprocessed_folder=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            use_percentages=100,
        )
        return self.comet_microstate_clusterer.compute_gev(data=all_data, maps=self.best_maps)

    def cluster_eeg_microstates(self, init, worker=None):
        """Perform modified K-means clustering iteration with stop check capability."""
        # Check if we should stop before starting
        if worker and hasattr(worker, "stopped") and worker.stopped:
            self._handle_clustering_stop()
            return None, 0.0, np.inf

        is_taahc = (
            self.clustering_method == "Topographic Atomize and Agglomerate Hierarchical Clustering"
        )

        # Initialize cluster centers (not needed for TAAHC but kept for consistency)
        initial_maps = None
        if not is_taahc:
            initial_maps = self.comet_data_initializer.initialize_cluster_centers(
                maps2use=self.maps2use, n_states=self.number_of_maps, initializer=self.initializer
            )
        else:
            # TAAHC is deterministic and time-consuming - MUST use only 1 repetition
            original_repeats = self.number_of_repeats
            self.number_of_repeats = 1
            if original_repeats != 1:
                self.logger.warning(
                    "CLUSTERING",
                    f"TAAHC: Enforcing 1 repetition (deterministic algorithm) - ignoring user setting of {original_repeats}",
                )
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(
                        "TAAHC: Enforcing 1 repetition (deterministic algorithm)", log_type="info"
                    )

        # Perform clustering based on selected method
        try:
            if self.clustering_method == "Modified K-Means Clustering (Pascual-Marqui et al. 1995)":
                maps_init, residual_init = self.comet_microstate_clusterer.modified_kmeans(
                    data=self.maps2use,
                    initial_maps=initial_maps,
                    worker=worker,  # Pass worker to check stopped flag
                    repetition_num=f"{init + 1}/{self.number_of_repeats}",  # Pass repetition number
                )
            elif self.clustering_method == "Modified K-Means Clustering with Spatial Similarity":
                maps_init, residual_init = (
                    self.comet_microstate_clusterer.modified_kmeans_similarity(
                        data=self.maps2use,
                        initial_maps=initial_maps,
                        metric=self.similarity_metric,
                        worker=worker,  # Pass worker to check stopped flag
                        repetition_num=f"{init + 1}/{self.number_of_repeats}",  # Pass repetition number
                    )
                )
            elif (
                self.clustering_method
                == "Topographic Atomize and Agglomerate Hierarchical Clustering"
            ):
                # For TAAHC, create a comprehensive progress callback
                def taahc_progress_callback(_current_step, _total_steps, message):
                    # Check if stopped
                    if worker and hasattr(worker, "stopped") and worker.stopped:
                        return False  # Signal to stop TAAHC

                    # Only log internal TAAHC progress, don't update main progress bar
                    # The main progress will be updated when the entire TAAHC iteration is completed
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            f"TAAHC internal progress: {message}", log_type="process"
                        )

                    return True  # Continue processing

                # Run TAAHC with progress tracking
                maps_init, residual_init = self.comet_microstate_clusterer.taahc(
                    data=self.maps2use,
                    metric=self.similarity_metric,
                    verbose=True,
                    progress_callback=taahc_progress_callback,
                    worker=worker,
                )
            else:
                # Unknown clustering method
                error_msg = f"Unknown clustering method: {self.clustering_method}"
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(error_msg, log_type="error")
                return None, 0.0, np.inf

        except Exception as e:
            error_msg = f"Clustering process failed: {str(e)}"
            self.logger.error("CLUSTERING", error_msg)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")
            return None, 0.0, np.inf

        # Check if stopped after clustering
        if worker and hasattr(worker, "stopped") and worker.stopped:
            self._handle_clustering_stop()
            return None, 0.0, np.inf

        # Only proceed if we have valid results
        if maps_init is not None:
            # Compute Global Explained Variance
            gev_init = self.comet_microstate_clusterer.compute_gev(
                data=self.maps2use, maps=maps_init
            )

            # Log iteration results
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                if is_taahc:
                    log_message = (
                        f"TAAHC Clustering Completed\n"
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
            if (
                self.best_maps is None
                or (hasattr(self, "best_gev") and gev_init > self.best_gev)
                or not hasattr(self, "best_gev")
            ):
                self.best_residual = (
                    residual_init.copy() if hasattr(residual_init, "copy") else residual_init
                )
                self.best_gev = gev_init
                self.best_maps = maps_init.copy() if hasattr(maps_init, "copy") else maps_init

                # Save the best maps immediately
                self.comet_microstate_io.export_microstates(
                    self.best_maps, self.eeg_info, self.microstate_maps_path
                )

                # Additional logging for best result updates
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    if is_taahc:
                        self.LogWindow.append_log(
                            f"✓ TAAHC clustering completed - GEV: {100 * gev_init:.3f}%"
                        )
                    else:
                        self.LogWindow.append_log(
                            f"✓ New best result found in iteration {init + 1} - GEV: {100 * gev_init:.3f}%"
                        )

            # Return the results
            return maps_init, gev_init, residual_init
        # No valid results
        error_msg = f"Clustering iteration {init + 1} failed - no maps generated"
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.append_log(error_msg, log_type="error")
        return None, 0.0, np.inf

    def _handle_clustering_stop(self):
        """Handle clustering stop - save best maps if any exist."""
        if self.best_maps is not None:
            # Compute final GEV
            self.best_gev = self.compute_gev_all_data()

            # Mark clustering as completed (even if stopped early)
            self.done_clustering = True

            # Log stop with current best results
            stop_message = (
                f"❌  [CLUSTERING] Stop Requested - Please Wait ...\n"
                f"⚠️ Clustering stopped by user\n"
                f"✓ Saved best maps found so far: {self.number_of_maps} microstates "
                f"(GEV: {100 * self.best_gev:.3f}%)"
            )

            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(stop_message, log_type="warning")
            else:
                self.logger.warning("CLUSTERING", stop_message)

            # Save configuration and logs
            if self.auto_save:
                self.save_config()
            self._save_logs()

            # Call clustering completion callback if set
            if self.clustering_completed_callback is not None:
                self.clustering_completed_callback()
        else:
            error_msg = "Clustering stopped - no maps were generated before stopping"
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")
            else:
                self.logger.error("CLUSTERING", error_msg)

    def backfit_eeg(self, eeg_path, eeg_name, worker=None):
        """Backfit microstate maps to an EEG file with safe-stop support."""
        if worker is not None and getattr(worker, "stopped", False):
            return

        eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)
        time_array = eeg.times * 1000

        # Log file processing start
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            file_info = (
                f"{eeg_name} | Duration: {eeg.times[-1]:.1f}s | {len(eeg.ch_names)} channels"
            )
            self.LogWindow.append_log(f"Backfitting {file_info}", log_type="file")

        labeled_segmentation, segmentation_fit = (
            self.comet_microstate_backfitter.perform_segmentation(
                eeg=eeg,
                filter_segments_less_than=int(
                    self.filter_segments_less_than_ms / (1000 / self.sample_rate)
                ),
            )
        )

        self.comet_segmentation_io.export_segmentation(
            output_folder=self.segmentation_path,
            filename=eeg_name,
            segmentation_array=labeled_segmentation,
            time_array=time_array,
            export_format=self.export_format,
        )

        # Log the backfitting progress
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            segments_count = len(labeled_segmentation)
            self.LogWindow.append_log(
                f"Successfully backfitted {eeg_name} ({segments_count} segments identified)",
                log_type="success",
            )

    def run_preprocessing(self):
        """Preprocess all EEG files in the input folder."""
        # Log section header
        self.logger.section_header("PREPROCESSING")

        # Start preprocessing
        self.logger.processing_start("PREPROCESSING", "Starting data preprocessing")

        # Reset directories based on current parameters
        self.reset_directories()

        # Ensure directories exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.preprocessed_data_path, exist_ok=True)

        # Only load raw files if list_eegs_path is empty or not set
        # This allows the NewStudyWindow to provide a pre-filtered list
        if not hasattr(self, "list_eegs_path") or not self.list_eegs_path:
            # Iterate through EEG files
            self.load_raw()

        # Check if any files were found
        if not hasattr(self, "list_eegs_path") or len(self.list_eegs_path) == 0:
            self.logger.error(
                "PREPROCESSING",
                f"No EEG files found in {self.input_folder} with extension {self.extension}",
            )
            return

        # Log preprocessing configuration
        preprocessing_steps = []
        if getattr(self, "temporal_filter_data", False):
            preprocessing_steps.append(
                f"Bandpass {self.lowcut_freq}-{self.highcut_freq} Hz ({self.filter_method.upper()})"
            )
        if getattr(self, "spatial_filter_data", False):
            preprocessing_steps.append("Spatial smoothing")
        if getattr(self, "downsample_data", False):
            preprocessing_steps.append(f"Downsampling to {self.sample_rate} Hz")
        if getattr(self, "prep_data", False):
            preprocessing_steps.append("Bad channel detection / interpolation")

        # Log study information and settings
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Study information
            study_info = {
                "Study Name": self.study_name,
                "Input Directory": self.input_folder,
                "Output Directory": self.save_dir,
                "Files Found": f"{len(self.list_eegs_path)} {self.datatype} EEG files with {self.extension} extension",
            }
            self.logger.settings_info("PREPROCESSING", study_info)

            # Log data selection mode between Files Found and preprocessing settings
            if getattr(self, "select_events_only", False) and getattr(self, "selected_event_label", None):
                selection_message = f"Data Selection: Only segments with event '{self.selected_event_label}'"
            else:
                selection_message = "Data Selection: Entire recording"
            self.logger.processing_info("PREPROCESSING", selection_message)

        # Build a list of EEG files
        self.zipped_eeg_files = list(zip(self.list_eegs_path, self.list_eegs))

        # Check channel consistency
        self.check_chan2rm()
        
        # Preprocessing settings (AFTER channel consistency check)
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            preprocessing_settings = {}
            if self.temporal_filter_data:
                preprocessing_settings["Bandpass Filter"] = (
                    f"{self.filter_method.upper()} method ({self.lowcut_freq}-{self.highcut_freq} Hz)"
                )
            else:
                preprocessing_settings["Bandpass Filter"] = "Disabled"

            preprocessing_settings["Spatial Filtering"] = (
                "Enabled" if self.spatial_filter_data else "Disabled"
            )
            preprocessing_settings["Downsampling"] = (
                f"{self.sample_rate} Hz" if self.downsample_data else "Disabled"
            )
            preprocessing_settings["Channels to Remove"] = (
                str(self.chan2rm) if self.chan2rm else "None"
            )
            # Check if interpolation is needed and which montage will be used
            if hasattr(self, 'missing_channels_per_file') and any(self.missing_channels_per_file.values()):
                montage_name = self.montage if hasattr(self, "montage") and self.montage else "standard_1020"
                preprocessing_settings["Channel Interpolation"] = f"Enabled (using {montage_name})"
            else:
                preprocessing_settings["Channel Interpolation"] = "Not needed"

            self.logger.settings_info("PREPROCESSING", preprocessing_settings)
            
            # Log channel consistency warning after preprocessing settings
            if hasattr(self, 'missing_channels_per_file') and any(self.missing_channels_per_file.values()):
                logger = get_logger()
                total_missing, files_with_missing, all_missing_channels = self.get_missing_channels_stats()
                logger.warning("PREPROCESSING", f"Channels are not consistent across all data files. {files_with_missing} files have missing channels. Total missing channels: {total_missing}. Missing channels: {', '.join(all_missing_channels)}")
        
        # Log channel consistency information
        total_missing, files_with_missing, all_missing_channels = self.get_missing_channels_stats()
        if total_missing > 0:
            montage_name = self.montage if hasattr(self, "montage") and self.montage else "standard_1020"
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(
                    f"Channel consistency check: {total_missing} missing channels across {files_with_missing} files. "
                    f"Missing channels: {', '.join(all_missing_channels)}. "
                    f"Will interpolate using montage: {montage_name}",
                    log_type="info"
                )

        # Start preprocessing
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.logger.processing_start(
                "PREPROCESSING", f"Processing {len(self.zipped_eeg_files)} EEG files"
            )
            self.LogWindow.setup_progress_dialog(
                window_title="Preprocessing ...",
                label_text="Preprocessing file ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.preprocess_eeg,
            )
            # Set callback to run when preprocessing worker thread finishes
            self.LogWindow.process_finished_callback = self._on_preprocessing_finished
        else:
            self.logger.processing_start(
                "PREPROCESSING", f"Preprocessing {len(self.zipped_eeg_files)} EEG files"
            )
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Preprocessing"):
                self.preprocess_eeg(eeg_path, eeg_name)
            # For non-GUI mode, call completion directly
            self._on_preprocessing_finished()

    def run_clustering(self):
        """Perform clustering on preprocessed EEG data with automatic or manual k selection
        Enhanced with proper TAAHC progress tracking and batch processing support.
        """
        # Log section header
        self.logger.section_header("CLUSTERING")

        # Log clustering configuration
        clustering_settings = {}
        clustering_settings["Clustering Method"] = getattr(self, "clustering_method", "Unknown")
        clustering_settings["Number of Maps"] = getattr(self, "number_of_maps", "Unknown")
        clustering_settings["Number of Repeats"] = getattr(self, "number_of_repeats", "Unknown")

        if getattr(self, "number_of_maps", None) == "auto":
            clustering_settings["K Range"] = (
                f"{getattr(self, 'kmin', 'Unknown')} to {getattr(self, 'kmax', 'Unknown')}"
            )
            clustering_settings["Stopping Mode"] = getattr(self, "stopping_mode", "Unknown")

        # Start clustering
        self.logger.processing_start("CLUSTERING", "Starting Microstate Clustering")
        self.logger.settings_info("CLUSTERING", clustering_settings)

        # Provide concise, high-signal details about the upcoming clustering
        # - K selection mode (user vs auto)
        # - Clustering input (GFP peaks vs random subset vs entire recordings)
        # - Effective repetitions and selection criterion (highest GEV)
        try:
            # K selection message
            if self.number_of_maps == "auto":
                k_selection_msg = (
                    f"K Selection: Automatic (kmin={getattr(self, 'kmin', 'Unknown')}, "
                    f"kmax={getattr(self, 'kmax', 'Unknown')}, "
                    f"stopping={getattr(self, 'stopping_mode', 'Unknown')})"
                )
            else:
                k_selection_msg = f"K Selection: User-specified (k={self.number_of_maps})"

            # Input selection message
            if self.number_of_maps == "auto":
                input_msg = "Clustering Input: GFP peaks (auto-k enforced)"
            else:
                use_pct = int(getattr(self, "use_percentages", 100) or 100)
                if use_pct >= 100:
                    input_msg = "Clustering Input: Entire recordings"
                else:
                    input_msg = f"Clustering Input: Random subset ({use_pct}%)"

            # Effective repetitions (TAAHC is deterministic → 1)
            is_taahc = (
                getattr(self, "clustering_method", "")
                == "Topographic Atomize and Agglomerate Hierarchical Clustering"
            )
            effective_repeats = 1 if is_taahc else getattr(self, "number_of_repeats", 1)
            repeats_msg = (
                f"Repetitions: {effective_repeats} (best solution selected by highest GEV)"
            )

            self.logger.processing_info("CLUSTERING", k_selection_msg)
            self.logger.processing_info("CLUSTERING", input_msg)
            self.logger.processing_info("CLUSTERING", repeats_msg)
        except Exception:
            # Logging should never break the flow
            pass

        # Add specific clustering parameters message with ⌛ emoji
        if self.number_of_maps != "auto":
            # Reflect effective repeats in the identifying message for deterministic methods
            is_taahc = (
                getattr(self, "clustering_method", "")
                == "Topographic Atomize and Agglomerate Hierarchical Clustering"
            )
            effective_repeats = 1 if is_taahc else getattr(self, "number_of_repeats", 1)
            self.logger.processing_info(
                "CLUSTERING",
                f"Identifying {self.number_of_maps} Microstate Maps with {effective_repeats} Repetitions...",
            )

        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Calculate total steps for clustering - only count clustering repetitions
            clustering_steps = self.number_of_repeats  # One step per repetition

            # Create a single task for the entire clustering process

            # Use setup_progress_dialog for worker thread with correct total steps
            self.LogWindow.setup_progress_dialog(
                window_title="Microstate Clustering...",
                label_text="Initializing clustering process...",
                tasks=clustering_steps,  # Pass only clustering steps
                processing_func=self._run_full_clustering_worker,
            )

            # Set callback to handle completion
            self.LogWindow.process_finished_callback = self._on_clustering_completed
        else:
            # Non-GUI mode - run directly
            self._run_full_clustering_direct()

    def _save_clustering_results(self):
        """Save clustering results to files."""
        try:
            # Create clustering results directory with proper naming convention
            clustering_results_path = os.path.join(
                self.save_dir, f"{self.study_name}_clustering_results"
            )
            os.makedirs(clustering_results_path, exist_ok=True)

            # Save microstate maps
            maps_file = os.path.join(clustering_results_path, "microstate_maps.npy")
            np.save(maps_file, self.best_maps)

            # Save clustering configuration and results
            if "clustering_results" not in self.config:
                self.config.add_section("clustering_results")
            self.config["clustering_results"]["maps_file"] = maps_file
            self.config["clustering_results"]["best_gev"] = str(self.best_gev)
            self.config["clustering_results"]["best_residual"] = str(self.best_residual)
            self.config["clustering_results"]["number_of_maps"] = str(self.number_of_maps)
            self.config["clustering_results"]["clustering_method"] = self.clustering_method

        except Exception as e:
            self.logger.error("CLUSTERING", f"Failed to save clustering results: {str(e)}")

    def _run_full_clustering_worker(self, _task_name):
        """Worker function for running the entire clustering process.
        This method is designed to be called by the LogWindow's worker thread.
        """
        # Ignore task_name parameter for step-based processing
        try:
            # Progress tracking variables - only count clustering repetitions
            total_steps = self.number_of_repeats  # Only clustering repetitions
            current_step = 0

            def update_progress(step_increment=1, message=""):
                nonlocal current_step
                # Handle both increment mode and absolute value mode
                if isinstance(step_increment, int) and step_increment >= 0:
                    # If step_increment is small (1-10), treat as increment
                    if step_increment <= 10:
                        current_step += step_increment
                    else:
                        # If step_increment is large, treat as absolute step value
                        current_step = step_increment
                else:
                    # If step_increment is a percentage (0-100), convert to step
                    current_step = int((step_increment / 100) * total_steps)

                    if (
                        hasattr(self, "LogWindow")
                        and self.LogWindow is not None
                        and hasattr(self.LogWindow, "worker_thread")
                        and self.LogWindow.worker_thread
                    ):
                        # Use current_step directly as the progress value (0 to total_steps)
                        # The progress bar range is already set to (0, total_steps)
                        progress_value = min(current_step, total_steps)
                        self.LogWindow.worker_thread.progress_updated.emit(progress_value, message)

            def check_stop():
                """Check if the process has been stopped by the user."""
                if (
                    hasattr(self, "LogWindow")
                    and self.LogWindow is not None
                    and hasattr(self.LogWindow, "worker_thread")
                    and self.LogWindow.worker_thread
                ):
                        return self.LogWindow.worker_thread.stopped
                return False

            # Step 1: Load EEG info
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.log_clustering_setup_step("Loading EEG information")
            if check_stop():
                return self._handle_stopped_clustering("Loading EEG information")
            self.load_eeg_info()

            # Step 1.5: Validate channel count consistency
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.log_clustering_setup_step("Validating channel consistency")
            if check_stop():
                return self._handle_stopped_clustering("Validating channel consistency")
            self.validate_channel_count_consistency()

            # Step 2: Calculate minimum distance size if smoothing GFP is enabled
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.log_clustering_setup_step("Calculating parameters")
            if check_stop():
                return self._handle_stopped_clustering("Calculating parameters")
            if self.smoothing_gfp:
                self.min_distance_size = int(
                    int(self.smoothing_distance) / (1000 / int(self.sample_rate))
                )
            else:
                self.min_distance_size = None

            # Step 3: Check if clustering method is supported
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.log_clustering_setup_step(
                    "Validating clustering method", self.clustering_method
                )
            available_methods = [
                "Modified K-Means Clustering (Pascual-Marqui et al. 1995)",
                "Modified K-Means Clustering with Spatial Similarity",
                "Topographic Atomize and Agglomerate Hierarchical Clustering",
            ]
            if self.clustering_method not in available_methods:
                error_msg = f"Clustering method '{self.clustering_method}' not supported."
                self.logger.error("CLUSTERING", error_msg)
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(error_msg, log_type="error")
                return False

            # Step 4: Determine if we need automatic k selection
            is_taahc = (
                self.clustering_method
                == "Topographic Atomize and Agglomerate Hierarchical Clustering"
            )

            # Step 5: Generate maps and peaks data for clustering
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.log_clustering_setup_step("Loading preprocessed data")
            if check_stop():
                return self._handle_stopped_clustering("Loading preprocessed data")

            # For auto-k selection, always use GFP peaks instead of random percentages
            auto_k_use_percentages = (
                None if self.number_of_maps == "auto" else self.use_percentages
            )

            self.maps2use, peaks = self.comet_data_initializer.generate_maps_and_peaks(
                preprocessed_folder=self.preprocessed_data_path,
                extension=self.extension,
                datatype=self.datatype,
                use_percentages=auto_k_use_percentages,
                min_dist=self.min_distance_size,
            )

            # Validate that the generated maps have the correct number of channels
            if hasattr(self, "eeg_info") and self.eeg_info is not None:
                expected_channels = len(self.eeg_info["ch_names"])
                actual_channels = self.maps2use.shape[0]
                if expected_channels != actual_channels:
                    error_msg = f"Channel count mismatch: EEG info has {expected_channels} channels, but maps have {actual_channels} channels"
                    self.logger.error("CLUSTERING", error_msg)
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(error_msg, log_type="error")
                    return False

            # Step 6: Handle automatic k selection
            if self.number_of_maps == "auto":
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.log_clustering_setup_step(
                        "Starting automatic optimization", f"k range: {self.kmin}-{self.kmax}"
                    )
                    # Add the processing message with ⌛ emoji
                    self.LogWindow.append_log(
                        f"⌛ Identifying {self.kmax - self.kmin + 1} optimal microstate maps..."
                    )
                if check_stop():
                    return self._handle_stopped_clustering("Starting automatic optimization")

                # Run automatic optimization with progress updates
                self._run_automatic_optimization_core_with_progress(update_progress, check_stop)

            # Step 7: After determining number_of_maps, perform actual clustering
            if self.number_of_maps and self.number_of_maps != "auto":
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.log_clustering_setup_step(
                        "Starting clustering",
                        f"{self.number_of_maps} maps, {self.number_of_repeats} repetitions",
                    )
                if check_stop():
                    return self._handle_stopped_clustering("Starting clustering repetitions")

                # Initialize microstate clusterer with special settings for TAAHC
                if is_taahc:
                    # TAAHC is deterministic and time-consuming - MUST use only 1 repetition
                    clustering_repeats = 1
                    if self.number_of_repeats != 1:
                        self.logger.warning(
                            "CLUSTERING",
                            f"TAAHC: Enforcing 1 repetition (deterministic algorithm) - ignoring user setting of {self.number_of_repeats}",
                        )
                else:
                    clustering_repeats = self.number_of_repeats

                self.comet_microstate_clusterer = MicrostateClusterer(
                    n_states=self.number_of_maps,
                    batch_size=self.batch_size,
                    n_inits=1,  # We'll handle repetitions manually
                    max_iter=self.max_iterations,
                    tolerance=self.clustering_tolerance,
                )

                # Perform clustering repetitions to find best solution
                best_maps = None
                best_gev = 0.0
                best_residual = np.inf

                if is_taahc:
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            "⌛ Running TAAHC clustering (deterministic algorithm)..."
                        )
                else:
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            f"⌛ Running {clustering_repeats} clustering repetitions..."
                        )

                # Run clustering repetitions with progress updates
                for completed_repetitions, init in enumerate(range(clustering_repeats), start=1):
                    if check_stop():
                        if is_taahc:
                            return self._handle_stopped_clustering(
                                "TAAHC clustering",
                                best_maps,
                                best_gev,
                                best_residual,
                                completed_repetitions,
                            )
                        return self._handle_stopped_clustering(
                            f"Clustering repetition {init + 1}/{clustering_repeats}",
                            best_maps,
                            best_gev,
                            best_residual,
                            completed_repetitions,
                        )

                    # Perform clustering without updating progress during internal processing
                    maps, gev, residual = self.cluster_eeg_microstates(init)

                    # Log detailed progress for each repetition
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        is_best = gev > best_gev
                        if is_taahc:
                            self.LogWindow.log_clustering_progress(
                                init + 1, clustering_repeats, gev, is_best
                            )
                        else:
                            self.LogWindow.log_clustering_progress(
                                init + 1, clustering_repeats, gev, is_best
                            )

                    # Only update progress after each repetition is completed
                    if is_taahc:
                        progress_msg = f"TAAHC Iteration {init + 1} completed"
                        update_progress(1, progress_msg)
                    else:
                        progress_msg = (
                            f"Clustering repetition {init + 1}/{clustering_repeats} completed"
                        )
                        update_progress(1, progress_msg)

                    if gev > best_gev:
                        best_gev = gev
                        best_maps = maps.copy()
                        best_residual = residual
                        if is_taahc:
                            self.logger.processing_info(
                                "CLUSTERING", f"TAAHC clustering completed - GEV: {best_gev:.4f}"
                            )
                        else:
                            self.logger.processing_info(
                                "CLUSTERING", f"New best GEV: {best_gev:.4f}"
                            )

                # Step 8: Store best results
                if best_maps is not None:
                    self.best_maps = best_maps
                    self.best_gev = best_gev
                    self.best_residual = best_residual

                    self.logger.processing_info(
                        "CLUSTERING",
                        f"Best GEV: {best_gev:.4f}, Best residual: {best_residual:.6f}",
                    )
                    self.logger.processing_success(
                        "CLUSTERING", "Clustering completed successfully!"
                    )

                    # Save clustering results
                    self._save_clustering_results()

                    # Update state flags
                    self.done_clustering = True
                    self.save_config()

                    return True

                self.logger.error("CLUSTERING", "Clustering failed - no valid results obtained")
                return False

            self.logger.error("CLUSTERING", "Number of maps not determined")
            return False

        except Exception as e:
            self.logger.error("CLUSTERING", f"Clustering process failed: {str(e)}")
            return False

    def _handle_stopped_clustering(
        self,
        stopped_at_step,
        best_maps=None,
        best_gev=0.0,
        best_residual=np.inf,
        completed_repetitions=0,
    ):
        """Handle clustering process that was stopped by the user.
        Returns partial results if available.
        """
        stop_msg = f"Clustering process stopped by user at: {stopped_at_step}"

        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.append_log(stop_msg, log_type="warning")

        # If we have partial results from clustering repetitions, save them
        if best_maps is not None and completed_repetitions > 0:
            self.best_maps = best_maps
            self.best_gev = best_gev
            self.best_residual = best_residual

            partial_msg = f"Partial results saved from {completed_repetitions} completed clustering repetitions"
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(partial_msg, log_type="info")
                self.LogWindow.append_log(f"Best GEV from partial results: {best_gev:.4f}")

            # Save partial clustering results
            self._save_clustering_results()

            # Update state flags
            self.done_clustering = True
            self.save_config()

            return True
        # No partial results available
        no_results_msg = "No clustering results available - process stopped too early"
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.append_log(no_results_msg, log_type="warning")

        return False

    def _run_full_clustering_direct(self):
        """Run full clustering process directly (non-GUI mode)."""
        return self._run_full_clustering_worker("full_clustering")

    def _run_clustering_repetition_worker(self, _task_name, init):
        """Worker function for running a single clustering repetition.
        This method is designed to be called by the LogWindow's worker thread.
        """
        maps, gev, residual = self.cluster_eeg_microstates(init)
        # Store result for later collection
        if not hasattr(self, "clustering_results"):
            self.clustering_results = []
        self.clustering_results.append((maps, gev, residual))
        return maps, gev, residual

    def _run_automatic_optimization_worker(self):
        """Run automatic optimization in worker thread.
        This method is designed to be called by the LogWindow's worker thread.
        """

        # Create optimizer with progress callback
        def progress_callback(_current, _total, message):
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                # Only log internal optimization progress, don't update main progress bar
                # The main progress will be updated when the entire optimization is completed
                self.LogWindow.append_log(f"Optimization progress: {message}", log_type="process")

        # Initialize optimizer
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
            max_iter=self.max_iterations,
            progress_callback=progress_callback,
            logger=self.logger,
        )

        # Run automatic optimization
        self._run_automatic_optimization_core()

    def _run_automatic_optimization_direct(self):
        """Run automatic optimization directly (non-GUI mode)."""
        # Initialize optimizer without progress callback
        # For auto-k selection, use only 1 repeat and force GFP peaks
        auto_k_n_inits = 1  # Force single repeat for auto-k selection
        self.logger.processing_info(
            "CLUSTERING", "Auto-k selection: Using single repeat (n_inits=1) for optimization"
        )

        self.comet_clusterer_optimizer = ClustererOptimizer(
            maps2use=self.maps2use,
            min_dist=self.min_distance_size,
            n_inits=auto_k_n_inits,  # Use single repeat for auto-k
            kmin=self.kmin,
            kmax=self.kmax,
            preprocessed_data_path=self.preprocessed_data_path,
            extension=self.extension,
            datatype=self.datatype,
            tolerance=self.clustering_tolerance,
            max_iter=self.max_iterations,
            logger=self.logger,
        )

        # Run automatic optimization
        self._run_automatic_optimization_core()

    def _run_automatic_optimization_core_with_progress(self, update_progress_func, check_stop_func):
        """Core automatic optimization logic using majority vote with progress updates."""
        try:
            # Create optimizer with progress callback that uses the provided update function
            def progress_callback(_current, _total, message):
                # This callback is called by the optimizer for each k value
                # Use the provided update_progress_func to maintain consistent progress tracking
                if update_progress_func:
                    # Calculate progress within the optimization phase
                    setup_steps = 5
                    optimization_steps = self.kmax - self.kmin + 1
                    # Calculate the current step within the optimization phase
                    optimization_current_step = setup_steps + int(
                        (_current / _total) * optimization_steps
                    )
                    # Pass the current step (not percentage)
                    update_progress_func(optimization_current_step, message)

            # Initialize optimizer with progress callback
            # For auto-k selection, use only 1 repeat and force GFP peaks
            auto_k_n_inits = 1  # Force single repeat for auto-k selection
            self.logger.processing_info(
                "CLUSTERING", "Auto-k selection: Using single repeat (n_inits=1) for optimization"
            )

            self.comet_clusterer_optimizer = ClustererOptimizer(
                maps2use=self.maps2use,
                min_dist=self.min_distance_size,
                n_inits=auto_k_n_inits,  # Use single repeat for auto-k
                kmin=self.kmin,
                kmax=self.kmax,
                preprocessed_data_path=self.preprocessed_data_path,
                extension=self.extension,
                datatype=self.datatype,
                tolerance=self.clustering_tolerance,
                max_iter=self.max_iterations,
                progress_callback=progress_callback,
                logger=self.logger,
            )

            # Connect LogWindow stop functionality to the optimizer
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                # Store reference to optimizer in LogWindow for stop functionality
                self.LogWindow.current_optimizer = self.comet_clusterer_optimizer

            # Check for stop before starting optimization
            if check_stop_func():
                return False

            # Use majority vote for auto-k selection
            if self.stopping_mode == "majority_vote":
                optimal_k = self.comet_clusterer_optimizer.find_optimal_k_majority_vote()
                self.optimization_results = self.comet_clusterer_optimizer.get_all_results()

                # Save optimization results for later visualization
                self.save_optimization_results(self.optimization_results)

                self.logger.processing_success(
                    "CLUSTERING", f"Optimal number of maps determined by majority vote: {optimal_k}"
                )
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(
                        f"Optimal number of maps determined by majority vote: {optimal_k}"
                    )

                    # Log individual method results
                    for method_name, result in self.optimization_results.items():
                        if method_name != "majority_vote":
                            self.LogWindow.append_log(
                                f"  {result.method_name}: k = {result.optimal_k}"
                            )
            else:
                # Fallback to single method (for backward compatibility)
                optimal_k, k_values, scores = self.comet_clusterer_optimizer.find_optimal_k(
                    optimizer_mode=self.stopping_mode, parameter_value=self.stopping_parameter
                )
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(f"Optimal number of maps determined: {optimal_k}")

            self.number_of_maps = optimal_k

            # Don't update progress here - only update when clustering repetitions are completed
            # if update_progress_func:
            #     update_progress_func(1, f"Automatic optimization completed - optimal k: {optimal_k}")

        except RuntimeError as e:
            if "stopped by user" in str(e):
                # Handle user-initiated stop
                self.logger.stop_requested("CLUSTERING")
                stop_msg = "Auto-k optimization stopped by user"
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(stop_msg, log_type="warning")
                # Fallback to default
                self.number_of_maps = 4
                self.logger.warning(
                    "CLUSTERING", f"Using fallback number of maps: {self.number_of_maps}"
                )
                # Don't update progress here - only update when clustering repetitions are completed
                # if update_progress_func:
                #     update_progress_func(1, "Auto-k optimization stopped by user")
            else:
                raise
        except Exception as e:
            error_msg = f"Automatic optimization failed: {str(e)}"
            self.logger.error("CLUSTERING", error_msg)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")
            # Fallback to default
            self.number_of_maps = 4
            self.logger.warning(
                "CLUSTERING", f"Using fallback number of maps: {self.number_of_maps}"
            )
            # Don't update progress here - only update when clustering repetitions are completed
            # if update_progress_func:
            #     update_progress_func(1, f"Automatic optimization failed: {str(e)}")

    def _run_automatic_optimization_core(self):
        """Core automatic optimization logic using majority vote."""
        try:
            # Use majority vote for auto-k selection
            if self.stopping_mode == "majority_vote":
                optimal_k = self.comet_clusterer_optimizer.find_optimal_k_majority_vote()
                self.optimization_results = self.comet_clusterer_optimizer.get_all_results()

                # Save optimization results for later visualization
                self.save_optimization_results(self.optimization_results)

                self.logger.processing_success(
                    "CLUSTERING", f"Optimal number of maps determined by majority vote: {optimal_k}"
                )
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(
                        f"Optimal number of maps determined by majority vote: {optimal_k}"
                    )

                    # Log individual method results
                    for method_name, result in self.optimization_results.items():
                        if method_name != "majority_vote":
                            self.LogWindow.append_log(
                                f"  {result.method_name}: k = {result.optimal_k}"
                            )
            else:
                # Fallback to single method (for backward compatibility)
                optimal_k, k_values, scores = self.comet_clusterer_optimizer.find_optimal_k(
                    optimizer_mode=self.stopping_mode, parameter_value=self.stopping_parameter
                )
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(f"Optimal number of maps determined: {optimal_k}")

            self.number_of_maps = optimal_k

        except Exception as e:
            error_msg = f"Automatic optimization failed: {str(e)}"
            self.logger.error("CLUSTERING", error_msg)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")
            # Fallback to default
            self.number_of_maps = 4
            self.logger.warning(
                "CLUSTERING", f"Using fallback number of maps: {self.number_of_maps}"
            )

    def run_microstate_labeling(self):
        """Label microstate maps."""
        # Check if best maps are available
        if self.best_maps is None:
            self.logger.error("LABELING", "No microstate maps available for labeling")
            return

        # Initialize microstate labeler
        self.comet_microstate_labeler = MicrostateLabeler(
            microstate_maps=self.best_maps,
            eeg_info=self.eeg_info,
            microstate_maps_path=self.microstate_maps_path,
        )

        # Log section header and start
        self.logger.section_header("LABELING")
        self.logger.processing_start("LABELING", "Starting microstate labeling")

        # Perform labeling
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

        # Log completion
        self.logger.processing_success("LABELING", "Microstate labeling completed successfully")

        # Save parameters
        if self.auto_save:
            self.save_config()

    def run_backfitting(self):
        """Backfit microstate maps to all EEG files."""
        # Log section header and start
        self.logger.section_header("BACKFITTING")
        self.logger.processing_start("BACKFITTING", "Starting microstate backfitting")

        # Log key backfitting parameters
        backfit_target = "GFP peaks" if self.backfit_to == "peaks" else "all time points"
        self.logger.processing_info("BACKFITTING", f"Backfitting Target: {backfit_target}")

        if self.filter_segments:
            filter_info = f"Segment Filtering: {self.filter_segments_option} (< {self.filter_segments_less_than}ms)"
        else:
            filter_info = "Segment Filtering: disabled"
        self.logger.processing_info("BACKFITTING", filter_info)

        # Check if best maps are available
        if self.best_maps is None:
            self.logger.error("BACKFITTING", "No microstate maps available for backfitting")
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
            self.logger.processing_start(
                "BACKFITTING", "Identifying optimal smoothing window length"
            )

            # Create progress dialog
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.setup_progress_dialog(
                    window_title="Backfitting ...",
                    label_text="Identifying the optimal length of the smoothing window ...",
                    max_value=len(self.list_eegs_path),
                )

            rm_max_len = 50
            len_win2rm_list = list(range(0, rm_max_len, int(1000 / self.sample_rate)))
            similarity_scores = np.empty((len(self.list_eegs_path), len(len_win2rm_list)))

            # Iterate through EEG files
            for eeg_idx, (eeg_path, eeg_name) in enumerate(
                zip(self.list_eegs_path, self.list_eegs)
            ):
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.update_progress(value=eeg_idx, text=f"{eeg_name}")
                # Removed individual file processing print for cleaner output

                eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, datatype=self.datatype)

                # Compute similarity scores for different segment removal lengths
                for idx_win2rm, len_win2rm in enumerate(len_win2rm_list):
                    similarity_scores[eeg_idx, idx_win2rm] = (
                        self.comet_microstate_backfitter.get_similarity_score(
                            eeg=eeg, rm_max_len=len_win2rm
                        )
                    )

            # Identify optimal length filter
            self.filter_segments_less_than_ms = (
                self.comet_microstate_backfitter.identify_optimal_length_filter(
                    similarity_scores=similarity_scores
                )
            )
            self.logger.processing_success(
                "BACKFITTING",
                f"Optimal filter length determined: {self.filter_segments_less_than_ms} ms",
            )
        else:
            if self.filter_segments:
                self.filter_segments_less_than_ms = self.filter_segments_less_than
            else:
                self.filter_segments_less_than_ms = 0

        # Log backfitting settings
        if self.backfit_to == "peaks":
            backfit_to_text = (
                "Backfitting microstates to the local peaks of the global field power."
            )
        else:
            backfit_to_text = "Backfitting microstates to all time points."

        if self.filter_segments_option == "remove":
            filter_segments_option_text = (
                f"Removing segments with less than "
                f"{self.filter_segments_less_than_ms}ms in duration."
            )
        elif self.filter_segments_option == "replace_high":
            filter_segments_option_text = (
                f"Replacing segments with less than {self.filter_segments_less_than_ms}ms"
                f"by the nearby microstate with higher occurrence."
            )
        elif self.filter_segments_option == "replace_half":
            filter_segments_option_text = (
                f"Replacing segments with less than {self.filter_segments_less_than_ms}ms"
                f"by half by the previous and half by the next dominant microstate."
            )
        elif self.filter_segments_option == "smooth":
            filter_segments_option_text = (
                f"Smoothing segments with window size {self.filter_segments_less_than_ms}ms"
                f" and lambda {self.lamb}."
            )
        else:
            filter_segments_option_text = "No filtering applied to short segments."

        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.append_log(
                f"Microstates Backfitting Settings:\n"
                f"* {backfit_to_text}\n"
                f"* {filter_segments_option_text}",
                log_type="settings",
            )
        else:
            self.logger.processing_info("BACKFITTING", "Microstates Backfitting Settings:")
            self.logger.processing_info("BACKFITTING", f"* {backfit_to_text}")
            self.logger.processing_info("BACKFITTING", f"* {filter_segments_option_text}")

        self.load_clean()
        self.zipped_eeg_files = list(zip(self.list_eegs_path, self.list_eegs))

        # Perform backfitting on all files
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.setup_progress_dialog(
                window_title="Backfitting ...",
                label_text="Backfitting microstates to data ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.backfit_eeg,
            )
            # Set callback to run when backfitting worker thread finishes
            self.LogWindow.process_finished_callback = self._on_backfitting_finished
        else:
            self.logger.processing_start(
                "BACKFITTING", f"Processing {len(self.zipped_eeg_files)} EEG files"
            )
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Backfitting"):
                self.backfit_eeg(eeg_path, eeg_name)
            # For non-GUI mode, call completion directly
            self._on_backfitting_finished()

    def run_feature_extraction(self):
        """Extract features from all segmentation files."""
        # Log section header and start
        self.logger.section_header("FEATURE_EXTRACTION")
        self.logger.processing_start("FEATURE_EXTRACTION", "Starting Feature Extraction")

        # Create features output directory
        os.makedirs(self.extracted_features_path, exist_ok=True)

        # Check if segmentation files exist
        self.segmentation_list_path, self.segmentation_list = self.comet_data_io.find_data(
            input_folder=self.segmentation_path, extension=self.export_format, pattern="*"
        )

        if not self.segmentation_list_path:
            self.logger.error(
                "FEATURE_EXTRACTION", f"No segmentation files found in {self.segmentation_path}"
            )
            return

        # Log feature extraction settings
        feature_settings = {
            "Features to Extract": ", ".join(self.feature_list),
            "Feature Modes": ", ".join(self.feature_mode),
            "Feature Types": ", ".join(self.feature_types),
        }

        if "sliding" in self.feature_mode:
            feature_settings["Sliding Window Size"] = self.sliding_window_size
            if hasattr(self, "pre_window_size"):
                feature_settings["Pre-Window Size"] = self.pre_window_size
            if hasattr(self, "post_window_size"):
                feature_settings["Post-Window Size"] = self.post_window_size

        self.logger.settings_info("FEATURE_EXTRACTION", feature_settings)

        # Initialize shared storage for thread-safe feature extraction
        COMET._shared_feature_results = {}

        # Create list of tasks for threading
        tasks = [
            (idx, path, name)
            for idx, (path, name) in enumerate(
                zip(self.segmentation_list_path, self.segmentation_list)
            )
        ]

        # Start feature extraction
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Use the logging window with a progress dialog
            self.logger.processing_start("FEATURE_EXTRACTION", "Extracting Features")
            self.LogWindow.setup_progress_dialog(
                window_title="Extracting Features ...",
                label_text="Extracting features ...",
                tasks=tasks,
                processing_func=self.extract_features_for_file_threadsafe,
            )
            # Defer result collection until the worker thread finishes
            self.LogWindow.process_finished_callback = self.collect_feature_extraction_results
        else:
            # Fallback: run sequentially in the main thread
            self.logger.processing_start("FEATURE_EXTRACTION", "Extracting Features")
            for task in tqdm(tasks, desc="Feature Extraction"):
                self.extract_features_for_file_threadsafe(*task)
            # Collect results immediately when done
            self.collect_feature_extraction_results()

    def extract_features_for_file_threadsafe(
        self, segmentation_idx, segmentation_path, segmentation_name, worker=None
    ):
        """Thread-safe feature extraction with stop check."""
        if worker is not None and getattr(worker, "stopped", False):
            return

        try:
            # Load segmentation
            segmentation_array = self.comet_segmentation_io.load_segmentation(
                segmentation_path=segmentation_path, import_format=self.export_format
            )

            # Convert to expected format for feature extraction
            if segmentation_array is not None:
                # Handle event-based sliding windows first
                if (
                    getattr(self, "event_based_sliding", False)
                    and "sliding" in self.feature_mode
                    and hasattr(self, "selected_events")
                    and self.selected_events
                ):
                    # Log that event-based processing is starting
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            f"🔄 Event-based sliding feature extraction enabled for {segmentation_name}",
                            log_type="info"
                        )
                        self.LogWindow.append_log(
                            f"   Selected events: {', '.join(self.selected_events)}",
                            log_type="info"
                        )
                    
                    try:
                        # Load preprocessed EEG to fetch accurate annotation timings
                        subject_name = os.path.splitext(segmentation_name)[0]
                        try:
                            preproc_eeg = self.get_preprocessed_eeg(subject_name)
                        except FileNotFoundError as e:
                            # Provide more helpful error message for missing files
                            error_msg = (
                                f"❌ Event-based sliding FAILED: Could not find preprocessed EEG file for {subject_name}. "
                                f"Please ensure the file was preprocessed and saved correctly. Error: {str(e)}"
                            )
                            self.logger.error("FEATURE_EXTRACTION", error_msg)
                            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                self.LogWindow.append_log(error_msg, log_type="error")
                            # Re-raise to prevent fallback - this is a critical error
                            raise RuntimeError(f"Event-based sliding failed: {str(e)}")
                        sfreq = preproc_eeg.info["sfreq"]
                        ann = preproc_eeg.annotations
                        
                        # Validate that annotations exist
                        if ann is None or len(ann) == 0:
                            raise RuntimeError(f"No annotations found in {segmentation_name}. Event-based sliding requires annotations.")
                        
                        # Log event-based processing start
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(
                                f"Starting event-based feature extraction for {segmentation_name} "
                                f"with events: {', '.join(self.selected_events)}",
                                log_type="info"
                            )
                            self.LogWindow.append_log(
                                f"   Found {len(ann)} annotations in file, sampling rate: {sfreq}Hz",
                                log_type="info"
                            )
                        
                        # Build windows for selected events
                        event_windows = []  # list of (start_idx,end_idx,label)
                        available_events = list(ann.description) if ann else []
                        
                        # Early validation: Check if ANY of the selected events exist in available events
                        matching_mode = getattr(self, "event_matching_mode", "partial")
                        has_any_match = False
                        
                        for selected_event in self.selected_events:
                            for available_event in available_events:
                                matched = False
                                
                                if matching_mode == "exact":
                                    matched = selected_event == available_event
                                elif matching_mode == "case_insensitive":
                                    matched = selected_event.lower() == available_event.lower()
                                else:  # "partial"
                                    matched = (selected_event == available_event or 
                                             selected_event.lower() == available_event.lower() or
                                             selected_event.lower() in available_event.lower() or 
                                             available_event.lower() in selected_event.lower())
                                
                                if matched:
                                    has_any_match = True
                                    break
                            if has_any_match:
                                break
                        
                        if not has_any_match:
                            available_str = ', '.join(sorted(set(available_events))) if available_events else 'None'
                            selected_str = ', '.join(self.selected_events) if self.selected_events else 'None'
                            raise RuntimeError(
                                f"Event-based sliding failed: None of the selected events [{selected_str}] "
                                f"match any available events [{available_str}] using {matching_mode} matching mode."
                            )
                        
                        # Get the total number of samples in the data
                        total_samples = preproc_eeg.n_times
                        
                        # Check segmentation array length
                        seg_length = len(segmentation_array) if len(segmentation_array.shape) == 1 else segmentation_array.shape[1]
                        
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(
                                f"Data info - EEG samples: {total_samples}, Segmentation length: {seg_length}, Sampling rate: {sfreq}Hz",
                                log_type="info"
                            )
                        
                        # Use the minimum of the two lengths to avoid out-of-bounds errors
                        max_valid_samples = min(total_samples, seg_length)
                        
                        # Log available events for debugging
                        if hasattr(self, "LogWindow") and self.LogWindow is not None and available_events:
                            unique_events = sorted(set(available_events))
                            self.LogWindow.append_log(
                                f"Available events in {segmentation_name}: {', '.join(unique_events)}",
                                log_type="info"
                            )
                            
                            # Log event details for debugging
                            for desc, onset, dur in zip(ann.description, ann.onset, ann.duration):
                                self.LogWindow.append_log(
                                    f"  Event '{desc}': onset={onset:.3f}s, duration={dur:.3f}s",
                                    log_type="info"
                                )
                        
                        # Process events using the already configured matching_mode
                        
                        # Get all annotation data for calculating windows
                        all_onsets = list(ann.onset)
                        all_durations = list(ann.duration)
                        all_descriptions = list(ann.description)
                        
                        for event_idx, (desc, onset, duration) in enumerate(zip(ann.description, ann.onset, ann.duration)):
                            matched = False
                            
                            # Apply matching based on mode
                            if matching_mode == "exact":
                                # Only exact matching
                                matched = desc in self.selected_events
                            
                            elif matching_mode == "case_insensitive":
                                # Exact match (case-insensitive)
                                for selected_event in self.selected_events:
                                    if desc.lower() == selected_event.lower():
                                        matched = True
                                        break
                            
                            else:  # default to "partial" for backward compatibility
                                # Try exact match first
                                matched = desc in self.selected_events
                                
                                # If no exact match, try case-insensitive match
                                if not matched:
                                    for selected_event in self.selected_events:
                                        if desc.lower() == selected_event.lower():
                                            matched = True
                                            break
                                
                                # If still no match, try partial/substring matching (both ways)
                                if not matched:
                                    for selected_event in self.selected_events:
                                        # Check if selected event is substring of desc or vice versa
                                        if (selected_event.lower() in desc.lower() or 
                                            desc.lower() in selected_event.lower()):
                                            matched = True
                                            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                                self.LogWindow.append_log(
                                                    f"Partial match: '{selected_event}' matched with '{desc}'",
                                                    log_type="info"
                                                )
                                            break
                            
                            if matched:
                                start_idx = int(round(onset * sfreq))
                                
                                # For onset-only annotations, create windows from current event to next event
                                # Find the next event after this one (any event, not just selected ones)
                                next_onset = None
                                for j in range(event_idx + 1, len(all_onsets)):
                                    if all_onsets[j] > onset:
                                        next_onset = all_onsets[j]
                                        break
                                
                                if next_onset is not None:
                                    # Use time until next event (regardless of what the next event is)
                                    window_duration = next_onset - onset
                                    end_idx = int(round(next_onset * sfreq))
                                    
                                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                        self.LogWindow.append_log(
                                            f"Event '{desc}' at {onset:.3f}s - window until next event at {next_onset:.3f}s ({window_duration:.3f}s)",
                                            log_type="info"
                                        )
                                else:
                                    # No next event, use window until end of data
                                    end_idx = max_valid_samples
                                    window_duration = (max_valid_samples - start_idx) / sfreq
                                    
                                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                        self.LogWindow.append_log(
                                            f"Event '{desc}' at {onset:.3f}s - window until end of data ({window_duration:.3f}s)",
                                            log_type="info"
                                        )
                                
                                # Ensure end_idx doesn't exceed valid data length
                                end_idx = min(end_idx, max_valid_samples)
                                
                                # Also ensure start_idx is within bounds
                                if start_idx >= max_valid_samples:
                                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                        self.LogWindow.append_log(
                                            f"Skipping event '{desc}' at {onset:.3f}s - start index {start_idx} exceeds data length {max_valid_samples}",
                                            log_type="warning"
                                        )
                                    continue
                                
                                if end_idx > start_idx:
                                    event_windows.append((start_idx, end_idx, desc))
                                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                        self.LogWindow.append_log(
                                            f"Added event window for '{desc}': samples {start_idx}-{end_idx} ({(end_idx-start_idx)/sfreq:.3f}s)",
                                            log_type="info"
                                        )
                                else:
                                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                        self.LogWindow.append_log(
                                            f"Skipping event '{desc}' at {onset:.3f}s - invalid window (start={start_idx}, end={end_idx})",
                                            log_type="warning"
                                        )
                        
                        if not event_windows:
                            # Provide detailed error message with available events
                            available_str = ', '.join(sorted(set(available_events))) if available_events else 'None'
                            selected_str = ', '.join(self.selected_events) if self.selected_events else 'None'
                            
                            error_msg = (
                                f"❌ Event-based sliding FAILED: No matching events found in {segmentation_name}.\n"
                                f"   Selected events: [{selected_str}]\n"
                                f"   Available events in file: [{available_str}]\n"
                                f"   Event matching mode: {getattr(self, 'event_matching_mode', 'partial')}"
                            )
                            
                            self.logger.error("FEATURE_EXTRACTION", error_msg)
                            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                self.LogWindow.append_log(error_msg, log_type="error")
                            # Raise RuntimeError to prevent silent fallback to fixed sliding
                            raise RuntimeError(f"Event-based sliding failed: No matching events found. Selected: {selected_str}, Available: {available_str}")
                        
                        # Log number of event windows found
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(
                                f"Found {len(event_windows)} event windows in {segmentation_name}",
                                log_type="info"
                            )
                        
                        # For event-based sliding, we need to create a custom sliding extraction
                        # that includes Window_index and Event_name columns properly
                        all_sliding_dfs = {ftype: [] for ftype in self.feature_types}
                        
                        # Process each event window
                        for window_index, (start_idx, end_idx, ev_label) in enumerate(event_windows):
                            # Ensure indices are within bounds of segmentation array
                            seg_len = len(segmentation_array) if len(segmentation_array.shape) == 1 else segmentation_array.shape[1]
                            safe_start = max(0, min(start_idx, seg_len - 1))
                            safe_end = max(safe_start + 1, min(end_idx, seg_len))
                            
                            window_labels = (
                                segmentation_array[safe_start:safe_end].tolist()
                                if len(segmentation_array.shape) == 1
                                else segmentation_array[0, safe_start:safe_end].tolist()
                            )
                            
                            if not window_labels:
                                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                    self.LogWindow.append_log(
                                        f"Empty window for event '{ev_label}' (indices {start_idx}-{end_idx})",
                                        log_type="warning"
                                    )
                                continue
                            
                            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                self.LogWindow.append_log(
                                    f"Processing event '{ev_label}' window: {len(window_labels)} samples",
                                    log_type="info"
                                )
                            
                            # Build minimal segmentation dict to reuse coordinator
                            # Create segmentation data for this event window
                            # Include event information for proper handling
                            seg_stub = {
                                "filename": segmentation_name,
                                "labels": window_labels,  # Add this field which is expected by the coordinator
                                "microstate_labels": window_labels,
                                "time": list(range(len(window_labels))),
                                "event_name": ev_label,  # Add event name
                                "window_index": window_index,  # Add window index
                            }
                            coordinator = FeatureExtractionCoordinator()
                            feat_res = coordinator.extract_features(
                                segmentation=seg_stub,
                                feature_list=self.feature_list,
                                feature_mode=["averaged"],
                                feature_types=self.feature_types,
                                sliding_window_size=1,
                                min_samples=None,
                            )
                            # Process the results for each feature type
                            averaged_dict = feat_res.get("averaged", {})
                            for ftype, df_list in averaged_dict.items():
                                for df in df_list:
                                    # Create a new DataFrame with proper structure for sliding mode
                                    # Get the feature columns (all except 'Filename')
                                    feature_cols = [col for col in df.columns if col != 'Filename']
                                    
                                    # Create rows for sliding format
                                    sliding_rows = []
                                    for feat_col in feature_cols:
                                        if feat_col in df.columns:
                                            sliding_rows.append({
                                                'Filename': segmentation_name,
                                                'Window_index': window_index,
                                                'Event_name': ev_label,
                                                'Feature': feat_col,
                                                'Value': df[feat_col].iloc[0] if not df.empty else None
                                            })
                                    
                                    # Create DataFrame in sliding format
                                    if sliding_rows:
                                        sliding_df = pd.DataFrame(sliding_rows)
                                        # Pivot to get the expected format
                                        pivoted_df = sliding_df.pivot_table(
                                            index=['Filename', 'Window_index', 'Event_name'],
                                            columns='Feature',
                                            values='Value'
                                        ).reset_index()
                                        
                                        # Add timing information
                                        pivoted_df['window_start_idx'] = start_idx
                                        pivoted_df['window_end_idx'] = end_idx
                                        pivoted_df['window_duration_ms'] = (end_idx - start_idx) * 1000 / sfreq
                                        
                                        all_sliding_dfs[ftype].append(pivoted_df)
                        
                        # Combine all DataFrames for each feature type
                        final_sliding_results = {"sliding": {}}
                        for ftype in self.feature_types:
                            if all_sliding_dfs[ftype]:
                                # Concatenate all DataFrames for this feature type
                                combined_df = pd.concat(all_sliding_dfs[ftype], ignore_index=True)
                                final_sliding_results["sliding"][ftype] = [combined_df]
                            else:
                                final_sliding_results["sliding"][ftype] = []
                        
                        # Log successful event-based extraction
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(
                                f"✅ Successfully extracted event-based features from {segmentation_name} ({len(event_windows)} event windows processed)",
                                log_type="success"
                            )
                        
                        # Store results
                        COMET._shared_feature_results[segmentation_idx] = {
                            "segmentation_name": segmentation_name,
                            "extracted_features": final_sliding_results,
                        }
                        return  # Skip standard sliding processing
                    except RuntimeError as _eb_err:
                        # RuntimeError indicates a critical event-based sliding error - do not fallback
                        error_msg = f"❌ Event-based feature extraction failed for {segmentation_name}: {str(_eb_err)}"
                        self.logger.error("FEATURE_EXTRACTION", error_msg)
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(
                                f"❌ Event-based extraction failed: {str(_eb_err)}",
                                log_type="error"
                            )
                        # Re-raise to stop processing - do not fallback to fixed sliding
                        raise
                    except Exception as _eb_err:
                        # For other unexpected errors, provide detailed error info and prevent fallback
                        error_msg = f"❌ Event-based feature extraction failed for {segmentation_name}: {str(_eb_err)}"
                        self.logger.error("FEATURE_EXTRACTION", error_msg)
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(
                                f"❌ Unexpected error in event-based extraction: {str(_eb_err)}. Event-based sliding requires debugging.",
                                log_type="error"
                            )
                        # Convert to RuntimeError to prevent silent fallback
                        raise RuntimeError(f"Event-based sliding failed with unexpected error: {str(_eb_err)}") from _eb_err

                # Check if we have epoched data with sliding features enabled OR ROF feature requested
                is_epoched_sliding = self.datatype == "epoched" and "sliding" in self.feature_mode
                needs_epoched_structure = self.datatype == "epoched" and "ROF" in self.feature_list

                if (is_epoched_sliding or needs_epoched_structure) and len(
                    segmentation_array.shape
                ) == 2:
                    # For epoched sliding features or ROF, preserve trial structure
                    # Use first trial for time array creation, but keep all trials for processing
                    labels = [str(item) for item in segmentation_array[0, :]]  # Ensure strings
                    # Store original segmentation array for feature extraction
                    original_segmentation_array = segmentation_array
                else:
                    # Standard processing - flatten if needed
                    if len(segmentation_array.shape) == 2:
                        labels = [str(item) for item in segmentation_array[0, :]]  # Ensure strings
                    else:
                        labels = [str(item) for item in segmentation_array]  # Ensure strings
                    original_segmentation_array = None

                # Retrieve accurate time points directly from segmentation file when available
                num_samples = len(labels)
                time = None
                try:
                    if self.export_format == ".csv":
                        df_time = pd.read_csv(segmentation_path, usecols=["time"])
                        time_unique = sorted(df_time["time"].unique())
                        if len(time_unique) == num_samples:
                            time = list(time_unique)
                    # TODO: handle other formats (pkl, hdf, json) similarly if needed
                except Exception as _e_time:
                    # Fallback to old behaviour if reading fails
                    time = None

                if time is None:
                    # Fallback: derive from sampling rate as before
                    if hasattr(self, "sample_rate") and self.sample_rate:
                        time_step = 1000 / self.sample_rate  # ms
                        if self.datatype == "epoched":
                            start_time = -1000
                            time = [start_time + i * time_step for i in range(num_samples)]
                        else:
                            time = [i * time_step for i in range(num_samples)]
                    else:
                        time = list(range(num_samples))

                # Load corresponding EEG data
                eeg_name = os.path.splitext(segmentation_name)[0]
                # Find the actual EEG file with the correct extension
                eeg_file = None
                preprocessed_files = os.listdir(self.preprocessed_data_path)
                for file in preprocessed_files:
                    if file.startswith(eeg_name + ".") or file == eeg_name:
                        eeg_file = os.path.join(self.preprocessed_data_path, file)
                        break

                if eeg_file is None:
                    raise FileNotFoundError(
                        f"Could not find EEG file for {eeg_name} in {self.preprocessed_data_path}"
                    )

                # Load the EEG data
                eeg = self.comet_data_io.load_eeg(eeg_file, self.datatype)

                # For epoched sliding features, preserve 3D structure (trials, channels, timepoints)
                # Otherwise use standard flattened structure
                if is_epoched_sliding:
                    eeg_data = eeg.get_data()  # Keep original 3D structure for epoched data
                else:
                    eeg_data = self.comet_data_io.get_eeg_data(
                        eeg, self.datatype
                    )  # Standard processing

                # Create segmentation dictionary in expected format
                segmentation = {
                    "labels": labels,
                    "time": time,
                    "filename": segmentation_name,
                    "eeg_data": eeg_data,
                    "microstate_maps": self.best_maps,
                    "microstate_labels": self.micro_labels,
                }

                # Add original segmentation data for epoched sliding processing or ROF
                if original_segmentation_array is not None:
                    segmentation["original_segmentation_array"] = original_segmentation_array
                    # For ROF calculation, also store the epoched labels directly
                    if needs_epoched_structure:
                        segmentation["epoched_labels"] = original_segmentation_array
            else:
                # Create empty segmentation if loading failed
                segmentation = {
                    "labels": [],
                    "time": [],
                    "filename": segmentation_name,
                    "eeg_data": None,
                    "microstate_maps": None,
                    "microstate_labels": None,
                }

            # Log file processing start
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                segments_count = (
                    len(segmentation["labels"]) if "labels" in segmentation else "unknown"
                )
                duration = segmentation["time"][-1] / 1000 if "time" in segmentation else "unknown"
                file_info = (
                    f"{segmentation_name} | Duration: {duration:.1f}s | Segments: {segments_count}"
                )
                self.LogWindow.append_log(f"Extracting features from {file_info}", log_type="file")

            # Extract features
            extracted_features = self.comet_feature_extractor.extract_features(
                segmentation=segmentation,
                feature_list=self.feature_list,
                feature_mode=self.feature_mode,
                feature_types=self.feature_types,
                sliding_window_size=self.sliding_window_size,
                pre_window_size=self.pre_window_size,
                post_window_size=self.post_window_size,
            )

            # Store in shared storage with thread-safe access
            COMET._shared_feature_results[segmentation_idx] = {
                "segmentation_name": segmentation_name,
                "extracted_features": extracted_features,
            }

            # Log successful feature extraction
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                feature_count = (
                    sum(len(mode_data) for mode_data in extracted_features.values())
                    if extracted_features
                    else 0
                )
                self.LogWindow.append_log(
                    f"Successfully extracted features from {segmentation_name} (modes: {feature_count})",
                    log_type="success",
                )

        except Exception as e:
            error_msg = f"Error extracting features from {segmentation_name}: {str(e)}"
            self.logger.error("FEATURE_EXTRACTION", error_msg)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")

    def collect_feature_extraction_results(self, _message=None):
        """Collect feature extraction results from shared storage and export them
        This is called when the worker thread finishes.
        """
        # Organize results by mode and type
        organized_results = {}

        # Initialize the structure for each mode and type
        for mode in self.feature_mode:
            organized_results[mode] = {}
            for feature_type in self.feature_types:
                organized_results[mode][feature_type] = []

        # Collect results from each file
        for _file_idx, file_data in COMET._shared_feature_results.items():
            if isinstance(file_data, dict) and "extracted_features" in file_data:
                extracted_features = file_data["extracted_features"]

                # Organize by mode and type
                for mode in self.feature_mode:
                    if mode in extracted_features:
                        for feature_type in self.feature_types:
                            if feature_type in extracted_features[mode]:
                                organized_results[mode][feature_type].extend(
                                    extracted_features[mode][feature_type]
                                )

                        # Handle ROF data separately
                        if "rof_data" in extracted_features[mode]:
                            if "rof_data" not in organized_results[mode]:
                                organized_results[mode]["rof_data"] = {}
                            organized_results[mode]["rof_data"].update(
                                extracted_features[mode]["rof_data"]
                            )

                        # Handle RTF data separately
                        if "rtf_data" in extracted_features[mode]:
                            if "rtf_data" not in organized_results[mode]:
                                organized_results[mode]["rtf_data"] = {}
                            organized_results[mode]["rtf_data"].update(
                                extracted_features[mode]["rtf_data"]
                            )

        # Combine DataFrames for each mode and type
        for mode in self.feature_mode:
            for feature_type in self.feature_types:
                if organized_results[mode][feature_type]:
                    # Combine all DataFrames for this mode and type
                    combined_df = pd.concat(
                        organized_results[mode][feature_type], ignore_index=True
                    )

                    # Create output directory if it doesn't exist
                    os.makedirs(self.extracted_features_path, exist_ok=True)

                    # Export the combined features
                    try:
                        self.comet_feature_io.export_features(
                            features_df=combined_df,
                            feature_type=feature_type,
                            feature_mode=mode,
                            output_folder=self.extracted_features_path,
                            export_format=self.export_format,
                        )

                        # Log successful export
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(
                                f"Exported {mode} features for {feature_type} ({len(combined_df)} records)",
                                log_type="success",
                            )

                    except Exception as export_error:
                        error_msg = (
                            f"Failed to export {mode} features for {feature_type}: {export_error}"
                        )
                        self.logger.error("FEATURE_EXTRACTION", error_msg)
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(error_msg, log_type="error")

        # Export aggregated ROF and RTF data (single file each) if available
        for mode in self.feature_mode:
            if mode in organized_results:
                # ROF export
                if "rof_data" in organized_results[mode]:
                    rof_data = organized_results[mode]["rof_data"]
                    if rof_data:
                        try:
                            self.comet_feature_io.export_rof_timeseries(
                                rof_data_dict=rof_data,
                                output_folder=self.extracted_features_path,
                                export_format=self.export_format,
                            )

                            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                self.LogWindow.append_log(
                                    f"Exported aggregated ROF time-series for {mode} mode ({len(rof_data)} files)",
                                    log_type="success",
                                )
                        except Exception as rof_export_error:
                            error_msg = (
                                f"Failed to export ROF data for {mode} mode: {rof_export_error}"
                            )
                            self.logger.error("FEATURE_EXTRACTION", error_msg)
                            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                self.LogWindow.append_log(error_msg, log_type="error")

                # RTF export
                if "rtf_data" in organized_results[mode]:
                    rtf_data = organized_results[mode]["rtf_data"]
                    if rtf_data:
                        try:
                            self.comet_feature_io.export_rtf_data(
                                rtf_data_dict=rtf_data,
                                output_folder=self.extracted_features_path,
                                export_format=self.export_format,
                            )

                            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                self.LogWindow.append_log(
                                    f"Exported aggregated RTF averages for {mode} mode ({len(rtf_data)} files)",
                                    log_type="success",
                                )
                        except Exception as rtf_export_error:
                            error_msg = (
                                f"Failed to export RTF data for {mode} mode: {rtf_export_error}"
                            )
                            self.logger.error("FEATURE_EXTRACTION", error_msg)
                            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                self.LogWindow.append_log(error_msg, log_type="error")

        # Set feature extraction flag
        self.done_extracting_features = True

        # Log completion
        self.logger.processing_success(
            "FEATURE_EXTRACTION", "Feature Extraction Completed Successfully"
        )

        # Log completion summary

        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Clear the callback to prevent infinite recursion
            self.LogWindow.process_finished_callback = None

        # Save parameters
        if self.auto_save:
            self.save_config()

        # Clear the shared storage to free memory
        COMET._shared_feature_results = {}

        # Notify any registered callbacks (e.g., GUI updates)
        if (
            hasattr(self, "feature_extraction_completed_callback")
            and self.feature_extraction_completed_callback is not None
        ):
            self.feature_extraction_completed_callback()

    def get_segmentation(self, subject_name):
        """Load a segmentation file on demand."""
        seg_path = os.path.join(self.segmentation_path, f"{subject_name}{self.export_format}")
        return self.comet_segmentation_io.load_segmentation(
            segmentation_path=seg_path, import_format=self.export_format
        )

    def source_localize_file(self, eeg_path, eeg_name, worker=None):
        """Perform source localization for a single file. Terminates early if stop requested."""
        if worker is not None and getattr(worker, "stopped", False):
            return

        success = self.comet_source_localizer.localize_single_file(eeg_path, eeg_name)

        # Log the source localization progress
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            if success:
                self.LogWindow.append_log(f"Source Localized: {eeg_name}")
            else:
                self.LogWindow.append_log(f"Error Source Localizing: {eeg_name}")

    def source_identify_file(self, eeg_path, eeg_name, worker=None):
        """Identify microstate sources for a single file with stop support."""
        if worker is not None and getattr(worker, "stopped", False):
            return

        success = self.comet_source_localizer.identify_sources_single_file(
            eeg_path, eeg_name, self.source_localization_method
        )

        # Log the source identification progress
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            if success:
                self.LogWindow.append_log(f"Microstate Sources Identified: {eeg_name}")
            else:
                self.LogWindow.append_log(f"Error Identifying Microstate Sources: {eeg_name}")

    def run_source_localization(self):
        """Perform source localization for microstates."""
        # Log section header and start
        self.logger.section_header("SOURCE_LOCALIZATION")
        self.logger.processing_start("SOURCE_LOCALIZATION", "Starting Source Localization")

        # Check if backfitting has been done
        if not self.done_backfitting:
            self.logger.error(
                "SOURCE_LOCALIZATION", "Backfitting must be completed before source localization"
            )
            return

        # Determine the subjects directory
        if self.use_anatomy == "individual":
            if hasattr(self, "individual_subjects_dir"):
                self.anatomy_subjects_dir = self.individual_subjects_dir
            else:
                self.logger.error(
                    "SOURCE_LOCALIZATION", "individual_subjects_dir not set for individual anatomy"
                )
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
            nperm=self.nperm,
            logger=self.logger,
        )

        # Make sure the stc_path is set correctly in the source localizer
        self.comet_source_localizer.stc_path = stc_path

        # Load EEG files to process
        list_eeg_path, list_eeg_name = self.comet_data_io.find_data(
            self.preprocessed_data_path, extension=self.extension, pattern="*"
        )
        self.zipped_eeg_files = list(zip(list_eeg_path, list_eeg_name))

        # Log source localization settings once
        self.comet_source_localizer.log_settings()

        # Perform source localization on all files
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.setup_progress_dialog(
                window_title="Source Localization ...",
                label_text="Localizing sources ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.source_localize_file,
            )
            # Set callback to run when source localization worker thread finishes
            self.LogWindow.process_finished_callback = self._on_source_localization_finished
        else:
            self.logger.processing_start(
                "SOURCE_LOCALIZATION", f"Processing {len(self.zipped_eeg_files)} EEG files"
            )
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Source Localization"):
                self.source_localize_file(eeg_path, eeg_name)
            # For non-GUI mode, call completion directly
            self._on_source_localization_finished()

    def run_identifying_microstate_sources(self):
        """Correlate sources and microstates."""
        # Log section header and start
        self.logger.section_header("SOURCE_LOCALIZATION")
        self.logger.processing_start(
            "SOURCE_LOCALIZATION", "Starting source-microstate correlation"
        )

        # Check if source localization has been done
        if not self.done_source_localization:
            self.logger.error(
                "SOURCE_LOCALIZATION", "Source localization must be completed before correlation"
            )
            return

        # Check if anatomy subjects directory is available
        try:
            _ = self.anatomy_subjects_dir
        except AttributeError:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
            self.anatomy_subjects_dir = os.path.dirname(fs_dir)

        # Make sure the necessary paths are set correctly in the source localizer
        if not hasattr(self, "comet_source_localizer") or self.comet_source_localizer is None:
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
                nperm=self.nperm,
                logger=self.logger,
            )

        # Ensure directories are created and paths are set
        if self.source_localization_method == "tess":
            self.tess_path = os.path.join(self.localized_sources_path, "tess_sources")
            os.makedirs(self.tess_path, exist_ok=True)
            self.comet_source_localizer.tess_path = self.tess_path
        elif self.source_localization_method == "avg":
            self.avg_sources_path = os.path.join(self.localized_sources_path, "avg_sources")
            os.makedirs(self.avg_sources_path, exist_ok=True)
            self.comet_source_localizer.avg_sources_path = self.avg_sources_path

        # Make sure stc_path is set
        stc_path = os.path.join(self.localized_sources_path, "stc")
        self.comet_source_localizer.stc_path = stc_path

        # Load EEG files to process
        list_eeg_path, list_eeg_name = self.comet_data_io.find_data(
            self.preprocessed_data_path, extension=self.extension, pattern="*"
        )
        self.zipped_eeg_files = list(zip(list_eeg_path, list_eeg_name))

        # Log source identification settings
        source_settings = {
            "Method": self.source_localization_method,
            "Number of Permutations": self.nperm,
            "Files to Process": len(self.zipped_eeg_files),
        }
        self.logger.settings_info("SOURCE_LOCALIZATION", source_settings)

        # Perform source identification on all files
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.logger.processing_start(
                "SOURCE_LOCALIZATION", f"Processing {len(self.zipped_eeg_files)} EEG files"
            )
            self.LogWindow.setup_progress_dialog(
                window_title="Source Identification ...",
                label_text=f"Identifying microstate sources using {self.source_localization_method} method ...",
                tasks=self.zipped_eeg_files,
                processing_func=self.source_identify_file,
            )
            # Set callback to run when source identification worker thread finishes
            self.LogWindow.process_finished_callback = self._on_source_identification_finished
        else:
            self.logger.processing_start(
                "SOURCE_LOCALIZATION", f"Processing {len(self.zipped_eeg_files)} EEG files"
            )
            for eeg_path, eeg_name in tqdm(self.zipped_eeg_files, desc="Source Identification"):
                self.source_identify_file(eeg_path, eeg_name)
            # For non-GUI mode, call completion directly
            self._on_source_identification_finished()

    def save_config(self):
        """Update and save the current configuration settings and program state to save_dir.
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
        # Save user-specified channels to remove (not interpolated channels)
        if isinstance(self.chan2rm, list):
            self.config["preprocessing_config"]["ch2rm"] = ", ".join(self.chan2rm)
        else:
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
        self.config["backfitting_config"]["filter_segments_less_than"] = str(
            self.filter_segments_less_than
        )
        self.config["backfitting_config"]["filter_segments_option"] = self.filter_segments_option
        self.config["backfitting_config"]["epsilon"] = str(self.epsilon)
        self.config["backfitting_config"]["b"] = str(self.b)
        self.config["backfitting_config"]["lamb"] = str(self.lamb)

        self.config["features_config"]["export_format"] = self.export_format
        self.config["features_config"]["feature_list"] = ", ".join(self.feature_list)
        self.config["features_config"]["feature_mode"] = ", ".join(self.feature_mode)
        self.config["features_config"]["feature_types"] = ", ".join(self.feature_types)
        self.config["features_config"]["sliding_window_size"] = str(self.sliding_window_size)
        self.config["features_config"]["event_based_sliding"] = str(self.event_based_sliding)
        self.config["features_config"]["selected_events"] = ", ".join(self.selected_events)
        self.config["features_config"]["event_matching_mode"] = self.event_matching_mode
        self.config["features_config"]["pre_window_size"] = str(self.pre_window_size)
        self.config["features_config"]["post_window_size"] = str(self.post_window_size)

        self.config["source_config"]["bem_solver"] = self.bem_solver
        self.config["source_config"]["inverse_method"] = self.inverse_method
        self.config["source_config"]["nperm"] = str(self.nperm)
        self.config["source_config"]["spacing"] = self.spacing
        self.config["source_config"]["source_localization_method"] = self.source_localization_method
        self.config["source_config"]["anatomy_subjects_dir"] = self.anatomy_subjects_dir

        # --- NEW: Save common events information ---
        if hasattr(self, "common_events"):
            if "events_config" not in self.config:
                self.config.add_section("events_config")
            self.config["events_config"]["common_events"] = ", ".join(self.common_events)
        # --- END NEW ---

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
            self.done_identifying_microstate_sources
        )

        # Note: optimization_results section is preserved if it exists
        # It should only be modified through save_optimization_results() method

        # Add timestamp for when config was last saved
        if "metadata" not in self.config:
            self.config.add_section("metadata")

        import time

        self.config["metadata"]["last_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        try:
            # Write config to file in save_dir
            with open(self.config_path, "w+", encoding="utf-8") as configfile:
                self.config.write(configfile)
        except Exception as e:
            self.logger.warning("CONFIGURATION", f"Failed to save configuration: {e}")

    def _save_logs(self):
        """Save current log content to the log_text attribute and to the log file."""
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.log_text = self.LogWindow.get_log_content()

        # Save logs to the separate log file
        self.save_logs_to_file()

    def save_logs_to_file(self):
        """Save current log content to the separate log file."""
        if hasattr(self, "log_file_path") and self.log_text:
            try:
                # Ensure the directory exists before writing
                log_dir = os.path.dirname(self.log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                with open(self.log_file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_text)
            except Exception as e:
                self.logger.warning("DATA_IO", f"Failed to save logs: {e}")

    def load_logs_from_file(self):
        """Load log content from the separate log file."""
        if hasattr(self, "log_file_path") and os.path.exists(self.log_file_path):
            try:
                with open(self.log_file_path, encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                self.logger.warning("DATA_IO", f"Failed to load logs: {e}")
                return ""
        return ""

    def load_optimization_results(self):
        """Load optimization results from configuration if available."""
        if hasattr(self, "optimization_results") and self.optimization_results:
            return self.optimization_results
        return None

    def save_optimization_results(self, results):
        """Save optimization results to configuration."""
        self.optimization_results = results

    def _launch_microstate_labeling(self):
        """Launch the microstate labeling window after clustering completion."""
        try:
            # Import the microstate labeling window
            from controllers.microstate_visualization_window import MicrostateVisualizationWindow

            # Check if we have a context available
            if not hasattr(self, "context") or self.context is None:
                self.logger.warning(
                    "VISUALIZATION", "Could not find context for microstate visualization window"
                )
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(
                        "Could not launch microstate labeling window - context not available",
                        log_type="warning",
                    )
                return

            # Get the main window from LogWindow if available
            main_window = None
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                main_window = getattr(self.LogWindow, "parent", None)
                if main_window is None:
                    main_window = self.LogWindow

            # Create the microstate visualization window with correct parameters
            microstate_window = MicrostateVisualizationWindow(
                self.context, main_window=main_window, tbx=self
            )
            microstate_window.plot_maps()  # Plot the microstates
            microstate_window.show()

            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log("Microstate labeling window launched", log_type="info")

        except Exception as e:
            error_msg = f"Failed to launch microstate labeling window: {str(e)}"
            self.logger.error("VISUALIZATION", error_msg)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")

    def _on_clustering_completed(self, _message=None):
        """Handle completion of the full clustering process."""
        try:
            # Clear the callback to prevent infinite recursion
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.process_finished_callback = None

            # Ensure logs are saved
            self._save_logs()

            # Launch microstate labeling window
            self._launch_microstate_labeling()

            # Notify any registered callbacks (e.g., GUI updates)
            if (
                hasattr(self, "clustering_completed_callback")
                and self.clustering_completed_callback is not None
            ):
                self.clustering_completed_callback()

        except Exception as e:
            error_msg = f"Error in clustering completion handler: {str(e)}"
            self.logger.error("CLUSTERING", error_msg)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")

    def _on_clustering_repetitions_finished(self, _message=None):
        """Collect clustering results from worker threads and find the best solution."""
        # Find the best result from all repetitions
        best_maps = None
        best_gev = 0.0
        best_residual = np.inf

        if hasattr(self, "clustering_results") and self.clustering_results:
            for maps, gev, residual in self.clustering_results:
                if gev > best_gev:
                    best_gev = gev
                    best_maps = maps.copy()
                    best_residual = residual

        # Store best results
        if best_maps is not None:
            self.best_maps = best_maps
            self.best_gev = best_gev
            self.best_residual = best_residual

            # Log completion
            self.logger.processing_success("CLUSTERING", "Clustering completed successfully")

            # Save clustering results
            self._save_clustering_results()

            # Update state flags
            self.done_clustering = True
            self.save_config()

            # Clear the callback to prevent infinite recursion
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.process_finished_callback = None

            # Ensure logs are saved
            self._save_logs()

            # Launch microstate labeling window
            self._launch_microstate_labeling()

        else:
            self.logger.error("CLUSTERING", "Clustering failed - no valid results obtained")

    def _on_clustering_iterations_finished(self, *_args, **_kwargs):
        """Finalize clustering once the worker thread has completed all iterations."""
        # Check if worker was stopped
        worker_stopped = False
        if (
            hasattr(self, "LogWindow")
            and self.LogWindow is not None
            and hasattr(self.LogWindow, "worker_thread")
            and self.LogWindow.worker_thread
        ):
                worker_stopped = self.LogWindow.worker_thread.stopped

        # Verify that clustering produced maps
        if self.best_maps is None:
            if worker_stopped:
                self.logger.warning(
                    "CLUSTERING", "Clustering stopped before any maps were generated"
                )
            else:
                self.logger.error(
                    "CLUSTERING", "Clustering failed - no microstate maps were generated"
                )
            return

        # Compute final GEV across all data
        self.best_gev = self.compute_gev_all_data()

        # Mark clustering as completed
        self.done_clustering = True

        # Log completion
        if worker_stopped:
            self.logger.warning("CLUSTERING", "Clustering stopped early but maps were saved")
        else:
            self.logger.processing_success("CLUSTERING", "Clustering completed successfully")

        # Clear the callback to prevent it from being triggered by other processes
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.process_finished_callback = None

        # Persist configuration and logs
        if self.auto_save:
            self.save_config()
        self._save_logs()

        # Notify any registered callbacks (e.g., GUI updates)
        if self.clustering_completed_callback is not None:
            self.clustering_completed_callback()

    def _on_preprocessing_finished(self, _message=None):
        """Handle preprocessing completion when worker thread finishes."""
        # Set preprocessing flag
        self.save_eeg_info(self.eeg_info_path)
        self.done_preprocessing = True

        # --- NEW: Determine common events across all files and persist ---
        if hasattr(self, "events_per_file") and self.events_per_file:
            event_sets = [set(ev.keys()) for ev in self.events_per_file.values() if ev]
            if event_sets:
                self.common_events = sorted(list(set.intersection(*event_sets)))
            else:
                self.common_events = []
        else:
            self.common_events = []

        # Store into config for later retrieval
        if not hasattr(self, "config"):
            self.config = self.create_default_config()
        if "events_config" not in self.config:
            self.config.add_section("events_config")
        self.config["events_config"]["common_events"] = ", ".join(self.common_events)
        # --- END NEW ---

        # Log completion with interpolation summary
        total_missing, files_with_missing, all_missing_channels = self.get_missing_channels_stats()
        if total_missing > 0:
            montage_name = self.montage if hasattr(self, "montage") and self.montage else "standard_1020"
            completion_msg = f"Preprocessing completed successfully. Interpolated {total_missing} missing channels across {files_with_missing} files using montage '{montage_name}'."
        else:
            completion_msg = "Preprocessing completed successfully. No channel interpolation was needed."
        
        self.logger.processing_success("PREPROCESSING", completion_msg)

        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Clear the callback to prevent it from being triggered by other processes
            self.LogWindow.process_finished_callback = None

        # Save configuration and parameters
        if self.auto_save:
            self.save_config()

        # Ensure logs are saved
        self._save_logs()

        # Notify any registered callbacks (e.g., GUI updates)
        if (
            hasattr(self, "preprocessing_completed_callback")
            and self.preprocessing_completed_callback is not None
        ):
            try:
                self.preprocessing_completed_callback()
            except Exception as cb_err:
                self.logger.warning(
                    "PREPROCESSING", f"Error in preprocessing completion callback: {cb_err}"
                )

    def _on_backfitting_finished(self, _message=None):
        """Handle backfitting completion when worker thread finishes."""
        # Set backfitting flag
        self.done_backfitting = True

        # Log completion
        self.logger.processing_success("BACKFITTING", "Backfitting completed successfully")

        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Clear the callback to prevent it from being triggered by other processes
            self.LogWindow.process_finished_callback = None

        # Save parameters
        if self.auto_save:
            self.save_config()

        # Ensure logs are saved
        self._save_logs()

        # Notify any registered callbacks (e.g., GUI updates)
        if (
            hasattr(self, "backfitting_completed_callback")
            and self.backfitting_completed_callback is not None
        ):
            self.backfitting_completed_callback()

    def _on_source_localization_finished(self, _message=None):
        """Handle source localization completion when worker thread finishes."""
        # Set source localization flag
        self.done_source_localization = True

        # Log completion
        self.logger.processing_success(
            "SOURCE_LOCALIZATION", "Source localization completed successfully"
        )

        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Clear the callback to prevent it from being triggered by other processes
            self.LogWindow.process_finished_callback = None

        # Save parameters
        if self.auto_save:
            self.save_config()

        # Ensure logs are saved
        self._save_logs()

        # Notify any registered callbacks (e.g., GUI updates)
        if (
            hasattr(self, "source_localization_completed_callback")
            and self.source_localization_completed_callback is not None
        ):
            self.source_localization_completed_callback()

    def _on_source_identification_finished(self, _message=None):
        """Handle source identification completion when worker thread finishes."""
        # Set source-microstate correlation flag
        self.done_identifying_microstate_sources = True

        # Log completion
        self.logger.processing_success(
            "SOURCE_LOCALIZATION", "Source-microstate correlation completed successfully"
        )

        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Clear the callback to prevent it from being triggered by other processes
            self.LogWindow.process_finished_callback = None

        # Save parameters
        if self.auto_save:
            self.save_config()

        # Ensure logs are saved
        self._save_logs()

        # Notify any registered callbacks (e.g., GUI updates)
        if (
            hasattr(self, "source_microstate_correlation_completed_callback")
            and self.source_microstate_correlation_completed_callback is not None
        ):
            self.source_microstate_correlation_completed_callback()
