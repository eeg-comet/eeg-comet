"""Core COMET workflow for EEG-COMET.

Provides the `COMET` class implementing preprocessing, clustering, labeling,
backfitting, feature extraction, source localization, and correlation routines.
"""

import contextlib
import glob
import os
import time
from configparser import ConfigParser

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from eeg_comet.backfitting_utils.microstate_backfitter import MicrostateBackfitter
from eeg_comet.backfitting_utils.segmentation_io import SegmentationIO
from eeg_comet.clustering_utils.clusterer_optimizer import (
    VALID_STOPPING_MODES,
    ClustererOptimizer,
)
from eeg_comet.clustering_utils.microstate_clusterer import MicrostateClusterer
from eeg_comet.clustering_utils.microstate_io import MicrostateIO
from eeg_comet.clustering_utils.microstate_labeler import MicrostateLabeler
from eeg_comet.clustering_utils.microstate_visualizer import reset_electrode_warning
from eeg_comet.controllers.logging_window import LogWindow
from eeg_comet.controllers.microstate_visualization_window import MicrostateVisualizationWindow
from eeg_comet.data_utils.data_initializer import DataInitializer
from eeg_comet.data_utils.data_io import DataIO
from eeg_comet.data_utils.data_preprocessor import DataPreprocessor
from eeg_comet.features_utils.feature_extractor import FeatureExtractionCoordinator, FeatureExtractor
from eeg_comet.features_utils.feature_helper import FeatureHelper
from eeg_comet.features_utils.feature_io import FeatureIO
from eeg_comet.gui_utils.terminal_logger import get_logger
from eeg_comet.sourcelocalization_utils.source_localizer import SourceLocalizer


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
        self.data_type = "raw"

        # Preprocessing
        self.temporal_filter_data = True
        self.filter_method = "fir"
        self.lowcut_freq = 2
        self.highcut_freq = 20
        self.downsample_data = True
        self.sampling_rate = 250
        self.spatial_filter_data = False
        self.auto_clean_data = False
        self.remove_channels = False
        self.channels_to_remove = ""
        self.prep_data = False

        # Clustering
        self.smoothing_gfp = False
        self.smoothing_distance = 10
        self.n_maps = 4
        self.choose_number_of_maps = "User"
        self.k_min = 2
        self.k_max = 10
        self.stopping_mode = "majority_vote"
        self.stopping_threshold = 10
        self.initializer = "Random"
        self.clustering_method = "Modified K-Means Clustering"
        self.max_iterations = 500
        self.clustering_tolerance = 1e-6
        self.similarity_metric = ""
        self.n_repeats = 5
        self.data_percentage = 100

        # Backfitting
        self.backfit_to = "all"
        self.identify_short_window = False
        self.filter_segments = False
        self.filter_segments_less_than = 20
        self.filter_segments_option = "smooth"
        self.convergence_epsilon = 1e-6
        self.half_window_size = 3
        self.smoothness_penalty = 5
        self.min_correlation_threshold = False
        self.filter_segments_less_than_ms = 0

        # Features
        self.export_format = ".csv"
        self.feature_list = ["OCC", "DUR", "COV"]
        self.feature_mode = ["averaged"]
        self.feature_types = ["real"]
        self.sliding_window_size = 1
        # How DUR is summarised across per-segment lengths. Default is
        # ``geometric`` (geometric mean of run lengths, robust to long-tail
        # outliers); ``arithmetic`` uses the mean of run lengths with the
        # (N-1)/fs interval convention and is algebraically consistent with
        # COV and OCC. See FeatureExtractor for all accepted values.
        self.duration_method = "geometric"
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
        self.n_permutations = 2000
        self.spacing = "ico3"
        self.source_localization_method = "tess"
        self.anatomy_subjects_dir = ""

        # Runtime/derived attributes used across methods
        self.config_last_saved = "Unknown"
        self.events_per_file = {}
        self.common_events = []
        self.labels_overall_confidence = None
        self.label_confidences = None  # Per-label confidence (label -> %)
        self.maps_to_use = None
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
        self.microstate_labels = []
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
        self.comet_feature_extractor = FeatureExtractionCoordinator(
            random_seed=getattr(self, "random_seed", None),
            duration_method=getattr(self, "duration_method", "geometric"),
        )
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
        
        # Initialize EEG info preservation for digitization points
        self._processed_eeg_info_captured = None

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
        config["io_config"]["pattern_content"] = "*"
        config["io_config"]["data_type"] = "raw"
        config["io_config"]["output_folder"] = ""

        # Set default preprocessing values
        config["preprocessing_config"]["temporal_filter_data"] = "True"
        config["preprocessing_config"]["filter_method"] = "fir"
        config["preprocessing_config"]["lowcut_freq"] = "2"
        config["preprocessing_config"]["highcut_freq"] = "20"
        config["preprocessing_config"]["downsample_data"] = "True"
        config["preprocessing_config"]["sampling_rate"] = "250"
        config["preprocessing_config"]["spatial_filter_data"] = "False"
        config["preprocessing_config"]["auto_clean_data"] = "False"
        config["preprocessing_config"]["remove_channels"] = "False"
        config["preprocessing_config"]["channels_to_remove"] = ""
        config["preprocessing_config"]["prep_data"] = "False"

        # Set default clustering values
        config["clustering_config"]["smoothing_gfp"] = "False"
        config["clustering_config"]["smoothing_distance"] = "10"
        config["clustering_config"]["n_maps"] = "4"
        config["clustering_config"]["k_min"] = "2"
        config["clustering_config"]["k_max"] = "10"
        config["clustering_config"]["stopping_mode"] = "majority_vote"
        config["clustering_config"]["stopping_threshold"] = "10"
        config["clustering_config"]["data_percentage"] = "100"
        config["clustering_config"]["initializer"] = "Random"
        config["clustering_config"][
            "clustering_method"
        ] = "Modified K-Means Clustering"
        config["clustering_config"]["max_iterations"] = "500"
        config["clustering_config"]["clustering_tolerance"] = "1e-6"
        config["clustering_config"]["similarity_metric"] = "Spatial Correlation"
        config["clustering_config"]["n_repeats"] = "5"
        # Set default backfitting values
        config["backfitting_config"]["backfit_to"] = "all"
        config["backfitting_config"]["identify_short_window"] = "False"
        config["backfitting_config"]["filter_segments"] = "False"
        config["backfitting_config"]["filter_segments_less_than"] = "20"
        config["backfitting_config"]["filter_segments_option"] = "smooth"
        config["backfitting_config"]["convergence_epsilon"] = "1e-6"
        config["backfitting_config"]["half_window_size"] = "3"
        config["backfitting_config"]["smoothness_penalty"] = "5"
        config["backfitting_config"]["min_correlation_threshold"] = "False"

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
        config["source_config"]["n_permutations"] = "2000"
        config["source_config"]["spacing"] = "ico3"
        config["source_config"]["source_localization_method"] = "tess"
        config["source_config"]["anatomy_subjects_dir"] = ""

        return config

    def reset_directories(self):
        """Reset and recreate directory structure when critical parameters change.
        Call this whenever study_name or output_folder are changed.
        """
        # Reset captured EEG info when directories are reset (new study)
        self._processed_eeg_info_captured = None
        
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
            # Create clustering results directory and set microstate_maps_path
            clustering_results_path = self.get_clustering_results_path()
            os.makedirs(clustering_results_path, exist_ok=True)
            self.microstate_maps_path = os.path.join(clustering_results_path, "microstate_maps.csv")
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
        # Reset electrode warning flag for new study
        reset_electrode_warning()
        
        # Input/Output Configs
        io_config = self.config["io_config"]
        self.study_name = io_config.get("study_name", "my_study")
        self.input_folder = io_config.get("input_folder", "")
        self.montage = io_config.get("montage", "")
        self.extension = io_config.get("extension", ".auto")
        self.pattern_content = io_config.get("pattern_content", io_config.get("pattern", "*"))
        self.data_type = io_config.get("data_type", io_config.get("datatype", "raw"))
        self.output_folder = io_config.get("output_folder", "")

        # Preprocessing Configs
        preprocessing_config = self.config["preprocessing_config"]
        # Historically written under [preprocessing_config] even though it pairs
        # with pattern_content in [io_config]; accept it from either section.
        self.load_all_files = io_config.getboolean(
            "load_all_files", preprocessing_config.getboolean("load_all_files", True)
        )
        self.temporal_filter_data = preprocessing_config.getboolean(
            "temporal_filter_data", preprocessing_config.getboolean("filter_data", True)
        )
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
            self.sampling_rate = preprocessing_config.getint(
                "sampling_rate", preprocessing_config.getint("sample_rate", 250)
            )
        else:
            self.sampling_rate = 250

        self.spatial_filter_data = preprocessing_config.getboolean("spatial_filter_data", False)
        self.auto_clean_data = preprocessing_config.getboolean("auto_clean_data", False)
        self.remove_channels = preprocessing_config.getboolean("remove_channels", False)
        if self.remove_channels:
            self.channels_to_remove = preprocessing_config.get(
                "channels_to_remove", preprocessing_config.get("ch2rm", "")
            )
        else:
            self.channels_to_remove = ""
        self.prep_data = preprocessing_config.getboolean("prep_data", False)

        # Clustering Configs
        clustering_config = self.config["clustering_config"]
        self.smoothing_gfp = clustering_config.getboolean("smoothing_gfp", False)
        if self.smoothing_gfp:
            self.smoothing_distance = clustering_config.getint("smoothing_distance", 10)
        else:
            self.smoothing_distance = 10

        n_maps_setting = clustering_config.get("n_maps", clustering_config.get("number_of_maps", "4"))
        self.n_maps = n_maps_setting if n_maps_setting == "auto" else int(n_maps_setting)
        self.choose_number_of_maps = "Auto" if n_maps_setting == "auto" else "User"

        if self.n_maps == "auto":
            self.k_min = clustering_config.getint("k_min", clustering_config.getint("kmin", 2))
            self.k_max = clustering_config.getint("k_max", clustering_config.getint("kmax", 10))
            self.stopping_mode = clustering_config.get("stopping_mode", "majority_vote")
            if self.stopping_mode not in VALID_STOPPING_MODES:
                # Caught here rather than deep in the optimisation run, where the
                # resulting error was swallowed and silently fell back to k=4.
                self.logger.warning(
                    "CLUSTERING",
                    f"Unknown stopping_mode '{self.stopping_mode}'; falling back to "
                    f"'majority_vote'. Valid options: {', '.join(VALID_STOPPING_MODES)}",
                )
                self.stopping_mode = "majority_vote"
            self.stopping_threshold = clustering_config.getint(
                "stopping_threshold", clustering_config.getint("stopping_parameter", 10)
            )
        else:
            self.k_min = 2
            self.k_max = 10
            self.stopping_mode = "majority_vote"
            self.stopping_threshold = 10

        self.initializer = clustering_config.get("initializer", "Random")
        self.clustering_method = clustering_config.get(
            "clustering_method", "Modified K-Means Clustering"
        )
        self.max_iterations = clustering_config.getint("max_iterations", 500)
        try:
            self.clustering_tolerance = clustering_config.getfloat("clustering_tolerance", 1e-6)
        except (ValueError, TypeError):
            # Handle case where clustering_tolerance is empty string or invalid
            self.clustering_tolerance = 1e-6
        self.similarity_metric = (
            clustering_config.get("similarity_metric", "Spatial Correlation")
            if self.clustering_method != "Modified K-Means Clustering"
            else ""
        )
        self.n_repeats = clustering_config.getint(
            "n_repeats", clustering_config.getint("number_of_repeats", 5)
        )

        # Handle data_percentage
        try:
            self.data_percentage = clustering_config.getint(
                "data_percentage", clustering_config.getint("use_percentages", 100)
            )
        except (ValueError, KeyError):
            self.data_percentage = 100

        # Allow data_percentage to be None for GFP peak detection
        # Note: None means use GFP peaks, values 20-100 mean use percentage of data

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

        try:
            self.convergence_epsilon = backfitting_config.getfloat(
                "convergence_epsilon", backfitting_config.getfloat("epsilon", 1e-6)
            )
        except (ValueError, TypeError):
            # Handle case where convergence_epsilon is empty string or invalid
            self.convergence_epsilon = 1e-6
        try:
            self.half_window_size = backfitting_config.getint(
                "half_window_size", backfitting_config.getint("b", 3)
            )
        except (ValueError, TypeError):
            self.half_window_size = 3
        try:
            self.smoothness_penalty = backfitting_config.getint(
                "smoothness_penalty", backfitting_config.getint("lamb", 5)
            )
        except (ValueError, TypeError):
            self.smoothness_penalty = 5
        _corr_thresh = backfitting_config.get("min_correlation_threshold", "False").strip()
        if _corr_thresh in ("", "False", "false", "0"):
            self.min_correlation_threshold = False
        else:
            try:
                self.min_correlation_threshold = float(_corr_thresh)
            except (ValueError, TypeError):
                self.min_correlation_threshold = False

        # Feature Extraction Configs
        features_config = self.config["features_config"]
        self.export_format = features_config.get("export_format", ".csv")

        feature_list_str = features_config.get("feature_list", "OCC, DUR, COV")
        self.feature_list = [x.strip() for x in feature_list_str.split(",")]

        feature_mode_str = features_config.get("feature_mode", "averaged")
        self.feature_mode = [x.strip() for x in feature_mode_str.split(",")]

        feature_types_str = features_config.get("feature_types", "real")
        self.feature_types = [x.strip() for x in feature_types_str.split(",")]

        # Aggregation method for per-segment microstate durations.
        # Accepted values: arithmetic | geometric | median | trimmed_mean.
        # ``geometric`` (default) uses the geometric mean of run lengths and
        # is robust to long-tail outliers; ``arithmetic`` uses the mean of
        # run lengths with the (N-1)/fs interval convention and is
        # algebraically consistent with COV and OCC.
        self.duration_method = features_config.get(
            "duration_method", "geometric"
        ).strip()
        if self.duration_method not in (
            "arithmetic", "geometric", "median", "trimmed_mean",
        ):
            self.logger.warning(
                "FEATURES",
                f"Unknown duration_method={self.duration_method!r}; "
                f"falling back to 'geometric'.",
            )
            self.duration_method = "geometric"
        # Re-instantiate the coordinator if it was already created (e.g. when
        # ``load_config`` is called after ``__init__``).
        if hasattr(self, "comet_feature_extractor") and self.comet_feature_extractor is not None:
            self.comet_feature_extractor.duration_method = self.duration_method

        if "OCC" in self.feature_list:
            try:
                self.sliding_window_size = features_config.getint(
                    "sliding_window_size", features_config.getint("window_size", 1)
                )
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
            "DUR_SD": "Duration Variability (ms)",
            "DUR_RMSSD": "Duration Irregularity (ms)",
            "COV_SD": "Coverage Variability (%)",
            "COV_RMSSD": "Coverage Irregularity (%)",
            "OCC_SD": "Occurrence Variability (Hz)",
            "OCC_RMSSD": "Occurrence Irregularity (Hz)",
        }

        # Source Localization Configs
        source_config = self.config["source_config"]
        self.use_anatomy = source_config.get("use_anatomy", "fsaverage")
        self.bem_solver = source_config.get("bem_solver", "mne")
        self.inverse_method = source_config.get("inverse_method", "dSPM")
        try:
            self.n_permutations = source_config.getint(
                "n_permutations", source_config.getint("nperm", 2000)
            )
        except (ValueError, TypeError):
            self.n_permutations = 2000
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

        # Load clustering results if available
        if "clustering_results" in self.config:
            clustering_results = self.config["clustering_results"]
            try:
                self.best_gev = clustering_results.getfloat("best_gev", 0.0)
            except (ValueError, TypeError):
                self.best_gev = 0.0
            
            try:
                self.best_residual = clustering_results.getfloat("best_residual", np.inf)
            except (ValueError, TypeError):
                self.best_residual = np.inf

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
        # Create clustering results directory and set microstate_maps_path
        clustering_results_path = self.get_clustering_results_path()
        os.makedirs(clustering_results_path, exist_ok=True)
        self.microstate_maps_path = os.path.join(clustering_results_path, "microstate_maps.csv")
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
        """Locate EEG file paths, with BIDS support."""
        # Check if load_all_files attribute exists, set it to True if not
        if not hasattr(self, "load_all_files"):
            self.load_all_files = True

        self.pattern = "*" if self.load_all_files else "*" + self.pattern_content + "*"
        
        # Check if we should exclude derivatives folder for BIDS raw data
        exclude_derivatives = (
            hasattr(self, 'bids_dataset') and self.bids_dataset and
            hasattr(self, 'bids_choice') and self.bids_choice == 'raw'
        )
        
        # Use regular data loading (input_folder is already set correctly for BIDS)
        self.list_eegs_path, self.list_eegs = self.comet_data_io.find_data(
            input_folder=self.input_folder, 
            extension=self.extension, 
            pattern=self.pattern,
            exclude_derivatives=exclude_derivatives
        )
        
        # Log BIDS info if applicable
        if hasattr(self, 'bids_dataset') and self.bids_dataset:
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                bids_type = "Derivatives" if hasattr(self, 'bids_choice') and self.bids_choice == 'derivative' else "Raw"
                self.LogWindow.append_log(
                    f"Loaded BIDS {bids_type} data: {len(self.list_eegs_path)} files found", 
                    log_type="info"
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
            self.best_maps, self.microstate_labels = self.comet_microstate_io.load_microstates(
                self.microstate_maps_path
            )
            return True
        except Exception as err:
            raise ValueError(f"Error loading microstate maps: {err}") from err

    def check_channels_to_remove(self):
        """Check the consistency of EEG channels across all data."""
        consistent_channels, missing_channels = self.comet_data_io.check_channels_to_remove(
            self.list_eegs_path, data_type=self.data_type, montage=self.montage
        )
        
        # Convert chan2rm to list if it's a string
        if isinstance(self.channels_to_remove, str):
            current_channels = [ch.strip() for ch in self.channels_to_remove.split(",") if ch.strip()] if self.channels_to_remove else []
        elif isinstance(self.channels_to_remove, list):
            current_channels = self.channels_to_remove
        else:
            current_channels = []
        
        # Combine current channels with missing channels
        self.channels_to_remove = list(set(current_channels).union(set(missing_channels)))
        self.ch_names = consistent_channels

    def save_eeg_info(self, eeg_info_path):
        """Save EEG information to a binary file using pickle.

        Args:
            eeg_info_path (str): Path where the EEG information will be saved
        """
        # Use captured processed EEG info if available (preserves digitization points)
        if hasattr(self, "_processed_eeg_info_captured") and self._processed_eeg_info_captured is not None:
            eeg_info = self._processed_eeg_info_captured.copy()
            eeg_info["description"] = self.study_name
            
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                dig_count = len(eeg_info['dig']) if eeg_info.get('dig') else 0
                self.LogWindow.append_log(f"Saving EEG info with {dig_count} digitization points from processed data", log_type="info")
        else:
            eeg_info = mne.create_info(
                ch_names=self.ch_names, ch_types=["eeg"] * len(self.ch_names), sfreq=self.sampling_rate
            )
            eeg_info["description"] = self.study_name

            # Set montage with better error handling and channel compatibility
            if hasattr(self, "montage") and self.montage:
                try:
                    montage = self.comet_data_io.load_montage(self.montage)
                    
                    # Filter montage to only include channels that exist in our data
                    temp_info = eeg_info.copy()
                    temp_info.set_montage(montage, match_case=False, on_missing="ignore")
                    
                    # Verify we have digitization points
                    if temp_info.get('dig') is not None and len(temp_info['dig']) > 0:
                        eeg_info = temp_info
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(f"Applied montage '{self.montage}' with {len(temp_info['dig'])} digitization points")
                    else:
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(f"Warning: No digitization points found after applying montage '{self.montage}'")
                            
                except Exception as e:
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(f"Failed to apply montage '{self.montage}': {e}")

        mne.io.write_info(eeg_info_path, eeg_info)

    def load_eeg_info(self):
        """Load EEG info from file instead of keeping it in memory."""
        self.eeg_info = mne.io.read_info(self.eeg_info_path)

    def detect_existing_processing_results(self):
        """Detect and update processing flags based on existing files in the study directory.
        
        This is useful when loading a study to ensure flags match actual data availability,
        especially for partial processing results.
        """
        # Helper to log messages safely (logger might not be available during early initialization)
        def safe_log(message):
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.processing_info("STUDY_LOADING", message)
        
        # Check for existing STC files (source localization results)
        if hasattr(self, "localized_sources_path"):
            stc_path = os.path.join(self.localized_sources_path, "stc")
            if os.path.exists(stc_path):
                successful_stc_files = self._count_successful_source_localizations(stc_path)
                if successful_stc_files > 0:
                    # Update flag if we have at least some STC files
                    if not self.done_source_localization:
                        self.done_source_localization = True
                        safe_log(f"Detected {successful_stc_files} existing source-localized files")
        
        # Check for existing correlation results
        if hasattr(self, "localized_sources_path") and self.done_source_localization:
            # Check TESS results
            tess_path = os.path.join(self.localized_sources_path, "tess_sources")
            if os.path.exists(tess_path):
                successful_tess = self._count_successful_correlations(tess_path)
                if successful_tess > 0 and not self.done_identifying_microstate_sources:
                    self.done_identifying_microstate_sources = True
                    safe_log(f"Detected {successful_tess} existing TESS correlation files")
            
            # Check AVG results
            avg_path = os.path.join(self.localized_sources_path, "avg_sources")
            if os.path.exists(avg_path):
                successful_avg = self._count_successful_correlations(avg_path)
                if successful_avg > 0 and not self.done_identifying_microstate_sources:
                    self.done_identifying_microstate_sources = True
                    safe_log(f"Detected {successful_avg} existing AVG correlation files")

    def preprocess_eeg(self, eeg_path, eeg_name, worker=None):
        """Preprocess a single EEG file. If the optional worker argument is supplied and its
        stopped flag is set, the function returns immediately so that the thread can
        terminate quickly when the user presses the STOP button.
        """
        # Early-exit if the user requested cancellation
        if worker is not None and getattr(worker, "stopped", False):
            return

        # Load EEG data
        eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, data_type=self.data_type)

        # Log file processing start
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            file_info = f"{eeg_name} | Channels: {len(eeg.ch_names)} | Duration: {eeg.times[-1]:.1f}s | Sampling Rate: {eeg.info['sfreq']}Hz"
            self.LogWindow.append_log(f"Processing {file_info}", log_type="file")

        # Apply montage if specified and track for channel filtering
        montage_obj = None
        if hasattr(self, "montage") and self.montage:
            montage_obj = self.comet_data_io.load_montage(self.montage)
            eeg.set_montage(montage_obj, match_case=False, on_missing="warn")

        # Remove channels if specified
        if hasattr(self, "channels_to_remove") and self.channels_to_remove:
            # Handle both string (comma-separated) and list formats
            if isinstance(self.channels_to_remove, str):
                channels_to_remove = [ch.strip() for ch in self.channels_to_remove.split(",") if ch.strip()]
            elif isinstance(self.channels_to_remove, list):
                channels_to_remove = [ch.strip() for ch in self.channels_to_remove if ch and str(ch).strip()]
            else:
                channels_to_remove = []
            
            if channels_to_remove:
                eeg.drop_channels(channels_to_remove, on_missing="ignore")

        # --- NEW: Extract event information and store for later use ---
        try:
            # Try to get events from annotations first
            events, event_id = mne.events_from_annotations(eeg)
        except Exception:
            events, event_id = None, {}
            
        # If no events found in annotations and this is a BIDS dataset, try to find BIDS events
        if (events is None or len(events) == 0) and hasattr(self, 'bids_dataset') and self.bids_dataset:
            bids_events_file = self._find_bids_events_file(eeg_path, eeg_name)
            if bids_events_file:
                try:
                    import pandas as pd
                    # Load BIDS events file
                    events_df = pd.read_csv(bids_events_file, sep='\t')
                    
                    if 'onset' in events_df.columns and 'trial_type' in events_df.columns:
                        # Convert BIDS events to MNE annotations
                        onset = events_df['onset'].values
                        duration = events_df.get('duration', [0] * len(onset)).values
                        description = events_df['trial_type'].values
                        
                        # Create annotations from BIDS events
                        annotations = mne.Annotations(onset=onset, duration=duration, description=description)
                        eeg.set_annotations(annotations)
                        
                        # Get events from the new annotations
                        events, event_id = mne.events_from_annotations(eeg)
                        
                        if hasattr(self, "LogWindow") and self.LogWindow is not None:
                            self.LogWindow.append_log(
                                f"Loaded BIDS events for {eeg_name}: {list(event_id.keys())}",
                                log_type="info"
                            )
                            
                except Exception as e:
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            f"Failed to load BIDS events for {eeg_name}: {str(e)}",
                            log_type="warning"
                        )
        
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
            apply_filter=self.temporal_filter_data,
            filter_method=self.filter_method,
            low_cut=self.lowcut_freq,
            high_cut=self.highcut_freq,
            apply_downsample=self.downsample_data,
            sampling_rate=self.sampling_rate,
            apply_spatial_smooth=self.spatial_filter_data,
            select_events_only=getattr(self, "select_events_only", False),
            selected_event_label=getattr(self, "selected_event_label", None),
            data_type=self.data_type,
        )

        # Store the processed EEG info from the first file to preserve digitization points
        if not hasattr(self, "_processed_eeg_info_captured") or self._processed_eeg_info_captured is None:
            self._processed_eeg_info_captured = preprocessed_eeg.info.copy()
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                dig_count = len(self._processed_eeg_info_captured['dig']) if self._processed_eeg_info_captured.get('dig') else 0
                self.LogWindow.append_log(f"Captured EEG info with {dig_count} digitization points from {eeg_name}", log_type="info")

        # Save preprocessed data
        name = os.path.splitext(eeg_name)[0]
        save_path = os.path.join(self.preprocessed_data_path, name)
        self.comet_data_io.export_eegs(
            eeg=preprocessed_eeg,
            save_path=save_path,
            extension=self.extension,
            data_type=self.data_type,
        )

        # Log successful processing
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.append_log(f"Successfully preprocessed {eeg_name}", log_type="success")

    def _find_bids_events_file(self, eeg_path, eeg_name):
        """Find the corresponding BIDS events file for an EEG file.
        
        Args:
            eeg_path (str): Full path to the EEG file
            eeg_name (str): Name of the EEG file
            
        Returns:
            str or None: Path to events file if found, None otherwise
        """
        try:
            # Get the directory containing the EEG file
            eeg_dir = os.path.dirname(eeg_path)
            
            # Extract the base name without extension
            eeg_stem = os.path.splitext(eeg_name)[0]
            
            # Look for events file with same base name
            events_file = os.path.join(eeg_dir, f"{eeg_stem}_events.tsv")
            if os.path.exists(events_file):
                return events_file
            
            # If in derivatives, also check the corresponding raw data folder
            if 'derivatives' in eeg_path and hasattr(self, 'bids_choice') and self.bids_choice == 'derivative':
                # Try to find the original BIDS root
                parts = eeg_path.split(os.sep)
                if 'derivatives' in parts:
                    deriv_idx = parts.index('derivatives')
                    # Reconstruct path to raw data
                    raw_parts = parts[:deriv_idx] + parts[deriv_idx+2:]  # Skip 'derivatives' and next folder
                    raw_eeg_dir = os.path.join(*raw_parts[:-1])  # Remove filename
                    raw_events_file = os.path.join(raw_eeg_dir, f"{eeg_stem}_events.tsv")
                    if os.path.exists(raw_events_file):
                        return raw_events_file
            
            return None
        except Exception:
            return None

    def get_preprocessed_eeg(self, subject_name):
        """Load a preprocessed EEG file on demand instead of keeping it in memory."""
        # Handle .auto extension by looking for actual saved file
        if self.extension == ".auto":
            # Try common extensions in order of preference
            possible_extensions = [".set", ".vhdr", ".edf"]
            for ext in possible_extensions:
                eeg_path = os.path.join(self.preprocessed_data_path, f"{subject_name}{ext}")
                if os.path.exists(eeg_path):
                    return self.comet_data_io.load_eeg(eeg_path=eeg_path, data_type=self.data_type)
            
            # If no file found with standard extensions, try to find any matching file
            pattern = f"{subject_name}.*"
            files = glob.glob(os.path.join(self.preprocessed_data_path, pattern))
            if files:
                # Use the first matching file
                return self.comet_data_io.load_eeg(eeg_path=files[0], data_type=self.data_type)
            else:
                raise FileNotFoundError(
                    f"No preprocessed EEG file found for {subject_name} in {self.preprocessed_data_path}"
                )
        else:
            # Use the specified extension
            eeg_path = os.path.join(self.preprocessed_data_path, f"{subject_name}{self.extension}")
            return self.comet_data_io.load_eeg(eeg_path=eeg_path, data_type=self.data_type)

    def compute_gev_all_data(self):
        """Compute Global Explained Variance for all data."""
        # normalize=False keeps raw amplitudes so GEV's GFP² weighting is
        # meaningful (unit-normalized topographies would flatten GFP).
        all_data, _ = self.comet_data_initializer.generate_maps_and_peaks(
            preprocessed_folder=self.preprocessed_data_path,
            extension=self.extension,
            data_type=self.data_type,
            data_percentage=100,
            normalize=False,
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
                maps_to_use=self.maps_to_use, n_states=self.n_maps, initializer=self.initializer
            )
        else:
            # TAAHC is deterministic and time-consuming - MUST use only 1 repetition
            original_repeats = self.n_repeats
            self.n_repeats = 1
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
            if self.clustering_method == "Modified K-Means Clustering":
                maps_init, residual_init = self.comet_microstate_clusterer.modified_kmeans(
                    data=self.maps_to_use,
                    initial_maps=initial_maps,
                    worker=worker,  # Pass worker to check stopped flag
                    repetition_num=f"{init + 1}/{self.n_repeats}",  # Pass repetition number
                )
            elif self.clustering_method == "Modified K-Means Clustering with Spatial Similarity":
                maps_init, residual_init = (
                    self.comet_microstate_clusterer.modified_kmeans_similarity(
                        data=self.maps_to_use,
                        initial_maps=initial_maps,
                        metric=self.similarity_metric,
                        worker=worker,  # Pass worker to check stopped flag
                        repetition_num=f"{init + 1}/{self.n_repeats}",  # Pass repetition number
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
                    data=self.maps_to_use,
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
            # Compute Global Explained Variance on entire dataset
            # Temporarily store the maps for GEV calculation
            temp_best_maps = self.best_maps
            self.best_maps = maps_init
            gev_init = self.compute_gev_all_data()
            self.best_maps = temp_best_maps

            # Log iteration results
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                if is_taahc:
                    log_message = (
                        f"TAAHC Clustering Completed\n"
                        f"✓ Hierarchical clustering processed {self.maps_to_use.shape[1]} timepoints\n"
                        f"✓ Global Explained Variance: {100 * gev_init:.3f}%"
                    )
                else:
                    log_message = (
                        f"Data Clustered [{init + 1}/{self.n_repeats}]\n"
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
                
                # Update microstate_labels to match the new number of states
                self.load_maps()

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

            # Ensure microstate_labels are updated for the new number of states
            try:
                self.load_maps()
            except Exception as e:
                self.logger.warning("CLUSTERING", f"Could not load maps after clustering stop: {e}")
                # Generate default labels if loading fails
                self.microstate_labels = [f"Microstate {i+1}" for i in range(self.n_maps)]

            # Mark clustering as completed (even if stopped early)
            self.done_clustering = True

            # Log stop with current best results
            stop_message = (
                f"❌  [CLUSTERING] Stop Requested - Please Wait ...\n"
                f"⚠️ Clustering stopped by user\n"
                f"✓ Saved best maps found so far: {self.n_maps} microstates "
                f"(GEV: {100 * self.best_gev:.3f}%%)"
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

        eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, data_type=self.data_type)
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
                    self.filter_segments_less_than_ms / (1000 / self.sampling_rate)
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
            preprocessing_steps.append(f"Downsampling to {self.sampling_rate} Hz")
        if getattr(self, "prep_data", False):
            preprocessing_steps.append("Bad channel detection / interpolation")

        # Log study information and settings
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Study information
            study_info = {
                "Study Name": self.study_name,
                "Input Directory": self.input_folder,
                "Output Directory": self.save_dir,
                "Files Found": f"{len(self.list_eegs_path)} {self.data_type} EEG files with {self.extension} extension",
            }
            self.logger.settings_info("PREPROCESSING", study_info)

            # Log data selection mode between Files Found and preprocessing settings
            if getattr(self, "select_events_only", False) and getattr(self, "selected_event_label", None):
                selection_message = f"Data Selection: Only segments with event '{self.selected_event_label}'"
            else:
                selection_message = "Data Selection: Entire recording"
            self.logger.processing_info("PREPROCESSING", selection_message)

            # Preprocessing settings
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
                f"{self.sampling_rate} Hz" if self.downsample_data else "Disabled"
            )
            preprocessing_settings["Channels to Remove"] = (
                str(self.channels_to_remove) if self.channels_to_remove else "None"
            )

            self.logger.settings_info("PREPROCESSING", preprocessing_settings)

        # Build a list of EEG files
        self.zipped_eeg_files = list(zip(self.list_eegs_path, self.list_eegs))

        # Check channel consistency
        self.check_channels_to_remove()

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
        """Perform clustering on preprocessed EEG data.

        Supports automatic or manual k selection, TAAHC progress tracking,
        and batch processing.
        """
        # Reset microstate labeling flag since new clustering will invalidate previous labels
        self.done_microstate_labeling = False
        
        # Log section header
        self.logger.section_header("CLUSTERING")

        # Log clustering configuration
        clustering_settings = {}
        clustering_settings["Clustering Method"] = getattr(self, "clustering_method", "Unknown")
        clustering_settings["Number of Maps"] = getattr(self, "n_maps", "Unknown")
        clustering_settings["Number of Repeats"] = getattr(self, "n_repeats", "Unknown")
        # Note: K Range and Stopping Mode are logged separately below for better formatting

        # Start clustering
        self.logger.processing_start("CLUSTERING", "Starting Microstate Clustering")
        
        # Add review references
        self.logger.processing_info("REVIEW", "    https://doi.org/10.1016/j.neubiorev.2014.12.010 ")
        self.logger.processing_info("REVIEW", "    https://doi.org/10.1016/j.neuroimage.2017.11.062")
        
        # Log optimization method information if using auto-k selection
        if self.n_maps == "auto":
            if getattr(self, 'stopping_mode', 'majority_vote') == "majority_vote":
                self.logger.processing_info("CLUSTERING", "Optimization Strategy: Ensemble method (majority vote across 10 criteria)")
                self.logger.processing_info("CLUSTERING", "Methods: GEV, Davies-Bouldin, Cross-Validation, Krzanowski-Lai, Silhouette, Dunn, Calinski-Harabasz, Gap, AIC, BIC")
            else:
                # Single method optimization
                method_descriptions = {
                    "gev": "Global Explained Variance (elbow point detection)",
                    "db": "Davies-Bouldin Index (cluster distinctiveness)", 
                    "cv": "Cross-Validation (explanatory power vs parsimony)",
                    "kl": "Krzanowski-Lai Criterion (relative improvement)",
                    "sil": "Silhouette Coefficient (clustering consistency)",
                    "dunn": "Dunn Index (cluster compactness and separation)",
                    "ch": "Calinski-Harabasz Index (variance ratios)",
                    "gap": "Gap Statistic (comparison to random distributions)",
                    "aic": "Akaike Information Criterion (model selection)",
                    "bic": "Bayesian Information Criterion (model selection)"
                }
                method_name = method_descriptions.get(getattr(self, 'stopping_mode', 'gev'), f"Single method: {getattr(self, 'stopping_mode', 'gev')}")
                self.logger.processing_info("CLUSTERING", f"Optimization Strategy: {method_name}")
            
            # Add key references for optimization methods
            self.logger.reference("CLUSTERING", "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x")  # Pascual-Marqui CV
            self.logger.reference("CLUSTERING", "https://doi.org/10.1016/0031-3203(87)90066-7")  # Silhouette
        
        # Note: clustering_settings will be logged later right before actual clustering starts

        # Provide concise clustering details
        try:
            # K range and input details
            if self.n_maps == "auto":
                kmin = getattr(self, 'k_min', 2)
                kmax = getattr(self, 'k_max', 10)
                
                # Determine optimization mode description
                if getattr(self, 'stopping_mode', 'majority_vote') == "majority_vote":
                    opt_mode = "ensemble voting"
                else:
                    opt_mode = f"single method ({getattr(self, 'stopping_mode', 'gev')})"
                
                self.logger.processing_info("CLUSTERING", f"K Range: {kmin} to {kmax} (optimization: {opt_mode})")
                self.logger.processing_info("CLUSTERING", "Clustering Input: GFP peaks (auto-k enforced)")
            else:
                self.logger.processing_info("CLUSTERING", f"K Value: {self.n_maps} (user-defined)")
                
                # Input selection message for user-defined k
                use_pct = getattr(self, "data_percentage", None)
                if use_pct is None:
                    input_msg = "Clustering Input: GFP peaks only"
                elif use_pct >= 100:
                    input_msg = "Clustering Input: Entire recordings"
                else:
                    input_msg = f"Clustering Input: Random {use_pct}% of data per repeat"
                self.logger.processing_info("CLUSTERING", input_msg)

            # Effective repetitions (simplified)
            is_taahc = (
                getattr(self, "clustering_method", "")
                == "Topographic Atomize and Agglomerate Hierarchical Clustering"
            )
            effective_repeats = 1 if is_taahc else getattr(self, "n_repeats", 1)
            
            if self.n_maps == "auto":
                self.logger.processing_info("CLUSTERING", "Auto-k selection: Using single repeat (n_repeats=1) for optimization")
            else:
                self.logger.processing_info("CLUSTERING", f"Repeating analysis {effective_repeats} times")
                
        except Exception:
            # Logging should never break the flow
            pass

        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            # Calculate total steps for clustering - only count clustering repetitions
            clustering_steps = self.n_repeats  # One step per repetition

            # Use setup_progress_dialog for worker thread with correct total steps
            self.LogWindow.setup_progress_dialog(
                window_title="Microstate Clustering ...",
                label_text="Initializing clustering process ...",
                tasks=clustering_steps,  # Pass only clustering steps
                processing_func=self._run_full_clustering_worker,
            )

            # Set callback to handle completion
            self.LogWindow.process_finished_callback = self._on_clustering_completed
        else:
            # Non-GUI mode - run directly
            self._run_full_clustering_direct()

    def get_clustering_results_path(self):
        """Get the path to the clustering results directory.
        
        Returns:
            str: Path to the clustering results directory.
        """
        return os.path.join(self.save_dir, f"{self.study_name}_clustering_results")

    def _save_clustering_results(self):
        """Save clustering results to files."""
        try:
            # Create clustering results directory with proper naming convention
            clustering_results_path = self.get_clustering_results_path()
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
            self.config["clustering_results"]["n_maps"] = str(self.n_maps)
            self.config["clustering_results"]["clustering_method"] = self.clustering_method

        except Exception as e:
            self.logger.error("CLUSTERING", f"Failed to save clustering results: {str(e)}")

    def _run_full_clustering_worker(self, _task_name, worker=None):
        """Worker function for running the entire clustering process.
        This method is designed to be called by the LogWindow's worker thread.
        
        Args:
            _task_name: Task name (ignored for step-based processing)
            worker: Worker thread instance for stop checking
        """
        # Ignore task_name parameter for step-based processing
        try:
            # Progress tracking variables - only count clustering repetitions
            total_steps = self.n_repeats  # Only clustering repetitions
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

                # Always emit progress update after updating current_step
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

            # Step 2: Calculate minimum distance size if smoothing GFP is enabled
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.log_clustering_setup_step("Calculating parameters")
            if check_stop():
                return self._handle_stopped_clustering("Calculating parameters")
            if self.smoothing_gfp:
                self.min_distance_size = int(
                    int(self.smoothing_distance) / (1000 / int(self.sampling_rate))
                )
            else:
                self.min_distance_size = None

            # Step 3: Check if clustering method is supported
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.log_clustering_setup_step(
                    "Validating clustering method", self.clustering_method
                )
            available_methods = [
                "Modified K-Means Clustering",
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
                None if self.n_maps == "auto" else self.data_percentage
            )

            self.maps_to_use, peaks = self.comet_data_initializer.generate_maps_and_peaks(
                preprocessed_folder=self.preprocessed_data_path,
                extension=self.extension,
                data_type=self.data_type,
                data_percentage=auto_k_use_percentages,
                min_dist=self.min_distance_size,
            )

            # Step 6: Handle automatic k selection
            if self.n_maps == "auto":
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.log_clustering_setup_step(
                        "Starting automatic optimization", f"k range: {self.k_min}-{self.k_max}"
                    )
                    # Add the processing message with ⌛ emoji
                    self.LogWindow.append_log(
                        f"⌛ Identifying {self.k_max - self.k_min + 1} optimal microstate maps ..."
                    )
                if check_stop():
                    return self._handle_stopped_clustering("Starting automatic optimization")

                # Run automatic optimization with progress updates
                optimization_success = self._run_automatic_optimization_core_with_progress(update_progress, check_stop)
                
                # If optimization was stopped or failed, reset state and return early
                if not optimization_success:
                    self.logger.warning("CLUSTERING", "❌ Auto-k optimization was stopped or failed")
                    
                    # Reset clustering state to allow restart
                    self.done_clustering = False
                    
                    # Clear any partial results
                    self.best_maps = None
                    self.best_gev = 0.0
                    self.best_residual = np.inf
                    
                    # Update UI message and call callback to reset main window
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log("🔄 Clustering ready to restart with new parameters", log_type="info")
                        self.LogWindow.set_progress_label("Ready for clustering")
                        self.LogWindow.set_progress_stop_enabled(False)
                    
                    # Call clustering completion callback to update main window UI
                    if self.clustering_completed_callback is not None:
                        self.clustering_completed_callback()
                    
                    return False

            # Step 7: After determining number_of_maps, perform actual clustering
            if self.n_maps and self.n_maps != "auto":
                # Log clustering method and settings right before actual clustering starts
                clustering_settings = {
                    "Clustering Method": getattr(self, "clustering_method", "Unknown"),
                    "Number of Maps": str(self.n_maps),
                    "Number of Repeats": str(getattr(self, "n_repeats", "Unknown"))
                }
                self.logger.settings_info("CLUSTERING", clustering_settings)
                
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.log_clustering_setup_step(
                        "Starting clustering",
                        f"{self.n_maps} maps, {self.n_repeats} repetitions",
                    )
                if check_stop():
                    return self._handle_stopped_clustering("Starting clustering repetitions")

                # Initialize microstate clusterer with special settings for TAAHC
                if is_taahc:
                    # TAAHC is deterministic and time-consuming - MUST use only 1 repetition
                    clustering_repeats = 1
                    if self.n_repeats != 1:
                        self.logger.warning(
                            "CLUSTERING",
                            f"TAAHC: Enforcing 1 repetition (deterministic algorithm) - ignoring user setting of {self.n_repeats}",
                        )
                else:
                    clustering_repeats = self.n_repeats

                self.comet_microstate_clusterer = MicrostateClusterer(
                    n_states=self.n_maps,
                    batch_size=self.batch_size,
                    n_repeats=1,  # We'll handle repetitions manually
                    max_iterations=self.max_iterations,
                    clustering_tolerance=self.clustering_tolerance,
                )

                # Perform clustering repetitions to find best solution
                best_maps = None
                best_gev = 0.0
                best_residual = np.inf

                if is_taahc:
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            "⌛ Running TAAHC clustering (deterministic algorithm) ..."
                        )
                else:
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            f"⌛ Running {clustering_repeats} clustering repetitions ..."
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

                    # For random subset mode, generate new random samples for each repeat
                    if (auto_k_use_percentages is not None and 
                        auto_k_use_percentages < 100 and 
                        not is_taahc):
                        # Check for stop before expensive data loading operation
                        if check_stop():
                            return self._handle_stopped_clustering(
                                f"Random data sampling for repetition {init + 1}/{clustering_repeats}",
                                best_maps,
                                best_gev,
                                best_residual,
                                completed_repetitions,
                            )
                        
                        # Generate new random subset for this repeat
                        temp_maps2use, temp_peaks = self.comet_data_initializer.generate_maps_and_peaks(
                            preprocessed_folder=self.preprocessed_data_path,
                            extension=self.extension,
                            data_type=self.data_type,
                            data_percentage=auto_k_use_percentages,
                            min_dist=self.min_distance_size,
                            random_seed=42 + init,  # Different seed for each repeat
                        )
                        # Temporarily store original maps2use
                        original_maps2use = self.maps_to_use
                        self.maps_to_use = temp_maps2use
                        
                        # Perform clustering with new random subset
                        maps, gev, residual = self.cluster_eeg_microstates(init, worker=worker)
                        
                        # Restore original maps2use
                        self.maps_to_use = original_maps2use
                    else:
                        # Use the same data for all repeats (GFP peaks or entire dataset)
                        maps, gev, residual = self.cluster_eeg_microstates(init, worker=worker)

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
                                "CLUSTERING", f"TAAHC clustering completed - GEV: {100 * self.best_gev:.3f}%"
                            )
                        else:
                            self.logger.processing_info(
                                "CLUSTERING", f"New best GEV: {100 * self.best_gev:.3f}%"
                            )

                # Step 8: Store best results
                if best_maps is not None:
                    self.best_maps = best_maps
                    self.best_gev = best_gev
                    self.best_residual = best_residual

                    # Ensure microstate_labels are updated for the new number of states
                    try:
                        self.load_maps()
                    except Exception as e:
                        self.logger.warning("CLUSTERING", f"Could not load maps after clustering: {e}")
                        # Generate default labels if loading fails
                        self.microstate_labels = [f"Microstate {i+1}" for i in range(self.n_maps)]

                    self.logger.processing_info(
                        "CLUSTERING",
                        f"Best GEV: {100 * self.best_gev:.3f}%, Best residual: {self.best_residual:.6f}",
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

            # Ensure microstate_labels are updated for the new number of states
            try:
                self.load_maps()
            except Exception as e:
                self.logger.warning("CLUSTERING", f"Could not load maps after partial clustering: {e}")
                # Generate default labels if loading fails
                self.microstate_labels = [f"Microstate {i+1}" for i in range(self.n_maps)]

            partial_msg = f"Partial results saved from {completed_repetitions} completed clustering repetitions"
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(partial_msg, log_type="info")
                self.LogWindow.append_log(f"Best GEV from partial results: {100 * self.best_gev:.3f}%")

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
        return self._run_full_clustering_worker("full_clustering", worker=None)

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
        clustering_results_path = self.get_clustering_results_path()
        self.comet_clusterer_optimizer = ClustererOptimizer(
            maps_to_use=self.maps_to_use,
            min_dist=self.min_distance_size,
            n_repeats=self.n_repeats,
            k_min=self.k_min,
            k_max=self.k_max,
            preprocessed_data_path=self.preprocessed_data_path,
            extension=self.extension,
            data_type=self.data_type,
            clustering_tolerance=self.clustering_tolerance,
            max_iterations=self.max_iterations,
            progress_callback=progress_callback,
            logger=self.logger,
            clustering_results_path=clustering_results_path,
        )

        # Run automatic optimization
        self._run_automatic_optimization_core()

    def _run_automatic_optimization_direct(self):
        """Run automatic optimization directly (non-GUI mode)."""
        # Initialize optimizer without progress callback
        # For auto-k selection, use only 1 repeat and force GFP peaks
        auto_k_n_inits = 1  # Force single repeat for auto-k selection

        clustering_results_path = self.get_clustering_results_path()
        self.comet_clusterer_optimizer = ClustererOptimizer(
            maps_to_use=self.maps_to_use,
            min_dist=self.min_distance_size,
            n_repeats=auto_k_n_inits,  # Use single repeat for auto-k
            k_min=self.k_min,
            k_max=self.k_max,
            preprocessed_data_path=self.preprocessed_data_path,
            extension=self.extension,
            data_type=self.data_type,
            clustering_tolerance=self.clustering_tolerance,
            max_iterations=self.max_iterations,
            logger=self.logger,
            clustering_results_path=clustering_results_path,
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
                    optimization_steps = self.k_max - self.k_min + 1
                    # Calculate the current step within the optimization phase
                    optimization_current_step = setup_steps + int(
                        (_current / _total) * optimization_steps
                    )
                    # Pass the current step (not percentage)
                    update_progress_func(optimization_current_step, message)

            # Initialize optimizer with progress callback
            # For auto-k selection, use only 1 repeat and force GFP peaks
            auto_k_n_inits = 1  # Force single repeat for auto-k selection

            clustering_results_path = self.get_clustering_results_path()
            self.comet_clusterer_optimizer = ClustererOptimizer(
                maps_to_use=self.maps_to_use,
                min_dist=self.min_distance_size,
                n_repeats=auto_k_n_inits,  # Use single repeat for auto-k
                k_min=self.k_min,
                k_max=self.k_max,
                preprocessed_data_path=self.preprocessed_data_path,
                extension=self.extension,
                data_type=self.data_type,
                clustering_tolerance=self.clustering_tolerance,
                max_iterations=self.max_iterations,
                progress_callback=progress_callback,
                logger=self.logger,
                clustering_results_path=clustering_results_path,
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
                # Single method optimization
                optimal_k, k_values, scores = self.comet_clusterer_optimizer.find_optimal_k(
                    optimizer_mode=self.stopping_mode, parameter_value=self.stopping_threshold
                )
                
                # Log to both console and GUI window
                self.logger.processing_success(
                    "CLUSTERING", f"Optimal k selected: {optimal_k}"
                )
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(f"Optimal number of maps determined: {optimal_k}")

            self.n_maps = optimal_k

            # Return True to indicate successful completion
            return True

        except RuntimeError as e:
            if "stopped by user" in str(e):
                # Handle user-initiated stop
                stop_msg = "⏹️ Auto-k optimization stopped by user"
                self.logger.warning("CLUSTERING", stop_msg)
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(stop_msg, log_type="warning")
                    self.LogWindow.append_log("💡 You can adjust parameters and restart clustering", log_type="info")
                
                # Reset to allow user to try again - don't set a default number_of_maps
                # Keep it as "auto" so user can restart optimization
                self.choose_number_of_maps = "user"  # Reset to user mode to prevent auto-retry
                self.n_maps = 4  # Fallback default
                
                # Clean up optimizer reference
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.current_optimizer = None
                    self.LogWindow.current_step = None
                
                self.logger.warning(
                    "CLUSTERING", f"Reset to user-defined mode with k={self.n_maps} (you can change this)"
                )
                
                # Return False to indicate optimization was stopped
                return False
            else:
                raise
        except Exception as e:
            error_msg = f"Automatic optimization failed: {str(e)}"
            self.logger.error("CLUSTERING", error_msg)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")
                self.LogWindow.append_log("💡 You can adjust parameters and restart clustering", log_type="info")
            
            # Clean up optimizer reference
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.current_optimizer = None
                self.LogWindow.current_step = None
            
            # Fallback to default
            self.choose_number_of_maps = "user"  # Reset to user mode 
            self.n_maps = 4
            self.logger.warning(
                "CLUSTERING", f"Reset to user-defined mode with k={self.n_maps} (you can change this)"
            )
            
            # Return False to indicate optimization failed
            return False

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
                # Single method optimization
                optimal_k, k_values, scores = self.comet_clusterer_optimizer.find_optimal_k(
                    optimizer_mode=self.stopping_mode, parameter_value=self.stopping_threshold
                )
                
                # Log to both console and GUI window
                self.logger.processing_success(
                    "CLUSTERING", f"Optimal k selected: {optimal_k}"
                )
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(f"Optimal number of maps determined: {optimal_k}")

            self.n_maps = optimal_k

        except Exception as e:
            error_msg = f"Automatic optimization failed: {str(e)}"
            self.logger.error("CLUSTERING", error_msg)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")
            # Fallback to default
            self.n_maps = 4
            self.logger.warning(
                "CLUSTERING", f"Using fallback number of maps: {self.n_maps}"
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
        microstate_labels, labels_overall_confidence, label_confidences = self.comet_microstate_labeler.do_labeling()

        # Save updated microstate maps with labels
        self.comet_microstate_io.export_microstates(
            self.best_maps, self.eeg_info, self.microstate_maps_path, headers=microstate_labels
        )

        # Update and save labels
        self.load_maps()
        self.labels_overall_confidence = labels_overall_confidence
        self.label_confidences = label_confidences  # Per-label confidence (label -> %)

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

        # Check if best maps are available
        if self.best_maps is None:
            self.logger.error("BACKFITTING", "No microstate maps available for backfitting")
            return

        # Create segmentation directory
        os.makedirs(self.segmentation_path, exist_ok=True)

        # Load preprocessed (cleaned) data paths FIRST before any processing
        # This ensures we work with data that has the same channels as the microstate maps
        self.load_clean()

        # Create an instance of the microstate backfitter
        self.comet_microstate_backfitter = MicrostateBackfitter(
            study_name=self.study_name,
            preprocessed_data_path=self.preprocessed_data_path,
            microstate_maps=self.best_maps,
            backfit_to=self.backfit_to,
            filter_segments=self.filter_segments,
            filter_segments_option=self.filter_segments_option,
            identify_short_window=self.identify_short_window,
            microstate_labels=self.microstate_labels,
            segmentation_path=self.segmentation_path,
            extension=self.extension,
            data_type=self.data_type,
            sampling_rate=self.sampling_rate,
            smoothing_parameters=[self.convergence_epsilon, self.half_window_size, self.smoothness_penalty],
            export_format=self.export_format,
            min_correlation_threshold=getattr(self, "min_correlation_threshold", False),
        )

        # Identify optimal window size if requested
        if self.identify_short_window:
            self.logger.processing_start(
                "BACKFITTING", "Identifying optimal smoothing window length"
            )

            # Use worker thread for optimal window identification to keep GUI responsive
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                # Calculate total progress steps: files + analysis + optional lambda optimization
                n_files = len(self.list_eegs_path)
                # If smoothing is selected, add steps for lambda optimization
                if (hasattr(self, "filter_segments_option") and 
                    self.filter_segments_option == "smooth"):
                    total_steps = n_files * 2 + 1  # Threshold analysis + lambda optimization + final
                else:
                    total_steps = n_files + 1  # Just threshold analysis + final
                
                # Set up progress dialog for window identification
                self.LogWindow.setup_progress_dialog(
                    window_title="Identifying Optimal Window ...",
                    label_text="Analyzing optimal parameters for segment filtering ...",
                    tasks=total_steps,
                    processing_func=self._identify_optimal_window_worker_with_progress,
                )
                # Set callback to run when window identification worker thread finishes
                self.LogWindow.process_finished_callback = self._on_window_identification_finished
            else:
                # Non-GUI mode - run directly
                self._identify_optimal_window_direct()
        else:
            if self.filter_segments:
                self.filter_segments_less_than_ms = self.filter_segments_less_than
            else:
                self.filter_segments_less_than_ms = 0

            # Continue with backfitting immediately if no window identification needed
            self._continue_backfitting_process()

    def _identify_optimal_window_worker_with_progress(self, _step_mode, worker=None):
        """Worker function for identifying optimal window length with simplified approach.
        
        Args:
            _step_mode: Unused parameter (LogWindow passes "step_based_processing")
            worker: Worker thread instance for progress updates and stop checking
        """
        if worker is not None and getattr(worker, "stopped", False):
            return

        n_files = len(self.list_eegs_path)
        
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            self.LogWindow.append_log("Starting optimal threshold analysis...", log_type="info")
        
        # Create progress callback for the backfitter
        def progress_callback(current, total, message):
            if worker is not None and getattr(worker, "stopped", False):
                return False
            
            # Update progress with actual computation status
            if hasattr(worker, 'progress_updated'):
                worker.progress_updated.emit(current, message)
                
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(message, log_type="info")
                
            return True
        
        try:
            # Load EEG data one at a time (more memory efficient)
            eeg_data_list = []
            for i, eeg_path in enumerate(self.list_eegs_path):
                if worker is not None and getattr(worker, "stopped", False):
                    return
                    
                try:
                    eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, data_type=self.data_type)
                    eeg_data_list.append(eeg)
                    
                except Exception as e:
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(f"Failed to load {eeg_path}: {str(e)}", log_type="warning")
                    continue
            
            if not eeg_data_list:
                self._optimal_threshold_result = self.filter_segments_less_than
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log("No data loaded, using default threshold", log_type="warning")
                return
            
            # Find optimal threshold for filtering short segments
            optimal_threshold = self.comet_microstate_backfitter.identify_optimal_length_filter(
                eeg_data_list, progress_callback=progress_callback
            )
            
            # Store threshold result
            self._optimal_threshold_result = optimal_threshold
            
            # Create threshold optimization plot using the first EEG file
            if eeg_data_list:
                try:
                    plot_path = self.comet_microstate_backfitter.plot_threshold_optimization(
                        eeg_data_list[0].get_data()
                    )
                    if plot_path and hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            f"📊 Threshold optimization plot saved: {plot_path}", 
                            log_type="info"
                        )
                except Exception as plot_err:
                    if hasattr(self, "LogWindow") and self.LogWindow is not None:
                        self.LogWindow.append_log(
                            f"⚠️ Could not create threshold optimization plot: {plot_err}", 
                            log_type="warning"
                        )
            
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(
                    f"✅ Optimal threshold determined: {optimal_threshold:.1f}ms", 
                    log_type="success"
                )
            
            # If smoothing is selected, also find optimal lambda (non-smoothness penalty)
            if (hasattr(self, "filter_segments_option") and 
                self.filter_segments_option == "smooth"):
                
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(
                        "Finding optimal smoothing parameter (λ) for segment filtering...", 
                        log_type="info"
                    )
                
                optimal_lambda = self.comet_microstate_backfitter.find_optimal_lambda_for_files(
                    eeg_data_list, optimal_threshold, progress_callback=progress_callback
                )
                
                # Store lambda result
                self._optimal_lambda_result = optimal_lambda
                
                if hasattr(self, "LogWindow") and self.LogWindow is not None:
                    self.LogWindow.append_log(
                        f"✅ Optimal smoothing parameter determined: λ={optimal_lambda}", 
                        log_type="success"
                    )
                
                # Final progress update with both parameters
                progress_callback(
                    n_files * 2 + 1, n_files * 2 + 1, 
                    f"Analysis complete - threshold: {optimal_threshold:.1f}ms, λ={optimal_lambda}"
                )
            else:
                # Final progress update with just threshold
                progress_callback(
                    n_files + 1, n_files + 1, 
                    f"Analysis complete - optimal threshold: {optimal_threshold:.1f}ms"
                )
            
        except Exception as e:
            # Handle any errors
            error_msg = f"Error in optimal threshold analysis: {str(e)}"
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")
            
            # Set fallback result
            self._optimal_threshold_result = self.filter_segments_less_than



    def _identify_optimal_window_direct(self):
        """Direct execution of optimal window identification using simplified method."""
        eeg_data_list = []
        
        # Load EEG data
        for eeg_path in self.list_eegs_path:
            try:
                eeg = self.comet_data_io.load_eeg(eeg_path=eeg_path, data_type=self.data_type)
                eeg_data_list.append(eeg)
            except Exception:
                continue  # Skip files that can't be loaded
        
        if eeg_data_list:
            # Find optimal threshold for filtering short segments
            self.filter_segments_less_than_ms = (
                self.comet_microstate_backfitter.identify_optimal_length_filter(eeg_data_list)
            )
            
            # Create threshold optimization plot using the first EEG file
            try:
                plot_path = self.comet_microstate_backfitter.plot_threshold_optimization(
                    eeg_data_list[0].get_data()
                )
                if plot_path:
                    self.logger.processing_info(
                        "BACKFITTING",
                        f"Threshold optimization plot saved: {plot_path}"
                    )
            except Exception as plot_err:
                self.logger.warning(
                    "BACKFITTING", 
                    f"Could not create threshold optimization plot: {plot_err}"
                )
            
            # If smoothing is selected, also find optimal lambda
            if (hasattr(self, "filter_segments_option") and 
                self.filter_segments_option == "smooth"):
                
                self.smoothness_penalty = self.comet_microstate_backfitter.find_optimal_lambda_for_files(
                    eeg_data_list, self.filter_segments_less_than_ms
                )
                
                # Log both parameters
                self.logger.processing_success(
                    "BACKFITTING",
                    f"Optimal parameters determined: {self.filter_segments_less_than_ms:.1f}ms, λ={self.smoothness_penalty}",
                )
            else:
                # Log just threshold
                self.logger.processing_success(
                    "BACKFITTING",
                    f"Optimal filter length determined: {self.filter_segments_less_than_ms:.1f}ms",
                )
        else:
            # Fallback if no data could be loaded
            self.filter_segments_less_than_ms = self.filter_segments_less_than
            self.logger.warning(
                "BACKFITTING", 
                "No EEG data could be loaded, using default filter value"
            )

        # Continue with backfitting process
        self._continue_backfitting_process()

    def _on_window_identification_finished(self, _message=None):
        """Handle completion of optimal window identification worker."""
        try:
            # Get threshold result from worker
            if hasattr(self, '_optimal_threshold_result'):
                self.filter_segments_less_than_ms = self._optimal_threshold_result
                # Clean up temporary result
                delattr(self, '_optimal_threshold_result')
            else:
                # Fallback if no result available
                self.filter_segments_less_than_ms = self.filter_segments_less_than
                self.logger.warning(
                    "BACKFITTING", 
                    "No threshold result from worker, using default filter value"
                )
            
            # Get lambda result from worker if smoothing was optimized
            if hasattr(self, '_optimal_lambda_result'):
                self.smoothness_penalty = self._optimal_lambda_result
                # Clean up temporary result
                delattr(self, '_optimal_lambda_result')
                
                # Log both results
                self.logger.processing_success(
                    "BACKFITTING",
                    f"Optimal parameters determined: {self.filter_segments_less_than_ms:.1f}ms, λ={self.smoothness_penalty}",
                )
            else:
                # Log just threshold result
                self.logger.processing_success(
                    "BACKFITTING",
                    f"Optimal filter length determined: {self.filter_segments_less_than_ms:.1f}ms",
                )

            # Clear the callback (worker cleanup handled by LogWindow)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.process_finished_callback = None

            # Continue with backfitting process
            self._continue_backfitting_process()

        except Exception as e:
            error_msg = f"Error in window identification completion: {str(e)}"
            self.logger.error("BACKFITTING", error_msg)
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.append_log(error_msg, log_type="error")
                self.LogWindow.process_finished_callback = None

            # Fallback and continue
            self.filter_segments_less_than_ms = self.filter_segments_less_than
            self._continue_backfitting_process()

    def _continue_backfitting_process(self):
        """Continue with the main backfitting process after optimal window identification."""
        # Note: load_clean() is now called at the start of run_backfitting()
        # to ensure preprocessed data paths are available for all backfitting operations
        self.zipped_eeg_files = list(zip(self.list_eegs_path, self.list_eegs))

        # Log backfitting process details before starting
        backfit_target = "GFP peaks" if self.backfit_to == "peaks" else "all time points"
        self.logger.processing_info("BACKFITTING", f"Backfitting microstates to {backfit_target}")
        
        # Log segment filtering details if enabled
        if self.filter_segments and hasattr(self, 'filter_segments_less_than_ms'):
            if self.filter_segments_option == "remove":
                self.logger.processing_info("BACKFITTING", 
                    f"Removing transient segments with durations less than {self.filter_segments_less_than_ms}ms")
            elif self.filter_segments_option == "replace_high":
                self.logger.processing_info("BACKFITTING", 
                    f"Replacing segments less than {self.filter_segments_less_than_ms}ms with nearby dominant microstates")
            elif self.filter_segments_option == "replace_half":
                self.logger.processing_info("BACKFITTING", 
                    f"Replacing segments less than {self.filter_segments_less_than_ms}ms using half-and-half method")
            elif self.filter_segments_option == "smooth":
                self.logger.processing_info("BACKFITTING", 
                    f"Smoothing segments: reject ≤ {self.filter_segments_less_than_ms}ms, half-window b={self.half_window_size}, lambda={self.smoothness_penalty}")

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

        # Log features to extract with full names and references
        self.logger.processing_info("FEATURE_EXTRACTION", "Features to Extract:")
        
        # Define feature names and references
        feature_info = {
            "OCC": ("Microstate Occurrence (OCC)", "https://doi.org/10.1016/0013-4694%2887%2990025-3"),
            "DUR": ("Microstate Duration (DUR)", "https://doi.org/10.1016/0013-4694%2887%2990025-3"),
            "COV": ("Microstate Coverage (COV)", "https://doi.org/10.1016/0013-4694%2887%2990025-3"),
            "GEV": ("Global Explained Variance (GEV)", "https://doi.org/10.1016/j.neuroimage.2012.05.060"),
            "TP": ("Transition Probability (TP)", "https://doi.org/10.1016/j.pscychresns.2004.05.007"),
            "HE": ("Hurst Exponent (HE)", "https://doi.org/10.1016/j.neuroimage.2016.07.050"),
            "ER": ("Entropy Rate (ER)", "https://doi.org/10.3389/fncom.2018.00070"),
            "LZC": ("Lempel-Ziv Complexity (LZC)", "https://doi.org/10.1038/s41598-020-74790-7"),
            "ERR": ("Entropy Representation (ERR)", "https://doi.org/10.1016/j.neuroimage.2023.120196"),
            "ROF": ("Relative Occurrence Frequency (ROF)", None),
            "RTF": ("Relative Transition Frequency (RTF)", None)
        }
        
        # Log each feature with its full name and reference
        # Group features that share the same reference
        basic_features = ["OCC", "DUR", "COV"]
        basic_features_present = [f for f in self.feature_list if f in basic_features]
        
        # Log basic features first (OCC, DUR, COV) without individual references
        for feature in basic_features_present:
            if feature in feature_info:
                feature_name, _ = feature_info[feature]
                self.logger.processing_info("FEATURE_EXTRACTION", feature_name)
        
        # Log single reference for basic features if any are present
        if basic_features_present:
            self.logger.reference("FEATURE_EXTRACTION", "https://doi.org/10.1016/0013-4694%2887%2990025-3")
        
        # Log other features with their individual references
        for feature in self.feature_list:
            if feature not in basic_features and feature in feature_info:
                feature_name, reference_url = feature_info[feature]
                self.logger.processing_info("FEATURE_EXTRACTION", feature_name)
                if reference_url:
                    self.logger.reference("FEATURE_EXTRACTION", reference_url)
        
        # Log other settings
        self.logger.processing_info("FEATURE_EXTRACTION", f"Feature Modes: {', '.join(self.feature_mode)}")
        self.logger.processing_info("FEATURE_EXTRACTION", f"Feature Types: {', '.join(self.feature_types)}")

        if "sliding" in self.feature_mode:
            self.logger.processing_info("FEATURE_EXTRACTION", f"Sliding Window Size: {self.sliding_window_size}")
            if hasattr(self, "pre_window_size"):
                self.logger.processing_info("FEATURE_EXTRACTION", f"Pre-Window Size: {self.pre_window_size}")
            if hasattr(self, "post_window_size"):
                self.logger.processing_info("FEATURE_EXTRACTION", f"Post-Window Size: {self.post_window_size}")

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
                # For epoched data, check if we need to transpose the segmentation array
                # Segmentation files store data as (timepoints, trials) but we need (trials, timepoints)
                # Apply transpose for ANY 2D array when shape suggests it needs transposing
                if len(segmentation_array.shape) == 2:
                    # Heuristic: if first dimension is larger, it's likely (timepoints, trials) and needs transpose
                    # Example: (1250 timepoints, 150 trials) should become (150 trials, 1250 timepoints)
                    if segmentation_array.shape[0] > segmentation_array.shape[1]:
                        segmentation_array = segmentation_array.T
                
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
                            
                            else:  # default to "partial" matching
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
                            coordinator = FeatureExtractionCoordinator(
                                random_seed=getattr(self, "random_seed", None),
                                duration_method=getattr(self, "duration_method", "geometric"),
                            )
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
                is_epoched_sliding = self.data_type == "epoched" and "sliding" in self.feature_mode
                needs_epoched_structure = self.data_type == "epoched" and ("ROF" in self.feature_list or "RTF" in self.feature_list or "pre_post" in self.feature_mode)

                # For epoched data with sliding mode, we need to preserve trial structure for pre/post extraction
                # For averaged mode, flatten everything
                if len(segmentation_array.shape) == 2:
                    if is_epoched_sliding:
                        # Keep trial structure for pre/post event extraction
                        # Use first trial's labels for time array generation
                        labels = [str(item) for item in segmentation_array[0, :]]
                        original_segmentation_array = segmentation_array
                    else:
                        # Flatten epoched segmentation for averaged mode: (trials, timepoints) -> (trials*timepoints,)
                        labels = [str(item) for item in segmentation_array.flatten()]
                        # Store original for ROF/RTF if needed
                        if needs_epoched_structure:
                            original_segmentation_array = segmentation_array
                        else:
                            original_segmentation_array = None
                else:
                    labels = [str(item) for item in segmentation_array]  # Ensure strings
                    original_segmentation_array = None

                # Retrieve accurate time points directly from segmentation file when available
                num_samples = len(labels)
                time = None
                time_single_epoch = None  # Store original single-epoch time for ROF/RTF and epoched sliding
                
                try:
                    if self.export_format == ".csv":
                        df_time = pd.read_csv(segmentation_path, usecols=["time"])
                        time_unique_array = df_time["time"].unique()
                        time_unique = sorted([float(t) for t in time_unique_array])  # Ensure it's a list of floats
                        
                        # For epoched data, handle time differently based on mode
                        if self.data_type == "epoched" and len(segmentation_array.shape) == 2:
                            n_trials = segmentation_array.shape[0]
                            n_times = len(time_unique)
                            # Store single-epoch time for ROF/RTF and epoched sliding
                            time_single_epoch = time_unique
                            
                            if is_epoched_sliding:
                                # For pre/post event extraction, use single epoch time
                                time = time_unique
                            elif n_trials * n_times == num_samples:
                                # For averaged mode with flattened data, replicate time for all trials
                                time = time_unique * n_trials  # Already a list, can multiply directly
                        elif len(time_unique) == num_samples:
                            time = time_unique
                except Exception as _e_time:
                    time = None

                if time is None:
                    # Fallback: derive from sampling rate as before
                    if hasattr(self, "sampling_rate") and self.sampling_rate:
                        time_step = 1000 / self.sampling_rate  # ms
                        if self.data_type == "epoched":
                            start_time = -1000
                            # For epoched data, create one epoch's worth of time
                            n_timepoints_per_epoch = len(original_segmentation_array[0]) if original_segmentation_array is not None else 1250
                            time_single_epoch = [start_time + i * time_step for i in range(n_timepoints_per_epoch)]
                            
                            if is_epoched_sliding:
                                # For pre/post event extraction, use single epoch time
                                time = time_single_epoch
                            else:
                                # For averaged mode, replicate for all trials
                                time = time_single_epoch * (num_samples // n_timepoints_per_epoch)
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
                eeg = self.comet_data_io.load_eeg(eeg_file, self.data_type)

                # For epoched sliding mode (pre/post event extraction), keep 3D structure
                # For averaged mode or ROF/RTF with averaged, flatten to 2D
                if is_epoched_sliding:
                    eeg_data = eeg.get_data()  # Keep 3D: (trials, channels, timepoints)
                else:
                    eeg_data = self.comet_data_io.get_eeg_data(
                        eeg, self.data_type
                    )  # Flatten to 2D: (channels, all_timepoints)

                # Create segmentation dictionary in expected format
                segmentation = {
                    "labels": labels,
                    "time": time,
                    "filename": segmentation_name,
                    "eeg_data": eeg_data,
                    "microstate_maps": self.best_maps,
                    "microstate_labels": self.microstate_labels,
                }

                # Add original segmentation data for epoched sliding processing or ROF
                if original_segmentation_array is not None:
                    segmentation["original_segmentation_array"] = original_segmentation_array
                    # For ROF/RTF calculation, store the epoched labels and single-epoch time
                    if needs_epoched_structure:
                        segmentation["epoched_labels"] = original_segmentation_array
                        # Store single-epoch time array for ROF/RTF (not the replicated one)
                        if time_single_epoch is not None:
                            segmentation["time_single_epoch"] = np.array(time_single_epoch)
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

            # For epoched data with multiple modes or pre_post mode, extract each mode separately with correct data structure
            if self.data_type == "epoched" and (len(self.feature_mode) > 1 or "pre_post" in self.feature_mode):
                extracted_features = {}
                
                # Extract each mode with appropriate data structure
                for mode in self.feature_mode:
                    if mode == "averaged":
                        # Extract averaged mode with flattened data structure
                        segmentation_averaged = segmentation.copy()
                        # Flatten labels and EEG data for averaged mode
                        labels_flat = [str(item) for item in segmentation_array.flatten()]
                        eeg_flat = self.comet_data_io.get_eeg_data(eeg, self.data_type)
                        time_flat = time_single_epoch * segmentation_array.shape[0] if time_single_epoch else time
                        segmentation_averaged["labels"] = labels_flat
                        segmentation_averaged["eeg_data"] = eeg_flat
                        segmentation_averaged["time"] = time_flat
                        # Add epoched_labels for ROF/RTF
                        if needs_epoched_structure:
                            segmentation_averaged["epoched_labels"] = segmentation_array
                            if time_single_epoch is not None:
                                segmentation_averaged["time_single_epoch"] = np.array(time_single_epoch)
                        
                        mode_features = self.comet_feature_extractor.extract_features(
                            segmentation=segmentation_averaged,
                            feature_list=self.feature_list,
                            feature_mode=[mode],
                            feature_types=self.feature_types,
                            sliding_window_size=self.sliding_window_size,
                            pre_window_size=self.pre_window_size,
                            post_window_size=self.post_window_size,
                            pre_event_window=getattr(self, 'pre_event_window', None),
                            post_event_window=getattr(self, 'post_event_window', None),
                        )
                        extracted_features.update(mode_features)
                        
                    elif mode == "sliding":
                        # Extract sliding mode with trial-preserved structure
                        mode_features = self.comet_feature_extractor.extract_features(
                            segmentation=segmentation,
                            feature_list=self.feature_list,
                            feature_mode=[mode],
                            feature_types=self.feature_types,
                            sliding_window_size=self.sliding_window_size,
                            pre_window_size=self.pre_window_size,
                            post_window_size=self.post_window_size,
                            pre_event_window=getattr(self, 'pre_event_window', None),
                            post_event_window=getattr(self, 'post_event_window', None),
                        )
                        extracted_features.update(mode_features)
                        
                    elif mode == "pre_post":
                        # Extract pre_post mode with flattened data structure (like averaged)
                        segmentation_pre_post = segmentation.copy()
                        # Flatten labels and EEG data for pre_post mode
                        labels_flat = [str(item) for item in segmentation_array.flatten()]
                        eeg_flat = self.comet_data_io.get_eeg_data(eeg, self.data_type)
                        time_flat = time_single_epoch if time_single_epoch else time
                        segmentation_pre_post["labels"] = labels_flat
                        segmentation_pre_post["eeg_data"] = eeg_flat
                        segmentation_pre_post["time"] = time_flat
                        # Add original segmentation array for pre_post extraction
                        segmentation_pre_post["original_segmentation_array"] = segmentation_array
                        
                        mode_features = self.comet_feature_extractor.extract_features(
                            segmentation=segmentation_pre_post,
                            feature_list=self.feature_list,
                            feature_mode=[mode],
                            feature_types=self.feature_types,
                            sliding_window_size=self.sliding_window_size,
                            pre_window_size=self.pre_window_size,
                            post_window_size=self.post_window_size,
                            pre_event_window=getattr(self, 'pre_event_window', None),
                            post_event_window=getattr(self, 'post_event_window', None),
                        )
                        extracted_features.update(mode_features)
            else:
                # Standard extraction for single mode or non-epoched data
                extracted_features = self.comet_feature_extractor.extract_features(
                    segmentation=segmentation,
                    feature_list=self.feature_list,
                    feature_mode=self.feature_mode,
                    feature_types=self.feature_types,
                    sliding_window_size=self.sliding_window_size,
                    pre_window_size=self.pre_window_size,
                    post_window_size=self.post_window_size,
                    pre_event_window=getattr(self, 'pre_event_window', None),
                    post_event_window=getattr(self, 'post_event_window', None),
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

                        # Handle variability data separately
                        if "variability_data" in extracted_features[mode]:
                            if "variability_data" not in organized_results[mode]:
                                organized_results[mode]["variability_data"] = {}
                            organized_results[mode]["variability_data"].update(
                                extracted_features[mode]["variability_data"]
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

                # Variability features export (SD and RMSSD for sliding mode)
                if "variability_data" in organized_results[mode]:
                    variability_data = organized_results[mode]["variability_data"]
                    if variability_data:
                        try:
                            self.comet_feature_io.export_variability_data(
                                variability_data_dict=variability_data,
                                output_folder=self.extracted_features_path,
                                export_format=self.export_format,
                            )

                            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                                self.LogWindow.append_log(
                                    f"Exported sliding variability features for {mode} mode ({len(variability_data)} files)",
                                    log_type="success",
                                )
                        except Exception as variability_export_error:
                            error_msg = (
                                f"Failed to export variability data for {mode} mode: {variability_export_error}"
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

    def _count_successful_source_localizations(self, stc_path):
        """Count how many files have been successfully source-localized.
        
        Args:
            stc_path: Path to the stc directory containing subject folders.
            
        Returns:
            int: Number of files with successfully generated stc data.
        """
        if not os.path.exists(stc_path):
            return 0
        
        # Count subdirectories in stc_path that contain stc files
        count = 0
        try:
            for subject_dir in os.listdir(stc_path):
                subject_path = os.path.join(stc_path, subject_dir)
                if os.path.isdir(subject_path):
                    # Check if this subject directory contains any stc files.
                    # Supported formats:
                    # - .h5: HDF5 format
                    # - .stc: MNE standard format (may have -lh.stc/-rh.stc hemispheres)
                    # - .pkl/.npy: pickle and numpy formats
                    stc_files = [f for f in os.listdir(subject_path) 
                                if f.endswith(('.stc', '.pkl', '.npy', '.h5', '-lh.stc', '-rh.stc'))]
                    if stc_files:
                        count += 1
        except Exception:
            return 0
        
        return count

    def _get_available_stc_files(self, stc_path):
        """Get list of file names that have successfully generated stc data.
        
        Args:
            stc_path: Path to the stc directory containing subject folders.
            
        Returns:
            list: List of file names (subject names) with stc data.
        """
        if not os.path.exists(stc_path):
            return []
        
        available_files = []
        try:
            for subject_dir in os.listdir(stc_path):
                subject_path = os.path.join(stc_path, subject_dir)
                if os.path.isdir(subject_path):
                    # Check if this subject directory contains any stc files
                    # Include .h5 which is the format used by MNE for HDF5 storage
                    stc_files = [f for f in os.listdir(subject_path) if f.endswith(('.stc', '.pkl', '.npy', '.h5'))]
                    if stc_files:
                        available_files.append(subject_dir)
        except Exception:
            return []
        
        return available_files

    def _count_successful_correlations(self, results_path):
        """Count how many files have successfully completed source-microstate correlation.
        
        Args:
            results_path: Path to the tess_sources or avg_sources directory.
            
        Returns:
            int: Number of files with successfully generated correlation results.
        """
        if not os.path.exists(results_path):
            return 0
        
        # Count files in the results directory
        count = 0
        try:
            # For TESS: count files ending with _z_scores.pkl or _filtered_z_scores.pkl
            # For AVG: count files ending with .pkl or .npy
            result_files = [f for f in os.listdir(results_path) 
                          if f.endswith(('.pkl', '.npy', '.csv'))]
            
            # Count unique subject names (each subject may have multiple result files)
            subject_names = set()
            for f in result_files:
                # Extract subject name from filename (remove extension and suffixes)
                base_name = f.replace('_z_scores.pkl', '').replace('_filtered_z_scores.pkl', '')
                base_name = base_name.replace('_p_values.pkl', '').replace('.pkl', '').replace('.npy', '')
                subject_names.add(base_name)
            
            count = len(subject_names)
        except Exception:
            return 0
        
        return count

    def _get_files_with_correlation_results(self, results_path):
        """Get list of file names that have correlation results.
        
        Args:
            results_path: Path to the tess_sources or avg_sources directory.
            
        Returns:
            list: List of file names (subject names) with correlation results.
        """
        if not os.path.exists(results_path):
            return []
        
        subject_names = set()
        try:
            # For TESS: look for files ending with _z_scores.pkl or _filtered_z_scores.pkl
            # For AVG: look for files ending with .pkl or .npy
            result_files = [f for f in os.listdir(results_path) 
                          if f.endswith(('.pkl', '.npy', '.csv'))]
            
            for f in result_files:
                # Extract subject name from filename (remove extension and suffixes)
                base_name = f.replace('_z_scores.pkl', '').replace('_filtered_z_scores.pkl', '')
                base_name = base_name.replace('_p_values.pkl', '').replace('.pkl', '').replace('.npy', '')
                subject_names.add(base_name)
        except Exception:
            return []
        
        return list(subject_names)

    def source_localize_file(self, eeg_path, eeg_name, worker=None):
        """Perform source localization for a single file. Terminates early if stop requested."""
        if worker is not None and getattr(worker, "stopped", False):
            return

        success = self.comet_source_localizer.localize_single_file(eeg_path, eeg_name)

        # Reduced logging frequency - only log errors or every Nth file to keep UI responsive
        # The progress bar provides visual feedback for each file
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            if not success:
                # Always log errors immediately
                self.LogWindow.append_log(f"❌ Error Source Localizing: {eeg_name}")
            # Success messages are shown in progress bar, not individual log entries

    def source_identify_file(self, eeg_path, eeg_name, worker=None):
        """Identify microstate sources for a single file with stop support."""
        if worker is not None and getattr(worker, "stopped", False):
            return

        success = self.comet_source_localizer.identify_sources_single_file(
            eeg_path, eeg_name, self.source_localization_method
        )

        # Reduced logging frequency - only log errors to keep UI responsive
        # The progress bar provides visual feedback for each file
        if hasattr(self, "LogWindow") and self.LogWindow is not None:
            if not success:
                # Always log errors immediately
                self.LogWindow.append_log(f"❌ Error Identifying Microstate Sources: {eeg_name}")
            # Success messages are shown in progress bar, not individual log entries

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
            data_type=self.data_type,
            bem_solver=self.bem_solver,
            inverse_method=self.inverse_method,
            spacing=self.spacing,
            microstate_maps=self.best_maps,
            n_permutations=self.n_permutations,
            logger=self.logger,
            random_seed=getattr(self, "random_seed", None),
        )

        # Make sure the stc_path is set correctly in the source localizer
        self.comet_source_localizer.stc_path = stc_path

        # Load all EEG files
        list_eeg_path, list_eeg_name = self.comet_data_io.find_data(
            self.preprocessed_data_path, extension=self.extension, pattern="*"
        )
        
        # Get list of files that already have STC data
        existing_stc_files = self._get_available_stc_files(stc_path)
        
        # Filter out files that already have STC data
        filtered_files = []
        skipped_files = []
        for eeg_path, eeg_name in zip(list_eeg_path, list_eeg_name):
            if eeg_name in existing_stc_files:
                skipped_files.append(eeg_name)
            else:
                filtered_files.append((eeg_path, eeg_name))
        
        self.zipped_eeg_files = filtered_files
        
        total_files = len(list_eeg_name)
        already_done = len(skipped_files)
        to_process = len(filtered_files)
        
        # Log information about what will be processed
        if already_done > 0:
            self.logger.processing_info(
                "SOURCE_LOCALIZATION",
                f"Found {already_done} files with existing source data - skipping these"
            )
        
        if to_process == 0:
            self.logger.processing_success(
                "SOURCE_LOCALIZATION",
                f"All {total_files} files already have source data - nothing to process"
            )
            # Properly handle completion when nothing needs processing
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.set_progress_label("All files already processed")
                self.LogWindow.set_progress_max(total_files)
                self.LogWindow.set_progress_value(total_files)
                self.LogWindow.set_progress_text(f"✅ All {total_files} files already completed")
                # Call completion callback
                if self.LogWindow.process_finished_callback:
                    self.LogWindow.process_finished_callback()
                    self.LogWindow.process_finished_callback = None
            else:
                # Non-GUI mode - just call completion
                self._on_source_localization_finished()
            return
        
        self.logger.processing_info(
            "SOURCE_LOCALIZATION",
            f"Processing {to_process} files ({already_done} already completed)"
        )

        # Log source localization settings once
        self.comet_source_localizer.log_settings()

        # Perform source localization on remaining files
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
                data_type=self.data_type,
                bem_solver=self.bem_solver,
                inverse_method=self.inverse_method,
                spacing=self.spacing,
                microstate_maps=self.best_maps,
                n_permutations=self.n_permutations,
                logger=self.logger,
                random_seed=getattr(self, "random_seed", None),
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

        # Get list of files with available stc data
        available_stc_files = self._get_available_stc_files(stc_path)
        
        if not available_stc_files:
            self.logger.error(
                "SOURCE_LOCALIZATION", 
                "No source-localized files found. Please complete source localization first."
            )
            return

        # Get list of files that already have correlation results
        if self.source_localization_method == "tess":
            results_path = os.path.join(self.localized_sources_path, "tess_sources")
        else:  # avg method
            results_path = os.path.join(self.localized_sources_path, "avg_sources")
        
        existing_correlation_files = self._get_files_with_correlation_results(results_path)

        # Load all EEG files and filter to only those with stc data but without correlation results
        list_eeg_path, list_eeg_name = self.comet_data_io.find_data(
            self.preprocessed_data_path, extension=self.extension, pattern="*"
        )
        
        # Filter to only include files that have stc data and don't have correlation results yet
        filtered_files = []
        skipped_files = []
        for eeg_path, eeg_name in zip(list_eeg_path, list_eeg_name):
            if eeg_name in available_stc_files:
                if eeg_name in existing_correlation_files:
                    skipped_files.append(eeg_name)
                else:
                    filtered_files.append((eeg_path, eeg_name))
        
        self.zipped_eeg_files = filtered_files
        
        total_preprocessed = len(list_eeg_name)
        files_with_stc = len(available_stc_files)
        already_done = len(skipped_files)
        to_process = len(filtered_files)
        
        # Log information about available files
        if files_with_stc < total_preprocessed:
            self.logger.warning(
                "SOURCE_LOCALIZATION",
                f"Source time series available for {files_with_stc}/{total_preprocessed} files"
            )
        
        if already_done > 0:
            self.logger.processing_info(
                "SOURCE_LOCALIZATION",
                f"Found {already_done} files with existing correlation results - skipping these"
            )
        
        if to_process == 0:
            self.logger.processing_success(
                "SOURCE_LOCALIZATION",
                f"All {files_with_stc} files with source data already have correlation results - nothing to process"
            )
            # Properly handle completion when nothing needs processing
            if hasattr(self, "LogWindow") and self.LogWindow is not None:
                self.LogWindow.set_progress_label("All files already processed")
                self.LogWindow.set_progress_max(files_with_stc)
                self.LogWindow.set_progress_value(files_with_stc)
                self.LogWindow.set_progress_text(f"✅ All {files_with_stc} files already completed")
                # Call completion callback
                if self.LogWindow.process_finished_callback:
                    self.LogWindow.process_finished_callback()
                    self.LogWindow.process_finished_callback = None
            else:
                # Non-GUI mode - just call completion
                self._on_source_identification_finished()
            return
        
        self.logger.processing_info(
            "SOURCE_LOCALIZATION",
            f"Processing {to_process} files ({already_done} already completed)"
        )

        # Log source identification settings
        source_settings = {
            "Method": self.source_localization_method,
            "Number of Permutations": self.n_permutations,
            "Files to Process": to_process,
        }
        self.logger.settings_info("SOURCE_LOCALIZATION", source_settings)

        # Log reference for TESS method if selected
        if self.source_localization_method == "tess":
            self.logger.reference("SOURCE_LOCALIZATION", "https://doi.org/10.1016/j.neuroimage.2014.04.002")

        # Perform source identification on remaining files
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
        self.config["io_config"]["load_all_files"] = str(self.load_all_files)
        self.config["io_config"]["data_type"] = self.data_type
        self.config["io_config"]["output_folder"] = self.output_folder

        self.config["preprocessing_config"]["temporal_filter_data"] = str(self.temporal_filter_data)
        self.config["preprocessing_config"]["filter_method"] = self.filter_method
        self.config["preprocessing_config"]["lowcut_freq"] = str(self.lowcut_freq)
        self.config["preprocessing_config"]["highcut_freq"] = str(self.highcut_freq)
        self.config["preprocessing_config"]["downsample_data"] = str(self.downsample_data)
        self.config["preprocessing_config"]["spatial_filter_data"] = str(self.spatial_filter_data)
        self.config["preprocessing_config"]["auto_clean_data"] = str(self.auto_clean_data)
        self.config["preprocessing_config"]["sampling_rate"] = str(self.sampling_rate)
        self.config["preprocessing_config"]["remove_channels"] = str(self.remove_channels)
        self.config["preprocessing_config"]["channels_to_remove"] = str(self.channels_to_remove)
        self.config["preprocessing_config"]["prep_data"] = str(self.prep_data)

        self.config["clustering_config"]["smoothing_gfp"] = str(self.smoothing_gfp)
        self.config["clustering_config"]["smoothing_distance"] = str(self.smoothing_distance)
        self.config["clustering_config"]["n_maps"] = str(self.n_maps)
        if self.n_maps == "auto":
            self.config["clustering_config"]["k_min"] = str(self.k_min)
            self.config["clustering_config"]["k_max"] = str(self.k_max)
            self.config["clustering_config"]["stopping_mode"] = self.stopping_mode
            self.config["clustering_config"]["stopping_threshold"] = str(self.stopping_threshold)
        self.config["clustering_config"]["data_percentage"] = str(self.data_percentage)
        self.config["clustering_config"]["initializer"] = self.initializer
        self.config["clustering_config"]["clustering_method"] = self.clustering_method
        self.config["clustering_config"]["max_iterations"] = str(self.max_iterations)
        self.config["clustering_config"]["clustering_tolerance"] = str(self.clustering_tolerance)
        self.config["clustering_config"]["similarity_metric"] = self.similarity_metric
        self.config["clustering_config"]["n_repeats"] = str(self.n_repeats)
        self.config["backfitting_config"]["backfit_to"] = self.backfit_to
        self.config["backfitting_config"]["identify_short_window"] = str(self.identify_short_window)
        self.config["backfitting_config"]["filter_segments"] = str(self.filter_segments)
        self.config["backfitting_config"]["filter_segments_less_than"] = str(
            self.filter_segments_less_than
        )
        self.config["backfitting_config"]["filter_segments_option"] = self.filter_segments_option
        self.config["backfitting_config"]["convergence_epsilon"] = str(self.convergence_epsilon)
        self.config["backfitting_config"]["half_window_size"] = str(self.half_window_size)
        self.config["backfitting_config"]["smoothness_penalty"] = str(self.smoothness_penalty)
        self.config["backfitting_config"]["min_correlation_threshold"] = (
            str(self.min_correlation_threshold) if self.min_correlation_threshold is not False else "False"
        )

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
        self.config["features_config"]["duration_method"] = self.duration_method

        self.config["source_config"]["use_anatomy"] = self.use_anatomy
        self.config["source_config"]["bem_solver"] = self.bem_solver
        self.config["source_config"]["inverse_method"] = self.inverse_method
        self.config["source_config"]["n_permutations"] = str(self.n_permutations)
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

            # Notify any registered callbacks (e.g., GUI updates)
            # The callback is responsible for opening the visualization window
            if (
                hasattr(self, "clustering_completed_callback")
                and self.clustering_completed_callback is not None
            ):
                self.clustering_completed_callback()
            else:
                # Only launch window directly if no callback is registered (non-GUI mode or standalone)
                self._launch_microstate_labeling()

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

            # Notify any registered callbacks (e.g., GUI updates)
            # The callback is responsible for opening the visualization window
            if (
                hasattr(self, "clustering_completed_callback")
                and self.clustering_completed_callback is not None
            ):
                self.clustering_completed_callback()
            else:
                # Only launch window directly if no callback is registered (non-GUI mode or standalone)
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

        # Log completion
        self.logger.processing_success("PREPROCESSING", "Preprocessing completed successfully")

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
        # Count how many files were successfully source-localized
        stc_path = os.path.join(self.localized_sources_path, "stc")
        successful_files = self._count_successful_source_localizations(stc_path)
        total_files = len(self.zipped_eeg_files) if hasattr(self, "zipped_eeg_files") else 0
        
        # Set source localization flag if at least some files succeeded
        if successful_files > 0:
            self.done_source_localization = True
            
            # Log completion with file count information
            if successful_files == total_files:
                self.logger.processing_success(
                    "SOURCE_LOCALIZATION", 
                    f"Source localization completed successfully for all {total_files} files"
                )
            else:
                self.logger.processing_success(
                    "SOURCE_LOCALIZATION", 
                    f"Source localization completed for {successful_files}/{total_files} files"
                )
                self.logger.warning(
                    "SOURCE_LOCALIZATION",
                    f"{total_files - successful_files} files failed source localization"
                )
        else:
            self.done_source_localization = False
            self.logger.error(
                "SOURCE_LOCALIZATION", 
                "Source localization failed for all files"
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
        # Count how many files were successfully processed for correlation
        if self.source_localization_method == "tess":
            results_path = os.path.join(self.localized_sources_path, "tess_sources")
        else:  # avg method
            results_path = os.path.join(self.localized_sources_path, "avg_sources")
        
        successful_files = self._count_successful_correlations(results_path)
        total_files = len(self.zipped_eeg_files) if hasattr(self, "zipped_eeg_files") else 0
        
        # Set correlation flag if at least some files succeeded
        if successful_files > 0:
            self.done_identifying_microstate_sources = True
            
            # Log completion with file count information
            if successful_files == total_files:
                self.logger.processing_success(
                    "SOURCE_LOCALIZATION", 
                    f"Source-microstate correlation completed for all {total_files} files"
                )
            else:
                self.logger.processing_success(
                    "SOURCE_LOCALIZATION", 
                    f"Source-microstate correlation completed for {successful_files}/{total_files} files"
                )
                if total_files > successful_files:
                    self.logger.warning(
                        "SOURCE_LOCALIZATION",
                        f"{total_files - successful_files} files failed correlation"
                    )
        else:
            self.done_identifying_microstate_sources = False
            self.logger.error(
                "SOURCE_LOCALIZATION", 
                "Source-microstate correlation failed for all files"
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

