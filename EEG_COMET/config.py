"""Typed configuration model for EEG-COMET.

Provides a single, typed, version-controlled :class:`CometConfig` dataclass
plus :meth:`from_ini` / :meth:`to_ini` round-trip helpers for the
``eeg_comet_config.ini`` file format.

Design goals:
- One source of truth for default values across the config surface.
- Every field is typed and has a stable default, so missing INI keys never
  raise ``KeyError`` mid-run.
- ``from_ini`` / ``to_ini`` are pure string <-> dataclass converters with no
  Qt dependency.
- ``CONFIG_FORMAT_VERSION`` records the on-disk format so study folders can
  be safely opened by versioned readers.
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

CONFIG_FORMAT_VERSION = "1.0"


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off", ""):
        return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_csv_list(value: Any, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    text = str(value).strip()
    if not text:
        return list(default)
    return [token.strip() for token in text.split(",") if token.strip()]


def _coerce_optional_float(value: Any) -> Union[float, bool]:
    """Return ``False`` for off/empty/blank values, else the parsed float.

    Used by ``min_correlation_threshold`` where ``False`` means "no threshold".
    """
    if value in (None, "", "False", "false", "0", False):
        return False
    try:
        return float(value)
    except (TypeError, ValueError):
        return False


@dataclass
class IOConfig:
    study_name: str = "my_study"
    input_folder: str = ""
    montage: str = ""
    extension: str = ".set"
    pattern: str = "*"
    datatype: str = "raw"
    output_folder: str = ""
    channel_location_dir: str = ""


@dataclass
class PreprocessingConfig:
    temporal_filter_data: bool = True
    filter_method: str = "fir"
    lowcut_freq: int = 2
    highcut_freq: int = 20
    downsample_data: bool = True
    sample_rate: int = 250
    spatial_filter_data: bool = False
    auto_clean_data: bool = False
    remove_channels: bool = False
    ch2rm: str = ""
    prep_data: bool = False


@dataclass
class ClusteringConfig:
    smoothing_gfp: bool = False
    smoothing_distance: int = 10
    number_of_maps: Union[int, str] = 4
    kmin: int = 2
    kmax: int = 10
    stopping_mode: str = "majority_vote"
    stopping_parameter: int = 10
    use_percentages: int = 100
    initializer: str = "Random"
    clustering_method: str = "Modified K-Means Clustering"
    max_iterations: int = 500
    clustering_tolerance: float = 1e-6
    similarity_metric: str = "Spatial Correlation"
    number_of_repeats: int = 5
    batch_size: Optional[int] = None


@dataclass
class BackfittingConfig:
    backfit_to: str = "all"
    identify_short_window: bool = False
    filter_segments: bool = False
    filter_segments_less_than: int = 20
    filter_segments_option: str = "smooth"
    epsilon: float = 1e-6
    b: int = 3
    lamb: int = 5
    min_correlation_threshold: Union[float, bool] = False


@dataclass
class FeaturesConfig:
    export_format: str = ".csv"
    feature_list: List[str] = field(default_factory=lambda: ["OCC", "DUR", "COV"])
    feature_mode: List[str] = field(default_factory=lambda: ["averaged"])
    feature_types: List[str] = field(default_factory=lambda: ["real"])
    sliding_window_size: int = 1
    event_based_sliding: bool = False
    selected_events: List[str] = field(default_factory=list)
    event_matching_mode: str = "partial"
    pre_window_size: int = 1
    post_window_size: int = 1


@dataclass
class SourceConfig:
    use_anatomy: str = "fsaverage"
    bem_solver: str = "mne"
    inverse_method: str = "dSPM"
    nperm: int = 2000
    spacing: str = "ico3"
    source_localization_method: str = "tess"
    anatomy_subjects_dir: str = ""


@dataclass
class StateFlags:
    done_preprocessing: bool = False
    done_clustering: bool = False
    done_microstate_labeling: bool = False
    done_backfitting: bool = False
    done_extracting_features: bool = False
    done_source_localization: bool = False
    done_identifying_microstate_sources: bool = False


@dataclass
class CometConfig:
    """Top-level typed config for an EEG-COMET study.

    Use :meth:`from_ini` / :meth:`to_ini` to round-trip with the
    ``eeg_comet_config.ini`` file format. Fields not present in the INI fall
    back to their dataclass defaults, so partially-populated studies load
    without error.
    """

    format_version: str = CONFIG_FORMAT_VERSION
    io: IOConfig = field(default_factory=IOConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    backfitting: BackfittingConfig = field(default_factory=BackfittingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    state: StateFlags = field(default_factory=StateFlags)
    common_events: List[str] = field(default_factory=list)
    random_seed: Optional[int] = None
    log_text: str = ""

    # ----------------------------- INI I/O -------------------------------- #

    def to_configparser(self) -> ConfigParser:
        cp = ConfigParser()
        cp["meta"] = {"format_version": self.format_version}

        cp["io_config"] = {
            "study_name": self.io.study_name,
            "input_folder": self.io.input_folder,
            "montage": str(self.io.montage) if self.io.montage is not None else "",
            "extension": self.io.extension,
            "pattern": self.io.pattern,
            "datatype": self.io.datatype,
            "output_folder": self.io.output_folder,
            "channel_location_dir": self.io.channel_location_dir,
        }

        pp = self.preprocessing
        cp["preprocessing_config"] = {
            "temporal_filter_data": str(pp.temporal_filter_data),
            "filter_method": pp.filter_method,
            "lowcut_freq": str(pp.lowcut_freq),
            "highcut_freq": str(pp.highcut_freq),
            "downsample_data": str(pp.downsample_data),
            "sample_rate": str(pp.sample_rate),
            "spatial_filter_data": str(pp.spatial_filter_data),
            "auto_clean_data": str(pp.auto_clean_data),
            "remove_channels": str(pp.remove_channels),
            "ch2rm": str(pp.ch2rm),
            "prep_data": str(pp.prep_data),
        }

        cl = self.clustering
        cp["clustering_config"] = {
            "smoothing_gfp": str(cl.smoothing_gfp),
            "smoothing_distance": str(cl.smoothing_distance),
            "number_of_maps": str(cl.number_of_maps),
            "kmin": str(cl.kmin),
            "kmax": str(cl.kmax),
            "stopping_mode": cl.stopping_mode,
            "stopping_parameter": str(cl.stopping_parameter),
            "use_percentages": str(cl.use_percentages) if cl.use_percentages is not None else "",
            "initializer": cl.initializer,
            "clustering_method": cl.clustering_method,
            "max_iterations": str(cl.max_iterations),
            "clustering_tolerance": str(cl.clustering_tolerance),
            "similarity_metric": cl.similarity_metric,
            "number_of_repeats": str(cl.number_of_repeats),
            "batch_size": "" if cl.batch_size is None else str(cl.batch_size),
        }

        bf = self.backfitting
        cp["backfitting_config"] = {
            "backfit_to": bf.backfit_to,
            "identify_short_window": str(bf.identify_short_window),
            "filter_segments": str(bf.filter_segments),
            "filter_segments_less_than": str(bf.filter_segments_less_than),
            "filter_segments_option": bf.filter_segments_option,
            "epsilon": str(bf.epsilon),
            "b": str(bf.b),
            "lamb": str(bf.lamb),
            "min_correlation_threshold": (
                "False" if bf.min_correlation_threshold is False
                else str(bf.min_correlation_threshold)
            ),
        }

        ft = self.features
        cp["features_config"] = {
            "export_format": ft.export_format,
            "feature_list": ", ".join(ft.feature_list),
            "feature_mode": ", ".join(ft.feature_mode),
            "feature_types": ", ".join(ft.feature_types),
            "sliding_window_size": str(ft.sliding_window_size),
            "event_based_sliding": str(ft.event_based_sliding),
            "selected_events": ", ".join(ft.selected_events),
            "event_matching_mode": ft.event_matching_mode,
            "pre_window_size": str(ft.pre_window_size),
            "post_window_size": str(ft.post_window_size),
        }

        sc = self.source
        cp["source_config"] = {
            "use_anatomy": sc.use_anatomy,
            "bem_solver": sc.bem_solver,
            "inverse_method": sc.inverse_method,
            "nperm": str(sc.nperm),
            "spacing": sc.spacing,
            "source_localization_method": sc.source_localization_method,
            "anatomy_subjects_dir": sc.anatomy_subjects_dir,
        }

        st = self.state
        cp["state_flags"] = {k: str(v) for k, v in asdict(st).items()}

        cp["events_config"] = {"common_events": ", ".join(self.common_events)}

        if self.random_seed is not None:
            cp["reproducibility"] = {"random_seed": str(self.random_seed)}

        return cp

    def to_ini(self, path: Union[str, Path]) -> None:
        cp = self.to_configparser()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            cp.write(fh)

    # ----------------------------- INI parse ------------------------------ #

    @classmethod
    def from_configparser(cls, cp: ConfigParser) -> "CometConfig":
        cfg = cls()

        if cp.has_section("meta"):
            cfg.format_version = cp["meta"].get("format_version", CONFIG_FORMAT_VERSION)

        if cp.has_section("io_config"):
            io = cp["io_config"]
            cfg.io = IOConfig(
                study_name=io.get("study_name", IOConfig.study_name),
                input_folder=io.get("input_folder", ""),
                montage=io.get("montage", ""),
                extension=io.get("extension", ".set"),
                pattern=io.get("pattern", io.get("pattern_content", "*")),
                datatype=io.get("datatype", "raw"),
                output_folder=io.get("output_folder", ""),
                channel_location_dir=io.get("channel_location_dir", ""),
            )

        if cp.has_section("preprocessing_config"):
            pp = cp["preprocessing_config"]
            cfg.preprocessing = PreprocessingConfig(
                temporal_filter_data=_coerce_bool(pp.get("temporal_filter_data"), True),
                filter_method=pp.get("filter_method", "fir"),
                lowcut_freq=_coerce_int(pp.get("lowcut_freq"), 2),
                highcut_freq=_coerce_int(pp.get("highcut_freq"), 20),
                downsample_data=_coerce_bool(pp.get("downsample_data"), True),
                sample_rate=_coerce_int(pp.get("sample_rate"), 250),
                spatial_filter_data=_coerce_bool(pp.get("spatial_filter_data"), False),
                auto_clean_data=_coerce_bool(pp.get("auto_clean_data"), False),
                remove_channels=_coerce_bool(pp.get("remove_channels"), False),
                ch2rm=pp.get("ch2rm", ""),
                prep_data=_coerce_bool(pp.get("prep_data"), False),
            )

        if cp.has_section("clustering_config"):
            cl = cp["clustering_config"]
            n_maps_raw = cl.get("number_of_maps", "4")
            n_maps: Union[int, str] = (
                "auto" if str(n_maps_raw).strip().lower() == "auto"
                else _coerce_int(n_maps_raw, 4)
            )
            batch = cl.get("batch_size", "").strip()
            cfg.clustering = ClusteringConfig(
                smoothing_gfp=_coerce_bool(cl.get("smoothing_gfp"), False),
                smoothing_distance=_coerce_int(cl.get("smoothing_distance"), 10),
                number_of_maps=n_maps,
                kmin=_coerce_int(cl.get("kmin"), 2),
                kmax=_coerce_int(cl.get("kmax"), 10),
                stopping_mode=cl.get("stopping_mode", "majority_vote"),
                stopping_parameter=_coerce_int(cl.get("stopping_parameter"), 10),
                use_percentages=_coerce_int(cl.get("use_percentages"), 100),
                initializer=cl.get("initializer", "Random"),
                clustering_method=cl.get(
                    "clustering_method", "Modified K-Means Clustering"
                ),
                max_iterations=_coerce_int(cl.get("max_iterations"), 500),
                clustering_tolerance=_coerce_float(cl.get("clustering_tolerance"), 1e-6),
                similarity_metric=cl.get("similarity_metric", "Spatial Correlation"),
                number_of_repeats=_coerce_int(cl.get("number_of_repeats"), 5),
                batch_size=None if not batch else _coerce_int(batch, 1000),
            )

        if cp.has_section("backfitting_config"):
            bf = cp["backfitting_config"]
            cfg.backfitting = BackfittingConfig(
                backfit_to=bf.get("backfit_to", "all"),
                identify_short_window=_coerce_bool(bf.get("identify_short_window"), False),
                filter_segments=_coerce_bool(bf.get("filter_segments"), False),
                filter_segments_less_than=_coerce_int(bf.get("filter_segments_less_than"), 20),
                filter_segments_option=bf.get("filter_segments_option", "smooth"),
                epsilon=_coerce_float(bf.get("epsilon"), 1e-6),
                b=_coerce_int(bf.get("b"), 3),
                lamb=_coerce_int(bf.get("lamb"), 5),
                min_correlation_threshold=_coerce_optional_float(
                    bf.get("min_correlation_threshold")
                ),
            )

        if cp.has_section("features_config"):
            ft = cp["features_config"]
            cfg.features = FeaturesConfig(
                export_format=ft.get("export_format", ".csv"),
                feature_list=_coerce_csv_list(ft.get("feature_list"), ["OCC", "DUR", "COV"]),
                feature_mode=_coerce_csv_list(ft.get("feature_mode"), ["averaged"]),
                feature_types=_coerce_csv_list(ft.get("feature_types"), ["real"]),
                sliding_window_size=_coerce_int(ft.get("sliding_window_size"), 1),
                event_based_sliding=_coerce_bool(ft.get("event_based_sliding"), False),
                selected_events=_coerce_csv_list(ft.get("selected_events"), []),
                event_matching_mode=ft.get("event_matching_mode", "partial"),
                pre_window_size=_coerce_int(ft.get("pre_window_size"), 1),
                post_window_size=_coerce_int(ft.get("post_window_size"), 1),
            )

        if cp.has_section("source_config"):
            sc = cp["source_config"]
            cfg.source = SourceConfig(
                use_anatomy=sc.get("use_anatomy", "fsaverage"),
                bem_solver=sc.get("bem_solver", "mne"),
                inverse_method=sc.get("inverse_method", "dSPM"),
                nperm=_coerce_int(sc.get("nperm"), 2000),
                spacing=sc.get("spacing", "ico3"),
                source_localization_method=sc.get("source_localization_method", "tess"),
                anatomy_subjects_dir=sc.get("anatomy_subjects_dir", ""),
            )

        if cp.has_section("state_flags"):
            sf = cp["state_flags"]
            cfg.state = StateFlags(
                **{
                    f.name: _coerce_bool(sf.get(f.name), False)
                    for f in fields(StateFlags)
                }
            )

        if cp.has_section("events_config"):
            cfg.common_events = _coerce_csv_list(
                cp["events_config"].get("common_events"), []
            )

        if cp.has_section("reproducibility"):
            cfg.random_seed = _coerce_int(
                cp["reproducibility"].get("random_seed"), 0
            ) or None

        return cfg

    @classmethod
    def from_ini(cls, path: Union[str, Path]) -> "CometConfig":
        cp = ConfigParser()
        cp.read(path, encoding="utf-8")
        return cls.from_configparser(cp)

    # ------------------------------- helpers ------------------------------ #

    def iter_field_groups(self) -> Iterable:
        """Yield ``(group_name, dataclass_instance)`` pairs for the config sections."""
        yield "io", self.io
        yield "preprocessing", self.preprocessing
        yield "clustering", self.clustering
        yield "backfitting", self.backfitting
        yield "features", self.features
        yield "source", self.source
        yield "state", self.state
