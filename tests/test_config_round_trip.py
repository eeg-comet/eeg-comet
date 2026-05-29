"""Round-trip tests for the typed :class:`CometConfig`.

Verify that ``to_ini`` followed by ``from_ini`` is the identity for every
default and for representative non-default values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eeg_comet.config import (
    CONFIG_FORMAT_VERSION,
    BackfittingConfig,
    ClusteringConfig,
    CometConfig,
    FeaturesConfig,
    IOConfig,
    PreprocessingConfig,
    SourceConfig,
    StateFlags,
)


def test_defaults_round_trip(tmp_path):
    cfg = CometConfig()
    out = tmp_path / "eeg_comet_config.ini"
    cfg.to_ini(out)
    loaded = CometConfig.from_ini(out)

    assert loaded.format_version == CONFIG_FORMAT_VERSION
    assert loaded.io == IOConfig()
    assert loaded.preprocessing == PreprocessingConfig()
    assert loaded.clustering == ClusteringConfig()
    assert loaded.backfitting == BackfittingConfig()
    assert loaded.features == FeaturesConfig()
    assert loaded.source == SourceConfig()
    assert loaded.state == StateFlags()


def test_non_default_values_round_trip(tmp_path):
    cfg = CometConfig(
        random_seed=1337,
        common_events=["resting", "task"],
        io=IOConfig(study_name="alpha", input_folder="C:/data"),
        clustering=ClusteringConfig(number_of_maps="auto", batch_size=10000, kmin=3, kmax=8),
        features=FeaturesConfig(
            feature_list=["OCC", "DUR", "COV", "GEV", "TP"],
            feature_mode=["averaged", "variability"],
            feature_types=["real", "surrogate"],
        ),
    )

    path = tmp_path / "eeg_comet_config.ini"
    cfg.to_ini(path)
    loaded = CometConfig.from_ini(path)

    assert loaded.random_seed == 1337
    assert loaded.common_events == ["resting", "task"]
    assert loaded.io.study_name == "alpha"
    assert loaded.clustering.number_of_maps == "auto"
    assert loaded.clustering.batch_size == 10000
    assert loaded.features.feature_list == ["OCC", "DUR", "COV", "GEV", "TP"]
    assert loaded.features.feature_mode == ["averaged", "variability"]
    assert loaded.features.feature_types == ["real", "surrogate"]


def test_min_correlation_threshold_off_then_value(tmp_path):
    cfg = CometConfig(backfitting=BackfittingConfig(min_correlation_threshold=False))
    p1 = tmp_path / "a.ini"
    cfg.to_ini(p1)
    assert CometConfig.from_ini(p1).backfitting.min_correlation_threshold is False

    cfg.backfitting.min_correlation_threshold = 0.7
    p2 = tmp_path / "b.ini"
    cfg.to_ini(p2)
    assert CometConfig.from_ini(p2).backfitting.min_correlation_threshold == pytest.approx(0.7)


def test_missing_sections_fall_back_to_defaults(tmp_path):
    p = tmp_path / "minimal.ini"
    p.write_text("[io_config]\nstudy_name = solo\n", encoding="utf-8")
    loaded = CometConfig.from_ini(p)
    assert loaded.io.study_name == "solo"
    assert loaded.preprocessing == PreprocessingConfig()
    assert loaded.clustering == ClusteringConfig()


def test_pattern_alias_back_compat(tmp_path):
    """The ``pattern_content`` key should populate ``pattern``."""
    p = tmp_path / "alias.ini"
    p.write_text(
        "[io_config]\npattern_content = sub-*_eeg\n", encoding="utf-8"
    )
    loaded = CometConfig.from_ini(p)
    assert loaded.io.pattern == "sub-*_eeg"
