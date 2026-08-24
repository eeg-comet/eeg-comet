"""Regression tests for correctness bugs found in the full-repository audit.

Each test pins down one defect that produced wrong scientific output rather
than merely wrong style: polarity handling in the similarity clusterer and in
K-Means++ seeding, the quality-control threshold being undone by segment
filtering, rejected timepoints leaking into features as if they were a
microstate class, and the transition matrix not being conditional.

The tests exercise small pure functions and static methods, so none of them
needs a full clustering run. They skip cleanly when the heavy runtime stack is
absent.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def np_mod():
    return pytest.importorskip("numpy")


@pytest.fixture
def clusterer_cls():
    pytest.importorskip("numpy")
    module = pytest.importorskip("eeg_comet.clustering_utils.microstate_clusterer")
    return module.MicrostateClusterer


@pytest.fixture
def initializer_cls():
    pytest.importorskip("numpy")
    module = pytest.importorskip("eeg_comet.data_utils.data_initializer")
    return module.DataInitializer


@pytest.fixture
def backfitter_cls():
    pytest.importorskip("numpy")
    module = pytest.importorskip("eeg_comet.backfitting_utils.microstate_backfitter")
    return module.MicrostateBackfitter


@pytest.fixture
def extractor_cls():
    pytest.importorskip("numpy")
    module = pytest.importorskip("eeg_comet.features_utils.feature_extractor")
    return module.FeatureExtractor


@pytest.fixture
def helper_cls():
    module = pytest.importorskip("eeg_comet.features_utils.feature_helper")
    return module.FeatureHelper


def _make_backfitter(backfitter_cls, np, n_states=4, labels=None, **overrides):
    labels = labels or [chr(ord("A") + i) for i in range(n_states)]
    rng = np.random.default_rng(0)
    maps = rng.standard_normal((n_states, 20))
    maps -= maps.mean(axis=1, keepdims=True)
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    kwargs = dict(
        study_name="study",
        preprocessed_data_path=".",
        microstate_maps=maps,
        backfit_to="all",
        filter_segments=True,
        filter_segments_option="smooth",
        identify_short_window=False,
        microstate_labels=labels,
        segmentation_path=".",
        extension=".set",
        data_type="continuous",
        sampling_rate=250,
        smoothing_parameters=[1e-6, 3, 5],
        export_format=".csv",
        min_correlation_threshold=False,
    )
    kwargs.update(overrides)
    return backfitter_cls(**kwargs)


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def test_similarity_clustering_treats_flipped_map_as_identical(clusterer_cls, np_mod):
    """A sign-flipped topography must be the closest map, not the furthest.

    The similarity was computed as ``1 - |d|`` where ``d = 1 - s``, which maps a
    perfectly anti-correlated topography to -1 instead of +1.
    """
    np = np_mod
    template = np.array([1.0, -1.0, 0.5, -0.5, 0.25])
    template -= template.mean()
    template /= np.linalg.norm(template)
    other = np.array([0.2, 0.3, -0.9, 0.4, 0.1])
    other -= other.mean()
    other /= np.linalg.norm(other)
    maps = np.vstack([template, other])

    # Two samples: one matching template exactly, one exactly sign-flipped.
    data = np.column_stack([template, -template])

    clusterer = clusterer_cls(n_states=2, max_iterations=1, clustering_tolerance=1e-12)
    fitted, _ = clusterer.modified_kmeans_similarity(
        data, maps.copy(), metric="Spatial Correlation", verbose=False
    )

    # Both samples belong to the template cluster, so it must stay aligned with
    # the template up to sign, and must not collapse to a near-zero vector.
    assert np.linalg.norm(fitted[0]) == pytest.approx(1.0, abs=1e-6)
    assert abs(float(fitted[0] @ template)) == pytest.approx(1.0, abs=1e-6)


def test_similarity_batch_and_nonbatch_paths_agree(clusterer_cls, np_mod):
    """Batching is an implementation detail and must not change the objective."""
    np = np_mod
    rng = np.random.default_rng(11)
    n_states, n_channels, n_samples = 3, 16, 120

    maps = rng.standard_normal((n_states, n_channels))
    maps -= maps.mean(axis=1, keepdims=True)
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)

    labels = rng.integers(0, n_states, n_samples)
    signs = rng.choice([-1.0, 1.0], n_samples)
    data = (maps[labels].T * signs) + 0.05 * rng.standard_normal((n_channels, n_samples))

    init = maps.copy()
    plain = clusterer_cls(n_states=n_states, max_iterations=20)
    batched = clusterer_cls(n_states=n_states, max_iterations=20, batch_size=32)

    maps_plain, _ = plain.modified_kmeans_similarity(
        data, init.copy(), metric="Spatial Correlation", verbose=False
    )
    maps_batched, _ = batched.modified_kmeans_similarity(
        data, init.copy(), metric="Spatial Correlation", verbose=False
    )

    # Compare polarity-invariantly, map for map.
    agreement = np.abs(np.sum(maps_plain * maps_batched, axis=1))
    assert np.all(agreement > 0.99), agreement


def test_kmeanspp_does_not_seed_duplicate_maps(initializer_cls, np_mod):
    """K-Means++ must sample far-apart seeds, not near-duplicates.

    The weight was ``|corr|`` (a similarity) used directly as if it were a
    distance, so the most redundant candidate was the most likely pick.
    """
    np = np_mod
    # One dominant topography repeated many times, plus a few distinct ones.
    duplicated = np.tile(np.eye(8, 1), (1, 30))
    distinct = np.eye(8)[:, 1:5]
    pool = np.hstack([duplicated, distinct])
    pool = pool / (np.linalg.norm(pool, axis=0, keepdims=True) + 1e-12)

    np.random.seed(3)
    centers = initializer_cls.initialize_cluster_centers(pool, 4, "K-Means++")

    similarity = np.abs(centers @ centers.T)
    np.fill_diagonal(similarity, 0.0)
    assert similarity.max() < 0.99, (
        f"seeded near-duplicate centres (max |corr| = {similarity.max():.3f})"
    )


# --------------------------------------------------------------------------
# Backfitting
# --------------------------------------------------------------------------


def test_correlation_threshold_survives_segment_filtering(backfitter_cls, np_mod):
    """min_correlation_threshold must hold for every filter option.

    Only 'remove' used to preserve rejects; 'smooth' (the default),
    'replace_half' and 'replace_high' refilled them, silently disabling the
    quality-control threshold.
    """
    np = np_mod
    n_states, n_channels = 4, 20
    rng = np.random.default_rng(3)
    maps = rng.standard_normal((n_states, n_channels))
    maps -= maps.mean(axis=1, keepdims=True)
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)

    clean = maps[np.repeat(np.arange(n_states), 125)].T * rng.uniform(1, 3, size=500)
    noise = rng.standard_normal((n_channels, 100)) * 0.5
    data = np.hstack([clean, noise])

    for option in ("remove", "smooth", "replace_half", "replace_high"):
        backfitter = _make_backfitter(
            backfitter_cls,
            np,
            n_states=n_states,
            filter_segments_option=option,
            min_correlation_threshold=0.5,
        )
        backfitter.microstate_maps = maps
        segmentation = backfitter.backfit_to_all(data, filter_segments_less_than=3)
        assert np.any(segmentation == -1), (
            f"option={option!r} discarded every correlation-threshold rejection"
        )


def _correlation_threshold_fixture(backfitter_cls, np, **overrides):
    """Build data where some samples cannot clear a 0.5 correlation threshold."""
    n_states, n_channels = 4, 20
    rng = np.random.default_rng(3)
    maps = rng.standard_normal((n_states, n_channels))
    maps -= maps.mean(axis=1, keepdims=True)
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)

    clean = maps[np.repeat(np.arange(n_states), 125)].T * rng.uniform(1, 3, size=500)
    noise = rng.standard_normal((n_channels, 100)) * 0.5
    data = np.hstack([clean, noise])

    backfitter = _make_backfitter(
        backfitter_cls,
        np,
        n_states=n_states,
        min_correlation_threshold=0.5,
        **overrides,
    )
    backfitter.microstate_maps = maps
    return backfitter, data


def test_flag_mode_leaves_no_unassigned_samples(backfitter_cls, np_mod):
    """'flag' must relabel sub-threshold samples so event-related analyses see no gaps."""
    np = np_mod
    for option in ("smooth", "replace_half", "replace_high"):
        backfitter, data = _correlation_threshold_fixture(
            backfitter_cls,
            np,
            filter_segments_option=option,
            correlation_rejection_mode="flag",
        )
        segmentation = backfitter.backfit_to_all(data, filter_segments_less_than=3)
        assert not np.any(segmentation == -1), (
            f"option={option!r} left gaps in 'flag' mode"
        )


def test_reject_and_flag_modes_differ(backfitter_cls, np_mod):
    """The two modes must actually produce different segmentations."""
    np = np_mod
    rejected, data = _correlation_threshold_fixture(
        backfitter_cls, np, correlation_rejection_mode="reject"
    )
    flagged, _ = _correlation_threshold_fixture(
        backfitter_cls, np, correlation_rejection_mode="flag"
    )

    seg_reject = rejected.backfit_to_all(data, filter_segments_less_than=3)
    seg_flag = flagged.backfit_to_all(data, filter_segments_less_than=3)

    assert np.any(seg_reject == -1)
    assert not np.any(seg_flag == -1)


def test_unknown_correlation_rejection_mode_is_rejected(backfitter_cls, np_mod):
    """A typo must fail loudly rather than silently picking a behaviour."""
    with pytest.raises(ValueError, match="correlation_rejection_mode"):
        _make_backfitter(
            backfitter_cls, np_mod, correlation_rejection_mode="preserve"
        )


def test_correlation_rejection_mode_survives_a_config_round_trip():
    config_mod = pytest.importorskip("eeg_comet.config")
    import tempfile
    from pathlib import Path

    cfg = config_mod.CometConfig()
    assert cfg.backfitting.correlation_rejection_mode == "reject"
    cfg.backfitting.correlation_rejection_mode = "flag"

    path = Path(tempfile.mkdtemp()) / "cfg.ini"
    cfg.to_ini(path)
    reloaded = config_mod.CometConfig.from_ini(path)
    assert reloaded.backfitting.correlation_rejection_mode == "flag"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_independent_ttest_switches_to_welch_when_variances_differ():
    """ttest_ind was called without equal_var=False, so Welch never happened."""
    module = pytest.importorskip("eeg_comet.controllers.compare_studies_window")
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(0)
    tight = rng.normal(0.0, 1.0, 40)
    wide = rng.normal(0.5, 8.0, 40)
    _, _, used_welch = module.independent_ttest(tight, wide)
    assert used_welch

    similar_a = rng.normal(0.0, 1.0, 40)
    similar_b = rng.normal(0.3, 1.0, 40)
    _, _, used_welch = module.independent_ttest(similar_a, similar_b)
    assert not used_welch


def test_independent_ttest_matches_scipy_for_both_branches():
    module = pytest.importorskip("eeg_comet.controllers.compare_studies_window")
    scipy_stats = pytest.importorskip("scipy.stats")
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 30)
    b = rng.normal(0.4, 6.0, 30)

    statistic, p_value, used_welch = module.independent_ttest(a, b)
    expected = scipy_stats.ttest_ind(a, b, equal_var=not used_welch)
    assert statistic == pytest.approx(expected.statistic)
    assert p_value == pytest.approx(expected.pvalue)


def test_gee_family_is_taken_from_the_selected_model():
    """Gamma/log was imported but never reachable, so skewed outcomes got Gaussian."""
    module = pytest.importorskip("eeg_comet.controllers.compare_studies_window")

    assert module.gee_family_from_model_name(
        "Generalized Estimating Equations (GEE, Gamma/log)"
    ) == "gamma"
    assert module.gee_family_from_model_name(
        "Generalized Estimating Equations (GEE, Gaussian/identity)"
    ) == "gaussian"
    assert module.gee_family_from_model_name("Linear Mixed Model (LMM)") == "gaussian"


def test_replace_high_handles_gap_next_to_final_sample(backfitter_cls, np_mod):
    """A rejected run ending at index n-2 used to raise IndexError."""
    np = np_mod
    segmentation = np.array([0, 0, 0, -1, 1])
    filled = backfitter_cls.fill_with_neighbors_with_higher_count(segmentation)
    assert -1 not in filled.tolist()


def test_replace_high_picks_the_longer_neighbouring_run(backfitter_cls, np_mod):
    """The gap should be absorbed by the neighbour that actually dominates."""
    np = np_mod
    segmentation = np.array([0] * 12 + [-1] * 2 + [1] * 3)
    filled = backfitter_cls.fill_with_neighbors_with_higher_count(segmentation)
    assert filled[12] == 0 and filled[13] == 0

    mirrored = np.array([0] * 3 + [-1] * 2 + [1] * 12)
    filled_mirrored = backfitter_cls.fill_with_neighbors_with_higher_count(mirrored)
    assert filled_mirrored[3] == 1 and filled_mirrored[4] == 1


def test_label_segments_is_correct_for_ten_or_more_maps(backfitter_cls, np_mod):
    """Sequential substring replacement corrupted states 10 and above."""
    np = np_mod
    labels = [chr(ord("A") + i) for i in range(12)]
    backfitter = _make_backfitter(backfitter_cls, np, n_states=12, labels=labels)

    labelled = backfitter.label_segments(np.array([0, 8, 9, 10, 11, -1]))
    assert list(labelled) == ["A", "I", "J", "K", "L", "NaN"]


def test_label_segments_marks_rejected_samples(backfitter_cls, np_mod):
    np = np_mod
    backfitter = _make_backfitter(backfitter_cls, np, n_states=4)
    labelled = backfitter.label_segments(np.array([-1, 0, 3]))
    assert list(labelled) == ["NaN", "A", "D"]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def test_run_collapsing_preserves_multicharacter_labels(helper_cls):
    """Collapsing joined the sequence into a string and walked characters."""
    collapsed = helper_cls().remove_repetition_sequence(["A", "A", "NaN", "NaN", "B"])
    assert list(collapsed) == ["A", "NaN", "B"]

    multi = helper_cls().remove_repetition_sequence(["MS1", "MS1", "MS10", "MS1"])
    assert list(multi) == ["MS1", "MS10", "MS1"]


def test_coverage_excludes_rejected_samples(extractor_cls):
    """Rejected timepoints were reported as a microstate called 'NaN'."""
    coverage = extractor_cls(
        input_sequence=["A", "A", "NaN", "NaN", "B", "B"], sampling_rate=250
    ).microstate_coverage()

    assert "NaN" not in coverage
    assert sum(coverage.values()) == pytest.approx(100.0)
    assert coverage["A"] == pytest.approx(50.0)


def test_duration_excludes_rejected_samples_but_still_splits_segments(extractor_cls):
    """A rejection gap is not a state, yet it must still break a run in two."""
    extractor = extractor_cls(
        input_sequence=["A"] * 4 + ["NaN"] * 2 + ["A"] * 4,
        sampling_rate=250,
        duration_method="arithmetic",
    )
    durations = extractor.microstate_duration()

    assert "NaN" not in durations
    # Two runs of four samples each, not one run of eight.
    assert durations["A"] == pytest.approx((4.0 - 1.0) * (1000.0 / 250))


def test_occurrence_counts_segments_split_by_a_rejection_gap(extractor_cls):
    """Dropping rejects before collapsing would merge the two A segments."""
    occurrence = extractor_cls(
        input_sequence=["A"] * 4 + ["NaN"] * 2 + ["A"] * 4, sampling_rate=250
    ).microstate_occurrence()

    assert "NaN" not in occurrence
    # Eight assigned samples at 250 Hz = 0.032 s of analysable data, 2 segments.
    assert occurrence["A"] == pytest.approx(2 / (8 / 250))


def test_occurrence_does_not_invent_states_from_label_characters(extractor_cls):
    """'NaN' used to be shredded into spurious 'N' and 'a' states."""
    occurrence = extractor_cls(
        input_sequence=["A", "A", "NaN", "B", "B"], sampling_rate=250
    ).microstate_occurrence()
    assert set(occurrence) == {"A", "B"}


def test_transition_probabilities_are_row_conditional(extractor_cls):
    """TP is documented as P(next=j | current=i), so each row must sum to 1."""
    transitions = extractor_cls(
        input_sequence=list("ABACABAC"), sampling_rate=250
    ).compute_transition_probabilities()

    row_totals = {}
    for pair, value in transitions.items():
        source = pair.split("_")[0]
        row_totals[source] = row_totals.get(source, 0.0) + value

    for source, total in row_totals.items():
        assert total == pytest.approx(1.0), f"row {source} sums to {total}"

    # Leaving A, the sequence goes to B and to C equally often.
    assert transitions["A_B"] == pytest.approx(0.5)
    assert transitions["A_C"] == pytest.approx(0.5)


def test_transition_probabilities_ignore_rejected_samples(extractor_cls):
    transitions = extractor_cls(
        input_sequence=["A", "A", "NaN", "B", "B"], sampling_rate=250
    ).compute_transition_probabilities()
    assert all("NaN" not in pair for pair in transitions)


def test_mmd_is_accepted_as_an_alias_for_duration(extractor_cls):
    """'MMD' ships in the default feature_list but produced no duration."""
    sequence = list("AABBBCC")
    via_mmd = extractor_cls(input_sequence=sequence, sampling_rate=250)
    via_dur = extractor_cls(input_sequence=sequence, sampling_rate=250)

    mmd_frame = via_mmd.extract_microstate_features(filename="f", feature_list=["MMD"])
    dur_frame = via_dur.extract_microstate_features(filename="f", feature_list=["DUR"])

    mmd_keys = [key for key in mmd_frame if key.startswith("DUR")]
    assert mmd_keys, "requesting MMD produced no duration columns"
    assert sorted(mmd_keys) == sorted(key for key in dur_frame if key.startswith("DUR"))


def test_lempel_ziv_handles_sequences_too_short_to_index(helper_cls):
    """Recordings collapsing to one or two runs used to raise IndexError."""
    assert helper_cls().compute_lempel_ziv_complexity(["A"]) == 0.0
    assert helper_cls().compute_lempel_ziv_complexity(["A", "B"]) == 0.0


def test_sliding_duration_uses_the_same_windows_as_coverage(extractor_cls):
    """DUR kept a trailing partial window that COV and OCC discarded."""
    # 3.6 s at 250 Hz: three whole 1 s windows plus a 0.6 s remainder.
    sequence = (list("ABCD") * 226)[:900]
    extractor = extractor_cls(
        input_sequence=sequence,
        sampling_rate=250,
        feature_mode="sliding",
        sliding_window_size=1,
    )

    assert len(extractor.microstate_duration()) == len(extractor.microstate_coverage())
    assert len(extractor.microstate_duration()) == len(extractor.microstate_occurrence())
    assert len(extractor.microstate_duration()) == 3


# --------------------------------------------------------------------------
# Convergence and threshold consistency
# --------------------------------------------------------------------------


def test_clustering_never_returns_maps_worse_than_an_earlier_iteration(
    clusterer_cls, np_mod
):
    """A residual increase used to satisfy the convergence test and be kept."""
    np = np_mod
    rng = np.random.default_rng(5)
    n_states, n_channels, n_samples = 4, 24, 300

    maps = rng.standard_normal((n_states, n_channels))
    maps -= maps.mean(axis=1, keepdims=True)
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    labels = rng.integers(0, n_states, n_samples)
    signs = rng.choice([-1.0, 1.0], n_samples)
    data = (maps[labels].T * signs) + 0.2 * rng.standard_normal((n_channels, n_samples))

    init = rng.standard_normal((n_states, n_channels))
    init /= np.linalg.norm(init, axis=1, keepdims=True)

    clusterer = clusterer_cls(n_states=n_states, max_iterations=50)
    _, residual = clusterer.modified_kmeans(data, init.copy(), verbose=False)

    # The returned residual must be no worse than the starting configuration.
    activation = init.dot(data)
    seg = np.argmax(np.abs(activation), axis=0)
    act_sum_sq = np.sum(np.sum(init[seg].T * data, axis=0) ** 2)
    initial_residual = abs(np.sum(data**2) - act_sum_sq) / float(
        n_samples * (n_channels - 1)
    )
    assert residual <= initial_residual + 1e-12


def test_threshold_search_and_production_filter_agree(backfitter_cls, np_mod):
    """The search used '<' while the applied filter used '<=' at the boundary."""
    np = np_mod
    backfitter = _make_backfitter(backfitter_cls, np)
    segmentation = np.array([0] * 5 + [1] * 20)

    marked = backfitter.mark_short_segments(segmentation.copy(), 5)
    searched = backfitter._filter_short_segments(segmentation.copy(), 5)

    assert list(marked) == list(searched)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_random_seed_zero_is_preserved():
    """`_coerce_int(...) or None` silently turned a seed of 0 into unseeded."""
    config_mod = pytest.importorskip("eeg_comet.config")
    import tempfile
    from pathlib import Path

    cfg = config_mod.CometConfig()
    cfg.random_seed = 0

    path = Path(tempfile.mkdtemp()) / "cfg.ini"
    cfg.to_ini(path)
    assert config_mod.CometConfig.from_ini(path).random_seed == 0


def test_duration_method_survives_a_config_round_trip():
    """duration_method was read at runtime but absent from the typed config."""
    config_mod = pytest.importorskip("eeg_comet.config")
    import tempfile
    from pathlib import Path

    cfg = config_mod.CometConfig()
    cfg.features.duration_method = "median"

    path = Path(tempfile.mkdtemp()) / "cfg.ini"
    cfg.to_ini(path)
    assert config_mod.CometConfig.from_ini(path).features.duration_method == "median"


def test_shipped_defaults_match_the_typed_defaults():
    """default_config.ini and config.py drifted apart on four parameters."""
    config_mod = pytest.importorskip("eeg_comet.config")
    from configparser import ConfigParser
    from pathlib import Path

    ini_path = Path(config_mod.__file__).parent / "default_config.ini"
    parser = ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(ini_path, encoding="utf-8")

    cfg = config_mod.CometConfig()
    assert parser.getint("clustering_config", "data_percentage") == (
        cfg.clustering.data_percentage
    )
    assert parser.getboolean("clustering_config", "smoothing_gfp") == (
        cfg.clustering.smoothing_gfp
    )
    assert parser.getboolean("backfitting_config", "filter_segments") == (
        cfg.backfitting.filter_segments
    )
    assert parser.get("clustering_config", "stopping_mode").strip() == (
        cfg.clustering.stopping_mode
    )
    assert parser.get("features_config", "duration_method").strip() == (
        cfg.features.duration_method
    )


def test_correction_combo_labels_are_accepted_by_statsmodels():
    """FDR-BH, FDR-TSBH and FDR-TSBKY were passed through lowercased and raised."""
    pytest.importorskip("statsmodels")
    from statsmodels.stats.multitest import multipletests

    module = pytest.importorskip("eeg_comet.controllers.compare_studies_window")

    labels = ["Bonferroni", "Holm", "Sidak", "Holm-Sidak", "Hommel",
              "FDR-BH", "FDR-TSBH", "FDR-TSBKY"]
    for label in labels:
        method = module._correction_method(label)
        multipletests([0.01, 0.04, 0.2], method=method)
