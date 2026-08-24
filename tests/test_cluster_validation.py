"""Regression tests for the cluster-number validation criteria.

These cover the parts of ``ClustererOptimizer`` that decide how many microstate
classes a dataset supports: the polarity invariance every criterion depends on,
the two criteria with their own published selection rules (Gap and
Krzanowski-Lai), and the consensus tally shared by the automatic pipeline and
the optimizer visualization window.

The metric implementations are static methods, so nothing here needs a full
clustering run. Tests skip cleanly when the heavy runtime stack is absent.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def optimizer_module():
    pytest.importorskip("numpy")
    return pytest.importorskip("eeg_comet.clustering_utils.clusterer_optimizer")


@pytest.fixture
def optimizer_cls(optimizer_module):
    return optimizer_module.ClustererOptimizer


@pytest.fixture
def polarity_case():
    """Structured topographies with arbitrary per-sample voltage signs."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)

    n_channels, n_clusters, n_samples = 24, 4, 400
    maps = rng.standard_normal((n_clusters, n_channels))
    maps -= maps.mean(axis=1, keepdims=True)
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)

    labels = rng.integers(0, n_clusters, n_samples)
    signs = rng.choice([-1.0, 1.0], n_samples)
    data = (maps[labels] * signs[:, None]).T + 0.2 * rng.standard_normal(
        (n_channels, n_samples)
    )
    data -= data.mean(axis=0, keepdims=True)
    return data, labels, maps


def _flip_polarity(data, maps, seed=11):
    """Flip the sign of half the samples and of every template."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    sample_signs = rng.choice([-1.0, 1.0], data.shape[1])
    return data * sample_signs, -maps


@pytest.mark.parametrize(
    "metric",
    ["davies_bouldin", "silhouette", "dunn", "calinski_harabasz", "aic", "bic", "cv"],
)
def test_criteria_are_polarity_invariant(optimizer_cls, polarity_case, metric):
    """Sign-flipping samples and templates must not change any criterion."""
    np = pytest.importorskip("numpy")
    data, labels, maps = polarity_case
    flipped_data, flipped_maps = _flip_polarity(data, maps)

    def score(d, m):
        if metric == "davies_bouldin":
            return optimizer_cls.compute_custom_davies_bouldin(d, labels, m)
        if metric == "silhouette":
            return optimizer_cls.silhouette_coefficient_correlation(d, labels)
        if metric == "dunn":
            return optimizer_cls.compute_dunn_index(d, labels, m)
        if metric == "calinski_harabasz":
            return optimizer_cls.compute_calinski_harabasz_index(d, labels, m)
        if metric == "cv":
            return optimizer_cls._compute_cross_validation_criterion_vectorized(d, m, labels)
        return optimizer_cls.compute_information_criteria(
            d, labels, m, criterion=metric.upper()
        )

    original = score(data, maps)
    flipped = score(flipped_data, flipped_maps)

    assert np.isfinite(original)
    assert original == pytest.approx(flipped, rel=1e-9, abs=1e-9)


def test_within_dispersion_is_polarity_invariant(optimizer_cls, polarity_case):
    np = pytest.importorskip("numpy")
    data, labels, _ = polarity_case
    flipped, _ = _flip_polarity(data, np.zeros((1, data.shape[0])))

    original = optimizer_cls._within_dispersion(data, labels, 4)
    assert original > 0
    assert original == pytest.approx(
        optimizer_cls._within_dispersion(flipped, labels, 4), rel=1e-9
    )


def test_kl_and_gap_share_one_dispersion_definition(optimizer_cls, polarity_case):
    """_compute_W_q must be _within_dispersion, not a second implementation."""
    data, labels, maps = polarity_case

    w_q = optimizer_cls._compute_W_q(data, labels, maps, max_samples=None)
    w_dispersion = optimizer_cls._within_dispersion(data, labels, maps.shape[0])

    assert w_q == pytest.approx(w_dispersion, rel=1e-12)


def test_gap_separates_structure_from_noise(optimizer_cls, polarity_case):
    """A real cluster structure must score above an unstructured baseline."""
    np = pytest.importorskip("numpy")
    data, labels, maps = polarity_case

    gap, s_k = optimizer_cls.compute_gap_statistic(data, labels, maps)

    rng = np.random.default_rng(3)
    noise = rng.standard_normal(data.shape)
    noise -= noise.mean(axis=0, keepdims=True)
    noise_labels = np.argmax(np.abs(maps @ noise), axis=0)
    noise_gap, _ = optimizer_cls.compute_gap_statistic(noise, noise_labels, maps)

    assert gap > 0
    assert s_k >= 0
    assert gap > noise_gap


def test_gap_statistic_is_reproducible(optimizer_cls, polarity_case):
    data, labels, maps = polarity_case
    first = optimizer_cls.compute_gap_statistic(data, labels, maps)
    second = optimizer_cls.compute_gap_statistic(data, labels, maps)
    assert first == second


def test_gap_uses_tibshirani_one_standard_error_rule(optimizer_cls):
    """The rule stops at the first k the next solution fails to beat."""
    k_values = [2, 3, 4, 5]
    gaps = [0.10, 0.50, 0.52, 0.55]
    standard_errors = [0.01, 0.01, 0.05, 0.01]

    # Gap(3) = 0.50 >= Gap(4) - s_4 = 0.47, so k=3 wins even though 5 is maximal.
    assert optimizer_cls._find_gap_optimal_k(k_values, gaps, standard_errors) == 3


def test_gap_falls_back_to_maximum_when_rule_never_fires(optimizer_cls):
    k_values = [2, 3, 4, 5]
    gaps = [0.1, 0.2, 0.3, 0.4]
    tiny = [0.001] * 4

    assert optimizer_cls._find_gap_optimal_k(k_values, gaps, tiny) == 5
    assert optimizer_cls._find_gap_optimal_k(k_values, gaps, None) == 5


def _make_optimizer(optimizer_cls, k_min=2, k_max=5):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    data = rng.standard_normal((8, 60))
    return optimizer_cls(data, k_min=k_min, k_max=k_max)


def test_krzanowski_lai_matches_published_ratio(optimizer_cls):
    """KL(q) = |DIFF(q)| / |DIFF(q+1)| with DIFF(q) = M_{q-1} - M_q."""
    np = pytest.importorskip("numpy")
    optimizer = _make_optimizer(optimizer_cls, k_min=2, k_max=5)

    scores = optimizer._compute_kl_scores_from_M_values({2: 10.0, 3: 6.0, 4: 4.0, 5: 3.5})

    assert np.isnan(scores[0])  # k_min has no preceding solution
    assert np.isnan(scores[-1])  # k_max has no following solution
    assert scores[1] == pytest.approx(abs(10.0 - 6.0) / abs(6.0 - 4.0))
    assert scores[2] == pytest.approx(abs(6.0 - 4.0) / abs(4.0 - 3.5))


def test_krzanowski_lai_abstains_on_degenerate_denominator(optimizer_cls):
    np = pytest.importorskip("numpy")
    optimizer = _make_optimizer(optimizer_cls, k_min=2, k_max=4)

    # M is flat between k=3 and k=4, making the published ratio undefined.
    scores = optimizer._compute_kl_scores_from_M_values({2: 10.0, 3: 6.0, 4: 6.0})
    assert np.isnan(scores[1])


def test_consensus_prefers_smaller_k_on_ties(optimizer_cls):
    votes = {"gev": 4, "cv": 4, "sil": 6, "dunn": 6}
    consensus, tally, excluded = optimizer_cls.tally_consensus(votes, k_min=2, k_max=10)

    assert consensus == 4
    assert tally == {4: 2, 6: 2}
    assert excluded == []


def test_consensus_sets_aside_boundary_picks(optimizer_cls):
    votes = {"db": 2, "ch": 2, "dunn": 2, "gev": 5, "cv": 6}
    consensus, tally, excluded = optimizer_cls.tally_consensus(votes, k_min=2, k_max=10)

    # Three criteria bottomed out at k_min; the interior ones decide, and the
    # tie between 5 and 6 resolves to the more parsimonious solution.
    assert consensus == 5
    assert sorted(excluded) == ["ch", "db", "dunn"]
    assert tally == {5: 1, 6: 1}


def test_consensus_falls_back_when_every_criterion_hits_a_boundary(optimizer_cls):
    votes = {"db": 2, "ch": 10, "dunn": 2}
    consensus, tally, excluded = optimizer_cls.tally_consensus(votes, k_min=2, k_max=10)

    assert consensus == 2
    assert tally == {2: 2, 10: 1}
    assert sorted(excluded) == ["ch", "db", "dunn"]


def test_consensus_ignores_criteria_without_a_vote(optimizer_cls):
    consensus, tally, _ = optimizer_cls.tally_consensus(
        {"gev": None, "cv": 5}, k_min=2, k_max=10
    )
    assert consensus == 5
    assert tally == {5: 1}


def test_consensus_reports_nothing_when_there_are_no_votes(optimizer_cls):
    consensus, tally, _ = optimizer_cls.tally_consensus({"gev": None}, k_min=2, k_max=10)
    assert consensus is None
    assert tally == {}


def test_valid_stopping_modes_match_the_documented_set(optimizer_module):
    """Guards against config docs drifting from what find_optimal_k accepts."""
    assert set(optimizer_module.VALID_STOPPING_MODES) == {
        "majority_vote",
        "gev",
        "cv",
        "db",
        "kl",
        "sil",
        "dunn",
        "ch",
        "gap",
        "aic",
        "bic",
    }
    # 'residual' was advertised in the config guide but never implemented.
    assert "residual" not in optimizer_module.VALID_STOPPING_MODES


def test_gev_threshold_selects_last_worthwhile_cluster(optimizer_cls):
    """stopping_threshold stops once an extra class buys less than the gain."""
    k_values = [2, 3, 4, 5, 6]
    # Relative gains: 33%, 25%, 4%, 4%.
    gev = [0.30, 0.40, 0.50, 0.52, 0.54]

    assert optimizer_cls._find_gev_threshold_elbow(k_values, gev, threshold=10.0) == 4
    assert optimizer_cls._find_gev_threshold_elbow(k_values, gev, threshold=1.0) == 6
