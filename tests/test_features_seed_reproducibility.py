"""Reproducibility checks for the seeded :class:`FeatureExtractionCoordinator`.

Confirms that constructing the coordinator with the same ``random_seed``
yields the same sequence of random draws, which is what guarantees identical
surrogate / random-baseline features across runs.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def coordinator_cls():
    pytest.importorskip("numpy")
    fe = pytest.importorskip("eeg_comet.features_utils.feature_extractor")
    return fe.FeatureExtractionCoordinator


def _draws(cls, seed):
    np = pytest.importorskip("numpy")
    coord = cls(random_seed=seed)
    rng_arr = coord._rng.standard_normal(8)  # noqa: SLF001
    arr = np.arange(10)
    coord._shuffle(arr)  # noqa: SLF001
    return rng_arr.tolist(), arr.tolist()


def test_same_seed_same_draws(coordinator_cls):
    a = _draws(coordinator_cls, 123)
    b = _draws(coordinator_cls, 123)
    assert a == b


def test_different_seeds_different_draws(coordinator_cls):
    a = _draws(coordinator_cls, 1)
    b = _draws(coordinator_cls, 2)
    assert a != b
