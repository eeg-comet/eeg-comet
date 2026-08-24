"""Pytest configuration shared by the EEG-COMET test suite.

Ensures the repository root is importable so tests can use the installed
package layout, e.g. ``from eeg_comet.config import CometConfig``. When the
package has been installed (``pip install -e .``) this is redundant but
harmless; when running straight from a checkout it lets ``import eeg_comet``
resolve against the ``eeg_comet/`` directory at the repo root.

A deterministic synthetic EEG fixture (32 channels, 30 s @ 250 Hz) is
exposed for clustering and feature checks. It is built lazily so collection
still works in environments where ``numpy`` is not installed; tests that
need the data skip themselves via ``pytest.importorskip``.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_optional_runtime_dependencies():
    """Allow the pure-Python modules to be imported without the heavy stack.

    ``eeg_comet/__init__`` eagerly imports ``comet``, which pulls in the ONNX
    auto-labeler and the source-localization solver. Those are optional at
    analysis time but their absence would otherwise skip every test that only
    needs, say, the feature maths.
    """
    if "onnxruntime" not in sys.modules:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

    if "invert" not in sys.modules:
        try:
            import invert  # noqa: F401
        except ImportError:
            stub = types.ModuleType("invert")
            stub.Solver = object
            sys.modules["invert"] = stub


_stub_optional_runtime_dependencies()


@pytest.fixture(scope="session")
def synthetic_eeg():
    """Return a deterministic ``(n_channels, n_samples)`` numpy array.

    Two of the four hidden microstate templates dominate, so any sane
    clustering should recover non-trivial GEV / coverage values.
    """
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed=42)

    n_channels, sfreq, duration = 32, 250, 30
    n_samples = sfreq * duration

    templates = rng.standard_normal((4, n_channels))
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)

    state_seq = rng.choice(4, size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    amplitudes = np.abs(rng.standard_normal(n_samples)) + 0.1

    eeg = (templates[state_seq] * amplitudes[:, None]).T
    eeg += 0.05 * rng.standard_normal((n_channels, n_samples))
    eeg -= eeg.mean(axis=0, keepdims=True)
    return eeg


@pytest.fixture
def tmp_study_dir(tmp_path):
    """A throwaway study directory shaped like a real EEG-COMET output."""
    study = tmp_path / "my_study"
    study.mkdir()
    (study / "my_study_clustering_results").mkdir()
    (study / "my_study_extracted_features").mkdir()
    (study / "my_study_segmentation").mkdir()
    return study


@pytest.fixture(autouse=True)
def _reset_pickle_warning_state():
    """Each test starts from a clean ``safe_io`` warned-paths cache."""
    try:
        from eeg_comet.data_utils import safe_io  # type: ignore
    except Exception:
        yield
        return
    safe_io._WARNED_PATHS.clear()  # noqa: SLF001
    yield
    safe_io._WARNED_PATHS.clear()  # noqa: SLF001
    os.environ.pop("EEG_COMET_ALLOW_PICKLE", None)
