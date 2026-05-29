"""Tests for ``eeg_comet/data_utils/safe_io.py``.

Cover the warning behaviour, the env-var opt-in, and the manifest
hash-mismatch rejection.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import pytest


@pytest.fixture
def safe_io():
    return pytest.importorskip("eeg_comet.data_utils.safe_io")


def _write_pickle(path: Path, payload):
    path.write_bytes(pickle.dumps(payload))


def test_safe_pickle_load_returns_original_object(tmp_path, safe_io):
    p = tmp_path / "obj.pkl"
    _write_pickle(p, {"k": [1, 2, 3]})
    assert safe_io.safe_pickle_load(p) == {"k": [1, 2, 3]}


def test_warning_emitted_once_per_path(tmp_path, safe_io, caplog):
    p = tmp_path / "obj.pkl"
    _write_pickle(p, [1])
    with caplog.at_level(logging.WARNING, logger=safe_io.__name__):
        safe_io.safe_pickle_load(p)
        safe_io.safe_pickle_load(p)
    warnings = [r for r in caplog.records if "Pickle deserialization" in r.getMessage()]
    assert len(warnings) == 1


def test_env_var_silences_warning(tmp_path, safe_io, caplog):
    os.environ["EEG_COMET_ALLOW_PICKLE"] = "1"
    p = tmp_path / "obj.pkl"
    _write_pickle(p, "x")
    with caplog.at_level(logging.WARNING, logger=safe_io.__name__):
        safe_io.safe_pickle_load(p)
    warnings = [r for r in caplog.records if "Pickle deserialization" in r.getMessage()]
    assert warnings == []


def test_manifest_mismatch_raises(tmp_path, safe_io):
    import json

    art = tmp_path / "obj.pkl"
    _write_pickle(art, [1, 2])
    (tmp_path / "study.json").write_text(
        json.dumps({"artifacts": {"obj.pkl": {"sha256": "deadbeef"}}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        safe_io.safe_pickle_load(art)
