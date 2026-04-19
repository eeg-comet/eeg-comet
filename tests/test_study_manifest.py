"""Tests for the versioned ``study.json`` manifest.

Verify the manifest's invariants:

- ``record_artifact`` produces a stable JSON document with relative paths.
- ``verify_artifact`` is permissive for un-recorded files.
- ``verify_artifact`` raises on a real hash mismatch.
- ``random_seed`` is preserved across writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def study_manifest():
    return pytest.importorskip("data_utils.study_manifest")


def test_record_and_verify_round_trip(tmp_path, study_manifest):
    artifact = tmp_path / "results" / "maps.csv"
    artifact.parent.mkdir()
    artifact.write_text("x,y\n1,2\n", encoding="utf-8")

    study_manifest.record_artifact(
        tmp_path, artifact, stage="clustering", random_seed=7
    )

    manifest = json.loads((tmp_path / "study.json").read_text(encoding="utf-8"))
    assert manifest["random_seed"] == 7
    assert "results/maps.csv" in manifest["artifacts"]
    assert manifest["artifacts"]["results/maps.csv"]["stage"] == "clustering"

    assert study_manifest.verify_artifact(tmp_path, artifact) is True


def test_verify_passes_for_unrecorded_artifact(tmp_path, study_manifest):
    art = tmp_path / "unknown.csv"
    art.write_text("noop", encoding="utf-8")
    assert study_manifest.verify_artifact(tmp_path, art) is True


def test_tampered_artifact_is_rejected(tmp_path, study_manifest):
    art = tmp_path / "data.bin"
    art.write_bytes(b"original")
    study_manifest.record_artifact(tmp_path, art, stage="features")

    art.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Manifest mismatch"):
        study_manifest.verify_artifact(tmp_path, art)


def test_random_seed_is_not_overwritten_once_set(tmp_path, study_manifest):
    art = tmp_path / "a.csv"
    art.write_text("a", encoding="utf-8")
    study_manifest.record_artifact(tmp_path, art, stage="x", random_seed=11)
    study_manifest.record_artifact(tmp_path, art, stage="x", random_seed=99)
    manifest = json.loads((tmp_path / "study.json").read_text(encoding="utf-8"))
    assert manifest["random_seed"] == 11
