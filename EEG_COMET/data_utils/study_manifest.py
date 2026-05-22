"""Versioned ``study.json`` manifest writer/verifier.

A study folder consists of an ``eeg_comet_config.ini`` plus a fan-out of
intermediate artifacts (``*_clustering_results/``, ``*_segmentation/``,
``*_extracted_features/``, ``*_localized_sources/``). The manifest written
by this module records a SHA-256, size, and stage for each artifact so that
loaders can detect tampering or corruption before unpickling.

Manifest layout::

   {
     "format_version": "1.0",
     "study_name": "my_study",
     "created_at": "2026-04-18T13:30:00Z",
     "random_seed": 42,
     "artifacts": {
       "my_study_clustering_results/microstate_maps.csv": {
         "sha256": "...",
         "size_bytes": 12345,
         "stage": "clustering"
       }
     }
   }

:func:`record_artifact` adds or updates an entry; :func:`verify_artifact`
re-hashes the file and raises on mismatch. The on-disk format is JSON, so
the manifest itself is safe to load and human-readable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

MANIFEST_FILENAME = "study.json"
MANIFEST_FORMAT_VERSION = "1.0"


def _hash_file(path: Union[str, os.PathLike], algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def manifest_path(study_dir: Union[str, os.PathLike]) -> Path:
    """Resolve the ``study.json`` location for a given study directory."""
    return Path(study_dir) / MANIFEST_FILENAME


def load_manifest(study_dir: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Read the manifest, returning a default skeleton if it doesn't exist."""
    path = manifest_path(study_dir)
    if not path.is_file():
        return {
            "format_version": MANIFEST_FORMAT_VERSION,
            "study_name": Path(study_dir).name,
            "created_at": _now_utc(),
            "random_seed": None,
            "artifacts": {},
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Corrupt manifest at {path}: {exc}") from exc


def save_manifest(study_dir: Union[str, os.PathLike], manifest: Dict[str, Any]) -> None:
    """Atomically write the manifest to ``<study_dir>/study.json``."""
    path = manifest_path(study_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def record_artifact(
    study_dir: Union[str, os.PathLike],
    artifact_path: Union[str, os.PathLike],
    *,
    stage: str,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Hash ``artifact_path`` and record it under the study's manifest.

    The artifact path may be absolute or relative to ``study_dir``; it is
    stored relative (with forward slashes) so the manifest is portable.
    """
    study_dir = Path(study_dir).resolve()
    artifact_path = Path(artifact_path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    rel = artifact_path.relative_to(study_dir).as_posix()
    digest = _hash_file(artifact_path)
    size = artifact_path.stat().st_size

    manifest = load_manifest(study_dir)
    if random_seed is not None and manifest.get("random_seed") is None:
        manifest["random_seed"] = int(random_seed)
    manifest.setdefault("artifacts", {})[rel] = {
        "sha256": digest,
        "size_bytes": size,
        "stage": stage,
        "updated_at": _now_utc(),
    }
    save_manifest(study_dir, manifest)
    return manifest


def verify_artifact(
    study_dir: Union[str, os.PathLike],
    artifact_path: Union[str, os.PathLike],
) -> bool:
    """Verify ``artifact_path`` against the manifest.

    Returns ``True`` if the file matches the recorded SHA-256, or if the
    manifest has no entry for it. Raises ``RuntimeError`` on hash mismatch.
    """
    manifest = load_manifest(study_dir)
    artifacts = manifest.get("artifacts", {}) or {}
    rel = Path(artifact_path).resolve().relative_to(Path(study_dir).resolve()).as_posix()
    entry = artifacts.get(rel)
    if not entry or "sha256" not in entry:
        return True
    actual = _hash_file(artifact_path)
    expected = str(entry["sha256"]).lower()
    if actual.lower() != expected:
        raise RuntimeError(
            f"Manifest mismatch for {artifact_path}: expected sha256={expected}, "
            f"got {actual}. Refusing to load."
        )
    return True
