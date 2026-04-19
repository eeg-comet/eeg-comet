"""Safe wrappers around pickle-based deserialization.

``pickle.load`` and ``pd.read_pickle`` execute arbitrary code embedded in the
file, so loading a study folder produced by an untrusted party is equivalent
to running their Python script. This module centralizes that risk:

* Every load goes through ``safe_pickle_load`` / ``safe_pd_read_pickle``.
* A one-time warning is emitted per session (per file path).
* The user can pre-acknowledge the risk by setting the environment variable
  ``EEG_COMET_ALLOW_PICKLE=1``.
* If a sibling ``study.json`` manifest is present and lists a SHA-256 for the
  artifact, the hash is verified before unpickling and a mismatch raises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

_LOGGER = logging.getLogger(__name__)
_WARNED_PATHS: set[str] = set()


def _allow_pickle_via_env() -> bool:
    return os.environ.get("EEG_COMET_ALLOW_PICKLE", "").lower() in ("1", "true", "yes")


def _hash_file(path: str | os.PathLike[str], algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_manifest(path: str | os.PathLike[str]) -> bool:
    """Verify ``path`` against a sibling ``study.json`` manifest if present.

    Returns ``True`` when no manifest is found, when the manifest does not
    list this artifact, or when the recorded SHA-256 matches. Raises
    ``RuntimeError`` on a hash mismatch.
    """
    p = Path(path).resolve()
    for ancestor in [p.parent, *p.parent.parents]:
        manifest = ancestor / "study.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError):
                return True
            artifacts = data.get("artifacts", {}) or {}
            rel = str(p.relative_to(ancestor)).replace(os.sep, "/")
            entry = artifacts.get(rel)
            if entry and "sha256" in entry:
                actual = _hash_file(p, "sha256")
                if actual.lower() != str(entry["sha256"]).lower():
                    raise RuntimeError(
                        f"Refusing to unpickle {p}: SHA-256 mismatch against "
                        f"manifest {manifest} (got {actual!r}, "
                        f"expected {entry['sha256']!r})."
                    )
            return True
    return True


def _emit_warning_once(path: str | os.PathLike[str]) -> None:
    key = str(Path(path).resolve())
    if key in _WARNED_PATHS:
        return
    _WARNED_PATHS.add(key)
    if _allow_pickle_via_env():
        _LOGGER.info(
            "Loading pickle artifact %s (EEG_COMET_ALLOW_PICKLE=1 set).", key
        )
        return
    _LOGGER.warning(
        "Loading pickle artifact %s. Pickle deserialization can execute "
        "arbitrary code; only load .pkl files from sources you trust. "
        "Re-export this study with export_format=.csv to silence this warning.",
        key,
    )


def safe_pickle_load(path: str | os.PathLike[str]) -> Any:
    """Read a Python pickle file with a session warning and manifest check."""
    _verify_manifest(path)
    _emit_warning_once(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def safe_pd_read_pickle(path: str | os.PathLike[str]) -> "pd.DataFrame":
    """Read a pandas pickle (typically a DataFrame) with the same guards."""
    _verify_manifest(path)
    _emit_warning_once(path)
    return pd.read_pickle(path)
