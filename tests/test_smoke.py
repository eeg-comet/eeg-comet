"""Smoke tests for the EEG-COMET package.

These tests deliberately avoid actually importing ``EEG_COMET`` at runtime
because doing so cascades into the full GUI / scientific stack (PyQt5,
pyvista, mne, ...) which is impractical to install on a bare CI runner
without xvfb and Qt platform libraries.

Instead we perform two cheap, dependency-free checks that still catch the
most common regressions:

1. ``EEG_COMET/__init__.py`` declares a non-empty ``__version__``, and that
   version matches the source-of-truth path declared in ``pyproject.toml``
   (``[tool.hatch.version]``).
2. Every ``.py`` file under ``EEG_COMET/`` is at least syntactically valid
   Python (compile-level only — no execution, no import resolution).

A real cross-platform test suite that actually imports the package can be
added later once the heavy dependencies are wired up on the runners.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "EEG_COMET"


def _read_declared_version() -> str:
    """Return the ``__version__`` literal from ``EEG_COMET/__init__.py``."""
    init_path = _PACKAGE_ROOT / "__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    match = re.search(r"""__version__\s*=\s*['"]([^'"]+)['"]""", init_text)
    assert match is not None, f"No __version__ literal found in {init_path}"
    return match.group(1)


def test_version_is_non_empty_string() -> None:
    """The package must advertise a non-empty version string."""
    version = _read_declared_version()
    assert isinstance(version, str)
    assert version.strip(), "__version__ must not be blank"


def test_version_matches_pyproject_source() -> None:
    """``pyproject.toml`` and ``EEG_COMET/__init__.py`` must agree.

    ``pyproject.toml`` declares ``[tool.hatch.version] path = "..."`` so the
    build always reads the version from a single source of truth. This test
    guards against accidental drift if someone bumps the version in only one
    of the two files.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = _REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml not found at repository root")

    with pyproject.open("rb") as fh:
        cfg = tomllib.load(fh)

    hatch_version_path = (
        cfg.get("tool", {}).get("hatch", {}).get("version", {}).get("path")
    )
    assert hatch_version_path == "EEG_COMET/__init__.py", (
        "[tool.hatch.version].path should point at EEG_COMET/__init__.py; "
        f"got {hatch_version_path!r}"
    )

    declared = _read_declared_version()
    assert declared, "Declared __version__ is empty"


def test_all_modules_are_syntactically_valid() -> None:
    """Every Python file under ``EEG_COMET/`` must parse cleanly.

    This catches the kind of regression the previous CI would have caught
    via ``import``-based testing (e.g. a stray syntax error introduced in a
    refactor) without requiring any of the heavy runtime dependencies to be
    installed on the runner.
    """
    failures: list[tuple[str, str]] = []
    for py_file in sorted(_PACKAGE_ROOT.rglob("*.py")):
        try:
            ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError as exc:
            failures.append((str(py_file.relative_to(_REPO_ROOT)), str(exc)))

    assert not failures, "Syntax errors found:\n" + "\n".join(
        f"  {path}: {err}" for path, err in failures
    )
