"""Defensive parsing helpers for Qt form inputs.

Provides :func:`parse_int` and :func:`parse_float` helpers that read text
from a Qt input widget (anything exposing ``.text()``), parse it to the
requested numeric type, clamp to an optional ``[minimum, maximum]`` range,
and fall back to a supplied default on empty or invalid input. A
``QMessageBox`` warning can optionally be shown to the user.
"""

from __future__ import annotations

from typing import Any

try:
    from PyQt5.QtWidgets import QMessageBox

    _HAVE_QT = True
except ImportError:  # pragma: no cover
    _HAVE_QT = False


def _widget_text(widget: Any) -> str:
    if widget is None:
        return ""
    if hasattr(widget, "text"):
        try:
            return str(widget.text()).strip()
        except Exception:
            return ""
    return str(widget).strip()


def _maybe_warn(parent, title: str, message: str, show_dialog: bool) -> None:
    if not show_dialog or not _HAVE_QT:
        return
    try:
        QMessageBox.warning(parent, title, message)
    except Exception:
        pass


def parse_int(
    widget: Any,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    field_name: str = "value",
    show_dialog: bool = False,
    parent: Any = None,
) -> int:
    """Parse the text of ``widget`` as an int, clamped to ``[minimum, maximum]``.

    Returns ``default`` when the text is empty or fails to parse. If
    ``show_dialog`` is True, a Qt ``QMessageBox`` is shown when the input is
    invalid or out of range; otherwise the failure is silent.
    """
    text = _widget_text(widget)
    if not text:
        return default
    try:
        value = int(float(text))
    except (TypeError, ValueError):
        _maybe_warn(
            parent,
            "Invalid input",
            f"{field_name!r}: expected an integer, got {text!r}. Using default {default}.",
            show_dialog,
        )
        return default
    if minimum is not None and value < minimum:
        _maybe_warn(
            parent,
            "Out of range",
            f"{field_name!r} must be >= {minimum}; clamped from {value} to {minimum}.",
            show_dialog,
        )
        value = minimum
    if maximum is not None and value > maximum:
        _maybe_warn(
            parent,
            "Out of range",
            f"{field_name!r} must be <= {maximum}; clamped from {value} to {maximum}.",
            show_dialog,
        )
        value = maximum
    return value


def parse_float(
    widget: Any,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    field_name: str = "value",
    show_dialog: bool = False,
    parent: Any = None,
) -> float:
    """Parse the text of ``widget`` as a float; same semantics as :func:`parse_int`."""
    text = _widget_text(widget)
    if not text:
        return default
    try:
        value = float(text)
    except (TypeError, ValueError):
        _maybe_warn(
            parent,
            "Invalid input",
            f"{field_name!r}: expected a number, got {text!r}. Using default {default}.",
            show_dialog,
        )
        return default
    if minimum is not None and value < minimum:
        _maybe_warn(
            parent,
            "Out of range",
            f"{field_name!r} must be >= {minimum}; clamped from {value} to {minimum}.",
            show_dialog,
        )
        value = minimum
    if maximum is not None and value > maximum:
        _maybe_warn(
            parent,
            "Out of range",
            f"{field_name!r} must be <= {maximum}; clamped from {value} to {maximum}.",
            show_dialog,
        )
        value = maximum
    return value
