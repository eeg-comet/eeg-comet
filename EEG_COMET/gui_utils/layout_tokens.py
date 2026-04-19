"""Shared layout tokens for the EEG-COMET GUI.

Centralizes the small set of pixel values that the application is allowed to
use for sizing. Controllers and helpers should pull spacing, control heights,
font sizes, and window minimums from here instead of hard-coding numbers.
"""

from __future__ import annotations

from typing import Tuple


class Spacing:
    """Padding and gap units (pixels)."""

    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    XXL = 32


class Radius:
    """Corner radius units (pixels)."""

    SM = 4
    MD = 6
    LG = 8


class ControlHeight:
    """Standard heights for interactive controls (pixels)."""

    XS = 24
    SM = 30
    MD = 36
    LG = 44


class FontSize:
    """Logical font sizes in points."""

    CAPTION = 11
    BODY = 13
    SUBHEAD = 14
    HEAD = 16
    BANNER = 18


class WindowMin:
    """Approved minimum window sizes (width, height)."""

    TINY_LOG: Tuple[int, int] = (480, 320)
    DIALOG: Tuple[int, int] = (640, 480)
    TOOL: Tuple[int, int] = (800, 600)
    MAIN: Tuple[int, int] = (960, 640)


class SplitterRatio:
    """Default stretch ratios for two-pane splitters."""

    CONTROLS_CANVAS: Tuple[int, int] = (1, 3)
    LIST_DETAIL: Tuple[int, int] = (1, 2)
    EQUAL: Tuple[int, int] = (1, 1)


class Branding:
    """Decorative pixel sizes that intentionally do not scale."""

    LOGO_SMALL = 60
    LOGO_LARGE = 120


PADDING: Tuple[int, int, int, int] = (Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
GAP: int = Spacing.SM
PANE_MIN: Tuple[int, int] = (160, 120)


__all__ = [
    "Spacing",
    "Radius",
    "ControlHeight",
    "FontSize",
    "WindowMin",
    "SplitterRatio",
    "Branding",
    "PADDING",
    "GAP",
    "PANE_MIN",
]
