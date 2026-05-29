"""Application-wide UI element scale.

Provides :class:`UIScale`, a singleton-style helper that resizes every Qt
widget in the application by a single multiplicative factor. The factor is
applied two ways at the same time so both QSS-styled controls and widgets
with hard-coded inline fonts (from `.ui` files) scale together:

1. The shared ``ui/theme.qss`` template is re-substituted with scaled
   pixel/point values and reapplied via :meth:`QApplication.setStyleSheet`.
2. Every existing widget is visited; the widget's *original* point size is
   captured on the first visit and then ``widget.setFont(...)`` is called
   with ``base * factor``.

The current scale index is persisted in :class:`QSettings` under
``ui/scale_index`` so the choice survives restarts.
"""

from __future__ import annotations

import re
import string
from typing import Dict, Optional

from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget

_SETTINGS_ORG = "EEG-COMET"
_SETTINGS_APP = "EEG-COMET"
_SETTINGS_KEY = "ui/scale_index"

_BASE_FONT_PROP = "_eegc_base_font_pt"
_BASE_QSS_PROP = "_eegc_base_qss"

# Matches numeric values immediately followed by a Qt stylesheet length unit
# (pt or px). Used to rescale inline ``setStyleSheet`` blocks such as
# ``QPushButton { padding: 8px 16px; font-size: 14pt; border-radius: 6px; }``
# that would otherwise override the application-wide scaled QSS.
_QSS_LENGTH_RE = re.compile(r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)(pt|px)(?![A-Za-z])")


# Discrete scale steps (factor relative to the unscaled UI). 1.0 is the
# default; users move up/down through the list with the menu actions.
SCALE_STEPS = (0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)
DEFAULT_INDEX = SCALE_STEPS.index(1.0)


# Token name -> base value (point size for fonts, pixels for everything else).
# These are the values that, at scale 1.0, must reproduce the un-scaled QSS.
BASE_TOKENS: Dict[str, int] = {
    # Typography (pt)
    "font_default": 13,
    "font_caption": 11,
    "font_subhead": 14,
    "font_head": 16,
    "font_banner_lg": 18,
    "font_banner_md": 16,
    # Buttons (px)
    "pad_v_btn": 6,
    "pad_h_btn": 14,
    "ctrl_h_btn": 30,
    # Inputs (px)
    "pad_v_input": 4,
    "pad_h_input": 10,
    "ctrl_h_input": 28,
    # ComboBox (px)
    "drop_w": 28,
    "arrow_size": 12,
    # Indicators (px)
    "indicator_size": 18,
    "indicator_spacing": 8,
    # Splitter (px)
    "splitter_thick": 6,
    # Tabs (px)
    "pad_v_tab": 6,
    "pad_h_tab": 14,
    # Banner (px)
    "pad_v_banner": 6,
    "pad_h_banner": 12,
    "ctrl_h_banner": 36,
    # GroupBox (px)
    "groupbox_margin": 12,
    "groupbox_pad_top": 12,
    "groupbox_title_pad_h": 6,
    # ProgressBar (px)
    "ctrl_h_progress": 22,
}


class UIScale:
    """Manage a global UI scale factor and apply it to the running app.

    Args:
      app: The :class:`QApplication` instance whose stylesheet and default
        font will be controlled.
      base_qss_template: Contents of ``ui/theme.qss``. The template uses
        ``${name}`` placeholders for the entries in :data:`BASE_TOKENS`.
      base_font_pt: Default Qt application font size (the size used at
        scale 1.0 for widgets that don't carry their own inline font).
      font_family: Default font family for the application.
      theme_qss: Optional theme override stylesheet (light/dark) appended
        to the scaled base QSS.
    """

    def __init__(
        self,
        app: QApplication,
        base_qss_template: str,
        *,
        base_font_pt: int = 14,
        font_family: str = "Calibri",
        theme_qss: str = "",
    ) -> None:
        self.app = app
        self.base_qss_template = base_qss_template
        self.base_font_pt = base_font_pt
        self.font_family = font_family
        self.theme_qss = theme_qss
        self._template = string.Template(base_qss_template)
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        try:
            stored = int(settings.value(_SETTINGS_KEY, DEFAULT_INDEX))
        except (TypeError, ValueError):
            stored = DEFAULT_INDEX
        self._index = max(0, min(len(SCALE_STEPS) - 1, stored))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def factor(self) -> float:
        """Current scale multiplier."""
        return SCALE_STEPS[self._index]

    @property
    def index(self) -> int:
        """Index into :data:`SCALE_STEPS`."""
        return self._index

    def set_theme(self, theme_qss: str) -> None:
        """Set the theme override QSS and reapply the combined stylesheet."""
        self.theme_qss = theme_qss
        self.apply()

    def increase(self) -> None:
        """Step up to the next discrete scale, if any."""
        if self._index < len(SCALE_STEPS) - 1:
            self._index += 1
            self.apply(persist=True)

    def decrease(self) -> None:
        """Step down to the previous discrete scale, if any."""
        if self._index > 0:
            self._index -= 1
            self.apply(persist=True)

    def reset(self) -> None:
        """Return to the unscaled (1.0) default."""
        if self._index != DEFAULT_INDEX:
            self._index = DEFAULT_INDEX
            self.apply(persist=True)

    def apply(self, *, persist: bool = False) -> None:
        """Render and apply the scaled stylesheet plus default font.

        Args:
          persist: When ``True``, also write the current scale index to
            :class:`QSettings`.
        """
        self.app.setStyleSheet(self._render_qss())
        self.app.setFont(self._scaled_default_font())
        for widget in self.app.allWidgets():
            self._scale_widget_font(widget)
            self._scale_widget_stylesheet(widget)
            widget.updateGeometry()
        if persist:
            QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(
                _SETTINGS_KEY, self._index
            )

    def attach_to_new_widget(self, widget: QWidget) -> None:
        """Hook a freshly created widget into the scale.

        Call when you build a window after :meth:`apply`; the helper
        records the widget's base font and updates it to the current
        scale immediately.
        """
        self._scale_widget_font(widget)
        self._scale_widget_stylesheet(widget)
        for child in widget.findChildren(QWidget):
            self._scale_widget_font(child)
            self._scale_widget_stylesheet(child)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scaled_tokens(self) -> Dict[str, int]:
        f = self.factor
        return {name: max(1, int(round(value * f))) for name, value in BASE_TOKENS.items()}

    def _render_qss(self) -> str:
        try:
            scaled_base = self._template.safe_substitute(self._scaled_tokens())
        except (KeyError, ValueError):
            scaled_base = self.base_qss_template
        if self.theme_qss:
            return scaled_base + "\n" + self.theme_qss
        return scaled_base

    def _scaled_default_font(self) -> QFont:
        font = QFont(self.font_family)
        font.setPointSizeF(self.base_font_pt * self.factor)
        return font

    def _scale_widget_font(self, widget: QWidget) -> None:
        """Set ``widget``'s font to ``base_pt * factor``.

        The first time a widget is visited its current point size is
        recorded as the widget's base size, so subsequent applies always
        scale from the original (avoiding compounding rounding errors).
        """
        base_pt = self._base_pt_for(widget)
        if base_pt is None:
            return
        font = widget.font()
        font.setPointSizeF(base_pt * self.factor)
        widget.setFont(font)

    def _scale_widget_stylesheet(self, widget: QWidget) -> None:
        """Rescale every ``Npt``/``Npx`` value in a widget's inline stylesheet.

        Per-widget ``setStyleSheet`` blocks defined in ``.ui`` files (e.g.,
        ``QPushButton { padding: 8px 16px; font-size: 14pt; }``) take
        precedence over the application-wide stylesheet for the properties
        they declare. Without this rewrite they would visually freeze
        button paddings and font sizes regardless of the current scale.
        """
        base_qss = self._base_qss_for(widget)
        if not base_qss:
            return
        scaled = _QSS_LENGTH_RE.sub(self._scale_length_match, base_qss)
        if scaled != widget.styleSheet():
            widget.setStyleSheet(scaled)

    def _scale_length_match(self, match: "re.Match[str]") -> str:
        value = float(match.group(1))
        unit = match.group(2)
        scaled = max(1, int(round(value * self.factor)))
        return f"{scaled}{unit}"

    def _base_qss_for(self, widget: QWidget) -> str:
        """Return the original (unscaled) stylesheet text for ``widget``.

        On first visit we cache whatever ``styleSheet()`` currently returns
        on a Qt dynamic property; subsequent visits read from that cache so
        repeated rescales always start from the original values.
        """
        stored = widget.property(_BASE_QSS_PROP)
        if stored is not None:
            return str(stored)
        current = widget.styleSheet() or ""
        widget.setProperty(_BASE_QSS_PROP, current)
        return current

    def _base_pt_for(self, widget: QWidget) -> Optional[float]:
        stored = widget.property(_BASE_FONT_PROP)
        if stored is not None:
            try:
                return float(stored)
            except (TypeError, ValueError):
                pass
        font = widget.font()
        size = font.pointSizeF()
        if size <= 0:
            size = float(self.base_font_pt)
        widget.setProperty(_BASE_FONT_PROP, size)
        return size


__all__ = [
    "UIScale",
    "SCALE_STEPS",
    "DEFAULT_INDEX",
    "BASE_TOKENS",
]
