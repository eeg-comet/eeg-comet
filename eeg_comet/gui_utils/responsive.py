"""Helpers that enforce a consistent responsive layout policy.

The functions in this module wrap the small set of operations that every
window in EEG-COMET should perform on its layouts so the behaviour stays
uniform: declare a minimum window size, scroll long forms instead of
clipping them, keep splitter panes from collapsing to zero, and persist
splitter sizes across sessions.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (
    QLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QWidget,
)

from .layout_tokens import PANE_MIN, SplitterRatio, WindowMin

_SETTINGS_ORG = "EEG-COMET"
_SETTINGS_APP = "EEG-COMET"


def apply_window_minimum(window: QWidget, kind: str = "tool") -> None:
    """Set a window's minimum size from the approved profiles in :class:`WindowMin`.

    Args:
      window: The window or dialog to configure.
      kind: One of ``"tiny_log"``, ``"dialog"``, ``"tool"`` (default), ``"main"``.
    """
    profile = {
        "tiny_log": WindowMin.TINY_LOG,
        "dialog": WindowMin.DIALOG,
        "tool": WindowMin.TOOL,
        "main": WindowMin.MAIN,
    }.get(kind.lower())
    if profile is None:
        raise ValueError(f"Unknown window minimum kind: {kind!r}")
    width, height = profile
    window.setMinimumSize(width, height)


def wrap_in_scroll_area(
    inner: QWidget,
    *,
    horizontal: Qt.ScrollBarPolicy = Qt.ScrollBarAsNeeded,
    vertical: Qt.ScrollBarPolicy = Qt.ScrollBarAsNeeded,
) -> QScrollArea:
    """Return a :class:`QScrollArea` that fully contains ``inner``.

    Sets ``widgetResizable=True`` so the inner widget tracks the viewport
    width and only scrolls when its preferred height exceeds the viewport.

    Args:
      inner: The widget to wrap. The caller is responsible for removing it
        from its previous parent layout before calling.
      horizontal: Horizontal scroll-bar policy.
      vertical: Vertical scroll-bar policy.
    """
    area = QScrollArea(inner.parentWidget())
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(horizontal)
    area.setVerticalScrollBarPolicy(vertical)
    area.setFrameShape(QScrollArea.NoFrame)
    inner.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
    )
    area.setWidget(inner)
    return area


def replace_with_scroll_area(inner: QWidget) -> QScrollArea:
    """Wrap ``inner`` in a scroll area while preserving its position in the
    parent layout.

    Looks up the parent layout, removes ``inner``, inserts an equivalent
    :class:`QScrollArea` in the same slot, and re-parents ``inner`` into
    the scroll area. Returns the new scroll area so the caller can keep a
    reference.
    """
    parent_layout = _layout_containing(inner)
    if parent_layout is None:
        return wrap_in_scroll_area(inner)

    index = parent_layout.indexOf(inner)
    item = parent_layout.takeAt(index)
    if item is not None:
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)

    area = wrap_in_scroll_area(inner)
    parent_layout.addWidget(area)
    return area


def scroll_wrap_tab(tab_widget: QTabWidget, index: int, *, title: Optional[str] = None) -> None:
    """Replace the page at ``index`` with a :class:`QScrollArea` wrapping the
    original page widget.

    Lets long forms inside a tab scroll vertically instead of being clipped
    when the user shrinks the window. Tab text and tooltip are preserved
    (or overridden via ``title``). The original page keeps its layout and
    is re-parented into the new scroll area.

    Args:
      tab_widget: The :class:`QTabWidget` to operate on.
      index: Index of the page to wrap.
      title: Optional override for the tab title; defaults to the existing
        tab text.
    """
    page = tab_widget.widget(index)
    if page is None:
        return
    label = title if title is not None else tab_widget.tabText(index)
    tooltip = tab_widget.tabToolTip(index)
    icon = tab_widget.tabIcon(index)
    enabled = tab_widget.isTabEnabled(index)

    page.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
    area = QScrollArea(tab_widget)
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setFrameShape(QScrollArea.NoFrame)

    tab_widget.removeTab(index)
    page.setParent(area)
    area.setWidget(page)
    tab_widget.insertTab(index, area, icon, label)
    tab_widget.setTabToolTip(index, tooltip)
    tab_widget.setTabEnabled(index, enabled)


def scroll_wrap_all_tabs(tab_widget: QTabWidget) -> None:
    """Apply :func:`scroll_wrap_tab` to every page in ``tab_widget``."""
    for index in range(tab_widget.count()):
        scroll_wrap_tab(tab_widget, index)


def configure_splitter(
    splitter: QSplitter,
    *,
    ratio: Sequence[int] = SplitterRatio.CONTROLS_CANVAS,
    collapsible: bool = False,
    pane_minimum: Tuple[int, int] = PANE_MIN,
    save_key: Optional[str] = None,
) -> None:
    """Apply the project-wide splitter policy.

    Sets per-pane stretch factors, prevents panes from collapsing to zero,
    enforces a minimum size on each pane, and (optionally) persists/restores
    the user's preferred sizes via :class:`QSettings`.

    Args:
      splitter: The splitter to configure.
      ratio: Stretch factors per pane. Padded with ``1`` if shorter than the
        splitter's pane count.
      collapsible: Whether panes are user-collapsible. Defaults to ``False``.
      pane_minimum: ``(width, height)`` minimum applied to each direct pane
        widget.
      save_key: Optional :class:`QSettings` key. When supplied, the splitter
        state is restored on configuration and saved whenever it moves.
    """
    count = splitter.count()
    if count == 0:
        return

    factors = list(ratio) + [1] * max(0, count - len(ratio))
    for index in range(count):
        splitter.setStretchFactor(index, factors[index])
        splitter.setCollapsible(index, collapsible)
        pane = splitter.widget(index)
        if pane is not None:
            min_w, min_h = pane_minimum
            current_min = pane.minimumSize()
            pane.setMinimumSize(
                max(min_w, current_min.width()), max(min_h, current_min.height())
            )

    if save_key:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        state = settings.value(f"splitter/{save_key}")
        if state is not None:
            try:
                splitter.restoreState(state)
            except Exception:
                pass

        def _save() -> None:
            QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(
                f"splitter/{save_key}", splitter.saveState()
            )

        splitter.splitterMoved.connect(lambda *_: _save())


def expand_canvas(canvas: QWidget, *, minimum: Tuple[int, int] = (320, 240)) -> None:
    """Apply the canonical size policy for a matplotlib/PyVista canvas.

    Args:
      canvas: The canvas widget.
      minimum: ``(width, height)`` floor so the canvas never disappears even
        when the splitter pane is dragged small.
    """
    canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
    width, height = minimum
    canvas.setMinimumSize(width, height)


def disable_show_maximized(window: QWidget) -> None:
    """Restore the system-default opening size by clearing any prior
    ``WindowMaximized`` state.

    Used when migrating windows away from unconditional ``showMaximized()``.
    """
    state = window.windowState()
    window.setWindowState(state & ~Qt.WindowMaximized)


def _layout_containing(widget: QWidget) -> Optional[QLayout]:
    """Return the layout that directly owns ``widget``, or ``None``."""
    parent = widget.parentWidget()
    if parent is None:
        return None
    layout = parent.layout()
    if layout is None:
        return None
    if layout.indexOf(widget) >= 0:
        return layout
    return _walk_layouts(layout, widget)


def _walk_layouts(layout: QLayout, widget: QWidget) -> Optional[QLayout]:
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        if item.widget() is widget:
            return layout
        child = item.layout()
        if child is not None:
            found = _walk_layouts(child, widget)
            if found is not None:
                return found
    return None


def iter_widgets(parent: QWidget) -> Iterable[QWidget]:
    """Yield ``parent`` and every descendant widget."""
    yield parent
    for child in parent.findChildren(QWidget):
        yield child


__all__ = [
    "apply_window_minimum",
    "wrap_in_scroll_area",
    "replace_with_scroll_area",
    "scroll_wrap_tab",
    "scroll_wrap_all_tabs",
    "configure_splitter",
    "expand_canvas",
    "disable_show_maximized",
    "iter_widgets",
]
