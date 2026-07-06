"""Logging window with hierarchical formatting and smart navigation.

Appended messages are kept as an ordered list of :class:`LogRecord`
objects (severity, processing step, message, owning section, and the
section timestamp). That model drives both the :class:`QTextEdit` body
and a step-grouped sidebar tree, and is persisted as a JSON sidecar so a
reloaded session rebuilds identically. Each rendered block still carries
:class:`LogBlockData` so the severity/step/search filters can hide lines
without rebuilding the document.

Public API (unchanged for callers outside this module):
    * :meth:`LogWindow.append_log` ``(message, log_type="info", step=None)``
    * :meth:`LogWindow.replace_log`
    * :meth:`LogWindow.set_log_content`
    * :meth:`LogWindow.get_log_content`
    * :meth:`LogWindow.setup_progress_dialog`
    * :meth:`LogWindow.update_progress`
    * :meth:`LogWindow.process_finished`
    * :meth:`LogWindow.set_progress_label` / ``set_progress_text`` /
      ``set_progress_max`` / ``set_progress_value`` /
      ``set_progress_stop_enabled``
    * :meth:`LogWindow.set_window_title` / ``set_label_text``
    * :meth:`LogWindow.stop_process` / ``show_hide_log_window`` /
      ``clear_logs`` / ``export_logs``
    * :meth:`LogWindow.log_clustering_setup_step` /
      ``log_clustering_progress`` / ``show_study_status``
"""

import json
import os.path
import re
import traceback
from datetime import datetime
from typing import List, Optional, Set

from PyQt5 import uic
from PyQt5.QtCore import (
    Q_ARG,
    QCoreApplication,
    QMetaObject,
    QSettings,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QTextBlockFormat,
    QTextBlockUserData,
    QTextCharFormat,
    QTextCursor,
)
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QSplitter,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from eeg_comet.gui_utils.responsive import apply_window_minimum


def _call_with_optional_worker(func, args, worker):
    """Invoke ``func`` with the given positional ``args``, passing ``worker=`` if accepted."""
    try:
        return func(*args, worker=worker)
    except TypeError:
        # The processing function doesn't accept ``worker``; call without it.
        return func(*args)


class Worker(QThread):
    """Background worker for step-based or task-based processing.

    Emits progress, finished, and error signals during execution. Supports two
    modes: step-based (when ``tasks`` is an ``int``) and task-based (when
    ``tasks`` is an iterable of work items).

    Signals:
      progress_updated (pyqtSignal): Emitted with (int value, str text).
      finished (pyqtSignal): Emitted with (str message, bool success).
      error (pyqtSignal): Emitted with a traceback string when an unhandled
        exception is raised by the processing function.
    """

    progress_updated = pyqtSignal(int, str)
    # ``finished(message, success)``: ``success`` is False when the processing
    # function returned False or raised, so the GUI can distinguish real
    # failures from successful runs.
    finished = pyqtSignal(str, bool)
    error = pyqtSignal(str)

    def __init__(self, tasks, processing_func):
        super().__init__()
        self.tasks = tasks
        self.processing_func = processing_func
        self.stopped = False
        if isinstance(tasks, int):
            self.total_tasks = tasks
            self.is_step_based = True
        else:
            self.total_tasks = len(tasks) if hasattr(tasks, "__len__") else 1
            self.is_step_based = False
        self.dynamic_total = None

    @staticmethod
    def _result_is_success(result) -> bool:
        if result is False:
            return False
        if isinstance(result, tuple) and result and result[0] is False:
            return False
        return True

    @staticmethod
    def _result_message(result, default: str) -> str:
        if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], str):
            return result[1]
        if isinstance(result, str):
            return result
        return default

    def run(self):
        if self.is_step_based:
            try:
                result = _call_with_optional_worker(
                    self.processing_func, ("step_based_processing",), self
                )
            except Exception:
                tb = traceback.format_exc()
                self.error.emit(tb)
                self.finished.emit("Processing failed (see log for details).", False)
                return

            if self.stopped:
                self.finished.emit("Process stopped by user.", False)
                return

            success = self._result_is_success(result)
            message = self._result_message(
                result,
                "Processing completed successfully."
                if success
                else "Processing finished with errors (see log).",
            )
            self.finished.emit(message, success)
            return

        total_tasks = len(self.tasks)
        all_success = True
        for idx, task in enumerate(self.tasks, start=1):
            if self.stopped:
                self.finished.emit("Process stopped by user.", False)
                return

            args = task if (isinstance(task, (tuple, list)) and not isinstance(task, str)) else (task,)
            try:
                result = _call_with_optional_worker(self.processing_func, args, self)
            except Exception:
                tb = traceback.format_exc()
                self.error.emit(tb)
                self.finished.emit(
                    f"Task {idx}/{total_tasks} failed (see log for details).", False
                )
                return

            if not self._result_is_success(result):
                all_success = False

            if self.dynamic_total is None:
                self.progress_updated.emit(idx, f"Completed task {idx} of {total_tasks}")

        if all_success:
            self.finished.emit("All tasks have been successfully processed.", True)
        else:
            self.finished.emit(
                "All tasks finished, but some reported failure (see log).", False
            )

    def stop(self):
        self.stopped = True


# ---------------------------------------------------------------------------
# Log formatting model
# ---------------------------------------------------------------------------

# Severity tier used in the toolbar. The keys here drive the filter
# checkboxes; the values list the underlying ``log_type`` strings each
# checkbox covers.
LEVEL_GROUPS = {
    "error":   {"error"},
    "warning": {"warning"},
    "success": {"success"},
    "process": {"process"},
    "info":    {"info", "file", "reference"},
}

LEVEL_ICON = {
    "error":     "\u274c",   # red cross
    "warning":   "\u26a0\ufe0f",
    "success":   "\u2705",
    "process":   "\u231b",
    "info":      "\u2139\ufe0f",
    "file":      "\U0001f4c1",
    "reference": "\U0001f4dd",
    "section":   "\u2756",
}

# Foregrounds chosen for legibility on both light and dark backgrounds.
LEVEL_FG = {
    "error":   "#d9534f",
    "warning": "#e0a800",
    "success": "#28a745",
}

SECTION_FG = "#34495e"
SETTINGS_FG = "#7f8c8d"

# Subtle background tint applied behind section header blocks.
SECTION_BG = "#eef2f6"

# Pixel column where entry message text starts (the emoji icon sits to the
# left of this tab stop, so messages stay aligned regardless of glyph width).
ENTRY_TAB_PX = 30.0

# Step inference: when a call site doesn't pass an explicit ``step``, the
# message body is scanned for these tokens (case-insensitive) so the step
# filter still works for legacy call sites.
STEP_KEYWORDS = (
    ("PREPROCESSING",       ("preprocess",)),
    ("CLUSTERING",          ("clustering", "microstate", "gev", "k-means", "kmeans", "taahc")),
    ("BACKFITTING",         ("backfit",)),
    ("FEATURE_EXTRACTION",  ("feature",)),
    ("SOURCE_LOCALIZATION", ("source",)),
    ("LABELING",            ("label",)),
    ("COREGISTRATION",      ("coregistr",)),
    ("OPTIMIZATION",        ("optim", "threshold")),
    ("STUDY_LOADING",       ("study", "load",)),
    ("CONFIGURATION",       ("setting", "config",)),
)

STEP_LABELS = {
    "PREPROCESSING":       "Preprocessing",
    "CLUSTERING":          "Clustering",
    "LABELING":            "Microstate Labeling",
    "BACKFITTING":         "Backfitting",
    "FEATURE_EXTRACTION":  "Feature Extraction",
    "SOURCE_LOCALIZATION": "Source Localization",
    "COREGISTRATION":      "Coregistration",
    "OPTIMIZATION":        "Optimization",
    "STUDY_LOADING":       "Study Loading",
    "CONFIGURATION":       "Configuration",
    "DATA_IO":             "Data I/O",
    "VISUALIZATION":       "Visualization",
    "VALIDATION":          "Validation",
    "MAINTENANCE":         "Maintenance",
    "REFERENCE":           "Reference",
    "REVIEW":              "Review",
    "SHUTDOWN":            "Shutdown",
}

# Strips at most one leading emoji + optional separator from the message,
# so call sites that pre-pend an icon don't end up with a doubled icon
# after the formatter adds its own per-level glyph.
_LEADING_EMOJI_RE = re.compile(
    r"^\s*"
    r"["
    r"\U0001F300-\U0001FAFF"  # Misc Symbols & Pictographs, Emoticons, etc.
    r"\U0001F000-\U0001F2FF"  # Mahjong / Domino / Enclosed Alphanumeric Suppl.
    r"\u2600-\u27BF"           # Misc Symbols, Dingbats, Misc Technical
    r"\u2300-\u23FF"           # Misc Technical (incl. \u231b hourglass)
    r"\u2100-\u214F"           # Letterlike Symbols (incl. \u2139 info)
    r"\u2700-\u27BF"           # Dingbats
    r"]"
    r"(?:\ufe0f)?"
    r"\s*[:\-\u2013\u2014]?\s*",
    re.UNICODE,
)

def _infer_step_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    low = text.lower()
    for step, keys in STEP_KEYWORDS:
        if any(k in low for k in keys):
            return step
    return None


class LogRecord:
    """A structured log entry: the source of truth for the window.

    The QTextEdit body and the sidebar tree are both rendered from a list
    of these records, and the list is what gets persisted (as JSON) so a
    reloaded session rebuilds identically instead of being re-parsed from
    formatted text.

    Attributes:
      kind: ``"entry"``, ``"section"``, ``"settings"`` or ``"raw"``
        (a verbatim line imported from a legacy plain-text log).
      level: Severity/``log_type`` for entries (``"info"``, ``"error"``,
        \u2026); ``"section"`` / ``"settings"`` for the other kinds.
      step: Processing step key (e.g. ``"CLUSTERING"``) or ``None``.
      message: Entry text, section title, or settings body.
      section_id: Id of the owning section (``0`` before any section).
      ts: Timestamp string. Only populated (and only shown) for sections.
    """

    __slots__ = ("kind", "level", "step", "message", "section_id", "ts")

    def __init__(self, kind, level, step, message, section_id, ts=""):
        self.kind = kind
        self.level = level or "info"
        self.step = step or None
        self.message = message or ""
        self.section_id = int(section_id or 0)
        self.ts = ts or ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "level": self.level,
            "step": self.step,
            "message": self.message,
            "section_id": self.section_id,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogRecord":
        return cls(
            kind=d.get("kind", "entry"),
            level=d.get("level", "info"),
            step=d.get("step"),
            message=d.get("message", ""),
            section_id=d.get("section_id", 0),
            ts=d.get("ts", ""),
        )


class LogBlockData(QTextBlockUserData):
    """Per-line metadata used by filters and the section sidebar.

    Attributes:
      level: One of the ``log_type`` values understood by
        :meth:`LogWindow.append_log` (``"error"``, ``"warning"``,
        ``"success"``, ``"process"``, ``"info"``, ``"file"``,
        ``"reference"``, ``"section"``).
      step: Processing step the entry belongs to (e.g. ``"CLUSTERING"``)
        or ``None`` when not associated with a specific step.
      section_id: Auto-incrementing id of the section the entry belongs
        to. ``0`` means "no section yet".
      role: One of ``"header"``, ``"subheader"``, ``"settings"``,
        ``"entry"`` or ``"blank"``. Determines whether filters can hide
        the line (headers are always shown).
    """

    __slots__ = ("level", "step", "section_id", "role")

    def __init__(self, level: str, step: Optional[str], section_id: int, role: str):
        super().__init__()
        self.level = level or "info"
        self.step = step or None
        self.section_id = int(section_id or 0)
        self.role = role or "entry"


class LogWindow(QWidget):
    """Log window with hierarchical formatting, filters, and section nav.

    The window starts from ``ui/LogWindow.ui`` and then injects a toolbar
    above the existing :class:`QTextEdit` plus a left :class:`QTreeWidget`
    sidebar (wrapped in a :class:`QSplitter`) that groups every section by
    its processing step, with the section time attached. Click a step or
    section to jump to it; use the toolbar to filter by severity / step or
    to incrementally search the document. Body text and the sidebar are
    both rendered from an ordered list of :class:`LogRecord` objects, which
    is also what gets persisted (as a JSON sidecar) for faithful reloads.

    Args:
      comet_instance: Optional COMET instance used to persist the log
        text to disk after every change.
    """

    def __init__(self, comet_instance=None):
        super().__init__()
        script_path = os.path.abspath(__file__)
        ui_path = os.path.join(os.path.dirname(script_path), "..", "ui", "LogWindow.ui")
        self.ui = uic.loadUi(ui_path, self)
        self.ui.setWindowTitle("EEG-COMET Log")
        apply_window_minimum(self, "tiny_log")
        self.ui.log_label.setProperty("role", "banner")
        self.ui.progress_stop_button.clicked.connect(self.stop_process)
        self.ui.progress_status_button.clicked.connect(self.show_study_status)
        self.ui.progress_stop_button.setEnabled(False)

        self.comet_instance = comet_instance
        self.worker_thread = None
        self.process_finished_callback = None
        self.current_optimizer = None
        self.current_step: Optional[str] = None

        self._is_saving_logs = False

        # Persisting reads the whole document and rewrites the log file, so a
        # burst of appends is coalesced into one debounced save to keep the UI
        # thread responsive.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._flush_log_to_comet)

        # Structured model: the ordered records that drive both the body
        # rendering and the sidebar tree, and that get persisted as JSON.
        self._records: List[LogRecord] = []
        # step key -> top-level QTreeWidgetItem for that step group.
        self._step_nodes = {}

        # Navigation / filter state.
        self._section_counter = 0
        self._current_section_id = 0
        self._active_levels: Set[str] = set(LEVEL_GROUPS.keys())
        self._active_step: Optional[str] = None
        self._search_text = ""

        self._build_navigation_ui()
        self._apply_log_font()
        self._restore_settings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _apply_log_font(self) -> None:
        """Monospaced body font + a fixed tab stop for the icon/message column.

        Emoji glyphs are not fixed-width, so alignment can't rely on space
        padding. Each entry is ``icon\\tmessage`` and the tab stop pins the
        message column at :data:`ENTRY_TAB_PX` regardless of glyph width.
        Wrapping is disabled so styled headers and long lines never fold;
        a horizontal scrollbar appears instead.
        """
        editor = self.ui.log_text_area
        families = QFontDatabase().families()
        preferred = ("Consolas", "Cascadia Mono", "Menlo", "DejaVu Sans Mono",
                     "Liberation Mono", "Courier New", "Monospace")
        family = next((f for f in preferred if f in families), preferred[-1])
        font = QFont(family)
        font.setStyleHint(QFont.Monospace)
        font.setPointSizeF(editor.font().pointSizeF() or 11.0)
        editor.setFont(font)

        editor.setLineWrapMode(QTextEdit.NoWrap)
        editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        try:
            editor.setTabStopDistance(ENTRY_TAB_PX)
        except AttributeError:
            editor.setTabStopWidth(int(ENTRY_TAB_PX))

    def _build_navigation_ui(self) -> None:
        """Inject the toolbar + sections sidebar around the existing QTextEdit."""
        vlayout = self.ui.verticalLayout
        log_text = self.ui.log_text_area

        idx = vlayout.indexOf(log_text)
        if idx < 0:
            return
        vlayout.removeWidget(log_text)

        toolbar = self._build_toolbar()
        vlayout.insertLayout(idx, toolbar)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setObjectName("log_splitter")
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(6)

        self._sections_tree = QTreeWidget(splitter)
        self._sections_tree.setObjectName("sections_tree")
        self._sections_tree.setHeaderHidden(True)
        self._sections_tree.setMinimumWidth(180)
        self._sections_tree.setMaximumWidth(460)
        self._sections_tree.setIndentation(12)
        self._sections_tree.itemActivated.connect(self._on_section_clicked)
        self._sections_tree.itemClicked.connect(self._on_section_clicked)

        splitter.addWidget(self._sections_tree)
        splitter.addWidget(log_text)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([200, 800])

        vlayout.insertWidget(idx + 1, splitter)
        self._splitter = splitter

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(6)

        self._search_box = QLineEdit(self)
        self._search_box.setObjectName("log_search_box")
        self._search_box.setPlaceholderText(
            "Search log\u2026  (Enter \u2192 next match)"
        )
        self._search_box.textChanged.connect(self._on_search_changed)
        self._search_box.returnPressed.connect(self._search_next)
        bar.addWidget(self._search_box, 1)

        self._level_buttons = {}
        order = ("error", "warning", "success", "process", "info")
        labels = {
            "error":   "Errors",
            "warning": "Warnings",
            "success": "Success",
            "process": "Processing",
            "info":    "Info",
        }
        for level in order:
            btn = QToolButton(self)
            btn.setObjectName(f"log_filter_{level}")
            btn.setText(LEVEL_ICON[level])
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setToolTip(f"Show / hide {labels[level]}")
            btn.toggled.connect(
                lambda checked, lvl=level: self._on_level_toggled(lvl, checked)
            )
            bar.addWidget(btn)
            self._level_buttons[level] = btn

        self._step_combo = QComboBox(self)
        self._step_combo.setObjectName("log_step_combo")
        self._step_combo.addItem("All steps", None)
        for step, label in STEP_LABELS.items():
            self._step_combo.addItem(label, step)
        self._step_combo.setToolTip("Filter to a single processing step")
        self._step_combo.currentIndexChanged.connect(self._on_step_changed)
        bar.addWidget(self._step_combo)

        def _btn(text, slot, tooltip=""):
            b = QToolButton(self)
            b.setText(text)
            if tooltip:
                b.setToolTip(tooltip)
            b.clicked.connect(slot)
            bar.addWidget(b)
            return b

        _btn("Collapse", self._collapse_all_sections, "Hide section bodies")
        _btn("Expand",   self._expand_all_sections,   "Show section bodies")
        _btn("End",      self._scroll_to_end,         "Jump to the end of the log")
        _btn("Copy",     self._copy_visible,          "Copy visible log to clipboard")
        _btn("Export\u2026", self._export_dialog,     "Export the log to a file")
        _btn("Clear",    self.clear_logs,             "Clear the log")

        return bar

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_main_thread() -> bool:
        return QCoreApplication.instance().thread() == QThread.currentThread()

    @staticmethod
    def _strip_leading_emoji(text: str) -> str:
        if not text:
            return ""
        return _LEADING_EMOJI_RE.sub("", text).strip()

    def _scroll_if_at_bottom(self) -> None:
        bar = self.ui.log_text_area.verticalScrollBar()
        if bar.value() >= bar.maximum() - 4:
            bar.setValue(bar.maximum())

    def _scroll_to_end(self) -> None:
        bar = self.ui.log_text_area.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ------------------------------------------------------------------
    # Public API: append / replace / read log content
    # ------------------------------------------------------------------

    def append_log(self, log, log_type: str = "info", step: Optional[str] = None) -> None:
        """Append a log entry.

        Args:
          log: The message body. A leading emoji is stripped to avoid
            doubling up with the per-level icon the formatter adds.
          log_type: One of ``"info"``, ``"success"``, ``"error"``,
            ``"warning"``, ``"process"``, ``"file"``, ``"reference"``,
            ``"section"`` or ``"settings"``.
          step: Optional processing step the entry belongs to (used by
            the step filter). When omitted, the formatter falls back to
            :attr:`current_step` and finally to keyword inference.
        """
        if log is None:
            return

        if log_type == "section":
            self._append_section_header(str(log), step=step)
            return
        if log_type == "settings":
            self._append_settings_block(str(log), step=step)
            return

        body = self._strip_leading_emoji(str(log))
        inferred_step = step or self.current_step or _infer_step_from_text(body)

        self._add_record("entry", log_type, inferred_step, body,
                         self._current_section_id)
        self._render_entry(body, log_type, inferred_step, self._current_section_id)
        self._save_log_to_comet()

    def _render_entry(self, body, log_type, step, section_id) -> None:
        # ``icon\tmessage``: the tab stop pins the message column so variable
        # width emoji don't break alignment.
        icon = LEVEL_ICON.get(log_type, LEVEL_ICON["info"])
        self._append_block(
            f"{icon}\t{body}",
            level=log_type,
            step=step,
            section_id=section_id,
            role="entry",
            color=LEVEL_FG.get(log_type),
            bold=log_type == "error",
        )

    def _append_section_header(self, title: str, step: Optional[str] = None,
                               ts: Optional[str] = None) -> None:
        title = (title or "Section").strip()
        self._section_counter += 1
        self._current_section_id = self._section_counter
        if not step:
            step = self.current_step or _infer_step_from_text(title)
        timestamp = ts or datetime.now().strftime("%H:%M:%S")

        self._add_record("section", "section", step, title,
                         self._current_section_id, ts=timestamp)
        self._render_section(title, step, timestamp, self._current_section_id)
        self._save_log_to_comet()

    def _render_section(self, title, step, timestamp, section_id) -> None:
        icon = LEVEL_ICON["section"]
        self._append_block(
            f"{icon}  {title}    \u00b7  {timestamp}",
            level="section", step=step, section_id=section_id,
            role="header", color=SECTION_FG, bold=True,
        )
        self._add_section_to_tree(step, title, timestamp, section_id)

    def _add_section_to_tree(self, step, title, timestamp, section_id) -> None:
        step_key = step or "GENERAL"
        parent = self._step_nodes.get(step_key)
        if parent is None:
            label = STEP_LABELS.get(
                step_key, (step_key or "General").replace("_", " ").title()
            )
            parent = QTreeWidgetItem([label])
            parent.setData(0, Qt.UserRole, section_id)
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            self._sections_tree.addTopLevelItem(parent)
            self._step_nodes[step_key] = parent

        child = QTreeWidgetItem([f"{title}    \u00b7  {timestamp}"])
        child.setData(0, Qt.UserRole, section_id)
        parent.addChild(child)
        parent.setExpanded(True)
        self._sections_tree.scrollToItem(child)

    def _append_settings_block(self, body: str, step: Optional[str] = None) -> None:
        if not step:
            step = self.current_step or _infer_step_from_text(body) or "CONFIGURATION"
        self._add_record("settings", "settings", step, body,
                         self._current_section_id)
        self._render_settings(body, step, self._current_section_id)
        self._save_log_to_comet()

    def _render_settings(self, body, step, section_id) -> None:
        self._append_block(
            "\U0001f4cb\tCONFIGURATION", level="info", step=step,
            section_id=section_id, role="settings", bold=True, color=SETTINGS_FG,
        )
        for raw in (body or "").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith(("=", "-", "\u2550", "\u2014")):
                continue
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                line = f"\t{key.strip()}: {value.strip()}"
            else:
                line = f"\t{stripped}"
            self._append_block(
                line, level="info", step=step,
                section_id=section_id, role="settings", color=SETTINGS_FG,
            )

    def _add_record(self, kind, level, step, message, section_id, ts="") -> None:
        self._records.append(LogRecord(kind, level, step, message, section_id, ts))

    # ------------------------------------------------------------------
    # Block insertion (thread-safe)
    # ------------------------------------------------------------------

    def _append_block(
        self,
        text: str,
        *,
        level: str,
        step: Optional[str],
        section_id: int,
        role: str,
        color: Optional[str] = None,
        bold: bool = False,
    ) -> None:
        if not self._is_main_thread():
            QMetaObject.invokeMethod(
                self,
                "_append_block_main",
                Qt.QueuedConnection,
                Q_ARG(str, text),
                Q_ARG(str, level or "info"),
                Q_ARG(str, step or ""),
                Q_ARG(int, int(section_id or 0)),
                Q_ARG(str, role or "entry"),
                Q_ARG(str, color or ""),
                Q_ARG(bool, bool(bold)),
            )
            return
        self._append_block_main(
            text, level or "info", step or "", int(section_id or 0),
            role or "entry", color or "", bool(bold),
        )

    @pyqtSlot(str, str, str, int, str, str, bool)
    def _append_block_main(
        self,
        text: str,
        level: str,
        step: str,
        section_id: int,
        role: str,
        color: str,
        bold: bool,
    ) -> None:
        doc = self.ui.log_text_area.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.End)

        non_empty_doc = doc.blockCount() > 1 or doc.firstBlock().length() > 1
        if non_empty_doc:
            cursor.insertBlock()

        cursor.setBlockFormat(self._block_format_for_role(role))

        char_fmt = QTextCharFormat()
        if color:
            char_fmt.setForeground(QColor(color))
        if bold:
            char_fmt.setFontWeight(QFont.Bold)
        if role == "header":
            # Render section headers as styled blocks instead of ASCII banners.
            base = self.ui.log_text_area.font().pointSizeF() or 11.0
            char_fmt.setFontPointSize(base + 1.0)
            char_fmt.setFontWeight(QFont.Bold)
        cursor.setCharFormat(char_fmt)

        cursor.insertText(text or "")

        block = cursor.block()
        block.setUserData(LogBlockData(
            level=level or "info",
            step=step or None,
            section_id=section_id,
            role=role or "entry",
        ))
        self._apply_filters_to_block(block)
        self._scroll_if_at_bottom()

    @staticmethod
    def _block_format_for_role(role: str) -> QTextBlockFormat:
        fmt = QTextBlockFormat()
        if role == "header":
            fmt.setBackground(QColor(SECTION_BG))
            fmt.setTopMargin(10.0)
            fmt.setBottomMargin(4.0)
            fmt.setLeftMargin(2.0)
        return fmt

    # ------------------------------------------------------------------
    # Replace / read content
    # ------------------------------------------------------------------

    def replace_log(self, log_text: str) -> None:
        self._load_from_plaintext(log_text or "")
        self._save_log_to_comet()

    def get_log_content(self) -> str:
        return self.ui.log_text_area.toPlainText()

    def set_log_content(self, log_content) -> None:
        # Prefer the structured JSON sidecar so a reloaded session rebuilds
        # exactly; fall back to a best-effort plain-text import otherwise.
        records = self._load_records_json()
        if records is not None:
            self._rebuild_from_records(records)
            return
        if not log_content:
            return
        if not isinstance(log_content, str):
            log_content = str(log_content)
        try:
            clean = log_content.encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            clean = log_content
        self._load_from_plaintext(clean)
        self._save_log_to_comet()

    @pyqtSlot()
    def _save_log_to_comet(self) -> None:
        """Schedule a debounced persist of the log to disk.

        The expensive read+write happens in :meth:`_flush_log_to_comet` when
        the timer fires.
        """
        if self.comet_instance is None:
            return
        if not self._is_main_thread():
            QMetaObject.invokeMethod(self, "_save_log_to_comet", Qt.QueuedConnection)
            return
        # Leaving a pending timer running bounds staleness during a continuous
        # stream instead of deferring the flush until it pauses.
        if not self._save_timer.isActive():
            self._save_timer.start()

    @pyqtSlot()
    def _flush_log_to_comet(self) -> None:
        """Persist the current log content to disk immediately (UI thread)."""
        if self.comet_instance is None or self._is_saving_logs:
            return
        if not self._is_main_thread():
            QMetaObject.invokeMethod(self, "_flush_log_to_comet", Qt.QueuedConnection)
            return
        self._is_saving_logs = True
        try:
            log_content = self.get_log_content()
            self.comet_instance.log_text = log_content
            self.comet_instance.save_logs_to_file()
            self._write_records_json()
        finally:
            self._is_saving_logs = False

    # ------------------------------------------------------------------
    # Structured model persistence / rebuild
    # ------------------------------------------------------------------

    def _records_json_path(self) -> Optional[str]:
        path = getattr(self.comet_instance, "log_file_path", None)
        if not path:
            return None
        root, _ = os.path.splitext(path)
        return root + ".records.json"

    def _write_records_json(self) -> None:
        path = self._records_json_path()
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._records], f, ensure_ascii=False)
        except Exception:
            pass

    def _load_records_json(self) -> Optional[List[LogRecord]]:
        path = self._records_json_path()
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        if not isinstance(data, list) or not data:
            # Empty/invalid sidecar: fall back to plain text so a good .txt
            # log isn't clobbered by an accidental empty rebuild.
            return None
        return [LogRecord.from_dict(d) for d in data if isinstance(d, dict)]

    def _reset_model(self) -> None:
        self.ui.log_text_area.clear()
        self._sections_tree.clear()
        self._step_nodes.clear()
        self._records = []
        self._section_counter = 0
        self._current_section_id = 0

    def _rebuild_from_records(self, records: List[LogRecord]) -> None:
        self._reset_model()
        editor = self.ui.log_text_area
        editor.setUpdatesEnabled(False)
        try:
            for rec in records:
                self._records.append(rec)
                if rec.kind == "section":
                    self._section_counter += 1
                    self._current_section_id = self._section_counter
                    rec.section_id = self._current_section_id
                    self._render_section(
                        rec.message, rec.step, rec.ts, self._current_section_id
                    )
                elif rec.kind == "settings":
                    self._render_settings(rec.message, rec.step, rec.section_id)
                elif rec.kind == "raw":
                    self._append_block(
                        rec.message, level=rec.level, step=rec.step,
                        section_id=rec.section_id, role="entry",
                    )
                else:
                    self._render_entry(
                        rec.message, rec.level, rec.step, rec.section_id
                    )
        finally:
            editor.setUpdatesEnabled(True)
        self._refilter_all()
        self._scroll_to_end()

    def _load_from_plaintext(self, text: str) -> None:
        """Best-effort import of a legacy plain-text log (no JSON sidecar)."""
        self._reset_model()
        editor = self.ui.log_text_area
        editor.setUpdatesEnabled(False)
        try:
            for line in (text or "").splitlines():
                if not line.strip():
                    continue
                # Legacy lines already carry their own text/glyphs; store them
                # as ``raw`` so they render verbatim now and after any reload.
                self._records.append(
                    LogRecord("raw", "info", self.current_step, line, 0)
                )
                self._append_block(
                    line, level="info", step=self.current_step,
                    section_id=0, role="entry",
                )
        finally:
            editor.setUpdatesEnabled(True)
        self._refilter_all()

    # ------------------------------------------------------------------
    # Filtering / search
    # ------------------------------------------------------------------

    def _on_level_toggled(self, level: str, checked: bool) -> None:
        if checked:
            self._active_levels.add(level)
        else:
            self._active_levels.discard(level)
        self._refilter_all()

    def _on_step_changed(self, _index: int) -> None:
        self._active_step = self._step_combo.currentData()
        self._refilter_all()

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text or ""
        self._highlight_matches()

    def _search_next(self) -> None:
        if not self._search_text:
            return
        editor = self.ui.log_text_area
        doc = editor.document()
        cursor = doc.find(self._search_text, editor.textCursor())
        if cursor.isNull():
            cursor = doc.find(self._search_text, QTextCursor(doc))
        if not cursor.isNull():
            editor.setTextCursor(cursor)
            editor.ensureCursorVisible()

    def _highlight_matches(self) -> None:
        editor = self.ui.log_text_area
        if not self._search_text:
            editor.setExtraSelections([])
            return
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#fff59d"))
        extras = []
        doc = editor.document()
        cursor = QTextCursor(doc)
        while True:
            cursor = doc.find(self._search_text, cursor)
            if cursor.isNull():
                break
            sel = QTextEdit.ExtraSelection()
            sel.cursor = cursor
            sel.format = fmt
            extras.append(sel)
        editor.setExtraSelections(extras)

    def _apply_filters_to_block(self, block) -> None:
        data = block.userData()
        if not isinstance(data, LogBlockData):
            block.setVisible(True)
            return

        # Section headers, sub-headers and blank gutter rows are always
        # visible so the document never collapses to zero rows under
        # heavy filtering.
        if data.role in ("header", "blank"):
            block.setVisible(True)
            return

        visible = True

        # Severity filter.
        if data.level != "section":
            groups = {grp for grp, levels in LEVEL_GROUPS.items()
                      if data.level in levels}
            if groups and not (groups & self._active_levels):
                visible = False

        # Step filter.
        if visible and self._active_step:
            if data.step and data.step != self._active_step:
                visible = False

        block.setVisible(visible)

    def _refilter_all(self) -> None:
        doc = self.ui.log_text_area.document()
        block = doc.firstBlock()
        while block.isValid():
            self._apply_filters_to_block(block)
            block = block.next()
        doc.markContentsDirty(0, doc.characterCount())
        self.ui.log_text_area.viewport().update()

    # ------------------------------------------------------------------
    # Section folding / sidebar
    # ------------------------------------------------------------------

    def _on_section_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        if item is None:
            return
        sid = item.data(0, Qt.UserRole)
        self._scroll_to_section(int(sid) if sid is not None else 0)

    def _scroll_to_section(self, section_id: int) -> None:
        if not section_id:
            return
        doc = self.ui.log_text_area.document()
        block = doc.firstBlock()
        while block.isValid():
            data = block.userData()
            if (
                isinstance(data, LogBlockData)
                and data.section_id == section_id
                and data.role == "header"
            ):
                cursor = QTextCursor(block)
                self.ui.log_text_area.setTextCursor(cursor)
                self.ui.log_text_area.ensureCursorVisible()
                return
            block = block.next()

    def _set_section_bodies_visible(self, visible: bool) -> None:
        doc = self.ui.log_text_area.document()
        block = doc.firstBlock()
        while block.isValid():
            data = block.userData()
            if isinstance(data, LogBlockData):
                if data.section_id and data.role not in ("header", "blank"):
                    if visible:
                        self._apply_filters_to_block(block)
                    else:
                        block.setVisible(False)
            block = block.next()
        doc.markContentsDirty(0, doc.characterCount())
        self.ui.log_text_area.viewport().update()

    def _collapse_all_sections(self) -> None:
        self._set_section_bodies_visible(False)

    def _expand_all_sections(self) -> None:
        self._set_section_bodies_visible(True)

    def _copy_visible(self) -> None:
        doc = self.ui.log_text_area.document()
        lines = []
        block = doc.firstBlock()
        while block.isValid():
            if block.isVisible():
                lines.append(block.text())
            block = block.next()
        QApplication.clipboard().setText("\n".join(lines))

    def _export_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export log", "eeg_comet_log.txt", "Text files (*.txt);;All files (*)",
        )
        if path:
            self.export_logs(path)

    # ------------------------------------------------------------------
    # Progress dialog API
    # ------------------------------------------------------------------

    def setup_progress_dialog(self, window_title, label_text, tasks, processing_func):
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.show()
        self.setWindowTitle(window_title)
        self.ui.progress_label.setText(label_text)
        self.ui.progress_bar.setValue(0)

        if "Preprocessing" in window_title:
            self.current_step = "PREPROCESSING"
        elif "Clustering" in window_title:
            self.current_step = "CLUSTERING"
            if isinstance(tasks, int):
                self.append_log(
                    f"Starting microstate clustering with {tasks} repetitions",
                    log_type="section",
                    step="CLUSTERING",
                )
        elif "Optimal Window" in window_title:
            self.current_step = "BACKFITTING"
        elif "Backfitting" in window_title:
            self.current_step = "BACKFITTING"
        elif "Feature" in window_title:
            self.current_step = "FEATURE_EXTRACTION"
        elif "Source" in window_title:
            self.current_step = "SOURCE_LOCALIZATION"
        else:
            self.current_step = "PROCESSING"

        if isinstance(tasks, int):
            total_tasks = tasks
        elif hasattr(tasks, "__len__") and not isinstance(tasks, str):
            total_tasks = len(tasks)
        else:
            total_tasks = 1

        self.ui.progress_bar.setRange(0, total_tasks)

        # Don't silently abandon a still-running worker; that would leak the
        # QThread and leave its signals connected to slots on this window.
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.append_log(
                "A background task is already running; ignoring new request",
                log_type="warning",
            )
            return

        self.worker_thread = Worker(tasks, processing_func)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.finished.connect(self.process_finished)
        self.worker_thread.error.connect(self._on_worker_error)
        self.worker_thread.start()

    def _on_worker_error(self, traceback_str):
        self.append_log(
            f"Unhandled exception in background worker:\n{traceback_str}",
            log_type="error",
        )

    def set_progress_label(self, text):
        self.ui.progress_label.setText(text)

    def set_progress_text(self, text):
        self.ui.progress_lineedit.setText(text)

    def set_progress_max(self, maximum):
        self.ui.progress_bar.setMaximum(int(maximum))

    def set_progress_value(self, value):
        self.ui.progress_bar.setValue(int(value))

    def set_progress_stop_enabled(self, enabled):
        self.ui.progress_stop_button.setEnabled(bool(enabled))

    def update_progress(self, value, text):
        self.ui.progress_stop_button.setEnabled(True)

        if value > self.ui.progress_bar.maximum():
            self.ui.progress_bar.setMaximum(value * 2)

        self.ui.progress_bar.setValue(value)
        self.ui.progress_lineedit.setText(text)

        if self.current_step == "CLUSTERING" and "completed" in text.lower():
            if "repetition" in text.lower():
                try:
                    parts = text.split()
                    rep_idx = parts.index("repetition") + 1
                    rep_num, total_reps = parts[rep_idx].split("/")
                    percentage = int((int(rep_num) / int(total_reps)) * 100)
                    self.append_log(
                        f"Repetition {rep_num}/{total_reps} completed ({percentage}%)",
                        log_type="success",
                        step="CLUSTERING",
                    )
                except (IndexError, ValueError):
                    self.append_log(text, log_type="success", step="CLUSTERING")
            elif "taahc" in text.lower():
                self.append_log(text, log_type="success", step="CLUSTERING")

        QApplication.processEvents()

    def process_finished(self, text=None, success=True):
        self.setWindowFlags(Qt.Window)
        self.show()
        if text is not None:
            self.ui.progress_lineedit.setText(text)
            if self.current_step == "CLUSTERING":
                if success:
                    self.append_log(
                        "Clustering process completed successfully.",
                        log_type="success",
                        step="CLUSTERING",
                    )
                else:
                    self.append_log(
                        f"Clustering process finished with errors: {text}",
                        log_type="warning",
                        step="CLUSTERING",
                    )
            elif not success:
                self.append_log(
                    f"{self.current_step or 'Process'} failed: {text}",
                    log_type="warning",
                    step=self.current_step,
                )

        self.ui.progress_stop_button.setEnabled(False)

        if self.process_finished_callback is not None:
            try:
                self.process_finished_callback(success=success)
            except TypeError:
                self.process_finished_callback()

    def set_window_title(self, title):
        self.setWindowTitle(title)

    def set_label_text(self, text):
        self.ui.progress_label.setText(text)

    def stop_process(self):
        if self.current_step and self.comet_instance and hasattr(self.comet_instance, "logger"):
            self.comet_instance.logger.stop_requested(self.current_step)

        if self.current_step == "CLUSTERING":
            self.append_log("Stopping clustering process\u2026",
                            log_type="warning", step="CLUSTERING")
            self.append_log("Saving partial results if available",
                            log_type="info", step="CLUSTERING")
        else:
            self.append_log(
                f"Stopping {self.current_step or 'process'}\u2026",
                log_type="warning",
                step=self.current_step,
            )

        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            if not self.worker_thread.wait(30000):
                self.append_log(
                    "Worker thread did not stop gracefully", log_type="warning"
                )

        if hasattr(self, "current_optimizer") and self.current_optimizer is not None:
            try:
                self.current_optimizer.stop()
                self.append_log("Optimization process stopped", log_type="info")
            except Exception as exc:
                self.append_log(
                    f"Error stopping optimizer: {exc}", log_type="warning"
                )
            finally:
                self.current_optimizer = None

        self.ui.progress_stop_button.setEnabled(False)
        self.ui.progress_label.setText("Process stopped by user")
        self.current_step = None
        self.append_log("Ready for new operations", log_type="success")

    def show_hide_log_window(self):
        self.setVisible(not self.isVisible())

    def clear_logs(self):
        self._reset_model()
        self._save_log_to_comet()

    def export_logs(self, file_path):
        log_content = self.get_log_content()
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(log_content)
            return True
        except Exception as exc:
            print(f"Error exporting logs: {exc}")
            return False

    # ------------------------------------------------------------------
    # Lifecycle / persistence
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        # Make sure any background work is stopped before the window goes away,
        # otherwise the QThread may keep emitting into deleted widgets.
        if self.worker_thread is not None and self.worker_thread.isRunning():
            try:
                self.worker_thread.stop()
            except Exception:
                pass
            try:
                self.worker_thread.wait(5000)
            except Exception:
                pass
        if hasattr(self, "current_optimizer") and self.current_optimizer is not None:
            try:
                self.current_optimizer.stop()
            except Exception:
                pass
            self.current_optimizer = None

        self.append_log("EEG-COMET Session Ended", log_type="section",
                        step="SHUTDOWN")
        self.append_log("Thank you for using EEG-COMET!", log_type="info",
                        step="SHUTDOWN")
        # Flush synchronously so a pending debounced save isn't lost.
        self._save_timer.stop()
        self._flush_log_to_comet()
        self._persist_settings()
        event.accept()

    def _restore_settings(self) -> None:
        settings = QSettings("EEG-COMET", "EEG-COMET")
        state = settings.value("logwindow/splitter")
        if state is not None:
            try:
                self._splitter.restoreState(state)
            except Exception:
                pass

    def _persist_settings(self) -> None:
        settings = QSettings("EEG-COMET", "EEG-COMET")
        try:
            settings.setValue("logwindow/splitter", self._splitter.saveState())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Clustering convenience helpers (used by comet.py / progress callbacks)
    # ------------------------------------------------------------------

    def log_clustering_setup_step(self, step_name, step_description=""):
        if self.current_step != "CLUSTERING":
            return
        if step_description:
            message = f"{step_name}: {step_description}"
        else:
            message = step_name
        self.append_log(message, log_type="process", step="CLUSTERING")

    def log_clustering_progress(self, repetition_num, total_repetitions, gev=None, is_best=False):
        if self.current_step != "CLUSTERING":
            return
        percentage = int((repetition_num / total_repetitions) * 100)
        base = f"Repetition {repetition_num}/{total_repetitions} ({percentage}%)"
        if gev is None:
            self.append_log(base, log_type="info", step="CLUSTERING")
            return
        gev_str = f"GEV {gev * 100:.2f}%"
        if is_best:
            self.append_log(
                f"{base} \u2014 {gev_str}  \u2605 best",
                log_type="success",
                step="CLUSTERING",
            )
        else:
            self.append_log(
                f"{base} \u2014 {gev_str}",
                log_type="info",
                step="CLUSTERING",
            )

    # ------------------------------------------------------------------
    # Study status report
    # ------------------------------------------------------------------

    def show_study_status(self):
        if self.comet_instance is None:
            self.append_log("No study loaded", log_type="warning")
            return

        lines = []
        lines.append("STUDY STATUS REPORT")
        if hasattr(self.comet_instance, "study_name") and self.comet_instance.study_name:
            lines.append(f"Study Name: {self.comet_instance.study_name}")
        if hasattr(self.comet_instance, "save_dir") and self.comet_instance.save_dir:
            lines.append(f"Save Directory: {self.comet_instance.save_dir}")
        lines.append("")
        lines.append("Processing steps:")

        steps = [
            ("Preprocessing", "done_preprocessing"),
            ("Clustering", "done_clustering"),
            ("Microstate Labeling", "done_microstate_labeling"),
            ("Backfitting", "done_backfitting"),
            ("Feature Extraction", "done_extracting_features"),
            ("Source Localization", "done_source_localization"),
            ("Source-Microstate Correlation", "done_identifying_microstate_sources"),
        ]
        for step_name, step_attr in steps:
            done = bool(getattr(self.comet_instance, step_attr, False))
            mark = "[done]" if done else "[----]"
            lines.append(f"  {mark}  {step_name}")

        if getattr(self.comet_instance, "done_clustering", False):
            lines.append("")
            lines.append("Clustering results:")
            if hasattr(self.comet_instance, "number_of_maps"):
                lines.append(
                    f"  Number of Microstates: {self.comet_instance.number_of_maps}"
                )
            best_gev = getattr(self.comet_instance, "best_gev", None)
            if best_gev:
                lines.append(
                    f"  Global Explained Variance: {best_gev * 100:.2f}%"
                )
            labels = getattr(self.comet_instance, "micro_labels", None)
            if labels:
                lines.append(f"  Microstate Labels: {', '.join(labels)}")

        if getattr(self.comet_instance, "list_eegs", None):
            lines.append("")
            lines.append("Data files:")
            lines.append(f"  Input Files: {len(self.comet_instance.list_eegs)}")

        self.append_log("Study Status", log_type="section", step="REVIEW")
        for line in lines:
            self.append_log(line, log_type="info", step="REVIEW")

        if hasattr(self.comet_instance, "logger") and self.comet_instance.logger:
            print("\n".join(lines))
