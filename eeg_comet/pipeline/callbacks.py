"""Callback contracts for pipeline stages.

A :class:`PipelineCallbacks` is the only object that pipeline code uses to
communicate with the surrounding application. Concrete implementations live
in the GUI layer and the CLI; :class:`NullCallbacks` provides a silent
default.
"""

from __future__ import annotations

from typing import Optional, Protocol


class PipelineCallbacks(Protocol):
    """Contract used by every pipeline stage to report progress and check cancellation."""

    def on_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress. ``total`` may be ``0`` for indeterminate stages."""

    def on_log(self, message: str, level: str = "info", section: Optional[str] = None) -> None:
        """Append a log message at ``level`` ('info'|'warning'|'error'|'debug')."""

    def should_stop(self) -> bool:
        """Return ``True`` when the user has requested cancellation."""


class NullCallbacks:
    """No-op implementation of :class:`PipelineCallbacks`."""

    def on_progress(self, current: int, total: int, message: str = "") -> None:
        return

    def on_log(self, message: str, level: str = "info", section: Optional[str] = None) -> None:
        return

    def should_stop(self) -> bool:
        return False
