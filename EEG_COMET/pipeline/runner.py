"""Facade exposing each pipeline stage as a single method.

:class:`PipelineRunner` provides a UI-free entry point to the analysis
pipeline. Each stage delegates to the corresponding ``COMET.run_*`` method
and reports progress, logs, and failure through the
:class:`PipelineCallbacks` interface supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .callbacks import NullCallbacks, PipelineCallbacks


@dataclass
class StageResult:
    """Return value from any pipeline stage.

    ``ok``      : whether the stage completed without raising.
    ``message`` : short human-readable summary, surfaced in the UI/log.
    ``payload`` : optional stage-specific data (e.g. cluster maps array).
    """

    ok: bool
    message: str = ""
    payload: Any = None


class PipelineRunner:
    """Stage-by-stage facade over a configured ``COMET`` instance.

    Parameters
    ----------
    comet:
        A configured ``EEG_COMET.comet.COMET`` instance. The runner does not
        mutate its config; callers configure the instance before invoking
        any stage.
    callbacks:
        Optional :class:`PipelineCallbacks`. When ``None``,
        :class:`NullCallbacks` is used so every stage runs silently.
    """

    def __init__(self, comet: Any, callbacks: Optional[PipelineCallbacks] = None) -> None:
        self.comet = comet
        self.callbacks: PipelineCallbacks = callbacks or NullCallbacks()

    def run_preprocessing(self) -> StageResult:
        return self._invoke("run_preprocessing", "preprocessing")

    def run_clustering(self) -> StageResult:
        return self._invoke("run_clustering", "clustering")

    def run_microstate_labeling(self) -> StageResult:
        return self._invoke("run_microstate_labeling", "labeling")

    def run_backfitting(self) -> StageResult:
        return self._invoke("run_backfitting", "backfitting")

    def run_feature_extraction(self) -> StageResult:
        return self._invoke("run_feature_extraction", "features")

    def run_source_localization(self) -> StageResult:
        return self._invoke("run_source_localization", "source")

    def run_identifying_microstate_sources(self) -> StageResult:
        return self._invoke("run_identifying_microstate_sources", "microstate_sources")

    def _invoke(self, method_name: str, stage: str) -> StageResult:
        """Run ``self.comet.<method_name>()`` with cancellation and log surface."""
        if self.callbacks.should_stop():
            return StageResult(ok=False, message=f"{stage} cancelled before start")

        method = getattr(self.comet, method_name, None)
        if method is None or not callable(method):
            return StageResult(
                ok=False,
                message=f"{stage}: COMET has no method {method_name!r}",
            )

        self.callbacks.on_log(f"Starting {stage}...", level="info", section=stage.upper())
        try:
            payload = method()
        except Exception as exc:  # noqa: BLE001
            self.callbacks.on_log(
                f"{stage} failed: {exc}", level="error", section=stage.upper()
            )
            return StageResult(ok=False, message=str(exc))

        if payload is False:
            return StageResult(ok=False, message=f"{stage} reported failure")
        if (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[0], bool)
            and not payload[0]
        ):
            return StageResult(ok=False, message=str(payload[1]))

        self.callbacks.on_log(
            f"Finished {stage}.", level="info", section=stage.upper()
        )
        return StageResult(ok=True, message=f"{stage} completed", payload=payload)
