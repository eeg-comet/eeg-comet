"""Tests for the :mod:`pipeline.runner` facade.

Verify the runner's behaviour without booting Qt or any of the heavy
MNE/numpy code by exercising it against a minimal ``COMET`` stand-in.
"""

from __future__ import annotations

import pytest

from pipeline import NullCallbacks, PipelineRunner
from pipeline.runner import StageResult


class FakeComet:
    """Minimal stand-in mimicking ``COMET``'s ``run_*`` surface."""

    def __init__(self, behaviours):
        self._behaviours = behaviours
        self.calls = []

    def __getattr__(self, item):
        if item in self._behaviours:
            def call():
                self.calls.append(item)
                return self._behaviours[item]()
            return call
        raise AttributeError(item)


class RecordingCallbacks:
    def __init__(self, stop_after=None):
        self.logs = []
        self.progress = []
        self.stop_after = stop_after
        self._calls = 0

    def on_progress(self, current, total, message=""):
        self.progress.append((current, total, message))

    def on_log(self, message, level="info", section=None):
        self.logs.append((level, section, message))

    def should_stop(self):
        self._calls += 1
        return self.stop_after is not None and self._calls > self.stop_after


def test_successful_stage_returns_ok():
    comet = FakeComet({"run_preprocessing": lambda: None})
    cb = RecordingCallbacks()
    result = PipelineRunner(comet, cb).run_preprocessing()
    assert result.ok is True
    assert "preprocessing completed" in result.message
    assert any("Starting preprocessing" in msg for _, _, msg in cb.logs)
    assert any("Finished preprocessing" in msg for _, _, msg in cb.logs)


def test_stage_returning_false_marks_failure():
    comet = FakeComet({"run_clustering": lambda: False})
    result = PipelineRunner(comet, NullCallbacks()).run_clustering()
    assert result.ok is False
    assert "reported failure" in result.message


def test_stage_returning_tuple_failure_propagates_message():
    comet = FakeComet({"run_backfitting": lambda: (False, "no maps loaded")})
    result = PipelineRunner(comet, NullCallbacks()).run_backfitting()
    assert result.ok is False
    assert result.message == "no maps loaded"


def test_exception_inside_stage_is_captured():
    def boom():
        raise ValueError("bad montage")
    comet = FakeComet({"run_microstate_labeling": boom})
    cb = RecordingCallbacks()
    result = PipelineRunner(comet, cb).run_microstate_labeling()
    assert result.ok is False
    assert "bad montage" in result.message
    error_logs = [msg for level, _, msg in cb.logs if level == "error"]
    assert any("labeling failed" in msg for msg in error_logs)


def test_should_stop_short_circuits_before_call():
    comet = FakeComet({"run_features": lambda: None})
    cb = RecordingCallbacks(stop_after=0)
    result = PipelineRunner(comet, cb).run_feature_extraction()
    assert result.ok is False
    assert comet.calls == []


def test_missing_method_returns_friendly_failure():
    class Empty:
        pass
    result = PipelineRunner(Empty(), NullCallbacks()).run_source_localization()
    assert result.ok is False
    assert "no method" in result.message
    assert isinstance(result, StageResult)
