"""Tests for ``EEG_COMET/gui_utils/parse_input.py``.

Verify that an empty, missing, or non-numeric ``QLineEdit`` text falls back
to the supplied default and that values outside ``[minimum, maximum]`` are
clamped. Uses a tiny stub object exposing ``text()`` to stand in for
``QLineEdit``.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def parse_input():
    return pytest.importorskip("gui_utils.parse_input")


class FakeLineEdit:
    """Minimal ``QLineEdit`` stand-in exposing ``text()``."""

    def __init__(self, value):
        self._value = value

    def text(self):
        return self._value


@pytest.mark.parametrize(
    "raw,expected",
    [("42", 42), ("  7 ", 7), ("0", 0)],
)
def test_parse_int_happy_path(parse_input, raw, expected):
    assert parse_input.parse_int(FakeLineEdit(raw), default=99) == expected


def test_parse_int_falls_back_on_empty(parse_input):
    assert parse_input.parse_int(FakeLineEdit(""), default=123) == 123


def test_parse_int_falls_back_on_invalid(parse_input):
    assert parse_input.parse_int(FakeLineEdit("abc"), default=5) == 5


def test_parse_int_clamps_to_min_max(parse_input):
    assert parse_input.parse_int(
        FakeLineEdit("-10"), default=0, minimum=1, maximum=20
    ) == 1
    assert parse_input.parse_int(
        FakeLineEdit("999"), default=0, minimum=1, maximum=20
    ) == 20


def test_parse_float_happy_path(parse_input):
    assert parse_input.parse_float(FakeLineEdit("1e-3"), default=0.0) == pytest.approx(0.001)


def test_parse_float_falls_back_on_invalid(parse_input):
    assert parse_input.parse_float(FakeLineEdit("nope"), default=2.5) == pytest.approx(2.5)


def test_parse_float_clamps(parse_input):
    assert parse_input.parse_float(
        FakeLineEdit("-1"), default=0.0, minimum=0.0, maximum=1.0
    ) == 0.0
