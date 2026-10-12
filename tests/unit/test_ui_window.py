"""Scheduling window of the app: defaults (now, now + 24 h), overrides and the text for the agent."""

from datetime import date, datetime, time

import pytest

from energy_advisor.ui.window import InvalidWindowError, default_window, window_context, window_from_inputs
from tests.helpers import BERLIN


def test_default_window_is_now_plus_24_hours():
    window = default_window(datetime(2026, 10, 12, 18, 37, 12, tzinfo=BERLIN))
    assert window.start == datetime(2026, 10, 12, 18, 37, tzinfo=BERLIN)
    assert window.deadline == datetime(2026, 10, 13, 18, 37, tzinfo=BERLIN)
    assert window.hours == 24


def test_default_window_is_24_real_hours_across_the_clock_change():
    window = default_window(datetime(2026, 10, 24, 12, 0, tzinfo=BERLIN))  # clocks go back that night
    assert window.deadline == datetime(2026, 10, 25, 11, 0, tzinfo=BERLIN)
    assert window.hours == 24


def test_deadline_can_be_overridden():
    window = window_from_inputs(date(2026, 10, 12), time(18, 0), date(2026, 10, 13), time(7, 0), BERLIN)
    assert window.hours == 13


def test_deadline_before_start_is_rejected():
    with pytest.raises(InvalidWindowError):
        window_from_inputs(date(2026, 10, 12), time(18, 0), date(2026, 10, 12), time(7, 0), BERLIN)


def test_context_text_names_both_times():
    text = window_context(default_window(datetime(2026, 10, 12, 18, 0, tzinfo=BERLIN)))
    assert "start 2026-10-12T18:00" in text
    assert "deadline 2026-10-13T18:00" in text
