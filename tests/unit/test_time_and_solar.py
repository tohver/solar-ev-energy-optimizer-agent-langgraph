"""Time-zone handling, daylight-saving days, sun position and synthetic weather."""

import random
from datetime import UTC, date, datetime, timedelta

import pytest

from energy_advisor.data.models import EnergyUsage, UTCDateTime
from energy_advisor.domain.climate import synthetic_weather
from energy_advisor.domain.solar import clear_sky_ghi, pv_energy_kwh, solar_elevation_deg
from energy_advisor.timeutils import (
    InvalidDateError,
    calendar_lines,
    day_bounds_utc,
    hours_between,
    hours_in_local_day,
    parse_day,
    parse_local_datetime,
    to_utc,
)
from tests.helpers import BERLIN

MUNICH = (48.1374, 11.5755)


# ----------------------------------------------------------------------------- days and DST
@pytest.mark.parametrize(
    ("day", "hours"),
    [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25), (date(2026, 10, 12), 24)],
)
def test_local_days_have_23_24_or_25_hours(day, hours):
    assert hours_in_local_day(day, BERLIN) == hours
    start, end = day_bounds_utc(day, day, BERLIN)
    assert len(list(hours_between(start, end))) == hours


def test_day_bounds_are_local_midnights_in_utc():
    start, end = day_bounds_utc(date(2026, 10, 12), date(2026, 10, 13), BERLIN)
    assert start == datetime(2026, 10, 11, 22, tzinfo=UTC)  # 00:00 CEST
    assert end == datetime(2026, 10, 13, 22, tzinfo=UTC)


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValueError, match="Naive"):
        to_utc(datetime(2026, 1, 1))  # noqa: DTZ001 - deliberately naive


@pytest.mark.parametrize("bad", ["14/10/2026", "2026-1-4", "2026-13-01", "", "tomorrow"])
def test_parse_day_is_strict(bad):
    with pytest.raises(InvalidDateError):
        parse_day(bad)


def test_parse_local_datetime():
    assert parse_local_datetime("2026-10-13T07:00", BERLIN) == datetime(2026, 10, 13, 7, tzinfo=BERLIN)
    assert parse_local_datetime("2026-10-13 7:30", BERLIN).minute == 30
    with pytest.raises(InvalidDateError):
        parse_local_datetime("13.10.2026 07:00", BERLIN)


def test_calendar_marks_today_and_tomorrow():
    text = calendar_lines(datetime(2026, 10, 12, 15, tzinfo=BERLIN), days=8)
    assert text.startswith("Monday = 2026-10-12 (today), Tuesday = 2026-10-13 (tomorrow)")
    assert "Monday = 2026-10-19" in text


def test_database_column_stores_utc_and_rejects_naive():
    column = UTCDateTime()
    stored = column.process_bind_param(datetime(2026, 10, 12, 9, tzinfo=BERLIN), None)
    assert stored == datetime(2026, 10, 12, 7)  # noqa: DTZ001 - SQLite stores naive UTC
    assert column.process_result_value(stored, None) == datetime(2026, 10, 12, 7, tzinfo=UTC)
    with pytest.raises(ValueError, match="Naive"):
        column.process_bind_param(datetime(2026, 10, 12, 9), None)  # noqa: DTZ001
    assert EnergyUsage.__table__.c.timestamp.type.__class__ is UTCDateTime


# ----------------------------------------------------------------------------- sun
def test_sun_is_high_in_june_and_low_in_december_at_noon():
    june = solar_elevation_deg(datetime(2026, 6, 21, 13, 15, tzinfo=BERLIN), *MUNICH)
    december = solar_elevation_deg(datetime(2026, 12, 21, 12, 15, tzinfo=BERLIN), *MUNICH)
    assert 63 < june < 67  # 90 - 48.1 + 23.4
    assert 16 < december < 20  # 90 - 48.1 - 23.4
    assert solar_elevation_deg(datetime(2026, 6, 21, 1, tzinfo=BERLIN), *MUNICH) < 0


def test_clear_sky_irradiance_and_pv_yield():
    assert clear_sky_ghi(-5) == 0
    assert 850 < clear_sky_ghi(65) < 1000
    assert pv_energy_kwh(500, peak_kw=8, performance_ratio=0.8) == pytest.approx(3.2)
    assert pv_energy_kwh(0, 8, 0.8) == 0


@pytest.mark.parametrize(
    ("day", "min_hours", "max_hours"), [(date(2026, 12, 21), 7, 10), (date(2026, 6, 21), 15, 18)]
)
def test_synthetic_weather_has_realistic_day_length(day, min_hours, max_hours):
    hours = synthetic_weather(day, day, *MUNICH, BERLIN, random.Random(1))
    sunny_hours = sum(1 for h in hours if h.irradiance_w_m2 > 5)
    assert len(hours) == 24
    assert min_hours <= sunny_hours <= max_hours


def test_synthetic_weather_is_deterministic_and_seasonal():
    winter = synthetic_weather(date(2026, 1, 10), date(2026, 1, 16), *MUNICH, BERLIN, random.Random(3))
    summer = synthetic_weather(date(2026, 7, 10), date(2026, 7, 16), *MUNICH, BERLIN, random.Random(3))
    assert winter == synthetic_weather(
        date(2026, 1, 10), date(2026, 1, 16), *MUNICH, BERLIN, random.Random(3)
    )
    mean = lambda hs: sum(h.temperature_c for h in hs) / len(hs)  # noqa: E731
    assert mean(summer) - mean(winter) > 12
    assert all(h.time.tzinfo is UTC for h in winter)
    assert winter[1].time - winter[0].time == timedelta(hours=1)
