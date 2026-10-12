"""Seeding, repository queries and the database tools (regressions F-04, F-07, F-08, F-10)."""

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select

from energy_advisor.agent.tools import build_tools
from energy_advisor.data.models import EnergyUsage, SolarGeneration
from energy_advisor.data.seed import seed_database
from tests.helpers import LAST_SEEDED_DAY

DAY = (LAST_SEEDED_DAY - timedelta(days=10)).isoformat()


@pytest.fixture
def tools(services):
    return {t.name: t for t in build_tools(services)}


def _count(repo, model):
    with repo.session() as s:
        return s.scalar(select(func.count()).select_from(model))


def test_seed_creates_expected_amount_of_data(repo):
    # 30 days x 24 hours x 3 device types; solar rows for the 13 daylight hours (06-18) of each day.
    assert _count(repo, EnergyUsage) == 30 * 24 * 3
    assert _count(repo, SolarGeneration) == 30 * 13


def test_f10_reseeding_replaces_data_instead_of_duplicating(repo):
    # F-10: re-running the setup used to append a second copy of the data.
    result = seed_database(repo, days=30, seed=42, end_date=LAST_SEEDED_DAY)
    assert result.deleted == {"energy_usage": 2160, "solar_generation": 390}
    assert _count(repo, EnergyUsage) == 2160


def test_seed_is_reproducible(repo):
    first = sum(r.consumption_kwh for r in repo.usage_between(*_range(repo)))
    seed_database(repo, days=30, seed=42, end_date=LAST_SEEDED_DAY)
    assert sum(r.consumption_kwh for r in repo.usage_between(*_range(repo))) == pytest.approx(first)


def test_seed_ends_on_requested_day(repo):
    last = repo.usage_between(*_range(repo))[-1].timestamp
    assert last.date() == LAST_SEEDED_DAY
    assert last.hour == 23


def _range(_repo):
    return datetime(2000, 1, 1), datetime(2100, 1, 1)


def test_f04_usage_output_is_aggregated_and_small(tools):
    # F-04: returning every raw row (2,160 for a month) overflowed the LLM context window.
    start = (LAST_SEEDED_DAY - timedelta(days=29)).isoformat()
    result = tools["query_energy_usage"].invoke(
        {"start_date": start, "end_date": LAST_SEEDED_DAY.isoformat()}
    )
    assert {"by_device_type", "by_device_name", "daily_totals", "avg_kwh_by_hour"} <= set(result)
    assert "records" not in result
    assert len(json.dumps(result, default=str)) < 10_000


def test_f04_solar_output_is_aggregated_and_small(tools):
    start = (LAST_SEEDED_DAY - timedelta(days=29)).isoformat()
    result = tools["query_solar_generation"].invoke(
        {"start_date": start, "end_date": LAST_SEEDED_DAY.isoformat()}
    )
    assert {"daily_totals", "avg_kwh_by_hour", "by_weather_condition"} <= set(result)
    assert result["average_daily_generation"] == pytest.approx(result["total_generation_kwh"] / 30, abs=0.01)
    assert len(json.dumps(result, default=str)) < 10_000


def test_f07_one_day_query_returns_only_that_day(tools):
    # F-07: the end of the range was inclusive, so midnight rows of the next day leaked in.
    result = tools["query_energy_usage"].invoke({"start_date": DAY, "end_date": DAY})
    assert result["total_records"] == 24 * 3
    assert list(result["daily_totals"]) == [DAY]


def test_f07_seven_day_query_returns_exactly_seven_days(tools):
    end = (LAST_SEEDED_DAY - timedelta(days=4)).isoformat()
    result = tools["query_energy_usage"].invoke({"start_date": DAY, "end_date": end})
    assert result["total_records"] == 7 * 24 * 3


@pytest.mark.parametrize("device_type", ["EV", "ev", " hvac ", "Appliance"])
def test_f08_device_filter_ignores_case_and_spaces(tools, device_type):
    # F-08: "ev" or "hvac" used to return zero rows silently.
    result = tools["query_energy_usage"].invoke(
        {"start_date": DAY, "end_date": DAY, "device_type": device_type}
    )
    assert result["total_records"] == 24


def test_f08_unknown_device_returns_error_with_options(tools):
    result = tools["query_energy_usage"].invoke({"start_date": DAY, "end_date": DAY, "device_type": "pool"})
    assert "error" in result
    assert result["available_device_types"] == ["EV", "HVAC", "appliance"]


def test_small_result_includes_raw_records(tools):
    result = tools["query_energy_usage"].invoke({"start_date": DAY, "end_date": DAY, "device_type": "EV"})
    assert len(result["records"]) == 24


def test_invalid_date_returns_error_for_the_llm(tools):
    result = tools["query_energy_usage"].invoke({"start_date": "01.10.2026", "end_date": DAY})
    assert "error" in result


def test_recent_summary_uses_injected_clock(tools):
    # NOW is 09:30 the day after the last seeded day: the last 24 h contain 14.5 seeded hours (09:30-24:00).
    result = tools["get_recent_energy_summary"].invoke({"hours": 24})
    breakdown = result["usage"]["device_breakdown"]
    assert set(breakdown) == {"EV", "HVAC", "appliance"}
    assert breakdown["EV"]["records"] == 14  # hours 10..23
    assert result["generation"]["average_weather"] in {"sunny", "partly_cloudy", "cloudy", "rainy"}
