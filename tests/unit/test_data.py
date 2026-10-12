"""Household model, seeding, repository queries and database tools (regressions F-04, F-07, F-08, F-10)."""

import json
import random
from datetime import UTC, date, datetime, timedelta

import pytest

from energy_advisor.agent.tools import build_tools
from energy_advisor.config import HouseholdSettings
from energy_advisor.data.aggregation import energy_balance
from energy_advisor.data.seed import build_history, generate_household, heat_pump_cop, seed_database
from energy_advisor.domain.weather import HourlyWeather
from energy_advisor.timeutils import day_bounds_utc
from tests.conftest import SEEDED_DAYS
from tests.helpers import BERLIN, LAST_SEEDED_DAY

DAY = (LAST_SEEDED_DAY - timedelta(days=5)).isoformat()
EVERYTHING = (datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC))


@pytest.fixture
def tools(services, repo):
    return {t.name: t for t in build_tools(services)}


def _weather_day(day, temperature, irradiance=0.0):
    start, end = day_bounds_utc(day, day, BERLIN)
    hours = []
    t = start
    while t < end:
        local_hour = t.astimezone(BERLIN).hour
        sun = irradiance if 9 <= local_hour <= 15 else 0.0
        hours.append(HourlyWeather(t, temperature, sun, "sunny" if sun else "cloudy"))
        t += timedelta(hours=1)
    return hours


# ----------------------------------------------------------------------------- household model
def test_heat_pump_cop_falls_in_the_cold():
    assert heat_pump_cop(-15) == 2.0
    assert heat_pump_cop(0) == 3.0
    assert heat_pump_cop(30) == 4.5


def test_heat_pump_uses_more_electricity_when_cold():
    def heat_pump_kwh(temperature):
        weather = _weather_day(LAST_SEEDED_DAY, temperature)
        usage, _ = generate_household(
            weather,
            dict.fromkeys((h.time for h in weather), 0.3),
            HouseholdSettings(),
            BERLIN,
            random.Random(1),
        )
        return sum(u.consumption_kwh for u in usage if u.device_type == "heat_pump")

    cold, mild, warm = heat_pump_kwh(-5), heat_pump_kwh(8), heat_pump_kwh(20)
    assert cold > mild > warm > 0  # warm: hot water only


def test_pv_rows_only_when_the_sun_shines_and_cost_is_kwh_times_price():
    weather = _weather_day(LAST_SEEDED_DAY, 15, irradiance=600)
    prices = {h.time: 0.25 + i / 1000 for i, h in enumerate(weather)}
    usage, solar = generate_household(weather, prices, HouseholdSettings(), BERLIN, random.Random(1))
    assert len(solar) == 7  # 09:00-15:00
    assert solar[0].generation_kwh == pytest.approx(8 * 0.6 * 0.8)
    assert all(u.cost_eur == pytest.approx(u.consumption_kwh * prices[u.timestamp], abs=1e-3) for u in usage)


# ----------------------------------------------------------------------------- seeding
def test_seed_covers_requested_days_in_local_time(repo):
    usage = repo.usage_between(*EVERYTHING)
    local_days = {u.timestamp.astimezone(BERLIN).date() for u in usage}
    assert min(local_days) == LAST_SEEDED_DAY - timedelta(days=SEEDED_DAYS - 1)
    assert max(local_days) == LAST_SEEDED_DAY
    base = [u for u in usage if u.device_type == "base_load"]
    assert len(base) == SEEDED_DAYS * 24  # base load every hour
    assert {u.device_type for u in usage} >= {"base_load", "heat_pump", "appliance", "EV"}


def test_f10_reseeding_replaces_data_instead_of_duplicating(services, repo):
    before = len(repo.usage_between(*EVERYTHING))
    s = services.settings
    history = build_history(
        services.weather, services.prices, s.household, BERLIN, SEEDED_DAYS, LAST_SEEDED_DAY, seed=42
    )
    result = seed_database(repo, history)
    assert result.deleted["energy_usage"] == before
    assert len(repo.usage_between(*EVERYTHING)) == before


def test_seed_is_reproducible(services, repo):
    s = services.settings
    a = build_history(services.weather, services.prices, s.household, BERLIN, 5, LAST_SEEDED_DAY, seed=7)
    b = build_history(services.weather, services.prices, s.household, BERLIN, 5, LAST_SEEDED_DAY, seed=7)
    assert [u.consumption_kwh for u in a.usage] == [u.consumption_kwh for u in b.usage]
    assert a.weather_source.startswith("simulated")  # offline in tests


def test_seed_rejects_too_many_days(services):
    s = services.settings
    with pytest.raises(ValueError, match="between 1 and 90"):
        build_history(services.weather, services.prices, s.household, BERLIN, 120, LAST_SEEDED_DAY, seed=1)


def test_seed_handles_daylight_saving_day(services):
    s = services.settings
    history = build_history(
        services.weather, services.prices, s.household, BERLIN, 1, date(2026, 10, 25), seed=1
    )
    assert sum(1 for u in history.usage if u.device_type == "base_load") == 25


# ----------------------------------------------------------------------------- tools
def test_f04_usage_output_is_aggregated_and_small(tools):
    start = (LAST_SEEDED_DAY - timedelta(days=SEEDED_DAYS - 1)).isoformat()
    result = tools["query_energy_usage"].invoke(
        {"start_date": start, "end_date": LAST_SEEDED_DAY.isoformat()}
    )
    assert {"by_device_type", "by_device_name", "daily_totals", "avg_kwh_by_hour"} <= set(result)
    assert "records" not in result
    assert len(json.dumps(result, default=str)) < 10_000


def test_f04_solar_output_is_aggregated_and_small(tools):
    start = (LAST_SEEDED_DAY - timedelta(days=SEEDED_DAYS - 1)).isoformat()
    result = tools["query_solar_generation"].invoke(
        {"start_date": start, "end_date": LAST_SEEDED_DAY.isoformat()}
    )
    assert {"daily_totals", "avg_kwh_by_hour", "by_weather_condition", "energy_balance"} <= set(result)
    assert result["average_daily_generation_kwh"] == pytest.approx(
        result["total_generation_kwh"] / SEEDED_DAYS, abs=0.01
    )
    assert len(json.dumps(result, default=str)) < 10_000


def test_f07_one_day_query_returns_only_that_local_day(tools):
    # F-07: the end of the range was inclusive, so midnight rows of the next day leaked in.
    result = tools["query_energy_usage"].invoke({"start_date": DAY, "end_date": DAY})
    assert list(result["daily_totals"]) == [DAY]
    assert result["by_device_type"]["base_load"]["hours_with_use"] == 24


@pytest.mark.parametrize("device_type", ["base_load", "BASE_LOAD", " Base_Load "])
def test_f08_device_filter_ignores_case_and_spaces(tools, device_type):
    result = tools["query_energy_usage"].invoke(
        {"start_date": DAY, "end_date": DAY, "device_type": device_type}
    )
    assert result["total_records"] == 24


def test_f08_unknown_device_returns_error_with_options(tools):
    result = tools["query_energy_usage"].invoke({"start_date": DAY, "end_date": DAY, "device_type": "pool"})
    assert "error" in result
    assert "heat_pump" in result["available_device_types"]


def test_reversed_or_invalid_dates_return_errors_for_the_llm(tools):
    assert "error" in tools["query_energy_usage"].invoke({"start_date": DAY, "end_date": "2026-01-01"})
    assert "error" in tools["query_solar_generation"].invoke({"start_date": "01.10.2026", "end_date": DAY})


def test_usage_costs_are_in_euro(tools):
    result = tools["query_energy_usage"].invoke({"start_date": DAY, "end_date": DAY})
    assert 0.2 < result["average_price_paid_eur_per_kwh"] < 0.6
    assert result["total_cost_eur"] == pytest.approx(
        sum(d["cost_eur"] for d in result["by_device_type"].values()), abs=0.05
    )


def test_energy_balance_adds_up(repo):
    balance = energy_balance(repo.usage_between(*EVERYTHING), repo.generation_between(*EVERYTHING), 0.078)
    assert balance["solar_self_consumed_kwh"] + balance["solar_exported_kwh"] == pytest.approx(
        balance["solar_generated_kwh"], abs=0.02
    )
    assert balance["solar_self_consumed_kwh"] + balance["grid_import_kwh"] == pytest.approx(
        balance["consumption_kwh"], abs=0.02
    )
    assert 0 <= balance["self_consumption_rate"] <= 1


def test_recent_summary_uses_injected_clock(tools):
    # NOW is 09:30 the day after the last seeded day: the last 24 h contain the seeded hours 09:30-24:00.
    result = tools["get_recent_energy_summary"].invoke({"hours": 24})
    assert result["usage"]["total_consumption_kwh"] > 0
    assert set(result["usage"]["device_breakdown"]) >= {"base_load", "heat_pump"}
