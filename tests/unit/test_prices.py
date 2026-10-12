"""Price chain, tariff conversion, hourly aggregation and the savings calculator."""

from datetime import date, datetime

import pytest

from energy_advisor.clients.prices.synthetic import SyntheticPriceProvider
from energy_advisor.config import LocationSettings, TariffSettings
from energy_advisor.domain.prices import PriceUnavailableError
from energy_advisor.services.prices import PriceService, assign_periods
from energy_advisor.services.savings import calculate_savings
from energy_advisor.services.tariff import retail_price
from energy_advisor.timeutils import InvalidDateError, day_bounds_utc
from tests.helpers import BERLIN, StaticPriceProvider, fixed_clock, hourly_prices

TARIFF = TariffSettings()


def _service(*providers):
    return PriceService(list(providers), TARIFF, LocationSettings(), fixed_clock)


# ----------------------------------------------------------------------------- tariff
def test_retail_price_formula():
    # (100 EUR/MWh = 0.10 EUR/kWh + 0.19 surcharges) x 1.19 VAT
    assert retail_price(100, TARIFF) == pytest.approx(0.3451)


def test_negative_spot_prices_lower_the_retail_price():
    assert retail_price(-50, TARIFF) < retail_price(0, TARIFF)


# ----------------------------------------------------------------------------- chain
def test_first_provider_that_works_is_used():
    broken, working = StaticPriceProvider(name="broken", fail=True), StaticPriceProvider(80, name="backup")
    result = _service(broken, working).get_prices("2026-10-13")
    assert result["source"] == "backup"
    assert result["is_estimate"] is False
    assert "broken" in result["note"]
    assert broken.calls == working.calls == 1


def test_incomplete_data_moves_on_to_the_next_provider():
    class HalfDay(StaticPriceProvider):
        def fetch(self, start, end):
            return super().fetch(start, end)[:12]

    result = _service(HalfDay(name="half"), StaticPriceProvider(name="full")).get_prices("2026-10-13")
    assert result["source"] == "full"


def test_no_provider_raises():
    with pytest.raises(PriceUnavailableError):
        _service(StaticPriceProvider(fail=True)).get_prices("2026-10-13")


def test_estimates_are_flagged_for_the_llm():
    result = _service(SyntheticPriceProvider(BERLIN)).get_prices("2026-10-13")
    assert result["is_estimate"] is True
    assert "ESTIMATE" in result["note"]


def test_default_day_is_today_in_local_time():
    assert _service(StaticPriceProvider()).get_prices()["date"] == "2026-10-02"


def test_invalid_date_is_rejected():
    with pytest.raises(InvalidDateError):
        _service(StaticPriceProvider()).get_prices("13.10.2026")


# ----------------------------------------------------------------------------- output
def test_quarter_hour_slots_are_averaged_to_hours():
    result = _service(StaticPriceProvider(120, slot_minutes=15)).get_prices("2026-10-13")
    assert len(result["hourly_rates"]) == 24
    assert result["hourly_rates"][0]["rate"] == retail_price(120, TARIFF)


@pytest.mark.parametrize(("day", "hours"), [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25)])
def test_daylight_saving_days(day, hours):
    result = _service(StaticPriceProvider(slot_minutes=15)).get_prices(day.isoformat())
    assert len(result["hourly_rates"]) == hours
    assert len({r["time"] for r in result["hourly_rates"]}) == hours  # "time" keeps the duplicate 02:00 apart


def test_summary_names_cheapest_and_most_expensive_hours():
    class Shaped(StaticPriceProvider):
        def fetch(self, start, end):
            values = [100.0] * 24
            values[3], values[13], values[19] = 20.0, -10.0, 300.0
            return hourly_prices(start, end, values)

    result = _service(Shaped()).get_prices("2026-10-13")
    summary = result["summary"]
    assert {"03:00", "13:00"} <= set(summary["cheapest_hours"])
    assert "19:00" in summary["most_expensive_hours"]
    assert summary["negative_spot_price_hours"] == ["13:00"]
    assert summary["min_rate"] == retail_price(-10, TARIFF)
    periods = {r["hour"]: r["period"] for r in result["hourly_rates"]}
    assert periods[13] == "off_peak"
    assert periods[19] == "peak"
    assert result["currency"] == "EUR"


def test_assign_periods_uses_terciles():
    labels = assign_periods([float(i) for i in range(24)])
    assert labels.count("off_peak") == labels.count("mid_peak") == labels.count("peak") == 8
    assert labels[0] == "off_peak"
    assert labels[23] == "peak"


def test_synthetic_profile_is_cheaper_at_midday_than_in_the_evening():
    start, end = day_bounds_utc(date(2026, 7, 1), date(2026, 7, 1), BERLIN)
    prices = {
        p.start.astimezone(BERLIN).hour: p.eur_per_mwh
        for p in SyntheticPriceProvider(BERLIN).fetch(start, end)
    }
    assert prices[13] < prices[3] < prices[19]


def test_hourly_retail_for_longer_ranges():
    start, end = day_bounds_utc(date(2026, 10, 1), date(2026, 10, 3), BERLIN)
    prices, source = _service(StaticPriceProvider(100)).hourly_retail(start, end)
    assert len(prices) == 72
    assert source == "static"
    assert all(isinstance(t, datetime) for t in prices)


# ----------------------------------------------------------------------------- savings
def _savings(**kwargs):
    defaults = {"device_type": "EV", "optimized_price_per_kwh": None, "co2_kg_per_kwh": 0.36}
    return calculate_savings(**{**defaults, **kwargs})


def test_f05_load_shifting_saves_money_but_no_co2():
    # F-05: 10 kWh moved from 0.42 to 0.30 EUR/kWh saves 1.20 EUR per day; the old version returned 0.
    result = _savings(
        current_usage_kwh=10, optimized_usage_kwh=10, price_per_kwh=0.42, optimized_price_per_kwh=0.30
    )
    assert result["savings_eur"] == pytest.approx(1.20)
    assert result["annual_savings_eur"] == pytest.approx(438.0)
    assert result["co2_saved_kg"] == 0


def test_f05_using_less_energy_saves_money_and_co2():
    result = _savings(
        device_type="heat_pump", current_usage_kwh=10, optimized_usage_kwh=9, price_per_kwh=0.35
    )
    assert result["savings_eur"] == pytest.approx(0.35)
    assert result["co2_saved_kg"] == pytest.approx(0.36)
    assert result["savings_percentage"] == pytest.approx(10.0)


def test_zero_usage_does_not_divide_by_zero():
    assert _savings(current_usage_kwh=0, optimized_usage_kwh=0, price_per_kwh=0.3)["savings_percentage"] == 0


def test_savings_tool_defaults_to_fixed_tariff(services):
    from energy_advisor.agent.tools import build_tools

    tool = {t.name: t for t in build_tools(services)}["calculate_energy_savings"]
    result = tool.invoke({"device_type": "heat_pump", "current_usage_kwh": 10, "optimized_usage_kwh": 9})
    assert result["price_per_kwh"] == TARIFF.fixed_price_eur_per_kwh
    assert "savings_eur" in result
