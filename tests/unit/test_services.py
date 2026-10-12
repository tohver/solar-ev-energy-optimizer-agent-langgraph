"""Weather, prices and savings."""

import json
from datetime import date

import httpx
import pytest
import respx

from energy_advisor.clients.open_meteo import (
    FORECAST_URL,
    GEOCODING_URL,
    OpenMeteoClient,
    condition_from_wmo,
)
from energy_advisor.config import PricingSettings
from energy_advisor.services.prices import InvalidDateError, MockPriceService
from energy_advisor.services.savings import calculate_savings
from energy_advisor.services.weather import WeatherService, simulated_forecast
from tests.helpers import FIXTURES, NOW


# ----------------------------------------------------------------------------- weather
@pytest.mark.parametrize(
    ("code", "condition"),
    [
        (None, "unknown"),
        (0, "sunny"),
        (1, "sunny"),
        (2, "partly_cloudy"),
        (3, "cloudy"),
        (45, "cloudy"),
        (61, "rainy"),
        (95, "rainy"),
    ],
)
def test_wmo_codes_map_to_sky_conditions(code, condition):
    assert condition_from_wmo(code) == condition


@respx.mock
def test_live_forecast_is_parsed_from_open_meteo_response():
    # Recorded real responses: 2 forecast days for San Francisco.
    respx.get(GEOCODING_URL).respond(
        json=json.loads((FIXTURES / "open_meteo_geocode.json").read_text("utf-8"))
    )
    respx.get(FORECAST_URL).respond(
        json=json.loads((FIXTURES / "open_meteo_forecast.json").read_text("utf-8"))
    )

    result = WeatherService(OpenMeteoClient(httpx.Client()), now=lambda: NOW).forecast(
        "San Francisco, CA", days=2
    )

    assert result["source"].startswith("open-meteo")
    assert result["resolved_location"].startswith("San Francisco")
    assert len(result["hourly"]) == 48
    assert set(result["hourly"][12]) >= {"date", "hour", "temperature_c", "condition", "solar_irradiance"}
    assert len(result["daily_conditions"]) == 2


@respx.mock
def test_geocoding_is_cached_per_location():
    geo = respx.get(GEOCODING_URL).respond(
        json=json.loads((FIXTURES / "open_meteo_geocode.json").read_text("utf-8"))
    )
    respx.get(FORECAST_URL).respond(
        json=json.loads((FIXTURES / "open_meteo_forecast.json").read_text("utf-8"))
    )
    client = OpenMeteoClient(httpx.Client())
    client.forecast("San Francisco, CA", 2)
    client.forecast("San Francisco, CA", 2)
    assert geo.call_count == 1


@respx.mock
def test_api_error_falls_back_to_simulated_forecast():
    respx.get(GEOCODING_URL).respond(status_code=503)
    result = WeatherService(OpenMeteoClient(httpx.Client()), now=lambda: NOW).forecast(
        "San Francisco, CA", days=1
    )
    assert result["source"] == "simulated"
    assert "HTTPStatusError" in result["note"]


@respx.mock
def test_unknown_location_falls_back_to_simulated_forecast():
    respx.get(GEOCODING_URL).respond(json={})  # Open-Meteo returns no "results" key
    result = WeatherService(OpenMeteoClient(httpx.Client()), now=lambda: NOW).forecast("Qwxzyplk", days=1)
    assert result["source"] == "simulated"
    assert "LocationNotFoundError" in result["note"]


def test_simulated_forecast_is_deterministic_and_complete():
    a = simulated_forecast("Munich", 3, NOW)
    b = simulated_forecast("Munich", 3, NOW)
    assert a == b
    assert len(a["hourly"]) == 72
    assert all(h["solar_irradiance"] == 0 for h in a["hourly"] if h["hour"] < 6 or h["hour"] > 18)
    assert list(a["daily_conditions"]) == ["2026-10-02", "2026-10-03", "2026-10-04"]


def test_days_are_clamped_to_supported_range(offline_http):
    service = WeatherService(OpenMeteoClient(offline_http), now=lambda: NOW)
    assert service.forecast("X", days=30)["forecast_days"] == 7
    assert service.forecast("X", days=0)["forecast_days"] == 1


# ----------------------------------------------------------------------------- prices
@pytest.fixture
def prices():
    return MockPriceService(PricingSettings(), today=lambda: date(2026, 10, 12))


def test_prices_follow_time_of_use_schedule(prices):
    result = prices.get_prices("2026-10-14")
    period = {h["hour"]: h["period"] for h in result["hourly_rates"]}
    assert len(period) == 24
    assert all(period[h] == "peak" for h in range(17, 21))
    assert period[22] == period[5] == "off_peak"
    assert period[6] == period[16] == period[21] == "mid_peak"


def test_prices_use_configured_tariff(prices):
    rates = {h["period"]: h["rate"] for h in prices.get_prices("2026-10-14")["hourly_rates"]}
    assert rates == {"off_peak": 0.084, "mid_peak": 0.156, "peak": 0.264}


def test_prices_default_to_today(prices):
    assert prices.get_prices()["date"] == "2026-10-12"


def test_prices_reject_invalid_date(prices):
    with pytest.raises(InvalidDateError):
        prices.get_prices("14/10/2026")


# ----------------------------------------------------------------------------- savings
def _savings(**kwargs):
    defaults = {"device_type": "EV", "optimized_price_per_kwh": None, "co2_kg_per_kwh": 0.4}
    return calculate_savings(**{**defaults, **kwargs})


def test_f05_load_shifting_saves_money():
    # F-05: 10 kWh moved from 0.264 to 0.084 saves 10 x 0.18 = 1.80 per day. The old version returned 0.
    result = _savings(
        current_usage_kwh=10, optimized_usage_kwh=10, price_per_kwh=0.264, optimized_price_per_kwh=0.084
    )
    assert result["savings_usd"] == pytest.approx(1.80)
    assert result["annual_savings_usd"] == pytest.approx(657.0)


def test_f05_load_shifting_saves_no_co2():
    result = _savings(
        current_usage_kwh=10, optimized_usage_kwh=10, price_per_kwh=0.264, optimized_price_per_kwh=0.084
    )
    assert result["co2_saved_kg"] == 0


def test_f05_using_less_energy_still_works():
    # 2 kWh less at 0.12 = 0.24 saved; 2 kWh x 0.4 kg/kWh = 0.8 kg CO2.
    result = _savings(device_type="HVAC", current_usage_kwh=10, optimized_usage_kwh=8, price_per_kwh=0.12)
    assert result["savings_usd"] == pytest.approx(0.24)
    assert result["co2_saved_kg"] == pytest.approx(0.8)
    assert result["savings_percentage"] == pytest.approx(20.0)


def test_savings_keys_follow_currency():
    result = _savings(current_usage_kwh=1, optimized_usage_kwh=0, price_per_kwh=0.3, currency="EUR")
    assert result["savings_eur"] == pytest.approx(0.3)


def test_zero_usage_does_not_divide_by_zero():
    assert _savings(current_usage_kwh=0, optimized_usage_kwh=0, price_per_kwh=0.1)["savings_percentage"] == 0


def test_savings_tool_defaults_to_tariff_price(services):
    from energy_advisor.agent.tools import build_tools

    tool = {t.name: t for t in build_tools(services)}["calculate_energy_savings"]
    result = tool.invoke({"device_type": "HVAC", "current_usage_kwh": 10, "optimized_usage_kwh": 8})
    assert result["price_per_kwh"] == 0.12
