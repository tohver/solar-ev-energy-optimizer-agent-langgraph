"""Weather client and service: recorded Open-Meteo responses (respx) and the synthetic fallback."""

import json
from datetime import UTC, date, datetime

import httpx
import pytest
import respx

from energy_advisor.clients.open_meteo import FORECAST_URL, GEOCODING_URL, OpenMeteoClient, parse_hourly
from energy_advisor.config import LocationSettings
from energy_advisor.domain.weather import condition_from_wmo
from energy_advisor.services.weather import LIVE_SOURCE, SYNTHETIC_SOURCE, WeatherService
from tests.helpers import BERLIN, FIXTURES, fixed_clock

MUNICH_RESPONSE = json.loads((FIXTURES / "open_meteo_munich.json").read_text("utf-8"))  # UTC, 2026-10-11..14
FRANKFURT_GEOCODE = json.loads((FIXTURES / "open_meteo_geocode_frankfurt.json").read_text("utf-8"))


def _service(http=None, clock=None):
    return WeatherService(
        OpenMeteoClient(http or httpx.Client()),
        LocationSettings(),
        clock or (lambda: datetime(2026, 10, 12, 9, tzinfo=BERLIN)),
    )


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


def test_radiation_is_shifted_to_the_hour_it_belongs_to():
    # Open-Meteo labels radiation with the END of the hour: the value at 11:00 UTC is for 10:00-11:00.
    start, end = datetime(2026, 10, 12, tzinfo=UTC), datetime(2026, 10, 13, tzinfo=UTC)
    hours = parse_hourly(MUNICH_RESPONSE, start, end)
    labels = MUNICH_RESPONSE["hourly"]["time"]
    radiation = dict(zip(labels, MUNICH_RESPONSE["hourly"]["shortwave_radiation"], strict=True))
    ten = next(h for h in hours if h.time.hour == 10)
    assert ten.irradiance_w_m2 == radiation["2026-10-12T11:00"]
    assert len(hours) == 24
    assert all(h.time.tzinfo is UTC for h in hours)


@respx.mock
def test_household_location_uses_fixed_coordinates_without_geocoding():
    geocode = respx.get(GEOCODING_URL)
    forecast = respx.get(FORECAST_URL).respond(json=MUNICH_RESPONSE)
    result = _service().forecast(None, days=2)

    assert not geocode.called
    params = forecast.calls.last.request.url.params
    assert (params["latitude"], params["timezone"]) == ("48.1374", "UTC")
    assert result["source"] == LIVE_SOURCE
    assert result["timezone"] == "Europe/Berlin"
    assert len(result["hourly"]) == 48
    assert result["hourly"][0]["date"] == "2026-10-12"
    assert result["hourly"][0]["hour"] == 0  # local midnight, not UTC midnight
    assert set(result["daily_conditions"]) == {"2026-10-12", "2026-10-13"}


@respx.mock
def test_munich_aliases_are_the_household():
    respx.get(FORECAST_URL).respond(json=MUNICH_RESPONSE)
    geocode = respx.get(GEOCODING_URL)
    for name in ("Munich", "München, Bayern", "munich, germany"):
        _service().forecast(name, days=1)
    assert not geocode.called


@respx.mock
def test_ambiguous_place_names_are_flagged():
    respx.get(GEOCODING_URL).respond(json=FRANKFURT_GEOCODE)
    respx.get(FORECAST_URL).respond(json=MUNICH_RESPONSE)  # content does not matter here
    result = _service().forecast("Frankfurt", days=1)
    assert result["resolved_location"].startswith("Frankfurt am Main")
    assert "ambiguous" in result["note"]


@respx.mock
def test_api_error_falls_back_to_synthetic_weather():
    respx.get(FORECAST_URL).respond(status_code=503)
    result = _service().forecast(None, days=1)
    assert result["source"] == SYNTHETIC_SOURCE
    assert "HTTPStatusError" in result["note"]
    assert len(result["hourly"]) == 24


@respx.mock
def test_incomplete_response_falls_back_to_synthetic_weather():
    truncated = json.loads(json.dumps(MUNICH_RESPONSE))
    for key, values in truncated["hourly"].items():
        truncated["hourly"][key] = values[:30]  # covers only part of the requested days
    respx.get(FORECAST_URL).respond(json=truncated)
    assert _service().forecast(None, days=2)["source"] == SYNTHETIC_SOURCE


@respx.mock
def test_unknown_location_falls_back_to_home():
    respx.get(GEOCODING_URL).respond(json={})
    respx.get(FORECAST_URL).respond(json=MUNICH_RESPONSE)
    result = _service().forecast("Qwxzyplk", days=1)
    assert "not found" in result["note"]


def test_days_are_clamped(offline_http):
    service = _service(offline_http, fixed_clock)
    assert service.forecast(None, days=30)["forecast_days"] == 7
    assert service.forecast(None, days=0)["forecast_days"] == 1


def test_history_offline_is_synthetic_and_complete(offline_http):
    series = _service(offline_http, fixed_clock).history(date(2026, 10, 1), date(2026, 10, 3))
    assert series.source == SYNTHETIC_SOURCE
    assert len(series.hours) == 72
