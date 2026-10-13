"""Client for the free Open-Meteo APIs (no key needed): geocoding and hourly weather.

All requests use ``timezone=UTC``: local wall-clock times are ambiguous on daylight-saving days,
so conversion to local time happens in our code.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import httpx

from energy_advisor.domain.weather import HourlyWeather, condition_from_wmo

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,shortwave_radiation"
CURRENT_VARIABLES = "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
USER_AGENT = "solar-ev-energy-optimizer (github.com/tohver/solar-ev-energy-optimizer-agent-langgraph)"


class LocationNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class Place:
    name: str
    latitude: float
    longitude: float
    country: str | None = None
    admin1: str | None = None
    timezone: str | None = None
    other_matches: int = 0

    @property
    def label(self) -> str:
        return ", ".join(p for p in (self.name, self.admin1, self.country) if p)


@dataclass(frozen=True)
class WeatherResponse:
    hours: list[HourlyWeather]
    current: dict[str, Any]


def default_http_client(timeout: float = 15.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})


class OpenMeteoClient:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or default_http_client()
        self.geocode = lru_cache(maxsize=64)(self._geocode)  # one lookup per place name

    def _geocode(self, location: str) -> Place:
        """Resolve 'Hamburg, Germany' to coordinates (searches the text before the first comma)."""
        name = location.split(",")[0].strip()
        resp = self._http.get(GEOCODING_URL, params={"name": name, "count": 5})
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            raise LocationNotFoundError(f"Location not found: {location}")
        best = results[0]  # Open-Meteo ranks by relevance and population
        # other places whose name contains the searched text, e.g. "Frankfurt" -> Frankfurt (Oder)
        same_name = [r for r in results[1:] if name.lower() in str(r.get("name", "")).lower()]
        return Place(
            name=best["name"],
            latitude=best["latitude"],
            longitude=best["longitude"],
            country=best.get("country"),
            admin1=best.get("admin1"),
            timezone=best.get("timezone"),
            other_matches=len(same_name),
        )

    def weather(self, latitude: float, longitude: float, start: datetime, end: datetime) -> WeatherResponse:
        """Hourly weather for the UTC interval ``[start, end)``, plus current conditions.

        Works for the past (about 3 months) and the future (up to 16 days).
        """
        resp = self._http.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": HOURLY_VARIABLES,
                "current": CURRENT_VARIABLES,
                "timezone": "UTC",
                "wind_speed_unit": "ms",
                "start_date": start.astimezone(UTC).date().isoformat(),
                # radiation is labelled with the END of its hour, so fetch one extra hour at the end
                "end_date": (end.astimezone(UTC) + timedelta(hours=1)).date().isoformat(),
            },
        )
        resp.raise_for_status()
        return WeatherResponse(parse_hourly(resp.json(), start, end), parse_current(resp.json()))


def parse_hourly(data: dict[str, Any], start: datetime, end: datetime) -> list[HourlyWeather]:
    """Convert Open-Meteo's column arrays into ``HourlyWeather`` rows for hours in ``[start, end)``.

    ``shortwave_radiation`` is the mean over the PRECEDING hour, so the value labelled 13:00 belongs
    to the hour 12:00-13:00; temperature and the other values are instantaneous.
    """
    h = data["hourly"]
    times = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in h["time"]]
    radiation_for_hour_start = {
        t - timedelta(hours=1): r for t, r in zip(times, h["shortwave_radiation"], strict=True)
    }
    rows = []
    for i, t in enumerate(times):
        if not (start <= t < end) or h["temperature_2m"][i] is None:
            continue
        rows.append(
            HourlyWeather(
                time=t,
                temperature_c=h["temperature_2m"][i],
                irradiance_w_m2=radiation_for_hour_start.get(t) or 0.0,
                condition=condition_from_wmo(h["weather_code"][i]),
                humidity=h["relative_humidity_2m"][i],
                wind_speed=h["wind_speed_10m"][i],
            )
        )
    return rows


def parse_current(data: dict[str, Any]) -> dict[str, Any]:
    cur = data.get("current") or {}
    return {
        "temperature_c": cur.get("temperature_2m"),
        "condition": condition_from_wmo(cur.get("weather_code")),
        "humidity": cur.get("relative_humidity_2m"),
        "wind_speed": cur.get("wind_speed_10m"),
    }
