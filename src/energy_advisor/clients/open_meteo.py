"""Client for the free Open-Meteo APIs (no key needed): geocoding and hourly forecast."""

from collections import Counter
from datetime import datetime
from functools import lru_cache
from typing import Any

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,shortwave_radiation"
CURRENT_VARIABLES = "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
DAYTIME_HOURS = range(8, 18)  # the daytime sky is what matters for solar


class LocationNotFoundError(ValueError):
    pass


def condition_from_wmo(code: int | None) -> str:
    """Map a WMO weather code (used by Open-Meteo) to the 4 sky conditions used across the project."""
    if code is None:
        return "unknown"
    if code <= 1:
        return "sunny"  # clear / mainly clear
    if code == 2:
        return "partly_cloudy"
    if code in (3, 45, 48):
        return "cloudy"  # overcast / fog
    return "rainy"  # drizzle, rain, snow, showers, thunderstorm


class OpenMeteoClient:
    def __init__(self, http: httpx.Client | None = None, timeout: float = 10.0) -> None:
        self._http = http or httpx.Client(timeout=timeout)
        # Cache per instance: the same place name is geocoded once.
        self.geocode = lru_cache(maxsize=64)(self._geocode)

    def _geocode(self, location: str) -> dict[str, Any]:
        """Resolve 'San Francisco, CA' to coordinates (uses the text before the first comma)."""
        name = location.split(",")[0].strip()
        resp = self._http.get(GEOCODING_URL, params={"name": name, "count": 1})
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            raise LocationNotFoundError(f"Location not found: {location}")
        r = results[0]
        return {
            "name": r.get("name"),
            "country": r.get("country"),
            "admin1": r.get("admin1"),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
        }

    def forecast(self, location: str, days: int) -> dict[str, Any]:
        """Live hourly forecast: temperature, humidity, wind, sky condition and solar irradiance (W/m2)."""
        place = self.geocode(location)
        resp = self._http.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "hourly": HOURLY_VARIABLES,
                "current": CURRENT_VARIABLES,
                "forecast_days": days,
                "timezone": "auto",
                "wind_speed_unit": "ms",
            },
        )
        resp.raise_for_status()
        return parse_forecast(resp.json(), location, place, days)


def parse_forecast(data: dict[str, Any], location: str, place: dict[str, Any], days: int) -> dict[str, Any]:
    """Convert the Open-Meteo JSON response into the tool's output format."""
    h = data["hourly"]
    hourly = []
    daytime: dict[str, list[str]] = {}
    for i, ts in enumerate(h["time"]):
        dt = datetime.fromisoformat(ts)
        condition = condition_from_wmo(h["weather_code"][i])
        hourly.append(
            {
                "date": dt.date().isoformat(),
                "hour": dt.hour,
                "temperature_c": h["temperature_2m"][i],
                "condition": condition,
                "solar_irradiance": h["shortwave_radiation"][i],
                "humidity": h["relative_humidity_2m"][i],
                "wind_speed": h["wind_speed_10m"][i],
            }
        )
        if dt.hour in DAYTIME_HOURS:
            daytime.setdefault(dt.date().isoformat(), []).append(condition)

    cur = data.get("current", {})
    resolved = f"{place['name']}, {place.get('admin1') or ''} {place.get('country') or ''}".strip()
    return {
        "location": location,
        "resolved_location": resolved,
        "source": "open-meteo.com (live forecast)",
        "timezone": data.get("timezone"),
        "forecast_days": days,
        "units": {"temperature": "C", "solar_irradiance": "W/m2", "wind_speed": "m/s", "humidity": "%"},
        "current": {
            "temperature_c": cur.get("temperature_2m"),
            "condition": condition_from_wmo(cur.get("weather_code")),
            "humidity": cur.get("relative_humidity_2m"),
            "wind_speed": cur.get("wind_speed_10m"),
        },
        "daily_conditions": {d: Counter(c).most_common(1)[0][0] for d, c in daytime.items()},
        "daily_peak_irradiance": {
            d: max(x["solar_irradiance"] or 0 for x in hourly if x["date"] == d) for d in daytime
        },
        "hourly": hourly,
    }
