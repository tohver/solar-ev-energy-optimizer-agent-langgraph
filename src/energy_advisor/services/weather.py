"""Weather: live Open-Meteo data with a deterministic synthetic fallback."""

import logging
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from energy_advisor.clients.open_meteo import OpenMeteoClient
from energy_advisor.config import LocationSettings
from energy_advisor.domain.climate import seed_for, synthetic_weather
from energy_advisor.domain.weather import HourlyWeather
from energy_advisor.timeutils import HOUR, Clock, day_bounds_utc

logger = logging.getLogger(__name__)
MAX_FORECAST_DAYS = 7
MIN_COVERAGE = 0.9  # share of the requested hours a live response must contain
DAYTIME_HOURS = range(8, 18)  # the daytime sky is what matters for solar

LIVE_SOURCE = "open-meteo.com (live)"
SYNTHETIC_SOURCE = "simulated (climate-based)"


@dataclass(frozen=True)
class ResolvedLocation:
    label: str
    latitude: float
    longitude: float
    tz: ZoneInfo
    note: str | None = None


@dataclass(frozen=True)
class WeatherSeries:
    hours: list[HourlyWeather]
    source: str
    current: dict[str, Any]
    note: str | None = None

    @property
    def is_live(self) -> bool:
        return self.source == LIVE_SOURCE


class WeatherService:
    def __init__(self, client: OpenMeteoClient, location: LocationSettings, clock: Clock) -> None:
        self._client = client
        self._home = location
        self._clock = clock

    # ------------------------------------------------------------------ location
    def resolve(self, place: str | None) -> ResolvedLocation:
        """The household's own location uses fixed coordinates; any other place is geocoded."""
        home = self._home
        if home.matches(place):
            return ResolvedLocation(home.name, home.latitude, home.longitude, home.tz)
        found = self._client.geocode(place or "")
        note = (
            f"'{place}' is ambiguous ({found.other_matches + 1} places share the name); used {found.label}."
            if found.other_matches
            else None
        )
        return ResolvedLocation(
            found.label, found.latitude, found.longitude, ZoneInfo(found.timezone or home.timezone), note
        )

    # ------------------------------------------------------------------ data
    def series(self, first_day: date, last_day: date, place: str | None = None) -> WeatherSeries:
        """Hourly weather for local days ``first_day..last_day`` (past ~3 months to +16 days)."""
        try:
            loc = self.resolve(place)
        except Exception as exc:
            logger.warning("Location %r could not be resolved: %s", place, exc)
            loc = ResolvedLocation(self._home.name, self._home.latitude, self._home.longitude, self._home.tz)
            fallback_note = f"Location '{place}' not found ({type(exc).__name__}); using {self._home.name}."
        else:
            fallback_note = None

        start, end = day_bounds_utc(first_day, last_day, loc.tz)
        expected = int((end - start) / HOUR)
        try:
            response = self._client.weather(loc.latitude, loc.longitude, start, end)
            if len(response.hours) < MIN_COVERAGE * expected:
                raise ValueError(f"incomplete response ({len(response.hours)}/{expected} hours)")
            notes = " ".join(n for n in (fallback_note, loc.note) if n) or None
            return WeatherSeries(response.hours, LIVE_SOURCE, response.current, notes)
        except Exception as exc:  # the agent must keep working when the API is down
            logger.warning("Live weather unavailable: %s", exc)
            rng = random.Random(seed_for(round(loc.latitude, 2), round(loc.longitude, 2), first_day))
            hours = synthetic_weather(first_day, last_day, loc.latitude, loc.longitude, loc.tz, rng)
            now = self._clock()
            current_hour = next((h for h in hours if h.time <= now < h.time + HOUR), hours[0])
            note = f"Live weather API unavailable ({type(exc).__name__}); using a simulated, climate-based forecast."
            return WeatherSeries(
                hours,
                SYNTHETIC_SOURCE,
                _current_from(current_hour),
                " ".join(filter(None, (fallback_note, note))),
            )

    def forecast(self, place: str | None = None, days: int = 3) -> dict[str, Any]:
        """Tool output: hourly forecast starting today (local time), with a daily summary."""
        days = max(1, min(int(days), MAX_FORECAST_DAYS))
        try:
            loc = self.resolve(place)
            tz, label = loc.tz, loc.label
        except Exception:
            tz, label = self._home.tz, self._home.name
        today = self._clock().astimezone(tz).date()
        data = self.series(today, today + timedelta(days=days - 1), place)
        return forecast_output(data, tz, place or self._home.name, label, days)

    def history(self, first_day: date, last_day: date) -> WeatherSeries:
        """Past weather at the household (for generating its history)."""
        return self.series(first_day, last_day, None)


def _current_from(hour: HourlyWeather) -> dict[str, Any]:
    return {
        "temperature_c": hour.temperature_c,
        "condition": hour.condition,
        "humidity": hour.humidity,
        "wind_speed": hour.wind_speed,
    }


def forecast_output(
    data: WeatherSeries, tz: ZoneInfo, requested: str, label: str, days: int
) -> dict[str, Any]:
    hourly = []
    daytime: dict[str, list[str]] = {}
    temps: dict[str, list[float]] = {}
    peak: dict[str, float] = {}
    for h in data.hours:
        local = h.time.astimezone(tz)
        day = local.date().isoformat()
        hourly.append(
            {
                "date": day,
                "hour": local.hour,
                "temperature_c": h.temperature_c,
                "condition": h.condition,
                "solar_irradiance": h.irradiance_w_m2,
            }
        )
        temps.setdefault(day, []).append(h.temperature_c)
        peak[day] = max(peak.get(day, 0.0), h.irradiance_w_m2)
        if local.hour in DAYTIME_HOURS:
            daytime.setdefault(day, []).append(h.condition)

    output: dict[str, Any] = {
        "location": requested,
        "resolved_location": label,
        "timezone": str(tz),
        "source": data.source,
        "forecast_days": days,
        "units": {"temperature": "C", "solar_irradiance": "W/m2 (mean over the hour)", "wind_speed": "m/s"},
        "current": data.current,
        "daily_conditions": {d: Counter(c).most_common(1)[0][0] for d, c in daytime.items()},
        "daily_temperature_c": {
            d: {"min": round(min(t), 1), "max": round(max(t), 1)} for d, t in temps.items()
        },
        "daily_peak_irradiance": {d: round(v, 1) for d, v in peak.items()},
        "hourly": hourly,
    }
    if data.note:
        output["note"] = data.note
    return output


def local_hour_label(value: datetime, tz: ZoneInfo) -> str:
    return value.astimezone(tz).strftime("%H:%M")
