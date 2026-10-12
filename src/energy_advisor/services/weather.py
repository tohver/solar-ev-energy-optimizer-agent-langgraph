"""Weather forecast: live Open-Meteo data with a deterministic simulated fallback."""

import hashlib
import logging
import math
import random
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from energy_advisor.clients.open_meteo import OpenMeteoClient

logger = logging.getLogger(__name__)

# Sunlight multiplier per sky condition (consistent with the generated history in the DB)
CONDITION_SOLAR_FACTOR = {"sunny": 1.0, "partly_cloudy": 0.6, "cloudy": 0.3, "rainy": 0.1}
CONDITION_WEIGHTS = {"sunny": 0.4, "partly_cloudy": 0.3, "cloudy": 0.2, "rainy": 0.1}
MAX_FORECAST_DAYS = 7


class WeatherService:
    def __init__(self, client: OpenMeteoClient, now: Callable[[], datetime] = datetime.now) -> None:
        self._client = client
        self._now = now

    def forecast(self, location: str, days: int = 3) -> dict[str, Any]:
        """Live forecast; if the API is unreachable or the location is unknown, a simulated one."""
        days = max(1, min(int(days), MAX_FORECAST_DAYS))
        try:
            return self._client.forecast(location, days)
        except Exception as exc:  # any failure of the external API must not break the agent
            logger.warning("Live weather unavailable for %r: %s", location, exc)
            forecast = simulated_forecast(location, days, self._now())
            forecast["note"] = (
                f"Live weather API unavailable ({type(exc).__name__}); using simulated forecast."
            )
            return forecast


def simulated_forecast(location: str, days: int, now: datetime) -> dict[str, Any]:
    """Simulated forecast, deterministic per (location, date) so repeated calls agree."""
    today: date = now.date()
    seed = int(hashlib.md5(f"{location.lower()}{today}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    base_temp = rng.uniform(12, 22)
    hourly = []
    daily_conditions: dict[str, str] = {}
    for d in range(days):
        day = (today + timedelta(days=d)).isoformat()
        condition = rng.choices(list(CONDITION_WEIGHTS), weights=list(CONDITION_WEIGHTS.values()))[0]
        daily_conditions[day] = condition
        for hour in range(24):
            # Bell-shaped irradiance between 6:00 and 18:00, peak at noon (max ~1000 W/m2)
            daylight = max(0.0, math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0
            irradiance = round(1000 * daylight * CONDITION_SOLAR_FACTOR[condition] * rng.uniform(0.9, 1.0), 1)
            temp = base_temp + 6 * math.sin(math.pi * (hour - 9) / 12) + rng.uniform(-1, 1)
            hourly.append(
                {
                    "date": day,
                    "hour": hour,
                    "temperature_c": round(temp, 1),
                    "condition": condition,
                    "solar_irradiance": irradiance,
                    "humidity": round(rng.uniform(40, 80), 1),
                    "wind_speed": round(rng.uniform(1, 12), 1),
                }
            )

    current = hourly[now.hour]
    return {
        "location": location,
        "source": "simulated",
        "forecast_days": days,
        "current": {k: current[k] for k in ("temperature_c", "condition", "humidity", "wind_speed")},
        "daily_conditions": daily_conditions,
        "hourly": hourly,
    }
