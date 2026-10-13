"""Synthetic but plausible weather, used when the live weather API is unavailable.

Day length and sun height come from the real sun position for the configured coordinates; temperatures
follow approximate Munich monthly climate normals. Deterministic for a given random generator.
"""

import math
import random
from datetime import date
from zoneinfo import ZoneInfo

from energy_advisor.domain.solar import hourly_clear_sky_ghi
from energy_advisor.domain.weather import CONDITION_SOLAR_FACTOR, HourlyWeather
from energy_advisor.timeutils import day_bounds_utc, hours_between

# Approximate monthly mean temperatures for Munich (deg C), January..December.
MONTHLY_MEAN_C = (-0.5, 0.8, 4.6, 8.8, 13.2, 16.4, 18.4, 18.0, 13.9, 9.4, 4.1, 0.9)
DIURNAL_AMPLITUDE_C = 4.5
# How often each sky condition occurs: winter half-year is cloudier.
SUMMER_WEIGHTS = {"sunny": 0.35, "partly_cloudy": 0.35, "cloudy": 0.2, "rainy": 0.1}
WINTER_WEIGHTS = {"sunny": 0.15, "partly_cloudy": 0.25, "cloudy": 0.4, "rainy": 0.2}


def condition_weights(month: int) -> dict[str, float]:
    return SUMMER_WEIGHTS if 4 <= month <= 9 else WINTER_WEIGHTS


def synthetic_weather(
    first_day: date,
    last_day: date,
    latitude: float,
    longitude: float,
    tz: ZoneInfo,
    rng: random.Random,
) -> list[HourlyWeather]:
    """Hourly weather for the local days ``first_day..last_day`` (inclusive)."""
    start, end = day_bounds_utc(first_day, last_day, tz)
    daily: dict[date, tuple[str, float]] = {}
    result = []
    for hour in hours_between(start, end):
        local = hour.astimezone(tz)
        day = local.date()
        if day not in daily:
            weights = condition_weights(day.month)
            condition = rng.choices(list(weights), weights=list(weights.values()))[0]
            daily[day] = (condition, MONTHLY_MEAN_C[day.month - 1] + rng.uniform(-3, 3))
        condition, mean_temp = daily[day]
        temperature = mean_temp + DIURNAL_AMPLITUDE_C * math.sin(math.pi * (local.hour - 9) / 12)
        irradiance = hourly_clear_sky_ghi(hour, latitude, longitude) * CONDITION_SOLAR_FACTOR[condition]
        result.append(
            HourlyWeather(
                time=hour,
                temperature_c=round(temperature + rng.uniform(-0.8, 0.8), 1),
                irradiance_w_m2=round(irradiance * rng.uniform(0.9, 1.0), 1),
                condition=condition,
                humidity=round(rng.uniform(55, 90), 1),
                wind_speed=round(rng.uniform(0.5, 6.0), 1),
            )
        )
    return result


def seed_for(*parts: object) -> int:
    """Stable seed from values (Python's hash() is randomized per process)."""
    text = "|".join(str(p) for p in parts)
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % (2**32)
