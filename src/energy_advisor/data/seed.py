"""Reproducible sample data for the demo household.

The generator is seeded (same seed, same data), idempotent (existing rows are replaced) and
writes everything in one transaction. The last generated day is ``end_date`` (yesterday by
default), so questions about "the last 7 days" always find data.
"""

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from energy_advisor.data.models import EnergyUsage, SolarGeneration
from energy_advisor.data.repository import EnergyRepository


@dataclass(frozen=True)
class DeviceProfile:
    base_kwh: float
    variation: float
    peak_hours: frozenset[int]


DEVICE_PROFILES: dict[str, DeviceProfile] = {
    "EV": DeviceProfile(10.0, 5.0, frozenset({18, 19, 20, 21})),
    "HVAC": DeviceProfile(2.0, 1.0, frozenset({12, 13, 14, 15, 16, 17})),
    "appliance": DeviceProfile(1.5, 0.5, frozenset({19, 20, 21, 22})),
}
APPLIANCE_NAMES = ("Dishwasher", "Washing Machine", "Dryer")
FIXED_DEVICE_NAMES = {"EV": "Tesla Model 3", "HVAC": "Main AC Unit"}

PEAK_PRICE, OFF_PEAK_PRICE = 0.15, 0.10  # simplified historical price, USD/kWh

# Sunlight multiplier per sky condition and how often each condition occurs.
WEATHER = {
    "sunny": (1.0, 0.4),
    "partly_cloudy": (0.6, 0.3),
    "cloudy": (0.3, 0.2),
    "rainy": (0.1, 0.1),
}
DAYLIGHT_HOURS = range(6, 19)  # 06:00-18:00 inclusive
PV_PEAK_KWH = 5.0
PEAK_IRRADIANCE = 800.0


@dataclass(frozen=True)
class SeedResult:
    first_day: date
    last_day: date
    usage_rows: int
    generation_rows: int
    deleted: dict[str, int]


def generate_usage(days: list[date], rng: random.Random) -> list[EnergyUsage]:
    rows = []
    for day in days:
        for hour in range(24):
            ts = datetime.combine(day, datetime.min.time()).replace(hour=hour)
            for device_type, p in DEVICE_PROFILES.items():
                is_peak = hour in p.peak_hours
                consumption = max(
                    0.0, (p.base_kwh + rng.uniform(-p.variation, p.variation)) * (1.5 if is_peak else 0.8)
                )
                name = FIXED_DEVICE_NAMES.get(device_type) or rng.choice(APPLIANCE_NAMES)
                rows.append(
                    EnergyUsage(
                        timestamp=ts,
                        consumption_kwh=consumption,
                        device_type=device_type,
                        device_name=name,
                        cost_usd=consumption * (PEAK_PRICE if is_peak else OFF_PEAK_PRICE),
                    )
                )
    return rows


def generate_solar(days: list[date], rng: random.Random) -> list[SolarGeneration]:
    rows = []
    conditions = list(WEATHER)
    weights = [WEATHER[c][1] for c in conditions]
    for day in days:
        condition = rng.choices(conditions, weights=weights)[0]
        multiplier = WEATHER[condition][0]
        for hour in DAYLIGHT_HOURS:
            hour_factor = 1 - abs(hour - 12) / 6  # bell-like curve, peak at noon
            generation = max(0.0, PV_PEAK_KWH * hour_factor * multiplier * rng.uniform(0.8, 1.2))
            base_temp = 20 + rng.uniform(-5, 5)
            temp_factor = 1.0 if 15 <= base_temp <= 35 else 0.9
            rows.append(
                SolarGeneration(
                    timestamp=datetime.combine(day, datetime.min.time()).replace(hour=hour),
                    generation_kwh=generation,
                    weather_condition=condition,
                    temperature_c=base_temp * temp_factor,
                    solar_irradiance=PEAK_IRRADIANCE * hour_factor * multiplier if generation > 0 else 0.0,
                )
            )
    return rows


def seed_database(
    repo: EnergyRepository, days: int = 30, seed: int = 42, end_date: date | None = None
) -> SeedResult:
    """Create the tables and replace their content with ``days`` days of sample data."""
    if days < 1:
        raise ValueError("days must be >= 1")
    last_day = end_date or (date.today() - timedelta(days=1))
    day_list = [last_day - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
    rng = random.Random(seed)

    usage = generate_usage(day_list, rng)
    solar = generate_solar(day_list, rng)

    repo.create_tables()
    deleted = repo.clear()
    repo.add_all(usage)
    repo.add_all(solar)
    return SeedResult(day_list[0], last_day, len(usage), len(solar), deleted)
