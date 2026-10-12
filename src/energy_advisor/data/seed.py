"""Reproducible history of the demo household in Munich.

The household has a PV system, an air-to-water heat pump, an EV with an 11 kW wallbox and the usual
appliances. Its history is driven by REAL inputs when available (past hourly weather from Open-Meteo,
past day-ahead prices) and by synthetic fallbacks otherwise:

- PV yield = installed kWp x irradiance / 1000 x performance ratio
- heat pump electricity = heat demand / COP, where heat demand grows with
  ``heating_limit - outdoor temperature`` and the COP (efficiency) falls in cold weather
- EV: daily driving drains the battery; it is charged at full wallbox power when the car comes home
  in the evening (the unoptimized habit the advisor should improve)
- appliances run as individual cycles (dishwasher, washing machine, dryer)
- base load follows a typical household day profile (morning and evening peaks)

The same seed and the same inputs always give the same data; re-seeding replaces the old data.
"""

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from energy_advisor.config import HouseholdSettings
from energy_advisor.data.models import EnergyUsage, SolarGeneration
from energy_advisor.data.repository import EnergyRepository
from energy_advisor.domain.solar import pv_energy_kwh
from energy_advisor.domain.weather import HourlyWeather
from energy_advisor.services.prices import PriceService
from energy_advisor.services.weather import WeatherService
from energy_advisor.timeutils import day_bounds_utc

MAX_DAYS = 90  # Open-Meteo's forecast API serves about 3 months of past weather

# Relative base load by local hour (mean ~1.0): low at night, peaks in the morning and evening.
BASE_LOAD_PROFILE = (
    0.6, 0.55, 0.5, 0.5, 0.5, 0.6, 0.9, 1.2, 1.1, 0.95, 0.9, 1.0,
    1.15, 1.0, 0.9, 0.9, 1.0, 1.3, 1.6, 1.7, 1.5, 1.3, 1.0, 0.8,
)  # fmt: skip
HOT_WATER_HOURS = {6: 0.8, 7: 0.6, 19: 0.6, 20: 0.5}  # kWh of heat for showers etc.


@dataclass(frozen=True)
class Appliance:
    name: str
    probability: float  # chance of one cycle per day
    start_hours: tuple[int, ...]
    kwh_per_hour: tuple[float, ...]  # energy in each hour of the cycle


DISHWASHER = Appliance("Dishwasher", 0.75, (19, 20, 21), (0.6, 0.4))
WASHING_MACHINE = Appliance("Washing machine", 0.5, tuple(range(8, 19)), (0.5, 0.3))
DRYER = Appliance("Heat-pump dryer", 0.6, (), (0.6, 0.6))  # runs 2 h after the washing machine
EV_NAME = "EV (11 kW wallbox)"
HEAT_PUMP_NAME = "Air-to-water heat pump"
BASE_LOAD_NAME = "Household base load"
EV_CHARGE_THRESHOLD_KWH = 15.0


@dataclass(frozen=True)
class History:
    usage: list[EnergyUsage]
    solar: list[SolarGeneration]
    first_day: date
    last_day: date
    weather_source: str
    price_source: str


@dataclass(frozen=True)
class SeedResult:
    first_day: date
    last_day: date
    usage_rows: int
    generation_rows: int
    deleted: dict[str, int]
    weather_source: str
    price_source: str


def heat_pump_cop(temperature_c: float) -> float:
    """Coefficient of performance: units of heat per unit of electricity; lower when it is cold."""
    return max(2.0, min(4.5, 3.0 + 0.1 * temperature_c))


def generate_household(
    weather: Sequence[HourlyWeather],
    price_of: Mapping[datetime, float],
    household: HouseholdSettings,
    tz: ZoneInfo,
    rng: random.Random,
) -> tuple[list[EnergyUsage], list[SolarGeneration]]:
    """Hourly consumption per device and PV generation for the given weather hours."""
    by_time = {h.time: h for h in weather}
    loads: dict[tuple[datetime, str, str], float] = defaultdict(float)
    solar: list[SolarGeneration] = []

    days: dict[date, list[HourlyWeather]] = defaultdict(list)
    for h in weather:
        days[h.time.astimezone(tz).date()].append(h)

    ev_deficit = rng.uniform(5, EV_CHARGE_THRESHOLD_KWH)
    for _day, hours in sorted(days.items()):
        hour_at = {h.time.astimezone(tz).hour: h.time for h in hours}  # local hour -> UTC start
        for h in hours:
            local_hour = h.time.astimezone(tz).hour
            # base load
            base = household.base_load_kw * BASE_LOAD_PROFILE[local_hour] * rng.uniform(0.85, 1.15)
            loads[(h.time, "base_load", BASE_LOAD_NAME)] += base
            # heat pump: space heating + hot water
            heat = household.heat_loss_kw_per_k * max(0.0, household.heating_limit_c - h.temperature_c)
            heat += HOT_WATER_HOURS.get(local_hour, 0.0)
            if heat > 0:
                loads[(h.time, "heat_pump", HEAT_PUMP_NAME)] += heat / heat_pump_cop(h.temperature_c)
            # PV
            if h.irradiance_w_m2 > 0:
                solar.append(
                    SolarGeneration(
                        timestamp=h.time,
                        generation_kwh=round(
                            pv_energy_kwh(
                                h.irradiance_w_m2, household.pv_peak_kw, household.pv_performance_ratio
                            ),
                            4,
                        ),
                        weather_condition=h.condition,
                        temperature_c=h.temperature_c,
                        solar_irradiance=h.irradiance_w_m2,
                    )
                )

        # EV: driving drains the battery; charge in the evening once enough is missing
        ev_deficit += (
            household.ev_daily_km * rng.uniform(0.5, 1.5) * household.ev_consumption_kwh_per_100km / 100
        )
        if ev_deficit >= EV_CHARGE_THRESHOLD_KWH and rng.random() < 0.85:
            arrival = rng.choice((17, 18, 19))
            remaining, hour = ev_deficit, arrival
            while remaining > 0.01 and hour in hour_at:
                energy = min(household.wallbox_kw, remaining)
                loads[(hour_at[hour], "EV", EV_NAME)] += energy
                remaining -= energy
                hour += 1
            ev_deficit = remaining

        # appliance cycles
        for appliance in (DISHWASHER, WASHING_MACHINE):
            if rng.random() >= appliance.probability:
                continue
            start = rng.choice(appliance.start_hours)
            _run_cycle(loads, hour_at, appliance, start, rng)
            if appliance is WASHING_MACHINE and rng.random() < DRYER.probability:
                _run_cycle(loads, hour_at, DRYER, start + 2, rng)

    usage = []
    for (ts, device_type, name), kwh in sorted(loads.items()):
        if ts not in by_time or kwh < 0.005:
            continue
        price = price_of[ts]
        usage.append(
            EnergyUsage(
                timestamp=ts,
                consumption_kwh=round(kwh, 4),
                device_type=device_type,
                device_name=name,
                price_eur_per_kwh=price,
                cost_eur=round(kwh * price, 4),
            )
        )
    return usage, solar


def _run_cycle(
    loads: dict[tuple[datetime, str, str], float],
    hour_at: Mapping[int, datetime],
    appliance: Appliance,
    start_hour: int,
    rng: random.Random,
) -> None:
    for offset, kwh in enumerate(appliance.kwh_per_hour):
        if start_hour + offset in hour_at:
            loads[(hour_at[start_hour + offset], "appliance", appliance.name)] += kwh * rng.uniform(0.9, 1.1)


def build_history(
    weather: WeatherService,
    prices: PriceService,
    household: HouseholdSettings,
    tz: ZoneInfo,
    days: int,
    last_day: date,
    seed: int,
) -> History:
    """Fetch past weather and prices (or their fallbacks) and generate the household's history."""
    if not 1 <= days <= MAX_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_DAYS}")
    first_day = last_day - timedelta(days=days - 1)
    series = weather.history(first_day, last_day)
    start, end = day_bounds_utc(first_day, last_day, tz)
    price_of, price_source = prices.hourly_retail(start, end)
    missing = [h.time for h in series.hours if h.time not in price_of]
    if missing:
        raise ValueError(f"Prices missing for {len(missing)} hours, e.g. {missing[0]}")
    usage, solar = generate_household(series.hours, price_of, household, tz, random.Random(seed))
    return History(usage, solar, first_day, last_day, series.source, price_source)


def seed_database(repo: EnergyRepository, history: History) -> SeedResult:
    """Create the tables and replace their content with ``history`` in one go."""
    deleted = repo.recreate_tables()
    repo.add_all(history.usage)
    repo.add_all(history.solar)
    return SeedResult(
        history.first_day,
        history.last_day,
        len(history.usage),
        len(history.solar),
        deleted,
        history.weather_source,
        history.price_source,
    )
