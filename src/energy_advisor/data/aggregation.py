"""Turn raw hourly rows into compact summaries the LLM can read.

Returning thousands of raw rows overflowed the model's context window (regression F-04), so the
tools return aggregates and include raw rows only for small result sets.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from energy_advisor.data.models import EnergyUsage, SolarGeneration

RAW_RECORDS_LIMIT = 40
RAW_RECORDS_NOTE = "Raw records omitted (too many); use the aggregated fields or query a shorter range."


def _round_group(groups: dict[str, dict[str, float]]) -> None:
    for g in groups.values():
        for key in ("consumption_kwh", "cost_usd", "generation_kwh"):
            if key in g:
                g[key] = round(g[key], 2)


def summarize_usage(records: Sequence[EnergyUsage]) -> dict[str, Any]:
    """Totals per device type, per device name and per day, plus the average kWh by hour of day."""
    by_type: dict[str, dict[str, float]] = {}
    by_name: dict[str, dict[str, float]] = {}
    daily: dict[str, dict[str, float]] = {}
    hourly: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        cost = r.cost_usd or 0.0
        for key, group in ((r.device_type or "unknown", by_type), (r.device_name or "unknown", by_name)):
            g = group.setdefault(key, {"consumption_kwh": 0.0, "cost_usd": 0.0, "records": 0})
            g["consumption_kwh"] += r.consumption_kwh
            g["cost_usd"] += cost
            g["records"] += 1
        day = daily.setdefault(r.timestamp.strftime("%Y-%m-%d"), {"consumption_kwh": 0.0, "cost_usd": 0.0})
        day["consumption_kwh"] += r.consumption_kwh
        day["cost_usd"] += cost
        hourly[r.device_type or "unknown"][r.timestamp.hour].append(r.consumption_kwh)

    for group in (by_type, by_name, daily):
        _round_group(group)

    summary: dict[str, Any] = {
        "total_records": len(records),
        "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
        "total_cost_usd": round(sum(r.cost_usd or 0.0 for r in records), 2),
        "by_device_type": by_type,
        "by_device_name": by_name,
        "daily_totals": daily,
        "avg_kwh_by_hour": {
            device: {hour: round(sum(v) / len(v), 2) for hour, v in sorted(hours.items())}
            for device, hours in hourly.items()
        },
    }
    if len(records) <= RAW_RECORDS_LIMIT:
        summary["records"] = [
            {
                "timestamp": r.timestamp.isoformat(),
                "consumption_kwh": r.consumption_kwh,
                "device_type": r.device_type,
                "device_name": r.device_name,
                "cost_usd": r.cost_usd,
            }
            for r in records
        ]
    else:
        summary["note"] = RAW_RECORDS_NOTE
    return summary


def summarize_generation(records: Sequence[SolarGeneration], days_in_range: int) -> dict[str, Any]:
    """Totals per day and per weather condition, plus the average kWh by hour of day."""
    daily: dict[str, dict[str, Any]] = {}
    by_weather: dict[str, dict[str, float]] = {}
    by_hour: dict[int, list[float]] = defaultdict(list)

    for r in records:
        day = daily.setdefault(
            r.timestamp.strftime("%Y-%m-%d"),
            {"generation_kwh": 0.0, "weather_condition": r.weather_condition},
        )
        day["generation_kwh"] += r.generation_kwh
        by_hour[r.timestamp.hour].append(r.generation_kwh)
        w = by_weather.setdefault(r.weather_condition or "unknown", {"generation_kwh": 0.0, "records": 0})
        w["generation_kwh"] += r.generation_kwh
        w["records"] += 1

    _round_group(daily)
    _round_group(by_weather)
    total = sum(r.generation_kwh for r in records)

    summary: dict[str, Any] = {
        "total_records": len(records),
        "total_generation_kwh": round(total, 2),
        "average_daily_generation": round(total / max(1, days_in_range), 2),
        "daily_totals": daily,
        "avg_kwh_by_hour": {h: round(sum(v) / len(v), 2) for h, v in sorted(by_hour.items())},
        "by_weather_condition": by_weather,
    }
    if len(records) <= RAW_RECORDS_LIMIT:
        summary["records"] = [
            {
                "timestamp": r.timestamp.isoformat(),
                "generation_kwh": r.generation_kwh,
                "weather_condition": r.weather_condition,
                "temperature_c": r.temperature_c,
                "solar_irradiance": r.solar_irradiance,
            }
            for r in records
        ]
    else:
        summary["note"] = RAW_RECORDS_NOTE
    return summary


def summarize_recent(
    usage: Sequence[EnergyUsage], generation: Sequence[SolarGeneration], hours: int
) -> dict[str, Any]:
    """Short overview of the last ``hours`` hours: consumption per device and total solar."""
    breakdown: dict[str, dict[str, float]] = {}
    for r in usage:
        d = breakdown.setdefault(
            r.device_type or "unknown", {"consumption_kwh": 0.0, "cost_usd": 0.0, "records": 0}
        )
        d["consumption_kwh"] += r.consumption_kwh
        d["cost_usd"] += r.cost_usd or 0.0
        d["records"] += 1
    _round_group(breakdown)

    conditions = Counter(r.weather_condition for r in generation if r.weather_condition)
    return {
        "time_period_hours": hours,
        "usage": {
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0.0 for r in usage), 2),
            "device_breakdown": breakdown,
        },
        "generation": {
            "total_generation_kwh": round(sum(r.generation_kwh for r in generation), 2),
            "average_weather": conditions.most_common(1)[0][0] if conditions else "unknown",
        },
    }
