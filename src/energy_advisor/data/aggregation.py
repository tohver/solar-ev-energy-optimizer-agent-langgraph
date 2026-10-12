"""Turn raw hourly rows into compact summaries the LLM can read.

Returning thousands of raw rows overflowed the model's context window (regression F-04), so the
tools return aggregates and include raw rows only for small result sets. Days and hours are local time.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from energy_advisor.data.models import EnergyUsage, SolarGeneration

RAW_RECORDS_LIMIT = 40
RAW_RECORDS_NOTE = "Raw records omitted (too many); use the aggregated fields or query a shorter range."


def _round_groups(groups: dict[str, dict[str, float]]) -> None:
    for g in groups.values():
        for key in ("consumption_kwh", "cost_eur", "generation_kwh"):
            if key in g:
                g[key] = round(g[key], 2)


def _local(ts: datetime, tz: ZoneInfo) -> datetime:
    return ts.astimezone(tz)


def summarize_usage(records: Sequence[EnergyUsage], tz: ZoneInfo, days_in_range: int) -> dict[str, Any]:
    """Totals per device type, per device and per day; average kWh per day; average kWh by hour of day."""
    by_type: dict[str, dict[str, float]] = {}
    by_name: dict[str, dict[str, float]] = {}
    daily: dict[str, dict[str, float]] = {}
    hourly: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for r in records:
        local = _local(r.timestamp, tz)
        cost = r.cost_eur or 0.0
        for key, group in ((r.device_type or "unknown", by_type), (r.device_name or "unknown", by_name)):
            g = group.setdefault(key, {"consumption_kwh": 0.0, "cost_eur": 0.0, "hours_with_use": 0})
            g["consumption_kwh"] += r.consumption_kwh
            g["cost_eur"] += cost
            g["hours_with_use"] += 1
        day = daily.setdefault(local.date().isoformat(), {"consumption_kwh": 0.0, "cost_eur": 0.0})
        day["consumption_kwh"] += r.consumption_kwh
        day["cost_eur"] += cost
        hourly[r.device_type or "unknown"][local.hour] += r.consumption_kwh

    for group in (by_type, by_name, daily):
        _round_groups(group)
    days = max(1, days_in_range)
    for g in (*by_type.values(), *by_name.values()):
        g["avg_kwh_per_day"] = round(g["consumption_kwh"] / days, 2)
        g["avg_price_eur_per_kwh"] = (
            round(g["cost_eur"] / g["consumption_kwh"], 4) if g["consumption_kwh"] else 0.0
        )

    total_kwh = sum(r.consumption_kwh for r in records)
    total_cost = sum(r.cost_eur or 0.0 for r in records)
    summary: dict[str, Any] = {
        "total_records": len(records),
        "days": days_in_range,
        "total_consumption_kwh": round(total_kwh, 2),
        "total_cost_eur": round(total_cost, 2),
        "average_price_paid_eur_per_kwh": round(total_cost / total_kwh, 4) if total_kwh else None,
        "by_device_type": by_type,
        "by_device_name": by_name,
        "daily_totals": daily,
        # average kWh in each local hour of the day, per device type (over the whole range)
        "avg_kwh_by_hour": {
            device: {hour: round(v / days, 3) for hour, v in sorted(hours.items())}
            for device, hours in hourly.items()
        },
        "note_on_cost": "Costs are consumption x hourly retail price, before subtracting own solar power.",
    }
    if len(records) <= RAW_RECORDS_LIMIT:
        summary["records"] = [
            {
                "time": _local(r.timestamp, tz).isoformat(timespec="minutes"),
                "consumption_kwh": r.consumption_kwh,
                "device_type": r.device_type,
                "device_name": r.device_name,
                "price_eur_per_kwh": r.price_eur_per_kwh,
                "cost_eur": r.cost_eur,
            }
            for r in records
        ]
    else:
        summary["note"] = RAW_RECORDS_NOTE
    return summary


def summarize_generation(
    records: Sequence[SolarGeneration], tz: ZoneInfo, days_in_range: int
) -> dict[str, Any]:
    """Totals per day and per weather condition, plus the average kWh by hour of day."""
    daily: dict[str, dict[str, Any]] = {}
    by_weather: dict[str, dict[str, float]] = {}
    by_hour: dict[int, float] = defaultdict(float)
    conditions: dict[str, Counter[str]] = defaultdict(Counter)

    for r in records:
        local = _local(r.timestamp, tz)
        day = daily.setdefault(local.date().isoformat(), {"generation_kwh": 0.0})
        day["generation_kwh"] += r.generation_kwh
        conditions[local.date().isoformat()][r.weather_condition or "unknown"] += 1
        by_hour[local.hour] += r.generation_kwh
        w = by_weather.setdefault(r.weather_condition or "unknown", {"generation_kwh": 0.0, "hours": 0})
        w["generation_kwh"] += r.generation_kwh
        w["hours"] += 1

    for d, values in daily.items():
        values["generation_kwh"] = round(values["generation_kwh"], 2)
        values["weather_condition"] = conditions[d].most_common(1)[0][0]
    _round_groups(by_weather)
    total = sum(r.generation_kwh for r in records)
    days = max(1, days_in_range)

    summary: dict[str, Any] = {
        "total_records": len(records),
        "total_generation_kwh": round(total, 2),
        "average_daily_generation_kwh": round(total / days, 2),
        "daily_totals": daily,
        "avg_kwh_by_hour": {h: round(v / days, 3) for h, v in sorted(by_hour.items())},
        "by_weather_condition": by_weather,
    }
    if len(records) <= RAW_RECORDS_LIMIT:
        summary["records"] = [
            {
                "time": _local(r.timestamp, tz).isoformat(timespec="minutes"),
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


def energy_balance(
    usage: Sequence[EnergyUsage], generation: Sequence[SolarGeneration], feed_in_eur_per_kwh: float
) -> dict[str, Any]:
    """Hour by hour: how much solar was used at home vs. exported, and how much came from the grid.

    Self-consumption rate = self-used solar / solar generated; autarky = self-used solar / consumption.
    """
    load: dict[datetime, float] = defaultdict(float)
    price: dict[datetime, list[tuple[float, float]]] = defaultdict(list)
    for r in usage:
        load[r.timestamp] += r.consumption_kwh
        if r.price_eur_per_kwh is not None:
            price[r.timestamp].append((r.consumption_kwh, r.price_eur_per_kwh))
    pv: dict[datetime, float] = defaultdict(float)
    for g in generation:
        pv[g.timestamp] += g.generation_kwh

    self_used = exported = imported = import_cost = self_value = 0.0
    for hour in set(load) | set(pv):
        used = min(load[hour], pv[hour])
        self_used += used
        exported += pv[hour] - used
        grid = load[hour] - used
        imported += grid
        weights = price.get(hour) or []
        hour_price = sum(k * p for k, p in weights) / sum(k for k, _ in weights) if weights else 0.0
        import_cost += grid * hour_price
        self_value += used * hour_price

    generated, consumed = sum(pv.values()), sum(load.values())
    return {
        "solar_generated_kwh": round(generated, 2),
        "consumption_kwh": round(consumed, 2),
        "solar_self_consumed_kwh": round(self_used, 2),
        "solar_exported_kwh": round(exported, 2),
        "grid_import_kwh": round(imported, 2),
        "self_consumption_rate": round(self_used / generated, 3) if generated else None,
        "autarky_rate": round(self_used / consumed, 3) if consumed else None,
        "grid_import_cost_eur": round(import_cost, 2),
        "savings_from_self_consumption_eur": round(self_value, 2),
        "feed_in_revenue_eur": round(exported * feed_in_eur_per_kwh, 2),
    }


def summarize_recent(
    usage: Sequence[EnergyUsage], generation: Sequence[SolarGeneration], hours: int
) -> dict[str, Any]:
    """Short overview of the last ``hours`` hours: consumption per device and total solar."""
    breakdown: dict[str, dict[str, float]] = {}
    for r in usage:
        d = breakdown.setdefault(r.device_type or "unknown", {"consumption_kwh": 0.0, "cost_eur": 0.0})
        d["consumption_kwh"] += r.consumption_kwh
        d["cost_eur"] += r.cost_eur or 0.0
    _round_groups(breakdown)

    conditions = Counter(r.weather_condition for r in generation if r.weather_condition)
    return {
        "time_period_hours": hours,
        "usage": {
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage), 2),
            "total_cost_eur": round(sum(r.cost_eur or 0.0 for r in usage), 2),
            "device_breakdown": breakdown,
        },
        "generation": {
            "total_generation_kwh": round(sum(r.generation_kwh for r in generation), 2),
            "most_common_weather": conditions.most_common(1)[0][0] if conditions else "unknown",
        },
    }
