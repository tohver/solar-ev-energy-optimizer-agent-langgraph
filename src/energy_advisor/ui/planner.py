"""Data for the app's planner view: the window of a chosen day and the rows for its chart and table."""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from energy_advisor.optimization.service import DevicePlan
from energy_advisor.timeutils import floor_hour, to_utc
from energy_advisor.ui.window import InvalidWindowError, SchedulingWindow, default_deadline

MAX_DAYS_AHEAD = 6


def day_window(day: date, now: datetime) -> SchedulingWindow:
    """Today: from now for 24 hours. A later day: that whole day (midnight to midnight)."""
    tz = now.tzinfo
    if not isinstance(tz, ZoneInfo):
        raise ValueError("now must carry a ZoneInfo time zone")
    today = now.date()
    if day < today:
        raise InvalidWindowError("Choose today or a later day.")
    if day > today + timedelta(days=MAX_DAYS_AHEAD):
        raise InvalidWindowError(f"Choose a day within the next {MAX_DAYS_AHEAD} days.")
    if day == today:
        rounded = now.replace(second=0, microsecond=0)
        start = rounded + timedelta(minutes=(-rounded.minute) % 15)  # next quarter hour, like the optimizer
        return SchedulingWindow(start, default_deadline(start))
    start = datetime.combine(day, time.min, tzinfo=tz)
    return SchedulingWindow(start, datetime.combine(day + timedelta(days=1), time.min, tzinfo=tz))


def _local(value: datetime, tz: ZoneInfo) -> datetime:
    """Naive local time for charts (they cannot show time zones)."""
    return value.astimezone(tz).replace(tzinfo=None)


def hourly_rows(plan: DevicePlan, tz: ZoneInfo) -> list[dict[str, Any]]:
    """Per hour of the window: household price (mean of its slots) and solar surplus (kWh)."""
    price_parts: dict[datetime, list[tuple[float, float]]] = defaultdict(list)
    surplus: dict[datetime, float] = defaultdict(float)
    for s in plan.window.slots:
        hour = floor_hour(s.start)
        price_parts[hour].append((s.hours, s.grid_price))
        surplus[hour] += s.solar_kw * s.hours
    return [
        {
            "time": _local(hour, tz),
            "price": round(sum(h * p for h, p in parts) / sum(h for h, _ in parts), 4),
            "solar_kwh": round(surplus[hour], 2),
        }
        for hour, parts in sorted(price_parts.items())
    ]


def window_bars(plan: DevicePlan, tz: ZoneInfo) -> list[dict[str, Any]]:
    """One bar per run block of each planned device."""
    bars = []
    for result in plan.results:
        if result.schedule is None:
            continue
        for block in result.schedule.blocks:
            bars.append(
                {
                    "device": result.device.name,
                    "start": _local(block.start, tz),
                    "end": _local(block.end, tz),
                    "kwh": round(block.energy_kwh, 2),
                    "solar_kwh": round(block.solar_kwh, 2),
                    "cost_eur": round(block.cost_eur, 2),
                }
            )
    return bars


def summary_rows(plan: DevicePlan, tz: ZoneInfo) -> list[dict[str, Any]]:
    """One row per device for the result table (or the reason it could not be planned)."""
    fmt = "%a %H:%M"
    rows: list[dict[str, Any]] = []
    for result in plan.results:
        device = result.device
        if result.schedule is None:
            rows.append({"Device": device.name, "Best time": f"not possible: {result.error}"})
            continue
        schedule = result.schedule
        times = ", ".join(
            f"{b.start.astimezone(tz):{fmt}}-{b.end.astimezone(tz):%H:%M}" for b in schedule.blocks
        )
        rows.append(
            {
                "Device": device.name,
                "Best time": times,
                "Energy (kWh)": round(schedule.energy_kwh, 1),
                "From solar (kWh)": round(schedule.solar_kwh, 1),
                "Cost (EUR)": round(schedule.cost_eur, 2),
                "Saving vs. starting at once (EUR)": round(schedule.savings_eur, 2),
            }
        )
    return rows


def window_label(plan: DevicePlan, tz: ZoneInfo) -> str:
    w = plan.window
    return f"{w.earliest.astimezone(tz):%a %d %b %H:%M} - {to_utc(w.latest).astimezone(tz):%a %d %b %H:%M}"
