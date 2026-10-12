"""Data for the app's planner view: the window of a chosen day and the rows for its chart and table."""

import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from energy_advisor.optimization.service import DevicePlan
from energy_advisor.timeutils import floor_hour, to_utc
from energy_advisor.ui.window import InvalidWindowError, SchedulingWindow, default_window

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
        return default_window(now)
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


def result_markdown(plan: DevicePlan, tz: ZoneInfo, feed_in_eur_per_kwh: float) -> str:
    """The result screen: per device the schedule table, the cost analysis and the solar contribution."""
    fmt = "%a %d %b %H:%M"
    start = plan.window.earliest.astimezone(tz)
    deadline = to_utc(plan.window.latest).astimezone(tz)
    parts = [f"**Window:** {start:{fmt}} - {deadline:{fmt}}"]
    if plan.window.curve.is_estimate:
        parts.append(
            "> ⚠️ Some prices in this window are **estimates**: day-ahead prices for the next day are published "
            "around 13:00."
        )
    for result in plan.results:
        device = result.device
        title = f"### {device.name}: {device.energy_kwh:g} kWh"
        if result.schedule is None:
            parts += [title, f"Not possible in this window: {result.error}"]
            continue
        s = result.schedule
        rows = [
            "| Time slot | Energy (kWh) | From solar (kWh) | From grid (kWh) | Cost (EUR) |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {b.start.astimezone(tz):%a %H:%M} - {b.end.astimezone(tz):%H:%M} | {b.energy_kwh:.2f} | "
                f"{b.solar_kwh:.2f} | {b.grid_kwh:.2f} | {b.cost_eur:.2f} |"
                for b in s.blocks
            ],
            f"| **Total** | **{s.energy_kwh:.2f}** | **{s.solar_kwh:.2f}** | **{s.grid_kwh:.2f}** | **{s.cost_eur:.2f}** |",
        ]
        analysis = [
            "**Cost analysis**",
            "",
            f"- Total cost: **{s.cost_eur:.2f} EUR** (average {s.average_price:.3f} EUR/kWh)",
        ]
        if math.isnan(s.baseline_cost_eur):
            analysis.append(f"- Starting immediately ({start:%a %H:%M}) is not possible within the window")
        else:
            analysis += [
                f"- Starting immediately ({start:%a %H:%M}) would cost {s.baseline_cost_eur:.2f} EUR",
                f"- **Saving: {s.savings_eur:.2f} EUR**",
            ]
        if s.solar_kwh > 0:
            solar = (
                f"**Solar:** {s.solar_kwh:.1f} kWh come from your PV system. They are valued at the feed-in tariff "
                f"({feed_in_eur_per_kwh:.3f} EUR/kWh) that you would otherwise receive for exporting them."
            )
        else:
            solar = (
                "**Solar:** no solar surplus is used: none is expected in these hours, or the devices planned "
                "before this one (largest first) already use it."
            )
        parts += [title, "\n".join(rows), "\n".join(analysis), solar]
    return "\n\n".join(parts)
