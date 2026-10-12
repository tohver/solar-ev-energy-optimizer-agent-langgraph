"""Glue between the optimizer and the data: price curve + solar forecast → schedule → tool output."""

import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from energy_advisor.config import Settings
from energy_advisor.domain.household import base_load_kw
from energy_advisor.domain.solar import pv_energy_kwh
from energy_advisor.optimization.scheduling import (
    Schedule,
    ScheduleError,
    Slot,
    clip_slots,
    find_cheapest_schedule,
)
from energy_advisor.services.prices import PriceCurve, PriceService
from energy_advisor.services.weather import WeatherService
from energy_advisor.timeutils import Clock, floor_hour, parse_local_datetime, to_utc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceSpec:
    """A flexible load the planner can schedule."""

    key: str
    name: str
    energy_kwh: float
    power_kw: float
    interruptible: bool  # True: may pause and modulate (EV); False: runs in one go (appliances)


# The MVP device set of the app; energies and powers are typical values (see guide.md, section 3).
DEVICE_PRESETS: dict[str, DeviceSpec] = {
    "ev": DeviceSpec("ev", "Electric car", 20.0, 11.0, True),
    "dishwasher": DeviceSpec("dishwasher", "Dishwasher", 1.0, 0.5, False),
    "washing_machine": DeviceSpec("washing_machine", "Washing machine", 0.8, 0.4, False),
}


@dataclass(frozen=True)
class WindowInputs:
    earliest: datetime
    latest: datetime
    slots: list[Slot]  # clipped to the window; household price and solar surplus per slot
    curve: PriceCurve
    solar_source: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DeviceResult:
    device: DeviceSpec
    schedule: Schedule | None
    error: str | None


@dataclass(frozen=True)
class DevicePlan:
    window: WindowInputs
    results: list[DeviceResult]
    solar_left_kwh: float


MAX_WINDOW = timedelta(days=7)
DEFAULT_WINDOW_HOURS = 24
DEFAULT_WINDOW = timedelta(hours=DEFAULT_WINDOW_HOURS)
QUARTER = timedelta(minutes=15)


class OptimizerService:
    def __init__(
        self, settings: Settings, prices: PriceService, weather: WeatherService, clock: Clock
    ) -> None:
        self._settings = settings
        self._prices = prices
        self._weather = weather
        self._clock = clock
        self._tz = settings.location.tz

    # ------------------------------------------------------------------ solar
    def solar_surplus(
        self, first_day: date, last_day: date
    ) -> tuple[dict[datetime, tuple[float, float]], str]:
        """Per UTC hour: (PV kWh, surplus kWh after the household base load), plus the forecast source."""
        h = self._settings.household
        series = self._weather.series(first_day, last_day)
        result = {}
        for w in series.hours:
            pv = pv_energy_kwh(w.irradiance_w_m2, h.pv_peak_kw, h.pv_performance_ratio)
            surplus = max(0.0, pv - base_load_kw(w.time.astimezone(self._tz).hour, h.base_load_kw))
            result[w.time] = (pv, surplus)
        return result, series.source

    def pv_forecast(self, days: int = 1) -> dict[str, Any]:
        """Tool output: expected PV yield per day and hour, and the hours with the most surplus."""
        days = max(1, min(int(days), 3))
        today = self._clock().astimezone(self._tz).date()
        surplus, source = self.solar_surplus(today, today + timedelta(days=days - 1))
        per_day: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        for t, (pv, extra) in sorted(surplus.items()):
            local = t.astimezone(self._tz)
            per_day[local.date().isoformat()].append((local.hour, pv, extra))
        h = self._settings.household
        return {
            "pv_system_kwp": h.pv_peak_kw,
            "forecast_source": source,
            "days": {
                day: {
                    "pv_kwh": round(sum(pv for _, pv, _ in rows), 1),
                    "surplus_kwh": round(sum(s for _, _, s in rows), 1),
                    "best_hours": [
                        f"{hr:02d}:00" for hr, _, extra in sorted(rows, key=lambda r: -r[2])[:4] if extra > 0
                    ],
                    "hourly": [
                        {"hour": hr, "pv_kwh": round(pv, 2), "surplus_kwh": round(s, 2)}
                        for hr, pv, s in rows
                        if pv > 0
                    ],
                }
                for day, rows in per_day.items()
            },
            "assumptions": (
                f"PV yield = {h.pv_peak_kw} kWp x irradiance/1000 x {h.pv_performance_ratio}; surplus = yield minus "
                f"a typical household base load (average {h.base_load_kw} kW)."
            ),
        }

    # ------------------------------------------------------------------ scheduling
    def window(
        self, start: datetime | None, deadline: datetime | None, use_solar: bool = True
    ) -> WindowInputs:
        """Everything a schedule needs for ``[start, deadline)``: the slots with household price and solar
        surplus, plus sources and notes. ``start`` defaults to now (a past start becomes now) and is rounded up
        to the next quarter hour; ``deadline`` defaults to start + 24 hours (noted in ``notes``)."""
        notes: list[str] = []
        now = to_utc(self._clock())
        begin = to_utc(start) if start else now
        if begin < now:
            notes.append(
                f"The start {begin.astimezone(self._tz):%Y-%m-%d %H:%M} is in the past; planning from now."
            )
            begin = now
        earliest = _ceil_quarter(begin)
        if deadline:
            latest = to_utc(deadline)
        else:
            latest = earliest + DEFAULT_WINDOW
            notes.append(
                f"No deadline given: assumed {DEFAULT_WINDOW_HOURS} hours from the start "
                f"({latest.astimezone(self._tz):%Y-%m-%d %H:%M}). Ask the user whether that is right."
            )
        if latest - earliest > MAX_WINDOW:
            raise ScheduleError("The window can be at most 7 days long.")
        if latest <= earliest:
            raise ScheduleError("The deadline must be after the start (and in the future).")

        curve = self._prices.curve(floor_hour(earliest), _ceil_hour(latest))
        surplus: dict[datetime, tuple[float, float]] = {}
        solar_source = "not used"
        if use_solar:
            try:
                surplus, solar_source = self.solar_surplus(
                    earliest.astimezone(self._tz).date(), latest.astimezone(self._tz).date()
                )
            except Exception as exc:
                logger.warning("Solar forecast unavailable: %s", exc)
                notes.append("Solar forecast unavailable; scheduled on grid prices only.")
        slots = clip_slots(
            [
                Slot(s.start, s.end, s.price_eur_per_kwh, surplus.get(floor_hour(s.start), (0.0, 0.0))[1])
                for s in curve.slots
            ],
            earliest,
            latest,
        )
        return WindowInputs(earliest, latest, slots, curve, solar_source, notes + curve.notes)

    def cheapest_window(
        self,
        energy_kwh: float,
        power_kw: float,
        start: str | None = None,
        deadline: str | None = None,
        interruptible: bool = True,
        use_solar: bool = True,
        device: str = "device",
    ) -> dict[str, Any]:
        """Tool backend: cheapest schedule between ``start`` (default: now) and ``deadline`` (default: start + 24 h)."""
        w = self.window(
            parse_local_datetime(start, self._tz) if start else None,
            parse_local_datetime(deadline, self._tz) if deadline else None,
            use_solar,
        )
        schedule = find_cheapest_schedule(
            w.slots,
            energy_kwh,
            power_kw,
            w.earliest,
            w.latest,
            self._settings.tariff.feed_in_tariff_eur_per_kwh,
            interruptible=interruptible,
            use_solar=use_solar,
        )
        return self._output(
            schedule,
            device,
            energy_kwh,
            power_kw,
            interruptible,
            w.earliest,
            w.latest,
            w.curve,
            w.solar_source,
            w.notes,
        )

    def plan_devices(
        self, devices: Sequence[DeviceSpec], start: datetime | None = None, deadline: datetime | None = None
    ) -> DevicePlan:
        """Schedule several devices in the same window, sharing the solar surplus.

        Devices are planned one after another, largest energy first; the solar energy a device takes is
        removed before the next one is planned, so the same surplus is never counted twice.
        """
        w = self.window(start, deadline)
        feed_in = self._settings.tariff.feed_in_tariff_eur_per_kwh
        remaining = {s.start: s.solar_kw * s.hours for s in w.slots}
        results: dict[str, DeviceResult] = {}
        for device in sorted(devices, key=lambda d: -d.energy_kwh):
            slots = [Slot(s.start, s.end, s.grid_price, remaining[s.start] / s.hours) for s in w.slots]
            try:
                schedule = find_cheapest_schedule(
                    slots,
                    device.energy_kwh,
                    device.power_kw,
                    w.earliest,
                    w.latest,
                    feed_in,
                    device.interruptible,
                )
            except ScheduleError as exc:
                results[device.key] = DeviceResult(device, None, str(exc))
                continue
            for slot_start, used in schedule.solar_by_slot.items():
                remaining[slot_start] = max(0.0, remaining[slot_start] - used)
            results[device.key] = DeviceResult(device, schedule, None)
        return DevicePlan(
            window=w,
            results=[results[d.key] for d in devices],  # in the user's order
            solar_left_kwh=sum(remaining.values()),
        )

    def _output(
        self,
        schedule: Schedule,
        device: str,
        energy_kwh: float,
        power_kw: float,
        interruptible: bool,
        earliest: datetime,
        latest: datetime,
        curve: PriceCurve,
        solar_source: str,
        notes: list[str],
    ) -> dict[str, Any]:
        def local(t: datetime) -> str:
            return t.astimezone(self._tz).strftime("%Y-%m-%d %H:%M")

        return {
            "device": device,
            "energy_kwh": energy_kwh,
            "power_kw": power_kw,
            "mode": "interruptible (may pause)" if interruptible else "contiguous (runs without pause)",
            "window": {
                "start": local(earliest),
                "deadline": local(latest),
                "timezone": str(self._tz),
            },
            "recommended_schedule": [
                {
                    "start": local(b.start),
                    "end": local(b.end),
                    "energy_kwh": round(b.energy_kwh, 2),
                    "average_power_kw": round(b.average_kw, 1),
                    "from_solar_kwh": round(b.solar_kwh, 2),
                    "from_grid_kwh": round(b.grid_kwh, 2),
                    "cost_eur": round(b.cost_eur, 2),
                }
                for b in schedule.blocks
            ],
            "finishes_at": local(max(b.end for b in schedule.blocks)),
            "total_cost_eur": round(schedule.cost_eur, 2),
            "average_price_eur_per_kwh": round(schedule.average_price, 4),
            "solar_kwh": round(schedule.solar_kwh, 2),
            "grid_kwh": round(schedule.grid_kwh, 2),
            "baseline_start_immediately": {
                "start": local(earliest),
                "cost_eur": None
                if math.isnan(schedule.baseline_cost_eur)
                else round(schedule.baseline_cost_eur, 2),
            },
            "savings_vs_baseline_eur": None
            if math.isnan(schedule.baseline_cost_eur)
            else round(schedule.savings_eur, 2),
            "prices": {"source": curve.source, "is_estimate": curve.is_estimate},
            "solar_forecast_source": solar_source,
            "assumptions": (
                "Own solar power is valued at the feed-in tariff "
                f"({self._settings.tariff.feed_in_tariff_eur_per_kwh} EUR/kWh) because using it means not exporting it."
            ),
            **({"notes": notes} if notes else {}),
        }


def _ceil_quarter(t: datetime) -> datetime:
    floored = t.replace(minute=t.minute - t.minute % 15, second=0, microsecond=0)
    return floored if floored == t else floored + QUARTER


def _ceil_hour(t: datetime) -> datetime:
    floored = floor_hour(t)
    return floored if floored == t else floored + timedelta(hours=1)
