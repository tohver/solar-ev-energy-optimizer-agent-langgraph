"""Electricity prices for the household: provider chain → coverage check → tariff → tool output."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from energy_advisor.clients.prices.base import PriceProvider
from energy_advisor.config import LocationSettings, TariffSettings
from energy_advisor.domain.prices import PriceSlot, PriceUnavailableError, SpotPrice
from energy_advisor.services.tariff import retail_price, tariff_assumptions
from energy_advisor.timeutils import Clock, day_bounds_utc, floor_hour, parse_day

logger = logging.getLogger(__name__)
MIN_COVERAGE = 0.99
PERIODS = ("off_peak", "mid_peak", "peak")
SUMMARY_HOURS = 4


@dataclass(frozen=True)
class PriceCurve:
    start: datetime
    end: datetime
    slots: list[PriceSlot]
    source: str
    is_estimate: bool
    attribution: str | None = None
    notes: list[str] = field(default_factory=list)

    def hourly(self) -> list[tuple[datetime, float, float]]:
        """(UTC hour start, mean spot EUR/MWh, mean retail EUR/kWh) per hour."""
        groups: dict[datetime, list[PriceSlot]] = defaultdict(list)
        for s in self.slots:
            groups[floor_hour(s.start)].append(s)
        return [
            (
                hour,
                round(mean(s.spot_eur_per_mwh for s in g), 2),
                round(mean(s.price_eur_per_kwh for s in g), 4),
            )
            for hour, g in sorted(groups.items())
        ]


def coverage(prices: Sequence[SpotPrice], start: datetime, end: datetime) -> float:
    covered = sum(((min(p.end, end) - max(p.start, start)) for p in prices), timedelta())
    return covered / (end - start)


class PriceService:
    def __init__(
        self,
        providers: Sequence[PriceProvider],
        tariff: TariffSettings,
        location: LocationSettings,
        clock: Clock,
    ) -> None:
        if not providers:
            raise ValueError("At least one price provider is required")
        self._providers = list(providers)
        self._tariff = tariff
        self._location = location
        self._tz: ZoneInfo = location.tz
        self._clock = clock

    # ------------------------------------------------------------------ core
    def spot_prices(
        self, start: datetime, end: datetime, providers: Sequence[PriceProvider] | None = None
    ) -> tuple[list[SpotPrice], PriceProvider, list[str]]:
        """First provider in the chain that covers ``[start, end)`` completely."""
        notes = []
        for provider in providers or self._providers:
            try:
                prices = [p for p in provider.fetch(start, end) if start <= p.start < end]
                if coverage(prices, start, end) < MIN_COVERAGE:
                    raise PriceUnavailableError("incomplete data")
                return sorted(prices, key=lambda p: p.start), provider, notes
            except Exception as exc:
                logger.info("Price provider %s failed: %s", provider.name, exc)
                notes.append(f"{provider.name}: {type(exc).__name__}: {exc}")
        raise PriceUnavailableError("No price provider could deliver prices. " + " | ".join(notes))

    def curve(self, start: datetime, end: datetime) -> PriceCurve:
        """Prices for ``[start, end)``. A multi-day range is first requested in one call from the real
        providers; if that fails (e.g. tomorrow is not published yet), each local day is resolved on its
        own, so today keeps real prices and only tomorrow is estimated."""
        days = sorted({h.astimezone(self._tz).date() for h in (start, end - timedelta(seconds=1))})
        if days[0] == days[-1]:
            return self._single_curve(start, end)
        real = [p for p in self._providers if not p.is_estimate]
        if real:
            try:
                return self._single_curve(start, end, real)
            except PriceUnavailableError:
                logger.info("Range not fully published; resolving day by day")
        parts = []
        day = days[0]
        while day <= days[-1]:
            day_start, day_end = day_bounds_utc(day, day, self._tz)
            parts.append(self._single_curve(max(start, day_start), min(end, day_end)))
            day += timedelta(days=1)
        return merge_curves(parts)

    def _single_curve(
        self, start: datetime, end: datetime, providers: Sequence[PriceProvider] | None = None
    ) -> PriceCurve:
        prices, provider, failures = self.spot_prices(start, end, providers)
        slots = [
            PriceSlot(p.start, p.end, p.eur_per_mwh, retail_price(p.eur_per_mwh, self._tariff))
            for p in prices
        ]
        notes = []
        if provider.is_estimate:
            notes.append(estimate_note(provider.name))
        if failures:
            notes.append("Unavailable sources: " + "; ".join(failures))
        return PriceCurve(start, end, slots, provider.name, provider.is_estimate, provider.attribution, notes)

    def day_curve(self, day: date) -> PriceCurve:
        return self.curve(*day_bounds_utc(day, day, self._tz))

    def hourly_retail(self, start: datetime, end: datetime) -> tuple[dict[datetime, float], str]:
        """Retail price per UTC hour for a longer interval (used to cost the household's history)."""
        curve = self.curve(start, end)
        return {hour: retail for hour, _, retail in curve.hourly()}, curve.source

    # ------------------------------------------------------------------ tool output
    def get_prices(self, day: str | None = None) -> dict[str, Any]:
        """Hourly retail prices for one local day, with cheapest/most expensive hours."""
        day_obj = parse_day(day) if day else self._clock().astimezone(self._tz).date()
        curve = self.day_curve(day_obj)
        return price_output(curve, day_obj, self._tz, self._location, self._tariff)


def merge_curves(parts: Sequence[PriceCurve]) -> PriceCurve:
    sources = list(dict.fromkeys(p.source for p in parts))
    notes = list(dict.fromkeys(n for p in parts for n in p.notes))
    return PriceCurve(
        start=parts[0].start,
        end=parts[-1].end,
        slots=[s for p in parts for s in p.slots],
        source=" + ".join(sources),
        is_estimate=any(p.is_estimate for p in parts),
        attribution=next((p.attribution for p in parts if p.attribution), None),
        notes=notes,
    )


def estimate_note(source: str) -> str:
    return (
        f"These prices are an ESTIMATE ({source}), not published market prices. Day-ahead prices for "
        "tomorrow are published around 13:00 CET; tell the user that the schedule is based on an estimate."
    )


def assign_periods(prices: Sequence[float]) -> list[str]:
    """Label each hour off_peak / mid_peak / peak by price terciles of the day."""
    ranked = sorted(range(len(prices)), key=lambda i: prices[i])
    labels = [""] * len(prices)
    for position, index in enumerate(ranked):
        labels[index] = PERIODS[min(2, position * 3 // max(1, len(prices)))]
    return labels


def price_output(
    curve: PriceCurve, day: date, tz: ZoneInfo, location: LocationSettings, tariff: TariffSettings
) -> dict[str, Any]:
    hourly = curve.hourly()
    retail = [r for _, _, r in hourly]
    periods = assign_periods(retail)
    rows = [
        {
            "hour": hour.astimezone(tz).hour,
            "time": hour.astimezone(tz).isoformat(timespec="minutes"),
            "rate": r,
            "spot_eur_per_mwh": spot,
            "period": period,
        }
        for (hour, spot, r), period in zip(hourly, periods, strict=True)
    ]
    by_price = sorted(rows, key=lambda x: x["rate"])
    output: dict[str, Any] = {
        "date": day.isoformat(),
        "timezone": str(tz),
        "bidding_zone": location.bidding_zone,
        "currency": tariff.currency,
        "unit": "EUR per kWh (retail, incl. surcharges and VAT)",
        "pricing_type": "dynamic tariff: day-ahead spot price + surcharges + VAT",
        "source": curve.source,
        "is_estimate": curve.is_estimate,
        "summary": {
            "average_rate": round(mean(retail), 4),
            "min_rate": min(retail),
            "max_rate": max(retail),
            "cheapest_hours": sorted(_label(r) for r in by_price[:SUMMARY_HOURS]),
            "most_expensive_hours": sorted(_label(r) for r in by_price[-SUMMARY_HOURS:]),
            "negative_spot_price_hours": [_label(r) for r in rows if r["spot_eur_per_mwh"] < 0],
        },
        "tariff_assumptions": tariff_assumptions(tariff),
        "hourly_rates": rows,
    }
    if curve.attribution:
        output["attribution"] = curve.attribution
    if curve.notes:
        output["note"] = " ".join(curve.notes)
    return output


def _label(row: dict[str, Any]) -> str:
    return f"{row['time'][11:16]}"
