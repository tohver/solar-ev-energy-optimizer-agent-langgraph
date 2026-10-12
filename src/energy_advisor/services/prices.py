"""Electricity prices.

Currently a mock time-of-use tariff (the same schedule every day). Phase 4 replaces it with real
day-ahead market prices behind the same ``get_prices`` interface.
"""

from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from energy_advisor.config import PricingSettings

PEAK_HOURS = range(17, 21)  # 17:00-21:00
MID_PEAK_HOURS = range(6, 22)  # 06:00-22:00 outside the peak window


class InvalidDateError(ValueError):
    pass


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise InvalidDateError(f"Invalid date '{value}'. Use YYYY-MM-DD format.") from exc


def period_for_hour(hour: int) -> str:
    if hour in PEAK_HOURS:
        return "peak"
    if hour in MID_PEAK_HOURS:
        return "mid_peak"
    return "off_peak"


class MockPriceService:
    def __init__(self, pricing: PricingSettings, today: Callable[[], date] = date.today) -> None:
        self._p = pricing
        self._today = today

    def get_prices(self, day: str | None = None) -> dict[str, Any]:
        """Hourly rates for one day: off-peak 22-06, peak 17-21 (with demand charge), mid-peak otherwise."""
        day_obj = parse_date(day) if day else self._today()
        multipliers = {
            "off_peak": self._p.off_peak_multiplier,
            "mid_peak": self._p.mid_peak_multiplier,
            "peak": self._p.peak_multiplier,
        }
        hourly_rates = []
        for hour in range(24):
            period = period_for_hour(hour)
            hourly_rates.append(
                {
                    "hour": hour,
                    "rate": round(self._p.base_price_per_kwh * multipliers[period], 4),
                    "period": period,
                    "demand_charge": self._p.peak_demand_charge if period == "peak" else 0.0,
                }
            )
        return {
            "date": day_obj.isoformat(),
            "pricing_type": "time_of_use",
            "source": "mock tariff",
            "currency": self._p.currency,
            "unit": "per_kWh",
            "hourly_rates": hourly_rates,
        }
