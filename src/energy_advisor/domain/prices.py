"""Electricity price data: wholesale spot prices and the retail price slots derived from them."""

from dataclasses import dataclass
from datetime import datetime, timedelta


class PriceUnavailableError(RuntimeError):
    """A provider has no (complete) prices for the requested interval."""


@dataclass(frozen=True)
class SpotPrice:
    """Day-ahead market price for one market time unit (15 or 60 minutes). Times are aware UTC."""

    start: datetime
    end: datetime
    eur_per_mwh: float


@dataclass(frozen=True)
class PriceSlot:
    """What the household pays during one slot."""

    start: datetime
    end: datetime
    spot_eur_per_mwh: float
    price_eur_per_kwh: float  # retail, including surcharges and VAT

    @property
    def hours(self) -> float:
        return (self.end - self.start) / timedelta(hours=1)
