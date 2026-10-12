"""Price estimate for periods without published prices (e.g. tomorrow before ~13:00).

For each hour of the target day: the average spot price of the same local hour over the previous
days, with the same weekday counted twice (weekends and weekdays differ). Always flagged as an estimate.
"""

from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from energy_advisor.clients.prices.base import PriceProvider
from energy_advisor.domain.prices import PriceUnavailableError, SpotPrice
from energy_advisor.timeutils import HOUR, Clock, day_bounds_utc, hours_between

SAME_WEEKDAY_WEIGHT = 2.0


class EstimatePriceProvider:
    is_estimate = True
    attribution = None

    def __init__(self, sources: Sequence[PriceProvider], tz: ZoneInfo, clock: Clock, lookback_days: int = 7):
        self._sources = list(sources)
        self._tz = tz
        self._clock = clock
        self._lookback = lookback_days
        self.name = f"estimate (weighted average of the previous {lookback_days} days)"

    def fetch(self, start: datetime, end: datetime) -> list[SpotPrice]:
        first_target = start.astimezone(self._tz).date()
        # history must be published: it ends with today's prices at the latest
        today = self._clock().astimezone(self._tz).date()
        history_end = min(first_target, today + timedelta(days=1))
        history = self._history(history_end - timedelta(days=self._lookback), history_end)

        by_hour: dict[int, list[tuple[int, float]]] = defaultdict(list)  # local hour -> (weekday, price)
        for p in history:
            local = p.start.astimezone(self._tz)
            by_hour[local.hour].append((local.weekday(), p.eur_per_mwh))

        prices = []
        for hour in hours_between(start, end):
            local = hour.astimezone(self._tz)
            samples = by_hour.get(local.hour) or [s for v in by_hour.values() for s in v]
            if not samples:
                raise PriceUnavailableError("no history to estimate from")
            weights = [SAME_WEEKDAY_WEIGHT if weekday == local.weekday() else 1.0 for weekday, _ in samples]
            value = sum(w * p for w, (_, p) in zip(weights, samples, strict=True)) / sum(weights)
            prices.append(SpotPrice(hour, hour + HOUR, round(value, 2)))
        return prices

    def _history(self, first: date, end_exclusive: date) -> list[SpotPrice]:
        start, end = day_bounds_utc(first, end_exclusive - timedelta(days=1), self._tz)
        errors = []
        for source in self._sources:
            try:
                return source.fetch(start, end)
            except Exception as exc:
                errors.append(f"{source.name}: {exc}")
        raise PriceUnavailableError("no price history available: " + "; ".join(errors))
