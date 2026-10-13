"""Last-resort price source: the typical daily shape of German day-ahead prices.

Night is moderate, the morning and especially the evening are expensive, and midday is cheap because
of solar power. A small deterministic variation per day makes days differ. Always flagged as an estimate.
"""

import random
from datetime import datetime
from zoneinfo import ZoneInfo

from energy_advisor.domain.climate import seed_for
from energy_advisor.domain.prices import SpotPrice
from energy_advisor.timeutils import HOUR, hours_between

# Typical EUR/MWh by local hour of day (spring/autumn); summer middays are cheaper, winter flatter.
TYPICAL_EUR_PER_MWH = (
    95, 90, 87, 85, 86, 92, 110, 130, 125, 105, 85, 72,
    64, 62, 67, 82, 105, 138, 160, 155, 138, 120, 108, 100,
)  # fmt: skip
MIDDAY_HOURS = range(10, 16)
SEASONAL_MIDDAY_FACTOR = {
    6: 0.6,
    7: 0.6,
    5: 0.75,
    8: 0.75,
    4: 0.85,
    9: 0.85,
    12: 1.15,
    1: 1.15,
    11: 1.1,
    2: 1.1,
}


class SyntheticPriceProvider:
    name = "synthetic profile (typical German day-ahead shape)"
    is_estimate = True
    attribution = None

    def __init__(self, tz: ZoneInfo) -> None:
        self._tz = tz

    def fetch(self, start: datetime, end: datetime) -> list[SpotPrice]:
        prices = []
        for hour in hours_between(start, end):
            local = hour.astimezone(self._tz)
            rng = random.Random(seed_for("price", local.date()))
            level = rng.uniform(0.8, 1.2)  # the same level for the whole local day
            base: float = TYPICAL_EUR_PER_MWH[local.hour]
            if local.hour in MIDDAY_HOURS:
                base *= SEASONAL_MIDDAY_FACTOR.get(local.month, 1.0)
            noise = random.Random(seed_for("price", hour)).uniform(-6, 6)
            prices.append(SpotPrice(hour, hour + HOUR, round(base * level + noise, 2)))
        return prices
