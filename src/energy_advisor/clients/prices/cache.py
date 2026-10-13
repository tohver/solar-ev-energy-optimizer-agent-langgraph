"""Disk cache for published day-ahead prices (decorator around a real provider).

Published day-ahead prices never change, so every slot fetched once can be reused forever. This
keeps repeated agent runs and evaluations fast and respects the providers' rate limits.
Estimates are never cached: only real providers are wrapped.
"""

import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from energy_advisor.clients.prices.base import PriceProvider
from energy_advisor.domain.prices import SpotPrice

logger = logging.getLogger(__name__)
COMPLETE = 0.99

SCHEMA = """
CREATE TABLE IF NOT EXISTS spot_price (
    provider TEXT NOT NULL,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    eur_per_mwh REAL NOT NULL,
    PRIMARY KEY (provider, start_ts)
)
"""


class CachedPriceProvider:
    def __init__(self, inner: PriceProvider, path: Path) -> None:
        self._inner = inner
        self._path = path
        self.name = inner.name
        self.is_estimate = inner.is_estimate
        self.attribution = inner.attribution

    def _connect(self) -> sqlite3.Connection:
        # the schema is ensured on every connection, so a deleted cache file is simply rebuilt
        self._path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self._path)
        con.execute(SCHEMA)
        return con

    def cached(self, start: datetime, end: datetime) -> list[SpotPrice]:
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT start_ts, end_ts, eur_per_mwh FROM spot_price "
                "WHERE provider = ? AND start_ts >= ? AND start_ts < ? ORDER BY start_ts",
                (self.name, int(start.timestamp()), int(end.timestamp())),
            ).fetchall()
        return [
            SpotPrice(datetime.fromtimestamp(s, UTC), datetime.fromtimestamp(e, UTC), p) for s, e, p in rows
        ]

    def fetch(self, start: datetime, end: datetime) -> list[SpotPrice]:
        hit = self.cached(start, end)
        covered = sum((p.end - p.start).total_seconds() for p in hit)
        if covered >= COMPLETE * (end - start).total_seconds():
            logger.debug("price cache hit: %s %s..%s", self.name, start, end)
            return hit
        prices = self._inner.fetch(start, end)
        with closing(self._connect()) as con, con:
            con.executemany(
                "INSERT OR REPLACE INTO spot_price VALUES (?, ?, ?, ?)",
                [
                    (self.name, int(p.start.timestamp()), int(p.end.timestamp()), p.eur_per_mwh)
                    for p in prices
                ],
            )
        return prices
