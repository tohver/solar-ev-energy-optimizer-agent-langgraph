"""Energy-Charts (Fraunhofer ISE): day-ahead prices for a bidding zone, no API key needed.

Data source Bundesnetzagentur | SMARD.de, licence CC BY 4.0 (attribution required). Since October
2025 the day-ahead market uses 15-minute slots. The API rate-limits bursts (HTTP 429), so results
should be cached (see ``cache.py``).
"""

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import httpx
from pydantic import BaseModel

from energy_advisor.domain.prices import PriceUnavailableError, SpotPrice

URL = "https://api.energy-charts.info/price"
ATTRIBUTION = "Energy-Charts (Fraunhofer ISE), data: Bundesnetzagentur | SMARD.de, CC BY 4.0"


class EnergyChartsResponse(BaseModel):
    unix_seconds: list[int]
    price: list[float | None]
    unit: str


class EnergyChartsProvider:
    name = "Energy-Charts day-ahead (EPEX spot)"
    is_estimate = False
    attribution = ATTRIBUTION

    def __init__(self, http: httpx.Client, bidding_zone: str = "DE-LU") -> None:
        self._http = http
        self._zone = bidding_zone

    def fetch(self, start: datetime, end: datetime) -> list[SpotPrice]:
        resp = self._http.get(
            URL,
            # the API's end is inclusive, so stop one second before the end of the interval
            params={"bzn": self._zone, "start": int(start.timestamp()), "end": int(end.timestamp()) - 1},
        )
        if resp.status_code == 404:
            raise PriceUnavailableError("no prices published for this period")
        resp.raise_for_status()
        return parse(EnergyChartsResponse.model_validate(resp.json()))


def parse(data: EnergyChartsResponse) -> list[SpotPrice]:
    if "MWh" not in data.unit:
        raise PriceUnavailableError(f"unexpected unit {data.unit!r}")
    times = [datetime.fromtimestamp(t, UTC) for t in data.unix_seconds]
    if not times:
        raise PriceUnavailableError("empty response")
    # slot length from the spacing of the timestamps (15 min since Oct 2025, 60 min before)
    step = min((b - a for a, b in pairwise(times)), default=timedelta(minutes=15))
    return [SpotPrice(t, t + step, float(p)) for t, p in zip(times, data.price, strict=True) if p is not None]
