"""aWATTar market data API: hourly day-ahead prices for Germany, no API key (fair use ~100 requests/day)."""

from datetime import UTC, datetime

import httpx
from pydantic import BaseModel

from energy_advisor.domain.prices import PriceUnavailableError, SpotPrice

URL = "https://api.awattar.de/v1/marketdata"


class AwattarEntry(BaseModel):
    start_timestamp: int  # milliseconds
    end_timestamp: int
    marketprice: float
    unit: str


class AwattarResponse(BaseModel):
    data: list[AwattarEntry]


class AwattarProvider:
    name = "aWATTar day-ahead (EPEX spot, hourly)"
    is_estimate = False
    attribution = "aWATTar market data API"

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def fetch(self, start: datetime, end: datetime) -> list[SpotPrice]:
        resp = self._http.get(
            URL, params={"start": int(start.timestamp() * 1000), "end": int(end.timestamp() * 1000)}
        )
        resp.raise_for_status()
        return parse(AwattarResponse.model_validate(resp.json()))


def parse(data: AwattarResponse) -> list[SpotPrice]:
    if not data.data:
        raise PriceUnavailableError("empty response")
    prices = []
    for e in data.data:
        if "mwh" not in e.unit.lower():
            raise PriceUnavailableError(f"unexpected unit {e.unit!r}")
        prices.append(
            SpotPrice(
                datetime.fromtimestamp(e.start_timestamp / 1000, UTC),
                datetime.fromtimestamp(e.end_timestamp / 1000, UTC),
                e.marketprice,
            )
        )
    return prices
