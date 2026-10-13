"""Constants and fakes shared by the tests."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage

from energy_advisor.domain.prices import PriceUnavailableError, SpotPrice
from energy_advisor.timeutils import hours_between

FIXTURES = Path(__file__).parent / "fixtures"
BERLIN = ZoneInfo("Europe/Berlin")
LAST_SEEDED_DAY = date(2026, 10, 1)  # the test database ends on this (local) day
NOW = datetime(2026, 10, 2, 9, 30, tzinfo=BERLIN)  # the clock injected into services and the agent


def fixed_clock() -> datetime:
    return NOW


class FakeToolModel(GenericFakeChatModel):
    """Scripted chat model that supports ``bind_tools`` and records what it was sent."""

    received: list[list[BaseMessage]] = []  # noqa: RUF012 - pydantic field, copied per instance

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class StaticPriceProvider:
    """Price provider returning a fixed EUR/MWh value for every hour (or failing on demand)."""

    is_estimate = False
    attribution = "test data"

    def __init__(
        self, eur_per_mwh: float = 100.0, name: str = "static", fail: bool = False, slot_minutes: int = 60
    ):
        self.name = name
        self._price = eur_per_mwh
        self._fail = fail
        self._slot = timedelta(minutes=slot_minutes)
        self.calls = 0

    def fetch(self, start: datetime, end: datetime) -> list[SpotPrice]:
        self.calls += 1
        if self._fail:
            raise PriceUnavailableError("provider down")
        prices = []
        t = start.astimezone(UTC)
        while t < end:
            prices.append(SpotPrice(t, t + self._slot, self._price))
            t += self._slot
        return prices


def hourly_prices(start: datetime, end: datetime, values: list[float]) -> list[SpotPrice]:
    hours = list(hours_between(start, end))
    return [SpotPrice(h, h + timedelta(hours=1), v) for h, v in zip(hours, values, strict=True)]
