"""Interface every price provider implements (a ``Protocol``: no inheritance needed)."""

from datetime import datetime
from typing import Protocol

from energy_advisor.domain.prices import SpotPrice


class PriceProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def is_estimate(self) -> bool: ...

    @property
    def attribution(self) -> str | None: ...

    def fetch(self, start: datetime, end: datetime) -> list[SpotPrice]:
        """Spot prices for slots starting in the UTC interval ``[start, end)``.

        Raises ``PriceUnavailableError`` (or an HTTP error) when the provider cannot deliver.
        """
        ...
