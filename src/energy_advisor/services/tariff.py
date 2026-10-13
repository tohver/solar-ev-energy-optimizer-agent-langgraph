"""Wholesale → retail: what a household with a dynamic tariff pays for a kWh."""

from typing import Any

from energy_advisor.config import TariffSettings


def retail_price(spot_eur_per_mwh: float, tariff: TariffSettings) -> float:
    """``(spot / 1000 + surcharges) * (1 + VAT)`` in EUR/kWh. Negative spot prices are passed through."""
    return round((spot_eur_per_mwh / 1000 + tariff.surcharges_eur_per_kwh) * (1 + tariff.vat_rate), 4)


def tariff_assumptions(tariff: TariffSettings) -> dict[str, Any]:
    return {
        "formula": "retail = (spot / 1000 + surcharges) x (1 + VAT)",
        "surcharges_eur_per_kwh_net": tariff.surcharges_eur_per_kwh,
        "vat_rate": tariff.vat_rate,
        "feed_in_tariff_eur_per_kwh": tariff.feed_in_tariff_eur_per_kwh,
        "fixed_tariff_eur_per_kwh_for_comparison": tariff.fixed_price_eur_per_kwh,
        "note": "Surcharges (grid fees, levies, electricity tax, supplier margin) are a typical assumption.",
    }
