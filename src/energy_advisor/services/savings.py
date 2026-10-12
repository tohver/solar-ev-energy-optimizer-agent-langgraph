"""Savings calculator for "use less" and "shift to a cheaper hour" optimizations."""

from typing import Any

DAYS_PER_YEAR = 365


def calculate_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float,
    optimized_price_per_kwh: float | None,
    co2_kg_per_kwh: float,
    currency: str = "EUR",
) -> dict[str, Any]:
    """Savings in kWh, money, percent, per year and CO2.

    Load shifting (same kWh, cheaper hour) saves money but no CO2 in this model; only reduced kWh
    reduce emissions.
    """
    new_price = price_per_kwh if optimized_price_per_kwh is None else optimized_price_per_kwh
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    current_cost = current_usage_kwh * price_per_kwh
    optimized_cost = optimized_usage_kwh * new_price
    savings = current_cost - optimized_cost
    percentage = savings / current_cost * 100 if current_cost > 0 else 0.0
    unit = currency.lower()

    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "price_per_kwh": price_per_kwh,
        "optimized_price_per_kwh": new_price,
        f"current_cost_{unit}": round(current_cost, 2),
        f"optimized_cost_{unit}": round(optimized_cost, 2),
        "savings_kwh": round(savings_kwh, 2),
        f"savings_{unit}": round(savings, 2),
        "savings_percentage": round(percentage, 1),
        f"annual_savings_{unit}": round(savings * DAYS_PER_YEAR, 2),
        "co2_saved_kg": round(savings_kwh * co2_kg_per_kwh, 2),
        "annual_co2_saved_kg": round(savings_kwh * co2_kg_per_kwh * DAYS_PER_YEAR, 1),
        "assumptions": (
            f"Annual figures assume the same saving every day; grid intensity {co2_kg_per_kwh} kg CO2/kWh "
            "(CO2 is only saved when kWh are reduced, not when usage is only shifted in time)."
        ),
    }
