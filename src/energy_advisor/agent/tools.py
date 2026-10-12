"""LangChain tools exposed to the agent.

The tools are thin wrappers: they validate input, call a service and return a JSON-friendly dict.
Errors are returned as ``{"error": ...}`` so the LLM can react instead of the run crashing.
Docstrings matter: the LLM reads them to decide which tool to call.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.tools import BaseTool, tool

from energy_advisor.data.aggregation import summarize_generation, summarize_recent, summarize_usage
from energy_advisor.services.container import Services
from energy_advisor.services.prices import InvalidDateError, parse_date
from energy_advisor.services.savings import calculate_savings

logger = logging.getLogger(__name__)


def build_tools(services: Services) -> list[BaseTool]:
    repo = services.repository
    settings = services.settings

    @tool
    def get_weather_forecast(location: str, days: int = 3) -> dict[str, Any]:
        """Get the hourly weather forecast for a location: temperature, sky condition and solar irradiance.

        Args:
            location: Place name, e.g. "San Francisco, CA".
            days: Number of days to forecast (1-7).
        """
        return services.weather.forecast(location, days)

    @tool
    def get_electricity_prices(date: str | None = None) -> dict[str, Any]:
        """Get hourly electricity rates (off_peak / mid_peak / peak) and demand charges for one day.

        Args:
            date: Date in YYYY-MM-DD format (defaults to today).
        """
        try:
            return services.prices.get_prices(date)
        except InvalidDateError as exc:
            return {"error": str(exc)}

    @tool
    def query_energy_usage(start_date: str, end_date: str, device_type: str | None = None) -> dict[str, Any]:
        """Query the household's historical energy consumption (aggregated) for a date range.

        Args:
            start_date: First day, YYYY-MM-DD.
            end_date: Last day (inclusive), YYYY-MM-DD.
            device_type: Optional filter, e.g. "EV", "HVAC" or "appliance" (case-insensitive).
        """
        try:
            start, end = parse_date(start_date), parse_date(end_date) + timedelta(days=1)
            records = repo.usage_between(_midnight(start), _midnight(end))
            if device_type:
                available = sorted({r.device_type for r in records if r.device_type})
                wanted = device_type.strip().lower()
                records = [r for r in records if (r.device_type or "").lower() == wanted]
                if not records and available:
                    return {
                        "error": f"No records for device_type '{device_type}' in this range.",
                        "available_device_types": available,
                    }
            return {
                "start_date": start_date,
                "end_date": end_date,
                "device_type": device_type,
                **summarize_usage(records),
            }
        except Exception as exc:
            logger.exception("query_energy_usage failed")
            return {"error": f"Failed to query energy usage: {exc}"}

    @tool
    def query_solar_generation(start_date: str, end_date: str) -> dict[str, Any]:
        """Query the household's historical solar generation (aggregated) for a date range.

        Args:
            start_date: First day, YYYY-MM-DD.
            end_date: Last day (inclusive), YYYY-MM-DD.
        """
        try:
            start, end = parse_date(start_date), parse_date(end_date) + timedelta(days=1)
            records = repo.generation_between(_midnight(start), _midnight(end))
            return {
                "start_date": start_date,
                "end_date": end_date,
                **summarize_generation(records, days_in_range=(end - start).days),
            }
        except Exception as exc:
            logger.exception("query_solar_generation failed")
            return {"error": f"Failed to query solar generation: {exc}"}

    @tool
    def get_recent_energy_summary(hours: int = 24) -> dict[str, Any]:
        """Get a summary of energy usage and solar generation over the last N hours.

        Args:
            hours: Number of hours to look back (default 24).
        """
        try:
            end = services.now()
            start = end - timedelta(hours=hours)
            return summarize_recent(
                repo.usage_between(start, end), repo.generation_between(start, end), hours
            )
        except Exception as exc:
            logger.exception("get_recent_energy_summary failed")
            return {"error": f"Failed to get recent energy summary: {exc}"}

    @tool
    def search_energy_tips(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the energy-saving knowledge base (RAG). Returns ranked tips with their source file to cite.

        Args:
            query: What to look for, e.g. "pre-cooling before peak hours".
            max_results: Maximum number of tips (1-10).
        """
        try:
            return services.knowledge_base.search(query, max_results)
        except Exception as exc:
            logger.exception("search_energy_tips failed")
            return {"error": f"Failed to search energy tips: {exc}"}

    @tool
    def calculate_energy_savings(
        device_type: str,
        current_usage_kwh: float,
        optimized_usage_kwh: float,
        price_per_kwh: float | None = None,
        optimized_price_per_kwh: float | None = None,
    ) -> dict[str, Any]:
        """Calculate energy, cost and CO2 savings of an optimization.

        Two kinds of optimization are supported:
        - Using LESS energy: pass a smaller optimized_usage_kwh.
        - Using the SAME energy at a CHEAPER time (load shifting, e.g. peak -> off-peak):
          keep optimized_usage_kwh equal to current_usage_kwh, pass the expensive rate as
          price_per_kwh and the cheaper rate as optimized_price_per_kwh.

        Args:
            device_type: Device being optimized.
            current_usage_kwh: Current energy use in kWh (per day / per cycle).
            optimized_usage_kwh: Energy use after the optimization (same period).
            price_per_kwh: Current price per kWh (defaults to the base tariff price).
            optimized_price_per_kwh: Price per kWh after the optimization (defaults to price_per_kwh).
        """
        try:
            return calculate_savings(
                device_type=device_type,
                current_usage_kwh=current_usage_kwh,
                optimized_usage_kwh=optimized_usage_kwh,
                price_per_kwh=settings.pricing.base_price_per_kwh if price_per_kwh is None else price_per_kwh,
                optimized_price_per_kwh=optimized_price_per_kwh,
                co2_kg_per_kwh=settings.household.co2_kg_per_kwh,
                currency=settings.pricing.currency,
            )
        except Exception as exc:
            logger.exception("calculate_energy_savings failed")
            return {"error": f"Failed to calculate savings: {exc}"}

    return [
        get_weather_forecast,
        get_electricity_prices,
        query_energy_usage,
        query_solar_generation,
        get_recent_energy_summary,
        search_energy_tips,
        calculate_energy_savings,
    ]


def _midnight(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time())
