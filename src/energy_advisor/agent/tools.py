"""LangChain tools exposed to the agent.

The tools are thin wrappers: they validate input, call a service and return a JSON-friendly dict.
Errors are returned as ``{"error": ...}`` so the LLM can react instead of the run crashing.
Docstrings matter: the LLM reads them to decide which tool to call.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import BaseTool, tool

from energy_advisor.data.aggregation import (
    energy_balance,
    summarize_generation,
    summarize_recent,
    summarize_usage,
)
from energy_advisor.optimization.scheduling import ScheduleError
from energy_advisor.services.container import Services
from energy_advisor.services.savings import calculate_savings
from energy_advisor.timeutils import InvalidDateError, day_bounds_utc, parse_day

logger = logging.getLogger(__name__)
DEVICE_TYPES = ("EV", "heat_pump", "appliance", "base_load")


def build_tools(services: Services, optimizer: bool = True) -> list[BaseTool]:
    """All agent tools; ``optimizer=False`` leaves out the scheduling tools (ablation runs)."""
    repo = services.repository
    settings = services.settings
    tz = settings.location.tz

    def utc_range(start_date: str, end_date: str) -> tuple[datetime, datetime, int]:
        first, last = parse_day(start_date), parse_day(end_date)
        if last < first:
            raise InvalidDateError("end_date must not be before start_date")
        start, end = day_bounds_utc(first, last, tz)
        return start, end, (last - first).days + 1

    @tool
    def get_weather_forecast(location: str | None = None, days: int = 3) -> dict[str, Any]:
        """Hourly weather forecast from today: temperature, sky condition and solar irradiance (W/m2).

        Args:
            location: Place name; leave empty for the household's own location (Munich).
            days: Number of days to forecast (1-7).
        """
        return services.weather.forecast(location, days)

    @tool
    def get_electricity_prices(date: str | None = None) -> dict[str, Any]:
        """Hourly household electricity prices (EUR/kWh) for one day under the dynamic tariff.

        Prices follow the day-ahead market; each hour is labelled off_peak / mid_peak / peak relative
        to that day. Check `is_estimate`: tomorrow's prices are only published around 13:00.
        To decide WHEN to run or charge a device, use find_cheapest_window instead (if available).

        Args:
            date: Day in YYYY-MM-DD format (defaults to today).
        """
        try:
            return services.prices.get_prices(date)
        except InvalidDateError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.exception("get_electricity_prices failed")
            return {"error": f"Failed to get electricity prices: {exc}"}

    @tool
    def query_energy_usage(start_date: str, end_date: str, device_type: str | None = None) -> dict[str, Any]:
        """The household's historical electricity consumption and cost (EUR) for a date range, aggregated.

        Args:
            start_date: First day, YYYY-MM-DD.
            end_date: Last day (inclusive), YYYY-MM-DD.
            device_type: Optional filter: "EV", "heat_pump", "appliance" or "base_load" (case-insensitive).
        """
        try:
            start, end, days = utc_range(start_date, end_date)
            records = repo.usage_between(start, end)
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
                **summarize_usage(records, tz, days),
            }
        except InvalidDateError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.exception("query_energy_usage failed")
            return {"error": f"Failed to query energy usage: {exc}"}

    @tool
    def query_solar_generation(start_date: str, end_date: str) -> dict[str, Any]:
        """The household's historical solar (PV) generation for a date range, aggregated, plus the
        energy balance: how much solar was used at home, exported, and how much came from the grid.

        Args:
            start_date: First day, YYYY-MM-DD.
            end_date: Last day (inclusive), YYYY-MM-DD.
        """
        try:
            start, end, days = utc_range(start_date, end_date)
            generation = repo.generation_between(start, end)
            balance = energy_balance(
                repo.usage_between(start, end), generation, settings.tariff.feed_in_tariff_eur_per_kwh
            )
            return {
                "start_date": start_date,
                "end_date": end_date,
                "pv_system_kwp": settings.household.pv_peak_kw,
                **summarize_generation(generation, tz, days),
                "energy_balance": balance,
            }
        except InvalidDateError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.exception("query_solar_generation failed")
            return {"error": f"Failed to query solar generation: {exc}"}

    @tool
    def get_recent_energy_summary(hours: int = 24) -> dict[str, Any]:
        """Summary of electricity use and solar generation over the last N hours.

        Args:
            hours: Number of hours to look back (default 24).
        """
        try:
            end = services.clock()
            start = end - timedelta(hours=hours)
            return summarize_recent(
                repo.usage_between(start, end), repo.generation_between(start, end), hours
            )
        except Exception as exc:
            logger.exception("get_recent_energy_summary failed")
            return {"error": f"Failed to get recent energy summary: {exc}"}

    @tool
    def find_cheapest_window(
        energy_kwh: float,
        power_kw: float,
        start: str | None = None,
        deadline: str | None = None,
        interruptible: bool = True,
        use_solar: bool = True,
        device: str = "device",
    ) -> dict[str, Any]:
        """Compute the cheapest time to run a flexible load, using hourly prices AND the solar forecast.

        ALWAYS use this for "when should I charge/run ..." questions instead of working out hours yourself.
        It searches between start (default: now) and deadline (default: 24 hours after the start) and returns
        the exact schedule, its cost in EUR, the energy from solar, and the saving compared with starting at
        the start time. Start and deadline come from the user (or from the window set in the app); never
        invent them. Without a deadline, tell the user that 24 hours were assumed and ask by when they need it.

        Args:
            energy_kwh: Energy the device needs, e.g. 30 for EV charging, 1.0 for a dishwasher cycle.
            power_kw: Power while running: EV wallbox 11; dishwasher about 0.5; washing machine about 0.4;
                dryer about 0.6; heat pump boost about 2.
            start: Local time "YYYY-MM-DDTHH:MM" from which the device may run (empty = now).
            deadline: Local time "YYYY-MM-DDTHH:MM" by which it must be finished (empty = start + 24 hours).
            interruptible: True if it may pause (EV charging); False if it must run in one go (appliances).
            use_solar: Consider own solar surplus (valued at the feed-in tariff).
            device: Name for the output, e.g. "EV".
        """
        try:
            return services.optimizer.cheapest_window(
                energy_kwh, power_kw, start, deadline, interruptible, use_solar, device
            )
        except (InvalidDateError, ScheduleError) as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.exception("find_cheapest_window failed")
            return {"error": f"Failed to compute a schedule: {exc}"}

    @tool
    def estimate_pv_generation(days: int = 1) -> dict[str, Any]:
        """Expected solar (PV) yield of the household for today and the next days, per hour, with the hours of
        highest surplus after the household's own base load.

        Args:
            days: Number of days starting today (1-3).
        """
        try:
            return services.optimizer.pv_forecast(days)
        except Exception as exc:
            logger.exception("estimate_pv_generation failed")
            return {"error": f"Failed to estimate PV generation: {exc}"}

    @tool
    def search_energy_tips(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the energy-saving knowledge base (RAG). Returns ranked tips with their source file to cite.

        Args:
            query: What to look for, e.g. "heat pump flow temperature".
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
        """Calculate energy, cost (EUR) and CO2 savings of an optimization.

        Two kinds of optimization are supported:
        - Using LESS energy: pass a smaller optimized_usage_kwh.
        - Using the SAME energy at a CHEAPER time (load shifting): keep optimized_usage_kwh equal to
          current_usage_kwh, pass the expensive rate as price_per_kwh and the cheaper rate as
          optimized_price_per_kwh. For own solar power use the feed-in tariff as its price.

        Args:
            device_type: Device being optimized.
            current_usage_kwh: Current energy use in kWh (per day / per cycle).
            optimized_usage_kwh: Energy use after the optimization (same period).
            price_per_kwh: Current price in EUR/kWh (defaults to a typical fixed tariff).
            optimized_price_per_kwh: Price after the optimization (defaults to price_per_kwh).
        """
        try:
            tariff = settings.tariff
            return calculate_savings(
                device_type=device_type,
                current_usage_kwh=current_usage_kwh,
                optimized_usage_kwh=optimized_usage_kwh,
                price_per_kwh=tariff.fixed_price_eur_per_kwh if price_per_kwh is None else price_per_kwh,
                optimized_price_per_kwh=optimized_price_per_kwh,
                co2_kg_per_kwh=settings.household.co2_kg_per_kwh,
                currency=tariff.currency,
            )
        except Exception as exc:
            logger.exception("calculate_energy_savings failed")
            return {"error": f"Failed to calculate savings: {exc}"}

    tools: list[BaseTool] = [
        search_energy_tips,
        get_weather_forecast,
        get_electricity_prices,
        query_energy_usage,
        query_solar_generation,
        get_recent_energy_summary,
        calculate_energy_savings,
    ]
    if optimizer:
        tools += [find_cheapest_window, estimate_pv_generation]
    return tools
