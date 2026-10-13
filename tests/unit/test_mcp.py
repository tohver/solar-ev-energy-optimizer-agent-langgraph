"""MCP server: tool list, calls through a real MCP client (in memory, offline) and a stdio round trip."""

import asyncio
import json
import os
import sys
from typing import Any

import pytest

pytest.importorskip("mcp")
from mcp.client.client import Client, StdioServerParameters

from energy_advisor.mcp_server import build_server
from energy_advisor.services.container import build_services
from tests.helpers import StaticPriceProvider, fixed_clock

EXPECTED_TOOLS = {
    "search_energy_tips",
    "get_weather_forecast",
    "get_electricity_prices",
    "query_energy_usage",
    "query_solar_generation",
    "get_recent_energy_summary",
    "calculate_energy_savings",
    "find_cheapest_window",
    "estimate_pv_generation",
    "plan_devices",
}


@pytest.fixture
def server(settings, offline_http):
    services = build_services(
        settings, http=offline_http, clock=fixed_clock, price_providers=[StaticPriceProvider(100)]
    )
    return build_server(settings, services)


def _call(server, name: str, arguments: dict[str, Any]) -> Any:
    async def run() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def _payload(result: Any) -> dict[str, Any]:
    data = result.structured_content or json.loads(result.content[0].text)
    return data.get("result", data)


def test_all_tools_are_listed_with_descriptions_and_schemas(server):
    async def run() -> Any:
        async with Client(server) as client:
            return (await client.list_tools()).tools, client.server_info

    tools, info = asyncio.run(run())
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert info.name == "energy-advisor"
    by_name = {t.name: t for t in tools}
    schema = by_name["find_cheapest_window"].input_schema
    assert schema["required"] == ["energy_kwh", "power_kw"]
    assert "deadline" in schema["properties"]
    assert "find_cheapest_window" in by_name["get_electricity_prices"].description
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


def test_prices_through_the_protocol(server):
    result = _call(server, "get_electricity_prices", {"date": "2026-10-03"})
    assert not result.is_error
    assert len(_payload(result)["hourly_rates"]) == 24


def test_scheduling_through_the_protocol(server):
    result = _call(server, "find_cheapest_window", {"energy_kwh": 11, "power_kw": 11, "use_solar": False})
    data = _payload(result)
    assert data["window"]["start"] == "2026-10-02 09:30"
    assert data["energy_kwh"] == 11


def test_plan_devices_returns_the_result_screen(server):
    result = _call(server, "plan_devices", {"devices": ["ev", "dishwasher"], "ev_kwh": 10})
    text = result.content[0].text
    assert "### Electric car: 10 kWh" in text
    assert "### Dishwasher: 1 kWh" in text
    assert "Cost analysis" in text


def test_invalid_input_comes_back_as_a_readable_error(server):
    result = _call(server, "find_cheapest_window", {"energy_kwh": 5, "power_kw": 11, "deadline": "tomorrow"})
    assert "Invalid date-time" in _payload(result)["error"]
    bad = _call(server, "plan_devices", {"devices": ["toaster"]})
    assert bad.is_error


def test_stdio_server_starts_and_lists_tools(tmp_path):
    """The real thing: the server as a separate process speaking MCP over stdin/stdout."""
    env = {
        **os.environ,
        "PATHS__DB": str(tmp_path / "energy.db"),
        "PATHS__PRICE_CACHE": str(tmp_path / "prices.sqlite"),
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "energy_advisor.cli", "mcp"], env=env)

    async def run() -> set[str]:
        async with Client(params) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert asyncio.run(run()) == EXPECTED_TOOLS
