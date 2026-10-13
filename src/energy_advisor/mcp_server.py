"""MCP server: the advisor's domain tools for any MCP client (Claude Desktop, Claude Code, VS Code, ...).

The server does not contain its own logic. It registers the same functions the LangGraph agent uses as
tools (``agent/tools.py``), so prices, cache, solar forecast and the optimizer behave exactly as in the
CLI and the app. Their type hints become the JSON schema of each tool and their docstrings the
description the client's model reads. One extra tool, ``plan_devices``, returns the app's result screen.

Run with ``energy-advisor mcp`` (stdio, for local clients) or ``energy-advisor mcp --transport http``.
See ``mcp.md`` for how it works and how to connect a client.
"""

from typing import Literal

from langchain_core.tools import StructuredTool
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from energy_advisor import __version__
from energy_advisor.agent.tools import build_tools
from energy_advisor.config import Settings, get_settings
from energy_advisor.optimization.service import DEVICE_PRESETS, DeviceSpec
from energy_advisor.services.container import Services, build_services
from energy_advisor.timeutils import parse_local_datetime
from energy_advisor.ui.planner import result_markdown

INSTRUCTIONS = """\
Energy advisor for a household in Munich, Germany, with a PV system, a heat pump, an electric car with an
11 kW wallbox and a dynamic electricity tariff (household price = German day-ahead price + fees + VAT).
All times are local (Europe/Berlin), written as YYYY-MM-DDTHH:MM; dates as YYYY-MM-DD.
- To decide WHEN to charge or run something, use find_cheapest_window (one device) or plan_devices
  (several devices sharing the solar surplus). Do not work out schedules yourself.
- If a result says is_estimate=true, tell the user the prices are estimates (tomorrow's prices are
  published around 13:00).
- Own solar power is valued at the feed-in tariff (what exporting it would earn).
"""

DeviceKey = Literal["ev", "dishwasher", "washing_machine"]

# Tools that only read data; the ones marked "open world" call external APIs (prices, weather).
OPEN_WORLD = {
    "get_weather_forecast",
    "get_electricity_prices",
    "find_cheapest_window",
    "estimate_pv_generation",
}


def build_server(settings: Settings | None = None, services: Services | None = None) -> MCPServer:
    """Create the server. Building it needs no network access and no API key."""
    settings = settings or get_settings()
    services = services or build_services(settings)
    tz = settings.location.tz
    server = MCPServer(
        "energy-advisor",
        title="Solar & EV Energy Advisor",
        instructions=INSTRUCTIONS,
        version=__version__,
    )

    # The agent's tools, reused as they are: same code path, same validation, same error messages.
    for lc_tool in build_tools(services):
        if not isinstance(lc_tool, StructuredTool) or lc_tool.func is None:
            raise TypeError(f"Tool {lc_tool.name} has no plain function to expose")
        server.add_tool(
            lc_tool.func,  # the plain Python function behind the LangChain tool
            name=lc_tool.name,
            description=lc_tool.description,
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=lc_tool.name in OPEN_WORLD),
        )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def plan_devices(
        devices: list[DeviceKey],
        start: str | None = None,
        deadline: str | None = None,
        ev_kwh: float = 20.0,
    ) -> str:
        """Best times for several devices in one time window, sharing the solar surplus (it is never counted
        twice). Returns a ready-to-show report: per device the time slots, energy from solar and grid, cost,
        the cost of starting immediately and the saving.

        Args:
            devices: Any of "ev" (electric car, 11 kW, may pause), "dishwasher" (1.0 kWh, 2 h in one go),
                "washing_machine" (0.8 kWh, 2 h in one go).
            start: Local time "YYYY-MM-DDTHH:MM" from which devices may run (empty = now).
            deadline: Local time "YYYY-MM-DDTHH:MM" by which they must be finished (empty = start + 24 h).
            ev_kwh: Energy to charge into the car in kWh.
        """
        specs = []
        for key in dict.fromkeys(devices):  # keep order, drop duplicates
            preset = DEVICE_PRESETS[key]
            if key == "ev":
                preset = DeviceSpec(preset.key, preset.name, ev_kwh, preset.power_kw, preset.interruptible)
            specs.append(preset)
        plan = services.optimizer.plan_devices(
            specs,
            parse_local_datetime(start, tz) if start else None,
            parse_local_datetime(deadline, tz) if deadline else None,
        )
        return result_markdown(plan, tz, settings.tariff.feed_in_tariff_eur_per_kwh)

    return server
