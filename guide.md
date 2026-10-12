# Project guide

This guide walks you through the Solar & EV Energy Optimizer: first as a user (what it can do and how to try
each feature), then as a reviewer (every assumption that was made and needs checking), and finally as a
developer (what lives in which module, how it works and why it was built that way). Every formula, and where
each piece of data is stored and for how long, is explained in [calculations.md](calculations.md).

Contents:

1. [Guided tour of the features](#1-guided-tour-of-the-features)
2. [What happens when you ask a question](#2-what-happens-when-you-ask-a-question)
3. [Assumptions to verify](#3-assumptions-to-verify)
4. [Module reference](#4-module-reference)
5. [Tests](#5-tests)

---

## 1. Guided tour of the features

Prerequisites: `uv sync --all-extras` and an `.env` file with `OPENAI_API_KEY` (see README). Every command
below is run as `uv run energy-advisor ...`.

### 1.1 Create the household's history: `seed-db`

```bash
uv run energy-advisor seed-db --days 30
```

Creates `data/energy_data.db` with 30 days of hourly consumption per device and PV generation, ending
yesterday. The output names its inputs, for example `weather: open-meteo.com (live) | prices: Energy-Charts
day-ahead (EPEX spot)`: the history is driven by **real past weather in Munich and real past electricity
prices**. Offline, it falls back to synthetic weather and prices and says so. Same seed, same data; re-running
replaces everything.

### 1.2 Build the knowledge base index: `build-index`

```bash
uv run energy-advisor build-index          # first time
uv run energy-advisor build-index --rebuild # after editing knowledge_base/*.txt
```

Splits the 8 documents in `knowledge_base/` into chunks, embeds them with OpenAI embeddings and stores them in
`data/vectorstore/` (Chroma).

### 1.3 Ask the agent: `ask`

```bash
uv run energy-advisor ask "<question>"
uv run energy-advisor ask "<question>" --export-svg answer.svg   # also save the output as an image
```

The output shows the tool calls (which data the agent fetched, with which arguments) and the answer. Questions
that exercise each feature:

| Feature | Try asking | Tool(s) behind it |
|---|---|---|
| Optimal charging window | "I need my EV charged with 30 kWh by 7:00 tomorrow. When should I charge it?" | `find_cheapest_window` |
| No deadline given | "When should I charge my EV with 20 kWh?" (plans for the next 24 h and asks you by when) | `find_cheapest_window` |
| Appliance scheduling | "My dishwasher needs 1 kWh. How much can I save by running it in the cheapest hours before 18:00 tomorrow instead of now?" | `find_cheapest_window` |
| Real electricity prices | "What will electricity cost tomorrow at 19:00?" | `get_electricity_prices` |
| Solar outlook | "Which of the next three days is best for charging from solar?" | `estimate_pv_generation` |
| Weather | "Will it be sunny tomorrow afternoon?" | `get_weather_forecast` |
| Usage history | "Which device used the most electricity in the last 7 days and what did it cost?" | `query_energy_usage` |
| Solar self-consumption | "How much of my solar power did I use myself last week?" | `query_solar_generation` (energy balance) |
| Savings and CO₂ | "If I move 10 kWh of charging from the evening to the cheapest hours, what do I save per year?" | `get_electricity_prices`, `calculate_energy_savings` |
| Knowledge base with sources | "What does section 14a mean for my wallbox?" | `search_energy_tips` |
| Recent summary | "How much did I use in the last 24 hours?" | `get_recent_energy_summary` |
| Other locations | "What is the weather in Hamburg tomorrow?" (ambiguous names such as "Frankfurt" are flagged) | `get_weather_forecast` |

### 1.4 Have a conversation: `chat` and the web app

```bash
uv run energy-advisor chat                                   # terminal conversation; 'exit' to quit
uv run --extra app streamlit run app/streamlit_app.py        # web app at http://localhost:8501
```

The agent remembers the conversation: ask "When should I charge my EV with 20 kWh?", and it plans for the next
24 hours, says so and asks by when; answer "By 7:00 tomorrow" and it recomputes with that deadline.

All settings of the web app are in the **left sidebar**, and only there:

- **From** (default: now, rounded up to the next quarter hour) and **Deadline** (default: From + 24 hours; it
  follows From until you change it yourself);
- **Devices**: Electric car, Dishwasher, Washing machine (several at once); for the car, the energy to charge;
- **OK**: computes the best times;
- **New conversation** (at the bottom): clears the chat memory and opens the chat.

The main area has three views:

- **Best times** (opens after OK): the result screen. For each device a table of the time slots with energy,
  solar share, grid share and cost, then the cost analysis (total, average price, cost of starting immediately,
  saving) and the solar contribution. Below it, a chart of the price, the solar surplus and the device runs.
  Devices share the solar surplus (largest first), so it is never counted twice. No LLM is involved.
- **Chat**: the conversation with memory. An empty chat shows a greeting and four example questions as small
  buttons; clicking one sends it. Scheduling questions use the sidebar's From and Deadline unless your message
  names other times.
- **Prices & solar**: hourly price and solar surplus for the sidebar's window, shown immediately.

### 1.5 Use the tools from Claude Desktop, Claude Code or VS Code: `mcp`

```bash
uv run --extra mcp energy-advisor mcp
```

Starts an MCP server with the advisor's tools. How it works and how to connect a client: [mcp.md](mcp.md).

### 1.6 Measure quality: `eval`

```bash
uv run energy-advisor eval                 # 15 cases, report in reports/
uv run energy-advisor eval --limit 3       # quick check
uv run energy-advisor eval --no-optimizer  # ablation: the same agent without the scheduling tools
```

Each case is answered by the agent and scored by an LLM judge (accuracy, relevance, completeness,
usefulness) plus deterministic tool-usage metrics. Output: Markdown, JSON and a heatmap PNG. The cases live in
`src/energy_advisor/evaluation/cases.yaml`; add your own without touching code.

### 1.7 Check retrieval quality: `rag-benchmark`

```bash
uv run energy-advisor rag-benchmark
```

For 12 labelled questions, shows at which rank the correct document appears for plain vector search, hybrid
search and hybrid search with LLM re-ranking, and summarizes with hit@3 and MRR (mean reciprocal rank: the
average of 1/rank of the right document, where 1.0 is perfect).

### 1.8 Charts: `report`

```bash
uv run energy-advisor report --out docs/img
```

Four PNG charts: the average day (consumption per device vs. solar yield), daily consumption vs. solar,
tomorrow's price curve with the solar forecast and the best solar window, and projected annual savings.

### 1.9 Change the household

Everything is configured in `.env` (see `.env.example`), for example:

```dotenv
LOCATION__NAME="Hamburg, Germany"
LOCATION__LATITUDE=53.55
LOCATION__LONGITUDE=9.99
TARIFF__SURCHARGES_EUR_PER_KWH=0.21
HOUSEHOLD__PV_PEAK_KW=10
```

Re-run `seed-db` afterwards so the history matches the new household.

---

## 2. What happens when you ask a question

Example: *"I need my EV charged with 30 kWh by 7:00 tomorrow."*

1. **CLI** (`cli.py: ask` or `chat`) or the web app loads `Settings` and builds all services
   (`services/container.py: build_services`).
2. **Agent** (`agent/graph.py: EnergyAdvisor.invoke`) adds a system context: current local date and time plus
   an 8-day calendar ("Monday = 2026-10-12 (today), Tuesday = 2026-10-13 (tomorrow), ...") and the household
   location (and, in the app, the scheduling window). This context is passed with every turn but not stored.
   The conversation itself is stored per thread by the checkpointer, so the agent sees earlier messages.
   It then runs the LangGraph loop.
3. **LLM decides** to call `find_cheapest_window(energy_kwh=30, power_kw=11, deadline="2026-10-13T07:00")`.
4. **Tool** (`agent/tools.py`) forwards to `optimization/service.py: OptimizerService.cheapest_window`:
   - window = from now (rounded up to the next quarter hour) to the deadline, or 24 hours if none was given;
   - prices for the window from `PriceService.curve` (Energy-Charts → aWATTar → estimate → synthetic, cached);
   - solar surplus per hour from the weather forecast (PV yield minus household base load);
   - `optimization/scheduling.py: find_cheapest_schedule` picks the cheapest slots.
5. **Result** (schedule, cost, solar share, saving versus starting now, price source, estimate flag) goes back
   to the LLM as a tool message; the LLM may call more tools (e.g. `search_energy_tips`) and then answers.
6. **CLI** prints the tool calls and the answer.

If any tool fails, its error becomes a message the LLM can react to; if the whole run fails, the user gets an
apology instead of a stack trace.

---

## 3. Assumptions to verify

These are decisions and numbers that were **not** given in the task and were chosen during development. They
are grouped by how much they affect results. Each item says where it lives and how to change it.

### 3.1 Prices and tariff (affect every EUR figure)

| # | Assumption | Value | Where | Notes |
|---|---|---|---|---|
| P1 | Household price under a dynamic tariff | `(spot / 1000 + surcharges) × (1 + VAT)` | `services/tariff.py` | Assumes the supplier passes the spot price through 1:1 and all other components are a fixed per-kWh amount. Real dynamic tariffs may add a monthly fee or a percentage markup. |
| P2 | Net surcharges (grid fees, levies, electricity tax, concession fee, supplier margin) | 0.19 EUR/kWh | `TariffSettings.surcharges_eur_per_kwh` | Typical value, not taken from a specific Munich tariff. Check against a real offer (e.g. SWM, Tibber, Ostrom). |
| P3 | VAT | 19 % | `TariffSettings.vat_rate` | |
| P4 | Feed-in tariff | 0.078 EUR/kWh | `TariffSettings.feed_in_tariff_eur_per_kwh` | Approximate EEG rate for new small systems; depends on commissioning date and size. |
| P5 | Fixed-tariff price used when the LLM gives no price to `calculate_energy_savings` | 0.35 EUR/kWh | `TariffSettings.fixed_price_eur_per_kwh` | |
| P6 | Day-ahead prices are published around 12:45-13:00 CET | – | `services/prices.py: estimate_note`, knowledge base | Before that, tomorrow is an estimate. |
| P7 | Estimate for unpublished days | weighted average of the same local hour over the previous 7 days, same weekday counted twice | `clients/prices/estimate.py` | Not a forecast; ignores weather and holidays. |
| P8 | Last-resort synthetic price profile | typical German day shape, ±20 % per day, seasonal midday factor | `clients/prices/synthetic.py` | Hand-made shape; only used when every real source and the estimate fail. |
| P9 | Cheap / middle / expensive labels | terciles of the day (8 hours each on a 24-hour day) | `services/prices.py: assign_periods` | Relative to the day, not absolute thresholds. |
| P10 | 15-minute prices shown to the LLM as hourly means | – | `PriceCurve.hourly` | The optimizer still uses the 15-minute slots. |
| P11 | Published prices never change, so the cache never expires | – | `clients/prices/cache.py` | Delete `data/price_cache.sqlite` to refetch. |
| P12 | Price coverage needed to accept a provider's answer | 99 % of the requested interval | `services/prices.py: MIN_COVERAGE` | |

### 3.2 Optimizer (affects every schedule)

| # | Assumption | Value | Where | Notes |
|---|---|---|---|---|
| O1 | Window start | the user's start (or the app's *From*), default **now**, rounded up to the next 15 minutes; a past start becomes now | `optimization/service.py: cheapest_window` | No assumptions about where the car is or when a device may run. |
| O2 | No deadline given | **24 hours after the start**, flagged in the output; the agent asks by when and recomputes when the user answers | `DEFAULT_WINDOW_HOURS` | In the app the deadline defaults to *From* + 24 h and can be changed. |
| O3 | Maximum window | 7 days | `MAX_WINDOW` | Beyond tomorrow, prices are estimates. |
| O4 | EV charging power can be modulated continuously between 0 and 11 kW | – | `scheduling.py: _fill_interruptible` | Real wallboxes have a minimum (about 1.4 kW single-phase, 4.1 kW three-phase). A partly used slot is reported as a lower average power. |
| O5 | Appliances run at constant power for `energy / power` hours | – | `scheduling.py: _run_contiguous` | Real cycles have heating peaks; the total cost is still right if the price is constant within a slot. |
| O6 | Contiguous loads can start only at slot boundaries (every 15 or 60 minutes) | – | `find_cheapest_schedule` | |
| O7 | Typical device powers suggested to the LLM | EV 11, dishwasher 0.5, washing machine 0.4, dryer 0.6, heat-pump boost 2 kW | tool docstring in `agent/tools.py` | The LLM may pick other values. |
| O8 | Own solar power is valued at the feed-in tariff (opportunity cost) | – | `scheduling.py: slot_energy_cost` | With the "solar peak law" new systems get no feed-in payment at negative prices; then the true opportunity cost would be 0. Not modelled. |
| O9 | Solar surplus = forecast PV yield − typical base load | – | `optimization/service.py: solar_surplus` | Ignores other loads running at the same time (heat pump, appliances). |
| O10 | Baseline for "saving" | start immediately at the beginning of the window | `find_cheapest_schedule` | |

### 3.3 Household model (affects the history and all "your usage" answers)

The household is synthetic; only the weather and the prices are real.

| # | Assumption | Value | Where |
|---|---|---|---|
| H1 | PV system | 8 kWp, performance ratio 0.8 | `HouseholdSettings` |
| H2 | PV yield from **horizontal** irradiance (no tilt, no orientation) | `kWp × GHI / 1000 × PR` | `domain/solar.py: pv_energy_kwh` |
| H3 | Heat pump heat demand | 0.15 kW per kelvin below 15 °C, plus hot water 0.8/0.6/0.6/0.5 kWh at 06/07/19/20 h | `HouseholdSettings`, `data/seed.py` |
| H4 | Heat pump efficiency | COP = 3.0 + 0.1 × outdoor °C, limited to 2.0-4.5 | `seed.py: heat_pump_cop` |
| H5 | EV | 17 kWh/100 km, 35 km/day (±50 %), charges at 11 kW when ≥ 15 kWh are missing (85 % chance per day), arrival at 17, 18 or 19 h | `HouseholdSettings`, `seed.py` |
| H6 | Base load | 0.3 kW on average with a household day profile (own approximation, not the official BDEW H0 profile), ±15 % | `domain/household.py` |
| H7 | Appliances | dishwasher 75 %/day, 1.0 kWh at 19-21 h; washing machine 50 %/day, 0.8 kWh at 8-18 h; heat-pump dryer after 60 % of washes, 1.2 kWh | `seed.py` |
| H8 | Cost per device | kWh × retail price of that hour, **before** subtracting own solar | `seed.py`, noted in tool output |
| H9 | CO₂ factor of the German grid | 0.36 kg/kWh; only reduced kWh save CO₂, shifted kWh do not | `HouseholdSettings.co2_kg_per_kwh` |
| H10 | History length | at most 90 days (Open-Meteo's past-weather window) | `seed.py: MAX_DAYS` |

### 3.4 Weather, location and time

| # | Assumption | Where |
|---|---|---|
| W1 | Munich = 48.1374 N, 11.5755 E, time zone Europe/Berlin, bidding zone DE-LU | `LocationSettings` |
| W2 | "Munich", "München", "Muenchen" (and an empty location) mean the household; other places are geocoded and the first Open-Meteo hit is used (ambiguity is reported) | `LocationSettings.matches`, `clients/open_meteo.py` |
| W3 | Sky condition from WMO codes: 0-1 sunny, 2 partly cloudy, 3/fog cloudy, everything else (rain, snow, storms) "rainy" | `domain/weather.py` |
| W4 | Daily condition = most common condition between 08:00 and 17:00 | `services/weather.py` |
| W5 | A live weather answer is accepted if it covers at least 90 % of the requested hours | `services/weather.py: MIN_COVERAGE` |
| W6 | Synthetic fallback weather: Munich monthly mean temperatures (approximate), ±3 K per day, 4.5 K daily swing, sky condition weights per season, clear-sky irradiance from the sun position (Haurwitz model) | `domain/climate.py`, `domain/solar.py` |
| W7 | A weekday that equals today means today only if that time is still ahead, otherwise the next one | system prompt |

### 3.5 Knowledge base content

The 8 documents in `knowledge_base/` were rewritten for German households from general knowledge. **Facts
and numbers to check** before publishing: UBA room-temperature recommendations; "each degree saves about
6 %"; typical household consumption (3,500-4,500 kWh, heat pump +3,000-5,000, EV +2,000-2,500); flow
temperature "2-3 % per kelvin"; §41a EnWG (dynamic tariffs since 2025); §14a EnWG modules 1-3 and the 4.2 kW
limit; wallbox registration (11 kW report, 22 kW approval); the solar peak law (no feed-in payment at negative
prices for systems since February 2025); 800 W balcony systems; PV yield 1,000-1,100 kWh/kWp in southern
Germany; battery cost 500-800 EUR/kWh; CO₂ factor 0.36 kg/kWh.

### 3.6 Agent and evaluation

| # | Assumption | Where |
|---|---|---|
| A1 | Agent model `gpt-4o-mini`, judge `gpt-4.1`, re-ranker `gpt-4o-mini`, embeddings `text-embedding-3-small` | `LLMSettings` |
| A2 | The prompt requires `search_energy_tips` in every answer; the evaluation therefore expects it in every case | `agent/prompts.py`, `cases.yaml` |
| A3 | Tool appropriateness = share of the used tools that were expected, so extra helpful tools lower the score | `evaluation/metrics.py` |
| A4 | Evaluation results are single runs (LLM output varies by about ±0.1) | README |
| A5 | Re-ranker drops passages scored below 3/10 (keeps at least one) | `RAGSettings.min_rerank_score` |
| A6 | Chunking 1,000 characters with 200 overlap | `RAGSettings` |

### 3.7 Repository and publishing

- `seed-db` **drops and recreates** the tables (generated data, no migrations).
- The README section "Origin" is written in the first person, on your behalf: check that it describes
  correctly what the course provided and what you built.
- Licence MIT, copyright holder "tohver".
- Your `.env` points to an OpenAI-compatible proxy (`OPENAI_BASE_URL`); public users will use the standard
  OpenAI endpoint.
- Data attribution: Energy-Charts / SMARD and Open-Meteo are CC BY 4.0 and are credited in the README and in the
  tool output.

---

## 4. Module reference

All code lives in `src/energy_advisor/`. The layers depend in one direction:
`cli → agent → services / optimization → clients / data / domain`. Business logic never imports LangChain,
so it is tested without an LLM.

### Top level

| File | What it does | How / why |
|---|---|---|
| `config.py` | `Settings` and its sections (`LocationSettings`, `TariffSettings`, `PriceSettings`, `HouseholdSettings`, `LLMSettings`, `RAGSettings`, `PathSettings`) | **pydantic-settings**: values from `.env`/environment with validation at startup; nested values via `__` (e.g. `TARIFF__VAT_RATE`). `SecretStr` hides the API key in logs. Unknown time zones are rejected. |
| `timeutils.py` | Time-zone-aware helpers: `make_clock`, `to_utc`, `parse_day`, `parse_local_datetime`, `day_bounds_utc`, `hours_between`, `calendar_lines` | Standard library `zoneinfo`. All arithmetic in UTC because Python does wall-clock arithmetic for local datetimes, which breaks on 23/25-hour DST days. Naive datetimes raise; ruff rule `DTZ` forbids them in the code. |
| `llm.py` | `make_chat_model`, `make_embeddings` | `langchain-openai`; the only place that knows the API key and base URL. |
| `logging_config.py` | `configure_logging` | **rich** log handler; third-party libraries only from WARNING up. |
| `mcp_server.py` | `build_server`: an MCP server (official `mcp` SDK, `MCPServer`) that registers the agent's tool functions as MCP tools, plus `plan_devices` | Type hints become the JSON schemas, docstrings the descriptions; no own logic, so agent, app, CLI and MCP share one code path. Details: [mcp.md](mcp.md). |
| `cli.py` | Commands `seed-db`, `build-index`, `rag-benchmark`, `ask`, `chat`, `mcp`, `eval`, `report` | **Typer** (CLI from type hints) and **rich** (tables, panels, SVG export). Thin: parses options, builds services, calls modules, prints. Expected errors (no API key, no index) become a one-line message and exit code 1. |

### `agent/`: the LangGraph agent

| File | What it does | How / why |
|---|---|---|
| `graph.py` | `build_graph` creates the loop `agent → tools → agent → … → END`; `EnergyAdvisor` wraps it (`invoke(question, context, thread_id)`, `history`, `from_settings`); `build_system_context` adds date, calendar and location | **LangGraph** `StateGraph` with `ToolNode(handle_tool_errors=True)` and `tools_condition` routing, compiled with a **checkpointer** (`InMemorySaver`) that stores each conversation under a thread id. The system context travels in the run config (`configurable.system_context`) and is added in the agent node, so it is fresh every turn and never stored. A recursion limit (25) stops endless loops; any failure returns a normal-shaped apology result. The clock is injected, so tests can freeze time. |
| `tools.py` | `build_tools(services, optimizer=True)` returns the 9 tools | LangChain `@tool` functions created inside a factory (closures over the injected services) instead of module globals, so tests can pass a temporary database and fake HTTP. Each tool validates input and returns `{"error": ...}` on expected problems, so the LLM can recover. Docstrings are written for the LLM: they decide which tool it calls. |
| `prompts.py` | `build_system_prompt(optimizer)` and `SYSTEM_PROMPT` | Built from parts so the ablation run can remove the optimizer sections while keeping everything else identical. |
| `trace.py` | `final_answer`, `tool_calls`, `tool_outputs`, `tools_with_errors` | Reads the message list of a result; used by the CLI and the evaluation. |

### `optimization/`: deterministic scheduling

| File | What it does | How / why |
|---|---|---|
| `scheduling.py` | `find_cheapest_schedule(slots, energy, power, earliest, deadline, feed_in, interruptible, use_solar)` returns a `Schedule` of `Block`s with cost, solar share and the baseline cost | **Plain Python, no LLM and no pandas**, because the result must be deterministic, exact and unit-testable. *Contiguous* loads: every possible start time is evaluated (`_run_contiguous`), the cheapest wins. *Interruptible* loads: each slot offers a "solar" chunk (priced at the feed-in tariff) and a "grid" chunk (priced at the retail price); the cheapest chunks are filled first (`_fill_interruptible`). This greedy approach is optimal because cost is linear in energy. `slot_energy_cost` takes solar first unless grid power is cheaper (deeply negative prices). |
| `service.py` | `OptimizerService.window` (window, price slots and solar surplus for an interval), `plan_devices` (several devices in one window, sharing the solar surplus; used by the app's planner), `DEVICE_PRESETS` (car 20 kWh at 11 kW, dishwasher 1.0 kWh at 0.5 kW, washing machine 0.8 kWh at 0.4 kW); `cheapest_window` (tool backend): builds the window, fetches the price curve and the solar surplus, calls the scheduler, formats the output in local time; `pv_forecast` (tool backend for `estimate_pv_generation`); `solar_surplus` | Connects the pure scheduler to the services. Window rules: start = now, deadline = the user's or now + 24 h (flagged), maximum 7 days. |

### `services/`: business logic around the data

| File | What it does | How / why |
|---|---|---|
| `container.py` | `build_services(settings, http, clock, price_providers)` → `Services`; `build_price_providers` builds the provider chain | **Dependency injection** in one place: nothing touches the network or needs an API key at construction time (LLM clients are created on first use). Tests pass an offline HTTP client, a fixed clock and fake providers. |
| `prices.py` | `PriceService`: `spot_prices` (provider chain + coverage check), `curve` (multi-day ranges: one call first, then day by day so today stays real and only tomorrow is estimated), `hourly_retail` (for the history), `get_prices` (tool output); `assign_periods`, `price_output`, `merge_curves` | **Strategy pattern with fallback chain**: providers are interchangeable objects tried in order. Every result carries `source`, `is_estimate`, notes and attribution so the LLM can be honest about estimates. |
| `tariff.py` | `retail_price`, `tariff_assumptions` | One formula, one place; the assumptions travel with every price answer. |
| `weather.py` | `WeatherService`: `resolve` (household coordinates or geocoding), `series` (live or synthetic hourly weather for local days), `forecast` (tool output), `history` (for seeding) | Live data must cover ≥ 90 % of the requested hours, otherwise the synthetic fallback is used and the output says so. |
| `savings.py` | `calculate_savings` | Pure function; supports "use less" and "shift to a cheaper time"; CO₂ only for reduced kWh. |

### `clients/`: external APIs

| File | What it does | How / why |
|---|---|---|
| `open_meteo.py` | `OpenMeteoClient.geocode` (cached per name, flags ambiguous names), `OpenMeteoClient.weather` (hourly + current), `parse_hourly`, `parse_current` | **httpx** with a timeout and a descriptive User-Agent. Requests use `timezone=UTC` because local hours are ambiguous on DST days. Radiation is labelled with the *end* of its hour and is shifted to the hour it belongs to. One endpoint serves both the past (~3 months) and the forecast (+16 days). |
| `prices/base.py` | `PriceProvider` | `typing.Protocol` (structural interface): any class with `name`, `is_estimate`, `attribution` and `fetch(start, end)` qualifies, without inheritance. |
| `prices/energy_charts.py` | `EnergyChartsProvider`, `parse` | 15-minute day-ahead prices for DE-LU from Fraunhofer ISE. Response validated with a **pydantic** model; the slot length is derived from the timestamps; the API's inclusive `end` is compensated. |
| `prices/awattar.py` | `AwattarProvider`, `parse` | Hourly prices, millisecond timestamps, pydantic validation. Fallback when Energy-Charts rate-limits (HTTP 429) or fails. |
| `prices/cache.py` | `CachedPriceProvider` | **Decorator pattern** around a real provider, storing every fetched slot in SQLite (`data/price_cache.sqlite`, standard library `sqlite3`). The schema is ensured on every connection, so a deleted cache file is simply rebuilt. |
| `prices/estimate.py` | `EstimatePriceProvider` | Builds an hourly estimate from the real providers' recent history (see P7). |
| `prices/synthetic.py` | `SyntheticPriceProvider` | Deterministic typical day shape, last resort (see P8). |

### `domain/`: plain data types and physics

| File | What it does | How / why |
|---|---|---|
| `weather.py` | `HourlyWeather`, `condition_from_wmo`, solar factors per sky condition | Frozen dataclasses: immutable, cheap, no framework. |
| `prices.py` | `SpotPrice`, `PriceSlot`, `PriceUnavailableError` | Same. All times are aware UTC. |
| `solar.py` | `solar_elevation_deg` (sun position from date, time and coordinates), `clear_sky_ghi` (Haurwitz model), `hourly_clear_sky_ghi`, `pv_energy_kwh` | Standard astronomical formulas in plain `math`, accurate to about 1°; gives correct day length for Munich in every season. |
| `climate.py` | `synthetic_weather`, `seed_for` | Climate-based fallback weather. `seed_for` makes a stable seed from values (Python's `hash()` changes between runs). |
| `household.py` | `BASE_LOAD_PROFILE`, `base_load_kw` | Shared by the history generator and the solar-surplus estimate. |

### `data/`: database and the household history

| File | What it does | How / why |
|---|---|---|
| `models.py` | `EnergyUsage`, `SolarGeneration`, `UTCDateTime` | **SQLAlchemy 2.0** typed models (`Mapped[...]`). `UTCDateTime` is a `TypeDecorator` that accepts only aware datetimes, stores naive UTC (SQLite has no time zones) and returns aware UTC. |
| `repository.py` | `EnergyRepository`: `recreate_tables`, `add_all` (one transaction), `usage_between`, `generation_between` (half-open `[start, end)`) | Repository pattern: the only place with SQL. Half-open ranges fixed regression F-07 (midnight rows leaking into the next day). |
| `aggregation.py` | `summarize_usage`, `summarize_generation`, `energy_balance`, `summarize_recent` | Turns hourly rows into compact summaries (per device, per local day, average by hour), because thousands of raw rows overflowed the LLM context (F-04). `energy_balance` matches load and PV hour by hour: self-consumed, exported, imported, self-consumption rate, autarky. |
| `seed.py` | `generate_household`, `build_history`, `seed_database`, `heat_pump_cop` | Seeded `random.Random` for reproducibility; inputs are real weather and real prices (or their fallbacks); see H1-H10. |

### `rag/`: knowledge base retrieval

| File | What it does | How / why |
|---|---|---|
| `ingest.py` | `load_documents`, `split_documents`, `build_index`, `make_chroma_client` | Files read directly (no extra loader library); `RecursiveCharacterTextSplitter` (1,000/200); **Chroma** persistent client with telemetry off; `--rebuild` deletes the collection first, so chunks are never duplicated (F-11). |
| `retrieval.py` | `tokenize`, `rrf_fuse` | **Reciprocal Rank Fusion**: each passage scores `Σ 1/(60 + rank)` over the vector and keyword rankings, so documents ranked well by either method rise. Pure function, unit-tested. |
| `rerank.py` | `LLMReranker`, `apply_scores`, `build_prompt` | A chat model with **structured output** (pydantic schema) scores candidates 0-10. `apply_scores` tolerates missing scores (F-06) and drops passages below 3. |
| `knowledge_base.py` | `KnowledgeBase`: `vector_search`, `bm25_search`, `hybrid_search`, `reranked_search`, `search` (tool output) | Owns the lazily loaded vector store and **BM25** index (`rank-bm25`, keyword ranking). Vectors find paraphrases, BM25 finds exact terms. If re-ranking fails, the fused order is used. |
| `benchmark.py` | `run_benchmark`, `rank_of`, `score` | hit@k and MRR on 12 labelled queries. |

### `evaluation/`: measuring the agent

| File | What it does | How / why |
|---|---|---|
| `cases.yaml` + `cases.py` | 15 cases (question, expected tools, expected answer); `load_cases` validates them | YAML so cases are data, not code; pydantic validation; unique ids enforced. |
| `judge.py` | `LLMJudge.evaluate` | **LLM-as-judge** with structured output; gets the question, the expected answer, the tool outputs as ground truth and the same calendar as the agent. Never raises: a failed judgement scores 0 with the reason. |
| `metrics.py` | `evaluate_tool_usage` | Deterministic: appropriateness, completeness, missing and extra tools, tools that returned errors. |
| `runner.py` | `run_evaluation` → list of `CaseResult` | Logs every tool call and argument per case. |
| `report.py` | `generate_report`, `render_markdown`, `save_report` | Averages, strengths/weaknesses, recommendations; Markdown + JSON files. |

### `ui/` and `app/`: the web app

| File | What it does | How / why |
|---|---|---|
| `ui/planner.py` | `result_markdown` (the result screen: slot table, cost analysis, solar contribution per device), `hourly_rows`, `window_bars`, `summary_rows`, `day_window` | Turns a `DevicePlan` into the result screen and chart rows; pure functions, unit-tested. |
| `ui/charts.py` | `plan_chart` (three stacked panels on one time axis: when the devices run, electricity price, solar surplus; each with a title and a legend), `price_panel`, `solar_panel`, `styled` | **Altair**; colours and monospace font of the README terminal screen; no overlays, so each element is readable on its own. |
| `ui/window.py` | `SchedulingWindow`, `default_window` (now to now + 24 h), `default_deadline`, `window_from_inputs`, `window_context` (text for the agent) | Pure functions so the page stays thin and the rules are unit-tested; 24 hours are real hours (also across a clock change). |
| `app/streamlit_app.py` | Sidebar with all settings (From, Deadline, devices, OK, New conversation); views Best times (result screen + chart), Chat (example questions, tool calls per answer), Prices & solar | `st.cache_data` keeps a computed plan for 15 minutes per window and device set, so switching views does not refetch. | **Altair** charts (bundled with Streamlit): planner = price line, solar bars and device bands on top, a device timeline below with a shared time axis. `st.segmented_control` switches views, `st.pills` selects devices. | **Streamlit** (`st.chat_message`, `st.chat_input`, `st.session_state`, `st.cache_resource` for one agent per server process). Outside the package because it is an application, not library code; installed with the `app` extra. |

### `viz/`: charts

| File | What it does | How / why |
|---|---|---|
| `charts.py` | Five chart functions returning a matplotlib `Figure`, `generate_charts` (data gathering + PNG files), `save_evaluation_heatmap` | **pandas** for grouping and pivoting the history (here a table library is the right tool), **matplotlib** with the `Agg` backend (files only). Optional `viz` extra, so the agent does not need these libraries. |

### Other folders

| Folder / file | Content |
|---|---|
| `knowledge_base/` | The 8 tip documents used by RAG. |
| `tests/unit/` | Offline tests (see below). `tests/integration/` real API tests, marked. `tests/fixtures/` recorded real API responses. |
| `docs/` | README images, the evaluation report. |
| `data/` (git-ignored) | `energy_data.db`, `vectorstore/`, `price_cache.sqlite`. |
| `reports/` (git-ignored) | Output of `eval` and `report`. |
| `.github/workflows/ci.yml` | GitHub Actions: lock check, ruff, format, mypy, tests with coverage ≥ 85 %. |
| `.streamlit/config.toml` | Theme of the web app: dark background `#292929`, text `#c5c8c6`, green accents, monospace font (the palette of `docs/img/ask-demo.svg`). |
| `.pre-commit-config.yaml` | Hooks before every commit: whitespace, YAML/TOML checks, large files, private keys, ruff, mypy, gitleaks (secret scanner). |
| `PLAN.md` / `CHANGELOG.md` | Development plan with implementation notes per phase / release history. |

---

## 5. Tests

```bash
uv run pytest                  # 209 offline tests, a few seconds
uv run pytest -m integration   # real Open-Meteo, Energy-Charts, aWATTar
uv run pytest -m llm           # real LLM and embeddings (costs a little)
```

| File | Covers |
|---|---|
| `test_config.py` | Settings defaults, environment overrides, secret masking, time-zone validation, location aliases, base URL reaching the clients (F-02). |
| `test_time_and_solar.py` | 23/24/25-hour days, UTC bounds, strict date parsing, the calendar, the UTC column type, sun position in June and December, synthetic day length. |
| `test_weather.py` | Recorded Munich forecast, radiation shift, no geocoding for the household, aliases, ambiguous "Frankfurt", fallbacks on errors and incomplete data. |
| `test_prices.py` | Retail formula, provider chain, incomplete data, estimates flagged, 15-minute → hourly, DST days, terciles, savings (F-05). |
| `test_price_providers.py` | Recorded Energy-Charts and aWATTar responses, HTTP errors, cache hits/misses/rebuild, estimate weighting, multi-day split, default chain fallbacks and attribution. |
| `test_data.py` | Heat pump vs. temperature, PV rows and costs, seeding (local days, reproducibility, re-seeding F-10, DST day), tool outputs (F-04, F-07, F-08), energy balance consistency. |
| `test_optimization.py` | Contiguous and interruptible schedules, merged blocks, solar valuation, negative prices, validation errors, tool outputs (overnight EV, dishwasher, 24 h default, past start, night-time appliance). |
| `test_rag.py` | Documents and topics, index without duplicates (F-03, F-11), RRF, BM25, re-ranker behaviour (F-06), fallbacks, benchmark maths. |
| `test_agent.py` | The 9 tools, graph nodes, system context, tool call → answer, tool error → message, failure → apology, ablation, conversation memory (follow-up answer, isolated threads, context not stored). Uses a scripted fake LLM. |
| `test_evaluation.py` | Cases file, tool metrics, judge clamping and failure, runner, report and files. |
| `test_cli.py` | Help, `seed-db` offline, clean errors without an API key (also for `chat`). |
| `test_ui_window.py` | Default window now to now + 24 h (also across a clock change), overriding the deadline, the text for the agent. |
| `test_app.py` | The Streamlit app rendered headless and offline (`AppTest`): sidebar layout (window, devices, OK first, New conversation last, removed texts), deadline following the start, result screen after OK, chat welcome and example questions, "New conversation" opens the chat, prices view without a button, error without an API key. |
| `test_mcp.py` | MCP server: ten tools with schemas and read-only hints, calls through a real MCP client (in memory), readable errors, a stdio round trip with the server as a separate process. |
| `test_planner.py` | Devices share the solar surplus (no double counting), user order kept, a device that does not fit gets an error, chart/table rows, day windows. |

How the tests stay offline and deterministic: HTTP calls are answered from recorded JSON files (**respx**) or
refused (an `httpx.MockTransport` that raises), the LLM is a scripted fake (`GenericFakeChatModel` subclass),
embeddings are `DeterministicFakeEmbedding`, the clock is fixed at 2026-10-02 09:30 Europe/Berlin, and every test
gets its own temporary database and settings.
