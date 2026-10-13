# The MCP server

The project includes an **MCP server** that gives AI assistants such as Claude Desktop, Claude Code or VS Code
(Copilot agent mode) access to the energy advisor's tools: real German electricity prices, the solar forecast,
the optimizer that finds the cheapest time to run devices, and the household's history. You ask your
assistant "When should I charge my car before 7:00 tomorrow?", and it calls this server to get the answer.

Contents:

1. [MCP in five minutes](#1-mcp-in-five-minutes)
2. [What this server offers](#2-what-this-server-offers)
3. [How it works inside](#3-how-it-works-inside)
4. [Setup](#4-setup)
5. [Connecting a client](#5-connecting-a-client)
6. [Example questions](#6-example-questions)
7. [Troubleshooting](#7-troubleshooting)
8. [Privacy and security](#8-privacy-and-security)
9. [Tests](#9-tests)

---

## 1. MCP in five minutes

**MCP (Model Context Protocol)** is an open standard for connecting AI assistants to tools and data. It
replaces one-off integrations ("a plugin for Claude", "a plugin for VS Code") with one protocol: write a
server once and every MCP-capable application can use it.

Three roles:

| Role | What it is | Here |
|---|---|---|
| **Host** | The application you talk to | Claude Desktop, Claude Code, VS Code |
| **Client** | The part of the host that speaks MCP to one server | built into the host |
| **Server** | A program that offers tools (and optionally data and prompt templates) | `energy-advisor mcp` |

What happens in a conversation:

1. When the host starts, it launches the server (or connects to it) and asks: *which tools do you have?*
   The server answers with each tool's **name**, a **description** written for the model, and a
   **JSON schema** of its parameters (names, types, which are required).
2. You ask a question. The host's language model sees the tool list and decides to call a tool, for
   example `find_cheapest_window` with `{"energy_kwh": 30, "power_kw": 11, "deadline": "2026-10-14T07:00"}`.
3. The host sends this call to the server. The server runs the tool and returns the result (JSON or text).
4. The model reads the result and writes the answer, or calls more tools first.

Messages are **JSON-RPC** (small JSON request/response objects). They travel over a **transport**:

- **stdio** (default, local): the host starts the server as a child process and the two talk through the
  process's standard input and output. Nothing listens on the network. This is how desktop apps usually run
  MCP servers.
- **Streamable HTTP**: the server runs as a small web service (here `http://127.0.0.1:8000/mcp`) and clients
  connect by URL. Useful when the client cannot start processes, or to share one running server.

The model never runs code on your machine directly. It can only call the tools the server offers, with
parameters that must match the schema.

---

## 2. What this server offers

Server name: `energy-advisor`. Ten tools, all **read-only**: they never change your data or devices. Times are
local (Europe/Berlin) as `YYYY-MM-DDTHH:MM`, dates as `YYYY-MM-DD`.

| Tool | What it answers | Main parameters | Needs |
|---|---|---|---|
| `find_cheapest_window` | When to run **one** device: exact time slots, cost, solar share, saving versus starting immediately | `energy_kwh`, `power_kw`, `start` (default now), `deadline` (default start + 24 h), `interruptible`, `use_solar` | internet |
| `plan_devices` | Best times for **several** devices together (car, dishwasher, washing machine) sharing the solar surplus. Returns the same result screen as the web app, as Markdown text | `devices` (`ev`, `dishwasher`, `washing_machine`), `start`, `deadline`, `ev_kwh` | internet |
| `get_electricity_prices` | Hourly household price for a day, cheapest and most expensive hours, whether prices are estimates | `date` | internet |
| `estimate_pv_generation` | Expected solar yield and surplus per hour, best solar hours, for 1-3 days | `days` | internet |
| `get_weather_forecast` | Hourly temperature, sky condition, solar irradiance | `location` (empty = Munich), `days` | internet |
| `query_energy_usage` | The household's past consumption and cost per device and day | `start_date`, `end_date`, `device_type` | history (`seed-db`) |
| `query_solar_generation` | Past PV production and the energy balance (self-consumed, exported, from the grid) | `start_date`, `end_date` | history (`seed-db`) |
| `get_recent_energy_summary` | Consumption and production of the last N hours | `hours` | history (`seed-db`) |
| `calculate_energy_savings` | Savings in EUR, kWh and CO₂ for "use less" or "shift to a cheaper hour" | current/new kWh, current/new price | – |
| `search_energy_tips` | Tips from the knowledge base for German households, with the source file | `query`, `max_results` | `OPENAI_API_KEY` + `build-index` |

The server also sends the client short **instructions** (MCP "server instructions"): the household, the time
format, "use the scheduling tools instead of working out schedules yourself", and "say when prices are
estimates".

Every tool is marked *read-only*. The tools that call external APIs are also marked *open world*. Hosts can
show these hints, for example to decide whether to ask before a call.

---

## 3. How it works inside

### 3.1 One code path for everything

The server (`src/energy_advisor/mcp_server.py`) contains no business logic of its own:

```
MCP client ──JSON-RPC──▶ MCPServer ──▶ the agent's tool functions (agent/tools.py)
                                         └─▶ services: PriceService, WeatherService, OptimizerService,
                                             EnergyRepository, KnowledgeBase (the same as in CLI and app)
```

- At start, `build_server()` builds the same **services** as the CLI and the web app (`build_services`) and the
  same **tools** as the LangGraph agent (`build_tools`). Building them touches no network and needs no API
  key, so the server starts in about a second.
- Each agent tool is registered with `server.add_tool(...)` as it is. The MCP SDK reads the function's **type
  hints** and turns them into the tool's JSON schema (`energy_kwh: float` becomes a required number,
  `deadline: str | None = None` an optional string), and uses the tool's **docstring** as its description. The
  descriptions the model sees are therefore exactly the ones the project's own agent sees.
- One extra tool, `plan_devices`, calls `OptimizerService.plan_devices` and formats the result with the same
  function as the web app's result screen (`ui/planner.py: result_markdown`).

A fix in the optimizer or the price service therefore immediately applies to the CLI, the web app, the
LangGraph agent and the MCP server alike.

### 3.2 A call, step by step

Example: your assistant calls `find_cheapest_window(energy_kwh=30, power_kw=11, deadline="2026-10-14T07:00")`.

1. The SDK checks the arguments against the schema (a text where a number is expected is rejected before any
   of our code runs).
2. The tool function parses the deadline as local time and calls the optimizer.
3. The optimizer fetches the prices for the window. Real prices come from the cache
   (`data/price_cache.sqlite`) or from Energy-Charts / aWATTar; missing days are estimated and flagged.
4. It fetches the solar forecast from Open-Meteo, computes the surplus, and finds the cheapest slots
   (see [calculations.md](calculations.md), section 5).
5. The result (schedule, cost, saving, price source, `is_estimate`, notes) goes back as JSON; the model turns
   it into an answer.

### 3.3 Data and state

- The server keeps no conversation state; each call stands alone. The conversation lives in the host.
- It shares the files of the project with the CLI and the app: the price cache, the household history
  (`data/energy_data.db`, created by `seed-db`) and the knowledge-base index (`data/vectorstore/`, created by
  `build-index`).
- Settings come from the project's `.env` file and environment variables, exactly as for the CLI (location,
  tariff, household, paths).

### 3.4 Errors

- Wrong argument **types** or unknown values (e.g. `devices: ["toaster"]`) are rejected by the SDK; the client
  receives an error result (`isError: true`) with the reason.
- Expected problems inside a tool (an invalid date, a deadline in the past, not enough time to charge 60 kWh
  before 07:00) come back as a normal result with an `"error"` field and a readable message, the same way the
  agent receives them, so the model can explain or retry with better parameters.

### 3.5 Why stdout must stay clean

With the stdio transport, the server's standard output **is** the protocol channel. Any stray text there
(a `print`, a log line) would corrupt the messages. The `mcp` command therefore sends all logging to standard
error, and only warnings and errors are logged.

---

## 4. Setup

1. **Install** the project with the MCP extra (once):

   ```bash
   uv sync --extra mcp
   ```

2. **Optional data** (only for the tools that need it):

   ```bash
   uv run energy-advisor seed-db      # household history: usage, solar, energy balance tools
   uv run energy-advisor build-index  # knowledge base: search_energy_tips (also needs OPENAI_API_KEY in .env)
   ```

   Prices, weather, solar forecast and scheduling work without any of this and without an API key.

3. **Try it** in a terminal:

   ```bash
   uv run --extra mcp energy-advisor mcp
   ```

   It prints nothing and waits: it is waiting for a client on its standard input. Stop it with Ctrl+C. (On
   Windows, use PowerShell; Git Bash does not pass Ctrl+C to native programs.)

4. **Find two values** for the client configuration:
   - the **absolute path of the project folder**, e.g. `C:/notebooks/portfolio_github/energy_optimizer_langgraph`;
   - the **path of uv**, if your client is a desktop app that may not see your terminal's PATH:
     `where uv` (Windows) or `which uv` (macOS/Linux), e.g. `C:/Users/<you>/.local/bin/uv.exe`.

The command every client runs is:

```
uv --directory <project folder> run --extra mcp energy-advisor mcp
```

`--directory` makes uv use the project's environment and `.env`, regardless of where the client starts it.

---

## 5. Connecting a client

### 5.1 MCP Inspector (to look at the server first)

The official debugging UI shows the tools, their schemas and lets you call them by hand. Requires Node.js.

```bash
npx @modelcontextprotocol/inspector uv --directory C:/notebooks/portfolio_github/energy_optimizer_langgraph run --extra mcp energy-advisor mcp
```

It opens a browser page: press **Connect**, then **Tools → List Tools**, choose `plan_devices`, enter
`["ev", "dishwasher"]` and **Run Tool**.

### 5.2 Claude Desktop

1. Open the configuration file (Claude Desktop: *Settings → Developer → Edit Config*):
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Add the server (use your paths; forward slashes work on Windows too):

   ```json
   {
     "mcpServers": {
       "energy-advisor": {
         "command": "C:/Users/<you>/.local/bin/uv.exe",
         "args": [
           "--directory", "C:/notebooks/portfolio_github/energy_optimizer_langgraph",
           "run", "--extra", "mcp", "energy-advisor", "mcp"
         ]
       }
     }
   }
   ```

3. Restart Claude Desktop completely (also from the system tray). The tools appear under the tools/connectors
   icon in the chat input.

### 5.3 Claude Code

```bash
claude mcp add energy-advisor -- uv --directory C:/notebooks/portfolio_github/energy_optimizer_langgraph run --extra mcp energy-advisor mcp
```

By default the server is added for the current project only; add `--scope user` to have it everywhere. Inside
Claude Code, `/mcp` shows the connected servers and their status.

### 5.4 VS Code (Copilot agent mode)

Create `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "energy-advisor": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory", "C:/notebooks/portfolio_github/energy_optimizer_langgraph",
        "run", "--extra", "mcp", "energy-advisor", "mcp"
      ]
    }
  }
}
```

VS Code shows a *Start* link above the server entry. In the Chat view, switch to **Agent** mode; the tools are
listed under the tools button.

### 5.5 HTTP instead of stdio

```bash
uv run --extra mcp energy-advisor mcp --transport http --port 8000
```

The server listens on `http://127.0.0.1:8000/mcp` (only on your own machine). Clients that support
Streamable HTTP connect with that URL, e.g. VS Code: `{"servers": {"energy-advisor": {"type": "http", "url":
"http://127.0.0.1:8000/mcp"}}}`. The server must be running before the client connects.

---

## 6. Example questions

Ask your assistant in normal language; it picks the tools:

- "When should I charge my EV with 30 kWh before 7:00 tomorrow, and what will it cost?"
- "Plan the car (25 kWh), the dishwasher and the washing machine for tomorrow."
- "What does electricity cost tomorrow at 19:00, and are these real prices or an estimate?"
- "Which of the next three days is best for charging from solar?"
- "How much of my solar power did I use myself last week?" (needs `seed-db`)
- "Give me tips for running my heat pump in winter." (needs `build-index` and an OpenAI key)

---

## 7. Troubleshooting

| Problem | Cause and fix |
|---|---|
| The server does not appear in the client | Wrong path or `uv` not found by the desktop app: use absolute paths for both, restart the client completely. Claude Desktop writes logs to `%APPDATA%\Claude\logs\mcp-server-energy-advisor.log` (Windows) or `~/Library/Logs/Claude/` (macOS). |
| "The MCP server needs the 'mcp' extra" | Run `uv sync --extra mcp` in the project folder, or keep `--extra mcp` in the client's command. |
| Running the command in a terminal "hangs" | That is normal for stdio: the server waits for a client. Use the Inspector to try it by hand. |
| History tools return an error or empty data | Run `uv run energy-advisor seed-db`. The history ends yesterday; re-run it after a few days. |
| `search_energy_tips` fails | Needs `OPENAI_API_KEY` in `.env` and `uv run energy-advisor build-index`. |
| Results say `is_estimate: true` | Tomorrow's day-ahead prices are published around 13:00; before that, prices are estimated from recent days. |
| Prices come from aWATTar or are estimated although it is afternoon | Energy-Charts may rate-limit (HTTP 429); the server falls back automatically. Published prices are cached, so later calls are fast. |
| Times look one hour off | All times are Munich local time (Europe/Berlin), including the summer/winter time change. |

---

## 8. Privacy and security

- **Read-only:** no tool changes files, settings or devices. The server only writes the price cache.
- **Local:** with stdio nothing listens on the network. In HTTP mode the server binds to `127.0.0.1` (your
  machine only) and has no authentication, so do not expose it to other machines.
- **What leaves your machine:** requests to the public APIs Energy-Charts, aWATTar and Open-Meteo (no personal
  data, only dates and the configured coordinates); embeddings and re-ranking requests to OpenAI when
  `search_energy_tips` is used. The **results** of tool calls are sent to the language model of the host you
  use (e.g. Anthropic for Claude Desktop), like everything else in that conversation.
- **Secrets:** the OpenAI key stays in `.env` on your machine; it is never part of a tool result.

---

## 9. Tests

`tests/unit/test_mcp.py` (part of `uv run pytest`, offline):

- the server lists all ten tools with descriptions, schemas and read-only hints;
- prices, scheduling and `plan_devices` work through a real MCP client connected in memory;
- invalid input comes back as a readable error;
- the server starts as a separate process and answers over stdio, exactly as a desktop client uses it.
