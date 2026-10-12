# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Conversation memory: LangGraph checkpointer with one thread per conversation; the system context is sent
  every turn but not stored. New CLI command `chat`.
- Streamlit web app (`app/streamlit_app.py`, extra `app`) with three views: chat (example questions, time
  window in the sidebar: From = now, Deadline = From + 24 h by default), **planner** (day + devices + OK ->
  best windows on a chart and in a table) and prices & solar.
- Multi-device planner (`OptimizerService.plan_devices`): devices share the solar surplus, so it is not counted
  twice.

### Changed
- Scheduling always searches from now until the user's deadline, for every device. Without a deadline the agent
  plans for the next 24 hours, says so and asks by when it is needed. Removed the assumption that an EV is only
  at home from 18:00.
- `find_cheapest_window` takes an optional `start` (default now) and `deadline` (default start + 24 h); the
  "after dinner = 20:00" prompt rule was removed.

## [1.0.0] - 2026-10-12

### Added
- Deterministic optimizer tools: `find_cheapest_window` (contiguous and interruptible loads, solar surplus
  valued at the feed-in tariff, saving versus starting immediately) and `estimate_pv_generation`.
- Real German day-ahead prices: Energy-Charts (15-minute) and aWATTar (hourly) providers, disk cache,
  7-day estimate for unpublished days, synthetic last-resort profile, retail tariff conversion.
- Munich household: coordinates, Europe/Berlin time zone, EUR, PV, heat pump with temperature-dependent COP,
  EV sessions, appliance cycles; history generated from real past weather and prices.
- Energy balance (self-consumption, export, grid import, autarky) in the solar history tool.
- Knowledge base rewritten for German households; 15 evaluation cases including negative prices and
  unpublished prices.
- mypy (strict), SVG export of answers, evaluation comparison in the README.

### Changed
- All time handling is time-zone aware (ruff `DTZ`), storage is UTC, DST days are tested.
- Open-Meteo is queried in UTC; radiation values are aligned with the hour they belong to.

## [0.2.0] - 2026-10-12

### Added
- Installable package (`src/energy_advisor`), typed `Settings`, dependency injection, SQLAlchemy 2.0.
- CLI replacing the notebooks: `seed-db`, `build-index`, `rag-benchmark`, `ask`, `eval`, `report`.
- Offline test suite with recorded HTTP responses, a fake LLM and fake embeddings; ruff, pre-commit, CI.

## [0.1.0]

- Course-project version: Jupyter notebooks, San Francisco, mock time-of-use tariff.
