"""Charts for the README and reports, written as PNG files (needs the ``viz`` extra).

Every chart is a function that takes plain data and returns a matplotlib ``Figure``;
``generate_charts`` gathers the data from the services and writes the files.
"""

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # render to files, never open a window

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from energy_advisor.data.aggregation import energy_balance
from energy_advisor.services.container import Services
from energy_advisor.services.prices import assign_periods
from energy_advisor.services.savings import calculate_savings
from energy_advisor.timeutils import day_bounds_utc, floor_hour

logger = logging.getLogger(__name__)

COLORS = {
    "EV": "#2a78d6",
    "heat_pump": "#eb6834",
    "appliance": "#1baf7a",
    "base_load": "#8a8984",
    "solar": "#eda100",
    "accent": "#2a78d6",
}
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "font.size": 10,
    "lines.linewidth": 2,
    "legend.frameon": False,
}


# ----------------------------------------------------------------------------- charts
def hourly_profile_chart(profile: pd.DataFrame, solar: pd.Series) -> Figure:
    """Average consumption per local hour for each device type, with the average solar yield behind it."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(
        solar.index, solar.values, color=COLORS["solar"], alpha=0.25, lw=0, step="mid", label="solar yield"
    )
    for device in profile.columns:
        ax.plot(
            profile.index, profile[device], color=COLORS.get(device, TEXT_2), label=device.replace("_", " ")
        )
    ax.set_title("Average hour of the day: consumption by device vs. solar yield")
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("kWh per hour")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    return fig


def daily_usage_vs_solar_chart(per_device: pd.DataFrame, solar_daily: pd.Series) -> Figure:
    """Stacked daily consumption by device (top) and daily solar generation (bottom), shared x-axis."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    bottom = None
    for device in per_device.columns:
        ax1.bar(
            per_device.index,
            per_device[device],
            bottom=bottom,
            width=0.8,
            color=COLORS.get(device, TEXT_2),
            label=device.replace("_", " "),
            edgecolor=SURFACE,
            linewidth=1,
        )
        bottom = per_device[device] if bottom is None else bottom + per_device[device]
    ax1.set_title("Daily consumption by device type")
    ax1.set_ylabel("kWh per day")
    ax1.set_ylim(0, per_device.sum(axis=1).max() * 1.2)  # headroom so the legend never covers bars
    ax1.legend(ncol=4, loc="upper left")

    ax2.bar(
        solar_daily.index,
        solar_daily.values,
        width=0.8,
        color=COLORS["solar"],
        edgecolor=SURFACE,
        linewidth=1,
    )
    ax2.set_title("Daily solar generation")
    ax2.set_ylabel("kWh per day")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def source_label(price_source: str, is_estimate: bool) -> str:
    """Short label for chart titles; the full source belongs in a caption."""
    if not is_estimate:
        return "day-ahead market"
    return "partly estimated" if "+" in price_source else "estimated"


def good_sun_runs(data: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Runs of consecutive hours with good sun (>= 50% of the window's peak irradiance) outside the most
    expensive third of the prices: the best hours for flexible loads. Returns (first hour, last hour) pairs."""
    if data.empty or data["solar_irradiance"].max() <= 0:
        return []
    good = (data["solar_irradiance"] >= 0.5 * data["solar_irradiance"].max()) & (data["period"] != "peak")
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for time, ok in good.items():
        if not ok:
            continue
        if runs and time - runs[-1][1] <= pd.Timedelta(hours=1):
            runs[-1] = (runs[-1][0], time)
        else:
            runs.append((time, time))
    return runs


def optimization_window_chart(data: pd.DataFrame, title: str, price_source: str) -> Figure:
    """Solar irradiance forecast (top) and household price (bottom) over a time window, with the best solar
    hours highlighted and the cheapest and most expensive hour labelled.

    ``data`` is indexed by the local start of each hour and has the columns ``solar_irradiance`` (W/m2),
    ``rate`` (EUR/kWh) and ``period`` (off_peak / mid_peak / peak).
    """
    half = pd.Timedelta(minutes=30)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = data.index + half  # centre of each hour
    ax1.fill_between(x, data["solar_irradiance"], color=COLORS["solar"], alpha=0.35, lw=0, step="mid")
    ax1.step(x, data["solar_irradiance"], where="mid", color=COLORS["solar"])
    ax1.set_title(title)
    ax1.set_ylabel("W/m²")

    ax2.step(x, data["rate"], where="mid", color=COLORS["accent"])
    for time, label in ((data["rate"].idxmin(), "cheapest"), (data["rate"].idxmax(), "most expensive")):
        ax2.annotate(
            f"{label}\n{data.loc[time, 'rate']:.3f} €",
            (pd.Timestamp(time) + half, data.loc[time, "rate"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=TEXT,
            fontsize=9,
        )
    ax2.set_title(f"Household electricity price ({price_source})")
    ax2.set_ylabel("EUR per kWh")
    ax2.set_xlabel("Local time")
    ax2.set_ylim(data["rate"].min() * 0.9, data["rate"].max() * 1.12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax2.set_xlim(data.index.min(), data.index.max() + pd.Timedelta(hours=1))

    runs = good_sun_runs(data)
    for first, last in runs:
        for ax in (ax1, ax2):
            ax.axvspan(first, last + pd.Timedelta(hours=1), color=COLORS["accent"], alpha=0.10, lw=0)
    if runs:
        first, last = max(runs, key=lambda r: r[1] - r[0])
        ax1.text(
            first + (last + pd.Timedelta(hours=1) - first) / 2,
            data["solar_irradiance"].max() * 0.5,
            f"best solar window\n{first:%H:%M}-{last + pd.Timedelta(hours=1):%H:%M}",
            ha="center",
            va="center",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def savings_chart(savings: pd.DataFrame) -> Figure:
    """Projected annual savings per recommendation (horizontal bars)."""
    savings = savings.sort_values("annual_savings")
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(savings["scenario"], savings["annual_savings"], color=COLORS["accent"], height=0.6)
    for bar, value in zip(bars, savings["annual_savings"], strict=True):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:,.0f} €",
            va="center",
            color=TEXT,
            fontsize=9,
        )
    ax.set_title("Projected annual savings per recommendation")
    ax.set_xlabel("EUR per year")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(1.0, savings["annual_savings"].max()) * 1.25)
    fig.tight_layout()
    return fig


def evaluation_heatmap(rows: Sequence[dict[str, Any]], metrics: Sequence[str]) -> Figure:
    """Heatmap of every metric for every test (single-hue sequential scale: darker = better)."""
    data = pd.DataFrame(rows).set_index("test_id")[list(metrics)]
    cmap = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(data) + 1.6))
    im = ax.imshow(data.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), [m.replace("_", "\n") for m in metrics], fontsize=9, color=TEXT_2)
    ax.xaxis.tick_top()  # metric names above the columns, where the eye starts reading
    ax.set_yticks(range(len(data)), data.index, fontsize=9, color=TEXT_2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(metrics))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(data))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    for (r, c), v in np.ndenumerate(data.values):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if v >= 0.6 else TEXT)
    ax.set_title("Evaluation scores per test (0-1)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).outline.set_visible(False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- data + files
def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_evaluation_heatmap(rows: Sequence[dict[str, Any]], metrics: Sequence[str], path: Path) -> Path:
    with plt.rc_context(STYLE):
        return _save(evaluation_heatmap(rows, metrics), path)


def window_frame(services: Services, start: datetime, end: datetime) -> tuple[pd.DataFrame, str, str, bool]:
    """Hourly irradiance, household price and price period for ``[start, end)``.

    Returns (data indexed by local hour, weather source, price source, prices are estimates).
    """
    tz = services.settings.location.tz
    first, last = start.astimezone(tz).date(), (end - timedelta(seconds=1)).astimezone(tz).date()
    weather = services.weather.series(first, last)
    irradiance = {h.time: h.irradiance_w_m2 for h in weather.hours}
    curve = services.prices.curve(floor_hour(start), end)
    hours = [
        (hour, retail)
        for hour, _, retail in curve.hourly()
        if start <= hour + timedelta(hours=1) and hour < end
    ]
    periods = assign_periods([retail for _, retail in hours])
    data = pd.DataFrame(
        {
            "solar_irradiance": [irradiance.get(hour, 0.0) for hour, _ in hours],
            "rate": [retail for _, retail in hours],
            "period": periods,
        },
        index=pd.DatetimeIndex([hour.astimezone(tz).replace(tzinfo=None) for hour, _ in hours]),
    )
    return data, weather.source, curve.source, curve.is_estimate


def history_profile(services: Services, days: int = 30) -> tuple[pd.DataFrame, pd.Series] | None:
    """Average consumption per local hour and device, and average solar yield per hour, over the last days.

    None if there is no history (run ``energy-advisor seed-db``).
    """
    tz = services.settings.location.tz
    today = services.clock().astimezone(tz).date()
    start, end = day_bounds_utc(today - timedelta(days=days), today - timedelta(days=1), tz)
    usage = services.repository.usage_between(start, end) if services.repository.exists() else []
    if not usage:
        return None
    generation = services.repository.generation_between(start, end)
    usage_df = pd.DataFrame(
        [{"t": r.timestamp.astimezone(tz), "device": r.device_type, "kwh": r.consumption_kwh} for r in usage]
    )
    n_days = usage_df["t"].map(lambda t: t.date()).nunique()
    profile = (
        usage_df.assign(hour=usage_df["t"].map(lambda t: t.hour))
        .groupby(["hour", "device"])["kwh"]
        .sum()
        .unstack(fill_value=0)
        .reindex(range(24), fill_value=0)
        / n_days
    )
    solar = pd.Series(0.0, index=range(24))
    for g in generation:
        solar[g.timestamp.astimezone(tz).hour] += g.generation_kwh / n_days
    return profile, solar


def generate_charts(services: Services, out_dir: Path, days: int = 30) -> list[Path]:
    """Query the history, forecast and prices, and write all charts to ``out_dir``."""
    settings = services.settings
    tz = settings.location.tz
    today = services.clock().astimezone(tz).date()
    start, end = day_bounds_utc(today - timedelta(days=days), today - timedelta(days=1), tz)
    usage = services.repository.usage_between(start, end)
    generation = services.repository.generation_between(start, end)
    if not usage:
        raise RuntimeError("No usage data found. Run `energy-advisor seed-db` first.")

    usage_df = pd.DataFrame(
        [{"t": r.timestamp.astimezone(tz), "device": r.device_type, "kwh": r.consumption_kwh} for r in usage]
    )
    usage_df["hour"] = usage_df["t"].map(lambda t: t.hour)
    usage_df["day"] = pd.to_datetime(usage_df["t"].map(lambda t: t.date()))
    n_days = usage_df["day"].nunique()
    profile = (
        usage_df.groupby(["hour", "device"])["kwh"]
        .sum()
        .unstack(fill_value=0)
        .reindex(range(24), fill_value=0)
    )
    profile = profile / n_days
    per_device = usage_df.pivot_table(index="day", columns="device", values="kwh", aggfunc="sum").fillna(0)

    solar_df = pd.DataFrame([{"t": g.timestamp.astimezone(tz), "kwh": g.generation_kwh} for g in generation])
    solar_profile = (
        solar_df.groupby(solar_df["t"].map(lambda t: t.hour))["kwh"].sum().reindex(range(24), fill_value=0)
        / n_days
    )
    solar_daily = solar_df.groupby(pd.to_datetime(solar_df["t"].map(lambda t: t.date())))["kwh"].sum()

    tomorrow = today + timedelta(days=1)
    window_start, window_end = day_bounds_utc(tomorrow, tomorrow, tz)
    frame, weather_source, price_source, is_estimate = window_frame(services, window_start, window_end)
    title = f"Solar irradiance forecast, {tomorrow} ({weather_source})"
    price_label = source_label(price_source, is_estimate)
    prices_df = frame
    tariff = settings.tariff
    cheap = float(np.mean(sorted(prices_df["rate"])[:4]))
    expensive = float(np.mean(sorted(prices_df["rate"])[-4:]))
    average = float(prices_df["rate"].mean())
    balance = energy_balance(usage, generation, tariff.feed_in_tariff_eur_per_kwh)
    exported_per_day = balance["solar_exported_kwh"] / n_days
    heat_pump_daily = per_device["heat_pump"].mean() if "heat_pump" in per_device else 0.0
    scenarios = [
        # name, device, current kWh/day, optimized kWh/day, current price, optimized price
        ("EV: evening -> cheapest hours (8 kWh/day)", "EV", 8, 8, expensive, cheap),
        ("EV: evening -> own solar (4 kWh/day)", "EV", 4, 4, expensive, tariff.feed_in_tariff_eur_per_kwh),
        ("Dishwasher: evening -> cheapest hours", "appliance", 1.0, 1.0, expensive, cheap),
        ("Heat pump: 1 °C lower setpoint (-6%)", "heat_pump", heat_pump_daily, heat_pump_daily * 0.94, average, average),
        ("Use exported solar at home (25%)", "solar", exported_per_day * 0.25, exported_per_day * 0.25,
         average, tariff.feed_in_tariff_eur_per_kwh),
    ]  # fmt: skip
    rows = []
    for name, device, cur, opt, price, new_price in scenarios:
        result = calculate_savings(
            device,
            round(cur, 2),
            round(opt, 2),
            round(price, 4),
            round(new_price, 4),
            settings.household.co2_kg_per_kwh,
            tariff.currency,
        )
        rows.append({"scenario": name, "annual_savings": result["annual_savings_eur"]})

    with plt.rc_context(STYLE):
        return [
            _save(hourly_profile_chart(profile, solar_profile), out_dir / "hourly_profile.png"),
            _save(daily_usage_vs_solar_chart(per_device, solar_daily), out_dir / "daily_usage_vs_solar.png"),
            _save(optimization_window_chart(frame, title, price_label), out_dir / "optimization_window.png"),
            _save(savings_chart(pd.DataFrame(rows)), out_dir / "annual_savings.png"),
        ]
