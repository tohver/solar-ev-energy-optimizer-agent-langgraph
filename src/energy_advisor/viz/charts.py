"""Charts for the README and reports, written as PNG files (needs the ``viz`` extra).

Every chart is a function that takes plain data and returns a matplotlib ``Figure``;
``generate_charts`` gathers the data from the services and writes the files.
"""

import logging
from collections.abc import Sequence
from datetime import datetime, time, timedelta
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

from energy_advisor.services.container import Services
from energy_advisor.services.prices import PEAK_HOURS
from energy_advisor.services.savings import calculate_savings

logger = logging.getLogger(__name__)

COLORS = {"EV": "#2a78d6", "HVAC": "#eb6834", "appliance": "#1baf7a", "solar": "#eda100", "accent": "#2a78d6"}
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
def hourly_profile_chart(avg_kwh_by_hour: dict[str, dict[int, float]]) -> Figure:
    """Average consumption per hour of day for each device type, with the peak-price window."""
    profile = pd.DataFrame(avg_kwh_by_hour).sort_index()
    profile.index = profile.index.astype(int)
    profile = profile.sort_index()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for device in profile.columns:
        ax.plot(profile.index, profile[device], color=COLORS.get(device, TEXT_2), label=device)

    # Direct labels at the line ends, nudged apart so they don't collide
    min_gap = (profile.max().max() - profile.min().min()) * 0.07
    placed: list[float] = []
    for y, device in sorted((profile[d].iloc[-1], d) for d in profile.columns):
        y_label = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_label)
        ax.annotate(
            device,
            (profile.index[-1], y),
            xytext=(profile.index[-1] + 0.4, y_label),
            textcoords="data",
            va="center",
            color=TEXT,
            fontsize=9,
        )

    ax.axvspan(PEAK_HOURS.start, PEAK_HOURS.stop, color=GRID, alpha=0.6, lw=0)
    ax.text(
        (PEAK_HOURS.start + PEAK_HOURS.stop) / 2,
        profile.max().max() * 0.62,
        f"peak price\n{PEAK_HOURS.start}-{PEAK_HOURS.stop}",
        ha="center",
        va="center",
        color=TEXT_2,
        fontsize=9,
    )
    ax.set_title("Average consumption per hour of day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("kWh per hour")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 24.5)
    ax.legend(loc="upper left")
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
            label=device,
            edgecolor=SURFACE,
            linewidth=1,
        )
        bottom = per_device[device] if bottom is None else bottom + per_device[device]
    ax1.set_title("Daily consumption by device type")
    ax1.set_ylabel("kWh per day")
    ax1.set_ylim(0, per_device.sum(axis=1).max() * 1.18)  # headroom so the legend never covers bars
    ax1.legend(ncol=3, loc="upper left")

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


def best_solar_window(weather: pd.DataFrame, prices: pd.DataFrame) -> pd.Index:
    """Hours with good sun (>= 50% of the day's peak irradiance) outside the peak-price period."""
    good_sun = weather["solar_irradiance"] >= 0.5 * weather["solar_irradiance"].max()
    return weather.index[good_sun & (prices["period"] != "peak")]


def optimization_window_chart(
    weather: pd.DataFrame, prices: pd.DataFrame, day: str, source: str, currency: str
) -> Figure:
    """Solar irradiance forecast and time-of-use price for one day, with the best window highlighted."""
    window = best_solar_window(weather, prices)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.fill_between(
        weather.index, weather["solar_irradiance"], color=COLORS["solar"], alpha=0.35, lw=0, step="mid"
    )
    ax1.step(weather.index, weather["solar_irradiance"], where="mid", color=COLORS["solar"])
    ax1.set_title(f"Solar irradiance forecast - {day} ({source})")
    ax1.set_ylabel("W/m²")

    ax2.step(prices.index, prices["rate"], where="mid", color=COLORS["accent"])
    ax2.set_title("Electricity price (time-of-use)")
    ax2.set_ylabel(f"{currency} per kWh")
    ax2.set_xlabel("Hour of day")
    ax2.set_xticks(range(0, 24, 2))
    for period, label_hour in (("off_peak", 2), ("mid_peak", 11), ("peak", 19)):
        rate = prices.loc[prices["period"] == period, "rate"].iloc[0]
        ax2.annotate(
            f"{period.replace('_', '-')}  {rate:.3f}",
            (label_hour, rate),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            color=TEXT,
            fontsize=9,
        )

    if len(window):
        for ax in (ax1, ax2):
            ax.axvspan(window.min() - 0.5, window.max() + 0.5, color=COLORS["accent"], alpha=0.10, lw=0)
        ax2.text(
            (window.min() + window.max()) / 2,
            (prices["rate"].min() + prices.loc[window, "rate"].min()) / 2,
            f"best solar window\n{window.min():02d}:00-{window.max() + 1:02d}:00",
            ha="center",
            va="center",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )
    fig.tight_layout()
    return fig


def savings_chart(savings: pd.DataFrame, currency: str) -> Figure:
    """Projected annual savings per recommendation (horizontal bars)."""
    savings = savings.sort_values("annual_savings")
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(savings["scenario"], savings["annual_savings"], color=COLORS["accent"], height=0.6)
    for bar, value in zip(bars, savings["annual_savings"], strict=True):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:,.0f} {currency}",
            va="center",
            color=TEXT,
            fontsize=9,
        )
    ax.set_title("Projected annual savings per recommendation")
    ax.set_xlabel(f"{currency} per year")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, savings["annual_savings"].max() * 1.25)
    fig.tight_layout()
    return fig


def evaluation_heatmap(rows: Sequence[dict[str, Any]], metrics: Sequence[str]) -> Figure:
    """Heatmap of every metric for every test (single-hue sequential scale: darker = better)."""
    data = pd.DataFrame(rows).set_index("test_id")[list(metrics)]
    cmap = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(data) + 1.6))
    im = ax.imshow(data.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), [m.replace("_", "\n") for m in metrics], fontsize=9, color=TEXT_2)
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


def generate_charts(services: Services, out_dir: Path, days: int = 30) -> list[Path]:
    """Query the history, forecast and prices, and write all charts to ``out_dir``."""
    end = services.now().date()
    start = end - timedelta(days=days)
    settings = services.settings
    currency = settings.pricing.currency

    start_dt, end_dt = datetime.combine(start, time.min), datetime.combine(end, time.min)
    usage = services.repository.usage_between(start_dt, end_dt)
    generation = services.repository.generation_between(start_dt, end_dt)
    if not usage:
        raise RuntimeError("No usage data found. Run `energy-advisor seed-db` first.")

    usage_df = pd.DataFrame(
        [{"timestamp": r.timestamp, "device": r.device_type, "kwh": r.consumption_kwh} for r in usage]
    )
    usage_df["hour"] = usage_df["timestamp"].dt.hour
    usage_df["day"] = usage_df["timestamp"].dt.normalize()
    avg_by_hour = usage_df.groupby(["device", "hour"])["kwh"].mean().unstack(0).round(2).to_dict()
    per_device = usage_df.pivot_table(index="day", columns="device", values="kwh", aggfunc="sum").fillna(0)
    solar_daily = (
        pd.DataFrame(
            [{"day": pd.Timestamp(r.timestamp).normalize(), "kwh": r.generation_kwh} for r in generation]
        )
        .groupby("day")["kwh"]
        .sum()
    )

    tomorrow = (end + timedelta(days=1)).isoformat()
    forecast = services.weather.forecast(settings.location.name, days=2)
    prices = services.prices.get_prices(tomorrow)
    weather_df = pd.DataFrame([h for h in forecast["hourly"] if h["date"] == tomorrow]).set_index("hour")
    prices_df = pd.DataFrame(prices["hourly_rates"]).set_index("hour")

    rates = prices_df.groupby("period")["rate"].first()
    peak, mid, off = rates["peak"], rates["mid_peak"], rates["off_peak"]
    hvac_daily = per_device["HVAC"].mean() if "HVAC" in per_device else 0.0
    scenarios = [
        # name, device, current kWh/day, optimized kWh/day, current rate, optimized rate
        ("EV charging: peak -> off-peak (10 kWh/day)", "EV", 10, 10, peak, off),
        ("Dishwasher: peak -> off-peak (1.5 kWh/day)", "appliance", 1.5, 1.5, peak, off),
        ("Pool pump: mid-peak -> off-peak (3 kWh/day)", "pool pump", 3, 3, mid, off),
        ("Thermostat +2°F in summer (-6% HVAC)", "HVAC", hvac_daily, hvac_daily * 0.94, mid, mid),
        ("Standby loads on smart plugs (-0.3 kWh/day)", "appliance", 0.3, 0.0, mid, mid),
    ]
    unit = currency.lower()
    savings_rows = []
    for name, device, cur, opt, rate, new_rate in scenarios:
        result = calculate_savings(
            device, round(cur, 2), round(opt, 2), rate, new_rate, settings.household.co2_kg_per_kwh, currency
        )
        savings_rows.append({"scenario": name, "annual_savings": result[f"annual_savings_{unit}"]})

    with plt.rc_context(STYLE):
        return [
            _save(hourly_profile_chart(avg_by_hour), out_dir / "hourly_profile.png"),
            _save(daily_usage_vs_solar_chart(per_device, solar_daily), out_dir / "daily_usage_vs_solar.png"),
            _save(
                optimization_window_chart(
                    weather_df, prices_df, tomorrow, forecast.get("source", ""), currency
                ),
                out_dir / "optimization_window.png",
            ),
            _save(savings_chart(pd.DataFrame(savings_rows), currency), out_dir / "annual_savings.png"),
        ]
