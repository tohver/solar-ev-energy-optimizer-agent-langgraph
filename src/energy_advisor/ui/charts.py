"""Altair charts of the app, in the palette of the README terminal screen (docs/img/ask-demo.svg).

Separate panels on one time axis instead of overlays; every panel has a title and a legend.
"""

from datetime import timedelta
from typing import Any

import altair as alt
import pandas as pd

# Palette of the terminal screen in the README (docs/img/ask-demo.svg)
TEXT, MUTED, GRID = "#c5c8c6", "#8a8d8b", "#3a3a3a"
BLUE, GREEN, OLIVE, PURPLE, SOLAR = "#68a0b3", "#00823d", "#98a84b", "#98729f", "#d7af5f"
DEVICE_COLORS = {"Electric car": BLUE, "Dishwasher": OLIVE, "Washing machine": PURPLE}


def styled(chart: Any) -> Any:
    """Dark, monospace chart style matching the app theme."""
    return (
        chart.configure(background="transparent", font="monospace")
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID, tickColor=GRID)
        .configure_axisY(
            minExtent=120, maxExtent=120
        )  # the same left margin in every panel: aligned time axes
        .configure_legend(
            labelColor=TEXT,
            titleColor=TEXT,
            orient="top",
            direction="horizontal",
            symbolSize=140,
            labelLimit=0,
        )
        .configure_title(color=BLUE, anchor="start", fontSize=14, offset=10)
        .configure_view(stroke=None)
    )


def price_panel(hours: pd.DataFrame, x: Any, height: int = 150) -> Any:
    data = hours.assign(series="Grid price (EUR/kWh)")
    return (
        alt.Chart(data, title="Electricity price")
        .mark_line(interpolate="step-after", strokeWidth=2.5)
        .encode(
            x=x,
            y=alt.Y("price:Q", title="EUR/kWh", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=alt.Scale(range=[BLUE]), legend=alt.Legend(title=None)),
            tooltip=[
                alt.Tooltip("time:T", title="hour", format="%a %H:%M"),
                alt.Tooltip("price:Q", format=".3f"),
            ],
        )
        .properties(height=height)
    )


def solar_panel(hours: pd.DataFrame, x: Any, height: int = 150) -> Any:
    data = hours.assign(series="Solar surplus (kWh)", end=hours["time"] + timedelta(hours=1))
    return (
        alt.Chart(data, title="Solar surplus")
        .mark_rect(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, stroke="#292929", strokeWidth=1)
        .encode(
            x=x,
            x2="end:T",
            y=alt.Y("solar_kwh:Q", title="kWh"),
            y2=alt.datum(0),  # bars from zero up to the value
            color=alt.Color("series:N", scale=alt.Scale(range=[SOLAR]), legend=alt.Legend(title=None)),
            tooltip=[
                alt.Tooltip("time:T", title="hour", format="%a %H:%M"),
                alt.Tooltip("solar_kwh:Q", title="kWh", format=".2f"),
            ],
        )
        .properties(height=height)
    )


def plan_chart(hours: pd.DataFrame, bars: pd.DataFrame) -> Any:
    """Three panels on one time axis: device runs, price, solar surplus. Each with a title and a legend."""
    x_scale = alt.Scale(domain=[hours["time"].min(), hours["time"].max() + timedelta(hours=1)])
    x = alt.X("time:T", title=None, scale=x_scale, axis=alt.Axis(format="%a %H:%M", labelAngle=0))
    present = [d for d in DEVICE_COLORS if d in set(bars["device"])]
    runs = (
        alt.Chart(bars, title="When your devices run")
        .mark_bar(cornerRadius=3, height=16)
        .encode(
            x=alt.X("start:T", title=None, scale=x_scale, axis=alt.Axis(format="%a %H:%M", labelAngle=0)),
            x2="end:T",
            y=alt.Y("device:N", title=None, sort=present, axis=alt.Axis(labelColor=TEXT)),
            color=alt.Color(
                "device:N",
                scale=alt.Scale(domain=present, range=[DEVICE_COLORS[d] for d in present]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("device:N"),
                alt.Tooltip("start:T", format="%a %H:%M"),
                alt.Tooltip("end:T", format="%a %H:%M"),
                alt.Tooltip("kwh:Q", title="kWh"),
                alt.Tooltip("cost_eur:Q", title="cost (EUR)"),
            ],
        )
        .properties(height=34 * len(present))
    )
    chart = alt.vconcat(runs, price_panel(hours, x), solar_panel(hours, x), spacing=28)
    return styled(chart.resolve_scale(x="shared", color="independent"))
