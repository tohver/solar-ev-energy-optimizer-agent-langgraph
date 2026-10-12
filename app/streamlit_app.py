"""Streamlit app for the energy advisor.

All settings live in the sidebar: the time window (From, Deadline) and the devices. OK computes the best
times and shows the result screen. The chat and the prices & solar view use the same window.

Run with:  uv run --extra app streamlit run app/streamlit_app.py
"""

from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from energy_advisor.agent.graph import EnergyAdvisor
from energy_advisor.agent.trace import final_answer, tool_calls
from energy_advisor.config import MissingApiKeyError, Settings, get_settings
from energy_advisor.optimization.service import DEVICE_PRESETS, DevicePlan, DeviceSpec
from energy_advisor.services.container import Services, build_services
from energy_advisor.ui.charts import BLUE, GREEN, OLIVE, PURPLE, plan_chart
from energy_advisor.ui.planner import hourly_rows, result_markdown, window_bars
from energy_advisor.ui.window import (
    InvalidWindowError,
    SchedulingWindow,
    default_deadline,
    default_window,
    window_context,
    window_from_inputs,
)
from energy_advisor.viz.charts import (
    STYLE,
    history_profile,
    hourly_profile_chart,
    optimization_window_chart,
    source_label,
    window_frame,
)

st.set_page_config(page_title="Solar & EV Energy Advisor", page_icon="⚡", layout="wide")

VIEWS = ["📅 Best times", "💬 Chat", "📈 Prices & solar"]
DEVICE_ICONS = {"ev": "🚗", "dishwasher": "🍽️", "washing_machine": "🧺"}
CSS = f"""
<style>
h1, h2, h3, h4 {{ color: {BLUE} !important; }}
.st-key-result {{ border: 1px solid {GREEN}; border-radius: 8px; padding: 1rem 1.4rem 0.6rem; }}
.st-key-result strong {{ color: {OLIVE}; }}
.st-key-result th {{ color: {BLUE}; }}
.st-key-result blockquote {{ border-left-color: {PURPLE}; }}
</style>
"""
SUGGESTIONS = [
    "When should I charge my EV with 20 kWh?",
    "What will electricity cost tomorrow at 19:00?",
    "How much of my solar power did I use myself last week?",
    "Which device used the most electricity in the last 7 days?",
]
WELCOME = (
    "Hi! I plan your EV charging and appliances around electricity prices and your solar power. "
    "Ask me anything, or pick one of the examples below."
)


@st.cache_resource
def load() -> tuple[Settings, Services, EnergyAdvisor]:
    """One set of services and one agent (with its conversation memory) per server process."""
    settings = get_settings()
    services = build_services(settings)
    return settings, services, EnergyAdvisor.from_settings(settings, services)


try:
    settings, services, advisor = load()
except MissingApiKeyError as exc:
    st.error(f"{exc}")
    st.stop()

tz = settings.location.tz
state = st.session_state
if "thread_id" not in state:
    state.thread_id = advisor.new_thread_id()
    state.chat = []  # list of {"role", "content", "tools"}
if "view" not in state:
    state.view = VIEWS[0]


def now() -> datetime:
    return services.clock().astimezone(tz)


@st.cache_data(ttl=900, show_spinner="Checking prices and the solar forecast...")
def plan_for(
    start_iso: str, deadline_iso: str, devices: tuple[DeviceSpec, ...], _services: Services
) -> DevicePlan:
    """Cached for 15 minutes per window and device set, so switching views does not refetch."""
    return _services.optimizer.plan_devices(
        list(devices), datetime.fromisoformat(start_iso), datetime.fromisoformat(deadline_iso)
    )


# ----------------------------------------------------------------------------- callbacks
def new_conversation() -> None:
    state.thread_id = advisor.new_thread_id()
    state.chat = []
    state.view = VIEWS[1]


def ask(question: str) -> None:
    state.pending = question
    state.view = VIEWS[1]


def start_changed() -> None:
    """Keep the default 24-hour range while the user has not chosen a deadline."""
    if not state.deadline_overridden:
        deadline = default_deadline(datetime.combine(state.from_date, state.from_time, tzinfo=tz))
        state.to_date, state.to_time = deadline.date(), deadline.time()


def deadline_changed() -> None:
    state.deadline_overridden = True


def selected_devices() -> tuple[DeviceSpec, ...]:
    devices = []
    for key in state.get("devices") or []:
        preset = DEVICE_PRESETS[key]
        if key == "ev":
            preset = DeviceSpec(
                preset.key, preset.name, float(state.ev_kwh), preset.power_kw, preset.interruptible
            )
        devices.append(preset)
    return tuple(devices)


def submit() -> None:
    state.submitted = (state.window_iso, selected_devices())
    state.view = VIEWS[0]


if "from_date" not in state:
    initial = default_window(now())
    state.from_date, state.from_time = initial.start.date(), initial.start.time()
    state.to_date, state.to_time = initial.deadline.date(), initial.deadline.time()
    state.deadline_overridden = False


# ----------------------------------------------------------------------------- sidebar: the only settings
with st.sidebar:
    c1, c2 = st.columns(2)
    c1.date_input("From", key="from_date", on_change=start_changed)
    c2.time_input("at", key="from_time", step=timedelta(minutes=15), on_change=start_changed)
    c3, c4 = st.columns(2)
    c3.date_input("Deadline", key="to_date", on_change=deadline_changed)
    c4.time_input("at ", key="to_time", step=timedelta(minutes=15), on_change=deadline_changed)
    window: SchedulingWindow | None
    try:
        window = window_from_inputs(state.from_date, state.from_time, state.to_date, state.to_time, tz)
        state.window_iso = (window.start.isoformat(), window.deadline.isoformat())
    except InvalidWindowError as exc:
        window = None
        st.warning(str(exc))

    st.pills(
        "Devices",
        options=list(DEVICE_PRESETS),
        format_func=lambda k: f"{DEVICE_ICONS[k]} {DEVICE_PRESETS[k].name}",
        selection_mode="multi",
        default=["ev"],
        key="devices",
    )
    if "ev" in (state.get("devices") or []):
        st.number_input("Car: energy to charge (kWh)", 1.0, 80.0, 20.0, 1.0, key="ev_kwh")
    st.button(
        "OK",
        type="primary",
        width="stretch",
        on_click=submit,
        disabled=window is None or not state.get("devices"),
    )

    st.space("large")
    st.divider()
    st.button("✏️ New conversation", on_click=new_conversation, width="stretch")


# ----------------------------------------------------------------------------- main area
st.markdown(CSS, unsafe_allow_html=True)
st.title("⚡ Solar & EV Energy Advisor")
view = st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed") or VIEWS[0]


def render_best_times() -> None:
    submitted = state.get("submitted")
    if not submitted:
        st.info("Choose the time window and the devices on the left, then press **OK**.")
        return
    (start_iso, deadline_iso), devices = submitted
    try:
        plan = plan_for(start_iso, deadline_iso, devices, services)
    except Exception as exc:
        st.error(f"Could not plan: {exc}")
        return
    with st.container(key="result"):
        st.markdown("#### Result")
        st.markdown(result_markdown(plan, tz, settings.tariff.feed_in_tariff_eur_per_kwh))
    bars = pd.DataFrame(window_bars(plan, tz))
    if not bars.empty:
        st.altair_chart(plan_chart(pd.DataFrame(hourly_rows(plan, tz)), bars), width="stretch")


def render_chat() -> None:
    if not state.chat:
        with st.chat_message("assistant"):
            st.markdown(WELCOME)
        cols = st.columns(len(SUGGESTIONS))
        for col, suggestion in zip(cols, SUGGESTIONS, strict=True):
            col.button(
                suggestion, type="tertiary", on_click=ask, args=(suggestion,), key=f"suggest_{suggestion}"
            )
    for message in state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tools"):
                with st.expander(f"Tool calls ({len(message['tools'])})"):
                    st.json(message["tools"])

    question = st.chat_input("Ask about charging, prices, solar, your usage...") or state.pop("pending", None)
    if not question:
        return
    state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    context = window_context(window) if window else None
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        result = advisor.invoke(question, context=context, thread_id=state.thread_id)
        answer = final_answer(result)
        calls = tool_calls({"messages": result["turn_messages"]})
        st.markdown(answer)
        if calls:
            with st.expander(f"Tool calls ({len(calls)})"):
                st.json(calls)
    state.chat.append({"role": "assistant", "content": answer, "tools": calls})


@st.cache_data(ttl=900, show_spinner="Fetching prices and the solar forecast...")
def window_data(
    start_iso: str, deadline_iso: str, _services: Services
) -> tuple[pd.DataFrame, str, str, bool]:
    return window_frame(_services, datetime.fromisoformat(start_iso), datetime.fromisoformat(deadline_iso))


@st.cache_data(ttl=3600, show_spinner=False)
def average_day(_services: Services) -> tuple[pd.DataFrame, pd.Series] | None:
    return history_profile(_services)


def render_prices() -> None:
    """The two charts of the README: the price/solar window and the household's average day."""
    if window is None:
        st.info("Choose a deadline after the start on the left.")
        return
    try:
        data, weather_source, price_source, is_estimate = window_data(
            window.start.isoformat(), window.deadline.isoformat(), services
        )
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        return
    simulated = "" if weather_source.startswith("open-meteo") else " (simulated weather)"
    with plt.rc_context(STYLE):
        fig = optimization_window_chart(
            data, f"Solar irradiance forecast{simulated}", source_label(price_source, is_estimate)
        )
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        "Top: expected solar irradiance. Bottom: what one kWh from the grid costs (incl. fees and VAT). "
        "Shaded: the best solar hours outside the most expensive third of the prices. "
        f"Prices: {price_source}. Weather: {weather_source}."
    )

    profile = average_day(services)
    if profile is None:
        st.info("The average-day chart needs the household history: run `energy-advisor seed-db`.")
        return
    with plt.rc_context(STYLE):
        fig = hourly_profile_chart(*profile)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        "Average of the last 30 days: when each device uses electricity, and when the PV system produces."
    )


renderers: dict[str, Any] = {VIEWS[0]: render_best_times, VIEWS[1]: render_chat, VIEWS[2]: render_prices}
renderers[view]()
