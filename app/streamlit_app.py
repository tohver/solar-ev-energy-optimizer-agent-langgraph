"""Streamlit chat app for the energy advisor.

Run with:  uv run --extra app streamlit run app/streamlit_app.py
"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from energy_advisor.agent.graph import EnergyAdvisor
from energy_advisor.agent.trace import final_answer, tool_calls
from energy_advisor.config import MissingApiKeyError, Settings, get_settings
from energy_advisor.services.container import Services, build_services
from energy_advisor.timeutils import floor_hour, to_utc
from energy_advisor.ui.window import (
    InvalidWindowError,
    SchedulingWindow,
    default_deadline,
    default_window,
    window_context,
    window_from_inputs,
)

st.set_page_config(page_title="Solar & EV Energy Advisor", page_icon="⚡", layout="wide")


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


# ----------------------------------------------------------------------------- scheduling window (sidebar)
def reset_window() -> None:
    window = default_window(services.clock().astimezone(tz))
    state.from_date, state.from_time = window.start.date(), window.start.time()
    state.to_date, state.to_time = window.deadline.date(), window.deadline.time()
    state.deadline_overridden = False


def start_changed() -> None:
    """Keep the default 24-hour range while the user has not chosen a deadline."""
    if not state.deadline_overridden:
        deadline = default_deadline(datetime.combine(state.from_date, state.from_time, tzinfo=tz))
        state.to_date, state.to_time = deadline.date(), deadline.time()


def deadline_changed() -> None:
    state.deadline_overridden = True


if "from_date" not in state:
    reset_window()

with st.sidebar:
    st.header("Scheduling window")
    st.caption("Used for questions like *When should I charge the car?* Default: from now for 24 hours.")
    c1, c2 = st.columns(2)
    c1.date_input("From", key="from_date", on_change=start_changed)
    c2.time_input("at", key="from_time", step=timedelta(minutes=15), on_change=start_changed)
    c3, c4 = st.columns(2)
    c3.date_input("Deadline", key="to_date", on_change=deadline_changed)
    c4.time_input("at ", key="to_time", step=timedelta(minutes=15), on_change=deadline_changed)
    st.button("Reset to now + 24 h", on_click=reset_window)
    use_window = st.toggle("Use this window in my questions", value=True, key="use_window")

    window: SchedulingWindow | None
    try:
        window = window_from_inputs(state.from_date, state.from_time, state.to_date, state.to_time, tz)
        st.caption(f"Window: {window.hours:.1f} hours ({tz.key})")
    except InvalidWindowError as exc:
        window = None
        st.warning(str(exc))

    st.divider()
    if st.button("New conversation"):
        state.thread_id = advisor.new_thread_id()
        state.chat = []
        st.rerun()
    st.caption(f"Household: {settings.location.name} · PV {settings.household.pv_peak_kw:g} kWp")


# ----------------------------------------------------------------------------- main area
st.title("⚡ Solar & EV Energy Advisor")
tab_chat, tab_window = st.tabs(["Chat", "Prices & solar in the window"])

with tab_chat:
    for message in state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tools"):
                with st.expander(f"Tool calls ({len(message['tools'])})"):
                    st.json(message["tools"])

    if question := st.chat_input("Ask about charging, prices, solar, your usage..."):
        state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        context = window_context(window) if (use_window and window) else None
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            result = advisor.invoke(question, context=context, thread_id=state.thread_id)
            answer = final_answer(result)
            calls = tool_calls({"messages": result["turn_messages"]})
            st.markdown(answer)
            if calls:
                with st.expander(f"Tool calls ({len(calls)})"):
                    st.json(calls)
        state.chat.append({"role": "assistant", "content": answer, "tools": calls})

with tab_window:
    if window is None:
        st.info("Choose a deadline after the start.")
    elif st.button("Show prices and solar surplus for this window"):
        start = floor_hour(max(to_utc(window.start), to_utc(services.clock())))
        end = to_utc(window.deadline)
        with st.spinner("Fetching prices and the forecast..."):
            curve = services.prices.curve(start, end + timedelta(hours=1))
            first, last = start.astimezone(tz).date(), end.astimezone(tz).date()
            surplus, solar_source = services.optimizer.solar_surplus(first, last)
        hours = pd.DataFrame(
            [
                {
                    "time": hour.astimezone(tz).replace(tzinfo=None),
                    "price (EUR/kWh)": retail,
                    "solar surplus (kWh)": surplus.get(hour, (0.0, 0.0))[1],
                }
                for hour, _, retail in curve.hourly()
                if start <= hour < end
            ]
        ).set_index("time")
        cheapest = hours["price (EUR/kWh)"].idxmin()
        sunniest = hours["solar surplus (kWh)"].idxmax()

        st.subheader("Electricity price per hour")
        st.line_chart(hours[["price (EUR/kWh)"]], x_label="Local time", y_label="EUR per kWh")
        st.caption(
            "What you pay for one kWh from the grid in each hour (dynamic tariff, incl. fees and VAT). "
            f"Cheapest hour: {cheapest:%a %H:%M} at {hours['price (EUR/kWh)'].min():.3f} EUR/kWh. "
            f"Source: {curve.source}{' - ESTIMATE, prices not published yet' if curve.is_estimate else ''}."
        )

        st.subheader("Solar surplus per hour")
        st.bar_chart(
            hours[["solar surplus (kWh)"]], x_label="Local time", y_label="kWh in that hour", color="#eda100"
        )
        if hours["solar surplus (kWh)"].max() > 0:
            best = f"Most surplus: {sunniest:%a %H:%M} with {hours['solar surplus (kWh)'].max():.1f} kWh. "
        else:
            best = "No solar surplus expected in this window. "
        st.caption(
            "Energy your PV system is expected to produce beyond the house's own base load, i.e. what is "
            "available for the car or appliances. It costs only the feed-in tariff you would otherwise "
            f"receive ({settings.tariff.feed_in_tariff_eur_per_kwh:.3f} EUR/kWh), so tall bars are the best "
            f"hours to run flexible devices. {best}Forecast: {solar_source}."
        )
