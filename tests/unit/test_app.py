"""Smoke tests of the Streamlit app, rendered headless (no browser, no network, no LLM call)."""

from datetime import timedelta

import httpx
import pytest

pytest.importorskip("streamlit")
import streamlit as st
from streamlit.testing.v1 import AppTest

from energy_advisor.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")
REMOVED_TEXTS = (
    "Household",
    "Use this window",
    "hours, local time",
    "Time window for chat",
    "Used when you ask",
)


def _offline_client(*_args: object, **_kwargs: object) -> httpx.Client:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network disabled in tests", request=request)

    return httpx.Client(transport=httpx.MockTransport(refuse))


@pytest.fixture(autouse=True)
def fresh_streamlit_cache():
    st.cache_resource.clear()  # the page caches the agent per process
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PATHS__DB", str(tmp_path / "energy.db"))
    monkeypatch.setenv("PATHS__PRICE_CACHE", str(tmp_path / "prices.sqlite"))
    # weather and price clients: offline -> synthetic weather and synthetic prices
    monkeypatch.setattr("energy_advisor.clients.open_meteo.default_http_client", _offline_client)
    monkeypatch.setattr("energy_advisor.services.container.default_http_client", _offline_client)


def _app(view: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    if view:
        at.session_state["view"] = view
    return at.run()


def _sidebar_texts(at: AppTest) -> str:
    parts = [c.value for c in at.sidebar.caption] + [m.value for m in at.sidebar.markdown]
    parts += [w.label for w in at.sidebar.toggle] + [s.value for s in at.sidebar.subheader]
    return " ".join(parts)


def test_sidebar_holds_window_devices_ok_and_new_conversation_last(app_env):
    at = _app()
    assert not at.exception
    from_date, to_date = at.sidebar.date_input
    from_time, to_time = at.sidebar.time_input
    assert to_date.value == from_date.value + timedelta(days=1)  # default: From + 24 h
    assert to_time.value == from_time.value
    labels = [b.label for b in at.sidebar.button]
    assert labels[0] == "OK"
    assert labels[-1].endswith("New conversation")
    assert not any(text in _sidebar_texts(at) for text in REMOVED_TEXTS)


def test_changing_the_start_moves_the_default_deadline(app_env):
    at = _app()
    from_date = at.sidebar.date_input[0]
    new_start = from_date.value + timedelta(days=2)
    from_date.set_value(new_start).run()
    assert at.sidebar.date_input[1].value == new_start + timedelta(days=1)


def test_ok_shows_the_result_screen(app_env):
    at = _app()
    assert any("press **OK**" in i.value for i in at.info)
    at.sidebar.button[0].click().run()
    assert not at.exception
    screen = " ".join(m.value for m in at.markdown)
    assert "### Electric car: 20 kWh" in screen
    assert "| Time slot | Energy (kWh) | From solar (kWh) | From grid (kWh) | Cost (EUR) |" in screen
    assert "Cost analysis" in screen
    assert "Saving:" in screen
    assert len(at.get("vega_lite_chart")) == 1  # runs, price and solar as stacked panels of one chart


def test_chat_view_shows_welcome_and_example_questions(app_env):
    at = _app("💬 Chat")
    assert not at.exception
    assert "Hi!" in at.chat_message[0].markdown[0].value
    assert len([b for b in at.button if b.key and b.key.startswith("suggest_")]) == 4
    assert at.chat_input


def test_new_conversation_returns_to_the_chat(app_env):
    at = _app("📈 Prices & solar")
    at.sidebar.button[-1].click().run()
    assert at.session_state["view"] == "💬 Chat"
    assert at.chat_input


def test_prices_view_renders_without_a_button(app_env):
    at = _app("📈 Prices & solar")
    assert not at.exception
    assert len(at.get("image")) == 1  # the README price/solar window chart (matplotlib)
    assert any("seed-db" in i.value for i in at.info)  # no history in the test: average-day chart explained
    assert not [b for b in at.main.button if b.label == "Show"]


def test_page_without_api_key_shows_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.error
