"""Smoke tests of the Streamlit app, rendered headless (no browser, no network, no LLM call)."""

from datetime import timedelta

import httpx
import pytest

pytest.importorskip("streamlit")
import streamlit as st
from streamlit.testing.v1 import AppTest

from energy_advisor.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")


def _offline_client(*_args: object, **_kwargs: object) -> httpx.Client:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network disabled in tests", request=request)

    return httpx.Client(transport=httpx.MockTransport(refuse))


@pytest.fixture(autouse=True)
def fresh_streamlit_cache():
    st.cache_resource.clear()  # the page caches the agent per process
    yield
    st.cache_resource.clear()


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PATHS__DB", str(tmp_path / "energy.db"))
    monkeypatch.setenv("PATHS__PRICE_CACHE", str(tmp_path / "prices.sqlite"))
    # weather and price clients: offline -> synthetic weather and synthetic prices
    monkeypatch.setattr("energy_advisor.clients.open_meteo.default_http_client", _offline_client)
    monkeypatch.setattr("energy_advisor.services.container.default_http_client", _offline_client)


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60)


def test_chat_view_shows_welcome_and_example_questions(app_env):
    at = _app().run()
    assert not at.exception
    assert "Hi!" in at.chat_message[0].markdown[0].value
    suggestions = [b.label for b in at.button if b.key and b.key.startswith("suggest_")]
    assert len(suggestions) == 4
    assert at.chat_input


def test_sidebar_window_defaults_to_now_plus_24_hours(app_env):
    at = _app().run()
    from_date, to_date = at.sidebar.date_input
    from_time, to_time = at.sidebar.time_input
    assert to_date.value == from_date.value + timedelta(days=1)
    assert to_time.value == from_time.value
    assert not [b for b in at.sidebar.button if "Reset" in b.label]


def test_changing_the_start_moves_the_default_deadline(app_env):
    at = _app().run()
    from_date = at.sidebar.date_input[0]
    new_start = from_date.value + timedelta(days=2)
    from_date.set_value(new_start).run()
    assert at.sidebar.date_input[1].value == new_start + timedelta(days=1)


def test_new_conversation_returns_to_the_chat(app_env):
    at = _app()
    at.session_state["view"] = "📅 Planner"
    at.run()
    next(b for b in at.sidebar.button if "New conversation" in b.label).click().run()
    assert at.session_state["view"] == "💬 Chat"
    assert at.chat_input


def test_planner_shows_best_windows_for_the_chosen_devices(app_env):
    at = _app()
    at.session_state["view"] = "📅 Planner"
    at.run()
    assert not at.exception
    next(b for b in at.button if b.label == "OK").click().run()
    assert not at.exception
    table = at.dataframe[0].value
    assert list(table["Device"]) == ["Electric car"]
    assert table["Best time"][0][:3].isalpha()  # e.g. "Mon 01:00-04:00"


def test_prices_view_renders_both_charts(app_env):
    at = _app()
    at.session_state["view"] = "📈 Prices & solar"
    at.run()
    next(b for b in at.button if b.label == "Show").click().run()
    assert not at.exception
    assert any("Electricity price" in m.value for m in at.markdown)


def test_page_without_api_key_shows_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = _app().run()
    assert at.error
