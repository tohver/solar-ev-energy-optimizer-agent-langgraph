"""Smoke test of the Streamlit page (rendered headless, no browser, no LLM call)."""

from datetime import timedelta

import pytest

pytest.importorskip("streamlit")
import streamlit as st
from streamlit.testing.v1 import AppTest

from energy_advisor.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")


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


def test_page_renders_with_a_default_24_hour_window(app_env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    from_date, to_date = at.sidebar.date_input
    from_time, to_time = at.sidebar.time_input
    start = from_date.value.isoformat() + from_time.value.isoformat()
    deadline = (to_date.value - timedelta(days=1)).isoformat() + to_time.value.isoformat()
    assert start == deadline  # deadline = start + 24 h (same clock time, next day)
    assert at.sidebar.toggle[0].value is True
    assert at.chat_input


def test_changing_the_start_moves_the_default_deadline(app_env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    from_date = at.sidebar.date_input[0]
    new_start = from_date.value + timedelta(days=2)
    from_date.set_value(new_start).run()
    assert at.sidebar.date_input[1].value == new_start + timedelta(days=1)


def test_page_without_api_key_shows_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.error
