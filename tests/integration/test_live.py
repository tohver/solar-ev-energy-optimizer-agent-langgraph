"""Tests against real services. Not run by default:

uv run pytest -m integration      # real Open-Meteo API (needs internet)
uv run pytest -m llm              # real LLM + embeddings (needs OPENAI_API_KEY, costs a little)
"""

import pytest

from energy_advisor.agent.graph import EnergyAdvisor
from energy_advisor.agent.trace import final_answer
from energy_advisor.clients.open_meteo import OpenMeteoClient
from energy_advisor.config import Settings
from energy_advisor.llm import make_embeddings
from energy_advisor.rag.ingest import build_index
from energy_advisor.services.container import build_services
from energy_advisor.services.weather import WeatherService


@pytest.fixture
def live_settings(settings):
    """Test paths, but the real API key and endpoint from .env."""
    real = Settings()
    if real.openai_api_key is None or not real.openai_api_key.get_secret_value():
        pytest.skip("OPENAI_API_KEY not configured")
    return settings.model_copy(
        update={"openai_api_key": real.openai_api_key, "openai_base_url": real.openai_base_url}
    )


@pytest.mark.integration
def test_live_weather_forecast():
    result = WeatherService(OpenMeteoClient()).forecast("San Francisco, CA", days=2)
    assert result["source"].startswith("open-meteo")
    assert len(result["hourly"]) == 48


@pytest.mark.llm
def test_search_energy_tips_end_to_end(live_settings):
    build_index(
        live_settings.paths.knowledge_base,
        live_settings.paths.vectorstore,
        make_embeddings(live_settings),
        live_settings.rag,
    )
    result = build_services(live_settings).knowledge_base.search("when to run the pool pump", max_results=3)
    assert 1 <= len(result["tips"]) <= 3
    assert all(tip["source"].endswith(".txt") for tip in result["tips"])


@pytest.mark.llm
def test_agent_answers_end_to_end(live_settings, repo):
    advisor = EnergyAdvisor.from_settings(
        live_settings, instructions="You are a test assistant. Answer in one short sentence."
    )
    result = advisor.invoke("Say hello.")
    assert final_answer(result)
    assert "error" not in result
