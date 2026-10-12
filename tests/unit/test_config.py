import pytest

from energy_advisor.config import MissingApiKeyError, Settings
from energy_advisor.llm import make_chat_model, make_embeddings


def test_defaults_without_env_file():
    s = Settings(_env_file=None)
    assert s.location.name == "San Francisco, CA"
    assert s.openai_api_key is None
    assert s.openai_base_url is None  # standard OpenAI endpoint, no course proxy by default


def test_nested_values_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("LLM__MODEL", "gpt-test")
    monkeypatch.setenv("HOUSEHOLD__CO2_KG_PER_KWH", "0.25")
    s = Settings(_env_file=None)
    assert s.llm.model == "gpt-test"
    assert s.household.co2_kg_per_kwh == 0.25


def test_home_environment_variable_does_not_break_settings(monkeypatch):
    # Every OS sets HOME; a settings field called "home" used to make Settings() crash.
    monkeypatch.setenv("HOME", "/home/someone")
    Settings(_env_file=None)


def test_api_key_is_masked_in_repr():
    s = Settings(_env_file=None, openai_api_key="sk-secret-value")
    assert "sk-secret-value" not in repr(s)


@pytest.mark.parametrize("key", [None, ""])
def test_missing_api_key_gives_actionable_error(key):
    with pytest.raises(MissingApiKeyError, match=r"\.env"):
        Settings(_env_file=None, openai_api_key=key).require_api_key()


def test_f02_configured_base_url_reaches_chat_and_embeddings():
    # F-02: embeddings must use the same endpoint as the LLM, otherwise a proxy key does not work.
    s = Settings(_env_file=None, openai_api_key="k", openai_base_url="https://proxy.example/v1")
    assert make_chat_model(s).openai_api_base == "https://proxy.example/v1"
    assert make_embeddings(s).openai_api_base == "https://proxy.example/v1"
