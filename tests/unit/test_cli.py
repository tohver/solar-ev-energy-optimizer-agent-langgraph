"""Smoke tests of the command-line interface."""

import httpx
import pytest
from typer.testing import CliRunner

from energy_advisor.cli import app

runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Point the CLI's settings at temporary paths; an empty key overrides the one in the real .env."""
    monkeypatch.setenv("PATHS__DB", str(tmp_path / "energy.db"))
    monkeypatch.setenv("PATHS__VECTORSTORE", str(tmp_path / "vectorstore"))
    monkeypatch.setenv("PATHS__REPORTS", str(tmp_path / "reports"))
    monkeypatch.setenv("OPENAI_API_KEY", "")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network disabled in tests", request=request)

    offline = httpx.Client(transport=httpx.MockTransport(refuse))
    monkeypatch.setattr("energy_advisor.clients.open_meteo.default_http_client", lambda *a, **k: offline)
    monkeypatch.setattr("energy_advisor.services.container.default_http_client", lambda *a, **k: offline)
    return tmp_path


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("seed-db", "build-index", "rag-benchmark", "ask", "eval", "report"):
        assert command in result.output


def test_seed_db_creates_database(cli_env):
    result = runner.invoke(app, ["seed-db", "--days", "3", "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert (cli_env / "energy.db").exists()
    assert "usage rows" in result.output
    assert "weather: simulated" in result.output  # offline fallback


def test_seed_db_rejects_more_than_90_days(cli_env):
    assert runner.invoke(app, ["seed-db", "--days", "120"]).exit_code != 0


def test_commands_needing_the_llm_fail_cleanly_without_key(cli_env):
    for args in (["ask", "Hello?"], ["build-index"], ["eval", "--limit", "1"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1, args
        assert "OPENAI_API_KEY is not set" in result.output


def test_report_without_data_fails_cleanly(cli_env):
    result = runner.invoke(app, ["report"])
    assert result.exit_code != 0


def test_chat_without_key_fails_cleanly(cli_env):
    result = runner.invoke(app, ["chat"], input="hello\nexit\n")
    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output
