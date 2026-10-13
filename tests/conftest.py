"""Shared fixtures. Every test gets its own temporary database and settings; nothing reads the real .env
and nothing touches the network (HTTP calls fail, so services use their offline fallbacks)."""

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from energy_advisor.config import PROJECT_ROOT, PathSettings, Settings, get_settings
from energy_advisor.data.repository import EnergyRepository
from energy_advisor.data.seed import build_history, seed_database
from energy_advisor.rag.ingest import build_index
from energy_advisor.rag.knowledge_base import KnowledgeBase
from energy_advisor.services.container import Services, build_services
from tests.helpers import LAST_SEEDED_DAY, fixed_clock

SETTINGS_ENV_PREFIXES = (
    "OPENAI_",
    "LLM__",
    "PATHS__",
    "RAG__",
    "LOCATION__",
    "TARIFF__",
    "PRICES__",
    "HOUSEHOLD__",
    "LOG_LEVEL",
)
SEEDED_DAYS = 14


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove settings-related environment variables so the developer's shell cannot leak into tests."""
    for name in list(os.environ):
        if name.upper().startswith(SETTINGS_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        paths=PathSettings(
            db=tmp_path / "energy.db",
            vectorstore=tmp_path / "vectorstore",
            knowledge_base=PROJECT_ROOT / "knowledge_base",
            reports=tmp_path / "reports",
            price_cache=tmp_path / "price_cache.sqlite",
        ),
    )


@pytest.fixture
def offline_http() -> httpx.Client:
    """HTTP client whose every request fails, so services fall back to their offline behaviour."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network disabled in tests", request=request)

    return httpx.Client(transport=httpx.MockTransport(refuse))


@pytest.fixture
def services(settings: Settings, offline_http: httpx.Client) -> Services:
    """Offline services: synthetic weather and synthetic prices, fixed clock."""
    return build_services(settings, http=offline_http, clock=fixed_clock)


@pytest.fixture
def repo(services: Services) -> EnergyRepository:
    """SEEDED_DAYS days of history ending on LAST_SEEDED_DAY (deterministic: synthetic inputs, seed 42)."""
    s = services.settings
    history = build_history(
        services.weather, services.prices, s.household, s.location.tz, SEEDED_DAYS, LAST_SEEDED_DAY, seed=42
    )
    seed_database(services.repository, history)
    return services.repository


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def knowledge_base(settings: Settings, fake_embeddings: DeterministicFakeEmbedding) -> KnowledgeBase:
    """Index of the real documents built with fake (offline, deterministic) embeddings; no re-ranker."""
    build_index(settings.paths.knowledge_base, settings.paths.vectorstore, fake_embeddings, settings.rag)
    return KnowledgeBase(settings.paths.vectorstore, settings.rag, lambda: fake_embeddings)
