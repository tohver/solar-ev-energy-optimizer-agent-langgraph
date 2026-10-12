"""Shared fixtures. Every test gets its own temporary database and settings; nothing reads the real .env."""

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from energy_advisor.config import PROJECT_ROOT, PathSettings, Settings, get_settings
from energy_advisor.data.repository import EnergyRepository
from energy_advisor.data.seed import seed_database
from energy_advisor.rag.ingest import build_index
from energy_advisor.rag.knowledge_base import KnowledgeBase
from energy_advisor.services.container import Services, build_services
from tests.helpers import LAST_SEEDED_DAY, NOW

SETTINGS_ENV_PREFIXES = (
    "OPENAI_",
    "LLM__",
    "PATHS__",
    "RAG__",
    "LOCATION__",
    "PRICING__",
    "HOUSEHOLD__",
    "LOG_LEVEL",
)


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
        ),
    )


@pytest.fixture
def repo(settings: Settings) -> EnergyRepository:
    """30 days of seeded sample data ending on LAST_SEEDED_DAY."""
    repository = EnergyRepository(settings.paths.db)
    seed_database(repository, days=30, seed=42, end_date=LAST_SEEDED_DAY)
    return repository


@pytest.fixture
def offline_http() -> httpx.Client:
    """HTTP client whose every request fails, so services fall back to their offline behaviour."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network disabled in tests", request=request)

    return httpx.Client(transport=httpx.MockTransport(refuse))


@pytest.fixture
def services(settings: Settings, repo: EnergyRepository, offline_http: httpx.Client) -> Services:
    return build_services(settings, http=offline_http, now=lambda: NOW)


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def knowledge_base(settings: Settings, fake_embeddings: DeterministicFakeEmbedding) -> KnowledgeBase:
    """Index of the real documents built with fake (offline, deterministic) embeddings; no re-ranker."""
    build_index(settings.paths.knowledge_base, settings.paths.vectorstore, fake_embeddings, settings.rag)
    return KnowledgeBase(settings.paths.vectorstore, settings.rag, lambda: fake_embeddings)
