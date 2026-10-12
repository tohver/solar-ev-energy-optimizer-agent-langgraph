"""Wiring: build every service from ``Settings`` in one place (dependency injection)."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import httpx

from energy_advisor.clients.open_meteo import OpenMeteoClient
from energy_advisor.config import Settings
from energy_advisor.data.repository import EnergyRepository
from energy_advisor.llm import make_chat_model, make_embeddings
from energy_advisor.rag.knowledge_base import KnowledgeBase
from energy_advisor.rag.rerank import LLMReranker
from energy_advisor.services.prices import MockPriceService
from energy_advisor.services.weather import WeatherService


@dataclass
class Services:
    settings: Settings
    repository: EnergyRepository
    weather: WeatherService
    prices: MockPriceService
    knowledge_base: KnowledgeBase
    now: Callable[[], datetime]


def build_services(
    settings: Settings,
    http: httpx.Client | None = None,
    now: Callable[[], datetime] = datetime.now,
) -> Services:
    """Nothing here touches the network or needs an API key; LLM clients are created on first use."""
    return Services(
        settings=settings,
        repository=EnergyRepository(settings.paths.db),
        weather=WeatherService(OpenMeteoClient(http), now=now),
        prices=MockPriceService(settings.pricing, today=lambda: now().date()),
        knowledge_base=KnowledgeBase(
            persist_dir=settings.paths.vectorstore,
            rag=settings.rag,
            embeddings_factory=lambda: make_embeddings(settings),
            reranker_factory=lambda: LLMReranker.from_llm(
                make_chat_model(settings, settings.llm.reranker_model), settings.rag.min_rerank_score
            ),
        ),
        now=now,
    )
