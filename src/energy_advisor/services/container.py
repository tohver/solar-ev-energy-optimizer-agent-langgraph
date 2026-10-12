"""Wiring: build every service from ``Settings`` in one place (dependency injection)."""

from dataclasses import dataclass

import httpx

from energy_advisor.clients.open_meteo import OpenMeteoClient
from energy_advisor.clients.prices.base import PriceProvider
from energy_advisor.clients.prices.synthetic import SyntheticPriceProvider
from energy_advisor.config import Settings
from energy_advisor.data.repository import EnergyRepository
from energy_advisor.llm import make_chat_model, make_embeddings
from energy_advisor.rag.knowledge_base import KnowledgeBase
from energy_advisor.rag.rerank import LLMReranker
from energy_advisor.services.prices import PriceService
from energy_advisor.services.weather import WeatherService
from energy_advisor.timeutils import Clock, make_clock


@dataclass
class Services:
    settings: Settings
    repository: EnergyRepository
    weather: WeatherService
    prices: PriceService
    knowledge_base: KnowledgeBase
    clock: Clock


def build_price_providers(settings: Settings, http: httpx.Client | None) -> list[PriceProvider]:
    return [SyntheticPriceProvider(settings.location.tz)]


def build_services(
    settings: Settings,
    http: httpx.Client | None = None,
    clock: Clock | None = None,
    price_providers: list[PriceProvider] | None = None,
) -> Services:
    """Nothing here touches the network or needs an API key; LLM clients are created on first use."""
    clock = clock or make_clock(settings.location.tz)
    return Services(
        settings=settings,
        repository=EnergyRepository(settings.paths.db),
        weather=WeatherService(OpenMeteoClient(http), settings.location, clock),
        prices=PriceService(
            price_providers or build_price_providers(settings, http),
            settings.tariff,
            settings.location,
            clock,
        ),
        knowledge_base=KnowledgeBase(
            persist_dir=settings.paths.vectorstore,
            rag=settings.rag,
            embeddings_factory=lambda: make_embeddings(settings),
            reranker_factory=lambda: LLMReranker.from_llm(
                make_chat_model(settings, settings.llm.reranker_model), settings.rag.min_rerank_score
            ),
        ),
        clock=clock,
    )
