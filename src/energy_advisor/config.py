"""Application settings: the single source of configuration.

Values come from (highest priority first): constructor arguments, environment variables,
the ``.env`` file in the project root, and the defaults below. Nested settings are overridden
with a double underscore, e.g. ``LLM__MODEL=gpt-4o`` or ``PATHS__DB=/tmp/energy.db``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LocationSettings(BaseModel):
    """Where the household is. Used as default context for the agent and for weather queries."""

    name: str = "San Francisco, CA"


class PricingSettings(BaseModel):
    """Mock time-of-use tariff (replaced by real day-ahead prices in phase 4)."""

    currency: str = "USD"
    base_price_per_kwh: float = 0.12
    off_peak_multiplier: float = 0.7
    mid_peak_multiplier: float = 1.3
    peak_multiplier: float = 2.2
    peak_demand_charge: float = 0.05


class HouseholdSettings(BaseModel):
    """Assumptions about the household used in calculations."""

    co2_kg_per_kwh: float = Field(default=0.4, description="Grid carbon intensity (kg CO2 per kWh)")


class LLMSettings(BaseModel):
    model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    reranker_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    recursion_limit: int = 25


class RAGSettings(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "energy_tips"
    candidates: int = 10
    rrf_k: int = 60
    min_rerank_score: int = 3


class PathSettings(BaseModel):
    db: Path = PROJECT_ROOT / "data" / "energy_data.db"
    vectorstore: Path = PROJECT_ROOT / "data" / "vectorstore"
    knowledge_base: Path = PROJECT_ROOT / "knowledge_base"
    reports: Path = PROJECT_ROOT / "reports"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    log_level: str = "INFO"

    location: LocationSettings = LocationSettings()
    pricing: PricingSettings = PricingSettings()
    household: HouseholdSettings = HouseholdSettings()
    llm: LLMSettings = LLMSettings()
    rag: RAGSettings = RAGSettings()
    paths: PathSettings = PathSettings()

    def require_api_key(self) -> SecretStr:
        """Return the API key or fail with a message that says how to fix it."""
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
            raise MissingApiKeyError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return self.openai_api_key


class MissingApiKeyError(RuntimeError):
    """Raised when a command needs the LLM but no API key is configured."""


@lru_cache
def get_settings() -> Settings:
    """Settings loaded once per process. Tests build their own ``Settings`` instead."""
    return Settings()
