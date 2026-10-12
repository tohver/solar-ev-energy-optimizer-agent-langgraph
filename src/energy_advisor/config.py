"""Application settings: the single source of configuration.

Values come from (highest priority first): constructor arguments, environment variables,
the ``.env`` file in the project root, and the defaults below. Nested settings are overridden
with a double underscore, e.g. ``LLM__MODEL=gpt-4o`` or ``TARIFF__SURCHARGES_EUR_PER_KWH=0.21``.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LocationSettings(BaseModel):
    """Where the household is. Weather, prices and "today" are all resolved for this place."""

    name: str = "Munich, Germany"
    latitude: float = 48.1374
    longitude: float = 11.5755
    timezone: str = "Europe/Berlin"
    bidding_zone: str = Field(default="DE-LU", description="Electricity market area of the day-ahead auction")
    aliases: list[str] = Field(default_factory=lambda: ["munich", "münchen", "muenchen"])

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown time zone: {value}") from exc
        return value

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def matches(self, place: str | None) -> bool:
        """True if ``place`` refers to this location (or is empty)."""
        if not place or not place.strip():
            return True
        text = place.lower()
        return self.name.split(",")[0].lower() in text or any(alias in text for alias in self.aliases)


class TariffSettings(BaseModel):
    """Converts the wholesale day-ahead price into what a household with a dynamic tariff pays.

    retail = (spot / 1000 + surcharges) * (1 + VAT). The surcharges (grid fees, levies, electricity
    tax, concession fee, supplier margin) vary by city and supplier; the default is a typical value.
    """

    currency: str = "EUR"
    surcharges_eur_per_kwh: float = Field(default=0.19, description="Net, without VAT")
    vat_rate: float = 0.19
    feed_in_tariff_eur_per_kwh: float = Field(default=0.078, description="Paid per exported solar kWh (EEG)")
    fixed_price_eur_per_kwh: float = Field(
        default=0.35, description="A typical fixed-price tariff, for comparison"
    )


class PriceSettings(BaseModel):
    """Where prices come from. Real providers are tried in order; then an estimate; then a synthetic profile."""

    providers: list[Literal["energy_charts", "awattar"]] = Field(
        default_factory=lambda: ["energy_charts", "awattar"]
    )
    estimate_lookback_days: int = 7
    http_timeout_s: float = 20.0


class HouseholdSettings(BaseModel):
    """The demo household: used to generate its history and in calculations."""

    pv_peak_kw: float = Field(default=8.0, description="Installed PV size (kWp)")
    pv_performance_ratio: float = Field(
        default=0.8, description="Share of the theoretical yield after losses"
    )
    wallbox_kw: float = 11.0
    ev_consumption_kwh_per_100km: float = 17.0
    ev_daily_km: float = 35.0
    heat_loss_kw_per_k: float = Field(
        default=0.15, description="Building heat loss per kelvin of temperature difference"
    )
    heating_limit_c: float = Field(
        default=15.0, description="No space heating above this outdoor temperature"
    )
    base_load_kw: float = Field(default=0.3, description="Average always-on household load")
    co2_kg_per_kwh: float = Field(default=0.36, description="German grid mix, approx. (UBA); configurable")


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
    price_cache: Path = PROJECT_ROOT / "data" / "price_cache.sqlite"


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
    tariff: TariffSettings = TariffSettings()
    prices: PriceSettings = PriceSettings()
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
