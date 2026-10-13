"""ORM tables for the household's energy history (SQLAlchemy 2.0 style).

Timestamps are stored as UTC (SQLite has no time zones) and always returned as aware UTC datetimes.
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Dialect, Float, String, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UTCDateTime(TypeDecorator[datetime]):
    """Accepts aware datetimes only; stores naive UTC; returns aware UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"Naive datetime cannot be stored: {value!r}")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None


class Base(DeclarativeBase):
    pass


class EnergyUsage(Base):
    """Consumption of one device during one hour (start of the hour in ``timestamp``)."""

    __tablename__ = "energy_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    consumption_kwh: Mapped[float] = mapped_column(Float)
    device_type: Mapped[str | None] = mapped_column(String(50))  # EV, heat_pump, appliance, base_load
    device_name: Mapped[str | None] = mapped_column(String(100))  # e.g. "Dishwasher"
    price_eur_per_kwh: Mapped[float | None] = mapped_column(Float)  # retail price of that hour
    cost_eur: Mapped[float | None] = mapped_column(Float)  # consumption x price (before solar offset)

    def __repr__(self) -> str:
        return f"<EnergyUsage {self.timestamp} {self.device_name} {self.consumption_kwh:.2f} kWh>"


class SolarGeneration(Base):
    """PV production during one hour."""

    __tablename__ = "solar_generation"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    generation_kwh: Mapped[float] = mapped_column(Float)
    weather_condition: Mapped[str | None] = mapped_column(String(50))  # sunny, partly_cloudy, cloudy, rainy
    temperature_c: Mapped[float | None] = mapped_column(Float)
    solar_irradiance: Mapped[float | None] = mapped_column(Float)  # W/m2, global horizontal

    def __repr__(self) -> str:
        return f"<SolarGeneration {self.timestamp} {self.generation_kwh:.2f} kWh {self.weather_condition}>"
