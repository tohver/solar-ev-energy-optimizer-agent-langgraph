"""ORM tables for the household's energy history (SQLAlchemy 2.0 style)."""

from datetime import datetime

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EnergyUsage(Base):
    """One hourly consumption reading of one device."""

    __tablename__ = "energy_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    consumption_kwh: Mapped[float] = mapped_column(Float)
    device_type: Mapped[str | None] = mapped_column(String(50))  # e.g. "EV", "HVAC", "appliance"
    device_name: Mapped[str | None] = mapped_column(String(100))  # e.g. "Tesla Model 3"
    cost_usd: Mapped[float | None] = mapped_column(Float)

    def __repr__(self) -> str:
        return f"<EnergyUsage {self.timestamp} {self.device_name} {self.consumption_kwh:.2f} kWh>"


class SolarGeneration(Base):
    """One hourly reading of the household's PV system."""

    __tablename__ = "solar_generation"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    generation_kwh: Mapped[float] = mapped_column(Float)
    weather_condition: Mapped[str | None] = mapped_column(String(50))  # sunny, cloudy, ...
    temperature_c: Mapped[float | None] = mapped_column(Float)
    solar_irradiance: Mapped[float | None] = mapped_column(Float)  # W/m2

    def __repr__(self) -> str:
        return f"<SolarGeneration {self.timestamp} {self.generation_kwh:.2f} kWh {self.weather_condition}>"
