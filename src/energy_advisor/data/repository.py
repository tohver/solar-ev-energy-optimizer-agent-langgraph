"""Database access for usage and solar history."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from energy_advisor.data.models import Base, EnergyUsage, SolarGeneration


class EnergyRepository:
    """Reads and writes the SQLite database. Date ranges are half-open: ``[start, end)``."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.engine: Engine = create_engine(f"sqlite:///{db_path.as_posix()}")
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._session_factory() as session, session.begin():
            yield session

    def exists(self) -> bool:
        return self.db_path.exists()

    def create_tables(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(self.engine)

    def clear(self) -> dict[str, int]:
        """Delete all rows (keeps the schema), so re-seeding never duplicates data."""
        with self.session() as s:
            usage = s.execute(delete(EnergyUsage)).rowcount
            solar = s.execute(delete(SolarGeneration)).rowcount
        return {"energy_usage": usage, "solar_generation": solar}

    def add_all(self, rows: Sequence[EnergyUsage | SolarGeneration]) -> int:
        """Insert many rows in a single transaction."""
        with self.session() as s:
            s.add_all(rows)
        return len(rows)

    def usage_between(self, start: datetime, end: datetime) -> list[EnergyUsage]:
        stmt = (
            select(EnergyUsage)
            .where(EnergyUsage.timestamp >= start, EnergyUsage.timestamp < end)
            .order_by(EnergyUsage.timestamp)
        )
        with self.session() as s:
            return list(s.scalars(stmt))

    def generation_between(self, start: datetime, end: datetime) -> list[SolarGeneration]:
        stmt = (
            select(SolarGeneration)
            .where(SolarGeneration.timestamp >= start, SolarGeneration.timestamp < end)
            .order_by(SolarGeneration.timestamp)
        )
        with self.session() as s:
            return list(s.scalars(stmt))
