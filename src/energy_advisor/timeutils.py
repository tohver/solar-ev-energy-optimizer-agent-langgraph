"""Time-zone-aware time helpers.

Rules used across the project:
- Every ``datetime`` is aware. Storage and arithmetic use UTC; local time is only for display and for
  grouping by "day" or "hour of day".
- Never subtract or add ``timedelta`` to local datetimes: Python does wall-clock arithmetic for
  datetimes that share a ``tzinfo``, which is wrong on daylight-saving days (23 or 25 hours).
"""

import re
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

Clock = Callable[[], datetime]
HOUR = timedelta(hours=1)
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LOCAL_DT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{1,2}):(\d{2})$")


class InvalidDateError(ValueError):
    pass


def make_clock(tz: ZoneInfo) -> Clock:
    """A clock returning the current local time in ``tz`` (aware)."""
    return lambda: datetime.now(tz)


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"Naive datetime not allowed: {value!r}")
    return value.astimezone(UTC)


def parse_day(value: str) -> date:
    """Strict ``YYYY-MM-DD``."""
    if not isinstance(value, str) or not _DAY_RE.match(value.strip()):
        raise InvalidDateError(f"Invalid date '{value}'. Use YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise InvalidDateError(f"Invalid date '{value}'. Use YYYY-MM-DD format.") from exc


def parse_local_datetime(value: str, tz: ZoneInfo) -> datetime:
    """``YYYY-MM-DDTHH:MM`` (or with a space) in local time, returned aware in ``tz``."""
    match = _LOCAL_DT_RE.match(value.strip()) if isinstance(value, str) else None
    if not match:
        raise InvalidDateError(f"Invalid date-time '{value}'. Use YYYY-MM-DDTHH:MM (local time).")
    day, hour, minute = match.groups()
    try:
        return datetime.combine(date.fromisoformat(day), time(int(hour), int(minute)), tzinfo=tz)
    except ValueError as exc:
        raise InvalidDateError(f"Invalid date-time '{value}'. Use YYYY-MM-DDTHH:MM (local time).") from exc


def local_midnight(day: date, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, time.min, tzinfo=tz)


def day_bounds_utc(first_day: date, last_day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """UTC interval ``[start, end)`` covering the local days ``first_day..last_day`` (inclusive)."""
    return to_utc(local_midnight(first_day, tz)), to_utc(local_midnight(last_day + timedelta(days=1), tz))


def hours_between(start: datetime, end: datetime) -> Iterator[datetime]:
    """UTC hour starts in ``[start, end)``; DST-safe because it steps in UTC."""
    current = to_utc(start).replace(minute=0, second=0, microsecond=0)
    if current < to_utc(start):
        current += HOUR
    end_utc = to_utc(end)
    while current < end_utc:
        yield current
        current += HOUR


def hours_in_local_day(day: date, tz: ZoneInfo) -> int:
    start, end = day_bounds_utc(day, day, tz)
    return int((end - start) / HOUR)


def floor_hour(value: datetime) -> datetime:
    return to_utc(value).replace(minute=0, second=0, microsecond=0)


def calendar_lines(now: datetime, days: int = 8) -> str:
    """'Monday = 2026-10-12 (today), Tuesday = 2026-10-13 (tomorrow), ...' in local time."""
    today = now.date()
    labels = {0: " (today)", 1: " (tomorrow)"}
    return ", ".join(
        f"{(today + timedelta(days=d)):%A} = {(today + timedelta(days=d)).isoformat()}{labels.get(d, '')}"
        for d in range(days)
    )
