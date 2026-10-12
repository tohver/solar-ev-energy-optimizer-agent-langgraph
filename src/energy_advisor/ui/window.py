"""The scheduling window chosen in the app: start (default now) and deadline (default start + 24 hours).

Pure functions, so the Streamlit page stays thin and the logic is unit-tested.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from energy_advisor.optimization.service import DEFAULT_WINDOW_HOURS
from energy_advisor.timeutils import to_utc

FORMAT = "%Y-%m-%dT%H:%M"


class InvalidWindowError(ValueError):
    pass


@dataclass(frozen=True)
class SchedulingWindow:
    start: datetime  # aware, local time
    deadline: datetime  # aware, local time

    @property
    def hours(self) -> float:
        return (to_utc(self.deadline) - to_utc(self.start)) / timedelta(hours=1)


def default_deadline(start: datetime) -> datetime:
    """Start + 24 hours of real time (also correct across a daylight-saving change)."""
    tz = start.tzinfo
    return (to_utc(start) + timedelta(hours=DEFAULT_WINDOW_HOURS)).astimezone(tz)


def default_window(now: datetime) -> SchedulingWindow:
    start = now.replace(second=0, microsecond=0)
    return SchedulingWindow(start, default_deadline(start))


def window_from_inputs(
    start_date: date, start_time: time, deadline_date: date, deadline_time: time, tz: ZoneInfo
) -> SchedulingWindow:
    start = datetime.combine(start_date, start_time, tzinfo=tz)
    deadline = datetime.combine(deadline_date, deadline_time, tzinfo=tz)
    if to_utc(deadline) <= to_utc(start):
        raise InvalidWindowError("The deadline must be after the start.")
    return SchedulingWindow(start, deadline)


def window_context(window: SchedulingWindow) -> str:
    """Text for the agent's system context, so scheduling tools use the window from the app."""
    return (
        "Scheduling window set by the user in the app: "
        f"start {window.start:{FORMAT}}, deadline {window.deadline:{FORMAT}} (local time). "
        "Use these as start and deadline for find_cheapest_window unless the user's message names other times."
    )
