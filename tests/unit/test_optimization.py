"""Deterministic scheduler and the optimizer tools."""

from datetime import UTC, datetime, timedelta

import pytest

from energy_advisor.agent.tools import build_tools
from energy_advisor.optimization.scheduling import ScheduleError, Slot, find_cheapest_schedule
from energy_advisor.services.container import build_services
from tests.helpers import BERLIN, StaticPriceProvider, fixed_clock, hourly_prices

T0 = datetime(2026, 10, 12, 0, tzinfo=UTC)
FEED_IN = 0.08


def hours(prices, solar=None):
    solar = solar or [0.0] * len(prices)
    return [
        Slot(T0 + timedelta(hours=i), T0 + timedelta(hours=i + 1), p, s)
        for i, (p, s) in enumerate(zip(prices, solar, strict=True))
    ]


def schedule(slots, energy, power, **kwargs):
    start = kwargs.pop("earliest", slots[0].start)
    end = kwargs.pop("deadline", slots[-1].end)
    return find_cheapest_schedule(slots, energy, power, start, end, FEED_IN, **kwargs)


# ----------------------------------------------------------------------------- contiguous
def test_contiguous_load_runs_in_the_cheapest_consecutive_hours():
    slots = hours([0.40, 0.30, 0.20, 0.25, 0.50, 0.10, 0.60])
    result = schedule(slots, energy=2.0, power=1.0, interruptible=False)
    block = result.blocks[0]
    # two consecutive hours: 2+3 cost 0.45, 5+6 cost 0.70, hour 5 alone is cheapest but must be paired
    assert (block.start, block.end) == (T0 + timedelta(hours=2), T0 + timedelta(hours=4))
    assert result.cost_eur == pytest.approx(0.45)
    assert result.baseline_cost_eur == pytest.approx(0.70)  # start immediately: hours 0+1
    assert result.savings_eur == pytest.approx(0.25)


def test_contiguous_load_must_finish_before_the_deadline():
    slots = hours([0.40, 0.30, 0.20, 0.25, 0.05, 0.05])
    result = schedule(slots, energy=2.0, power=1.0, interruptible=False, deadline=T0 + timedelta(hours=4))
    assert result.blocks[0].end <= T0 + timedelta(hours=4)


def test_partial_hours_are_charged_proportionally():
    result = schedule(hours([0.30, 0.10]), energy=0.5, power=1.0, interruptible=False)
    assert result.blocks[0].start == T0 + timedelta(hours=1)
    assert result.blocks[0].end == T0 + timedelta(hours=1, minutes=30)
    assert result.cost_eur == pytest.approx(0.05)


# ----------------------------------------------------------------------------- interruptible
def test_interruptible_load_uses_the_cheapest_hours_anywhere():
    slots = hours([0.40, 0.10, 0.50, 0.12, 0.45, 0.11])
    result = schedule(slots, energy=33.0, power=11.0)
    used = {b.start.hour for b in result.blocks}
    assert used == {1, 3, 5}
    assert result.energy_kwh == pytest.approx(33.0)
    assert result.cost_eur == pytest.approx(11 * (0.10 + 0.12 + 0.11))
    assert len(result.blocks) == 3  # not adjacent, so three blocks


def test_adjacent_hours_are_merged_and_partial_hours_run_at_lower_power():
    result = schedule(hours([0.10, 0.11, 0.50]), energy=16.5, power=11.0)
    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert (block.start, block.end) == (T0, T0 + timedelta(hours=2))
    assert block.average_kw == pytest.approx(8.25)


def test_solar_surplus_is_used_first_and_valued_at_the_feed_in_tariff():
    slots = hours([0.30, 0.30, 0.20], solar=[0.0, 4.0, 0.0])
    result = schedule(slots, energy=6.0, power=11.0)
    assert result.solar_kwh == pytest.approx(4.0)
    assert result.grid_kwh == pytest.approx(2.0)
    # 4 kWh solar at 0.08 + 2 kWh in the cheapest grid hour (0.20)
    assert result.cost_eur == pytest.approx(4 * FEED_IN + 2 * 0.20)


def test_solar_is_ignored_when_disabled():
    slots = hours([0.30, 0.30, 0.20], solar=[0.0, 4.0, 0.0])
    assert schedule(slots, energy=6.0, power=11.0, use_solar=False).solar_kwh == 0


def test_grid_is_preferred_when_it_is_cheaper_than_the_feed_in_tariff():
    # negative grid prices: consuming from the grid earns money, so solar is exported instead
    slots = hours([-0.05, 0.30], solar=[5.0, 0.0])
    result = schedule(slots, energy=5.0, power=11.0)
    assert result.solar_kwh == 0
    assert result.cost_eur == pytest.approx(-0.25)


# ----------------------------------------------------------------------------- validation
def test_not_enough_time_is_explained():
    with pytest.raises(ScheduleError, match=r"at most 22.0 kWh"):
        schedule(hours([0.1, 0.1]), energy=30.0, power=11.0)


@pytest.mark.parametrize(("energy", "power"), [(0, 11), (10, 0), (-1, 11)])
def test_invalid_inputs(energy, power):
    with pytest.raises(ScheduleError):
        schedule(hours([0.1, 0.1]), energy=energy, power=power)


def test_deadline_before_start():
    with pytest.raises(ScheduleError, match="deadline"):
        schedule(hours([0.1]), energy=1, power=1, earliest=T0 + timedelta(hours=1), deadline=T0)


# ----------------------------------------------------------------------------- tools
class EveningPeak(StaticPriceProvider):
    """Expensive 17-21 local time, cheap 01-05, normal otherwise."""

    def fetch(self, start, end):
        out = []
        for p in hourly_prices(start, end, [0.0] * int((end - start) / timedelta(hours=1))):
            h = p.start.astimezone(BERLIN).hour
            out.append(type(p)(p.start, p.end, 300.0 if 17 <= h < 21 else 20.0 if 1 <= h < 5 else 120.0))
        return out


@pytest.fixture
def tools(settings, offline_http):
    services = build_services(settings, http=offline_http, clock=fixed_clock, price_providers=[EveningPeak()])
    return {t.name: t for t in build_tools(services)}


def test_ev_charges_overnight_before_the_deadline(tools):
    result = tools["find_cheapest_window"].invoke(
        {"energy_kwh": 30, "power_kw": 11, "deadline": "2026-10-03T07:00",
         "use_solar": False, "device": "EV"}
    )  # fmt: skip
    blocks = result["recommended_schedule"]
    assert blocks[0]["start"] == "2026-10-03 01:00"
    assert result["finishes_at"] <= "2026-10-03 07:00"
    assert result["savings_vs_baseline_eur"] > 0
    assert result["prices"]["is_estimate"] is False


def test_dishwasher_avoids_the_evening_peak(tools):
    result = tools["find_cheapest_window"].invoke(
        {"energy_kwh": 1.0, "power_kw": 0.5, "deadline": "2026-10-03T12:00",
         "interruptible": False, "use_solar": False}
    )  # fmt: skip
    start = result["recommended_schedule"][0]["start"]
    assert "2026-10-03 01:00" <= start <= "2026-10-03 03:00"
    assert result["mode"].startswith("contiguous")


def test_solar_forecast_is_used_offline(tools):
    result = tools["find_cheapest_window"].invoke(
        {"energy_kwh": 3, "power_kw": 11, "deadline": "2026-10-02T17:00"}
    )
    assert result["solar_forecast_source"].startswith("simulated")


@pytest.mark.parametrize(
    "args",
    [
        {"energy_kwh": 5, "power_kw": 11, "deadline": "tomorrow 7am"},
        {"energy_kwh": 500, "power_kw": 11, "deadline": "2026-10-02T12:00"},
        {"energy_kwh": 5, "power_kw": 11, "deadline": "2026-10-20T12:00"},
    ],
)
def test_tool_errors_are_returned_to_the_llm(tools, args):
    assert "error" in tools["find_cheapest_window"].invoke(args)


def test_pv_forecast_tool(tools):
    result = tools["estimate_pv_generation"].invoke({"days": 5})
    assert len(result["days"]) == 3  # clamped to 1-3
    today = result["days"]["2026-10-02"]
    assert today["pv_kwh"] >= today["surplus_kwh"] >= 0
    assert all(h["pv_kwh"] > 0 for h in today["hourly"])


def test_without_deadline_the_next_24_hours_are_used_and_flagged(tools):
    result = tools["find_cheapest_window"].invoke({"energy_kwh": 10, "power_kw": 11, "use_solar": False})
    assert result["window"]["start"] == "2026-10-02 09:30"  # now
    assert result["window"]["deadline"] == "2026-10-03 09:30"
    assert any("No deadline given" in n for n in result["notes"])


def test_window_always_starts_now(tools):
    result = tools["find_cheapest_window"].invoke(
        {"energy_kwh": 5, "power_kw": 11, "deadline": "2026-10-04T07:00"}
    )
    assert result["window"] == {
        "start": "2026-10-02 09:30",
        "deadline": "2026-10-04 07:00",
        "timezone": "Europe/Berlin",
    }
    assert "notes" not in result or not any("No deadline" in n for n in result["notes"])


def test_deadline_in_the_past_is_an_error(tools):
    assert "error" in tools["find_cheapest_window"].invoke(
        {"energy_kwh": 5, "power_kw": 11, "deadline": "2026-10-01T07:00"}
    )


def test_window_starts_now_and_any_hour_is_allowed(tools):
    # a washing machine may run at night: no quiet hours are assumed
    result = tools["find_cheapest_window"].invoke(
        {
            "energy_kwh": 0.8,
            "power_kw": 0.4,
            "deadline": "2026-10-03T09:00",
            "interruptible": False,
            "use_solar": False,
        }
    )
    assert result["window"]["start"] == "2026-10-02 09:30"
    assert "2026-10-03 01:00" <= result["recommended_schedule"][0]["start"] <= "2026-10-03 03:00"


def test_start_can_be_given_and_the_deadline_defaults_to_24_hours_after_it(tools):
    result = tools["find_cheapest_window"].invoke(
        {"energy_kwh": 5, "power_kw": 11, "start": "2026-10-04T20:00", "use_solar": False}
    )  # 2.5 days ahead: no error any more
    assert result["window"]["start"] == "2026-10-04 20:00"
    assert result["window"]["deadline"] == "2026-10-05 20:00"
    assert any("No deadline given" in n for n in result["notes"])


def test_start_in_the_past_becomes_now(tools):
    result = tools["find_cheapest_window"].invoke(
        {"energy_kwh": 5, "power_kw": 11, "start": "2026-10-01T08:00", "deadline": "2026-10-03T08:00"}
    )
    assert result["window"]["start"] == "2026-10-02 09:30"
    assert any("in the past" in n for n in result["notes"])
