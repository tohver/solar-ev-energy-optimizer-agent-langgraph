"""Multi-device planner (shared solar surplus) and the planner view's data helpers."""

from datetime import UTC, date, datetime, timedelta

import pytest

from energy_advisor.optimization.scheduling import Slot
from energy_advisor.optimization.service import DEVICE_PRESETS, DeviceSpec, WindowInputs
from energy_advisor.services.prices import PriceCurve
from energy_advisor.ui.planner import day_window, hourly_rows, summary_rows, window_bars
from energy_advisor.ui.window import InvalidWindowError
from tests.helpers import BERLIN

T0 = datetime(2026, 10, 13, 8, tzinfo=UTC)
EV = DeviceSpec("ev", "Electric car", 6.0, 11.0, True)
DISHWASHER = DEVICE_PRESETS["dishwasher"]  # 1 kWh at 0.5 kW, in one go


@pytest.fixture
def optimizer(services, monkeypatch):
    """Optimizer whose window is 4 hours at 0.30 EUR/kWh with 5 kWh of solar surplus in the 2nd hour only."""
    slots = [
        Slot(T0 + timedelta(hours=i), T0 + timedelta(hours=i + 1), 0.30, 5.0 if i == 1 else 0.0)
        for i in range(4)
    ]
    curve = PriceCurve(T0, T0 + timedelta(hours=4), [], "test prices", False)
    window = WindowInputs(T0, T0 + timedelta(hours=4), slots, curve, "test solar")
    monkeypatch.setattr(services.optimizer, "window", lambda *a, **k: window)
    return services.optimizer


def test_devices_share_the_solar_surplus_instead_of_counting_it_twice(optimizer):
    plan = optimizer.plan_devices([DISHWASHER, EV])
    by_name = {r.device.name: r.schedule for r in plan.results}
    total_solar = sum(s.solar_kwh for s in by_name.values() if s)
    assert total_solar == pytest.approx(5.0)  # not 5 + 1
    assert by_name["Electric car"].solar_kwh == pytest.approx(5.0)  # largest device is planned first
    assert plan.solar_left_kwh == pytest.approx(0.0)


def test_results_keep_the_users_order(optimizer):
    plan = optimizer.plan_devices([DISHWASHER, EV])
    assert [r.device.key for r in plan.results] == ["dishwasher", "ev"]


def test_a_device_that_does_not_fit_gets_an_error_and_the_others_are_still_planned(optimizer):
    too_big = DeviceSpec("ev", "Electric car", 100.0, 11.0, True)  # 4 h x 11 kW = 44 kWh max
    plan = optimizer.plan_devices([too_big, DISHWASHER])
    assert plan.results[0].schedule is None
    assert "Not enough time" in plan.results[0].error
    assert plan.results[1].schedule is not None


def test_view_helpers(optimizer):
    plan = optimizer.plan_devices([EV, DISHWASHER])
    hours = hourly_rows(plan, BERLIN)
    assert [h["solar_kwh"] for h in hours] == [0.0, 5.0, 0.0, 0.0]
    assert hours[0]["time"] == datetime(2026, 10, 13, 10, 0, tzinfo=BERLIN).replace(
        tzinfo=None
    )  # naive local
    bars = window_bars(plan, BERLIN)
    assert {b["device"] for b in bars} == {"Electric car", "Dishwasher"}
    rows = summary_rows(plan, BERLIN)
    assert rows[0]["Device"] == "Electric car"
    assert rows[0]["From solar (kWh)"] == 5.0


def test_planning_without_devices_still_returns_the_window(optimizer):
    plan = optimizer.plan_devices([])
    assert plan.results == []
    assert len(hourly_rows(plan, BERLIN)) == 4


# ----------------------------------------------------------------------------- day window
NOW = datetime(2026, 10, 12, 18, 37, tzinfo=BERLIN)


def test_today_means_from_now_for_24_hours():
    window = day_window(date(2026, 10, 12), NOW)  # 18:37
    assert window.start == datetime(2026, 10, 12, 18, 45, tzinfo=BERLIN)  # next quarter hour
    assert window.deadline == datetime(2026, 10, 13, 18, 45, tzinfo=BERLIN)


def test_a_later_day_means_that_whole_day():
    window = day_window(date(2026, 10, 14), NOW)
    assert window.start == datetime(2026, 10, 14, 0, 0, tzinfo=BERLIN)
    assert window.deadline == datetime(2026, 10, 15, 0, 0, tzinfo=BERLIN)


@pytest.mark.parametrize("day", [date(2026, 10, 11), date(2026, 10, 30)])
def test_past_or_far_days_are_rejected(day):
    with pytest.raises(InvalidWindowError):
        day_window(day, NOW)


def test_chart_has_three_titled_panels_with_legends(optimizer):
    import pandas as pd

    from energy_advisor.ui.charts import plan_chart

    plan = optimizer.plan_devices([EV, DISHWASHER])
    spec = plan_chart(
        pd.DataFrame(hourly_rows(plan, BERLIN)), pd.DataFrame(window_bars(plan, BERLIN))
    ).to_dict()
    titles = [panel["title"] for panel in spec["vconcat"]]
    assert titles == ["When your devices run", "Electricity price", "Solar surplus"]
    assert all("legend" in panel["encoding"]["color"] for panel in spec["vconcat"])
    assert spec["resolve"]["scale"]["x"] == "shared"
