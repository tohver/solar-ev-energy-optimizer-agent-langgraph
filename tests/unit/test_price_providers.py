"""Real price providers (recorded responses), the disk cache, the estimate and the full provider chain."""

import json
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
import respx

from energy_advisor.clients.prices import awattar, energy_charts
from energy_advisor.clients.prices.awattar import AwattarProvider
from energy_advisor.clients.prices.cache import CachedPriceProvider
from energy_advisor.clients.prices.energy_charts import EnergyChartsProvider
from energy_advisor.clients.prices.estimate import EstimatePriceProvider
from energy_advisor.clients.prices.synthetic import SyntheticPriceProvider
from energy_advisor.config import LocationSettings, TariffSettings
from energy_advisor.domain.prices import PriceUnavailableError
from energy_advisor.services.container import build_services
from energy_advisor.services.prices import PriceService
from energy_advisor.timeutils import day_bounds_utc
from tests.helpers import BERLIN, FIXTURES, StaticPriceProvider, fixed_clock, hourly_prices

DAY = date(2026, 10, 11)
START, END = day_bounds_utc(DAY, DAY, BERLIN)
ENERGY_CHARTS = json.loads((FIXTURES / "energy_charts_2026-10-11.json").read_text("utf-8"))
AWATTAR = json.loads((FIXTURES / "awattar_2026-10-11.json").read_text("utf-8"))


# ----------------------------------------------------------------------------- Energy-Charts
@respx.mock
def test_energy_charts_returns_quarter_hours_for_the_local_day():
    route = respx.get(energy_charts.URL).respond(json=ENERGY_CHARTS)
    prices = EnergyChartsProvider(httpx.Client()).fetch(START, END)

    params = route.calls.last.request.url.params
    assert params["bzn"] == "DE-LU"
    assert int(params["end"]) == int(END.timestamp()) - 1  # the API's end is inclusive
    assert len(prices) == 96
    assert prices[0].start == START
    assert prices[0].end - prices[0].start == timedelta(minutes=15)
    assert all(p.start.tzinfo is UTC for p in prices)


@respx.mock
@pytest.mark.parametrize("status", [404, 429, 500])
def test_energy_charts_errors_raise(status):
    respx.get(energy_charts.URL).respond(status_code=status)
    with pytest.raises((PriceUnavailableError, httpx.HTTPStatusError)):
        EnergyChartsProvider(httpx.Client()).fetch(START, END)


def test_energy_charts_skips_missing_values_and_checks_unit():
    data = energy_charts.EnergyChartsResponse(
        unix_seconds=[0, 900, 1800], price=[1.0, None, 3.0], unit="EUR / MWh"
    )
    assert [p.eur_per_mwh for p in energy_charts.parse(data)] == [1.0, 3.0]
    with pytest.raises(PriceUnavailableError, match="unit"):
        energy_charts.parse(data.model_copy(update={"unit": "EUR / kWh"}))


# ----------------------------------------------------------------------------- aWATTar
@respx.mock
def test_awattar_returns_hours_in_milliseconds_api():
    route = respx.get(awattar.URL).respond(json=AWATTAR)
    prices = AwattarProvider(httpx.Client()).fetch(START, END)
    assert route.calls.last.request.url.params["start"] == str(int(START.timestamp() * 1000))
    assert len(prices) == 24
    assert prices[0].start == START
    assert prices[0].end - prices[0].start == timedelta(hours=1)


@respx.mock
def test_awattar_empty_response_raises():
    respx.get(awattar.URL).respond(json={"object": "list", "data": []})
    with pytest.raises(PriceUnavailableError):
        AwattarProvider(httpx.Client()).fetch(START, END)


# ----------------------------------------------------------------------------- cache
def test_cache_serves_repeated_requests_without_calling_the_provider(tmp_path):
    inner = StaticPriceProvider(90)
    cached = CachedPriceProvider(inner, tmp_path / "cache.sqlite")
    first = cached.fetch(START, END)
    second = cached.fetch(START, END)
    assert inner.calls == 1
    assert first == second
    assert cached.name == inner.name
    assert cached.is_estimate is False


def test_cache_survives_a_new_instance_and_fetches_missing_days(tmp_path):
    path = tmp_path / "cache.sqlite"
    CachedPriceProvider(StaticPriceProvider(), path).fetch(START, END)
    inner = StaticPriceProvider()
    later = CachedPriceProvider(inner, path)
    later.fetch(START, END)
    assert inner.calls == 0
    later.fetch(*day_bounds_utc(DAY, DAY + timedelta(days=1), BERLIN))  # second day not cached yet
    assert inner.calls == 1


def test_cache_does_not_store_failures(tmp_path):
    cached = CachedPriceProvider(StaticPriceProvider(fail=True), tmp_path / "cache.sqlite")
    with pytest.raises(PriceUnavailableError):
        cached.fetch(START, END)
    assert cached.cached(START, END) == []


# ----------------------------------------------------------------------------- estimate
class HistoryProvider(StaticPriceProvider):
    """Hour-of-day profile: cheap at 13:00, expensive at 19:00; Sundays 50 EUR cheaper."""

    def fetch(self, start, end):
        self.calls += 1
        hours = hourly_prices(start, end, [0.0] * int((end - start) / timedelta(hours=1)))
        out = []
        for p in hours:
            local = p.start.astimezone(BERLIN)
            value = {13: 20.0, 19: 200.0}.get(local.hour, 100.0) - (50 if local.weekday() == 6 else 0)
            out.append(type(p)(p.start, p.end, value))
        return out


def test_estimate_follows_the_recent_daily_profile():
    target = day_bounds_utc(date(2026, 10, 14), date(2026, 10, 14), BERLIN)  # a Wednesday, two days ahead
    clock = lambda: datetime(2026, 10, 12, 9, tzinfo=BERLIN)  # noqa: E731
    prices = EstimatePriceProvider([HistoryProvider()], BERLIN, clock).fetch(*target)
    by_hour = {p.start.astimezone(BERLIN).hour: p.eur_per_mwh for p in prices}
    assert len(prices) == 24
    assert by_hour[13] < by_hour[3] < by_hour[19]


def test_estimate_weights_the_same_weekday_double():
    target = day_bounds_utc(date(2026, 10, 18), date(2026, 10, 18), BERLIN)  # a Sunday
    clock = lambda: datetime(2026, 10, 17, 9, tzinfo=BERLIN)  # noqa: E731
    prices = EstimatePriceProvider([HistoryProvider()], BERLIN, clock).fetch(*target)
    sunday_3am = next(p.eur_per_mwh for p in prices if p.start.astimezone(BERLIN).hour == 3)
    # 7 days of history: 1 Sunday (weight 2, 50 EUR) and 6 other days (100 EUR) -> (2*50 + 6*100) / 8
    assert sunday_3am == pytest.approx(87.5)


def test_estimate_needs_history():
    clock = lambda: datetime(2026, 10, 12, 9, tzinfo=BERLIN)  # noqa: E731
    with pytest.raises(PriceUnavailableError, match="history"):
        EstimatePriceProvider([StaticPriceProvider(fail=True)], BERLIN, clock).fetch(START, END)


# ----------------------------------------------------------------------------- chain
def test_multi_day_range_keeps_real_prices_where_published():
    class PublishedUntilTomorrow(StaticPriceProvider):
        def fetch(self, start, end):
            limit = day_bounds_utc(date(2026, 10, 3), date(2026, 10, 3), BERLIN)[1]
            prices = [p for p in super().fetch(start, end) if p.start < limit]
            if not prices:
                raise PriceUnavailableError("not published")
            return prices

    real = PublishedUntilTomorrow(100, name="real")
    service = PriceService(
        [real, SyntheticPriceProvider(BERLIN)], TariffSettings(), LocationSettings(), fixed_clock
    )
    curve = service.curve(*day_bounds_utc(date(2026, 10, 2), date(2026, 10, 4), BERLIN))
    assert curve.is_estimate
    assert curve.source.startswith("real + synthetic")
    real_days = {s.start.astimezone(BERLIN).date() for s in curve.slots if s.spot_eur_per_mwh == 100}
    assert real_days == {date(2026, 10, 2), date(2026, 10, 3)}


@respx.mock
def test_default_chain_falls_back_from_energy_charts_to_awattar(settings):
    respx.get(energy_charts.URL).respond(status_code=429)
    respx.get(awattar.URL).respond(json=AWATTAR)
    services = build_services(settings, http=httpx.Client(), clock=fixed_clock)
    result = services.prices.get_prices(DAY.isoformat())
    assert result["source"].startswith("aWATTar")
    assert result["is_estimate"] is False
    assert "Energy-Charts" in result["note"]


@respx.mock
def test_default_chain_ends_with_a_flagged_synthetic_profile_when_everything_is_down(settings):
    respx.get(energy_charts.URL).respond(status_code=503)
    respx.get(awattar.URL).respond(status_code=503)
    services = build_services(settings, http=httpx.Client(), clock=fixed_clock)
    result = services.prices.get_prices(DAY.isoformat())
    assert result["source"].startswith("synthetic")
    assert result["is_estimate"] is True


@respx.mock
def test_real_prices_carry_attribution(settings):
    respx.get(energy_charts.URL).respond(json=ENERGY_CHARTS)
    services = build_services(settings, http=httpx.Client(), clock=fixed_clock)
    result = services.prices.get_prices(DAY.isoformat())
    assert "CC BY 4.0" in result["attribution"]
    assert len(result["hourly_rates"]) == 24
