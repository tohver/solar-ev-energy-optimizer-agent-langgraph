"""Cheapest time to run a flexible load: deterministic, so the LLM never does schedule arithmetic itself.

Each time slot has a grid price and possibly a solar surplus. Solar energy is valued at the feed-in
tariff: using it at home means giving up that payment (its opportunity cost). Two modes:

- **contiguous** (dishwasher, washing machine): the device runs without pause at constant power; every
  possible start time is evaluated and the cheapest is chosen.
- **interruptible** (EV charging): energy can be split over any slots and the power can be modulated
  (a wallbox following the solar surplus); filling the cheapest capacity first is optimal because cost is
  linear in energy (a greedy algorithm).

The result is compared with the naive baseline "start immediately at the earliest start".
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

HOUR = timedelta(hours=1)


class ScheduleError(ValueError):
    pass


@dataclass(frozen=True)
class Slot:
    start: datetime
    end: datetime
    grid_price: float  # EUR/kWh
    solar_kw: float = 0.0  # average solar surplus power available in this slot

    @property
    def hours(self) -> float:
        return (self.end - self.start) / HOUR


@dataclass
class Block:
    """A continuous period in which the device runs."""

    start: datetime
    end: datetime
    energy_kwh: float = 0.0
    grid_kwh: float = 0.0
    solar_kwh: float = 0.0
    cost_eur: float = 0.0

    @property
    def average_kw(self) -> float:
        hours = (self.end - self.start) / HOUR
        return self.energy_kwh / hours if hours else 0.0


@dataclass
class Schedule:
    blocks: list[Block]
    energy_kwh: float
    cost_eur: float
    grid_kwh: float
    solar_kwh: float
    baseline_cost_eur: float
    notes: list[str] = field(default_factory=list)

    @property
    def savings_eur(self) -> float:
        return self.baseline_cost_eur - self.cost_eur

    @property
    def average_price(self) -> float:
        return self.cost_eur / self.energy_kwh if self.energy_kwh else 0.0


def slot_energy_cost(
    slot: Slot, energy_kwh: float, feed_in: float, use_solar: bool
) -> tuple[float, float, float]:
    """(cost, grid kWh, solar kWh) of drawing ``energy_kwh`` within one slot, cheapest source first."""
    solar_capacity = slot.solar_kw * slot.hours if use_solar else 0.0
    if feed_in <= slot.grid_price:
        solar = min(energy_kwh, solar_capacity)
        grid = energy_kwh - solar
    else:  # grid cheaper than giving up feed-in (deeply negative prices): take the grid power
        solar, grid = 0.0, energy_kwh
    return grid * slot.grid_price + solar * feed_in, grid, solar


def _window_slots(slots: list[Slot], earliest: datetime, deadline: datetime) -> list[Slot]:
    """Slots clipped to ``[earliest, deadline)``; the first and last one may be partial."""
    clipped = []
    for s in sorted(slots, key=lambda s: s.start):
        start, end = max(s.start, earliest), min(s.end, deadline)
        if start < end:
            clipped.append(Slot(start, end, s.grid_price, s.solar_kw))
    return clipped


def _run_contiguous(
    slots: list[Slot], start: datetime, energy_kwh: float, power_kw: float, feed_in: float, use_solar: bool
) -> list[Block] | None:
    """Cost of running from ``start`` at constant power until the energy is delivered; None if it does not fit."""
    end = start + timedelta(hours=energy_kwh / power_kw)
    block = Block(start, end)
    covered = timedelta()
    for s in slots:
        overlap_start, overlap_end = max(s.start, start), min(s.end, end)
        if overlap_start >= overlap_end:
            continue
        covered += overlap_end - overlap_start
        energy = power_kw * ((overlap_end - overlap_start) / HOUR)
        part = Slot(overlap_start, overlap_end, s.grid_price, s.solar_kw)
        cost, grid, solar = slot_energy_cost(part, energy, feed_in, use_solar)
        block.energy_kwh += energy
        block.grid_kwh += grid
        block.solar_kwh += solar
        block.cost_eur += cost
    if covered < (end - start) - timedelta(seconds=1):
        return None
    return [block]


def _fill_interruptible(
    slots: list[Slot], energy_kwh: float, power_kw: float, feed_in: float, use_solar: bool
) -> list[Block]:
    # every slot offers up to two "chunks": solar capacity (at feed-in) and the rest of the power (at grid price)
    chunks = []
    for i, s in enumerate(slots):
        capacity = power_kw * s.hours
        solar = min(capacity, s.solar_kw * s.hours) if use_solar and feed_in <= s.grid_price else 0.0
        if solar > 0:
            chunks.append((feed_in, s.start, i, "solar", solar))
        if capacity - solar > 0:
            chunks.append((s.grid_price, s.start, i, "grid", capacity - solar))
    used: dict[int, dict[str, float]] = {}
    remaining = energy_kwh
    for _price, _start, i, source, amount in sorted(chunks, key=lambda c: (c[0], c[1])):
        if remaining <= 1e-9:
            break
        take = min(amount, remaining)
        used.setdefault(i, {"solar": 0.0, "grid": 0.0})[source] += take
        remaining -= take

    # An interruptible load (EV) can modulate its power: a partly used slot means running at reduced
    # average power for the whole slot (e.g. following the solar surplus), not a short burst.
    blocks: list[Block] = []
    for i in sorted(used):
        s, parts = slots[i], used[i]
        if blocks and blocks[-1].end == s.start:
            b = blocks[-1]
            b.end = s.end
        else:
            b = Block(s.start, s.end)
            blocks.append(b)
        b.energy_kwh += parts["solar"] + parts["grid"]
        b.grid_kwh += parts["grid"]
        b.solar_kwh += parts["solar"]
        b.cost_eur += parts["solar"] * feed_in + parts["grid"] * s.grid_price
    return blocks


def find_cheapest_schedule(
    slots: list[Slot],
    energy_kwh: float,
    power_kw: float,
    earliest: datetime,
    deadline: datetime,
    feed_in: float,
    interruptible: bool = True,
    use_solar: bool = True,
) -> Schedule:
    if energy_kwh <= 0 or power_kw <= 0:
        raise ScheduleError("energy_kwh and power_kw must be positive")
    if deadline <= earliest:
        raise ScheduleError("The deadline must be after the earliest start")
    window = _window_slots(slots, earliest, deadline)
    available_h = sum(s.hours for s in window)
    if available_h * power_kw < energy_kwh - 1e-6:
        raise ScheduleError(
            f"Not enough time: at {power_kw} kW the window delivers at most {available_h * power_kw:.1f} kWh "
            f"of the {energy_kwh} kWh needed."
        )

    baseline = _run_contiguous(window, window[0].start, energy_kwh, power_kw, feed_in, use_solar)
    if interruptible:
        blocks = _fill_interruptible(window, energy_kwh, power_kw, feed_in, use_solar)
    else:
        candidates = [
            run
            for s in window
            if (run := _run_contiguous(window, s.start, energy_kwh, power_kw, feed_in, use_solar)) is not None
        ]
        if not candidates:
            raise ScheduleError("The device cannot run without pause inside the window.")
        blocks = min(candidates, key=lambda run: (round(run[0].cost_eur, 6), run[0].start))

    return Schedule(
        blocks=blocks,
        energy_kwh=sum(b.energy_kwh for b in blocks),
        cost_eur=sum(b.cost_eur for b in blocks),
        grid_kwh=sum(b.grid_kwh for b in blocks),
        solar_kwh=sum(b.solar_kwh for b in blocks),
        baseline_cost_eur=sum(b.cost_eur for b in baseline) if baseline else float("nan"),
    )
