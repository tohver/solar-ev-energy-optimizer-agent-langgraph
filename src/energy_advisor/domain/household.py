"""Typical household load shape, shared by the history generator and the solar-surplus estimate."""

# Relative base load by local hour (mean ~1.0): low at night, peaks in the morning and evening.
BASE_LOAD_PROFILE = (
    0.6, 0.55, 0.5, 0.5, 0.5, 0.6, 0.9, 1.2, 1.1, 0.95, 0.9, 1.0,
    1.15, 1.0, 0.9, 0.9, 1.0, 1.3, 1.6, 1.7, 1.5, 1.3, 1.0, 0.8,
)  # fmt: skip


def base_load_kw(local_hour: int, average_kw: float) -> float:
    return average_kw * BASE_LOAD_PROFILE[local_hour]
