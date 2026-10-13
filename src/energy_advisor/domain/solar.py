"""Sun position, clear-sky irradiance and a simple PV yield model."""

import math
from datetime import datetime, timedelta

from energy_advisor.timeutils import to_utc

SOLAR_CONSTANT_FACTOR = 1098.0  # Haurwitz clear-sky model coefficient (W/m2)


def solar_elevation_deg(when: datetime, latitude: float, longitude: float) -> float:
    """Approximate sun elevation above the horizon (degrees). Accurate to about 1 degree, enough here."""
    t = to_utc(when)
    day_of_year = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60
    gamma = 2 * math.pi / 365 * (day_of_year - 1 + (hour - 12) / 24)  # fractional year (radians)
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
    )
    equation_of_time_min = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    solar_time_min = hour * 60 + equation_of_time_min + 4 * longitude
    hour_angle = math.radians(solar_time_min / 4 - 180)
    lat = math.radians(latitude)
    sin_elevation = math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(declination) * math.cos(
        hour_angle
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elevation))))


def clear_sky_ghi(elevation_deg: float) -> float:
    """Global horizontal irradiance under a clear sky (Haurwitz model), W/m2."""
    if elevation_deg <= 0:
        return 0.0
    cos_zenith = math.sin(math.radians(elevation_deg))
    return SOLAR_CONSTANT_FACTOR * cos_zenith * math.exp(-0.057 / cos_zenith)


def hourly_clear_sky_ghi(hour_start: datetime, latitude: float, longitude: float) -> float:
    """Mean clear-sky irradiance over one hour (sampled at quarter past, half past and quarter to)."""
    samples = [hour_start + timedelta(minutes=m) for m in (15, 30, 45)]
    return sum(clear_sky_ghi(solar_elevation_deg(s, latitude, longitude)) for s in samples) / len(samples)


def pv_energy_kwh(
    irradiance_w_m2: float, peak_kw: float, performance_ratio: float, hours: float = 1.0
) -> float:
    """PV yield: installed kWp x irradiance relative to standard test conditions (1000 W/m2) x losses.

    Uses horizontal irradiance, so tilt and orientation are ignored (a conservative simplification).
    """
    return max(0.0, peak_kw * (irradiance_w_m2 or 0.0) / 1000 * performance_ratio * hours)
