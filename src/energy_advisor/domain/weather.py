"""Weather data shared by the forecast tool, the sample-data generator and the optimizer."""

from dataclasses import dataclass
from datetime import datetime

# Sunlight multiplier per sky condition, relative to a clear sky.
CONDITION_SOLAR_FACTOR = {"sunny": 1.0, "partly_cloudy": 0.6, "cloudy": 0.3, "rainy": 0.12}
CONDITION_WMO_CODE = {"sunny": 0, "partly_cloudy": 2, "cloudy": 3, "rainy": 61}


@dataclass(frozen=True)
class HourlyWeather:
    """Weather during one hour. ``time`` is the aware UTC start of the hour."""

    time: datetime
    temperature_c: float
    irradiance_w_m2: float  # global horizontal irradiance (GHI), hourly mean
    condition: str
    humidity: float | None = None
    wind_speed: float | None = None


def condition_from_wmo(code: int | None) -> str:
    """Map a WMO weather code (used by Open-Meteo) to the 4 sky conditions used across the project."""
    if code is None:
        return "unknown"
    if code <= 1:
        return "sunny"  # clear / mainly clear
    if code == 2:
        return "partly_cloudy"
    if code in (3, 45, 48):
        return "cloudy"  # overcast / fog
    return "rainy"  # drizzle, rain, snow, showers, thunderstorm
