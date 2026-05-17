"""Annual temperature climate indices for milestone 005.

Pure xarray helpers that compute simple intensity and frequency indices
from M004 daily standard products. No NetCDF I/O, no CDS, no shapefile
or rioxarray dependency. ``xarray`` is imported lazily inside the
functions that need it so this module is cheap to import.

Index families implemented:

- ``Tmx``: annual max of daily ``tmean``.
- ``Tmn``: annual min of daily ``tmean``.
- ``TXx``: annual max of daily ``tmax``.
- ``TNn``: annual min of daily ``tmin``.
- ``DTR``: annual mean of daily ``(tmax - tmin)``.
- ``SU``: annual count of days with ``tmax > 30 degC`` (summer days).
- ``TR``: annual count of days with ``tmin > 20 degC`` (tropical nights).

Out of scope for this milestone:

- ``TX90p`` / ``TN10p`` (percentile indices),
- ``WSDI`` (warm spell duration index),
- precipitation and wind indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ANNUAL_FREQUENCY = "YE"
SUMMER_DAY_THRESHOLD_DEGC = 30.0
TROPICAL_NIGHT_THRESHOLD_DEGC = 20.0

INDEX_TMX = "Tmx"
INDEX_TMN = "Tmn"
INDEX_TXX = "TXx"
INDEX_TNN = "TNn"
INDEX_DTR = "DTR"
INDEX_SU = "SU"
INDEX_TR = "TR"


class TemperatureIndexError(ValueError):
    """Raised when a temperature index cannot be computed from the inputs."""


@dataclass(frozen=True)
class TemperatureIndexSpec:
    index_id: str
    required_variables: tuple[str, ...]
    description: str
    units: str


TEMPERATURE_INDEX_SPECS: tuple[TemperatureIndexSpec, ...] = (
    TemperatureIndexSpec(
        index_id=INDEX_TMX,
        required_variables=("tmean",),
        description="Annual maximum of daily mean temperature",
        units="degC",
    ),
    TemperatureIndexSpec(
        index_id=INDEX_TMN,
        required_variables=("tmean",),
        description="Annual minimum of daily mean temperature",
        units="degC",
    ),
    TemperatureIndexSpec(
        index_id=INDEX_TXX,
        required_variables=("tmax",),
        description="Annual maximum of daily maximum temperature",
        units="degC",
    ),
    TemperatureIndexSpec(
        index_id=INDEX_TNN,
        required_variables=("tmin",),
        description="Annual minimum of daily minimum temperature",
        units="degC",
    ),
    TemperatureIndexSpec(
        index_id=INDEX_DTR,
        required_variables=("tmax", "tmin"),
        description="Annual mean of daily temperature range (tmax - tmin)",
        units="degC",
    ),
    TemperatureIndexSpec(
        index_id=INDEX_SU,
        required_variables=("tmax",),
        description=f"Annual count of days with tmax > {SUMMER_DAY_THRESHOLD_DEGC} degC",
        units="days",
    ),
    TemperatureIndexSpec(
        index_id=INDEX_TR,
        required_variables=("tmin",),
        description=f"Annual count of days with tmin > {TROPICAL_NIGHT_THRESHOLD_DEGC} degC",
        units="days",
    ),
)

TEMPERATURE_INDEX_SPECS_BY_ID: dict[str, TemperatureIndexSpec] = {
    spec.index_id: spec for spec in TEMPERATURE_INDEX_SPECS
}


def _require_variable(data: dict[str, Any], variable: str, index_id: str):
    if variable not in data:
        raise TemperatureIndexError(
            f"index {index_id!r} requires variable {variable!r} but only "
            f"{sorted(data)} were provided"
        )
    return data[variable]


def _annual_max(da, *, index_id: str, units: str):
    out = da.resample(time=ANNUAL_FREQUENCY).max()
    out.name = index_id
    out.attrs = {"units": units, "description": TEMPERATURE_INDEX_SPECS_BY_ID[index_id].description}
    return out


def _annual_min(da, *, index_id: str, units: str):
    out = da.resample(time=ANNUAL_FREQUENCY).min()
    out.name = index_id
    out.attrs = {"units": units, "description": TEMPERATURE_INDEX_SPECS_BY_ID[index_id].description}
    return out


def compute_Tmx(daily: dict[str, Any]):
    tmean = _require_variable(daily, "tmean", INDEX_TMX)
    return _annual_max(tmean, index_id=INDEX_TMX, units="degC")


def compute_Tmn(daily: dict[str, Any]):
    tmean = _require_variable(daily, "tmean", INDEX_TMN)
    return _annual_min(tmean, index_id=INDEX_TMN, units="degC")


def compute_TXx(daily: dict[str, Any]):
    tmax = _require_variable(daily, "tmax", INDEX_TXX)
    return _annual_max(tmax, index_id=INDEX_TXX, units="degC")


def compute_TNn(daily: dict[str, Any]):
    tmin = _require_variable(daily, "tmin", INDEX_TNN)
    return _annual_min(tmin, index_id=INDEX_TNN, units="degC")


def compute_DTR(daily: dict[str, Any]):
    tmax = _require_variable(daily, "tmax", INDEX_DTR)
    tmin = _require_variable(daily, "tmin", INDEX_DTR)
    dtr_daily = tmax - tmin
    out = dtr_daily.resample(time=ANNUAL_FREQUENCY).mean()
    out.name = INDEX_DTR
    out.attrs = {
        "units": "degC",
        "description": TEMPERATURE_INDEX_SPECS_BY_ID[INDEX_DTR].description,
    }
    return out


def compute_SU(daily: dict[str, Any], *, threshold_degc: float = SUMMER_DAY_THRESHOLD_DEGC):
    import xarray as xr

    tmax = _require_variable(daily, "tmax", INDEX_SU)
    flag = (tmax > threshold_degc).astype("int32")
    out = flag.resample(time=ANNUAL_FREQUENCY).sum()
    out.name = INDEX_SU
    out.attrs = {
        "units": "days",
        "description": TEMPERATURE_INDEX_SPECS_BY_ID[INDEX_SU].description,
        "threshold_degc": threshold_degc,
    }
    return out


def compute_TR(daily: dict[str, Any], *, threshold_degc: float = TROPICAL_NIGHT_THRESHOLD_DEGC):
    import xarray as xr

    tmin = _require_variable(daily, "tmin", INDEX_TR)
    flag = (tmin > threshold_degc).astype("int32")
    out = flag.resample(time=ANNUAL_FREQUENCY).sum()
    out.name = INDEX_TR
    out.attrs = {
        "units": "days",
        "description": TEMPERATURE_INDEX_SPECS_BY_ID[INDEX_TR].description,
        "threshold_degc": threshold_degc,
    }
    return out


TEMPERATURE_INDEX_FUNCTIONS = {
    INDEX_TMX: compute_Tmx,
    INDEX_TMN: compute_Tmn,
    INDEX_TXX: compute_TXx,
    INDEX_TNN: compute_TNn,
    INDEX_DTR: compute_DTR,
    INDEX_SU: compute_SU,
    INDEX_TR: compute_TR,
}


def compute_index(index_id: str, daily: dict[str, Any]):
    """Dispatch to the correct compute function for ``index_id``."""
    if index_id not in TEMPERATURE_INDEX_FUNCTIONS:
        raise TemperatureIndexError(
            f"unknown temperature index_id {index_id!r}; expected one of "
            f"{sorted(TEMPERATURE_INDEX_FUNCTIONS)}"
        )
    return TEMPERATURE_INDEX_FUNCTIONS[index_id](daily)


__all__ = [
    "ANNUAL_FREQUENCY",
    "INDEX_DTR",
    "INDEX_SU",
    "INDEX_TMN",
    "INDEX_TMX",
    "INDEX_TNN",
    "INDEX_TR",
    "INDEX_TXX",
    "SUMMER_DAY_THRESHOLD_DEGC",
    "TEMPERATURE_INDEX_FUNCTIONS",
    "TEMPERATURE_INDEX_SPECS",
    "TEMPERATURE_INDEX_SPECS_BY_ID",
    "TROPICAL_NIGHT_THRESHOLD_DEGC",
    "TemperatureIndexError",
    "TemperatureIndexSpec",
    "compute_DTR",
    "compute_SU",
    "compute_TNn",
    "compute_TR",
    "compute_TXx",
    "compute_Tmn",
    "compute_Tmx",
    "compute_index",
]
