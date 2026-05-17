"""Annual precipitation climate indices for milestone 007.

Pure xarray helpers that compute the seven legacy ETCCDI-style
precipitation indices from M006 daily ``pr`` standard products. No
NetCDF I/O, no CDS, no shapefile / rioxarray / icclim / xclim / dask
dependency. ``numpy`` and ``xarray`` are imported lazily inside the
helpers that need them.

Indices implemented (matching
``90_legacy_review/OtraSenda-Clima-main/Clima_obs/Markdowns/3_README-calculo_indices_climaticos_lluvia.md``):

- ``PRCPTOT``: annual sum of wet-day precipitation (``pr >= 1.0 mm``);
  units ``mm``.
- ``RX1day``: annual maximum daily precipitation; units ``mm``.
- ``R95p``: annual sum of precipitation on days where ``pr`` exceeds
  the period-wide wet-day p95 (``RRwn95``); units ``mm``. The
  baseline is the full extent of the daily ``pr`` series passed in;
  attrs record that policy.
- ``CDD``: annual max consecutive run of dry days
  (``pr < 1.0 mm``); units ``days``.
- ``CWD``: annual max consecutive run of wet days
  (``pr >= 1.0 mm``); units ``days``.
- ``R10mm``: annual count of days with ``pr >= 10.0 mm``;
  units ``days``.
- ``R20mm``: annual count of days with ``pr >= 20.0 mm``;
  units ``days``.

Out of scope: spell-duration indices (``WSDI``), temperature
percentile indices (``TX90p`` / ``TN10p``), wind indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ANNUAL_FREQUENCY = "YE"
WET_DAY_THRESHOLD_MM = 1.0
R10_THRESHOLD_MM = 10.0
R20_THRESHOLD_MM = 20.0
R95P_PERCENTILE = 0.95

R95P_BASELINE_POLICY = "period_full_extent"

INDEX_PRCPTOT = "PRCPTOT"
INDEX_RX1DAY = "RX1day"
INDEX_R95P = "R95p"
INDEX_CDD = "CDD"
INDEX_CWD = "CWD"
INDEX_R10MM = "R10mm"
INDEX_R20MM = "R20mm"


class PrecipitationIndexError(ValueError):
    """Raised when a precipitation index cannot be computed from the inputs."""


@dataclass(frozen=True)
class PrecipitationIndexSpec:
    index_id: str
    required_variables: tuple[str, ...]
    description: str
    units: str


PRECIPITATION_INDEX_SPECS: tuple[PrecipitationIndexSpec, ...] = (
    PrecipitationIndexSpec(
        index_id=INDEX_PRCPTOT,
        required_variables=("pr",),
        description=f"Annual sum of wet-day precipitation (pr >= {WET_DAY_THRESHOLD_MM} mm)",
        units="mm",
    ),
    PrecipitationIndexSpec(
        index_id=INDEX_RX1DAY,
        required_variables=("pr",),
        description="Annual maximum daily precipitation",
        units="mm",
    ),
    PrecipitationIndexSpec(
        index_id=INDEX_R95P,
        required_variables=("pr",),
        description=(
            "Annual sum of precipitation on days where pr exceeds the "
            "period-wide wet-day p95 (RRwn95)"
        ),
        units="mm",
    ),
    PrecipitationIndexSpec(
        index_id=INDEX_CDD,
        required_variables=("pr",),
        description=f"Annual max consecutive run of dry days (pr < {WET_DAY_THRESHOLD_MM} mm)",
        units="days",
    ),
    PrecipitationIndexSpec(
        index_id=INDEX_CWD,
        required_variables=("pr",),
        description=f"Annual max consecutive run of wet days (pr >= {WET_DAY_THRESHOLD_MM} mm)",
        units="days",
    ),
    PrecipitationIndexSpec(
        index_id=INDEX_R10MM,
        required_variables=("pr",),
        description=f"Annual count of days with pr >= {R10_THRESHOLD_MM} mm",
        units="days",
    ),
    PrecipitationIndexSpec(
        index_id=INDEX_R20MM,
        required_variables=("pr",),
        description=f"Annual count of days with pr >= {R20_THRESHOLD_MM} mm",
        units="days",
    ),
)

PRECIPITATION_INDEX_SPECS_BY_ID: dict[str, PrecipitationIndexSpec] = {
    spec.index_id: spec for spec in PRECIPITATION_INDEX_SPECS
}


def _require_pr(daily: dict[str, Any], index_id: str):
    if "pr" not in daily:
        raise PrecipitationIndexError(
            f"index {index_id!r} requires variable 'pr' but only "
            f"{sorted(daily)} were provided"
        )
    return daily["pr"]


def _annual_resample(da):
    return da.resample(time=ANNUAL_FREQUENCY)


def _set_attrs(da, *, index_id: str, units: str, extra: dict[str, Any] | None = None):
    spec = PRECIPITATION_INDEX_SPECS_BY_ID[index_id]
    attrs = {"units": units, "description": spec.description}
    if extra:
        attrs.update(extra)
    da.attrs = attrs
    da.name = index_id
    return da


def compute_PRCPTOT(daily: dict[str, Any]):
    """Annual sum of wet-day precipitation (pr >= 1 mm).

    Mask-preserving: cells where the input ``pr`` is NaN for an entire
    resample bucket stay NaN in the output (``min_count=1``). Valid
    cells with zero wet days still produce ``0.0`` (an honest count of
    no wet-day precipitation, not a missing observation).
    """
    pr = _require_pr(daily, INDEX_PRCPTOT)
    # ``pr.where(pr >= 1.0, 0.0)`` alone would convert NaN inputs to 0.0
    # because NaN comparisons are False. Restore the NaN mask afterwards.
    wet = pr.where(pr >= WET_DAY_THRESHOLD_MM, 0.0).where(~pr.isnull())
    out = _annual_resample(wet).sum(skipna=True, min_count=1)
    return _set_attrs(
        out,
        index_id=INDEX_PRCPTOT,
        units="mm",
        extra={"wet_day_threshold_mm": WET_DAY_THRESHOLD_MM},
    )


def compute_RX1day(daily: dict[str, Any]):
    """Annual maximum daily precipitation."""
    pr = _require_pr(daily, INDEX_RX1DAY)
    out = _annual_resample(pr).max()
    return _set_attrs(out, index_id=INDEX_RX1DAY, units="mm")


def compute_R95p(daily: dict[str, Any]):
    """Annual sum of precipitation on days exceeding the period-wide wet-day p95.

    Mask-preserving: cells where the input ``pr`` is NaN for an entire
    resample bucket stay NaN. A valid cell that never has wet days
    (RRwn95 is NaN for that cell) produces ``0.0`` -- the legacy
    ETCCDI convention -- since the cell is not missing data, it just
    has no extreme days to sum.
    """
    pr = _require_pr(daily, INDEX_R95P)
    wet_days_ref = pr.where(pr >= WET_DAY_THRESHOLD_MM)
    rrwn95 = wet_days_ref.quantile(R95P_PERCENTILE, dim="time", skipna=True)
    # Drop the scalar quantile coord so the broadcast against ``pr`` is clean
    # and the saved attribute is the policy text, not the percentile value.
    rrwn95 = rrwn95.reset_coords(drop=True) if "quantile" in rrwn95.coords else rrwn95
    # ``pr.where(pr > rrwn95, 0.0)`` alone would convert NaN inputs to 0.0
    # because NaN comparisons are False. Restore the NaN mask after.
    extreme = pr.where(pr > rrwn95, 0.0).where(~pr.isnull())
    out = _annual_resample(extreme).sum(skipna=True, min_count=1)
    return _set_attrs(
        out,
        index_id=INDEX_R95P,
        units="mm",
        extra={
            "wet_day_threshold_mm": WET_DAY_THRESHOLD_MM,
            "percentile": R95P_PERCENTILE,
            "baseline_period_policy": R95P_BASELINE_POLICY,
        },
    )


def _max_run_length_1d(arr_1d) -> int:
    """Longest run of ``True`` along a 1D boolean array."""
    max_run = 0
    current = 0
    for value in arr_1d:
        if bool(value):
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return int(max_run)


def _max_run_reducer_along_axis(arr, axis: int):
    """xarray-compatible reducer: longest run of ``True`` along ``axis``.

    ``arr`` is the underlying numpy array for one resample/groupby
    bucket. ``axis`` is the position of the time dimension.
    """
    import numpy as np

    return np.apply_along_axis(_max_run_length_1d, axis, arr)


def _annual_max_run(da, *, condition_predicate, index_id: str, units: str, extra: dict[str, Any]):
    """Mask-preserving annual max run length.

    NaN inputs make the predicate False (NaN comparisons return False),
    so a fully-masked cell would otherwise yield a spurious run length
    of zero. Compute the count of valid days per year-cell and set the
    run length to NaN wherever no valid input exists.
    """
    condition = condition_predicate(da)
    out = _annual_resample(condition).reduce(_max_run_reducer_along_axis, dim="time")
    valid_per_year = _annual_resample(~da.isnull()).sum()
    out = out.astype("float32").where(valid_per_year > 0)
    return _set_attrs(out, index_id=index_id, units=units, extra=extra)


def compute_CDD(daily: dict[str, Any]):
    """Annual max consecutive run of dry days (pr < 1 mm)."""
    pr = _require_pr(daily, INDEX_CDD)
    return _annual_max_run(
        pr,
        condition_predicate=lambda da: (da < WET_DAY_THRESHOLD_MM),
        index_id=INDEX_CDD,
        units="days",
        extra={"wet_day_threshold_mm": WET_DAY_THRESHOLD_MM},
    )


def compute_CWD(daily: dict[str, Any]):
    """Annual max consecutive run of wet days (pr >= 1 mm)."""
    pr = _require_pr(daily, INDEX_CWD)
    return _annual_max_run(
        pr,
        condition_predicate=lambda da: (da >= WET_DAY_THRESHOLD_MM),
        index_id=INDEX_CWD,
        units="days",
        extra={"wet_day_threshold_mm": WET_DAY_THRESHOLD_MM},
    )


def _annual_threshold_count(da, *, threshold_mm: float, index_id: str):
    """Mask-preserving annual count of days above ``threshold_mm``.

    Boolean comparisons with NaN return False, so a fully-masked cell
    would otherwise be counted as zero. Restore the NaN mask on the
    flag and use ``min_count=1`` so all-NaN year-cells stay NaN.
    """
    flag = (da >= threshold_mm).astype("float32").where(~da.isnull())
    out = _annual_resample(flag).sum(skipna=True, min_count=1)
    return _set_attrs(
        out,
        index_id=index_id,
        units="days",
        extra={"threshold_mm": threshold_mm},
    )


def compute_R10mm(daily: dict[str, Any]):
    """Annual count of days with pr >= 10 mm."""
    pr = _require_pr(daily, INDEX_R10MM)
    return _annual_threshold_count(pr, threshold_mm=R10_THRESHOLD_MM, index_id=INDEX_R10MM)


def compute_R20mm(daily: dict[str, Any]):
    """Annual count of days with pr >= 20 mm."""
    pr = _require_pr(daily, INDEX_R20MM)
    return _annual_threshold_count(pr, threshold_mm=R20_THRESHOLD_MM, index_id=INDEX_R20MM)


PRECIPITATION_INDEX_FUNCTIONS = {
    INDEX_PRCPTOT: compute_PRCPTOT,
    INDEX_RX1DAY: compute_RX1day,
    INDEX_R95P: compute_R95p,
    INDEX_CDD: compute_CDD,
    INDEX_CWD: compute_CWD,
    INDEX_R10MM: compute_R10mm,
    INDEX_R20MM: compute_R20mm,
}


def compute_precipitation_index(index_id: str, daily: dict[str, Any]):
    """Dispatch to the correct compute function for ``index_id``."""
    if index_id not in PRECIPITATION_INDEX_FUNCTIONS:
        raise PrecipitationIndexError(
            f"unknown precipitation index_id {index_id!r}; expected one of "
            f"{sorted(PRECIPITATION_INDEX_FUNCTIONS)}"
        )
    return PRECIPITATION_INDEX_FUNCTIONS[index_id](daily)


__all__ = [
    "ANNUAL_FREQUENCY",
    "INDEX_CDD",
    "INDEX_CWD",
    "INDEX_PRCPTOT",
    "INDEX_R10MM",
    "INDEX_R20MM",
    "INDEX_R95P",
    "INDEX_RX1DAY",
    "PRECIPITATION_INDEX_FUNCTIONS",
    "PRECIPITATION_INDEX_SPECS",
    "PRECIPITATION_INDEX_SPECS_BY_ID",
    "PrecipitationIndexError",
    "PrecipitationIndexSpec",
    "R10_THRESHOLD_MM",
    "R20_THRESHOLD_MM",
    "R95P_BASELINE_POLICY",
    "R95P_PERCENTILE",
    "WET_DAY_THRESHOLD_MM",
    "compute_CDD",
    "compute_CWD",
    "compute_PRCPTOT",
    "compute_R10mm",
    "compute_R20mm",
    "compute_R95p",
    "compute_RX1day",
    "compute_precipitation_index",
]
