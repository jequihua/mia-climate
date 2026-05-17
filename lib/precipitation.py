"""Hourly-to-daily precipitation transformations for milestone 006.

Pure xarray helpers that convert ERA5-Land hourly accumulated total
precipitation (``tp``, meters) into a daily standard product
(``pr``, mm/day). All scientific operations are isolated as small
functions so the pipeline composition is reviewable and testable
against synthetic NetCDF fixtures.

The first supported policy is ``legacy_utc_minus_7``, which matches
the documented intent of the legacy script
(`90_legacy_review/OtraSenda-Clima-main/Clima_obs/SCRIPTs/procesamiento_nombre_recorte_conversion_precip.py`)
without reproducing its `time`/`valid_time` coordinate-mixing bug or
its double-diff dance. The legacy `Etc/GMT+7` timezone is reduced to
a deterministic ``-7h`` clock shift on the time coordinate; that is
the part the legacy code actually relied on for daily aggregation.

Heavy imports (``numpy``, ``xarray``, ``pandas``) are loaded lazily
inside the helpers that need them so ``import lib.precipitation``
costs nothing for planning.
"""

from __future__ import annotations

from typing import Any

PR_PROJECT_VARIABLE = "pr"
PR_SOURCE_VARIABLE_CANDIDATES = ("tp", "total_precipitation")

PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7 = "legacy_utc_minus_7"
SUPPORTED_PRECIPITATION_POLICIES = frozenset(
    {PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7}
)

METERS_TO_MILLIMETERS = 1000.0
LEGACY_UTC_OFFSET_HOURS = -7
SOURCE_UNITS_METERS = "m"
DAILY_UNITS_MM = "mm/day"


class PrecipitationError(ValueError):
    """Raised when a precipitation transformation cannot be applied."""


# ---------------------------------------------------------------------------
# Variable normalization
# ---------------------------------------------------------------------------


def rename_to_pr(dataset, *, source_variable: str | None = None):
    """Rename the source CDS precipitation variable to ``pr``.

    If ``source_variable`` is ``None``, pick the first candidate from
    ``PR_SOURCE_VARIABLE_CANDIDATES`` that is present, then fall back
    to inferring a single data variable. Errors are explicit when no
    candidate is found or when multiple data variables are present
    and none of them is a known candidate.
    """
    if source_variable is None:
        for candidate in PR_SOURCE_VARIABLE_CANDIDATES:
            if candidate in dataset.data_vars:
                source_variable = candidate
                break
    if source_variable is None:
        data_vars = list(dataset.data_vars)
        if len(data_vars) == 1:
            source_variable = data_vars[0]
        else:
            raise PrecipitationError(
                "could not infer precipitation source variable: dataset has "
                f"{len(data_vars)} data variables ({data_vars}); expected one of "
                f"{list(PR_SOURCE_VARIABLE_CANDIDATES)}"
            )
    if source_variable not in dataset.data_vars:
        raise PrecipitationError(
            f"precipitation source variable {source_variable!r} not found; "
            f"available data variables: {list(dataset.data_vars)}"
        )
    if source_variable == PR_PROJECT_VARIABLE:
        return dataset
    if PR_PROJECT_VARIABLE in dataset.data_vars:
        raise PrecipitationError(
            f"cannot rename {source_variable!r} to {PR_PROJECT_VARIABLE!r}: "
            f"{PR_PROJECT_VARIABLE!r} already exists"
        )
    return dataset.rename({source_variable: PR_PROJECT_VARIABLE})


# ---------------------------------------------------------------------------
# Time-coordinate shift
# ---------------------------------------------------------------------------


def apply_utc_offset(dataset, *, offset_hours: int):
    """Shift the ``time`` coordinate by ``offset_hours`` hours.

    Used to express UTC hourly accumulations on a fixed-offset local
    clock before daily aggregation. The shift is naive (no DST), which
    is the explicit behavior the legacy chain relied on.
    """
    import numpy as np

    if "time" not in dataset.coords and "time" not in dataset.dims:
        raise PrecipitationError(
            "dataset must carry a 'time' coordinate before applying a UTC offset; "
            "call normalize_dimensions() first"
        )
    delta = np.timedelta64(offset_hours, "h")
    shifted = dataset["time"].values + delta
    return dataset.assign_coords(time=shifted)


# ---------------------------------------------------------------------------
# Deaccumulation and daily aggregation
# ---------------------------------------------------------------------------


def deaccumulate_hourly_tp(da_accumulated):
    """Return hourly increments from an hourly accumulated tp series.

    ERA5-Land ``tp`` is an hourly accumulation in meters that resets at
    a forecast-day boundary. A simple ``diff`` along ``time`` yields
    the hourly precipitation increment at every step except the reset
    (which becomes negative). Clamping the result to ``>= 0`` is
    documented in the legacy markdown as the safe-by-default handling
    of the reset and of tiny negative artifacts; this is the same
    clamp the new helper applies. The leading hour is dropped because
    diff loses one sample; that limitation is recorded in
    `90_legacy_review/legacy_risks.md`.
    """
    if "time" not in da_accumulated.dims:
        raise PrecipitationError(
            "expected a 'time' dimension on the accumulated tp DataArray"
        )
    increments = da_accumulated.diff(dim="time")
    return increments.where(increments >= 0, 0)


def aggregate_to_daily_mm(da_hourly_increment_m):
    """Sum hourly increments per local-clock day and convert m -> mm.

    The result variable preserves the input name (typically ``pr``).
    Daily ``time`` stamps come from xarray's ``"1D"`` resample (one
    timestamp per day at 00:00). The output records the conversion
    factor and source units as attributes so a future reader can
    audit the math without re-running the pipeline.
    """
    if "time" not in da_hourly_increment_m.dims:
        raise PrecipitationError(
            "expected a 'time' dimension on the hourly increment DataArray"
        )
    daily_m = da_hourly_increment_m.resample(time="1D").sum()
    daily_mm = daily_m * METERS_TO_MILLIMETERS
    daily_mm.name = da_hourly_increment_m.name or PR_PROJECT_VARIABLE
    daily_mm.attrs = dict(da_hourly_increment_m.attrs)
    daily_mm.attrs["units"] = DAILY_UNITS_MM
    daily_mm.attrs["source_units"] = SOURCE_UNITS_METERS
    daily_mm.attrs["preprocessing_conversion_factor"] = METERS_TO_MILLIMETERS
    return daily_mm


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess_precipitation_dataset(
    dataset,
    *,
    policy: str,
    region_geojson: dict[str, Any] | None = None,
    source_variable: str | None = None,
):
    """Apply the requested precipitation policy to an hourly ERA5-Land dataset.

    Steps for ``legacy_utc_minus_7``:

    1. ``normalize_dimensions`` -- rename ``valid_time``/``latitude``/
       ``longitude`` to ``time``/``lat``/``lon`` (reuses M004 helper).
    2. ``rename_to_pr`` -- rename the source CDS variable to ``pr``.
    3. ``apply_utc_offset(offset_hours=-7)`` -- fixed clock shift,
       matching the legacy ``Etc/GMT+7`` choice but without the
       pandas tz dance that confused the legacy chain.
    4. ``deaccumulate_hourly_tp`` -- diff along time, clamp negatives
       to zero (handles the daily reset and float artifacts).
    5. ``aggregate_to_daily_mm`` -- resample daily sum, multiply by
       1000 m -> mm, record ``units = "mm/day"`` and provenance attrs.
    6. ``apply_region_mask`` -- polygon mask if ``region_geojson``
       was provided (reuses M004 helper).
    """
    if policy not in SUPPORTED_PRECIPITATION_POLICIES:
        raise PrecipitationError(
            f"unsupported precipitation_policy {policy!r}; expected one of "
            f"{sorted(SUPPORTED_PRECIPITATION_POLICIES)}"
        )
    # Lazy imports of M004 helpers so this module stays cheap to load.
    from .preprocessing import (  # noqa: PLC0415 - lazy import on purpose
        apply_region_mask,
        normalize_dimensions,
    )

    dataset = normalize_dimensions(dataset)
    dataset = rename_to_pr(dataset, source_variable=source_variable)
    dataset = apply_utc_offset(dataset, offset_hours=LEGACY_UTC_OFFSET_HOURS)
    increment_m = deaccumulate_hourly_tp(dataset[PR_PROJECT_VARIABLE])
    daily_mm = aggregate_to_daily_mm(increment_m)
    daily_mm.attrs["precipitation_policy"] = policy
    out = daily_mm.to_dataset(name=PR_PROJECT_VARIABLE)
    if region_geojson is not None:
        out = apply_region_mask(out, region_geojson=region_geojson)
    return out


__all__ = [
    "DAILY_UNITS_MM",
    "LEGACY_UTC_OFFSET_HOURS",
    "METERS_TO_MILLIMETERS",
    "PR_PROJECT_VARIABLE",
    "PR_SOURCE_VARIABLE_CANDIDATES",
    "PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7",
    "PrecipitationError",
    "SOURCE_UNITS_METERS",
    "SUPPORTED_PRECIPITATION_POLICIES",
    "aggregate_to_daily_mm",
    "apply_utc_offset",
    "deaccumulate_hourly_tp",
    "preprocess_precipitation_dataset",
    "rename_to_pr",
]
