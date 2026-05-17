"""ERA5-Land download planning helpers.

Pure helpers for milestone 002. They build a deterministic, reviewable
"download plan" manifest from a region manifest produced by milestone 001.

No code in this module talks to Copernicus, imports ``cdsapi``, opens
NetCDF files, or hits the network. The planned request payloads are
designed so a future adapter can hand them to the CDS API without
recomputation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .regions import (
    REGION_MANIFEST_REQUIRED_FIELDS,
    RegionValidationError,
    load_region_manifest,
)

DAILY_STATISTICS_DATASET = "derived-era5-land-daily-statistics"
HOURLY_PRECIPITATION_DATASET = "reanalysis-era5-land"

DAILY_STATISTICS_FREQUENCY = "6_hourly"
DAILY_STATISTICS_TIME_ZONE = "utc-07:00"

HOURLY_PRECIPITATION_DATA_FORMAT = "netcdf"
HOURLY_PRECIPITATION_DOWNLOAD_FORMAT = "unarchived"

MIN_PLANNABLE_YEAR = 1950
MAX_PLANNABLE_YEAR = 2100
PLANNABLE_YEAR_WINDOW_RATIONALE = (
    "ERA5-Land begins in 1950-01-01 per the CDS dataset documentation; an "
    "upper bound of 2100 is a generous future-proof cap well beyond any "
    "near-term run. This is a planning guard against typos and inverted "
    "ranges, not a CDS availability claim."
)

MONTHS = tuple(f"{m:02d}" for m in range(1, 13))
DAYS = tuple(f"{d:02d}" for d in range(1, 32))
HOURS = tuple(f"{h:02d}:00" for h in range(0, 24))

PRECIPITATION_SEMESTERS = (
    {"chunk_id": "H1", "months": tuple(f"{m:02d}" for m in range(1, 7))},
    {"chunk_id": "H2", "months": tuple(f"{m:02d}" for m in range(7, 13))},
)


@dataclass(frozen=True)
class DailyStatisticVariable:
    project_variable: str
    cds_variable: str
    daily_statistic: str


DAILY_STATISTIC_VARIABLES: tuple[DailyStatisticVariable, ...] = (
    DailyStatisticVariable("tmax", "2m_temperature", "daily_maximum"),
    DailyStatisticVariable("tmin", "2m_temperature", "daily_minimum"),
    DailyStatisticVariable("tmean", "2m_temperature", "daily_mean"),
    DailyStatisticVariable("u10m", "10m_u_component_of_wind", "daily_mean"),
    DailyStatisticVariable("v10m", "10m_v_component_of_wind", "daily_mean"),
)

HOURLY_PRECIPITATION_PROJECT_VARIABLE = "pr"
HOURLY_PRECIPITATION_CDS_VARIABLE = "total_precipitation"


class DownloadPlanError(ValueError):
    """Raised when a download plan cannot be constructed."""


def validate_year_range(start_year: int, end_year: int) -> tuple[int, int]:
    """Confirm ``start_year`` and ``end_year`` form a usable inclusive range."""
    if not isinstance(start_year, int) or isinstance(start_year, bool):
        raise DownloadPlanError(f"start_year must be an int, got {type(start_year).__name__}")
    if not isinstance(end_year, int) or isinstance(end_year, bool):
        raise DownloadPlanError(f"end_year must be an int, got {type(end_year).__name__}")
    if start_year > end_year:
        raise DownloadPlanError(
            f"start_year ({start_year}) must be <= end_year ({end_year})"
        )
    if start_year < MIN_PLANNABLE_YEAR or end_year > MAX_PLANNABLE_YEAR:
        raise DownloadPlanError(
            f"year range [{start_year}, {end_year}] is outside the plannable window "
            f"[{MIN_PLANNABLE_YEAR}, {MAX_PLANNABLE_YEAR}]"
        )
    return start_year, end_year


def _check_region_manifest(manifest: dict[str, Any]) -> None:
    missing = [f for f in REGION_MANIFEST_REQUIRED_FIELDS if f not in manifest]
    if missing:
        raise DownloadPlanError(
            f"region manifest is missing required fields: {missing}"
        )


def _daily_statistics_request_id(project_variable: str, year: int) -> str:
    return f"era5_daily_stats__{project_variable}__{year}"


def _hourly_precipitation_request_id(year: int, chunk_id: str) -> str:
    return f"era5_hourly_pr__{year}_{chunk_id}"


def _daily_statistics_output_path(project_variable: str, year: int) -> str:
    return f"raw/era5_land/daily_statistics/{project_variable}/{year}.nc"


def _hourly_precipitation_output_path(year: int, chunk_id: str) -> str:
    return f"raw/era5_land/hourly_precipitation/{year}_{chunk_id}.nc"


def build_daily_statistics_requests(
    *, region_manifest: dict[str, Any], start_year: int, end_year: int
) -> list[dict[str, Any]]:
    """One request per (project variable, year) for ERA5-Land daily statistics."""
    _check_region_manifest(region_manifest)
    validate_year_range(start_year, end_year)
    area = list(region_manifest["bbox_north_west_south_east"])
    requests: list[dict[str, Any]] = []
    for variable in DAILY_STATISTIC_VARIABLES:
        for year in range(start_year, end_year + 1):
            payload = {
                "variable": [variable.cds_variable],
                "daily_statistic": variable.daily_statistic,
                "year": str(year),
                "month": list(MONTHS),
                "day": list(DAYS),
                "time_zone": DAILY_STATISTICS_TIME_ZONE,
                "frequency": DAILY_STATISTICS_FREQUENCY,
                "area": area,
            }
            requests.append(
                {
                    "request_id": _daily_statistics_request_id(
                        variable.project_variable, year
                    ),
                    "request_kind": "daily_statistics",
                    "dataset": DAILY_STATISTICS_DATASET,
                    "project_variables": [variable.project_variable],
                    "cds_variables": [variable.cds_variable],
                    "year": year,
                    "output_path": _daily_statistics_output_path(
                        variable.project_variable, year
                    ),
                    "payload": payload,
                }
            )
    return requests


def build_hourly_precipitation_requests(
    *, region_manifest: dict[str, Any], start_year: int, end_year: int
) -> list[dict[str, Any]]:
    """Two semester requests per year for ERA5-Land hourly total precipitation."""
    _check_region_manifest(region_manifest)
    validate_year_range(start_year, end_year)
    area = list(region_manifest["bbox_north_west_south_east"])
    requests: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        for semester in PRECIPITATION_SEMESTERS:
            chunk_id: str = semester["chunk_id"]
            months: tuple[str, ...] = semester["months"]
            payload = {
                "variable": [HOURLY_PRECIPITATION_CDS_VARIABLE],
                "year": str(year),
                "month": list(months),
                "day": list(DAYS),
                "time": list(HOURS),
                "data_format": HOURLY_PRECIPITATION_DATA_FORMAT,
                "download_format": HOURLY_PRECIPITATION_DOWNLOAD_FORMAT,
                "area": area,
            }
            requests.append(
                {
                    "request_id": _hourly_precipitation_request_id(year, chunk_id),
                    "request_kind": "hourly_precipitation",
                    "dataset": HOURLY_PRECIPITATION_DATASET,
                    "project_variables": [HOURLY_PRECIPITATION_PROJECT_VARIABLE],
                    "cds_variables": [HOURLY_PRECIPITATION_CDS_VARIABLE],
                    "year": year,
                    "chunk_id": chunk_id,
                    "output_path": _hourly_precipitation_output_path(year, chunk_id),
                    "payload": payload,
                }
            )
    return requests


def build_download_manifest(
    *,
    region_manifest: dict[str, Any],
    region_manifest_path: Path,
    start_year: int,
    end_year: int,
    created_by: str = "scripts/01_plan_downloads.py",
) -> dict[str, Any]:
    """Assemble the deterministic ERA5-Land download plan manifest."""
    _check_region_manifest(region_manifest)
    validate_year_range(start_year, end_year)
    daily_requests = build_daily_statistics_requests(
        region_manifest=region_manifest,
        start_year=start_year,
        end_year=end_year,
    )
    precip_requests = build_hourly_precipitation_requests(
        region_manifest=region_manifest,
        start_year=start_year,
        end_year=end_year,
    )
    requests = daily_requests + precip_requests
    datasets = {
        DAILY_STATISTICS_DATASET: {
            "request_kind": "daily_statistics",
            "project_variables": [v.project_variable for v in DAILY_STATISTIC_VARIABLES],
            "cds_variables": sorted({v.cds_variable for v in DAILY_STATISTIC_VARIABLES}),
            "request_count": len(daily_requests),
            "frequency": DAILY_STATISTICS_FREQUENCY,
            "time_zone": DAILY_STATISTICS_TIME_ZONE,
            "chunking": "per_variable_per_year",
        },
        HOURLY_PRECIPITATION_DATASET: {
            "request_kind": "hourly_precipitation",
            "project_variables": [HOURLY_PRECIPITATION_PROJECT_VARIABLE],
            "cds_variables": [HOURLY_PRECIPITATION_CDS_VARIABLE],
            "request_count": len(precip_requests),
            "data_format": HOURLY_PRECIPITATION_DATA_FORMAT,
            "download_format": HOURLY_PRECIPITATION_DOWNLOAD_FORMAT,
            "chunking": "semester_per_year",
            "time_zone_policy": "open: precipitation timezone is not yet decided; see 90_legacy_review/migration_decision_log.md open decisions",
        },
    }
    manifest: dict[str, Any] = {
        "manifest_type": "era5_land_download_plan",
        "region_id": region_manifest["region_id"],
        "region_manifest_path": _as_posix(region_manifest_path),
        "region_geometry_hash": region_manifest["geometry_hash"],
        "bbox_north_west_south_east": list(region_manifest["bbox_north_west_south_east"]),
        "start_year": start_year,
        "end_year": end_year,
        "datasets": datasets,
        "requests": requests,
        "created_by": created_by,
        "requires_network": False,
        "download_execution_status": "planned_only",
    }
    return manifest


def write_download_manifest(output_path: Path, manifest: dict[str, Any]) -> None:
    """Write the download manifest as deterministic JSON with a trailing newline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")


def plan_downloads(
    *,
    region_manifest_path: Path,
    output_path: Path,
    start_year: int,
    end_year: int,
    created_by: str = "scripts/01_plan_downloads.py",
) -> dict[str, Any]:
    """End-to-end: load region manifest, build plan, return the manifest dict."""
    try:
        region_manifest = load_region_manifest(region_manifest_path)
    except RegionValidationError as exc:
        raise DownloadPlanError(str(exc)) from exc
    return build_download_manifest(
        region_manifest=region_manifest,
        region_manifest_path=region_manifest_path,
        start_year=start_year,
        end_year=end_year,
        created_by=created_by,
    )


def _as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


__all__ = [
    "DAILY_STATISTICS_DATASET",
    "DAILY_STATISTICS_FREQUENCY",
    "DAILY_STATISTICS_TIME_ZONE",
    "DAILY_STATISTIC_VARIABLES",
    "DailyStatisticVariable",
    "DownloadPlanError",
    "HOURLY_PRECIPITATION_CDS_VARIABLE",
    "HOURLY_PRECIPITATION_DATASET",
    "HOURLY_PRECIPITATION_DATA_FORMAT",
    "HOURLY_PRECIPITATION_DOWNLOAD_FORMAT",
    "HOURLY_PRECIPITATION_PROJECT_VARIABLE",
    "MAX_PLANNABLE_YEAR",
    "MIN_PLANNABLE_YEAR",
    "PRECIPITATION_SEMESTERS",
    "build_daily_statistics_requests",
    "build_download_manifest",
    "build_hourly_precipitation_requests",
    "plan_downloads",
    "validate_year_range",
    "write_download_manifest",
]
