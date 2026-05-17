"""ERA5-Land daily preprocessing helpers (milestones 004 and 006).

Consumes the M003 acquisition manifest plus the M002 download manifest
and the M001 region manifest. For each daily-statistics request the
helpers normalize coordinate/dimension names, rename the source CDS
variable to the project variable (``tmax``, ``tmin``, ``tmean``,
``u10m``, ``v10m``), convert temperature variables from Kelvin to
Celsius, mask grid cells outside the region polygon, and write a daily
standard product under ``intermediate/daily/{project_variable}/{year}.nc``.

Precipitation requests (``request_kind == "hourly_precipitation"``)
behave by policy:

- When ``precipitation_policy`` is ``None`` (M004 behavior, also the
  default when ``scripts/03_preprocess_daily.py`` is invoked without
  ``--precipitation-policy``), each precipitation chunk is recorded as
  ``status = deferred`` with ``reason = precipitation_policy_open``.
- When ``precipitation_policy`` is a supported value (M006 first
  shipped ``legacy_utc_minus_7``), H1 + H2 chunks for the same year
  are collapsed into one planning unit that produces a daily
  ``intermediate/daily/pr/{year}.nc`` via ``lib.precipitation``.

Precipitation indices, DST-aware policies, region-specific civil-time
policies, live CDS downloads, Docker, and cloud are explicitly out of
scope; see ``90_legacy_review/migration_decision_log.md``.

Heavy scientific dependencies (``numpy``, ``xarray``, ``scipy``,
``shapely``, ``h5netcdf``) are imported lazily inside the functions
that need them, so ``import lib.preprocessing`` works in a stripped
environment and is cheap when only manifest planning is needed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .acquisition import (
    MANIFEST_TYPE_ACQUISITION,
    MANIFEST_TYPE_DOWNLOAD_PLAN,
    resolve_target_path,
)
from .precipitation import (
    PR_PROJECT_VARIABLE,
    SUPPORTED_PRECIPITATION_POLICIES,
)
from .regions import (
    RegionValidationError,
    iter_features,
    load_geojson,
    load_region_manifest,
)

MANIFEST_TYPE_PREPROCESSING = "era5_land_preprocessing_run"

MODE_DRY_RUN = "dry-run"
MODE_EXECUTE = "execute"
VALID_MODES = frozenset({MODE_DRY_RUN, MODE_EXECUTE})

STATUS_PLANNED = "planned"
STATUS_PREPROCESSED = "preprocessed"
STATUS_SKIPPED = "skipped_existing"
STATUS_FAILED = "failed"
STATUS_MISSING_INPUT = "missing_input"
STATUS_DEFERRED = "deferred"

EXECUTION_STATUS_PLANNED = "planned_only"
EXECUTION_STATUS_PREPROCESSED = "preprocessed"
EXECUTION_STATUS_COMPLETE_EXISTING = "complete_existing"
EXECUTION_STATUS_PARTIAL = "partial"
EXECUTION_STATUS_FAILED = "failed"

REASON_PRECIPITATION_OPEN = "precipitation_policy_open"
REASON_SOURCE_NOT_FOUND = "source_not_found"
REASON_UNSUPPORTED_VARIABLE = "unsupported_project_variable"
REASON_SOURCE_NOT_DOWNLOADED = "source_acquisition_status_not_downloaded"
REASON_UNSUPPORTED_PRECIPITATION_POLICY = "unsupported_precipitation_policy"

KELVIN_TO_CELSIUS = 273.15

TEMPERATURE_PROJECT_VARIABLES = frozenset({"tmax", "tmin", "tmean"})
WIND_PROJECT_VARIABLES = frozenset({"u10m", "v10m"})
SUPPORTED_PROJECT_VARIABLES = TEMPERATURE_PROJECT_VARIABLES | WIND_PROJECT_VARIABLES

ACQUISITION_MANIFEST_REQUIRED_FIELDS = (
    "manifest_type",
    "region_id",
    "region_geometry_hash",
    "mode",
    "output_root",
    "results",
)

DOWNLOAD_MANIFEST_REQUIRED_FIELDS = (
    "manifest_type",
    "region_id",
    "region_geometry_hash",
    "requests",
)


class PreprocessingError(ValueError):
    """Raised when preprocessing inputs cannot be validated or executed."""


@dataclass(frozen=True)
class PreprocessingResult:
    request_id: str
    project_variable: str | None
    year: int | None
    source_path: str
    output_path: str
    status: str
    reason: str | None = None
    error: str | None = None
    source_chunks: tuple[dict[str, Any], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "request_id": self.request_id,
            "project_variable": self.project_variable,
            "year": self.year,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "status": self.status,
        }
        if self.reason is not None:
            record["reason"] = self.reason
        if self.error is not None:
            record["error"] = self.error
        if self.source_chunks is not None:
            record["source_chunks"] = [dict(c) for c in self.source_chunks]
        return record


# ---------------------------------------------------------------------------
# Manifest loading and joining
# ---------------------------------------------------------------------------


def compute_manifest_hash(path: Path) -> str:
    """Return a stable ``sha256:<hex>`` hash of the manifest file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_acquisition_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate an M003 acquisition manifest."""
    data = _read_json_object(path, label="acquisition manifest")
    _require_fields(data, ACQUISITION_MANIFEST_REQUIRED_FIELDS, path, label="acquisition manifest")
    if data["manifest_type"] != MANIFEST_TYPE_ACQUISITION:
        raise PreprocessingError(
            f"acquisition manifest manifest_type must be {MANIFEST_TYPE_ACQUISITION!r}, "
            f"got {data['manifest_type']!r}"
        )
    if not isinstance(data["results"], list):
        raise PreprocessingError("acquisition manifest 'results' must be a list")
    return data


def load_download_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate an M002 download manifest."""
    data = _read_json_object(path, label="download manifest")
    _require_fields(data, DOWNLOAD_MANIFEST_REQUIRED_FIELDS, path, label="download manifest")
    if data["manifest_type"] != MANIFEST_TYPE_DOWNLOAD_PLAN:
        raise PreprocessingError(
            f"download manifest manifest_type must be {MANIFEST_TYPE_DOWNLOAD_PLAN!r}, "
            f"got {data['manifest_type']!r}"
        )
    if not isinstance(data["requests"], list):
        raise PreprocessingError("download manifest 'requests' must be a list")
    return data


def load_region_manifest_for_preprocessing(path: Path) -> dict[str, Any]:
    """Reuse the M001 strict region-manifest validator with the M004 exception."""
    try:
        return load_region_manifest(path)
    except RegionValidationError as exc:
        raise PreprocessingError(f"region manifest invalid: {exc}") from exc


def load_region_geojson(region_manifest: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load the raw GeoJSON referenced by ``region_manifest['geometry_path']``."""
    raw_path = Path(region_manifest["geometry_path"])
    if not raw_path.is_absolute() and repo_root is not None:
        raw_path = repo_root / raw_path
    return load_geojson(raw_path)


def assert_provenance_consistency(
    *,
    acquisition: dict[str, Any],
    download: dict[str, Any],
    region: dict[str, Any],
) -> None:
    """Confirm region_id and geometry hash agree across the three manifests.

    A chained ``a != b != c`` here would short-circuit on the first equal
    pair and miss a single-manifest mismatch (e.g. acquisition == download
    but region disagrees). Compare via a set so any difference raises.
    """
    region_ids = {
        acquisition["region_id"],
        download["region_id"],
        region["region_id"],
    }
    if len(region_ids) > 1:
        raise PreprocessingError(
            "region_id disagrees between manifests: "
            f"acquisition={acquisition['region_id']!r} "
            f"download={download['region_id']!r} "
            f"region={region['region_id']!r}"
        )
    if acquisition["region_geometry_hash"] != download["region_geometry_hash"]:
        raise PreprocessingError(
            "region_geometry_hash disagrees between acquisition and download manifests"
        )
    if download["region_geometry_hash"] != region["geometry_hash"]:
        raise PreprocessingError(
            "region_geometry_hash disagrees between download and region manifests"
        )


def join_results_to_requests(
    *,
    acquisition: dict[str, Any],
    download: dict[str, Any],
) -> list[dict[str, Any]]:
    """For each acquisition result, attach the matching download request."""
    requests_by_id = {r["request_id"]: r for r in download["requests"]}
    joined: list[dict[str, Any]] = []
    for result in acquisition["results"]:
        request = requests_by_id.get(result["request_id"])
        if request is None:
            raise PreprocessingError(
                f"acquisition result {result['request_id']!r} has no matching "
                "request in the download manifest"
            )
        joined.append({"result": result, "request": request})
    return joined


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------


def daily_output_path(project_variable: str, year: int) -> str:
    """Relative POSIX path for a daily standard product."""
    return f"intermediate/daily/{project_variable}/{year}.nc"


# ---------------------------------------------------------------------------
# In-memory transformations (xarray-aware, imported lazily)
# ---------------------------------------------------------------------------


def normalize_dimensions(dataset):
    """Rename ``valid_time`` -> ``time``, ``latitude``/``longitude`` -> ``lat``/``lon``.

    Idempotent: already-normalized datasets pass through unchanged.
    """
    rename = {}
    for source, target in (
        ("valid_time", "time"),
        ("latitude", "lat"),
        ("longitude", "lon"),
    ):
        if source in dataset.coords or source in dataset.dims:
            if target in dataset.coords or target in dataset.dims:
                raise PreprocessingError(
                    f"cannot rename {source!r} to {target!r}: both already exist"
                )
            rename[source] = target
    if rename:
        dataset = dataset.rename(rename)
    return dataset


def rename_to_project_variable(dataset, *, project_variable: str, source_variable: str | None = None):
    """Rename the source data variable to ``project_variable``.

    If ``source_variable`` is ``None``, infer the single non-coordinate data
    variable from the dataset. Errors are explicit when the source
    variable is missing or when more than one candidate is present.
    """
    if project_variable not in SUPPORTED_PROJECT_VARIABLES:
        raise PreprocessingError(
            f"unsupported project_variable {project_variable!r}; "
            f"expected one of {sorted(SUPPORTED_PROJECT_VARIABLES)}"
        )
    if source_variable is None:
        data_vars = list(dataset.data_vars)
        if len(data_vars) != 1:
            raise PreprocessingError(
                f"cannot infer source variable: dataset has {len(data_vars)} "
                f"data variables ({data_vars}); pass source_variable explicitly"
            )
        source_variable = data_vars[0]
    if source_variable not in dataset.data_vars:
        raise PreprocessingError(
            f"source variable {source_variable!r} not found in dataset; "
            f"available data variables: {list(dataset.data_vars)}"
        )
    if source_variable == project_variable:
        return dataset
    if project_variable in dataset.data_vars:
        raise PreprocessingError(
            f"cannot rename {source_variable!r} to {project_variable!r}: "
            f"{project_variable!r} already exists"
        )
    return dataset.rename({source_variable: project_variable})


def convert_temperature_if_needed(dataset, *, project_variable: str):
    """Convert Kelvin to Celsius for temperature variables; no-op otherwise."""
    if project_variable not in TEMPERATURE_PROJECT_VARIABLES:
        return dataset
    if project_variable not in dataset.data_vars:
        raise PreprocessingError(
            f"temperature conversion requested but {project_variable!r} is "
            f"not in dataset; have {list(dataset.data_vars)}"
        )
    da = dataset[project_variable] - KELVIN_TO_CELSIUS
    da.attrs = dict(dataset[project_variable].attrs)
    da.attrs["units"] = "degC"
    da.attrs["preprocessing_offset_kelvin"] = KELVIN_TO_CELSIUS
    out = dataset.copy()
    out[project_variable] = da
    return out


def build_polygon_mask(dataset, *, region_geojson: dict[str, Any]):
    """Return a 2D ``(lat, lon)`` boolean mask: True inside the region union."""
    import numpy as np
    import xarray as xr
    from shapely import contains_xy
    from shapely.geometry import shape
    from shapely.ops import unary_union

    for coord in ("lat", "lon"):
        if coord not in dataset.coords:
            raise PreprocessingError(
                f"dataset must carry a '{coord}' coordinate before masking; "
                "call normalize_dimensions() first"
            )
    geometries = []
    for feature in iter_features(region_geojson):
        geom_dict = feature.get("geometry")
        if not isinstance(geom_dict, dict):
            continue
        geometries.append(shape(geom_dict))
    if not geometries:
        raise PreprocessingError("region geojson has no usable geometries")
    polygon = unary_union(geometries)
    lon2d, lat2d = np.meshgrid(dataset.lon.values, dataset.lat.values)
    mask = contains_xy(polygon, lon2d, lat2d)
    return xr.DataArray(mask, dims=("lat", "lon"), coords={"lat": dataset.lat, "lon": dataset.lon})


def apply_region_mask(dataset, *, region_geojson: dict[str, Any]):
    """Set values outside the region polygon to NaN; preserves dataset shape."""
    mask = build_polygon_mask(dataset, region_geojson=region_geojson)
    masked = dataset.where(mask)
    return masked


def preprocess_dataset(
    dataset,
    *,
    project_variable: str,
    region_geojson: dict[str, Any] | None,
    source_variable: str | None = None,
):
    """Full transformation: normalize coords/dims, rename, K->C, mask."""
    dataset = normalize_dimensions(dataset)
    dataset = rename_to_project_variable(
        dataset, project_variable=project_variable, source_variable=source_variable
    )
    dataset = convert_temperature_if_needed(dataset, project_variable=project_variable)
    if region_geojson is not None:
        dataset = apply_region_mask(dataset, region_geojson=region_geojson)
    return dataset


# ---------------------------------------------------------------------------
# Planning and execution
# ---------------------------------------------------------------------------


def _select_project_variable(request: dict[str, Any]) -> str | None:
    project_vars = request.get("project_variables") or []
    if len(project_vars) == 1:
        return project_vars[0]
    return None


def _iter_planning_units(
    joined: list[dict[str, Any]],
    *,
    precipitation_policy: str | None,
):
    """Yield planning units in input order.

    When ``precipitation_policy`` is set, all ``hourly_precipitation``
    requests for the same year are collapsed into one
    ``precipitation_group`` unit emitted at the position of the year's
    first chunk in the input list. Subsequent chunks of the same year
    are absorbed into that group. When ``precipitation_policy`` is
    None (M004 behavior), precipitation requests are passed through as
    individual ``single`` units which the daily-stats path will defer.
    """
    pr_emitted: set[Any] = set()
    for entry in joined:
        request = entry["request"]
        request_kind = request.get("request_kind")
        if request_kind == "hourly_precipitation" and precipitation_policy is not None:
            if precipitation_policy not in SUPPORTED_PRECIPITATION_POLICIES:
                yield {
                    "kind": "precipitation_unsupported_policy",
                    "entry": entry,
                    "policy": precipitation_policy,
                }
                continue
            year = request.get("year")
            if year in pr_emitted:
                continue
            pr_emitted.add(year)
            chunks = sorted(
                [
                    e
                    for e in joined
                    if e["request"].get("request_kind") == "hourly_precipitation"
                    and e["request"].get("year") == year
                ],
                key=lambda e: e["request"].get("chunk_id", ""),
            )
            yield {"kind": "precipitation_group", "year": year, "chunks": chunks}
            continue
        yield {"kind": "single", "entry": entry}


def _classify_request(
    joined: dict[str, Any],
) -> tuple[str | None, int | None, str | None]:
    """Return ``(project_variable, year, deferral_reason)``.

    A non-``None`` ``deferral_reason`` means the request should never be
    preprocessed by this milestone (precipitation, unsupported variable,
    etc.) and the result will carry ``status = deferred``.
    """
    request = joined["request"]
    request_kind = request.get("request_kind")
    project_variable = _select_project_variable(request)
    year = request.get("year")
    if request_kind == "hourly_precipitation":
        return None, year, REASON_PRECIPITATION_OPEN
    if request_kind != "daily_statistics":
        return project_variable, year, f"unsupported_request_kind:{request_kind}"
    if project_variable is None or project_variable not in SUPPORTED_PROJECT_VARIABLES:
        return project_variable, year, REASON_UNSUPPORTED_VARIABLE
    if not isinstance(year, int):
        return project_variable, None, "missing_year"
    return project_variable, year, None


def _build_source_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": e["request"].get("chunk_id"),
            "request_id": e["result"]["request_id"],
            "source_path": e["result"].get("target_path", ""),
        }
        for e in chunks
    ]


def plan_results(
    joined: list[dict[str, Any]],
    *,
    output_root: Path,
    precipitation_policy: str | None = None,
) -> list[PreprocessingResult]:
    """Build dry-run preprocessing records without any I/O."""
    results: list[PreprocessingResult] = []
    for unit in _iter_planning_units(joined, precipitation_policy=precipitation_policy):
        if unit["kind"] == "precipitation_unsupported_policy":
            entry = unit["entry"]
            results.append(
                PreprocessingResult(
                    request_id=entry["result"]["request_id"],
                    project_variable=None,
                    year=entry["request"].get("year"),
                    source_path=entry["result"].get("target_path", ""),
                    output_path="",
                    status=STATUS_DEFERRED,
                    reason=f"{REASON_UNSUPPORTED_PRECIPITATION_POLICY}:{unit['policy']}",
                )
            )
            continue
        if unit["kind"] == "precipitation_group":
            year = unit["year"]
            chunks = unit["chunks"]
            target = resolve_target_path(
                output_root, daily_output_path(PR_PROJECT_VARIABLE, year)
            )
            source_chunks = _build_source_chunks(chunks)
            primary_path = source_chunks[0]["source_path"] if source_chunks else ""
            missing = [
                c
                for c in source_chunks
                if not c["source_path"] or not Path(c["source_path"]).exists()
            ]
            reason = REASON_SOURCE_NOT_FOUND if missing else None
            results.append(
                PreprocessingResult(
                    request_id=f"era5_hourly_pr__{year}",
                    project_variable=PR_PROJECT_VARIABLE,
                    year=year,
                    source_path=primary_path,
                    output_path=_as_posix(target),
                    status=STATUS_PLANNED,
                    reason=reason,
                    source_chunks=tuple(source_chunks),
                )
            )
            continue
        # kind == "single": daily-stats or precipitation-with-no-policy (deferred)
        entry = unit["entry"]
        result_record = entry["result"]
        project_variable, year, reason = _classify_request(entry)
        if reason is not None:
            results.append(
                PreprocessingResult(
                    request_id=result_record["request_id"],
                    project_variable=project_variable,
                    year=year,
                    source_path=result_record.get("target_path", ""),
                    output_path="",
                    status=STATUS_DEFERRED,
                    reason=reason,
                )
            )
            continue
        assert project_variable is not None and year is not None
        output_path = daily_output_path(project_variable, year)
        target = resolve_target_path(output_root, output_path)
        source_path = result_record.get("target_path", "")
        results.append(
            PreprocessingResult(
                request_id=result_record["request_id"],
                project_variable=project_variable,
                year=year,
                source_path=source_path,
                output_path=_as_posix(target),
                status=STATUS_PLANNED,
                reason=REASON_SOURCE_NOT_FOUND if source_path and not Path(source_path).exists() else None,
            )
        )
    return results


def execute_results(
    joined: list[dict[str, Any]],
    *,
    output_root: Path,
    region_geojson: dict[str, Any],
    overwrite: bool = False,
    precipitation_policy: str | None = None,
) -> list[PreprocessingResult]:
    """Open NetCDF inputs, transform, and write daily standard products.

    Imports xarray lazily; raises ``PreprocessingError`` if the engine
    cannot open the source file. Per-request exceptions are converted
    to ``STATUS_FAILED`` records so the manifest captures everything.
    """
    import xarray as xr

    results: list[PreprocessingResult] = []
    for unit in _iter_planning_units(joined, precipitation_policy=precipitation_policy):
        if unit["kind"] == "precipitation_unsupported_policy":
            entry = unit["entry"]
            results.append(
                PreprocessingResult(
                    request_id=entry["result"]["request_id"],
                    project_variable=None,
                    year=entry["request"].get("year"),
                    source_path=entry["result"].get("target_path", ""),
                    output_path="",
                    status=STATUS_DEFERRED,
                    reason=f"{REASON_UNSUPPORTED_PRECIPITATION_POLICY}:{unit['policy']}",
                )
            )
            continue
        if unit["kind"] == "precipitation_group":
            results.append(
                _execute_precipitation_group(
                    unit,
                    output_root=output_root,
                    region_geojson=region_geojson,
                    overwrite=overwrite,
                    precipitation_policy=precipitation_policy,
                )
            )
            continue
        # kind == "single"
        entry = unit["entry"]
        result_record = entry["result"]
        acquisition_status = result_record.get("status")
        project_variable, year, reason = _classify_request(entry)
        source_path = result_record.get("target_path", "")
        if reason is not None:
            results.append(
                PreprocessingResult(
                    request_id=result_record["request_id"],
                    project_variable=project_variable,
                    year=year,
                    source_path=source_path,
                    output_path="",
                    status=STATUS_DEFERRED,
                    reason=reason,
                )
            )
            continue
        assert project_variable is not None and year is not None
        target = resolve_target_path(output_root, daily_output_path(project_variable, year))
        output_posix = _as_posix(target)
        if not source_path or not Path(source_path).exists():
            results.append(
                PreprocessingResult(
                    request_id=result_record["request_id"],
                    project_variable=project_variable,
                    year=year,
                    source_path=source_path,
                    output_path=output_posix,
                    status=STATUS_MISSING_INPUT,
                    reason=REASON_SOURCE_NOT_FOUND
                    if acquisition_status in (None, "planned")
                    else REASON_SOURCE_NOT_DOWNLOADED,
                )
            )
            continue
        if target.exists() and not overwrite:
            results.append(
                PreprocessingResult(
                    request_id=result_record["request_id"],
                    project_variable=project_variable,
                    year=year,
                    source_path=source_path,
                    output_path=output_posix,
                    status=STATUS_SKIPPED,
                )
            )
            continue
        try:
            with xr.open_dataset(source_path) as ds:
                processed = preprocess_dataset(
                    ds.load(),
                    project_variable=project_variable,
                    region_geojson=region_geojson,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            processed.to_netcdf(target)
        except Exception as exc:  # noqa: BLE001 - adapter boundary captures failures into the manifest
            results.append(
                PreprocessingResult(
                    request_id=result_record["request_id"],
                    project_variable=project_variable,
                    year=year,
                    source_path=source_path,
                    output_path=output_posix,
                    status=STATUS_FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(
            PreprocessingResult(
                request_id=result_record["request_id"],
                project_variable=project_variable,
                year=year,
                source_path=source_path,
                output_path=output_posix,
                status=STATUS_PREPROCESSED,
            )
        )
    return results


def _execute_precipitation_group(
    unit: dict[str, Any],
    *,
    output_root: Path,
    region_geojson: dict[str, Any],
    overwrite: bool,
    precipitation_policy: str | None,
) -> PreprocessingResult:
    import xarray as xr

    from .precipitation import preprocess_precipitation_dataset

    assert precipitation_policy is not None  # caller guarantees
    year = unit["year"]
    chunks = unit["chunks"]
    source_chunks = _build_source_chunks(chunks)
    target = resolve_target_path(
        output_root, daily_output_path(PR_PROJECT_VARIABLE, year)
    )
    output_posix = _as_posix(target)
    primary_path = source_chunks[0]["source_path"] if source_chunks else ""
    request_id = f"era5_hourly_pr__{year}"
    missing = [
        c
        for c in source_chunks
        if not c["source_path"] or not Path(c["source_path"]).exists()
    ]
    if missing:
        return PreprocessingResult(
            request_id=request_id,
            project_variable=PR_PROJECT_VARIABLE,
            year=year,
            source_path=primary_path,
            output_path=output_posix,
            status=STATUS_MISSING_INPUT,
            reason=REASON_SOURCE_NOT_FOUND,
            source_chunks=tuple(source_chunks),
        )
    if target.exists() and not overwrite:
        return PreprocessingResult(
            request_id=request_id,
            project_variable=PR_PROJECT_VARIABLE,
            year=year,
            source_path=primary_path,
            output_path=output_posix,
            status=STATUS_SKIPPED,
            source_chunks=tuple(source_chunks),
        )
    try:
        per_chunk_datasets: list[Any] = []
        for chunk in source_chunks:
            with xr.open_dataset(chunk["source_path"]) as ds:
                per_chunk_datasets.append(ds.load())
        if len(per_chunk_datasets) == 1:
            combined = per_chunk_datasets[0]
        else:
            # CDS chunks carry ``valid_time``; concat along whichever time
            # dimension the chunks already have. ``preprocess_precipitation_dataset``
            # will rename to ``time`` afterwards via ``normalize_dimensions``.
            first_dims = per_chunk_datasets[0].dims
            time_dim = "valid_time" if "valid_time" in first_dims else "time"
            combined = xr.concat(per_chunk_datasets, dim=time_dim, join="exact")
        processed = preprocess_precipitation_dataset(
            combined,
            policy=precipitation_policy,
            region_geojson=region_geojson,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        processed.to_netcdf(target)
    except Exception as exc:  # noqa: BLE001 - adapter boundary captures failures into the manifest
        return PreprocessingResult(
            request_id=request_id,
            project_variable=PR_PROJECT_VARIABLE,
            year=year,
            source_path=primary_path,
            output_path=output_posix,
            status=STATUS_FAILED,
            error=f"{type(exc).__name__}: {exc}",
            source_chunks=tuple(source_chunks),
        )
    return PreprocessingResult(
        request_id=request_id,
        project_variable=PR_PROJECT_VARIABLE,
        year=year,
        source_path=primary_path,
        output_path=output_posix,
        status=STATUS_PREPROCESSED,
        source_chunks=tuple(source_chunks),
    )


def derive_execution_status(mode: str, results: list[PreprocessingResult]) -> str:
    """Summarize per-result statuses into a manifest-level execution_status.

    ``deferred`` counts (precipitation without a policy, unsupported
    variable, unsupported precipitation policy) are informational and
    do not by themselves flip the run into a failure state. The status
    reflects what happened to the actionable requests -- daily
    statistics under M004 and, when ``--precipitation-policy`` is set,
    daily precipitation under M006.
    """
    if mode == MODE_DRY_RUN:
        return EXECUTION_STATUS_PLANNED
    actionable = [r for r in results if r.status != STATUS_DEFERRED]
    if not actionable:
        return EXECUTION_STATUS_PLANNED
    statuses = {r.status for r in actionable}
    has_failure = STATUS_FAILED in statuses or STATUS_MISSING_INPUT in statuses
    has_success = STATUS_PREPROCESSED in statuses
    has_skipped = STATUS_SKIPPED in statuses
    if statuses == {STATUS_FAILED} or statuses == {STATUS_MISSING_INPUT} or statuses == {STATUS_FAILED, STATUS_MISSING_INPUT}:
        return EXECUTION_STATUS_FAILED
    if has_failure:
        return EXECUTION_STATUS_PARTIAL
    if statuses == {STATUS_SKIPPED}:
        return EXECUTION_STATUS_COMPLETE_EXISTING
    if has_success:
        return EXECUTION_STATUS_PREPROCESSED
    return EXECUTION_STATUS_PARTIAL


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------


def build_preprocessing_manifest(
    *,
    acquisition_manifest: dict[str, Any],
    acquisition_manifest_path: Path,
    acquisition_manifest_hash: str,
    download_manifest_path: Path,
    download_manifest_hash: str,
    region_manifest: dict[str, Any],
    region_manifest_path: Path,
    region_manifest_hash: str,
    mode: str,
    output_root: Path,
    results: list[PreprocessingResult],
    created_by: str,
    precipitation_policy: str | None = None,
) -> dict[str, Any]:
    """Assemble the deterministic preprocessing manifest dict.

    ``precipitation_policy`` is added to the manifest only when it is
    set, preserving the M004 reference schema (which carries no
    precipitation policy) byte-identically.
    """
    if mode not in VALID_MODES:
        raise PreprocessingError(f"mode {mode!r} is not one of {sorted(VALID_MODES)}")
    counts = {
        STATUS_PLANNED: 0,
        STATUS_PREPROCESSED: 0,
        STATUS_SKIPPED: 0,
        STATUS_FAILED: 0,
        STATUS_MISSING_INPUT: 0,
        STATUS_DEFERRED: 0,
    }
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    manifest: dict[str, Any] = {
        "manifest_type": MANIFEST_TYPE_PREPROCESSING,
        "acquisition_manifest_path": _as_posix(acquisition_manifest_path),
        "acquisition_manifest_hash": acquisition_manifest_hash,
        "download_manifest_path": _as_posix(download_manifest_path),
        "download_manifest_hash": download_manifest_hash,
        "region_manifest_path": _as_posix(region_manifest_path),
        "region_manifest_hash": region_manifest_hash,
        "region_id": region_manifest["region_id"],
        "region_geometry_hash": region_manifest["geometry_hash"],
        "mode": mode,
        "output_root": _as_posix(output_root),
        "request_count": len(results),
        "planned_count": counts[STATUS_PLANNED],
        "preprocessed_count": counts[STATUS_PREPROCESSED],
        "skipped_count": counts[STATUS_SKIPPED],
        "failed_count": counts[STATUS_FAILED],
        "missing_input_count": counts[STATUS_MISSING_INPUT],
        "deferred_count": counts[STATUS_DEFERRED],
        "results": [r.to_dict() for r in results],
        "created_by": created_by,
        "requires_network": False,
        "execution_status": derive_execution_status(mode, results),
    }
    if precipitation_policy is not None:
        manifest["precipitation_policy"] = precipitation_policy
    return manifest


def write_preprocessing_manifest(output_path: Path, manifest: dict[str, Any]) -> None:
    """Write the preprocessing manifest as deterministic JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Request selection
# ---------------------------------------------------------------------------


def select_joined_records(
    joined: list[dict[str, Any]],
    *,
    request_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter joined records by an optional id allow-list and/or count limit."""
    selected = list(joined)
    if request_ids is not None:
        wanted = list(request_ids)
        if not wanted:
            raise PreprocessingError("--request-id filter provided but list is empty")
        wanted_set = set(wanted)
        available = {entry["result"]["request_id"] for entry in selected}
        unknown = sorted(wanted_set - available)
        if unknown:
            raise PreprocessingError(f"unknown request_id(s): {unknown}")
        selected = [
            entry for entry in selected if entry["result"]["request_id"] in wanted_set
        ]
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise PreprocessingError(f"limit must be a positive int, got {type(limit).__name__}")
        if limit <= 0:
            raise PreprocessingError(f"limit must be a positive int, got {limit}")
        selected = selected[:limit]
    if not selected:
        raise PreprocessingError("no requests remain after filtering")
    return selected


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise PreprocessingError(f"{label} not found: {path}")
    if not path.is_file():
        raise PreprocessingError(f"{label} path is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreprocessingError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PreprocessingError(
            f"{label} must be a JSON object, got {type(data).__name__}: {path}"
        )
    return data


def _require_fields(
    data: dict[str, Any],
    fields: tuple[str, ...],
    path: Path,
    *,
    label: str,
) -> None:
    missing = [f for f in fields if f not in data]
    if missing:
        raise PreprocessingError(
            f"{label} {path} is missing required fields: {missing}"
        )


def _as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


__all__ = [
    "ACQUISITION_MANIFEST_REQUIRED_FIELDS",
    "DOWNLOAD_MANIFEST_REQUIRED_FIELDS",
    "EXECUTION_STATUS_COMPLETE_EXISTING",
    "EXECUTION_STATUS_FAILED",
    "EXECUTION_STATUS_PARTIAL",
    "EXECUTION_STATUS_PLANNED",
    "EXECUTION_STATUS_PREPROCESSED",
    "KELVIN_TO_CELSIUS",
    "MANIFEST_TYPE_PREPROCESSING",
    "MODE_DRY_RUN",
    "MODE_EXECUTE",
    "PreprocessingError",
    "PreprocessingResult",
    "PR_PROJECT_VARIABLE",
    "REASON_PRECIPITATION_OPEN",
    "REASON_SOURCE_NOT_DOWNLOADED",
    "REASON_SOURCE_NOT_FOUND",
    "REASON_UNSUPPORTED_PRECIPITATION_POLICY",
    "REASON_UNSUPPORTED_VARIABLE",
    "SUPPORTED_PRECIPITATION_POLICIES",
    "STATUS_DEFERRED",
    "STATUS_FAILED",
    "STATUS_MISSING_INPUT",
    "STATUS_PLANNED",
    "STATUS_PREPROCESSED",
    "STATUS_SKIPPED",
    "SUPPORTED_PROJECT_VARIABLES",
    "TEMPERATURE_PROJECT_VARIABLES",
    "VALID_MODES",
    "WIND_PROJECT_VARIABLES",
    "apply_region_mask",
    "assert_provenance_consistency",
    "build_polygon_mask",
    "build_preprocessing_manifest",
    "compute_manifest_hash",
    "convert_temperature_if_needed",
    "daily_output_path",
    "derive_execution_status",
    "execute_results",
    "join_results_to_requests",
    "load_acquisition_manifest",
    "load_download_manifest",
    "load_region_geojson",
    "load_region_manifest_for_preprocessing",
    "normalize_dimensions",
    "plan_results",
    "preprocess_dataset",
    "rename_to_project_variable",
    "select_joined_records",
    "write_preprocessing_manifest",
]
