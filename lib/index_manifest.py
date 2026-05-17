"""Shared index-manifest helpers for milestones 005+.

Loads a preprocessing manifest (M004 daily-statistics or M006 daily
precipitation), plans per-index source paths, runs index computations
in execute mode, and writes a deterministic ``index_manifest.json``.

Two index families are wired in:

- **Temperature** (M005): seven simple annual indices over
  ``tmax`` / ``tmin`` / ``tmean``.
- **Precipitation** (M007): seven ETCCDI-style annual indices over
  daily ``pr``.

The script ``scripts/04_compute_indices.py`` selects a family via
``--index-family`` (default ``temperature`` to keep the M005 reference
command byte-identical). The wire-level manifest schema is identical
across families.

Heavy dependencies (``xarray``, ``numpy``) are imported lazily inside
the helpers that need them so ``import lib.index_manifest`` works
without them and dry-run planning stays cheap.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .acquisition import resolve_target_path
from .indices_precipitation import (
    PRECIPITATION_INDEX_FUNCTIONS,
    PRECIPITATION_INDEX_SPECS,
    PRECIPITATION_INDEX_SPECS_BY_ID,
    PrecipitationIndexError,
)
from .indices_temperature import (
    TEMPERATURE_INDEX_FUNCTIONS,
    TEMPERATURE_INDEX_SPECS,
    TEMPERATURE_INDEX_SPECS_BY_ID,
    TemperatureIndexError,
)

MANIFEST_TYPE_INDEX = "era5_land_index_run"
MANIFEST_TYPE_PREPROCESSING = "era5_land_preprocessing_run"

MODE_DRY_RUN = "dry-run"
MODE_EXECUTE = "execute"
VALID_MODES = frozenset({MODE_DRY_RUN, MODE_EXECUTE})

STATUS_PLANNED = "planned"
STATUS_COMPUTED = "computed"
STATUS_SKIPPED = "skipped_existing"
STATUS_FAILED = "failed"
STATUS_MISSING_INPUT = "missing_input"

EXECUTION_STATUS_PLANNED = "planned_only"
EXECUTION_STATUS_COMPUTED = "computed"
EXECUTION_STATUS_COMPLETE_EXISTING = "complete_existing"
EXECUTION_STATUS_PARTIAL = "partial"
EXECUTION_STATUS_FAILED = "failed"

REASON_REQUIRED_VARIABLE_NOT_IN_PLAN = "required_variable_not_in_preprocessing_plan"
REASON_SOURCE_NOT_FOUND = "source_not_found"
REASON_SOURCE_NOT_PREPROCESSED = "source_preprocessing_status_not_preprocessed"

INDEX_FAMILY_TEMPERATURE = "temperature"
INDEX_FAMILY_PRECIPITATION = "precipitation"
INDEX_FAMILY_ALL = "all"
SUPPORTED_INDEX_FAMILIES = frozenset(
    {INDEX_FAMILY_TEMPERATURE, INDEX_FAMILY_PRECIPITATION, INDEX_FAMILY_ALL}
)

PREPROCESSING_MANIFEST_REQUIRED_FIELDS = (
    "manifest_type",
    "region_id",
    "region_geometry_hash",
    "results",
)


class IndexManifestError(ValueError):
    """Raised when an index manifest cannot be planned or executed."""


@dataclass(frozen=True)
class IndexResult:
    index_id: str
    required_variables: tuple[str, ...]
    source_paths: tuple[str, ...]
    output_path: str
    status: str
    reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "index_id": self.index_id,
            "required_variables": list(self.required_variables),
            "source_paths": list(self.source_paths),
            "output_path": self.output_path,
            "status": self.status,
        }
        if self.reason is not None:
            record["reason"] = self.reason
        if self.error is not None:
            record["error"] = self.error
        return record


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def compute_manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_preprocessing_manifest(path: Path) -> dict[str, Any]:
    """Read and minimally validate an M004 preprocessing manifest."""
    if not path.exists():
        raise IndexManifestError(f"preprocessing manifest not found: {path}")
    if not path.is_file():
        raise IndexManifestError(f"preprocessing manifest path is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexManifestError(
            f"preprocessing manifest is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise IndexManifestError(
            f"preprocessing manifest must be a JSON object: {path}"
        )
    missing = [f for f in PREPROCESSING_MANIFEST_REQUIRED_FIELDS if f not in data]
    if missing:
        raise IndexManifestError(
            f"preprocessing manifest {path} is missing required fields: {missing}"
        )
    if data["manifest_type"] != MANIFEST_TYPE_PREPROCESSING:
        raise IndexManifestError(
            f"preprocessing manifest manifest_type must be "
            f"{MANIFEST_TYPE_PREPROCESSING!r}, got {data['manifest_type']!r}"
        )
    if not isinstance(data["results"], list):
        raise IndexManifestError("preprocessing manifest 'results' must be a list")
    return data


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------


def group_preprocessing_results_by_variable(
    preprocessing: dict[str, Any],
    *,
    require_executed: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Group preprocessing results by ``project_variable``, sorted by year.

    If ``require_executed`` is True, only ``preprocessed`` /
    ``skipped_existing`` results are kept (use in execute mode). Otherwise
    every non-deferred / non-failed result with a project_variable counts
    (use in dry-run planning).
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in preprocessing["results"]:
        project_variable = result.get("project_variable")
        if not project_variable:
            continue
        status = result.get("status")
        if require_executed and status not in ("preprocessed", "skipped_existing"):
            continue
        grouped[project_variable].append(result)
    for variable, items in grouped.items():
        items.sort(key=lambda r: (r.get("year") or 0, r.get("request_id", "")))
    return dict(grouped)


def specs_for_family(family: str) -> list[Any]:
    """Return the index-spec list for ``family``.

    ``temperature`` -> the M005 seven temperature specs.
    ``precipitation`` -> the M007 seven precipitation specs.
    ``all`` -> temperature first, then precipitation, preserving the
    M005 default ordering when only temperature was selected.
    """
    if family == INDEX_FAMILY_TEMPERATURE:
        return list(TEMPERATURE_INDEX_SPECS)
    if family == INDEX_FAMILY_PRECIPITATION:
        return list(PRECIPITATION_INDEX_SPECS)
    if family == INDEX_FAMILY_ALL:
        return list(TEMPERATURE_INDEX_SPECS) + list(PRECIPITATION_INDEX_SPECS)
    raise IndexManifestError(
        f"unknown index family {family!r}; expected one of "
        f"{sorted(SUPPORTED_INDEX_FAMILIES)}"
    )


def _dispatch_compute(index_id: str, daily: dict[str, Any]):
    """Run the index computation by id, dispatching across families."""
    if index_id in TEMPERATURE_INDEX_FUNCTIONS:
        return TEMPERATURE_INDEX_FUNCTIONS[index_id](daily)
    if index_id in PRECIPITATION_INDEX_FUNCTIONS:
        return PRECIPITATION_INDEX_FUNCTIONS[index_id](daily)
    raise IndexManifestError(
        f"unknown index_id {index_id!r}; expected one of "
        f"{sorted(set(TEMPERATURE_INDEX_FUNCTIONS) | set(PRECIPITATION_INDEX_FUNCTIONS))}"
    )


def index_output_path(index_id: str) -> str:
    """Relative POSIX path for an annual index NetCDF product."""
    return f"derived/indices/{index_id}.nc"


def select_index_specs(
    specs: Iterable[Any],
    *,
    index_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[Any]:
    """Filter the configured index specs by an optional id allow-list and limit."""
    selected = list(specs)
    if index_ids is not None:
        wanted = list(index_ids)
        if not wanted:
            raise IndexManifestError("--index-id filter provided but list is empty")
        available = {spec.index_id for spec in selected}
        unknown = sorted(set(wanted) - available)
        if unknown:
            raise IndexManifestError(f"unknown index_id(s): {unknown}")
        wanted_set = set(wanted)
        selected = [spec for spec in selected if spec.index_id in wanted_set]
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise IndexManifestError(
                f"limit must be a positive int, got {type(limit).__name__}"
            )
        if limit <= 0:
            raise IndexManifestError(f"limit must be a positive int, got {limit}")
        selected = selected[:limit]
    if not selected:
        raise IndexManifestError("no indices remain after filtering")
    return selected


def plan_index_results(
    specs: list[Any],
    *,
    preprocessing: dict[str, Any],
    output_root: Path,
) -> list[IndexResult]:
    """Build dry-run records: one per index, no I/O."""
    grouped = group_preprocessing_results_by_variable(preprocessing, require_executed=False)
    results: list[IndexResult] = []
    for spec in specs:
        source_paths, missing_variables = _collect_source_paths(spec, grouped)
        output_target = resolve_target_path(output_root, index_output_path(spec.index_id))
        reason = (
            f"{REASON_REQUIRED_VARIABLE_NOT_IN_PLAN}: {sorted(missing_variables)}"
            if missing_variables
            else None
        )
        results.append(
            IndexResult(
                index_id=spec.index_id,
                required_variables=tuple(spec.required_variables),
                source_paths=tuple(source_paths),
                output_path=_as_posix(output_target),
                status=STATUS_PLANNED,
                reason=reason,
            )
        )
    return results


def _collect_source_paths(
    spec: Any,
    grouped: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], set[str]]:
    source_paths: list[str] = []
    missing_variables: set[str] = set()
    for variable in spec.required_variables:
        if variable not in grouped or not grouped[variable]:
            missing_variables.add(variable)
            continue
        for result in grouped[variable]:
            source_paths.append(result["output_path"])
    return source_paths, missing_variables


# ---------------------------------------------------------------------------
# Execute mode
# ---------------------------------------------------------------------------


def execute_index_results(
    specs: list[Any],
    *,
    preprocessing: dict[str, Any],
    output_root: Path,
    overwrite: bool = False,
) -> list[IndexResult]:
    """Open daily standard products, compute each index, write NetCDF outputs."""
    import xarray as xr  # noqa: F401  -- imported lazily; used inside _open_daily_variable

    grouped = group_preprocessing_results_by_variable(preprocessing, require_executed=True)
    results: list[IndexResult] = []
    for spec in specs:
        output_target = resolve_target_path(output_root, index_output_path(spec.index_id))
        output_posix = _as_posix(output_target)
        source_paths, missing_variables = _collect_source_paths(spec, grouped)
        if missing_variables:
            results.append(
                IndexResult(
                    index_id=spec.index_id,
                    required_variables=tuple(spec.required_variables),
                    source_paths=tuple(source_paths),
                    output_path=output_posix,
                    status=STATUS_MISSING_INPUT,
                    reason=f"{REASON_SOURCE_NOT_PREPROCESSED}: {sorted(missing_variables)}",
                )
            )
            continue
        missing_files = [p for p in source_paths if not Path(p).exists()]
        if missing_files:
            results.append(
                IndexResult(
                    index_id=spec.index_id,
                    required_variables=tuple(spec.required_variables),
                    source_paths=tuple(source_paths),
                    output_path=output_posix,
                    status=STATUS_MISSING_INPUT,
                    reason=f"{REASON_SOURCE_NOT_FOUND}: {missing_files}",
                )
            )
            continue
        if output_target.exists() and not overwrite:
            results.append(
                IndexResult(
                    index_id=spec.index_id,
                    required_variables=tuple(spec.required_variables),
                    source_paths=tuple(source_paths),
                    output_path=output_posix,
                    status=STATUS_SKIPPED,
                )
            )
            continue
        try:
            daily = _load_daily_variables(spec.required_variables, grouped)
            index_da = _dispatch_compute(spec.index_id, daily)
            output_target.parent.mkdir(parents=True, exist_ok=True)
            index_da.to_netcdf(output_target)
        except Exception as exc:  # noqa: BLE001 - adapter boundary captures failures into the manifest
            results.append(
                IndexResult(
                    index_id=spec.index_id,
                    required_variables=tuple(spec.required_variables),
                    source_paths=tuple(source_paths),
                    output_path=output_posix,
                    status=STATUS_FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(
            IndexResult(
                index_id=spec.index_id,
                required_variables=tuple(spec.required_variables),
                source_paths=tuple(source_paths),
                output_path=output_posix,
                status=STATUS_COMPUTED,
            )
        )
    return results


def _load_daily_variables(
    required_variables: Iterable[str],
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Open per-year daily NetCDFs individually and concatenate along ``time``.

    We avoid ``xr.open_mfdataset`` because it requires ``dask``. Daily
    products are small (one variable, one year each), so loading
    eagerly and concatenating is cheap and keeps the dependency
    surface narrow.
    """
    import xarray as xr

    loaded: dict[str, Any] = {}
    for variable in required_variables:
        paths = [r["output_path"] for r in grouped[variable]]
        per_year: list[Any] = []
        for path in paths:
            with xr.open_dataset(path) as ds:
                if variable not in ds.data_vars:
                    raise IndexManifestError(
                        f"variable {variable!r} not present in {path}; "
                        f"data_vars = {list(ds.data_vars)}"
                    )
                per_year.append(ds[variable].load())
        if len(per_year) == 1:
            loaded[variable] = per_year[0]
        else:
            loaded[variable] = xr.concat(per_year, dim="time")
    return loaded


def derive_execution_status(mode: str, results: list[IndexResult]) -> str:
    if mode == MODE_DRY_RUN:
        return EXECUTION_STATUS_PLANNED
    if not results:
        return EXECUTION_STATUS_PLANNED
    statuses = {r.status for r in results}
    has_failure = STATUS_FAILED in statuses or STATUS_MISSING_INPUT in statuses
    has_success = STATUS_COMPUTED in statuses
    if statuses == {STATUS_FAILED} or statuses == {STATUS_MISSING_INPUT} or statuses == {
        STATUS_FAILED,
        STATUS_MISSING_INPUT,
    }:
        return EXECUTION_STATUS_FAILED
    if has_failure:
        return EXECUTION_STATUS_PARTIAL
    if statuses == {STATUS_SKIPPED}:
        return EXECUTION_STATUS_COMPLETE_EXISTING
    if has_success:
        return EXECUTION_STATUS_COMPUTED
    return EXECUTION_STATUS_PARTIAL


# ---------------------------------------------------------------------------
# Manifest building / writing
# ---------------------------------------------------------------------------


def build_index_manifest(
    *,
    preprocessing_manifest: dict[str, Any],
    preprocessing_manifest_path: Path,
    preprocessing_manifest_hash: str,
    mode: str,
    output_root: Path,
    results: list[IndexResult],
    created_by: str,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise IndexManifestError(f"mode {mode!r} is not one of {sorted(VALID_MODES)}")
    counts = {
        STATUS_PLANNED: 0,
        STATUS_COMPUTED: 0,
        STATUS_SKIPPED: 0,
        STATUS_FAILED: 0,
        STATUS_MISSING_INPUT: 0,
    }
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    manifest: dict[str, Any] = {
        "manifest_type": MANIFEST_TYPE_INDEX,
        "preprocessing_manifest_path": _as_posix(preprocessing_manifest_path),
        "preprocessing_manifest_hash": preprocessing_manifest_hash,
        "region_id": preprocessing_manifest["region_id"],
        "region_geometry_hash": preprocessing_manifest["region_geometry_hash"],
        "mode": mode,
        "output_root": _as_posix(output_root),
        "index_count": len(results),
        "planned_count": counts[STATUS_PLANNED],
        "computed_count": counts[STATUS_COMPUTED],
        "skipped_count": counts[STATUS_SKIPPED],
        "failed_count": counts[STATUS_FAILED],
        "missing_input_count": counts[STATUS_MISSING_INPUT],
        "results": [r.to_dict() for r in results],
        "created_by": created_by,
        "requires_network": False,
        "execution_status": derive_execution_status(mode, results),
    }
    return manifest


def write_index_manifest(output_path: Path, manifest: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")


def _as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


__all__ = [
    "EXECUTION_STATUS_COMPLETE_EXISTING",
    "EXECUTION_STATUS_COMPUTED",
    "EXECUTION_STATUS_FAILED",
    "EXECUTION_STATUS_PARTIAL",
    "EXECUTION_STATUS_PLANNED",
    "INDEX_FAMILY_ALL",
    "INDEX_FAMILY_PRECIPITATION",
    "INDEX_FAMILY_TEMPERATURE",
    "IndexManifestError",
    "IndexResult",
    "MANIFEST_TYPE_INDEX",
    "MANIFEST_TYPE_PREPROCESSING",
    "MODE_DRY_RUN",
    "MODE_EXECUTE",
    "PRECIPITATION_INDEX_FUNCTIONS",
    "PRECIPITATION_INDEX_SPECS",
    "PRECIPITATION_INDEX_SPECS_BY_ID",
    "PREPROCESSING_MANIFEST_REQUIRED_FIELDS",
    "PrecipitationIndexError",
    "REASON_REQUIRED_VARIABLE_NOT_IN_PLAN",
    "REASON_SOURCE_NOT_FOUND",
    "REASON_SOURCE_NOT_PREPROCESSED",
    "STATUS_COMPUTED",
    "STATUS_FAILED",
    "STATUS_MISSING_INPUT",
    "STATUS_PLANNED",
    "STATUS_SKIPPED",
    "SUPPORTED_INDEX_FAMILIES",
    "TEMPERATURE_INDEX_FUNCTIONS",
    "TEMPERATURE_INDEX_SPECS",
    "TEMPERATURE_INDEX_SPECS_BY_ID",
    "TemperatureIndexError",
    "VALID_MODES",
    "build_index_manifest",
    "compute_manifest_hash",
    "derive_execution_status",
    "execute_index_results",
    "group_preprocessing_results_by_variable",
    "index_output_path",
    "load_preprocessing_manifest",
    "plan_index_results",
    "select_index_specs",
    "specs_for_family",
    "write_index_manifest",
]
