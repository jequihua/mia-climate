"""Local validation/regression foundation for milestone 009.

Reads the M008 pipeline manifest and the downstream M001-M007
manifests it points at, runs a stable ordered set of graph and
side-effect checks, and writes a deterministic
``validation_report.json``. Heavy NetCDF helpers
(``numpy`` / ``xarray``) are imported lazily inside the product
validators so ``import lib.validation`` stays cheap and dry-run
validation never pulls them in.

Two surfaces:

- ``run_validation`` -- top-level orchestrator used by
  ``scripts/06_validate_outputs.py``. Returns the stable ordered
  list of ``CheckResult`` records that feed ``build_validation_report``.
- ``validate_daily_product`` / ``validate_index_product`` -- pure
  per-NetCDF validators tested with synthetic fixtures. The
  canonical dry-run skips both because no products exist yet; the
  helpers run unchanged once owner-authorized NetCDF appears under
  ``runs/{run_id}/intermediate/`` or ``runs/{run_id}/derived/``.

Live CDS, Docker, cloud, and legacy-NetCDF numeric comparison are
out of scope for M009.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_TYPE_VALIDATION = "era5_land_validation_report"
PIPELINE_MANIFEST_TYPE = "era5_land_pipeline_run"

MODE_DRY_RUN = "dry-run"
SUPPORTED_MODES = frozenset({MODE_DRY_RUN})

STATUS_PASSED = "passed"
STATUS_WARNING = "warning"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"

EXECUTION_STATUS_PASSED = "passed"
EXECUTION_STATUS_PASSED_WITH_WARNINGS = "passed_with_warnings"
EXECUTION_STATUS_FAILED = "failed"

# Check IDs in canonical order. ``CANONICAL_CHECK_ORDER`` is the
# definitive ordering for the ``checks`` array in the report.
CHECK_PIPELINE_MANIFEST_EXISTS = "pipeline_manifest_exists"
CHECK_PIPELINE_MANIFEST_TYPE = "pipeline_manifest_type"
CHECK_PIPELINE_EXECUTION_STATUS = "pipeline_execution_status"
CHECK_PIPELINE_STEP_COUNT = "pipeline_step_count"
CHECK_PIPELINE_STEP_ORDER = "pipeline_step_order"
CHECK_PIPELINE_STEP_OUTPUTS_EXIST = "pipeline_step_outputs_exist"
CHECK_PIPELINE_STEP_HASHES_MATCH = "pipeline_step_output_hashes_match"
CHECK_PIPELINE_REQUIRES_NETWORK = "pipeline_requires_network_false"
CHECK_REGION_ID_CONSISTENT = "region_id_consistent"
CHECK_REGION_HASH_CONSISTENT = "region_geometry_hash_consistent"
CHECK_MANIFESTS_DRY_RUN_ONLY = "downstream_manifests_dry_run_only"
CHECK_NO_NC_FILES = "no_netcdf_files_in_output_root"
CHECK_NO_RAW_DIR = "no_raw_directory"
CHECK_NO_INTERMEDIATE_DIR = "no_intermediate_directory"
CHECK_NO_DERIVED_DIR = "no_derived_directory"
CHECK_DAILY_PRODUCTS = "daily_product_schemas"
CHECK_INDEX_PRODUCTS = "index_product_schemas"

CANONICAL_CHECK_ORDER: tuple[str, ...] = (
    CHECK_PIPELINE_MANIFEST_EXISTS,
    CHECK_PIPELINE_MANIFEST_TYPE,
    CHECK_PIPELINE_EXECUTION_STATUS,
    CHECK_PIPELINE_STEP_COUNT,
    CHECK_PIPELINE_STEP_ORDER,
    CHECK_PIPELINE_STEP_OUTPUTS_EXIST,
    CHECK_PIPELINE_STEP_HASHES_MATCH,
    CHECK_PIPELINE_REQUIRES_NETWORK,
    CHECK_REGION_ID_CONSISTENT,
    CHECK_REGION_HASH_CONSISTENT,
    CHECK_MANIFESTS_DRY_RUN_ONLY,
    CHECK_NO_NC_FILES,
    CHECK_NO_RAW_DIR,
    CHECK_NO_INTERMEDIATE_DIR,
    CHECK_NO_DERIVED_DIR,
    CHECK_DAILY_PRODUCTS,
    CHECK_INDEX_PRODUCTS,
)

EXPECTED_STEP_ORDER: tuple[str, ...] = (
    "validate_region",
    "plan_downloads",
    "acquire_daily_stats_dry_run",
    "acquire_precipitation_dry_run",
    "preprocess_daily_stats_dry_run",
    "preprocess_precipitation_dry_run",
    "indices_temperature_dry_run",
    "indices_precipitation_dry_run",
)

# Map M001-M007 manifest_type -> mode-field expectation for the
# downstream-manifests-dry-run-only check. The region manifest and the
# download plan manifest are special-cased (no ``mode`` field).
DOWNSTREAM_MANIFEST_MODE_FIELD: dict[str, str | None] = {
    None: None,                                 # M001 region manifest has no manifest_type
    "era5_land_download_plan": None,            # M002 has download_execution_status, not mode
    "era5_land_acquisition_run": MODE_DRY_RUN,
    "era5_land_preprocessing_run": MODE_DRY_RUN,
    "era5_land_index_run": MODE_DRY_RUN,
}

TEMPERATURE_DAILY_VARIABLES = ("tmax", "tmin", "tmean", "u10m", "v10m")
PRECIPITATION_DAILY_VARIABLE = "pr"
TEMPERATURE_DAILY_UNITS = "degC"
WIND_DAILY_UNITS_CANDIDATES = ("m/s", "m s-1", "K")  # wind preserves source units in M004
PRECIPITATION_DAILY_UNITS = "mm/day"

INDEX_TEMPERATURE_IDS = ("Tmx", "Tmn", "TXx", "TNn", "DTR", "SU", "TR")
INDEX_PRECIPITATION_IDS = ("PRCPTOT", "RX1day", "R95p", "CDD", "CWD", "R10mm", "R20mm")


class ValidationError(ValueError):
    """Raised when validation cannot run (bad inputs); distinct from check failures."""


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str
    severity: str
    message: str
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "check_id": self.check_id,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
        }
        if self.artifact_path is not None:
            record["artifact_path"] = self.artifact_path
        return record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _as_posix(p: Path | str) -> str:
    return str(p).replace("\\", "/")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ValidationError(f"{label} not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(f"{label} must be a JSON object: {path}")
    return data


def load_pipeline_manifest(path: Path) -> dict[str, Any]:
    return _read_json_object(path, label="pipeline manifest")


# ---------------------------------------------------------------------------
# Manifest-graph check builders
# ---------------------------------------------------------------------------


def _check_manifest_type(pipeline: dict[str, Any], path: Path) -> CheckResult:
    actual = pipeline.get("manifest_type")
    if actual == PIPELINE_MANIFEST_TYPE:
        return CheckResult(
            CHECK_PIPELINE_MANIFEST_TYPE, STATUS_PASSED, SEVERITY_INFO,
            f"manifest_type is {PIPELINE_MANIFEST_TYPE!r}",
            artifact_path=_as_posix(path),
        )
    return CheckResult(
        CHECK_PIPELINE_MANIFEST_TYPE, STATUS_FAILED, SEVERITY_ERROR,
        f"manifest_type {actual!r} != expected {PIPELINE_MANIFEST_TYPE!r}",
        artifact_path=_as_posix(path),
    )


def _check_execution_status(pipeline: dict[str, Any], path: Path) -> CheckResult:
    status = pipeline.get("execution_status")
    if status == "completed_dry_run":
        return CheckResult(
            CHECK_PIPELINE_EXECUTION_STATUS, STATUS_PASSED, SEVERITY_INFO,
            "pipeline execution_status is 'completed_dry_run'",
            artifact_path=_as_posix(path),
        )
    return CheckResult(
        CHECK_PIPELINE_EXECUTION_STATUS, STATUS_FAILED, SEVERITY_ERROR,
        f"pipeline execution_status {status!r} != 'completed_dry_run'",
        artifact_path=_as_posix(path),
    )


def _check_step_count(pipeline: dict[str, Any], path: Path) -> CheckResult:
    steps = pipeline.get("steps") or []
    if len(steps) == len(EXPECTED_STEP_ORDER):
        return CheckResult(
            CHECK_PIPELINE_STEP_COUNT, STATUS_PASSED, SEVERITY_INFO,
            f"pipeline carries {len(steps)} steps",
            artifact_path=_as_posix(path),
        )
    return CheckResult(
        CHECK_PIPELINE_STEP_COUNT, STATUS_FAILED, SEVERITY_ERROR,
        f"step_count {len(steps)} != expected {len(EXPECTED_STEP_ORDER)}",
        artifact_path=_as_posix(path),
    )


def _check_step_order(pipeline: dict[str, Any], path: Path) -> CheckResult:
    ids = tuple(s.get("step_id") for s in pipeline.get("steps") or [])
    if ids == EXPECTED_STEP_ORDER:
        return CheckResult(
            CHECK_PIPELINE_STEP_ORDER, STATUS_PASSED, SEVERITY_INFO,
            "pipeline step_id order matches the canonical M008 sequence",
            artifact_path=_as_posix(path),
        )
    return CheckResult(
        CHECK_PIPELINE_STEP_ORDER, STATUS_FAILED, SEVERITY_ERROR,
        f"pipeline step_id order {ids!r} != expected {EXPECTED_STEP_ORDER!r}",
        artifact_path=_as_posix(path),
    )


def _check_step_outputs_exist(pipeline: dict[str, Any]) -> CheckResult:
    missing: list[str] = []
    for step in pipeline.get("steps") or []:
        op = step.get("output_path")
        if not op or not Path(op).exists():
            missing.append(op or "<no output_path>")
    if not missing:
        return CheckResult(
            CHECK_PIPELINE_STEP_OUTPUTS_EXIST, STATUS_PASSED, SEVERITY_INFO,
            "every pipeline step's declared output_path exists on disk",
        )
    return CheckResult(
        CHECK_PIPELINE_STEP_OUTPUTS_EXIST, STATUS_FAILED, SEVERITY_ERROR,
        f"missing step outputs: {missing}",
    )


def _check_step_output_hashes(pipeline: dict[str, Any]) -> CheckResult:
    mismatched: list[str] = []
    for step in pipeline.get("steps") or []:
        op = step.get("output_path")
        expected = step.get("output_hash")
        if not op or expected is None:
            continue
        try:
            actual = compute_file_hash(Path(op))
        except FileNotFoundError:
            mismatched.append(f"{op}: file missing")
            continue
        if actual != expected:
            mismatched.append(f"{op}: expected {expected} got {actual}")
    if not mismatched:
        return CheckResult(
            CHECK_PIPELINE_STEP_HASHES_MATCH, STATUS_PASSED, SEVERITY_INFO,
            "every pipeline step's output_hash matches the file bytes",
        )
    return CheckResult(
        CHECK_PIPELINE_STEP_HASHES_MATCH, STATUS_FAILED, SEVERITY_ERROR,
        f"output_hash mismatches: {mismatched}",
    )


def _check_requires_network(pipeline: dict[str, Any], path: Path) -> CheckResult:
    if pipeline.get("requires_network") is False:
        return CheckResult(
            CHECK_PIPELINE_REQUIRES_NETWORK, STATUS_PASSED, SEVERITY_INFO,
            "pipeline manifest declares requires_network = false",
            artifact_path=_as_posix(path),
        )
    return CheckResult(
        CHECK_PIPELINE_REQUIRES_NETWORK, STATUS_FAILED, SEVERITY_ERROR,
        f"pipeline manifest requires_network={pipeline.get('requires_network')!r} != False",
        artifact_path=_as_posix(path),
    )


def _load_downstream_manifests(pipeline: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    """Open each pipeline step's JSON output and return ``[(path, data), ...]``."""
    out: list[tuple[Path, dict[str, Any]]] = []
    for step in pipeline.get("steps") or []:
        op = step.get("output_path")
        if not op:
            continue
        p = Path(op)
        if not p.exists() or not p.suffix == ".json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.append((p, data))
    return out


def _check_region_id_consistent(downstream: list[tuple[Path, dict[str, Any]]]) -> CheckResult:
    ids: set[str] = set()
    for path, data in downstream:
        rid = data.get("region_id")
        if rid is not None:
            ids.add(rid)
    if len(ids) <= 1:
        return CheckResult(
            CHECK_REGION_ID_CONSISTENT, STATUS_PASSED, SEVERITY_INFO,
            f"region_id consistent across downstream manifests: {sorted(ids)}",
        )
    return CheckResult(
        CHECK_REGION_ID_CONSISTENT, STATUS_FAILED, SEVERITY_ERROR,
        f"region_id disagreement across downstream manifests: {sorted(ids)}",
    )


def _check_region_hash_consistent(downstream: list[tuple[Path, dict[str, Any]]]) -> CheckResult:
    hashes: set[str] = set()
    for path, data in downstream:
        h = data.get("region_geometry_hash") or data.get("geometry_hash")
        if h is not None:
            hashes.add(h)
    if len(hashes) <= 1:
        return CheckResult(
            CHECK_REGION_HASH_CONSISTENT, STATUS_PASSED, SEVERITY_INFO,
            f"region geometry hash consistent across downstream manifests: {sorted(hashes)}",
        )
    return CheckResult(
        CHECK_REGION_HASH_CONSISTENT, STATUS_FAILED, SEVERITY_ERROR,
        f"region geometry hash disagreement across downstream manifests: {sorted(hashes)}",
    )


def _check_downstream_dry_run_only(downstream: list[tuple[Path, dict[str, Any]]]) -> CheckResult:
    offenders: list[str] = []
    for path, data in downstream:
        mt = data.get("manifest_type")
        expected_mode = DOWNSTREAM_MANIFEST_MODE_FIELD.get(mt)
        if expected_mode is not None:
            actual_mode = data.get("mode")
            if actual_mode != expected_mode:
                offenders.append(f"{path.name}: mode={actual_mode!r} != {expected_mode!r}")
        # requires_network must be false where present; absence is fine for M001 region manifest.
        if "requires_network" in data and data["requires_network"] is not False:
            offenders.append(f"{path.name}: requires_network={data['requires_network']!r} != False")
    if not offenders:
        return CheckResult(
            CHECK_MANIFESTS_DRY_RUN_ONLY, STATUS_PASSED, SEVERITY_INFO,
            "every downstream manifest declares mode=dry-run (where applicable) and requires_network=false",
        )
    return CheckResult(
        CHECK_MANIFESTS_DRY_RUN_ONLY, STATUS_FAILED, SEVERITY_ERROR,
        f"downstream manifests not dry-run-only: {offenders}",
    )


# ---------------------------------------------------------------------------
# Side-effect policy checks
# ---------------------------------------------------------------------------


def _check_no_netcdf_files(output_root: Path) -> CheckResult:
    found = [str(p) for p in output_root.rglob("*.nc")]
    found.extend(str(p) for p in output_root.rglob("*.nc4"))
    if not found:
        return CheckResult(
            CHECK_NO_NC_FILES, STATUS_PASSED, SEVERITY_INFO,
            f"no NetCDF files under {_as_posix(output_root)}",
            artifact_path=_as_posix(output_root),
        )
    return CheckResult(
        CHECK_NO_NC_FILES, STATUS_FAILED, SEVERITY_ERROR,
        f"unexpected NetCDF files present under {_as_posix(output_root)}: {found}",
        artifact_path=_as_posix(output_root),
    )


def _check_no_directory(output_root: Path, subdir: str, *, check_id: str) -> CheckResult:
    target = output_root / subdir
    if not target.exists():
        return CheckResult(
            check_id, STATUS_PASSED, SEVERITY_INFO,
            f"no {subdir!r} directory under {_as_posix(output_root)}",
            artifact_path=_as_posix(target),
        )
    return CheckResult(
        check_id, STATUS_FAILED, SEVERITY_ERROR,
        f"unexpected {subdir!r} directory present under {_as_posix(output_root)}",
        artifact_path=_as_posix(target),
    )


# ---------------------------------------------------------------------------
# Product-schema validators (lazy NetCDF imports)
# ---------------------------------------------------------------------------


def validate_daily_product(
    path: Path,
    *,
    project_variable: str,
    expected_year: int | None = None,
    expected_units: str | tuple[str, ...] | None = None,
) -> CheckResult:
    """Validate a single daily-product NetCDF.

    Checks: variable name present; ``time`` / ``lat`` / ``lon`` coords;
    units match expectation if given; year coverage matches
    ``expected_year`` when given; at least one finite cell exists;
    not an all-NaN product. Returns a single ``CheckResult`` whose
    ``check_id`` is namespaced ``daily/{project_variable}/{year}``.
    """
    import numpy as np
    import xarray as xr

    check_id = f"daily/{project_variable}/{expected_year or 'any'}"
    if not path.exists():
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"daily product missing: {path}",
                           artifact_path=_as_posix(path))
    try:
        with xr.open_dataset(path) as ds:
            ds.load()
    except Exception as exc:  # noqa: BLE001 -- propagate as check failure
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"could not open daily product {path}: {exc}",
                           artifact_path=_as_posix(path))
    if project_variable not in ds.data_vars:
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"variable {project_variable!r} not in {list(ds.data_vars)}",
                           artifact_path=_as_posix(path))
    da = ds[project_variable]
    for coord in ("time", "lat", "lon"):
        if coord not in da.coords and coord not in da.dims:
            return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                               f"coord/dim {coord!r} missing from {project_variable}",
                               artifact_path=_as_posix(path))
    if expected_units is not None:
        units = da.attrs.get("units")
        candidates = (expected_units,) if isinstance(expected_units, str) else tuple(expected_units)
        if units not in candidates:
            return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                               f"units {units!r} not in expected {candidates!r}",
                               artifact_path=_as_posix(path))
    if expected_year is not None:
        years = sorted({int(np.datetime64(t, "Y").astype(int) + 1970) for t in da["time"].values})
        if years != [expected_year]:
            return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                               f"year coverage {years!r} != expected [{expected_year}]",
                               artifact_path=_as_posix(path))
    arr = np.asarray(da.values, dtype="float64")
    if np.isnan(arr).all():
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"all values are NaN for {project_variable}",
                           artifact_path=_as_posix(path))
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"no finite values for {project_variable}",
                           artifact_path=_as_posix(path))
    # All-zero is suspicious for precipitation / temperature; warn rather than fail.
    finite_values = arr[finite_mask]
    if finite_values.size > 0 and (finite_values == 0).all():
        return CheckResult(check_id, STATUS_WARNING, SEVERITY_WARNING,
                           f"all finite values are zero for {project_variable}",
                           artifact_path=_as_posix(path))
    return CheckResult(check_id, STATUS_PASSED, SEVERITY_INFO,
                       f"daily product {project_variable} for {expected_year or 'any'} ok",
                       artifact_path=_as_posix(path))


def validate_index_product(
    path: Path,
    *,
    index_id: str,
    expected_units: str | tuple[str, ...] | None = None,
) -> CheckResult:
    """Validate a single annual index-product NetCDF."""
    import numpy as np
    import xarray as xr

    check_id = f"index/{index_id}"
    if not path.exists():
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"index product missing: {path}",
                           artifact_path=_as_posix(path))
    try:
        with xr.open_dataset(path) as ds:
            ds.load()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"could not open index product {path}: {exc}",
                           artifact_path=_as_posix(path))
    if index_id not in ds.data_vars:
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"index variable {index_id!r} not in {list(ds.data_vars)}",
                           artifact_path=_as_posix(path))
    da = ds[index_id]
    if "time" not in da.coords and "time" not in da.dims:
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"index {index_id!r} missing 'time' axis",
                           artifact_path=_as_posix(path))
    if expected_units is not None:
        units = da.attrs.get("units")
        candidates = (expected_units,) if isinstance(expected_units, str) else tuple(expected_units)
        if units not in candidates:
            return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                               f"index {index_id!r} units {units!r} not in expected {candidates!r}",
                               artifact_path=_as_posix(path))
    arr = np.asarray(da.values, dtype="float64")
    if np.isnan(arr).all():
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"index {index_id!r} is all-NaN",
                           artifact_path=_as_posix(path))
    if not np.isfinite(arr).any():
        return CheckResult(check_id, STATUS_FAILED, SEVERITY_ERROR,
                           f"index {index_id!r} has no finite values",
                           artifact_path=_as_posix(path))
    return CheckResult(check_id, STATUS_PASSED, SEVERITY_INFO,
                       f"index product {index_id!r} ok",
                       artifact_path=_as_posix(path))


def _check_daily_products(output_root: Path) -> CheckResult:
    intermediate = output_root / "intermediate" / "daily"
    if not intermediate.exists():
        return CheckResult(
            CHECK_DAILY_PRODUCTS, STATUS_SKIPPED, SEVERITY_INFO,
            f"no daily products to validate under {_as_posix(intermediate)} (dry-run state)",
            artifact_path=_as_posix(intermediate),
        )
    # When products exist, this check would iterate and aggregate.
    # M009 implements per-file validators via validate_daily_product
    # and leaves the aggregator for the slice that first writes real
    # daily products.
    return CheckResult(
        CHECK_DAILY_PRODUCTS, STATUS_WARNING, SEVERITY_WARNING,
        f"daily products present under {_as_posix(intermediate)} but per-file validation aggregation is deferred",
        artifact_path=_as_posix(intermediate),
    )


def _check_index_products(output_root: Path) -> CheckResult:
    derived = output_root / "derived" / "indices"
    if not derived.exists():
        return CheckResult(
            CHECK_INDEX_PRODUCTS, STATUS_SKIPPED, SEVERITY_INFO,
            f"no index products to validate under {_as_posix(derived)} (dry-run state)",
            artifact_path=_as_posix(derived),
        )
    return CheckResult(
        CHECK_INDEX_PRODUCTS, STATUS_WARNING, SEVERITY_WARNING,
        f"index products present under {_as_posix(derived)} but per-file validation aggregation is deferred",
        artifact_path=_as_posix(derived),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_validation(
    *,
    pipeline_manifest_path: Path,
    output_root: Path,
    mode: str = MODE_DRY_RUN,
) -> list[CheckResult]:
    """Run the canonical check sequence and return results in stable order."""
    if mode not in SUPPORTED_MODES:
        raise ValidationError(f"mode {mode!r} is not one of {sorted(SUPPORTED_MODES)}")

    results: list[CheckResult] = []
    if not pipeline_manifest_path.exists():
        results.append(CheckResult(
            CHECK_PIPELINE_MANIFEST_EXISTS, STATUS_FAILED, SEVERITY_ERROR,
            f"pipeline manifest not found: {pipeline_manifest_path}",
            artifact_path=_as_posix(pipeline_manifest_path),
        ))
        for cid in CANONICAL_CHECK_ORDER[1:]:
            results.append(CheckResult(
                cid, STATUS_SKIPPED, SEVERITY_INFO,
                "skipped: pipeline manifest missing",
            ))
        return results
    results.append(CheckResult(
        CHECK_PIPELINE_MANIFEST_EXISTS, STATUS_PASSED, SEVERITY_INFO,
        f"pipeline manifest found at {_as_posix(pipeline_manifest_path)}",
        artifact_path=_as_posix(pipeline_manifest_path),
    ))

    try:
        pipeline = load_pipeline_manifest(pipeline_manifest_path)
    except ValidationError as exc:
        results.append(CheckResult(
            CHECK_PIPELINE_MANIFEST_TYPE, STATUS_FAILED, SEVERITY_ERROR,
            f"pipeline manifest unreadable: {exc}",
            artifact_path=_as_posix(pipeline_manifest_path),
        ))
        for cid in CANONICAL_CHECK_ORDER[2:]:
            results.append(CheckResult(cid, STATUS_SKIPPED, SEVERITY_INFO,
                                       "skipped: pipeline manifest unreadable"))
        return results

    results.append(_check_manifest_type(pipeline, pipeline_manifest_path))
    results.append(_check_execution_status(pipeline, pipeline_manifest_path))
    results.append(_check_step_count(pipeline, pipeline_manifest_path))
    results.append(_check_step_order(pipeline, pipeline_manifest_path))
    results.append(_check_step_outputs_exist(pipeline))
    results.append(_check_step_output_hashes(pipeline))
    results.append(_check_requires_network(pipeline, pipeline_manifest_path))

    downstream = _load_downstream_manifests(pipeline)
    results.append(_check_region_id_consistent(downstream))
    results.append(_check_region_hash_consistent(downstream))
    results.append(_check_downstream_dry_run_only(downstream))

    results.append(_check_no_netcdf_files(output_root))
    results.append(_check_no_directory(output_root, "raw", check_id=CHECK_NO_RAW_DIR))
    results.append(_check_no_directory(output_root, "intermediate", check_id=CHECK_NO_INTERMEDIATE_DIR))
    results.append(_check_no_directory(output_root, "derived", check_id=CHECK_NO_DERIVED_DIR))

    results.append(_check_daily_products(output_root))
    results.append(_check_index_products(output_root))

    return results


def derive_execution_status(results: list[CheckResult]) -> str:
    has_failed = any(r.status == STATUS_FAILED for r in results)
    has_warning = any(r.status == STATUS_WARNING for r in results)
    if has_failed:
        return EXECUTION_STATUS_FAILED
    if has_warning:
        return EXECUTION_STATUS_PASSED_WITH_WARNINGS
    return EXECUTION_STATUS_PASSED


def _extract_region_provenance(
    pipeline: dict[str, Any] | None,
    output_root: Path,
) -> tuple[str, str]:
    """Pull (region_id, region_geometry_hash) from the M001 region manifest if available."""
    if pipeline is None:
        return "", ""
    for step in pipeline.get("steps") or []:
        if step.get("step_id") == "validate_region":
            op = step.get("output_path")
            if op and Path(op).exists():
                try:
                    data = json.loads(Path(op).read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return "", ""
                rid = data.get("region_id") or ""
                rh = data.get("geometry_hash") or data.get("region_geometry_hash") or ""
                return rid, rh
    return "", ""


def build_validation_report(
    *,
    pipeline_manifest: dict[str, Any] | None,
    pipeline_manifest_path: Path,
    pipeline_manifest_hash: str | None,
    mode: str,
    output_root: Path,
    results: list[CheckResult],
    created_by: str,
) -> dict[str, Any]:
    if mode not in SUPPORTED_MODES:
        raise ValidationError(f"mode {mode!r} is not supported")
    counts = {STATUS_PASSED: 0, STATUS_WARNING: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    run_id = (pipeline_manifest or {}).get("run_id", "")
    region_id, region_geometry_hash = _extract_region_provenance(pipeline_manifest, output_root)
    report: dict[str, Any] = {
        "manifest_type": MANIFEST_TYPE_VALIDATION,
        "pipeline_manifest_path": _as_posix(pipeline_manifest_path),
        "pipeline_manifest_hash": pipeline_manifest_hash or "",
        "region_id": region_id,
        "region_geometry_hash": region_geometry_hash,
        "mode": mode,
        "run_id": run_id,
        "output_root": _as_posix(output_root),
        "check_count": len(results),
        "passed_count": counts[STATUS_PASSED],
        "warning_count": counts[STATUS_WARNING],
        "failed_count": counts[STATUS_FAILED],
        "skipped_count": counts[STATUS_SKIPPED],
        "checks": [r.to_dict() for r in results],
        "created_by": created_by,
        "requires_network": False,
        "execution_status": derive_execution_status(results),
    }
    return report


def write_validation_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")


__all__ = [
    "CANONICAL_CHECK_ORDER",
    "CHECK_DAILY_PRODUCTS",
    "CHECK_INDEX_PRODUCTS",
    "CHECK_MANIFESTS_DRY_RUN_ONLY",
    "CHECK_NO_DERIVED_DIR",
    "CHECK_NO_INTERMEDIATE_DIR",
    "CHECK_NO_NC_FILES",
    "CHECK_NO_RAW_DIR",
    "CHECK_PIPELINE_EXECUTION_STATUS",
    "CHECK_PIPELINE_MANIFEST_EXISTS",
    "CHECK_PIPELINE_MANIFEST_TYPE",
    "CHECK_PIPELINE_REQUIRES_NETWORK",
    "CHECK_PIPELINE_STEP_COUNT",
    "CHECK_PIPELINE_STEP_HASHES_MATCH",
    "CHECK_PIPELINE_STEP_ORDER",
    "CHECK_PIPELINE_STEP_OUTPUTS_EXIST",
    "CHECK_REGION_HASH_CONSISTENT",
    "CHECK_REGION_ID_CONSISTENT",
    "CheckResult",
    "EXECUTION_STATUS_FAILED",
    "EXECUTION_STATUS_PASSED",
    "EXECUTION_STATUS_PASSED_WITH_WARNINGS",
    "EXPECTED_STEP_ORDER",
    "INDEX_PRECIPITATION_IDS",
    "INDEX_TEMPERATURE_IDS",
    "MANIFEST_TYPE_VALIDATION",
    "MODE_DRY_RUN",
    "PRECIPITATION_DAILY_UNITS",
    "PRECIPITATION_DAILY_VARIABLE",
    "PIPELINE_MANIFEST_TYPE",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "STATUS_FAILED",
    "STATUS_PASSED",
    "STATUS_SKIPPED",
    "STATUS_WARNING",
    "SUPPORTED_MODES",
    "TEMPERATURE_DAILY_UNITS",
    "TEMPERATURE_DAILY_VARIABLES",
    "ValidationError",
    "WIND_DAILY_UNITS_CANDIDATES",
    "build_validation_report",
    "compute_file_hash",
    "derive_execution_status",
    "load_pipeline_manifest",
    "run_validation",
    "validate_daily_product",
    "validate_index_product",
    "write_validation_report",
]
