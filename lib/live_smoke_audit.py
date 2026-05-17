"""Live smoke audit foundation for milestone 011.

Two modes:

- ``preflight`` (default): no network, no NetCDF reads. Hashes the
  canonical M010 preflight plan and the M010 safety-corrections
  review file, confirms the expected scratch ``--expected-output-root``
  is not the canonical ``runs/dev_region`` root, and writes a
  deterministic
  ``runs/dev_region/live_smoke_audit_plan.json``.
- ``audit``: reads an existing M010 execute report (default
  ``runs/live_smoke_tmax_2000/live_smoke_report.json``) and the four
  expected product files under the scratch root. Hashes every
  artifact, runs the M009 ``validate_daily_product`` /
  ``validate_index_product`` helpers on the daily and index NetCDFs,
  and writes a deterministic audit report. Never imports ``cdsapi``.

This module is the second half of the M010 boundary contract:
M010 produces the live-smoke plan; M011 audits the products that
plan was supposed to write. Both halves must be runnable from a
fresh checkout with no credentials and no `.nc` files committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Re-export a handful of M010 constants so callers do not need to
# import both modules. Keep the import light (no cdsapi, no xarray).
from . import live_smoke as _live_smoke
from .live_smoke import (
    ALLOWED_INDEX_IDS,
    ALLOWED_REQUEST_IDS,
    CANONICAL_DEV_OUTPUT_ROOT,
    CONFIRMATION_TOKEN,
    DEFAULT_OUTPUT_ROOT as M010_DEFAULT_OUTPUT_ROOT,
    DEFAULT_REQUEST_ID,
    MANIFEST_TYPE_LIVE_SMOKE,
    MODE_EXECUTE as M010_MODE_EXECUTE,
    PREFLIGHT_EXECUTION_STATUS_READY as M010_PREFLIGHT_STATUS_READY,
    _resolves_to_canonical_dev_root,
)

MANIFEST_TYPE_AUDIT = "era5_land_live_smoke_audit"

MODE_PREFLIGHT = "preflight"
MODE_AUDIT = "audit"
SUPPORTED_MODES = frozenset({MODE_PREFLIGHT, MODE_AUDIT})

DEFAULT_EXPECTED_OUTPUT_ROOT = M010_DEFAULT_OUTPUT_ROOT
DEFAULT_LIVE_REPORT_NAME = "live_smoke_report.json"
DEFAULT_AUDIT_REPORT_NAME = "live_smoke_audit_report.json"
DEFAULT_AUDIT_PLAN_PATH = "runs/dev_region/live_smoke_audit_plan.json"

PREFLIGHT_EXECUTION_STATUS_READY = "ready_for_live_smoke_audit"
PREFLIGHT_EXECUTION_STATUS_BLOCKED = "blocked"
AUDIT_EXECUTION_STATUS_PASSED = "audit_passed"
AUDIT_EXECUTION_STATUS_PASSED_WITH_WARNINGS = "audit_passed_with_warnings"
AUDIT_EXECUTION_STATUS_FAILED = "audit_failed"

CHECK_STATUS_PASSED = "passed"
CHECK_STATUS_FAILED = "failed"
CHECK_STATUS_WARNING = "warning"

# Preflight check ids in stable order.
CHECK_M010_PLAN_PRESENT = "m010_preflight_plan_present"
CHECK_M010_PLAN_READY = "m010_preflight_plan_ready"
CHECK_M010_SAFETY_REVIEW_PRESENT = "m010_safety_corrections_review_present"
CHECK_EXPECTED_OUTPUT_ROOT_IS_SCRATCH = "expected_output_root_is_scratch"
CHECK_NO_CREDENTIALS_REQUIRED = "no_credentials_required_for_audit_preflight"
CHECK_NO_NETCDF_REQUIRED_FOR_PREFLIGHT = "no_netcdf_required_for_audit_preflight"

CANONICAL_PREFLIGHT_CHECK_ORDER: tuple[str, ...] = (
    CHECK_M010_PLAN_PRESENT,
    CHECK_M010_PLAN_READY,
    CHECK_M010_SAFETY_REVIEW_PRESENT,
    CHECK_EXPECTED_OUTPUT_ROOT_IS_SCRATCH,
    CHECK_NO_CREDENTIALS_REQUIRED,
    CHECK_NO_NETCDF_REQUIRED_FOR_PREFLIGHT,
)

# Audit check ids in stable order.
CHECK_LIVE_REPORT_PRESENT = "live_report_present"
CHECK_LIVE_REPORT_MANIFEST_TYPE = "live_report_manifest_type"
CHECK_LIVE_REPORT_MODE_EXECUTE = "live_report_mode_execute"
CHECK_LIVE_REPORT_REQUIRES_NETWORK = "live_report_requires_network_true"
CHECK_LIVE_REPORT_REQUEST_ID = "live_report_request_id_matches"
CHECK_LIVE_REPORT_STEPS_SUCCEEDED = "live_report_all_steps_succeeded"
CHECK_LIVE_REPORT_OUTPUT_ROOT_SCRATCH = "live_report_output_root_is_scratch"
CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED = (
    "live_report_output_root_matches_expected_root"
)
CHECK_PRODUCTS_PRESENT = "expected_products_present"
CHECK_DAILY_PRODUCT_SCHEMA = "daily_tmax_product_schema"
CHECK_TXX_PRODUCT_SCHEMA = "TXx_index_product_schema"
CHECK_SU_PRODUCT_SCHEMA = "SU_index_product_schema"

CANONICAL_AUDIT_CHECK_ORDER: tuple[str, ...] = (
    CHECK_LIVE_REPORT_PRESENT,
    CHECK_LIVE_REPORT_MANIFEST_TYPE,
    CHECK_LIVE_REPORT_MODE_EXECUTE,
    CHECK_LIVE_REPORT_REQUIRES_NETWORK,
    CHECK_LIVE_REPORT_REQUEST_ID,
    CHECK_LIVE_REPORT_STEPS_SUCCEEDED,
    CHECK_LIVE_REPORT_OUTPUT_ROOT_SCRATCH,
    CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED,
    CHECK_PRODUCTS_PRESENT,
    CHECK_DAILY_PRODUCT_SCHEMA,
    CHECK_TXX_PRODUCT_SCHEMA,
    CHECK_SU_PRODUCT_SCHEMA,
)


class LiveSmokeAuditError(ValueError):
    """Raised on bad inputs (missing config, unsupported mode, etc.)."""


@dataclass(frozen=True)
class CheckRecord:
    check_id: str
    status: str
    message: str
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "check_id": self.check_id,
            "status": self.status,
            "message": self.message,
        }
        if self.artifact_path is not None:
            record["artifact_path"] = self.artifact_path
        return record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_posix(p: Path | str) -> str:
    return str(p).replace("\\", "/")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LiveSmokeAuditError(f"could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise LiveSmokeAuditError(f"{path} is not a JSON object")
    return data


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise LiveSmokeAuditError(f"config not found: {path}")
    if not path.is_file():
        raise LiveSmokeAuditError(f"config path is not a file: {path}")
    data = _read_json(path)
    for required in ("run", "live_smoke"):
        if required not in data:
            raise LiveSmokeAuditError(
                f"config {path} missing required top-level key {required!r}"
            )
    return data


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


def _check_m010_plan(plan_path: Path) -> tuple[CheckRecord, CheckRecord, dict[str, Any] | None]:
    """Return (present_check, ready_check, parsed_plan_or_None)."""
    if not plan_path.exists():
        present = CheckRecord(
            CHECK_M010_PLAN_PRESENT, CHECK_STATUS_FAILED,
            f"M010 preflight plan missing: {plan_path}",
            artifact_path=_as_posix(plan_path),
        )
        ready = CheckRecord(
            CHECK_M010_PLAN_READY, CHECK_STATUS_FAILED,
            "skipped: M010 preflight plan missing",
        )
        return present, ready, None
    try:
        plan = _read_json(plan_path)
    except LiveSmokeAuditError as exc:
        present = CheckRecord(
            CHECK_M010_PLAN_PRESENT, CHECK_STATUS_FAILED,
            f"M010 preflight plan unreadable: {exc}",
            artifact_path=_as_posix(plan_path),
        )
        ready = CheckRecord(
            CHECK_M010_PLAN_READY, CHECK_STATUS_FAILED,
            "skipped: M010 preflight plan unreadable",
        )
        return present, ready, None
    present = CheckRecord(
        CHECK_M010_PLAN_PRESENT, CHECK_STATUS_PASSED,
        f"M010 preflight plan present and readable",
        artifact_path=_as_posix(plan_path),
    )
    status = plan.get("execution_status")
    if status != M010_PREFLIGHT_STATUS_READY:
        ready = CheckRecord(
            CHECK_M010_PLAN_READY, CHECK_STATUS_FAILED,
            f"M010 preflight plan execution_status={status!r}, "
            f"expected {M010_PREFLIGHT_STATUS_READY!r}",
            artifact_path=_as_posix(plan_path),
        )
    else:
        ready = CheckRecord(
            CHECK_M010_PLAN_READY, CHECK_STATUS_PASSED,
            f"M010 preflight plan execution_status="
            f"{M010_PREFLIGHT_STATUS_READY!r}",
            artifact_path=_as_posix(plan_path),
        )
    return present, ready, plan


def _check_m010_safety_review(review_path: Path) -> CheckRecord:
    if not review_path.exists():
        return CheckRecord(
            CHECK_M010_SAFETY_REVIEW_PRESENT, CHECK_STATUS_FAILED,
            f"M010 safety-corrections review missing: {review_path}",
            artifact_path=_as_posix(review_path),
        )
    return CheckRecord(
        CHECK_M010_SAFETY_REVIEW_PRESENT, CHECK_STATUS_PASSED,
        "M010 safety-corrections review present",
        artifact_path=_as_posix(review_path),
    )


def _check_expected_output_root_is_scratch(expected_output_root: Path) -> CheckRecord:
    posix = _as_posix(expected_output_root).rstrip("/")
    canonical = CANONICAL_DEV_OUTPUT_ROOT.rstrip("/")
    if not posix:
        return CheckRecord(
            CHECK_EXPECTED_OUTPUT_ROOT_IS_SCRATCH, CHECK_STATUS_FAILED,
            "expected_output_root must not be empty",
        )
    if posix == canonical or _resolves_to_canonical_dev_root(expected_output_root):
        return CheckRecord(
            CHECK_EXPECTED_OUTPUT_ROOT_IS_SCRATCH, CHECK_STATUS_FAILED,
            f"expected_output_root {posix!r} resolves to the canonical dev "
            f"root; choose a scratch path under runs/live_smoke*/",
        )
    return CheckRecord(
        CHECK_EXPECTED_OUTPUT_ROOT_IS_SCRATCH, CHECK_STATUS_PASSED,
        f"expected_output_root {posix!r} is a scratch directory",
    )


def _check_no_credentials_required_for_audit_preflight() -> CheckRecord:
    return CheckRecord(
        CHECK_NO_CREDENTIALS_REQUIRED, CHECK_STATUS_PASSED,
        "audit preflight does not import cdsapi, does not read ~/.cdsapirc, "
        "and does not inspect CDSAPI_* env vars",
    )


def _check_no_netcdf_required_for_audit_preflight() -> CheckRecord:
    return CheckRecord(
        CHECK_NO_NETCDF_REQUIRED_FOR_PREFLIGHT, CHECK_STATUS_PASSED,
        "audit preflight does not require NetCDF files on disk; full audit "
        "is the audit mode entry point",
    )


def run_preflight_checks(
    *,
    m010_plan_path: Path,
    m010_safety_review_path: Path,
    expected_output_root: Path,
) -> tuple[list[CheckRecord], dict[str, Any] | None]:
    """Run the six preflight checks; return records and parsed M010 plan."""
    checks: list[CheckRecord] = []
    present, ready, plan = _check_m010_plan(m010_plan_path)
    checks.append(present)
    checks.append(ready)
    checks.append(_check_m010_safety_review(m010_safety_review_path))
    checks.append(_check_expected_output_root_is_scratch(expected_output_root))
    checks.append(_check_no_credentials_required_for_audit_preflight())
    checks.append(_check_no_netcdf_required_for_audit_preflight())
    return checks, plan


# ---------------------------------------------------------------------------
# Preflight orchestration
# ---------------------------------------------------------------------------


def _hash_prerequisites(
    *,
    m010_plan_path: Path,
    m010_safety_review_path: Path,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, p in (
        ("m010_live_smoke_plan", m010_plan_path),
        ("m010_safety_corrections_review", m010_safety_review_path),
    ):
        if not p.exists():
            continue
        out.append({
            "name": name,
            "path": _as_posix(p),
            "hash": compute_file_hash(p),
        })
    return out


def _expected_artifact_paths(
    expected_output_root: Path, request_id: str
) -> dict[str, str]:
    # The M010 execute plan writes one daily-statistics NetCDF, one
    # daily product, and two indices. Mirror M010's _build_planned_outputs.
    year = 2000
    project_variable = "tmax"
    raw_dir = "raw/era5_land/daily_statistics"
    return {
        "raw_target_path": _as_posix(
            expected_output_root / raw_dir / project_variable
            / f"{year}.nc"
        ),
        "daily_tmax_path": _as_posix(
            expected_output_root / "intermediate" / "daily"
            / project_variable / f"{year}.nc"
        ),
        "TXx_index_path": _as_posix(
            expected_output_root / "derived" / "indices" / "TXx.nc"
        ),
        "SU_index_path": _as_posix(
            expected_output_root / "derived" / "indices" / "SU.nc"
        ),
        "live_report_path": _as_posix(
            expected_output_root / DEFAULT_LIVE_REPORT_NAME
        ),
        "audit_report_path": _as_posix(
            expected_output_root / DEFAULT_AUDIT_REPORT_NAME
        ),
    }


def run_preflight(
    *,
    m010_plan_path: Path,
    m010_safety_review_path: Path,
    expected_output_root: Path,
    request_id: str = DEFAULT_REQUEST_ID,
) -> dict[str, Any]:
    checks, m010_plan = run_preflight_checks(
        m010_plan_path=m010_plan_path,
        m010_safety_review_path=m010_safety_review_path,
        expected_output_root=expected_output_root,
    )
    overall_passed = all(c.status == CHECK_STATUS_PASSED for c in checks)
    execution_status = (
        PREFLIGHT_EXECUTION_STATUS_READY if overall_passed
        else PREFLIGHT_EXECUTION_STATUS_BLOCKED
    )
    prerequisite_hashes = _hash_prerequisites(
        m010_plan_path=m010_plan_path,
        m010_safety_review_path=m010_safety_review_path,
    )
    expected_artifacts = _expected_artifact_paths(expected_output_root, request_id)
    plan: dict[str, Any] = {
        "manifest_type": MANIFEST_TYPE_AUDIT,
        "mode": MODE_PREFLIGHT,
        "requires_network": False,
        "request_id": request_id,
        "allowed_request_ids": sorted(ALLOWED_REQUEST_IDS),
        "allowed_indices": list(ALLOWED_INDEX_IDS),
        "expected_output_root": _as_posix(expected_output_root),
        "expected_artifacts": expected_artifacts,
        "prerequisite_artifacts": prerequisite_hashes,
        "preflight_checks": [c.to_dict() for c in checks],
        "audit_checks_planned": list(CANONICAL_AUDIT_CHECK_ORDER),
        "created_by": "scripts/08_audit_live_smoke.py",
        "execution_status": execution_status,
        "confirmation_token_required": CONFIRMATION_TOKEN,
    }
    return plan


# ---------------------------------------------------------------------------
# Audit mode
# ---------------------------------------------------------------------------


def _hash_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": _as_posix(path),
            "exists": False,
            "hash": None,
            "byte_size": None,
        }
    data = path.read_bytes()
    return {
        "path": _as_posix(path),
        "exists": True,
        "hash": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "byte_size": len(data),
    }


def _check_live_report_present(report_path: Path) -> CheckRecord:
    if not report_path.exists():
        return CheckRecord(
            CHECK_LIVE_REPORT_PRESENT, CHECK_STATUS_FAILED,
            f"live smoke report missing: {report_path}",
            artifact_path=_as_posix(report_path),
        )
    return CheckRecord(
        CHECK_LIVE_REPORT_PRESENT, CHECK_STATUS_PASSED,
        "live smoke report present",
        artifact_path=_as_posix(report_path),
    )


def _check_live_report_schema(report: dict[str, Any]) -> list[CheckRecord]:
    """Run the four single-field schema checks against the live report."""
    checks: list[CheckRecord] = []
    mt = report.get("manifest_type")
    if mt == MANIFEST_TYPE_LIVE_SMOKE:
        checks.append(CheckRecord(
            CHECK_LIVE_REPORT_MANIFEST_TYPE, CHECK_STATUS_PASSED,
            f"manifest_type={MANIFEST_TYPE_LIVE_SMOKE!r}",
        ))
    else:
        checks.append(CheckRecord(
            CHECK_LIVE_REPORT_MANIFEST_TYPE, CHECK_STATUS_FAILED,
            f"manifest_type={mt!r}, expected {MANIFEST_TYPE_LIVE_SMOKE!r}",
        ))
    mode = report.get("mode")
    if mode == M010_MODE_EXECUTE:
        checks.append(CheckRecord(
            CHECK_LIVE_REPORT_MODE_EXECUTE, CHECK_STATUS_PASSED,
            f"mode={M010_MODE_EXECUTE!r}",
        ))
    else:
        checks.append(CheckRecord(
            CHECK_LIVE_REPORT_MODE_EXECUTE, CHECK_STATUS_FAILED,
            f"mode={mode!r}, expected {M010_MODE_EXECUTE!r}",
        ))
    requires_network = report.get("requires_network")
    if requires_network is True:
        checks.append(CheckRecord(
            CHECK_LIVE_REPORT_REQUIRES_NETWORK, CHECK_STATUS_PASSED,
            "requires_network=true",
        ))
    else:
        checks.append(CheckRecord(
            CHECK_LIVE_REPORT_REQUIRES_NETWORK, CHECK_STATUS_FAILED,
            f"requires_network={requires_network!r}, expected true",
        ))
    request_id = report.get("request_id")
    if request_id == DEFAULT_REQUEST_ID:
        checks.append(CheckRecord(
            CHECK_LIVE_REPORT_REQUEST_ID, CHECK_STATUS_PASSED,
            f"request_id={DEFAULT_REQUEST_ID!r}",
        ))
    else:
        checks.append(CheckRecord(
            CHECK_LIVE_REPORT_REQUEST_ID, CHECK_STATUS_FAILED,
            f"request_id={request_id!r}, expected {DEFAULT_REQUEST_ID!r}",
        ))
    return checks


def _check_live_report_steps_succeeded(report: dict[str, Any]) -> CheckRecord:
    steps = report.get("steps") or []
    if not steps:
        return CheckRecord(
            CHECK_LIVE_REPORT_STEPS_SUCCEEDED, CHECK_STATUS_FAILED,
            "live report has no steps",
        )
    bad = [
        f"{s.get('step_id')}={s.get('status')}"
        for s in steps
        if s.get("status") != "succeeded"
    ]
    if bad:
        return CheckRecord(
            CHECK_LIVE_REPORT_STEPS_SUCCEEDED, CHECK_STATUS_FAILED,
            f"not every step succeeded: {bad}",
        )
    return CheckRecord(
        CHECK_LIVE_REPORT_STEPS_SUCCEEDED, CHECK_STATUS_PASSED,
        f"all {len(steps)} steps succeeded",
    )


def _check_live_report_output_root_is_scratch(report: dict[str, Any]) -> CheckRecord:
    output_root = report.get("output_root", "")
    if not output_root:
        return CheckRecord(
            CHECK_LIVE_REPORT_OUTPUT_ROOT_SCRATCH, CHECK_STATUS_FAILED,
            "live report missing output_root",
        )
    if _resolves_to_canonical_dev_root(Path(output_root)):
        return CheckRecord(
            CHECK_LIVE_REPORT_OUTPUT_ROOT_SCRATCH, CHECK_STATUS_FAILED,
            f"live report output_root {output_root!r} resolves to the "
            f"canonical dev root",
        )
    return CheckRecord(
        CHECK_LIVE_REPORT_OUTPUT_ROOT_SCRATCH, CHECK_STATUS_PASSED,
        f"live report output_root {output_root!r} is a scratch directory",
    )


def _check_live_report_output_root_matches_expected(
    report: dict[str, Any], expected_output_root: Path,
) -> CheckRecord:
    """Guard against auditing products that did not come from the
    operator-supplied scratch root.

    A reviewer who points ``--expected-output-root`` at directory A but
    feeds the audit a live report whose ``output_root`` is directory B
    would otherwise still get a green audit (the four expected products
    happen to exist under A, validators happen to pass), even though
    the live run wrote to B. Without this check, the audit's artifact
    hashes and product-validation results have no provenance tie to the
    M010 execute run.
    """
    output_root = report.get("output_root", "")
    if not output_root:
        return CheckRecord(
            CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED, CHECK_STATUS_FAILED,
            "live report missing output_root",
        )
    try:
        report_resolved = Path(output_root).resolve(strict=False)
        expected_resolved = expected_output_root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        return CheckRecord(
            CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED, CHECK_STATUS_FAILED,
            f"could not resolve output roots for comparison: {exc}",
        )
    if report_resolved != expected_resolved:
        return CheckRecord(
            CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED, CHECK_STATUS_FAILED,
            f"live report output_root {output_root!r} resolves to "
            f"{_as_posix(report_resolved)!r}, which does not match "
            f"expected_output_root {_as_posix(expected_resolved)!r}",
        )
    return CheckRecord(
        CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED, CHECK_STATUS_PASSED,
        f"live report output_root resolves to the operator-supplied "
        f"expected_output_root",
    )


def _check_products_present(
    expected: dict[str, str],
) -> tuple[CheckRecord, list[dict[str, Any]]]:
    artifact_records: list[dict[str, Any]] = []
    missing: list[str] = []
    for key in ("raw_target_path", "daily_tmax_path", "TXx_index_path", "SU_index_path"):
        p = Path(expected[key])
        record = _hash_artifact(p)
        record["role"] = key
        artifact_records.append(record)
        if not record["exists"]:
            missing.append(_as_posix(p))
    if missing:
        return (
            CheckRecord(
                CHECK_PRODUCTS_PRESENT, CHECK_STATUS_FAILED,
                f"expected products missing: {missing}",
            ),
            artifact_records,
        )
    return (
        CheckRecord(
            CHECK_PRODUCTS_PRESENT, CHECK_STATUS_PASSED,
            "all four expected products present and hashable",
        ),
        artifact_records,
    )


def _validate_products(
    daily_path: Path,
    txx_path: Path,
    su_path: Path,
) -> tuple[list[CheckRecord], list[dict[str, Any]]]:
    """Run M009 per-file validators; return both audit-side CheckRecords
    and the raw M009 CheckResult dicts for the report."""
    from . import validation as v  # lazy

    raw_results: list[dict[str, Any]] = []
    audit_records: list[CheckRecord] = []

    daily_result = v.validate_daily_product(
        daily_path,
        project_variable="tmax",
        expected_year=2000,
        expected_units=v.TEMPERATURE_DAILY_UNITS,
    )
    raw_results.append(daily_result.to_dict())
    audit_records.append(_audit_record_from_validator(
        CHECK_DAILY_PRODUCT_SCHEMA, daily_result, daily_path,
    ))

    txx_result = v.validate_index_product(
        txx_path, index_id="TXx", expected_units="degC",
    )
    raw_results.append(txx_result.to_dict())
    audit_records.append(_audit_record_from_validator(
        CHECK_TXX_PRODUCT_SCHEMA, txx_result, txx_path,
    ))

    su_result = v.validate_index_product(
        su_path, index_id="SU", expected_units="days",
    )
    raw_results.append(su_result.to_dict())
    audit_records.append(_audit_record_from_validator(
        CHECK_SU_PRODUCT_SCHEMA, su_result, su_path,
    ))
    return audit_records, raw_results


def _audit_record_from_validator(
    audit_check_id: str, result: Any, path: Path,
) -> CheckRecord:
    from . import validation as v  # lazy

    if result.status == v.STATUS_PASSED:
        status = CHECK_STATUS_PASSED
    elif result.status == v.STATUS_WARNING:
        status = CHECK_STATUS_WARNING
    else:
        status = CHECK_STATUS_FAILED
    return CheckRecord(
        audit_check_id, status, result.message, artifact_path=_as_posix(path),
    )


def run_audit(
    *,
    expected_output_root: Path,
    live_report_path: Path,
    request_id: str = DEFAULT_REQUEST_ID,
    product_validator: Callable[..., tuple[list[CheckRecord], list[dict[str, Any]]]] | None = None,
    skip_product_validation: bool = False,
) -> dict[str, Any]:
    """Audit an M010 execute run.

    ``product_validator`` lets tests inject a fake validator. The
    canonical audit uses the M009 helpers via ``_validate_products``.
    ``skip_product_validation=True`` records all three product checks
    as skipped (used when the upstream live report itself is broken or
    the products are missing -- there's nothing useful to validate).
    """
    expected_artifacts = _expected_artifact_paths(expected_output_root, request_id)
    audit_checks: list[CheckRecord] = []

    present_check = _check_live_report_present(live_report_path)
    audit_checks.append(present_check)

    if present_check.status != CHECK_STATUS_PASSED:
        for cid in (
            CHECK_LIVE_REPORT_MANIFEST_TYPE,
            CHECK_LIVE_REPORT_MODE_EXECUTE,
            CHECK_LIVE_REPORT_REQUIRES_NETWORK,
            CHECK_LIVE_REPORT_REQUEST_ID,
            CHECK_LIVE_REPORT_STEPS_SUCCEEDED,
            CHECK_LIVE_REPORT_OUTPUT_ROOT_SCRATCH,
            CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED,
            CHECK_PRODUCTS_PRESENT,
            CHECK_DAILY_PRODUCT_SCHEMA,
            CHECK_TXX_PRODUCT_SCHEMA,
            CHECK_SU_PRODUCT_SCHEMA,
        ):
            audit_checks.append(CheckRecord(
                cid, CHECK_STATUS_FAILED, "skipped: live report missing",
            ))
        return _assemble_audit_report(
            expected_output_root=expected_output_root,
            live_report_path=live_report_path,
            live_report=None,
            request_id=request_id,
            expected_artifacts=expected_artifacts,
            artifact_records=[],
            audit_checks=audit_checks,
            product_validations=[],
        )

    live_report = _read_json(live_report_path)
    audit_checks.extend(_check_live_report_schema(live_report))
    audit_checks.append(_check_live_report_steps_succeeded(live_report))
    audit_checks.append(_check_live_report_output_root_is_scratch(live_report))
    audit_checks.append(_check_live_report_output_root_matches_expected(
        live_report, expected_output_root,
    ))

    products_check, artifact_records = _check_products_present(expected_artifacts)
    audit_checks.append(products_check)

    product_validations: list[dict[str, Any]] = []
    if products_check.status != CHECK_STATUS_PASSED or skip_product_validation:
        reason = (
            "skipped: products missing"
            if products_check.status != CHECK_STATUS_PASSED
            else "skipped: caller requested no product validation"
        )
        for cid in (
            CHECK_DAILY_PRODUCT_SCHEMA, CHECK_TXX_PRODUCT_SCHEMA, CHECK_SU_PRODUCT_SCHEMA,
        ):
            audit_checks.append(CheckRecord(cid, CHECK_STATUS_FAILED, reason))
    else:
        validator = product_validator or _default_product_validator
        validator_checks, product_validations = validator(
            daily_path=Path(expected_artifacts["daily_tmax_path"]),
            txx_path=Path(expected_artifacts["TXx_index_path"]),
            su_path=Path(expected_artifacts["SU_index_path"]),
        )
        audit_checks.extend(validator_checks)

    return _assemble_audit_report(
        expected_output_root=expected_output_root,
        live_report_path=live_report_path,
        live_report=live_report,
        request_id=request_id,
        expected_artifacts=expected_artifacts,
        artifact_records=artifact_records,
        audit_checks=audit_checks,
        product_validations=product_validations,
    )


def _default_product_validator(
    *, daily_path: Path, txx_path: Path, su_path: Path,
) -> tuple[list[CheckRecord], list[dict[str, Any]]]:
    return _validate_products(daily_path, txx_path, su_path)


def _derive_audit_execution_status(checks: list[CheckRecord]) -> str:
    if any(c.status == CHECK_STATUS_FAILED for c in checks):
        return AUDIT_EXECUTION_STATUS_FAILED
    if any(c.status == CHECK_STATUS_WARNING for c in checks):
        return AUDIT_EXECUTION_STATUS_PASSED_WITH_WARNINGS
    return AUDIT_EXECUTION_STATUS_PASSED


def _assemble_audit_report(
    *,
    expected_output_root: Path,
    live_report_path: Path,
    live_report: dict[str, Any] | None,
    request_id: str,
    expected_artifacts: dict[str, str],
    artifact_records: list[dict[str, Any]],
    audit_checks: list[CheckRecord],
    product_validations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "manifest_type": MANIFEST_TYPE_AUDIT,
        "mode": MODE_AUDIT,
        "requires_network": False,
        "request_id": request_id,
        "expected_output_root": _as_posix(expected_output_root),
        "expected_artifacts": expected_artifacts,
        "live_report_path": _as_posix(live_report_path),
        "live_report_hash": (
            compute_file_hash(live_report_path)
            if live_report_path.exists() else None
        ),
        "live_report_request_id": (
            live_report.get("request_id") if live_report is not None else None
        ),
        "live_report_execution_status": (
            live_report.get("execution_status") if live_report is not None else None
        ),
        "artifact_hashes": artifact_records,
        "audit_checks": [c.to_dict() for c in audit_checks],
        "product_validations": product_validations,
        "created_by": "scripts/08_audit_live_smoke.py",
        "execution_status": _derive_audit_execution_status(audit_checks),
    }


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------


def write_plan(output_path: Path, plan: dict[str, Any]) -> None:
    """Write the audit plan/report as deterministic JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")


__all__ = [
    "ALLOWED_INDEX_IDS",
    "ALLOWED_REQUEST_IDS",
    "AUDIT_EXECUTION_STATUS_FAILED",
    "AUDIT_EXECUTION_STATUS_PASSED",
    "AUDIT_EXECUTION_STATUS_PASSED_WITH_WARNINGS",
    "CANONICAL_AUDIT_CHECK_ORDER",
    "CANONICAL_DEV_OUTPUT_ROOT",
    "CANONICAL_PREFLIGHT_CHECK_ORDER",
    "CHECK_DAILY_PRODUCT_SCHEMA",
    "CHECK_EXPECTED_OUTPUT_ROOT_IS_SCRATCH",
    "CHECK_LIVE_REPORT_MANIFEST_TYPE",
    "CHECK_LIVE_REPORT_MODE_EXECUTE",
    "CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED",
    "CHECK_LIVE_REPORT_OUTPUT_ROOT_SCRATCH",
    "CHECK_LIVE_REPORT_PRESENT",
    "CHECK_LIVE_REPORT_REQUEST_ID",
    "CHECK_LIVE_REPORT_REQUIRES_NETWORK",
    "CHECK_LIVE_REPORT_STEPS_SUCCEEDED",
    "CHECK_M010_PLAN_PRESENT",
    "CHECK_M010_PLAN_READY",
    "CHECK_M010_SAFETY_REVIEW_PRESENT",
    "CHECK_NO_CREDENTIALS_REQUIRED",
    "CHECK_NO_NETCDF_REQUIRED_FOR_PREFLIGHT",
    "CHECK_PRODUCTS_PRESENT",
    "CHECK_STATUS_FAILED",
    "CHECK_STATUS_PASSED",
    "CHECK_STATUS_WARNING",
    "CHECK_SU_PRODUCT_SCHEMA",
    "CHECK_TXX_PRODUCT_SCHEMA",
    "CONFIRMATION_TOKEN",
    "CheckRecord",
    "DEFAULT_AUDIT_PLAN_PATH",
    "DEFAULT_AUDIT_REPORT_NAME",
    "DEFAULT_EXPECTED_OUTPUT_ROOT",
    "DEFAULT_LIVE_REPORT_NAME",
    "DEFAULT_REQUEST_ID",
    "LiveSmokeAuditError",
    "MANIFEST_TYPE_AUDIT",
    "MODE_AUDIT",
    "MODE_PREFLIGHT",
    "PREFLIGHT_EXECUTION_STATUS_BLOCKED",
    "PREFLIGHT_EXECUTION_STATUS_READY",
    "SUPPORTED_MODES",
    "compute_file_hash",
    "load_config",
    "run_audit",
    "run_preflight",
    "run_preflight_checks",
    "write_plan",
]
