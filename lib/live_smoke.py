"""Live-smoke readiness orchestration for milestone 010.

Two modes:

- ``preflight`` (default): no network, no NetCDF writes. Walks the
  canonical M001-M009 manifests, hash-checks them via the M009
  validator, looks up the configured smoke ``request_id`` in the
  M002 download manifest, and writes a deterministic plan to
  ``runs/{run_id}/live_smoke_plan.json``.
- ``execute``: owner-authorized live ERA5-Land smoke. Only runs if
  the operator passes a confirmation token equal to
  ``I_UNDERSTAND_THIS_USES_CDS`` and chooses a scratch
  ``--output-root`` other than ``runs/dev_region``. Orchestrates
  exactly four existing scripts in order:

  1. ``scripts/02_download_era5_land.py --mode execute``
     for one ``request_id`` (default ``era5_daily_stats__tmax__2000``).
  2. ``scripts/03_preprocess_daily.py --mode execute``
     for the produced acquisition manifest.
  3. ``scripts/04_compute_indices.py --mode execute --index-id TXx
     --index-id SU`` -- the two indices computable from ``tmax``
     alone.
  4. M009 ``validate_daily_product`` / ``validate_index_product``
     applied to the produced ``tmax`` / ``TXx`` / ``SU`` NetCDF
     files.

This module never imports ``cdsapi`` at module scope. The execute
path leaves the lazy import inside ``lib.acquisition`` /
``lib.preprocessing`` / ``lib.validation`` untouched. Automated
tests inject fake step runners and never call the real scripts in
execute mode.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

MANIFEST_TYPE_LIVE_SMOKE = "era5_land_live_smoke_plan"

MODE_PREFLIGHT = "preflight"
MODE_EXECUTE = "execute"
SUPPORTED_MODES = frozenset({MODE_PREFLIGHT, MODE_EXECUTE})

CONFIRMATION_TOKEN = "I_UNDERSTAND_THIS_USES_CDS"

DEFAULT_REQUEST_ID = "era5_daily_stats__tmax__2000"
DEFAULT_OUTPUT_ROOT = "runs/live_smoke_tmax_2000"
ALLOWED_REQUEST_IDS = frozenset({DEFAULT_REQUEST_ID})
ALLOWED_INDEX_IDS = ("TXx", "SU")
CANONICAL_DEV_OUTPUT_ROOT = "runs/dev_region"

CHECK_CANONICAL_MANIFESTS_PRESENT = "canonical_manifests_present"
CHECK_M009_VALIDATION_PASSED = "m009_validation_passed"
CHECK_REQUEST_IN_DOWNLOAD_MANIFEST = "request_in_download_manifest"
CHECK_REQUEST_IN_ALLOWLIST = "request_in_allowlist"
CHECK_OUTPUT_ROOT_IS_SCRATCH = "output_root_is_scratch"
CHECK_NO_CREDENTIALS_REQUIRED = "no_credentials_required_for_preflight"

CANONICAL_PREFLIGHT_CHECK_ORDER: tuple[str, ...] = (
    CHECK_CANONICAL_MANIFESTS_PRESENT,
    CHECK_M009_VALIDATION_PASSED,
    CHECK_REQUEST_IN_DOWNLOAD_MANIFEST,
    CHECK_REQUEST_IN_ALLOWLIST,
    CHECK_OUTPUT_ROOT_IS_SCRATCH,
    CHECK_NO_CREDENTIALS_REQUIRED,
)

STEP_STATUS_PLANNED = "planned"
STEP_STATUS_SUCCEEDED = "succeeded"
STEP_STATUS_FAILED = "failed"
STEP_STATUS_SKIPPED = "skipped"

STEP_ACQUIRE_ONE_REQUEST = "acquire_one_request"
STEP_PREPROCESS_ONE_REQUEST = "preprocess_one_request"
STEP_INDICES_ONE_REQUEST = "indices_one_request"
STEP_VALIDATE_PRODUCTS = "validate_products"

CANONICAL_EXECUTE_STEP_ORDER: tuple[str, ...] = (
    STEP_ACQUIRE_ONE_REQUEST,
    STEP_PREPROCESS_ONE_REQUEST,
    STEP_INDICES_ONE_REQUEST,
    STEP_VALIDATE_PRODUCTS,
)

PREFLIGHT_EXECUTION_STATUS_READY = "ready_for_owner_authorized_live_test"
PREFLIGHT_EXECUTION_STATUS_BLOCKED = "blocked"
EXECUTE_EXECUTION_STATUS_COMPLETED = "completed_live_smoke"
EXECUTE_EXECUTION_STATUS_FAILED = "failed"
EXECUTE_EXECUTION_STATUS_PARTIAL = "partial"

CHECK_STATUS_PASSED = "passed"
CHECK_STATUS_FAILED = "failed"

CANONICAL_PREREQUISITE_MANIFESTS: tuple[str, ...] = (
    "region_manifest_path",
    "download_manifest_path",
    "pipeline_manifest_path",
    "validation_report_path",
)


class LiveSmokeError(ValueError):
    """Raised when the live-smoke runner cannot orchestrate the canonical sequence."""


@dataclass(frozen=True)
class CheckRecord:
    check_id: str
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "status": self.status, "message": self.message}


@dataclass(frozen=True)
class StepRecord:
    step_id: str
    description: str
    script: str
    argv: tuple[str, ...]
    status: str
    exit_code: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "step_id": self.step_id,
            "description": self.description,
            "script": self.script,
            "argv": list(self.argv),
            "status": self.status,
            "exit_code": self.exit_code,
        }
        if self.error is not None:
            record["error"] = self.error
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
        raise LiveSmokeError(f"could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise LiveSmokeError(f"{path} is not a JSON object")
    return data


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise LiveSmokeError(f"config not found: {path}")
    if not path.is_file():
        raise LiveSmokeError(f"config path is not a file: {path}")
    data = _read_json(path)
    for required in ("run", "live_smoke"):
        if required not in data:
            raise LiveSmokeError(f"config {path} missing required top-level key {required!r}")
    return data


def _extract_request(download_manifest: dict[str, Any], request_id: str) -> dict[str, Any] | None:
    for request in download_manifest.get("requests", []):
        if isinstance(request, dict) and request.get("request_id") == request_id:
            return request
    return None


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


def _check_canonical_manifests_present(config: dict[str, Any]) -> tuple[CheckRecord, list[dict[str, Any]]]:
    run = config["run"]
    found: list[dict[str, Any]] = []
    missing: list[str] = []
    for key in CANONICAL_PREREQUISITE_MANIFESTS:
        path_str = run.get(key)
        if not path_str:
            missing.append(key)
            continue
        p = Path(path_str)
        if not p.exists():
            missing.append(f"{key}: {path_str}")
            continue
        found.append({
            "name": key.removesuffix("_path"),
            "path": _as_posix(p),
            "hash": compute_file_hash(p),
        })
    if missing:
        return (
            CheckRecord(
                CHECK_CANONICAL_MANIFESTS_PRESENT, CHECK_STATUS_FAILED,
                f"missing canonical manifests: {missing}",
            ),
            found,
        )
    return (
        CheckRecord(
            CHECK_CANONICAL_MANIFESTS_PRESENT, CHECK_STATUS_PASSED,
            f"every canonical manifest is present and hashable ({len(found)} files)",
        ),
        found,
    )


def _check_m009_validation_passed(config: dict[str, Any]) -> CheckRecord:
    """Re-run the M009 validator dry-run against the canonical pipeline manifest.

    The validator is imported lazily so ``lib.live_smoke`` itself does
    not pull ``numpy`` / ``xarray`` into the preflight import graph.
    """
    from . import validation as v  # lazy

    run = config["run"]
    pipeline_path = Path(run.get("pipeline_manifest_path", ""))
    output_root = Path(run.get("output_root", ""))
    if not pipeline_path.exists():
        return CheckRecord(
            CHECK_M009_VALIDATION_PASSED, CHECK_STATUS_FAILED,
            f"pipeline manifest missing for M009 cross-check: {pipeline_path}",
        )
    results = v.run_validation(
        pipeline_manifest_path=pipeline_path,
        output_root=output_root,
        mode=v.MODE_DRY_RUN,
    )
    status = v.derive_execution_status(results)
    if status == v.EXECUTION_STATUS_PASSED:
        return CheckRecord(
            CHECK_M009_VALIDATION_PASSED, CHECK_STATUS_PASSED,
            "M009 validation reports execution_status='passed' against the canonical manifests",
        )
    failed_ids = [r.check_id for r in results if r.status == v.STATUS_FAILED]
    return CheckRecord(
        CHECK_M009_VALIDATION_PASSED, CHECK_STATUS_FAILED,
        f"M009 validation execution_status={status!r}; failed checks: {failed_ids}",
    )


def _check_request_in_download_manifest(
    config: dict[str, Any], request_id: str
) -> tuple[CheckRecord, dict[str, Any] | None]:
    run = config["run"]
    dl_path = Path(run.get("download_manifest_path", ""))
    if not dl_path.exists():
        return (
            CheckRecord(
                CHECK_REQUEST_IN_DOWNLOAD_MANIFEST, CHECK_STATUS_FAILED,
                f"download manifest missing: {dl_path}",
            ),
            None,
        )
    download = _read_json(dl_path)
    request = _extract_request(download, request_id)
    if request is None:
        return (
            CheckRecord(
                CHECK_REQUEST_IN_DOWNLOAD_MANIFEST, CHECK_STATUS_FAILED,
                f"request_id {request_id!r} not found in {dl_path}",
            ),
            None,
        )
    return (
        CheckRecord(
            CHECK_REQUEST_IN_DOWNLOAD_MANIFEST, CHECK_STATUS_PASSED,
            f"request_id {request_id!r} found in download manifest",
        ),
        request,
    )


def _check_request_in_allowlist(request_id: str) -> CheckRecord:
    if request_id in ALLOWED_REQUEST_IDS:
        return CheckRecord(
            CHECK_REQUEST_IN_ALLOWLIST, CHECK_STATUS_PASSED,
            f"request_id {request_id!r} is in the M010 smoke allowlist",
        )
    return CheckRecord(
        CHECK_REQUEST_IN_ALLOWLIST, CHECK_STATUS_FAILED,
        f"request_id {request_id!r} is not in the smoke allowlist {sorted(ALLOWED_REQUEST_IDS)}",
    )


def _resolves_to_canonical_dev_root(output_root: Path) -> bool:
    """Return True iff ``output_root`` points at the canonical dev root.

    Catches both the relative form (``Path("runs/dev_region")``) and any
    absolute form (``Path("runs/dev_region").resolve()`` or a different
    cwd-anchored absolute path). The comparison is on
    ``Path.resolve()``, which is filesystem-aware but does not require
    the directory to exist.
    """
    try:
        resolved = output_root.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    try:
        canonical_resolved = Path(CANONICAL_DEV_OUTPUT_ROOT).resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return resolved == canonical_resolved


def _check_output_root_is_scratch(output_root: Path) -> CheckRecord:
    posix = _as_posix(output_root).rstrip("/")
    canonical = CANONICAL_DEV_OUTPUT_ROOT.rstrip("/")
    if not posix:
        return CheckRecord(
            CHECK_OUTPUT_ROOT_IS_SCRATCH, CHECK_STATUS_FAILED,
            "output_root must not be empty",
        )
    if posix == canonical or _resolves_to_canonical_dev_root(output_root):
        return CheckRecord(
            CHECK_OUTPUT_ROOT_IS_SCRATCH, CHECK_STATUS_FAILED,
            f"output_root {posix!r} resolves to the canonical dev root; choose a scratch path under runs/live_smoke*/",
        )
    return CheckRecord(
        CHECK_OUTPUT_ROOT_IS_SCRATCH, CHECK_STATUS_PASSED,
        f"output_root {posix!r} is a scratch directory (not the canonical dev root)",
    )


def _check_no_credentials_required() -> CheckRecord:
    """Structural check: preflight must not need CDS credentials.

    No filesystem lookup of ``~/.cdsapirc`` and no environment-variable
    read. The check is documentation + a discipline pin: if the
    preflight ever needs credentials, this check should be updated to
    reflect that and the M010 boundary contract should change.
    """
    return CheckRecord(
        CHECK_NO_CREDENTIALS_REQUIRED, CHECK_STATUS_PASSED,
        "preflight does not import cdsapi, does not read ~/.cdsapirc, and does not inspect CDSAPI_* env vars",
    )


def run_preflight_checks(
    config: dict[str, Any],
    *,
    request_id: str,
    output_root: Path,
) -> tuple[list[CheckRecord], list[dict[str, Any]], dict[str, Any] | None]:
    """Run the six preflight checks and return (records, prerequisite_manifests, request_record)."""
    checks: list[CheckRecord] = []
    cm_check, prerequisites = _check_canonical_manifests_present(config)
    checks.append(cm_check)
    if cm_check.status == CHECK_STATUS_PASSED:
        checks.append(_check_m009_validation_passed(config))
    else:
        checks.append(CheckRecord(
            CHECK_M009_VALIDATION_PASSED, CHECK_STATUS_FAILED,
            "skipped: canonical manifests missing",
        ))
    request_record = None
    if cm_check.status == CHECK_STATUS_PASSED:
        rid_check, request_record = _check_request_in_download_manifest(config, request_id)
        checks.append(rid_check)
    else:
        checks.append(CheckRecord(
            CHECK_REQUEST_IN_DOWNLOAD_MANIFEST, CHECK_STATUS_FAILED,
            "skipped: canonical manifests missing",
        ))
    checks.append(_check_request_in_allowlist(request_id))
    checks.append(_check_output_root_is_scratch(output_root))
    checks.append(_check_no_credentials_required())
    return checks, prerequisites, request_record


# ---------------------------------------------------------------------------
# Step argv builders
# ---------------------------------------------------------------------------


def _argv_acquire(config: dict[str, Any], request_id: str, output_root: Path) -> list[str]:
    run = config["run"]
    return [
        "--download-manifest", run["download_manifest_path"],
        "--output", _as_posix(output_root / "acquisition_manifest.json"),
        "--output-root", _as_posix(output_root),
        "--mode", "execute",
        "--request-id", request_id,
    ]


def _argv_preprocess(config: dict[str, Any], output_root: Path) -> list[str]:
    run = config["run"]
    return [
        "--acquisition-manifest", _as_posix(output_root / "acquisition_manifest.json"),
        "--download-manifest", run["download_manifest_path"],
        "--region-manifest", run["region_manifest_path"],
        "--output", _as_posix(output_root / "preprocessing_manifest.json"),
        "--output-root", _as_posix(output_root),
        "--mode", "execute",
    ]


def _argv_indices(output_root: Path) -> list[str]:
    argv = [
        "--preprocessing-manifest", _as_posix(output_root / "preprocessing_manifest.json"),
        "--output", _as_posix(output_root / "index_manifest.json"),
        "--output-root", _as_posix(output_root),
        "--mode", "execute",
    ]
    for index_id in ALLOWED_INDEX_IDS:
        argv.extend(["--index-id", index_id])
    return argv


def _build_planned_outputs(request: dict[str, Any], output_root: Path) -> dict[str, str]:
    project_vars = request.get("project_variables") or ["tmax"]
    var = project_vars[0]
    year = request.get("year", 2000)
    return {
        "raw_target_path": _as_posix(output_root / request["output_path"]),
        f"{var}_daily_product": _as_posix(output_root / "intermediate" / "daily" / var / f"{year}.nc"),
        "TXx_index_product": _as_posix(output_root / "derived" / "indices" / "TXx.nc"),
        "SU_index_product": _as_posix(output_root / "derived" / "indices" / "SU.nc"),
    }


def _build_planned_steps(
    config: dict[str, Any], request_id: str, output_root: Path
) -> list[StepRecord]:
    return [
        StepRecord(
            step_id=STEP_ACQUIRE_ONE_REQUEST,
            description="Download exactly one ERA5-Land daily-statistics request via cdsapi",
            script="02_download_era5_land.py",
            argv=tuple(_argv_acquire(config, request_id, output_root)),
            status=STEP_STATUS_PLANNED,
            exit_code=-1,
        ),
        StepRecord(
            step_id=STEP_PREPROCESS_ONE_REQUEST,
            description="Preprocess the acquired daily-statistics NetCDF into a daily standard product",
            script="03_preprocess_daily.py",
            argv=tuple(_argv_preprocess(config, output_root)),
            status=STEP_STATUS_PLANNED,
            exit_code=-1,
        ),
        StepRecord(
            step_id=STEP_INDICES_ONE_REQUEST,
            description=f"Compute the two tmax-only indices: {list(ALLOWED_INDEX_IDS)}",
            script="04_compute_indices.py",
            argv=tuple(_argv_indices(output_root)),
            status=STEP_STATUS_PLANNED,
            exit_code=-1,
        ),
        StepRecord(
            step_id=STEP_VALIDATE_PRODUCTS,
            description="Validate the produced tmax / TXx / SU NetCDF schemas via M009 helpers",
            script="(in-process lib.validation)",
            argv=tuple(),
            status=STEP_STATUS_PLANNED,
            exit_code=-1,
        ),
    ]


# ---------------------------------------------------------------------------
# Preflight orchestration
# ---------------------------------------------------------------------------


def run_preflight(
    config: dict[str, Any],
    *,
    request_id: str,
    output_root: Path,
) -> dict[str, Any]:
    """Run preflight checks, build the deterministic plan dict."""
    checks, prerequisites, request_record = run_preflight_checks(
        config, request_id=request_id, output_root=output_root,
    )
    overall_passed = all(c.status == CHECK_STATUS_PASSED for c in checks)
    execution_status = (
        PREFLIGHT_EXECUTION_STATUS_READY if overall_passed else PREFLIGHT_EXECUTION_STATUS_BLOCKED
    )
    steps = _build_planned_steps(config, request_id, output_root)
    planned_outputs: dict[str, str] = {}
    if request_record is not None:
        planned_outputs = _build_planned_outputs(request_record, output_root)
    plan: dict[str, Any] = {
        "manifest_type": MANIFEST_TYPE_LIVE_SMOKE,
        "mode": MODE_PREFLIGHT,
        "requires_network": False,
        "run_id": Path(output_root.name).name or "live_smoke",
        "output_root": _as_posix(output_root),
        "request_id": request_id,
        "request": request_record if request_record is not None else {},
        "planned_outputs": planned_outputs,
        "prerequisite_manifests": prerequisites,
        "preflight_checks": [c.to_dict() for c in checks],
        "steps": [s.to_dict() for s in steps],
        "created_by": "scripts/07_run_live_smoke.py",
        "execution_status": execution_status,
        "confirmation_token_required": CONFIRMATION_TOKEN,
    }
    return plan


# ---------------------------------------------------------------------------
# Execute orchestration (fake-runner friendly)
# ---------------------------------------------------------------------------


def _default_step_runner(
    step: StepRecord, *, repo_root: Path,
) -> tuple[int, str | None]:
    """Real-script runner used in execute mode. Tests inject a fake."""
    if step.script == "(in-process lib.validation)":
        # Special case: validation step is in-process; no script to load.
        # Returning (0, None) so the caller can run validators separately.
        return 0, None
    import importlib.util

    script_path = repo_root / "scripts" / step.script
    if not script_path.exists():
        return 2, f"script not found: {script_path}"
    spec = importlib.util.spec_from_file_location(
        f"live_smoke__script__{step.script.replace('.', '_')}",
        script_path,
    )
    if spec is None or spec.loader is None:
        return 2, f"could not load script: {script_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        return 2, f"script has no callable main(): {script_path}"
    try:
        exit_code = int(main_fn(list(step.argv)))
    except SystemExit as exc:
        exit_code = int(exc.code) if exc.code is not None else 0
    except Exception as exc:  # noqa: BLE001 - capture into the manifest
        return 2, f"{type(exc).__name__}: {exc}"
    return exit_code, None


def _validate_smoke_products(
    output_root: Path, request: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Run the M009 per-file validators on the produced products."""
    from . import validation as v  # lazy

    project_vars = request.get("project_variables") or ["tmax"]
    var = project_vars[0]
    year = int(request.get("year", 2000))
    daily_path = output_root / "intermediate" / "daily" / var / f"{year}.nc"
    txx_path = output_root / "derived" / "indices" / "TXx.nc"
    su_path = output_root / "derived" / "indices" / "SU.nc"
    validations: list[dict[str, Any]] = []
    all_passed = True

    daily_result = v.validate_daily_product(
        daily_path,
        project_variable=var,
        expected_year=year,
        expected_units=v.TEMPERATURE_DAILY_UNITS if var in v.TEMPERATURE_DAILY_VARIABLES else None,
    )
    validations.append(daily_result.to_dict())
    if daily_result.status != v.STATUS_PASSED:
        all_passed = False
    for index_path, index_id, expected_units in (
        (txx_path, "TXx", "degC"),
        (su_path, "SU", "days"),
    ):
        result = v.validate_index_product(
            index_path, index_id=index_id, expected_units=expected_units,
        )
        validations.append(result.to_dict())
        if result.status != v.STATUS_PASSED:
            all_passed = False
    return validations, all_passed


def run_execute(
    config: dict[str, Any],
    *,
    request_id: str,
    output_root: Path,
    repo_root: Path,
    confirm_live: str | None = None,
    step_runner: Callable[..., tuple[int, str | None]] | None = None,
    skip_product_validation: bool = False,
) -> dict[str, Any]:
    """Run the four-step canonical sequence; tests inject ``step_runner``.

    ``confirm_live`` must equal :data:`CONFIRMATION_TOKEN` to proceed.
    The check fires before any step runner is selected so a direct
    library caller cannot reach :func:`_default_step_runner` (and
    therefore cannot reach live CDS) without proving operator intent.

    ``skip_product_validation=True`` lets tests with fake runners avoid
    needing real NetCDF on disk; the canonical execute path always
    validates.
    """
    if confirm_live != CONFIRMATION_TOKEN:
        raise LiveSmokeError(
            "execute mode refused: confirm_live must equal "
            f"{CONFIRMATION_TOKEN!r} to authorize a live CDS call"
        )
    if _resolves_to_canonical_dev_root(output_root):
        raise LiveSmokeError(
            f"execute mode refused: output_root {_as_posix(output_root)!r} "
            f"resolves to the canonical dev root {CANONICAL_DEV_OUTPUT_ROOT!r}; "
            "choose a scratch path under runs/live_smoke*/"
        )
    runner = step_runner or _default_step_runner

    # Pre-execute checks: the same preflight set without the
    # "no credentials" structural pin (credentials *are* allowed here).
    checks, prerequisites, request_record = run_preflight_checks(
        config, request_id=request_id, output_root=output_root,
    )
    blocking_failures = [c for c in checks if c.status == CHECK_STATUS_FAILED
                        and c.check_id != CHECK_NO_CREDENTIALS_REQUIRED]
    planned_steps = _build_planned_steps(config, request_id, output_root)
    if blocking_failures or request_record is None:
        skipped = tuple(
            StepRecord(
                step_id=s.step_id, description=s.description, script=s.script,
                argv=s.argv, status=STEP_STATUS_SKIPPED, exit_code=-1,
                error=f"blocked by preflight check {blocking_failures[0].check_id!r}"
                      if blocking_failures else "no request record",
            )
            for s in planned_steps
        )
        return _assemble_execute_plan(
            config=config, request_id=request_id, request=request_record,
            output_root=output_root, checks=checks, prerequisites=prerequisites,
            steps=skipped, validations=[],
            execution_status=EXECUTE_EXECUTION_STATUS_FAILED,
        )

    results: list[StepRecord] = []
    aborted = False
    for step in planned_steps[:3]:  # acquire / preprocess / indices
        if aborted:
            results.append(StepRecord(
                step_id=step.step_id, description=step.description, script=step.script,
                argv=step.argv, status=STEP_STATUS_SKIPPED, exit_code=-1,
            ))
            continue
        rc, error = runner(step, repo_root=repo_root)
        status = STEP_STATUS_SUCCEEDED if rc == 0 else STEP_STATUS_FAILED
        results.append(StepRecord(
            step_id=step.step_id, description=step.description, script=step.script,
            argv=step.argv, status=status, exit_code=rc, error=error,
        ))
        if status != STEP_STATUS_SUCCEEDED:
            aborted = True

    validations: list[dict[str, Any]] = []
    validate_step = planned_steps[3]
    if aborted:
        results.append(StepRecord(
            step_id=validate_step.step_id, description=validate_step.description,
            script=validate_step.script, argv=validate_step.argv,
            status=STEP_STATUS_SKIPPED, exit_code=-1,
        ))
    else:
        if skip_product_validation:
            validate_rc = 0
            validations = [{
                "check_id": "validation_skipped",
                "status": "skipped",
                "severity": "info",
                "message": "product validation skipped by caller (test mode)",
            }]
        else:
            try:
                validations, val_passed = _validate_smoke_products(output_root, request_record)
                validate_rc = 0 if val_passed else 2
            except Exception as exc:  # noqa: BLE001
                validate_rc = 2
                validations = [{
                    "check_id": "validation_error",
                    "status": "failed",
                    "severity": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }]
        status = STEP_STATUS_SUCCEEDED if validate_rc == 0 else STEP_STATUS_FAILED
        results.append(StepRecord(
            step_id=validate_step.step_id, description=validate_step.description,
            script=validate_step.script, argv=validate_step.argv,
            status=status, exit_code=validate_rc,
        ))

    statuses = {r.status for r in results}
    if statuses == {STEP_STATUS_SUCCEEDED}:
        execution_status = EXECUTE_EXECUTION_STATUS_COMPLETED
    elif STEP_STATUS_SUCCEEDED in statuses and (STEP_STATUS_FAILED in statuses or STEP_STATUS_SKIPPED in statuses):
        execution_status = EXECUTE_EXECUTION_STATUS_PARTIAL
    else:
        execution_status = EXECUTE_EXECUTION_STATUS_FAILED

    return _assemble_execute_plan(
        config=config, request_id=request_id, request=request_record,
        output_root=output_root, checks=checks, prerequisites=prerequisites,
        steps=tuple(results), validations=validations,
        execution_status=execution_status,
    )


def _assemble_execute_plan(
    *,
    config: dict[str, Any],
    request_id: str,
    request: dict[str, Any] | None,
    output_root: Path,
    checks: list[CheckRecord],
    prerequisites: list[dict[str, Any]],
    steps: tuple[StepRecord, ...],
    validations: list[dict[str, Any]],
    execution_status: str,
) -> dict[str, Any]:
    planned_outputs = (
        _build_planned_outputs(request, output_root) if request is not None else {}
    )
    return {
        "manifest_type": MANIFEST_TYPE_LIVE_SMOKE,
        "mode": MODE_EXECUTE,
        "requires_network": True,
        "run_id": Path(output_root.name).name or "live_smoke",
        "output_root": _as_posix(output_root),
        "request_id": request_id,
        "request": request if request is not None else {},
        "planned_outputs": planned_outputs,
        "prerequisite_manifests": prerequisites,
        "preflight_checks": [c.to_dict() for c in checks],
        "steps": [s.to_dict() for s in steps],
        "product_validations": validations,
        "created_by": "scripts/07_run_live_smoke.py",
        "execution_status": execution_status,
        "confirmation_token_required": CONFIRMATION_TOKEN,
    }


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------


def write_plan(output_path: Path, plan: dict[str, Any]) -> None:
    """Write the live-smoke plan as deterministic JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")


__all__ = [
    "ALLOWED_INDEX_IDS",
    "ALLOWED_REQUEST_IDS",
    "CANONICAL_DEV_OUTPUT_ROOT",
    "CANONICAL_EXECUTE_STEP_ORDER",
    "CANONICAL_PREFLIGHT_CHECK_ORDER",
    "CANONICAL_PREREQUISITE_MANIFESTS",
    "CHECK_CANONICAL_MANIFESTS_PRESENT",
    "CHECK_M009_VALIDATION_PASSED",
    "CHECK_NO_CREDENTIALS_REQUIRED",
    "CHECK_OUTPUT_ROOT_IS_SCRATCH",
    "CHECK_REQUEST_IN_ALLOWLIST",
    "CHECK_REQUEST_IN_DOWNLOAD_MANIFEST",
    "CHECK_STATUS_FAILED",
    "CHECK_STATUS_PASSED",
    "CONFIRMATION_TOKEN",
    "CheckRecord",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_REQUEST_ID",
    "EXECUTE_EXECUTION_STATUS_COMPLETED",
    "EXECUTE_EXECUTION_STATUS_FAILED",
    "EXECUTE_EXECUTION_STATUS_PARTIAL",
    "LiveSmokeError",
    "MANIFEST_TYPE_LIVE_SMOKE",
    "MODE_EXECUTE",
    "MODE_PREFLIGHT",
    "PREFLIGHT_EXECUTION_STATUS_BLOCKED",
    "PREFLIGHT_EXECUTION_STATUS_READY",
    "STEP_ACQUIRE_ONE_REQUEST",
    "STEP_INDICES_ONE_REQUEST",
    "STEP_PREPROCESS_ONE_REQUEST",
    "STEP_STATUS_FAILED",
    "STEP_STATUS_PLANNED",
    "STEP_STATUS_SKIPPED",
    "STEP_STATUS_SUCCEEDED",
    "STEP_VALIDATE_PRODUCTS",
    "SUPPORTED_MODES",
    "StepRecord",
    "_resolves_to_canonical_dev_root",
    "compute_file_hash",
    "load_config",
    "run_execute",
    "run_preflight",
    "run_preflight_checks",
    "write_plan",
]
