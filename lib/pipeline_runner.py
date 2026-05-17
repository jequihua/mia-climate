"""Local dry-run pipeline orchestration for milestone 008.

This module composes the M001-M007 script boundaries into one
auditable canonical dry-run sequence and writes a deterministic
``pipeline_manifest.json`` summarizing each step's command argv,
exit code, output path, and output SHA-256 hash.

The runner does **not** perform live CDS downloads, does **not**
open NetCDF in dry-run, and does **not** add new science. It only
re-invokes existing scripts via their ``main(argv)`` functions and
records the result.

``cdsapi`` and other heavy scientific dependencies are imported
lazily by the downstream scripts on the execute paths; this module
does not import them at module scope, and dry-run never triggers
the lazy load.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MANIFEST_TYPE_PIPELINE = "era5_land_pipeline_run"

MODE_DRY_RUN = "dry-run"
SUPPORTED_MODES = frozenset({MODE_DRY_RUN})

STEP_STATUS_SUCCEEDED = "succeeded"
STEP_STATUS_FAILED = "failed"
STEP_STATUS_SKIPPED = "skipped"

EXECUTION_STATUS_COMPLETED_DRY_RUN = "completed_dry_run"
EXECUTION_STATUS_FAILED = "failed"

# Step identifiers. Eight canonical steps for the M001-M007 dry-run sequence.
STEP_VALIDATE_REGION = "validate_region"
STEP_PLAN_DOWNLOADS = "plan_downloads"
STEP_ACQUIRE_DAILY_STATS = "acquire_daily_stats_dry_run"
STEP_ACQUIRE_PRECIPITATION = "acquire_precipitation_dry_run"
STEP_PREPROCESS_DAILY_STATS = "preprocess_daily_stats_dry_run"
STEP_PREPROCESS_PRECIPITATION = "preprocess_precipitation_dry_run"
STEP_INDICES_TEMPERATURE = "indices_temperature_dry_run"
STEP_INDICES_PRECIPITATION = "indices_precipitation_dry_run"

CANONICAL_STEP_ORDER: tuple[str, ...] = (
    STEP_VALIDATE_REGION,
    STEP_PLAN_DOWNLOADS,
    STEP_ACQUIRE_DAILY_STATS,
    STEP_ACQUIRE_PRECIPITATION,
    STEP_PREPROCESS_DAILY_STATS,
    STEP_PREPROCESS_PRECIPITATION,
    STEP_INDICES_TEMPERATURE,
    STEP_INDICES_PRECIPITATION,
)


class PipelineRunnerError(ValueError):
    """Raised when the pipeline runner cannot orchestrate the canonical sequence."""


@dataclass(frozen=True)
class StepSpec:
    step_id: str
    description: str
    script_filename: str
    build_argv: Callable[[dict[str, Any]], list[str]]
    output_path_key: Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class StepResult:
    step_id: str
    script: str
    argv: tuple[str, ...]
    output_path: str
    output_hash: str | None
    exit_code: int
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "step_id": self.step_id,
            "script": self.script,
            "argv": list(self.argv),
            "output_path": self.output_path,
            "exit_code": self.exit_code,
            "status": self.status,
        }
        if self.output_hash is not None:
            record["output_hash"] = self.output_hash
        if self.error is not None:
            record["error"] = self.error
        return record


# ---------------------------------------------------------------------------
# Step argv builders
# ---------------------------------------------------------------------------


def _argv_validate_region(config: dict[str, Any]) -> list[str]:
    region = config["region"]
    run = config["run"]
    return [
        "--region-id", region["region_id"],
        "--geometry", region["geometry_path"],
        "--output", run["region_manifest_path"],
    ]


def _argv_plan_downloads(config: dict[str, Any]) -> list[str]:
    run = config["run"]
    plan = config["download_plan"]
    return [
        "--region-manifest", run["region_manifest_path"],
        "--output", run["download_manifest_path"],
        "--start-year", str(plan["start_year"]),
        "--end-year", str(plan["end_year"]),
    ]


def _argv_acquire_daily_stats(config: dict[str, Any]) -> list[str]:
    run = config["run"]
    pipeline = config["pipeline"]
    return [
        "--download-manifest", run["download_manifest_path"],
        "--output", run["acquisition_manifest_path"],
        "--output-root", run["output_root"],
        "--mode", MODE_DRY_RUN,
        "--limit", str(pipeline["acquisition_daily_stats_limit"]),
    ]


def _argv_acquire_precipitation(config: dict[str, Any]) -> list[str]:
    run = config["run"]
    pipeline = config["pipeline"]
    argv = [
        "--download-manifest", run["download_manifest_path"],
        "--output", run["acquisition_manifest_precipitation_dry_run_path"],
        "--output-root", run["output_root"],
        "--mode", MODE_DRY_RUN,
    ]
    for request_id in pipeline["precipitation_acquisition_request_ids"]:
        argv.extend(["--request-id", request_id])
    return argv


def _argv_preprocess_daily_stats(config: dict[str, Any]) -> list[str]:
    run = config["run"]
    return [
        "--acquisition-manifest", run["acquisition_manifest_path"],
        "--download-manifest", run["download_manifest_path"],
        "--region-manifest", run["region_manifest_path"],
        "--output", run["preprocessing_manifest_path"],
        "--output-root", run["output_root"],
        "--mode", MODE_DRY_RUN,
    ]


def _argv_preprocess_precipitation(config: dict[str, Any]) -> list[str]:
    run = config["run"]
    pipeline = config["pipeline"]
    return [
        "--acquisition-manifest", run["acquisition_manifest_precipitation_dry_run_path"],
        "--download-manifest", run["download_manifest_path"],
        "--region-manifest", run["region_manifest_path"],
        "--output", run["preprocessing_manifest_precipitation_path"],
        "--output-root", run["output_root"],
        "--mode", MODE_DRY_RUN,
        "--precipitation-policy", pipeline["precipitation_policy"],
    ]


def _argv_indices_temperature(config: dict[str, Any]) -> list[str]:
    run = config["run"]
    return [
        "--preprocessing-manifest", run["preprocessing_manifest_path"],
        "--output", run["index_manifest_path"],
        "--output-root", run["output_root"],
        "--mode", MODE_DRY_RUN,
    ]


def _argv_indices_precipitation(config: dict[str, Any]) -> list[str]:
    run = config["run"]
    return [
        "--preprocessing-manifest", run["preprocessing_manifest_precipitation_path"],
        "--output", run["index_manifest_precipitation_path"],
        "--output-root", run["output_root"],
        "--mode", MODE_DRY_RUN,
        "--index-family", "precipitation",
    ]


# ---------------------------------------------------------------------------
# Step spec table
# ---------------------------------------------------------------------------


STEP_SPECS: tuple[StepSpec, ...] = (
    StepSpec(
        step_id=STEP_VALIDATE_REGION,
        description="Validate the canonical region polygon and write the region manifest.",
        script_filename="00_validate_region.py",
        build_argv=_argv_validate_region,
        output_path_key=lambda c: c["run"]["region_manifest_path"],
    ),
    StepSpec(
        step_id=STEP_PLAN_DOWNLOADS,
        description="Plan the ERA5-Land download requests for the configured year range.",
        script_filename="01_plan_downloads.py",
        build_argv=_argv_plan_downloads,
        output_path_key=lambda c: c["run"]["download_manifest_path"],
    ),
    StepSpec(
        step_id=STEP_ACQUIRE_DAILY_STATS,
        description="Dry-run daily-statistics acquisition (first N requests).",
        script_filename="02_download_era5_land.py",
        build_argv=_argv_acquire_daily_stats,
        output_path_key=lambda c: c["run"]["acquisition_manifest_path"],
    ),
    StepSpec(
        step_id=STEP_ACQUIRE_PRECIPITATION,
        description="Dry-run precipitation acquisition for the configured H1+H2 chunks.",
        script_filename="02_download_era5_land.py",
        build_argv=_argv_acquire_precipitation,
        output_path_key=lambda c: c["run"]["acquisition_manifest_precipitation_dry_run_path"],
    ),
    StepSpec(
        step_id=STEP_PREPROCESS_DAILY_STATS,
        description="Dry-run daily-statistics preprocessing.",
        script_filename="03_preprocess_daily.py",
        build_argv=_argv_preprocess_daily_stats,
        output_path_key=lambda c: c["run"]["preprocessing_manifest_path"],
    ),
    StepSpec(
        step_id=STEP_PREPROCESS_PRECIPITATION,
        description="Dry-run precipitation preprocessing under the configured policy.",
        script_filename="03_preprocess_daily.py",
        build_argv=_argv_preprocess_precipitation,
        output_path_key=lambda c: c["run"]["preprocessing_manifest_precipitation_path"],
    ),
    StepSpec(
        step_id=STEP_INDICES_TEMPERATURE,
        description="Dry-run temperature climate indices.",
        script_filename="04_compute_indices.py",
        build_argv=_argv_indices_temperature,
        output_path_key=lambda c: c["run"]["index_manifest_path"],
    ),
    StepSpec(
        step_id=STEP_INDICES_PRECIPITATION,
        description="Dry-run precipitation climate indices.",
        script_filename="04_compute_indices.py",
        build_argv=_argv_indices_precipitation,
        output_path_key=lambda c: c["run"]["index_manifest_precipitation_path"],
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_file_hash(path: Path) -> str:
    """Return a stable ``sha256:<hex>`` over the bytes of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_config(path: Path) -> dict[str, Any]:
    """Read and minimally validate a pipeline config (e.g. configs/rbmn_local.json)."""
    if not path.exists():
        raise PipelineRunnerError(f"config not found: {path}")
    if not path.is_file():
        raise PipelineRunnerError(f"config path is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineRunnerError(f"config is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PipelineRunnerError(
            f"config must be a JSON object, got {type(data).__name__}: {path}"
        )
    for required_top in ("region", "run", "download_plan", "pipeline"):
        if required_top not in data:
            raise PipelineRunnerError(
                f"config {path} is missing required top-level key: {required_top!r}"
            )
    return data


def _as_posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _load_script_main(script_filename: str, *, repo_root: Path) -> Callable[[list[str] | None], int]:
    """Load a digit-prefixed script via ``importlib.util`` and return its ``main``.

    The numbered script filenames (e.g. ``02_download_era5_land.py``) are
    not importable as a normal module, so the tests and now this runner
    use ``importlib.util`` to load them on demand. Loading is deferred
    until step execution so ``import lib.pipeline_runner`` stays light.
    """
    import importlib.util

    script_path = repo_root / "scripts" / script_filename
    if not script_path.exists():
        raise PipelineRunnerError(f"script not found: {script_path}")
    module_name = f"pipeline_runner__script__{script_filename.replace('.', '_').replace('/', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise PipelineRunnerError(f"could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        raise PipelineRunnerError(f"script has no callable main(): {script_path}")
    return main_fn  # type: ignore[return-value]


def default_step_runner(
    spec: StepSpec, argv: list[str], *, repo_root: Path
) -> tuple[int, str | None]:
    """Default real-script runner: load the script via importlib and call main(argv).

    Returns ``(exit_code, error_message_or_None)``. Exceptions are
    captured as exit code 2 plus an error message so the manifest
    records the failure rather than crashing the runner.
    """
    try:
        main_fn = _load_script_main(spec.script_filename, repo_root=repo_root)
        exit_code = int(main_fn(argv))
    except SystemExit as exc:
        # argparse / scripts that call ``raise SystemExit(rc)``.
        exit_code = int(exc.code) if exc.code is not None else 0
        return exit_code, None
    except Exception as exc:  # noqa: BLE001 - capture into manifest, do not abort the runner
        return 2, f"{type(exc).__name__}: {exc}"
    return exit_code, None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    config: dict[str, Any],
    *,
    repo_root: Path,
    mode: str = MODE_DRY_RUN,
    step_runner: Callable[..., tuple[int, str | None]] | None = None,
) -> list[StepResult]:
    """Run the canonical eight-step dry-run sequence in order.

    Stops on the first failed step; subsequent steps are recorded as
    ``status = "skipped"`` (with exit_code -1 and no output_hash).

    ``step_runner`` defaults to ``default_step_runner``; tests inject a
    fake to avoid invoking the real scripts.
    """
    if mode not in SUPPORTED_MODES:
        raise PipelineRunnerError(
            f"mode {mode!r} is not one of {sorted(SUPPORTED_MODES)}; "
            "M008 only supports dry-run"
        )
    runner = step_runner or default_step_runner
    results: list[StepResult] = []
    aborted = False
    for spec in STEP_SPECS:
        argv = list(spec.build_argv(config))
        output_path_str = _as_posix(spec.output_path_key(config))
        if aborted:
            results.append(
                StepResult(
                    step_id=spec.step_id,
                    script=spec.script_filename,
                    argv=tuple(argv),
                    output_path=output_path_str,
                    output_hash=None,
                    exit_code=-1,
                    status=STEP_STATUS_SKIPPED,
                )
            )
            continue
        exit_code, error = runner(spec, argv, repo_root=repo_root)
        output_hash: str | None = None
        if exit_code == 0:
            output_file = Path(output_path_str)
            if output_file.exists():
                output_hash = compute_file_hash(output_file)
            else:
                # Step claimed success but did not produce its declared output.
                exit_code = 2
                error = f"declared output missing: {output_path_str}"
        status = STEP_STATUS_SUCCEEDED if exit_code == 0 else STEP_STATUS_FAILED
        results.append(
            StepResult(
                step_id=spec.step_id,
                script=spec.script_filename,
                argv=tuple(argv),
                output_path=output_path_str,
                output_hash=output_hash,
                exit_code=exit_code,
                status=status,
                error=error,
            )
        )
        if status != STEP_STATUS_SUCCEEDED:
            aborted = True
    return results


def derive_execution_status(results: list[StepResult]) -> str:
    """Top-level pipeline status: completed only if every step succeeded."""
    if results and all(r.status == STEP_STATUS_SUCCEEDED for r in results):
        return EXECUTION_STATUS_COMPLETED_DRY_RUN
    return EXECUTION_STATUS_FAILED


def build_pipeline_manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_hash: str,
    mode: str,
    results: list[StepResult],
    created_by: str,
) -> dict[str, Any]:
    """Assemble the deterministic pipeline manifest dict."""
    if mode not in SUPPORTED_MODES:
        raise PipelineRunnerError(f"mode {mode!r} is not supported")
    succeeded = sum(1 for r in results if r.status == STEP_STATUS_SUCCEEDED)
    failed = sum(1 for r in results if r.status == STEP_STATUS_FAILED)
    skipped = sum(1 for r in results if r.status == STEP_STATUS_SKIPPED)
    run = config["run"]
    failed_step: str | None = None
    for r in results:
        if r.status == STEP_STATUS_FAILED:
            failed_step = r.step_id
            break
    manifest: dict[str, Any] = {
        "manifest_type": MANIFEST_TYPE_PIPELINE,
        "config_path": _as_posix(config_path),
        "config_hash": config_hash,
        "mode": mode,
        "run_id": run["run_id"],
        "output_root": run["output_root"],
        "step_count": len(results),
        "succeeded_count": succeeded,
        "failed_count": failed,
        "skipped_count": skipped,
        "steps": [r.to_dict() for r in results],
        "created_by": created_by,
        "requires_network": False,
        "execution_status": derive_execution_status(results),
    }
    if failed_step is not None:
        manifest["failed_step"] = failed_step
    return manifest


def write_pipeline_manifest(output_path: Path, manifest: dict[str, Any]) -> None:
    """Write the pipeline manifest as deterministic JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")


__all__ = [
    "CANONICAL_STEP_ORDER",
    "EXECUTION_STATUS_COMPLETED_DRY_RUN",
    "EXECUTION_STATUS_FAILED",
    "MANIFEST_TYPE_PIPELINE",
    "MODE_DRY_RUN",
    "PipelineRunnerError",
    "STEP_ACQUIRE_DAILY_STATS",
    "STEP_ACQUIRE_PRECIPITATION",
    "STEP_INDICES_PRECIPITATION",
    "STEP_INDICES_TEMPERATURE",
    "STEP_PLAN_DOWNLOADS",
    "STEP_PREPROCESS_DAILY_STATS",
    "STEP_PREPROCESS_PRECIPITATION",
    "STEP_SPECS",
    "STEP_STATUS_FAILED",
    "STEP_STATUS_SKIPPED",
    "STEP_STATUS_SUCCEEDED",
    "STEP_VALIDATE_REGION",
    "SUPPORTED_MODES",
    "StepResult",
    "StepSpec",
    "build_pipeline_manifest",
    "compute_file_hash",
    "default_step_runner",
    "derive_execution_status",
    "load_config",
    "run_pipeline",
    "write_pipeline_manifest",
]
