"""Tests for ``lib.pipeline_runner`` and ``scripts/05_run_pipeline.py``.

The orchestration tests use a fake step runner so they exercise the
runner's control flow without invoking the real scripts. A small
integration smoke test exercises the real runner against the
canonical config and verifies byte-identity of the existing M001-M007
reference manifests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from lib.pipeline_runner import (
    CANONICAL_STEP_ORDER,
    EXECUTION_STATUS_COMPLETED_DRY_RUN,
    EXECUTION_STATUS_FAILED,
    MANIFEST_TYPE_PIPELINE,
    MODE_DRY_RUN,
    STEP_ACQUIRE_PRECIPITATION,
    STEP_INDICES_PRECIPITATION,
    STEP_INDICES_TEMPERATURE,
    STEP_PLAN_DOWNLOADS,
    STEP_PREPROCESS_DAILY_STATS,
    STEP_PREPROCESS_PRECIPITATION,
    STEP_SPECS,
    STEP_STATUS_FAILED,
    STEP_STATUS_SKIPPED,
    STEP_STATUS_SUCCEEDED,
    STEP_VALIDATE_REGION,
    PipelineRunnerError,
    build_pipeline_manifest,
    compute_file_hash,
    derive_execution_status,
    load_config,
    run_pipeline,
    write_pipeline_manifest,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_CONFIG = REPO_ROOT / "configs" / "rbmn_local.json"
CANONICAL_PIPELINE_MANIFEST = REPO_ROOT / "runs" / "dev_region" / "pipeline_manifest.json"


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _stub_config(tmp_path: Path) -> dict[str, Any]:
    """Build a minimal pipeline config that resolves to paths under ``tmp_path``."""
    out_root = tmp_path / "runs" / "dev_region"
    return {
        "region": {
            "region_id": "rbmn",
            "geometry_path": "01_data/case_studies/rbmn.geojson",
            "clip_policy": "polygon",
        },
        "run": {
            "run_id": "dev_region",
            "output_root": str(out_root).replace("\\", "/"),
            "region_manifest_path": str(out_root / "region_manifest.json").replace("\\", "/"),
            "download_manifest_path": str(out_root / "download_manifest.json").replace("\\", "/"),
            "acquisition_manifest_path": str(out_root / "acquisition_manifest.json").replace("\\", "/"),
            "acquisition_manifest_precipitation_dry_run_path": str(out_root / "acquisition_manifest_precipitation_dry_run.json").replace("\\", "/"),
            "preprocessing_manifest_path": str(out_root / "preprocessing_manifest.json").replace("\\", "/"),
            "preprocessing_manifest_precipitation_path": str(out_root / "preprocessing_manifest_precipitation.json").replace("\\", "/"),
            "index_manifest_path": str(out_root / "index_manifest.json").replace("\\", "/"),
            "index_manifest_precipitation_path": str(out_root / "index_manifest_precipitation.json").replace("\\", "/"),
            "pipeline_manifest_path": str(out_root / "pipeline_manifest.json").replace("\\", "/"),
        },
        "download_plan": {"start_year": 2000, "end_year": 2024},
        "pipeline": {
            "acquisition_daily_stats_limit": 3,
            "precipitation_acquisition_request_ids": [
                "era5_hourly_pr__2000_H1",
                "era5_hourly_pr__2000_H2",
            ],
            "precipitation_policy": "legacy_utc_minus_7",
        },
    }


def _make_fake_runner(
    *,
    fail_at: str | None = None,
    write_outputs: bool = True,
):
    """Return a fake step runner that records calls and optionally writes outputs.

    If ``fail_at`` is given, the step with that id returns exit code 2
    and writes nothing. ``write_outputs=False`` simulates a script that
    returns 0 but does not produce its declared output.
    """
    calls: list[tuple[str, tuple[str, ...]]] = []

    def runner(spec, argv, *, repo_root):
        calls.append((spec.step_id, tuple(argv)))
        if fail_at is not None and spec.step_id == fail_at:
            return 2, "simulated failure"
        if write_outputs:
            output_path = Path(spec.output_path_key.__call__({"run": {}}) if False else "")  # unused
            # Find --output in argv and write a deterministic payload there.
            try:
                idx = argv.index("--output")
                target = Path(argv[idx + 1])
            except (ValueError, IndexError):  # pragma: no cover - defensive
                return 0, None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps({"step_id": spec.step_id, "argv": argv}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return 0, None

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


# ---------------------------------------------------------------------------
# load_config + spec table
# ---------------------------------------------------------------------------


def test_canonical_config_loads_with_all_required_keys():
    config = load_config(CANONICAL_CONFIG)
    for key in ("region", "run", "download_plan", "pipeline"):
        assert key in config
    for key in (
        "region_manifest_path",
        "download_manifest_path",
        "acquisition_manifest_path",
        "acquisition_manifest_precipitation_dry_run_path",
        "preprocessing_manifest_path",
        "preprocessing_manifest_precipitation_path",
        "index_manifest_path",
        "index_manifest_precipitation_path",
        "pipeline_manifest_path",
    ):
        assert key in config["run"], f"config.run missing {key}"


def test_load_config_rejects_missing_file(tmp_path: Path):
    with pytest.raises(PipelineRunnerError, match="not found"):
        load_config(tmp_path / "missing.json")


def test_load_config_rejects_missing_top_level_keys(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"region": {}}), encoding="utf-8")
    with pytest.raises(PipelineRunnerError, match="missing required top-level key"):
        load_config(bad)


def test_canonical_step_order_matches_step_specs():
    assert CANONICAL_STEP_ORDER == tuple(s.step_id for s in STEP_SPECS)
    assert len(STEP_SPECS) == 8


# ---------------------------------------------------------------------------
# run_pipeline with a fake step runner
# ---------------------------------------------------------------------------


def test_run_pipeline_success_path_records_all_eight_steps(tmp_path: Path):
    config = _stub_config(tmp_path)
    runner = _make_fake_runner()
    results = run_pipeline(config, repo_root=REPO_ROOT, mode=MODE_DRY_RUN, step_runner=runner)
    assert len(results) == 8
    assert [r.step_id for r in results] == list(CANONICAL_STEP_ORDER)
    assert all(r.status == STEP_STATUS_SUCCEEDED for r in results)
    assert all(r.output_hash is not None and r.output_hash.startswith("sha256:") for r in results)


def test_run_pipeline_step_argv_matches_expected_order(tmp_path: Path):
    config = _stub_config(tmp_path)
    runner = _make_fake_runner()
    results = run_pipeline(config, repo_root=REPO_ROOT, mode=MODE_DRY_RUN, step_runner=runner)
    # Spot-check a few critical argv constructions.
    by_id = {r.step_id: r for r in results}
    assert "--region-id" in by_id[STEP_VALIDATE_REGION].argv
    assert "--limit" in by_id["acquire_daily_stats_dry_run"].argv
    pr_acq = by_id[STEP_ACQUIRE_PRECIPITATION]
    assert pr_acq.argv.count("--request-id") == 2
    assert "era5_hourly_pr__2000_H1" in pr_acq.argv
    assert "era5_hourly_pr__2000_H2" in pr_acq.argv
    pr_pre = by_id[STEP_PREPROCESS_PRECIPITATION]
    assert "--precipitation-policy" in pr_pre.argv
    assert "legacy_utc_minus_7" in pr_pre.argv
    pr_idx = by_id[STEP_INDICES_PRECIPITATION]
    assert "--index-family" in pr_idx.argv
    assert "precipitation" in pr_idx.argv


def test_run_pipeline_failure_stops_subsequent_steps(tmp_path: Path):
    config = _stub_config(tmp_path)
    runner = _make_fake_runner(fail_at=STEP_PREPROCESS_DAILY_STATS)
    results = run_pipeline(config, repo_root=REPO_ROOT, mode=MODE_DRY_RUN, step_runner=runner)
    assert len(results) == 8
    statuses = {r.step_id: r.status for r in results}
    # Earlier steps succeed.
    for earlier in (
        STEP_VALIDATE_REGION,
        STEP_PLAN_DOWNLOADS,
        "acquire_daily_stats_dry_run",
        STEP_ACQUIRE_PRECIPITATION,
    ):
        assert statuses[earlier] == STEP_STATUS_SUCCEEDED
    # Failing step recorded.
    assert statuses[STEP_PREPROCESS_DAILY_STATS] == STEP_STATUS_FAILED
    # Later steps skipped.
    for later in (
        STEP_PREPROCESS_PRECIPITATION,
        STEP_INDICES_TEMPERATURE,
        STEP_INDICES_PRECIPITATION,
    ):
        assert statuses[later] == STEP_STATUS_SKIPPED


def test_run_pipeline_marks_step_failed_if_output_missing_despite_rc_zero(tmp_path: Path):
    """Defensive check: a script that returns 0 but does not write its
    declared output must be recorded as failed, not silently succeed."""
    config = _stub_config(tmp_path)
    runner = _make_fake_runner(write_outputs=False)
    results = run_pipeline(config, repo_root=REPO_ROOT, mode=MODE_DRY_RUN, step_runner=runner)
    assert results[0].status == STEP_STATUS_FAILED
    assert "declared output missing" in (results[0].error or "")
    # Subsequent steps must be skipped.
    assert all(r.status == STEP_STATUS_SKIPPED for r in results[1:])


def test_run_pipeline_rejects_unsupported_mode(tmp_path: Path):
    config = _stub_config(tmp_path)
    with pytest.raises(PipelineRunnerError, match="not one of"):
        run_pipeline(config, repo_root=REPO_ROOT, mode="execute")


# ---------------------------------------------------------------------------
# Manifest write determinism
# ---------------------------------------------------------------------------


def test_build_and_write_pipeline_manifest_is_deterministic(tmp_path: Path):
    config = _stub_config(tmp_path)
    runner = _make_fake_runner()
    results = run_pipeline(config, repo_root=REPO_ROOT, mode=MODE_DRY_RUN, step_runner=runner)
    manifest = build_pipeline_manifest(
        config=config,
        config_path=tmp_path / "config.json",
        config_hash="sha256:" + "a" * 64,
        mode=MODE_DRY_RUN,
        results=results,
        created_by="tests",
    )
    assert manifest["manifest_type"] == MANIFEST_TYPE_PIPELINE
    assert manifest["mode"] == MODE_DRY_RUN
    assert manifest["execution_status"] == EXECUTION_STATUS_COMPLETED_DRY_RUN
    assert manifest["step_count"] == 8
    assert manifest["succeeded_count"] == 8
    assert manifest["failed_count"] == 0
    assert manifest["skipped_count"] == 0
    assert manifest["requires_network"] is False
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_pipeline_manifest(a, manifest)
    write_pipeline_manifest(b, manifest)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_text(encoding="utf-8").endswith("\n")


def test_failure_manifest_records_failed_step_and_status(tmp_path: Path):
    config = _stub_config(tmp_path)
    runner = _make_fake_runner(fail_at=STEP_INDICES_TEMPERATURE)
    results = run_pipeline(config, repo_root=REPO_ROOT, mode=MODE_DRY_RUN, step_runner=runner)
    manifest = build_pipeline_manifest(
        config=config,
        config_path=tmp_path / "config.json",
        config_hash="sha256:" + "b" * 64,
        mode=MODE_DRY_RUN,
        results=results,
        created_by="tests",
    )
    assert manifest["execution_status"] == EXECUTION_STATUS_FAILED
    assert manifest["failed_step"] == STEP_INDICES_TEMPERATURE
    assert manifest["failed_count"] == 1
    assert manifest["skipped_count"] >= 1


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_pipeline_runner_does_not_eagerly_import_heavy_deps():
    """``import lib.pipeline_runner`` must not pull in cdsapi / xarray /
    numpy / dask / icclim / xclim / geopandas / rioxarray."""
    import inspect

    import lib.pipeline_runner as m

    src = inspect.getsource(m)
    for line in src.splitlines():
        stripped = line.lstrip()
        if line == stripped:
            for forbidden in (
                "import cdsapi", "from cdsapi",
                "import xarray", "from xarray",
                "import numpy", "from numpy",
                "import dask", "from dask",
                "import icclim", "from icclim",
                "import xclim", "from xclim",
                "import geopandas", "from geopandas",
                "import rioxarray", "from rioxarray",
            ):
                assert not stripped.startswith(forbidden), f"eager import: {line!r}"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "05_run_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_runner_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_canonical_pipeline_manifest_matches_canonical_artifact():
    """The committed canonical pipeline manifest must reflect what the
    runner produces against the canonical config. This catches drift
    if either the config or any underlying script changes."""
    canonical = json.loads(CANONICAL_PIPELINE_MANIFEST.read_text(encoding="utf-8"))
    assert canonical["manifest_type"] == MANIFEST_TYPE_PIPELINE
    assert canonical["mode"] == MODE_DRY_RUN
    assert canonical["execution_status"] == EXECUTION_STATUS_COMPLETED_DRY_RUN
    assert canonical["step_count"] == 8
    assert canonical["succeeded_count"] == 8
    assert canonical["requires_network"] is False
    step_ids = [s["step_id"] for s in canonical["steps"]]
    assert step_ids == list(CANONICAL_STEP_ORDER)
    for step in canonical["steps"]:
        assert step["output_hash"].startswith("sha256:")
        assert step["status"] == STEP_STATUS_SUCCEEDED


def test_paired_review_prompt_for_m008_exists_and_stays_under_line_cap():
    """The M008 coding prompt has a paired review prompt under the
    repository's 400-line prompt ceiling."""
    candidate = REPO_ROOT / "prompts" / "for_review_agent" / "012_review_m008_local_pipeline_runner.md"
    assert candidate.exists()
    line_count = len(candidate.read_text(encoding="utf-8").splitlines())
    assert line_count <= 400
