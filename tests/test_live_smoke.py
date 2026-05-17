"""Tests for ``lib.live_smoke`` and ``scripts/07_run_live_smoke.py``.

Preflight tests run the real preflight against the canonical M001-M009
manifests (read-only). Execute-mode tests inject fake step runners and
never call ``cdsapi``, never need credentials, and never write NetCDF
under ``runs/dev_region/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from lib.live_smoke import (
    ALLOWED_INDEX_IDS,
    ALLOWED_REQUEST_IDS,
    CANONICAL_DEV_OUTPUT_ROOT,
    CANONICAL_EXECUTE_STEP_ORDER,
    CANONICAL_PREFLIGHT_CHECK_ORDER,
    CHECK_M009_VALIDATION_PASSED,
    CHECK_OUTPUT_ROOT_IS_SCRATCH,
    CHECK_REQUEST_IN_ALLOWLIST,
    CHECK_STATUS_FAILED,
    CHECK_STATUS_PASSED,
    CONFIRMATION_TOKEN,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REQUEST_ID,
    EXECUTE_EXECUTION_STATUS_COMPLETED,
    EXECUTE_EXECUTION_STATUS_FAILED,
    MANIFEST_TYPE_LIVE_SMOKE,
    MODE_EXECUTE,
    MODE_PREFLIGHT,
    PREFLIGHT_EXECUTION_STATUS_BLOCKED,
    PREFLIGHT_EXECUTION_STATUS_READY,
    STEP_ACQUIRE_ONE_REQUEST,
    STEP_INDICES_ONE_REQUEST,
    STEP_PREPROCESS_ONE_REQUEST,
    STEP_STATUS_FAILED,
    STEP_STATUS_SKIPPED,
    STEP_STATUS_SUCCEEDED,
    STEP_VALIDATE_PRODUCTS,
    LiveSmokeError,
    load_config,
    run_execute,
    run_preflight,
    write_plan,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_CONFIG = REPO_ROOT / "configs" / "rbmn_local.json"
CANONICAL_LIVE_SMOKE_PLAN = REPO_ROOT / "runs" / "dev_region" / "live_smoke_plan.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_with_repo_paths() -> dict[str, Any]:
    """Load the canonical config but resolve its run paths relative to REPO_ROOT.

    The config stores POSIX-relative paths (e.g.
    ``runs/dev_region/region_manifest.json``); the preflight reads them
    via ``Path(...).exists()`` against the current working directory.
    For tests that run from arbitrary cwd, rewrite each run path to be
    absolute under REPO_ROOT so existence checks succeed regardless of
    where pytest is invoked from.
    """
    cfg = json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))
    for key, value in list(cfg["run"].items()):
        if isinstance(value, str) and value.startswith("runs/"):
            cfg["run"][key] = str((REPO_ROOT / value).resolve()).replace("\\", "/")
    cfg["region"]["geometry_path"] = str(
        (REPO_ROOT / cfg["region"]["geometry_path"]).resolve()
    ).replace("\\", "/")
    return cfg


def _make_fake_runner(*, fail_at: str | None = None):
    """Return a fake execute runner that records call order."""
    calls: list[tuple[str, tuple[str, ...]]] = []

    def runner(step, *, repo_root):
        calls.append((step.step_id, tuple(step.argv)))
        if fail_at is not None and step.step_id == fail_at:
            return 2, "simulated step failure"
        return 0, None

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_reads_canonical():
    cfg = load_config(CANONICAL_CONFIG)
    assert "run" in cfg and "live_smoke" in cfg
    assert cfg["run"]["live_smoke_plan_path"].endswith("live_smoke_plan.json")
    assert cfg["live_smoke"]["confirmation_token"] == CONFIRMATION_TOKEN
    assert DEFAULT_REQUEST_ID in cfg["live_smoke"]["allowed_request_ids"]


def test_load_config_rejects_missing_file(tmp_path: Path):
    with pytest.raises(LiveSmokeError, match="not found"):
        load_config(tmp_path / "missing.json")


def test_load_config_rejects_missing_top_level_keys(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"region": {}}), encoding="utf-8")
    with pytest.raises(LiveSmokeError, match="missing required top-level key"):
        load_config(bad)


# ---------------------------------------------------------------------------
# Preflight success against the canonical manifests
# ---------------------------------------------------------------------------


def test_preflight_against_canonical_is_ready(tmp_path: Path):
    cfg = _config_with_repo_paths()
    plan = run_preflight(
        cfg,
        request_id=DEFAULT_REQUEST_ID,
        output_root=tmp_path / "live_smoke_tmax_2000",
    )
    assert plan["manifest_type"] == MANIFEST_TYPE_LIVE_SMOKE
    assert plan["mode"] == MODE_PREFLIGHT
    assert plan["requires_network"] is False
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_READY
    assert plan["request_id"] == DEFAULT_REQUEST_ID
    assert plan["confirmation_token_required"] == CONFIRMATION_TOKEN
    assert [c["check_id"] for c in plan["preflight_checks"]] == list(
        CANONICAL_PREFLIGHT_CHECK_ORDER
    )
    assert all(c["status"] == CHECK_STATUS_PASSED for c in plan["preflight_checks"])
    assert [s["step_id"] for s in plan["steps"]] == list(CANONICAL_EXECUTE_STEP_ORDER)
    assert all(s["status"] == "planned" for s in plan["steps"])
    # Prereq manifests carry sha256: hashes computed from the canonical files.
    assert {m["name"] for m in plan["prerequisite_manifests"]} == {
        "region_manifest", "download_manifest",
        "pipeline_manifest", "validation_report",
    }
    for m in plan["prerequisite_manifests"]:
        assert m["hash"].startswith("sha256:")


def test_preflight_step_argv_carries_smoke_constraints(tmp_path: Path):
    cfg = _config_with_repo_paths()
    plan = run_preflight(
        cfg,
        request_id=DEFAULT_REQUEST_ID,
        output_root=tmp_path / "live_smoke_tmax_2000",
    )
    by_id = {s["step_id"]: s for s in plan["steps"]}
    # Acquire step carries --mode execute, --request-id, --limit absent
    acq = by_id[STEP_ACQUIRE_ONE_REQUEST]
    assert "--mode" in acq["argv"] and "execute" in acq["argv"]
    assert "--request-id" in acq["argv"]
    assert DEFAULT_REQUEST_ID in acq["argv"]
    # Indices step requests both allowed indices, never tmin/tmean
    idx = by_id[STEP_INDICES_ONE_REQUEST]
    assert idx["argv"].count("--index-id") == len(ALLOWED_INDEX_IDS)
    for index_id in ALLOWED_INDEX_IDS:
        assert index_id in idx["argv"]
    # Tmin / tmean indices must NOT appear
    for forbidden in ("Tmx", "Tmn", "TNn", "DTR", "TR"):
        assert forbidden not in idx["argv"]


# ---------------------------------------------------------------------------
# Preflight failure modes (blocked)
# ---------------------------------------------------------------------------


def test_preflight_rejects_unsupported_request_id(tmp_path: Path):
    cfg = _config_with_repo_paths()
    plan = run_preflight(
        cfg,
        request_id="era5_daily_stats__tmax__1850",
        output_root=tmp_path / "live_smoke_other",
    )
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_BLOCKED
    by_id = {c["check_id"]: c for c in plan["preflight_checks"]}
    assert by_id[CHECK_REQUEST_IN_ALLOWLIST]["status"] == CHECK_STATUS_FAILED


def test_preflight_rejects_canonical_dev_output_root(tmp_path: Path):
    """A live-smoke run targeting runs/dev_region must be blocked.

    Preflight is technically read-only, but the output_root field will
    later drive the execute path; blocking here keeps the safety
    contract uniform across modes.
    """
    cfg = _config_with_repo_paths()
    plan = run_preflight(
        cfg,
        request_id=DEFAULT_REQUEST_ID,
        output_root=Path(CANONICAL_DEV_OUTPUT_ROOT),
    )
    by_id = {c["check_id"]: c for c in plan["preflight_checks"]}
    assert by_id[CHECK_OUTPUT_ROOT_IS_SCRATCH]["status"] == CHECK_STATUS_FAILED
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_BLOCKED


def test_preflight_m009_failure_blocks(tmp_path: Path, monkeypatch):
    """Tamper a downstream manifest under a copied repo state so M009 fails."""
    # Stage a tmp repo: copy canonical manifests, then mutate one.
    staged_runs = tmp_path / "runs" / "dev_region"
    staged_runs.mkdir(parents=True)
    manifest_names = (
        "region_manifest.json", "download_manifest.json",
        "acquisition_manifest.json", "acquisition_manifest_precipitation_dry_run.json",
        "preprocessing_manifest.json", "preprocessing_manifest_precipitation.json",
        "index_manifest.json", "index_manifest_precipitation.json",
        "pipeline_manifest.json", "validation_report.json",
    )
    canonical_runs_root = REPO_ROOT / "runs" / "dev_region"
    for fname in manifest_names:
        shutil.copyfile(canonical_runs_root / fname, staged_runs / fname)
    # Rewrite the staged pipeline manifest so every step's output_path
    # points at the staged copy of that file. Without this, M009's hash
    # check rehashes the *original* canonical files and passes even
    # after we tamper with the staged copies.
    pipeline_path = staged_runs / "pipeline_manifest.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    canonical_posix = str(canonical_runs_root.resolve()).replace("\\", "/")
    staged_posix = str(staged_runs.resolve()).replace("\\", "/")
    for step in pipeline.get("steps", []):
        op = step.get("output_path")
        if isinstance(op, str):
            op_norm = op.replace("\\", "/")
            if op_norm.startswith(canonical_posix):
                step["output_path"] = op_norm.replace(canonical_posix, staged_posix, 1)
            elif op_norm.startswith("runs/dev_region/"):
                step["output_path"] = op_norm.replace(
                    "runs/dev_region/",
                    staged_posix + "/",
                    1,
                )
    pipeline_path.write_text(
        json.dumps(pipeline, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Mutate the staged region manifest so its hash no longer matches
    # what the pipeline manifest records under steps[*].output_hash.
    region_path = staged_runs / "region_manifest.json"
    region_path.write_bytes(region_path.read_bytes() + b" ")  # one extra byte
    cfg = _config_with_repo_paths()
    # Point config paths at the tmp staging copy.
    for key in (
        "region_manifest_path", "download_manifest_path",
        "acquisition_manifest_path", "acquisition_manifest_precipitation_dry_run_path",
        "preprocessing_manifest_path", "preprocessing_manifest_precipitation_path",
        "index_manifest_path", "index_manifest_precipitation_path",
        "pipeline_manifest_path", "validation_report_path",
    ):
        fname = Path(cfg["run"][key]).name
        cfg["run"][key] = str(staged_runs / fname).replace("\\", "/")
    cfg["run"]["output_root"] = str(staged_runs).replace("\\", "/")
    plan = run_preflight(
        cfg,
        request_id=DEFAULT_REQUEST_ID,
        output_root=tmp_path / "live_smoke_tmax_2000",
    )
    by_id = {c["check_id"]: c for c in plan["preflight_checks"]}
    assert by_id[CHECK_M009_VALIDATION_PASSED]["status"] == CHECK_STATUS_FAILED
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_BLOCKED


# ---------------------------------------------------------------------------
# Execute orchestration (fake runner only -- never real CDS)
# ---------------------------------------------------------------------------


def test_execute_fake_runner_records_four_steps_in_order(tmp_path: Path):
    cfg = _config_with_repo_paths()
    runner = _make_fake_runner()
    plan = run_execute(
        cfg,
        request_id=DEFAULT_REQUEST_ID,
        output_root=tmp_path / "live_smoke_tmax_2000",
        repo_root=REPO_ROOT,
        confirm_live=CONFIRMATION_TOKEN,
        step_runner=runner,
        skip_product_validation=True,
    )
    assert plan["mode"] == MODE_EXECUTE
    assert plan["requires_network"] is True
    assert plan["execution_status"] == EXECUTE_EXECUTION_STATUS_COMPLETED
    # Fake runner saw three callable scripts (validation is in-process).
    call_ids = [c[0] for c in runner.calls]
    assert call_ids == [STEP_ACQUIRE_ONE_REQUEST, STEP_PREPROCESS_ONE_REQUEST, STEP_INDICES_ONE_REQUEST]
    step_statuses = [(s["step_id"], s["status"]) for s in plan["steps"]]
    # All four steps recorded; validation skipped per the test flag but
    # still counts as succeeded.
    assert step_statuses == [
        (STEP_ACQUIRE_ONE_REQUEST, STEP_STATUS_SUCCEEDED),
        (STEP_PREPROCESS_ONE_REQUEST, STEP_STATUS_SUCCEEDED),
        (STEP_INDICES_ONE_REQUEST, STEP_STATUS_SUCCEEDED),
        (STEP_VALIDATE_PRODUCTS, STEP_STATUS_SUCCEEDED),
    ]


def test_execute_failed_acquisition_skips_downstream_steps(tmp_path: Path):
    cfg = _config_with_repo_paths()
    runner = _make_fake_runner(fail_at=STEP_ACQUIRE_ONE_REQUEST)
    plan = run_execute(
        cfg,
        request_id=DEFAULT_REQUEST_ID,
        output_root=tmp_path / "live_smoke_tmax_2000",
        repo_root=REPO_ROOT,
        confirm_live=CONFIRMATION_TOKEN,
        step_runner=runner,
        skip_product_validation=True,
    )
    statuses = {s["step_id"]: s["status"] for s in plan["steps"]}
    assert statuses[STEP_ACQUIRE_ONE_REQUEST] == STEP_STATUS_FAILED
    for downstream in (
        STEP_PREPROCESS_ONE_REQUEST, STEP_INDICES_ONE_REQUEST, STEP_VALIDATE_PRODUCTS,
    ):
        assert statuses[downstream] == STEP_STATUS_SKIPPED
    assert plan["execution_status"] in {EXECUTE_EXECUTION_STATUS_FAILED, "partial"}


def test_execute_without_confirm_live_raises_before_runner(tmp_path: Path):
    """Direct library call without ``confirm_live`` must refuse before any
    step runner -- real or fake -- can execute. Without this gate the
    default ``_default_step_runner`` could reach live CDS."""
    cfg = _config_with_repo_paths()
    runner = _make_fake_runner()
    with pytest.raises(LiveSmokeError, match="confirm_live"):
        run_execute(
            cfg,
            request_id=DEFAULT_REQUEST_ID,
            output_root=tmp_path / "live_smoke_tmax_2000",
            repo_root=REPO_ROOT,
            step_runner=runner,
            skip_product_validation=True,
        )
    assert runner.calls == []  # type: ignore[attr-defined]


def test_execute_with_wrong_confirm_live_raises_before_runner(tmp_path: Path):
    cfg = _config_with_repo_paths()
    runner = _make_fake_runner()
    with pytest.raises(LiveSmokeError, match="confirm_live"):
        run_execute(
            cfg,
            request_id=DEFAULT_REQUEST_ID,
            output_root=tmp_path / "live_smoke_tmax_2000",
            repo_root=REPO_ROOT,
            confirm_live="yes-please",
            step_runner=runner,
            skip_product_validation=True,
        )
    assert runner.calls == []  # type: ignore[attr-defined]


def test_execute_rejects_absolute_canonical_dev_output_root(tmp_path: Path):
    """An absolute path to ``runs/dev_region`` must be refused at the
    library layer, not only by the CLI string-equality check."""
    cfg = _config_with_repo_paths()
    runner = _make_fake_runner()
    absolute_canonical = (REPO_ROOT / CANONICAL_DEV_OUTPUT_ROOT).resolve()
    with pytest.raises(LiveSmokeError, match="canonical dev root"):
        run_execute(
            cfg,
            request_id=DEFAULT_REQUEST_ID,
            output_root=absolute_canonical,
            repo_root=REPO_ROOT,
            confirm_live=CONFIRMATION_TOKEN,
            step_runner=runner,
            skip_product_validation=True,
        )
    assert runner.calls == []  # type: ignore[attr-defined]


def test_preflight_rejects_absolute_canonical_dev_output_root(tmp_path: Path):
    """Preflight check must also catch the absolute-path bypass so the
    safety contract is uniform across modes."""
    cfg = _config_with_repo_paths()
    absolute_canonical = (REPO_ROOT / CANONICAL_DEV_OUTPUT_ROOT).resolve()
    plan = run_preflight(
        cfg,
        request_id=DEFAULT_REQUEST_ID,
        output_root=absolute_canonical,
    )
    by_id = {c["check_id"]: c for c in plan["preflight_checks"]}
    assert by_id[CHECK_OUTPUT_ROOT_IS_SCRATCH]["status"] == CHECK_STATUS_FAILED
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_BLOCKED


# ---------------------------------------------------------------------------
# Config-vs-code consistency
# ---------------------------------------------------------------------------


def test_config_constants_match_code_constants():
    """The live_smoke section of ``configs/rbmn_local.json`` and the
    constants in ``lib/live_smoke.py`` are two surfaces describing the
    same contract. They must agree exactly; otherwise the operator can
    pass a token or request id that one layer accepts and the other
    silently rewrites."""
    cfg = json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))
    live = cfg["live_smoke"]
    assert live["confirmation_token"] == CONFIRMATION_TOKEN
    assert tuple(sorted(live["allowed_request_ids"])) == tuple(sorted(ALLOWED_REQUEST_IDS))
    assert tuple(live["allowed_indices"]) == tuple(ALLOWED_INDEX_IDS)
    assert live["default_request_id"] == DEFAULT_REQUEST_ID


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "07_run_live_smoke.py"
    spec = importlib.util.spec_from_file_location("live_smoke_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_cli_preflight_against_canonical_exits_zero(tmp_path: Path, capsys):
    main = _load_script_main()
    output = tmp_path / "plan.json"
    rc = main(
        [
            "--config", str(CANONICAL_CONFIG),
            "--mode", "preflight",
            "--output", str(output),
            "--output-root", str(tmp_path / "live_smoke_tmax_2000"),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote live-smoke plan" in captured.out
    assert "execution_status=ready_for_owner_authorized_live_test" in captured.out
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["manifest_type"] == MANIFEST_TYPE_LIVE_SMOKE
    assert loaded["mode"] == MODE_PREFLIGHT
    assert loaded["requires_network"] is False


def test_cli_execute_without_confirmation_exits_2(tmp_path: Path, capsys):
    main = _load_script_main()
    rc = main(
        [
            "--config", str(CANONICAL_CONFIG),
            "--mode", "execute",
            "--output", str(tmp_path / "live_smoke_run.json"),
            "--output-root", str(tmp_path / "live_smoke_tmax_2000"),
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "--confirm-live" in captured.err


def test_cli_execute_with_wrong_confirmation_exits_2(tmp_path: Path, capsys):
    main = _load_script_main()
    rc = main(
        [
            "--config", str(CANONICAL_CONFIG),
            "--mode", "execute",
            "--output", str(tmp_path / "live_smoke_run.json"),
            "--output-root", str(tmp_path / "live_smoke_tmax_2000"),
            "--confirm-live", "yes",
        ]
    )
    assert rc == 2


def test_cli_execute_with_canonical_dev_output_root_exits_2(tmp_path: Path, capsys):
    main = _load_script_main()
    rc = main(
        [
            "--config", str(CANONICAL_CONFIG),
            "--mode", "execute",
            "--output", str(tmp_path / "live_smoke_run.json"),
            "--output-root", "runs/dev_region",
            "--confirm-live", CONFIRMATION_TOKEN,
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "runs/dev_region" in captured.err


def test_cli_execute_with_absolute_canonical_dev_output_root_exits_2(tmp_path: Path, capsys):
    """A user typing the absolute path to runs/dev_region must hit the
    same exit-2 refusal as the relative form. Without resolved-path
    comparison the CLI's string-equality check passes and execute mode
    proceeds against the canonical reference root."""
    main = _load_script_main()
    absolute_canonical = (REPO_ROOT / CANONICAL_DEV_OUTPUT_ROOT).resolve()
    rc = main(
        [
            "--config", str(CANONICAL_CONFIG),
            "--mode", "execute",
            "--output", str(tmp_path / "live_smoke_run.json"),
            "--output-root", str(absolute_canonical),
            "--confirm-live", CONFIRMATION_TOKEN,
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "canonical" in captured.err.lower() or "dev_region" in captured.err


def test_cli_preflight_rerun_is_byte_identical(tmp_path: Path):
    main = _load_script_main()
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    output_root = tmp_path / "live_smoke_tmax_2000"
    for target in (out_a, out_b):
        rc = main(
            [
                "--config", str(CANONICAL_CONFIG),
                "--mode", "preflight",
                "--output", str(target),
                "--output-root", str(output_root),
            ]
        )
        assert rc == 0
    assert out_a.read_bytes() == out_b.read_bytes()


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_live_smoke_module_does_not_eagerly_import_heavy_deps():
    """``import lib.live_smoke`` must not pull in cdsapi / xarray /
    numpy / dask."""
    import inspect

    import lib.live_smoke as m

    src = inspect.getsource(m)
    for line in src.splitlines():
        stripped = line.lstrip()
        if line == stripped:
            for forbidden in (
                "import cdsapi", "from cdsapi",
                "import xarray", "from xarray",
                "import numpy", "from numpy",
                "import dask", "from dask",
            ):
                assert not stripped.startswith(forbidden), f"eager import: {line!r}"


# ---------------------------------------------------------------------------
# Canonical-artifact pin + side-effect policy
# ---------------------------------------------------------------------------


def test_canonical_live_smoke_plan_matches_committed_artifact():
    canonical = json.loads(CANONICAL_LIVE_SMOKE_PLAN.read_text(encoding="utf-8"))
    assert canonical["manifest_type"] == MANIFEST_TYPE_LIVE_SMOKE
    assert canonical["mode"] == MODE_PREFLIGHT
    assert canonical["requires_network"] is False
    assert canonical["execution_status"] == PREFLIGHT_EXECUTION_STATUS_READY
    assert canonical["request_id"] == DEFAULT_REQUEST_ID
    assert [c["check_id"] for c in canonical["preflight_checks"]] == list(
        CANONICAL_PREFLIGHT_CHECK_ORDER
    )
    assert all(c["status"] == CHECK_STATUS_PASSED for c in canonical["preflight_checks"])
    assert [s["step_id"] for s in canonical["steps"]] == list(CANONICAL_EXECUTE_STEP_ORDER)


def test_no_nc_files_under_dev_region():
    """The M010 work must not leave NetCDF files under the canonical dev root."""
    dev_root = REPO_ROOT / "runs" / "dev_region"
    nc = list(dev_root.rglob("*.nc")) + list(dev_root.rglob("*.nc4"))
    assert nc == []
