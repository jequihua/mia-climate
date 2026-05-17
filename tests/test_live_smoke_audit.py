"""Tests for ``lib.live_smoke_audit`` and ``scripts/08_audit_live_smoke.py``.

Preflight tests hash the canonical M010 plan + the M010 safety-corrections
review file (read-only). Audit-mode tests build a fake M010 execute
report + synthetic NetCDF fixtures in tmp directories; they never call
``cdsapi`` and never write under ``runs/dev_region/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from lib.live_smoke_audit import (
    AUDIT_EXECUTION_STATUS_FAILED,
    AUDIT_EXECUTION_STATUS_PASSED,
    CANONICAL_AUDIT_CHECK_ORDER,
    CANONICAL_PREFLIGHT_CHECK_ORDER,
    CHECK_DAILY_PRODUCT_SCHEMA,
    CHECK_EXPECTED_OUTPUT_ROOT_IS_SCRATCH,
    CHECK_LIVE_REPORT_MODE_EXECUTE,
    CHECK_LIVE_REPORT_PRESENT,
    CHECK_LIVE_REPORT_STEPS_SUCCEEDED,
    CHECK_M010_PLAN_PRESENT,
    CHECK_M010_PLAN_READY,
    CHECK_M010_SAFETY_REVIEW_PRESENT,
    CHECK_PRODUCTS_PRESENT,
    CHECK_STATUS_FAILED,
    CHECK_STATUS_PASSED,
    CHECK_SU_PRODUCT_SCHEMA,
    CHECK_TXX_PRODUCT_SCHEMA,
    CONFIRMATION_TOKEN,
    DEFAULT_AUDIT_REPORT_NAME,
    DEFAULT_EXPECTED_OUTPUT_ROOT,
    DEFAULT_LIVE_REPORT_NAME,
    DEFAULT_REQUEST_ID,
    MANIFEST_TYPE_AUDIT,
    MODE_AUDIT,
    MODE_PREFLIGHT,
    PREFLIGHT_EXECUTION_STATUS_BLOCKED,
    PREFLIGHT_EXECUTION_STATUS_READY,
    LiveSmokeAuditError,
    load_config,
    run_audit,
    run_preflight,
    write_plan,
)
from lib.live_smoke import (
    CANONICAL_DEV_OUTPUT_ROOT,
    MANIFEST_TYPE_LIVE_SMOKE,
    MODE_EXECUTE as M010_MODE_EXECUTE,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_CONFIG = REPO_ROOT / "configs" / "rbmn_local.json"
CANONICAL_M010_PLAN = REPO_ROOT / "runs" / "dev_region" / "live_smoke_plan.json"
CANONICAL_M010_SAFETY_REVIEW = (
    REPO_ROOT / "05_governance" / "reviews"
    / "review_m010_live_smoke_safety_corrections.md"
)
CANONICAL_AUDIT_PLAN = (
    REPO_ROOT / "runs" / "dev_region" / "live_smoke_audit_plan.json"
)


# ---------------------------------------------------------------------------
# Fixtures: fake M010 execute report + synthetic NetCDFs
# ---------------------------------------------------------------------------


def _make_fake_m010_execute_report(scratch_root: Path) -> dict[str, Any]:
    """Build a fake M010 execute report mirroring the real schema."""
    return {
        "manifest_type": MANIFEST_TYPE_LIVE_SMOKE,
        "mode": M010_MODE_EXECUTE,
        "requires_network": True,
        "run_id": scratch_root.name,
        "output_root": str(scratch_root).replace("\\", "/"),
        "request_id": DEFAULT_REQUEST_ID,
        "request": {
            "request_id": DEFAULT_REQUEST_ID,
            "year": 2000,
            "project_variables": ["tmax"],
            "output_path": "raw/era5_land/daily_statistics/tmax/2000.nc",
        },
        "planned_outputs": {},
        "prerequisite_manifests": [],
        "preflight_checks": [],
        "steps": [
            {"step_id": "acquire_one_request", "status": "succeeded", "exit_code": 0},
            {"step_id": "preprocess_one_request", "status": "succeeded", "exit_code": 0},
            {"step_id": "indices_one_request", "status": "succeeded", "exit_code": 0},
            {"step_id": "validate_products", "status": "succeeded", "exit_code": 0},
        ],
        "product_validations": [],
        "created_by": "scripts/07_run_live_smoke.py",
        "execution_status": "completed_live_smoke",
        "confirmation_token_required": CONFIRMATION_TOKEN,
    }


def _write_daily_tmax_2000(path: Path) -> None:
    times = np.array(
        ["2000-01-15", "2000-06-15", "2000-12-15"], dtype="datetime64[ns]",
    )
    values = np.array(
        [[[25.0, 26.0], [24.0, 25.5]],
         [[35.0, 36.0], [34.0, 35.5]],
         [[22.0, 23.0], [21.0, 22.5]]],
        dtype="float32",
    )
    da = xr.DataArray(
        values, dims=("time", "lat", "lon"),
        coords={
            "time": times,
            "lat": [22.0, 21.5],
            "lon": [-105.6, -105.4],
        },
        name="tmax", attrs={"units": "degC"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(path, engine="h5netcdf")


def _write_annual_index(path: Path, *, index_id: str, units: str, value: float) -> None:
    times = np.array(["2000-12-31"], dtype="datetime64[ns]")
    da = xr.DataArray(
        np.array([[[value, value], [value, value]]], dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={
            "time": times,
            "lat": [22.0, 21.5],
            "lon": [-105.6, -105.4],
        },
        name=index_id, attrs={"units": units},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(path, engine="h5netcdf")


def _stage_scratch_products(scratch_root: Path) -> None:
    _write_daily_tmax_2000(
        scratch_root / "raw" / "era5_land" / "daily_statistics"
        / "tmax" / "2000.nc"
    )
    _write_daily_tmax_2000(
        scratch_root / "intermediate" / "daily" / "tmax" / "2000.nc"
    )
    _write_annual_index(
        scratch_root / "derived" / "indices" / "TXx.nc",
        index_id="TXx", units="degC", value=36.0,
    )
    _write_annual_index(
        scratch_root / "derived" / "indices" / "SU.nc",
        index_id="SU", units="days", value=180.0,
    )


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_reads_canonical():
    cfg = load_config(CANONICAL_CONFIG)
    assert "run" in cfg and "live_smoke" in cfg
    assert cfg["run"]["live_smoke_audit_plan_path"].endswith(
        "live_smoke_audit_plan.json"
    )


def test_load_config_rejects_missing_file(tmp_path: Path):
    with pytest.raises(LiveSmokeAuditError, match="not found"):
        load_config(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# Preflight success and failure modes
# ---------------------------------------------------------------------------


def test_preflight_against_canonical_is_ready(tmp_path: Path):
    plan = run_preflight(
        m010_plan_path=CANONICAL_M010_PLAN,
        m010_safety_review_path=CANONICAL_M010_SAFETY_REVIEW,
        expected_output_root=tmp_path / "live_smoke_tmax_2000",
    )
    assert plan["manifest_type"] == MANIFEST_TYPE_AUDIT
    assert plan["mode"] == MODE_PREFLIGHT
    assert plan["requires_network"] is False
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_READY
    assert plan["request_id"] == DEFAULT_REQUEST_ID
    assert plan["confirmation_token_required"] == CONFIRMATION_TOKEN
    assert [c["check_id"] for c in plan["preflight_checks"]] == list(
        CANONICAL_PREFLIGHT_CHECK_ORDER
    )
    assert all(c["status"] == CHECK_STATUS_PASSED for c in plan["preflight_checks"])
    assert plan["audit_checks_planned"] == list(CANONICAL_AUDIT_CHECK_ORDER)
    assert {p["name"] for p in plan["prerequisite_artifacts"]} == {
        "m010_live_smoke_plan", "m010_safety_corrections_review",
    }
    for record in plan["prerequisite_artifacts"]:
        assert record["hash"].startswith("sha256:")


def test_preflight_missing_m010_plan_blocks(tmp_path: Path):
    plan = run_preflight(
        m010_plan_path=tmp_path / "missing_plan.json",
        m010_safety_review_path=CANONICAL_M010_SAFETY_REVIEW,
        expected_output_root=tmp_path / "live_smoke_tmax_2000",
    )
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_BLOCKED
    by_id = {c["check_id"]: c for c in plan["preflight_checks"]}
    assert by_id[CHECK_M010_PLAN_PRESENT]["status"] == CHECK_STATUS_FAILED
    assert by_id[CHECK_M010_PLAN_READY]["status"] == CHECK_STATUS_FAILED


def test_preflight_m010_plan_not_ready_blocks(tmp_path: Path):
    plan_path = tmp_path / "live_smoke_plan.json"
    plan_path.write_text(
        json.dumps({"execution_status": "blocked"}), encoding="utf-8",
    )
    plan = run_preflight(
        m010_plan_path=plan_path,
        m010_safety_review_path=CANONICAL_M010_SAFETY_REVIEW,
        expected_output_root=tmp_path / "live_smoke_tmax_2000",
    )
    by_id = {c["check_id"]: c for c in plan["preflight_checks"]}
    assert by_id[CHECK_M010_PLAN_READY]["status"] == CHECK_STATUS_FAILED
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_BLOCKED


def test_preflight_missing_safety_review_blocks(tmp_path: Path):
    plan = run_preflight(
        m010_plan_path=CANONICAL_M010_PLAN,
        m010_safety_review_path=tmp_path / "missing_review.md",
        expected_output_root=tmp_path / "live_smoke_tmax_2000",
    )
    by_id = {c["check_id"]: c for c in plan["preflight_checks"]}
    assert by_id[CHECK_M010_SAFETY_REVIEW_PRESENT]["status"] == CHECK_STATUS_FAILED
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_BLOCKED


def test_preflight_rejects_canonical_dev_output_root_relative(tmp_path: Path):
    plan = run_preflight(
        m010_plan_path=CANONICAL_M010_PLAN,
        m010_safety_review_path=CANONICAL_M010_SAFETY_REVIEW,
        expected_output_root=Path(CANONICAL_DEV_OUTPUT_ROOT),
    )
    by_id = {c["check_id"]: c for c in plan["preflight_checks"]}
    assert by_id[CHECK_EXPECTED_OUTPUT_ROOT_IS_SCRATCH]["status"] == CHECK_STATUS_FAILED
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_BLOCKED


def test_preflight_rejects_canonical_dev_output_root_absolute(tmp_path: Path):
    absolute = (REPO_ROOT / CANONICAL_DEV_OUTPUT_ROOT).resolve()
    plan = run_preflight(
        m010_plan_path=CANONICAL_M010_PLAN,
        m010_safety_review_path=CANONICAL_M010_SAFETY_REVIEW,
        expected_output_root=absolute,
    )
    by_id = {c["check_id"]: c for c in plan["preflight_checks"]}
    assert by_id[CHECK_EXPECTED_OUTPUT_ROOT_IS_SCRATCH]["status"] == CHECK_STATUS_FAILED
    assert plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_BLOCKED


# ---------------------------------------------------------------------------
# Audit mode -- happy path with synthetic NetCDF
# ---------------------------------------------------------------------------


def test_audit_happy_path_records_hashes_and_validates_products(tmp_path: Path):
    scratch_root = tmp_path / "live_smoke_tmax_2000"
    scratch_root.mkdir()
    live_report_path = scratch_root / DEFAULT_LIVE_REPORT_NAME
    report = _make_fake_m010_execute_report(scratch_root)
    live_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )
    _stage_scratch_products(scratch_root)

    audit = run_audit(
        expected_output_root=scratch_root,
        live_report_path=live_report_path,
    )
    assert audit["manifest_type"] == MANIFEST_TYPE_AUDIT
    assert audit["mode"] == MODE_AUDIT
    assert audit["requires_network"] is False
    assert audit["execution_status"] == AUDIT_EXECUTION_STATUS_PASSED
    assert [c["check_id"] for c in audit["audit_checks"]] == list(
        CANONICAL_AUDIT_CHECK_ORDER
    )
    for c in audit["audit_checks"]:
        assert c["status"] == CHECK_STATUS_PASSED, c
    roles = {a["role"] for a in audit["artifact_hashes"]}
    assert roles == {
        "raw_target_path", "daily_tmax_path", "TXx_index_path", "SU_index_path",
    }
    for record in audit["artifact_hashes"]:
        assert record["exists"] is True
        assert record["hash"].startswith("sha256:")
        assert isinstance(record["byte_size"], int) and record["byte_size"] > 0
    # M009 product validators ran in-process for all three NetCDFs.
    assert len(audit["product_validations"]) == 3
    assert audit["live_report_hash"].startswith("sha256:")


# ---------------------------------------------------------------------------
# Audit mode -- failure modes
# ---------------------------------------------------------------------------


def test_audit_missing_live_report_fails_and_skips_downstream(tmp_path: Path):
    scratch_root = tmp_path / "live_smoke_tmax_2000"
    scratch_root.mkdir()
    audit = run_audit(
        expected_output_root=scratch_root,
        live_report_path=scratch_root / DEFAULT_LIVE_REPORT_NAME,
    )
    by_id = {c["check_id"]: c for c in audit["audit_checks"]}
    assert by_id[CHECK_LIVE_REPORT_PRESENT]["status"] == CHECK_STATUS_FAILED
    # Every downstream audit check is recorded as failed/skipped, never silently dropped.
    for cid in CANONICAL_AUDIT_CHECK_ORDER:
        assert cid in by_id
        if cid != CHECK_LIVE_REPORT_PRESENT:
            assert by_id[cid]["status"] == CHECK_STATUS_FAILED
    assert audit["execution_status"] == AUDIT_EXECUTION_STATUS_FAILED
    assert audit["product_validations"] == []


def test_audit_mode_not_execute_in_live_report_fails(tmp_path: Path):
    scratch_root = tmp_path / "live_smoke_tmax_2000"
    scratch_root.mkdir()
    live_report_path = scratch_root / DEFAULT_LIVE_REPORT_NAME
    report = _make_fake_m010_execute_report(scratch_root)
    report["mode"] = "preflight"  # wrong: audit expects an execute report
    live_report_path.write_text(json.dumps(report), encoding="utf-8")
    _stage_scratch_products(scratch_root)
    audit = run_audit(
        expected_output_root=scratch_root, live_report_path=live_report_path,
    )
    by_id = {c["check_id"]: c for c in audit["audit_checks"]}
    assert by_id[CHECK_LIVE_REPORT_MODE_EXECUTE]["status"] == CHECK_STATUS_FAILED
    assert audit["execution_status"] == AUDIT_EXECUTION_STATUS_FAILED


def test_audit_failed_step_in_live_report_fails(tmp_path: Path):
    scratch_root = tmp_path / "live_smoke_tmax_2000"
    scratch_root.mkdir()
    live_report_path = scratch_root / DEFAULT_LIVE_REPORT_NAME
    report = _make_fake_m010_execute_report(scratch_root)
    report["steps"][1]["status"] = "failed"
    report["steps"][1]["exit_code"] = 2
    live_report_path.write_text(json.dumps(report), encoding="utf-8")
    _stage_scratch_products(scratch_root)
    audit = run_audit(
        expected_output_root=scratch_root, live_report_path=live_report_path,
    )
    by_id = {c["check_id"]: c for c in audit["audit_checks"]}
    assert by_id[CHECK_LIVE_REPORT_STEPS_SUCCEEDED]["status"] == CHECK_STATUS_FAILED
    assert audit["execution_status"] == AUDIT_EXECUTION_STATUS_FAILED


def test_audit_missing_products_fails_without_invoking_validators(tmp_path: Path):
    scratch_root = tmp_path / "live_smoke_tmax_2000"
    scratch_root.mkdir()
    live_report_path = scratch_root / DEFAULT_LIVE_REPORT_NAME
    report = _make_fake_m010_execute_report(scratch_root)
    live_report_path.write_text(json.dumps(report), encoding="utf-8")
    # Intentionally do not stage products.
    audit = run_audit(
        expected_output_root=scratch_root, live_report_path=live_report_path,
    )
    by_id = {c["check_id"]: c for c in audit["audit_checks"]}
    assert by_id[CHECK_PRODUCTS_PRESENT]["status"] == CHECK_STATUS_FAILED
    for cid in (CHECK_DAILY_PRODUCT_SCHEMA, CHECK_TXX_PRODUCT_SCHEMA, CHECK_SU_PRODUCT_SCHEMA):
        assert by_id[cid]["status"] == CHECK_STATUS_FAILED
    # No xarray was invoked: product_validations should be empty.
    assert audit["product_validations"] == []
    assert audit["execution_status"] == AUDIT_EXECUTION_STATUS_FAILED


def test_audit_mismatched_output_root_fails_even_with_passing_validator(tmp_path: Path):
    """An audit run that hashes products under ``expected_root`` but
    reads a live report whose ``output_root`` points at ``report_root``
    must fail, even if the four expected products exist under
    ``expected_root`` and a fake validator returns all-passing results.

    Without the ``live_report_output_root_matches_expected_root`` check,
    audit_passed could be reported with no provenance tie between the
    audited products and the M010 execute run that supposedly wrote
    them.
    """
    expected_root = tmp_path / "live_smoke_tmax_2000"
    expected_root.mkdir()
    report_root = tmp_path / "live_smoke_tmax_2000_other"
    report_root.mkdir()

    # Build a live report whose output_root points at report_root.
    report = _make_fake_m010_execute_report(report_root)
    live_report_path = expected_root / DEFAULT_LIVE_REPORT_NAME
    live_report_path.write_text(json.dumps(report), encoding="utf-8")

    # Stage all four products under expected_root only -- so
    # expected_products_present and the fake validator both happily
    # pass. Without the new check, the audit would be green.
    _stage_scratch_products(expected_root)

    from lib.live_smoke_audit import (
        CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED,
        CheckRecord,
    )

    def passing_fake_validator(*, daily_path, txx_path, su_path):
        return (
            [
                CheckRecord(CHECK_DAILY_PRODUCT_SCHEMA, CHECK_STATUS_PASSED, "ok"),
                CheckRecord(CHECK_TXX_PRODUCT_SCHEMA, CHECK_STATUS_PASSED, "ok"),
                CheckRecord(CHECK_SU_PRODUCT_SCHEMA, CHECK_STATUS_PASSED, "ok"),
            ],
            [{"check_id": "fake/daily", "status": "passed"}],
        )

    audit = run_audit(
        expected_output_root=expected_root,
        live_report_path=live_report_path,
        product_validator=passing_fake_validator,
    )
    by_id = {c["check_id"]: c for c in audit["audit_checks"]}
    # The new check must fail explicitly...
    mismatch = by_id[CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED]
    assert mismatch["status"] == CHECK_STATUS_FAILED
    # ...with a message that names both resolved paths so a human
    # operator can see exactly which roots disagreed.
    assert "does not match" in mismatch["message"]
    # And the overall execution_status must be audit_failed even
    # though every other check (products present, fake validators)
    # would otherwise pass.
    assert audit["execution_status"] == AUDIT_EXECUTION_STATUS_FAILED


def test_audit_matched_output_root_passes(tmp_path: Path):
    """Inverse of the mismatch test: when the live report's output_root
    resolves to the same directory as ``--expected-output-root``, the
    new check should pass and the audit should still pass overall."""
    expected_root = tmp_path / "live_smoke_tmax_2000"
    expected_root.mkdir()
    report = _make_fake_m010_execute_report(expected_root)
    live_report_path = expected_root / DEFAULT_LIVE_REPORT_NAME
    live_report_path.write_text(json.dumps(report), encoding="utf-8")
    _stage_scratch_products(expected_root)

    audit = run_audit(
        expected_output_root=expected_root, live_report_path=live_report_path,
    )
    from lib.live_smoke_audit import CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED
    by_id = {c["check_id"]: c for c in audit["audit_checks"]}
    assert by_id[CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED]["status"] == CHECK_STATUS_PASSED
    assert audit["execution_status"] == AUDIT_EXECUTION_STATUS_PASSED


def test_audit_absolute_vs_relative_output_root_matches(tmp_path: Path):
    """Path.resolve() must treat an absolute scratch path and the same
    path expressed via a different prefix as the same directory.

    Without resolve()-based comparison this would falsely flag a
    mismatch when the operator typed the absolute path on the CLI but
    the M010 execute report recorded the relative form."""
    expected_root = (tmp_path / "live_smoke_tmax_2000").resolve()
    expected_root.mkdir()
    # Live report stores the absolute resolved string; expected is the
    # same directory expressed as a less-canonical (but equivalent) path.
    report = _make_fake_m010_execute_report(expected_root)
    live_report_path = expected_root / DEFAULT_LIVE_REPORT_NAME
    live_report_path.write_text(json.dumps(report), encoding="utf-8")
    _stage_scratch_products(expected_root)

    # Pass the un-resolved tmp_path-relative form on purpose.
    audit = run_audit(
        expected_output_root=tmp_path / "live_smoke_tmax_2000",
        live_report_path=live_report_path,
    )
    from lib.live_smoke_audit import CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED
    by_id = {c["check_id"]: c for c in audit["audit_checks"]}
    assert by_id[CHECK_LIVE_REPORT_OUTPUT_ROOT_MATCHES_EXPECTED]["status"] == CHECK_STATUS_PASSED


def test_audit_canonical_dev_output_root_in_live_report_fails(tmp_path: Path):
    scratch_root = tmp_path / "live_smoke_tmax_2000"
    scratch_root.mkdir()
    live_report_path = scratch_root / DEFAULT_LIVE_REPORT_NAME
    report = _make_fake_m010_execute_report(scratch_root)
    report["output_root"] = CANONICAL_DEV_OUTPUT_ROOT
    live_report_path.write_text(json.dumps(report), encoding="utf-8")
    _stage_scratch_products(scratch_root)
    audit = run_audit(
        expected_output_root=scratch_root, live_report_path=live_report_path,
    )
    from lib.live_smoke_audit import CHECK_LIVE_REPORT_OUTPUT_ROOT_SCRATCH
    by_id = {c["check_id"]: c for c in audit["audit_checks"]}
    assert by_id[CHECK_LIVE_REPORT_OUTPUT_ROOT_SCRATCH]["status"] == CHECK_STATUS_FAILED


def test_audit_with_fake_validator_records_validator_results(tmp_path: Path):
    """Audit can run with an injected validator -- useful for tests that
    do not want to stage real NetCDF files but do want to verify the
    audit pipeline records validator output."""
    scratch_root = tmp_path / "live_smoke_tmax_2000"
    scratch_root.mkdir()
    live_report_path = scratch_root / DEFAULT_LIVE_REPORT_NAME
    report = _make_fake_m010_execute_report(scratch_root)
    live_report_path.write_text(json.dumps(report), encoding="utf-8")
    _stage_scratch_products(scratch_root)

    from lib.live_smoke_audit import CheckRecord

    def fake_validator(*, daily_path, txx_path, su_path):
        return (
            [
                CheckRecord(CHECK_DAILY_PRODUCT_SCHEMA, CHECK_STATUS_PASSED, "ok"),
                CheckRecord(CHECK_TXX_PRODUCT_SCHEMA, CHECK_STATUS_PASSED, "ok"),
                CheckRecord(CHECK_SU_PRODUCT_SCHEMA, CHECK_STATUS_PASSED, "ok"),
            ],
            [{"check_id": "fake/daily", "status": "passed"}],
        )

    audit = run_audit(
        expected_output_root=scratch_root,
        live_report_path=live_report_path,
        product_validator=fake_validator,
    )
    assert audit["execution_status"] == AUDIT_EXECUTION_STATUS_PASSED
    assert audit["product_validations"] == [
        {"check_id": "fake/daily", "status": "passed"},
    ]


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "08_audit_live_smoke.py"
    spec = importlib.util.spec_from_file_location("audit_smoke_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_cli_preflight_against_canonical_exits_zero(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    main = _load_script_main()
    output = tmp_path / "audit_plan.json"
    rc = main(
        [
            "--config", str(CANONICAL_CONFIG),
            "--mode", "preflight",
            "--output", str(output),
            "--expected-output-root", str(tmp_path / "live_smoke_tmax_2000"),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote live-smoke audit" in captured.out
    assert "execution_status=ready_for_live_smoke_audit" in captured.out
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["manifest_type"] == MANIFEST_TYPE_AUDIT
    assert loaded["mode"] == MODE_PREFLIGHT


def test_cli_preflight_rerun_is_byte_identical(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    main = _load_script_main()
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    expected_root = tmp_path / "live_smoke_tmax_2000"
    for target in (out_a, out_b):
        rc = main(
            [
                "--config", str(CANONICAL_CONFIG),
                "--mode", "preflight",
                "--output", str(target),
                "--expected-output-root", str(expected_root),
            ]
        )
        assert rc == 0
    assert out_a.read_bytes() == out_b.read_bytes()


def test_cli_preflight_against_canonical_dev_output_root_exits_1(tmp_path: Path, capsys, monkeypatch):
    """Preflight against runs/dev_region must block (exit 1)."""
    monkeypatch.chdir(REPO_ROOT)
    main = _load_script_main()
    rc = main(
        [
            "--config", str(CANONICAL_CONFIG),
            "--mode", "preflight",
            "--output", str(tmp_path / "audit_plan.json"),
            "--expected-output-root", "runs/dev_region",
        ]
    )
    assert rc == 1


def test_cli_audit_against_synthetic_scratch_exits_zero(tmp_path: Path, capsys):
    scratch_root = tmp_path / "live_smoke_tmax_2000"
    scratch_root.mkdir()
    live_report_path = scratch_root / DEFAULT_LIVE_REPORT_NAME
    report = _make_fake_m010_execute_report(scratch_root)
    live_report_path.write_text(json.dumps(report), encoding="utf-8")
    _stage_scratch_products(scratch_root)
    main = _load_script_main()
    rc = main(
        [
            "--config", str(CANONICAL_CONFIG),
            "--mode", "audit",
            "--output", str(scratch_root / DEFAULT_AUDIT_REPORT_NAME),
            "--expected-output-root", str(scratch_root),
            "--live-report", str(live_report_path),
        ]
    )
    assert rc == 0
    audit = json.loads(
        (scratch_root / DEFAULT_AUDIT_REPORT_NAME).read_text(encoding="utf-8")
    )
    assert audit["execution_status"] == AUDIT_EXECUTION_STATUS_PASSED


# ---------------------------------------------------------------------------
# Import safety + canonical-artifact pin
# ---------------------------------------------------------------------------


def test_live_smoke_audit_module_does_not_eagerly_import_heavy_deps():
    """``import lib.live_smoke_audit`` must not pull in cdsapi / xarray /
    numpy / dask. The audit-mode product validators load them lazily."""
    import inspect

    import lib.live_smoke_audit as m

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


def test_canonical_audit_plan_matches_committed_artifact():
    canonical = json.loads(CANONICAL_AUDIT_PLAN.read_text(encoding="utf-8"))
    assert canonical["manifest_type"] == MANIFEST_TYPE_AUDIT
    assert canonical["mode"] == MODE_PREFLIGHT
    assert canonical["requires_network"] is False
    assert canonical["execution_status"] == PREFLIGHT_EXECUTION_STATUS_READY
    assert canonical["request_id"] == DEFAULT_REQUEST_ID
    assert [c["check_id"] for c in canonical["preflight_checks"]] == list(
        CANONICAL_PREFLIGHT_CHECK_ORDER
    )
    assert all(c["status"] == CHECK_STATUS_PASSED for c in canonical["preflight_checks"])
    assert canonical["audit_checks_planned"] == list(CANONICAL_AUDIT_CHECK_ORDER)


def test_no_nc_files_under_dev_region():
    """M011 must not leave NetCDF files or scratch directories under
    the canonical dev root."""
    dev_root = REPO_ROOT / "runs" / "dev_region"
    nc = list(dev_root.rglob("*.nc")) + list(dev_root.rglob("*.nc4"))
    assert nc == []
    for scratch_name in ("raw", "intermediate", "derived"):
        assert not (dev_root / scratch_name).exists()
