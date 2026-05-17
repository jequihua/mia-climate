"""Tests for ``lib.validation`` and ``scripts/06_validate_outputs.py``.

Manifest-graph checks run against synthetic mini pipelines built in
``tmp_path``. Product-level validators run against synthetic NetCDF
fixtures so the suite never depends on real CDS-acquired data. A
single integration smoke test exercises the real validator against
the canonical M008 pipeline manifest and verifies the report's
top-level shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from lib.validation import (
    CANONICAL_CHECK_ORDER,
    CHECK_DAILY_PRODUCTS,
    CHECK_INDEX_PRODUCTS,
    CHECK_MANIFESTS_DRY_RUN_ONLY,
    CHECK_NO_DERIVED_DIR,
    CHECK_NO_INTERMEDIATE_DIR,
    CHECK_NO_NC_FILES,
    CHECK_NO_RAW_DIR,
    CHECK_PIPELINE_EXECUTION_STATUS,
    CHECK_PIPELINE_MANIFEST_EXISTS,
    CHECK_PIPELINE_MANIFEST_TYPE,
    CHECK_PIPELINE_REQUIRES_NETWORK,
    CHECK_PIPELINE_STEP_COUNT,
    CHECK_PIPELINE_STEP_HASHES_MATCH,
    CHECK_PIPELINE_STEP_ORDER,
    CHECK_PIPELINE_STEP_OUTPUTS_EXIST,
    CHECK_REGION_HASH_CONSISTENT,
    CHECK_REGION_ID_CONSISTENT,
    EXECUTION_STATUS_FAILED,
    EXECUTION_STATUS_PASSED,
    EXECUTION_STATUS_PASSED_WITH_WARNINGS,
    EXPECTED_STEP_ORDER,
    MANIFEST_TYPE_VALIDATION,
    MODE_DRY_RUN,
    PIPELINE_MANIFEST_TYPE,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    STATUS_WARNING,
    ValidationError,
    build_validation_report,
    compute_file_hash,
    derive_execution_status,
    run_validation,
    validate_daily_product,
    validate_index_product,
    write_validation_report,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_PIPELINE_MANIFEST = REPO_ROOT / "runs" / "dev_region" / "pipeline_manifest.json"
CANONICAL_OUTPUT_ROOT = REPO_ROOT / "runs" / "dev_region"
CANONICAL_VALIDATION_REPORT = REPO_ROOT / "runs" / "dev_region" / "validation_report.json"


# ---------------------------------------------------------------------------
# Synthetic manifest pipeline builder
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _make_downstream(
    *,
    manifest_type: str | None,
    region_id: str,
    region_geometry_hash: str,
    mode: str | None,
    requires_network: bool | None = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"region_id": region_id}
    if manifest_type is not None:
        data["manifest_type"] = manifest_type
    # M001 region manifest uses ``geometry_hash``; downstream uses ``region_geometry_hash``.
    if manifest_type is None:
        data["geometry_hash"] = region_geometry_hash
    else:
        data["region_geometry_hash"] = region_geometry_hash
    if mode is not None:
        data["mode"] = mode
    if requires_network is not None:
        data["requires_network"] = requires_network
    if extra:
        data.update(extra)
    return data


def _build_synthetic_pipeline(
    tmp_path: Path,
    *,
    region_id: str = "rbmn",
    region_geometry_hash: str = "sha256:" + "0" * 64,
) -> tuple[Path, dict[str, Any]]:
    """Build a tmp_path/runs/dev_region/ tree with eight valid downstream
    manifests and a pipeline manifest stitching them together. Returns
    (pipeline_path, pipeline_dict)."""
    out_root = tmp_path / "runs" / "dev_region"
    out_root.mkdir(parents=True, exist_ok=True)

    file_specs = [
        ("validate_region", "region_manifest.json", "00_validate_region.py",
         _make_downstream(manifest_type=None, region_id=region_id, region_geometry_hash=region_geometry_hash, mode=None, requires_network=None)),
        ("plan_downloads", "download_manifest.json", "01_plan_downloads.py",
         _make_downstream(manifest_type="era5_land_download_plan", region_id=region_id, region_geometry_hash=region_geometry_hash, mode=None)),
        ("acquire_daily_stats_dry_run", "acquisition_manifest.json", "02_download_era5_land.py",
         _make_downstream(manifest_type="era5_land_acquisition_run", region_id=region_id, region_geometry_hash=region_geometry_hash, mode="dry-run")),
        ("acquire_precipitation_dry_run", "acquisition_manifest_precipitation_dry_run.json", "02_download_era5_land.py",
         _make_downstream(manifest_type="era5_land_acquisition_run", region_id=region_id, region_geometry_hash=region_geometry_hash, mode="dry-run")),
        ("preprocess_daily_stats_dry_run", "preprocessing_manifest.json", "03_preprocess_daily.py",
         _make_downstream(manifest_type="era5_land_preprocessing_run", region_id=region_id, region_geometry_hash=region_geometry_hash, mode="dry-run")),
        ("preprocess_precipitation_dry_run", "preprocessing_manifest_precipitation.json", "03_preprocess_daily.py",
         _make_downstream(manifest_type="era5_land_preprocessing_run", region_id=region_id, region_geometry_hash=region_geometry_hash, mode="dry-run")),
        ("indices_temperature_dry_run", "index_manifest.json", "04_compute_indices.py",
         _make_downstream(manifest_type="era5_land_index_run", region_id=region_id, region_geometry_hash=region_geometry_hash, mode="dry-run")),
        ("indices_precipitation_dry_run", "index_manifest_precipitation.json", "04_compute_indices.py",
         _make_downstream(manifest_type="era5_land_index_run", region_id=region_id, region_geometry_hash=region_geometry_hash, mode="dry-run")),
    ]
    steps = []
    for step_id, filename, script, data in file_specs:
        target = out_root / filename
        _write_json(target, data)
        steps.append({
            "step_id": step_id,
            "script": script,
            "argv": [],
            "output_path": str(target).replace("\\", "/"),
            "exit_code": 0,
            "status": "succeeded",
            "output_hash": compute_file_hash(target),
        })

    pipeline = {
        "manifest_type": PIPELINE_MANIFEST_TYPE,
        "config_path": "configs/rbmn_local.json",
        "config_hash": "sha256:" + "a" * 64,
        "mode": "dry-run",
        "run_id": "dev_region",
        "output_root": str(out_root).replace("\\", "/"),
        "step_count": 8,
        "succeeded_count": 8,
        "failed_count": 0,
        "skipped_count": 0,
        "steps": steps,
        "created_by": "tests",
        "requires_network": False,
        "execution_status": "completed_dry_run",
    }
    pipeline_path = out_root / "pipeline_manifest.json"
    _write_json(pipeline_path, pipeline)
    return pipeline_path, pipeline


# ---------------------------------------------------------------------------
# run_validation: graph-level success
# ---------------------------------------------------------------------------


def test_run_validation_canonical_synthetic_passes(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    out_root = pipeline_path.parent
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=out_root)
    # 17 checks, in canonical order, 15 passed + 2 skipped (daily / index products).
    assert [r.check_id for r in results] == list(CANONICAL_CHECK_ORDER)
    assert sum(1 for r in results if r.status == STATUS_PASSED) == 15
    assert sum(1 for r in results if r.status == STATUS_SKIPPED) == 2
    assert sum(1 for r in results if r.status == STATUS_FAILED) == 0
    assert derive_execution_status(results) == EXECUTION_STATUS_PASSED


def test_run_validation_rejects_unsupported_mode(tmp_path: Path):
    with pytest.raises(ValidationError, match="not one of"):
        run_validation(
            pipeline_manifest_path=tmp_path / "missing.json",
            output_root=tmp_path,
            mode="execute",
        )


# ---------------------------------------------------------------------------
# Manifest-graph failure modes
# ---------------------------------------------------------------------------


def test_missing_pipeline_manifest_does_not_crash(tmp_path: Path):
    results = run_validation(
        pipeline_manifest_path=tmp_path / "does_not_exist.json",
        output_root=tmp_path,
    )
    assert results[0].check_id == CHECK_PIPELINE_MANIFEST_EXISTS
    assert results[0].status == STATUS_FAILED
    # All subsequent checks must be skipped, not failed or absent.
    assert [r.status for r in results[1:]] == [STATUS_SKIPPED] * (len(CANONICAL_CHECK_ORDER) - 1)


def test_tampered_step_output_hash_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    # Tamper with one downstream manifest after the pipeline manifest was built.
    region_path = pipeline_path.parent / "region_manifest.json"
    region_path.write_text(
        region_path.read_text(encoding="utf-8").replace("rbmn", "tampered"),
        encoding="utf-8",
    )
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    hash_check = next(r for r in results if r.check_id == CHECK_PIPELINE_STEP_HASHES_MATCH)
    assert hash_check.status == STATUS_FAILED
    assert "output_hash mismatches" in hash_check.message


def test_missing_step_output_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    (pipeline_path.parent / "index_manifest_precipitation.json").unlink()
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    existence_check = next(r for r in results if r.check_id == CHECK_PIPELINE_STEP_OUTPUTS_EXIST)
    assert existence_check.status == STATUS_FAILED
    assert "missing step outputs" in existence_check.message


def test_region_id_drift_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    # Mutate one downstream manifest to a different region_id, then refresh
    # only that step's output_hash so the hash check still passes (so we can
    # isolate the region-id check).
    bad_path = pipeline_path.parent / "preprocessing_manifest.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["region_id"] = "another_region"
    bad_path.write_text(json.dumps(bad, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    for step in pipeline["steps"]:
        if step["output_path"].endswith("preprocessing_manifest.json"):
            step["output_hash"] = compute_file_hash(bad_path)
    pipeline_path.write_text(json.dumps(pipeline, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    rid = next(r for r in results if r.check_id == CHECK_REGION_ID_CONSISTENT)
    assert rid.status == STATUS_FAILED


def test_pipeline_manifest_type_mismatch_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["manifest_type"] = "wrong_kind"
    pipeline_path.write_text(json.dumps(pipeline, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    mt = next(r for r in results if r.check_id == CHECK_PIPELINE_MANIFEST_TYPE)
    assert mt.status == STATUS_FAILED


def test_pipeline_step_order_drift_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    # Swap two adjacent steps.
    pipeline["steps"][0], pipeline["steps"][1] = pipeline["steps"][1], pipeline["steps"][0]
    pipeline_path.write_text(json.dumps(pipeline, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    order = next(r for r in results if r.check_id == CHECK_PIPELINE_STEP_ORDER)
    assert order.status == STATUS_FAILED


def test_pipeline_requires_network_true_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["requires_network"] = True
    pipeline_path.write_text(json.dumps(pipeline, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    rn = next(r for r in results if r.check_id == CHECK_PIPELINE_REQUIRES_NETWORK)
    assert rn.status == STATUS_FAILED


def test_downstream_non_dry_run_mode_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    bad_path = pipeline_path.parent / "preprocessing_manifest.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["mode"] = "execute"
    bad_path.write_text(json.dumps(bad, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    for step in pipeline["steps"]:
        if step["output_path"].endswith("preprocessing_manifest.json"):
            step["output_hash"] = compute_file_hash(bad_path)
    pipeline_path.write_text(json.dumps(pipeline, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    dr = next(r for r in results if r.check_id == CHECK_MANIFESTS_DRY_RUN_ONLY)
    assert dr.status == STATUS_FAILED


# ---------------------------------------------------------------------------
# Side-effect policy checks
# ---------------------------------------------------------------------------


def test_nc_file_under_output_root_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    (pipeline_path.parent / "stray.nc").write_bytes(b"fake")
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    nc = next(r for r in results if r.check_id == CHECK_NO_NC_FILES)
    assert nc.status == STATUS_FAILED


def test_raw_directory_under_output_root_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    (pipeline_path.parent / "raw").mkdir()
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    raw = next(r for r in results if r.check_id == CHECK_NO_RAW_DIR)
    assert raw.status == STATUS_FAILED


def test_intermediate_directory_under_output_root_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    (pipeline_path.parent / "intermediate").mkdir()
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    inter = next(r for r in results if r.check_id == CHECK_NO_INTERMEDIATE_DIR)
    assert inter.status == STATUS_FAILED


def test_derived_directory_under_output_root_is_detected(tmp_path: Path):
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    (pipeline_path.parent / "derived").mkdir()
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    der = next(r for r in results if r.check_id == CHECK_NO_DERIVED_DIR)
    assert der.status == STATUS_FAILED


# ---------------------------------------------------------------------------
# Daily-product validator (synthetic NetCDF fixtures)
# ---------------------------------------------------------------------------


def _write_daily_tmax(path: Path, *, year: int = 2000, all_nan: bool = False,
                     wrong_units: bool = False, wrong_var: str | None = None,
                     wrong_year: int | None = None) -> None:
    times = np.array(
        [f"{wrong_year or year}-01-15", f"{wrong_year or year}-06-15", f"{wrong_year or year}-12-15"],
        dtype="datetime64[ns]",
    )
    lat = [22.0, 21.5]
    lon = [-105.6, -105.4]
    if all_nan:
        values = np.full((3, 2, 2), np.nan, dtype="float32")
    else:
        values = np.array(
            [[[25.0, 26.0], [24.0, 25.5]], [[35.0, 36.0], [34.0, 35.5]], [[22.0, 23.0], [21.0, 22.5]]],
            dtype="float32",
        )
    var_name = wrong_var or "tmax"
    units = "K" if wrong_units else "degC"
    da = xr.DataArray(
        values, dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
        name=var_name, attrs={"units": units},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(path, engine="h5netcdf")


def test_validate_daily_product_accepts_valid_tmax(tmp_path: Path):
    p = tmp_path / "tmax_2000.nc"
    _write_daily_tmax(p, year=2000)
    res = validate_daily_product(p, project_variable="tmax", expected_year=2000, expected_units="degC")
    assert res.status == STATUS_PASSED


def test_validate_daily_product_rejects_wrong_variable(tmp_path: Path):
    p = tmp_path / "wrong.nc"
    _write_daily_tmax(p, wrong_var="t2m")
    res = validate_daily_product(p, project_variable="tmax", expected_year=2000, expected_units="degC")
    assert res.status == STATUS_FAILED
    assert "tmax" in res.message


def test_validate_daily_product_rejects_wrong_units(tmp_path: Path):
    p = tmp_path / "wrong_units.nc"
    _write_daily_tmax(p, wrong_units=True)
    res = validate_daily_product(p, project_variable="tmax", expected_year=2000, expected_units="degC")
    assert res.status == STATUS_FAILED
    assert "units" in res.message


def test_validate_daily_product_rejects_all_nan(tmp_path: Path):
    p = tmp_path / "all_nan.nc"
    _write_daily_tmax(p, all_nan=True)
    res = validate_daily_product(p, project_variable="tmax", expected_year=2000, expected_units="degC")
    assert res.status == STATUS_FAILED
    assert "NaN" in res.message


def test_validate_daily_product_rejects_wrong_year_coverage(tmp_path: Path):
    p = tmp_path / "wrong_year.nc"
    _write_daily_tmax(p, year=2000, wrong_year=2001)
    res = validate_daily_product(p, project_variable="tmax", expected_year=2000, expected_units="degC")
    assert res.status == STATUS_FAILED
    assert "year coverage" in res.message


def test_validate_daily_product_missing_coord_is_detected(tmp_path: Path):
    p = tmp_path / "no_lon.nc"
    times = np.array(["2000-01-01"], dtype="datetime64[ns]")
    da = xr.DataArray(
        np.array([[1.0, 2.0]], dtype="float32"),
        dims=("time", "lat"),
        coords={"time": times, "lat": [22.0, 21.5]},
        name="tmax", attrs={"units": "degC"},
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(p, engine="h5netcdf")
    res = validate_daily_product(p, project_variable="tmax", expected_year=2000, expected_units="degC")
    assert res.status == STATUS_FAILED
    assert "lon" in res.message


def test_validate_daily_product_pr_units_are_mm_per_day(tmp_path: Path):
    p = tmp_path / "pr_2000.nc"
    times = np.array(
        ["2000-01-15", "2000-07-15", "2000-12-15"], dtype="datetime64[ns]"
    )
    da = xr.DataArray(
        np.array([[[0.1, 5.0], [12.0, 25.0]], [[0.0, 1.0], [3.0, 8.0]], [[0.5, 2.0], [4.0, 9.0]]], dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [22.0, 21.5], "lon": [-105.6, -105.4]},
        name="pr", attrs={"units": "mm/day"},
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(p, engine="h5netcdf")
    res = validate_daily_product(p, project_variable="pr", expected_year=2000, expected_units="mm/day")
    assert res.status == STATUS_PASSED


def test_validate_daily_product_warns_on_all_zero(tmp_path: Path):
    p = tmp_path / "zero.nc"
    times = np.array(["2000-01-15", "2000-07-15"], dtype="datetime64[ns]")
    da = xr.DataArray(
        np.zeros((2, 2, 2), dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [22.0, 21.5], "lon": [-105.6, -105.4]},
        name="pr", attrs={"units": "mm/day"},
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(p, engine="h5netcdf")
    res = validate_daily_product(p, project_variable="pr", expected_year=2000, expected_units="mm/day")
    assert res.status == STATUS_WARNING
    assert "zero" in res.message


# ---------------------------------------------------------------------------
# Index-product validator
# ---------------------------------------------------------------------------


def _write_annual_index(path: Path, *, index_id: str, units: str, values: list[float],
                        all_nan: bool = False) -> None:
    n = len(values)
    times = np.array([f"{2000 + i}-12-31" for i in range(n)], dtype="datetime64[ns]")
    arr = (np.full(n, np.nan, dtype="float32") if all_nan else np.array(values, dtype="float32"))
    da = xr.DataArray(arr, dims=("time",), coords={"time": times}, name=index_id, attrs={"units": units})
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(path, engine="h5netcdf")


def test_validate_index_product_accepts_valid_TXx(tmp_path: Path):
    p = tmp_path / "TXx.nc"
    _write_annual_index(p, index_id="TXx", units="degC", values=[35.0, 36.5, 34.2])
    res = validate_index_product(p, index_id="TXx", expected_units="degC")
    assert res.status == STATUS_PASSED


def test_validate_index_product_rejects_wrong_index_id(tmp_path: Path):
    p = tmp_path / "wrong.nc"
    _write_annual_index(p, index_id="other", units="degC", values=[1.0])
    res = validate_index_product(p, index_id="TXx", expected_units="degC")
    assert res.status == STATUS_FAILED


def test_validate_index_product_rejects_all_nan(tmp_path: Path):
    p = tmp_path / "nan.nc"
    _write_annual_index(p, index_id="TXx", units="degC", values=[0.0], all_nan=True)
    res = validate_index_product(p, index_id="TXx", expected_units="degC")
    assert res.status == STATUS_FAILED
    assert "all-NaN" in res.message


def test_validate_index_product_rejects_wrong_units(tmp_path: Path):
    p = tmp_path / "wrong_units.nc"
    _write_annual_index(p, index_id="TXx", units="K", values=[300.0])
    res = validate_index_product(p, index_id="TXx", expected_units="degC")
    assert res.status == STATUS_FAILED
    assert "units" in res.message


# ---------------------------------------------------------------------------
# Report writing + determinism
# ---------------------------------------------------------------------------


def test_build_and_write_validation_report_is_deterministic(tmp_path: Path):
    pipeline_path, pipeline = _build_synthetic_pipeline(tmp_path)
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    report = build_validation_report(
        pipeline_manifest=pipeline,
        pipeline_manifest_path=pipeline_path,
        pipeline_manifest_hash=compute_file_hash(pipeline_path),
        mode=MODE_DRY_RUN,
        output_root=pipeline_path.parent,
        results=results,
        created_by="tests",
    )
    assert report["manifest_type"] == MANIFEST_TYPE_VALIDATION
    assert report["mode"] == MODE_DRY_RUN
    assert report["requires_network"] is False
    assert report["execution_status"] == EXECUTION_STATUS_PASSED
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_validation_report(a, report)
    write_validation_report(b, report)
    assert a.read_bytes() == b.read_bytes()


def test_failed_check_flips_execution_status(tmp_path: Path):
    pipeline_path, pipeline = _build_synthetic_pipeline(tmp_path)
    (pipeline_path.parent / "raw").mkdir()
    results = run_validation(pipeline_manifest_path=pipeline_path, output_root=pipeline_path.parent)
    report = build_validation_report(
        pipeline_manifest=pipeline,
        pipeline_manifest_path=pipeline_path,
        pipeline_manifest_hash=compute_file_hash(pipeline_path),
        mode=MODE_DRY_RUN,
        output_root=pipeline_path.parent,
        results=results,
        created_by="tests",
    )
    assert report["execution_status"] == EXECUTION_STATUS_FAILED
    assert report["failed_count"] >= 1


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_validation_module_does_not_eagerly_import_heavy_deps():
    import inspect

    import lib.validation as m

    src = inspect.getsource(m)
    for line in src.splitlines():
        stripped = line.lstrip()
        if line == stripped:
            for forbidden in (
                "import xarray", "from xarray",
                "import numpy", "from numpy",
                "import dask", "from dask",
                "import geopandas", "from geopandas",
                "import rioxarray", "from rioxarray",
                "import icclim", "from icclim",
                "import xclim", "from xclim",
            ):
                assert not stripped.startswith(forbidden), f"eager import: {line!r}"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "06_validate_outputs.py"
    spec = importlib.util.spec_from_file_location("validate_outputs_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_script_main_canonical_dry_run_exits_zero(tmp_path: Path, capsys):
    main = _load_script_main()
    output = tmp_path / "validation_report.json"
    rc = main(
        [
            "--pipeline-manifest", str(CANONICAL_PIPELINE_MANIFEST),
            "--output", str(output),
            "--output-root", str(CANONICAL_OUTPUT_ROOT),
            "--mode", "dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote validation report" in captured.out
    assert "execution_status: passed" in captured.out


def test_script_main_exits_2_when_pipeline_argument_path_passed_but_unreadable(tmp_path: Path):
    main = _load_script_main()
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    rc = main(
        [
            "--pipeline-manifest", str(bad),
            "--output", str(tmp_path / "report.json"),
            "--output-root", str(tmp_path),
            "--mode", "dry-run",
        ]
    )
    assert rc == 2


def test_script_main_exits_1_when_a_check_fails(tmp_path: Path):
    main = _load_script_main()
    # Build a synthetic pipeline that fails a side-effect check (intermediate/ exists).
    pipeline_path, _ = _build_synthetic_pipeline(tmp_path)
    (pipeline_path.parent / "intermediate").mkdir()
    rc = main(
        [
            "--pipeline-manifest", str(pipeline_path),
            "--output", str(tmp_path / "report.json"),
            "--output-root", str(pipeline_path.parent),
            "--mode", "dry-run",
        ]
    )
    assert rc == 1


def test_canonical_validation_report_matches_canonical_artifact():
    """The committed canonical validation report must reflect what the
    validator produces against the canonical pipeline manifest."""
    canonical = json.loads(CANONICAL_VALIDATION_REPORT.read_text(encoding="utf-8"))
    assert canonical["manifest_type"] == MANIFEST_TYPE_VALIDATION
    assert canonical["mode"] == MODE_DRY_RUN
    assert canonical["execution_status"] == EXECUTION_STATUS_PASSED
    assert canonical["requires_network"] is False
    assert canonical["check_count"] == len(CANONICAL_CHECK_ORDER)
    assert canonical["failed_count"] == 0
