"""Tests for ``lib.index_manifest`` and ``scripts/04_compute_indices.py``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from lib.index_manifest import (
    EXECUTION_STATUS_COMPLETE_EXISTING,
    EXECUTION_STATUS_COMPUTED,
    EXECUTION_STATUS_FAILED,
    EXECUTION_STATUS_PLANNED,
    MANIFEST_TYPE_INDEX,
    MANIFEST_TYPE_PREPROCESSING,
    MODE_DRY_RUN,
    MODE_EXECUTE,
    REASON_REQUIRED_VARIABLE_NOT_IN_PLAN,
    STATUS_COMPUTED,
    STATUS_FAILED,
    STATUS_MISSING_INPUT,
    STATUS_PLANNED,
    STATUS_SKIPPED,
    TEMPERATURE_INDEX_SPECS,
    IndexManifestError,
    IndexResult,
    build_index_manifest,
    compute_manifest_hash,
    derive_execution_status,
    execute_index_results,
    group_preprocessing_results_by_variable,
    index_output_path,
    load_preprocessing_manifest,
    plan_index_results,
    select_index_specs,
    write_index_manifest,
)
from lib.indices_temperature import TemperatureIndexSpec


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_PREPROCESSING_MANIFEST = (
    REPO_ROOT / "runs" / "dev_region" / "preprocessing_manifest.json"
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _stub_preprocessing_result(
    project_variable: str,
    year: int,
    *,
    status: str = "planned",
    output_path: str | None = None,
) -> dict[str, Any]:
    op = (
        output_path
        if output_path is not None
        else f"intermediate/daily/{project_variable}/{year}.nc"
    )
    return {
        "request_id": f"era5_daily_stats__{project_variable}__{year}",
        "project_variable": project_variable,
        "year": year,
        "source_path": f"raw/era5_land/daily_statistics/{project_variable}/{year}.nc",
        "output_path": op,
        "status": status,
    }


def _stub_preprocessing_manifest(*results: dict[str, Any], region_id: str = "rbmn") -> dict[str, Any]:
    return {
        "manifest_type": MANIFEST_TYPE_PREPROCESSING,
        "acquisition_manifest_path": "runs/dev_region/acquisition_manifest.json",
        "acquisition_manifest_hash": "sha256:" + "a" * 64,
        "download_manifest_path": "runs/dev_region/download_manifest.json",
        "download_manifest_hash": "sha256:" + "b" * 64,
        "region_manifest_path": "runs/dev_region/region_manifest.json",
        "region_manifest_hash": "sha256:" + "c" * 64,
        "region_id": region_id,
        "region_geometry_hash": "sha256:" + "0" * 64,
        "mode": "execute",
        "output_root": "runs/dev_region",
        "request_count": len(results),
        "planned_count": 0,
        "preprocessed_count": len([r for r in results if r["status"] == "preprocessed"]),
        "skipped_count": 0,
        "failed_count": 0,
        "missing_input_count": 0,
        "deferred_count": 0,
        "results": list(results),
        "created_by": "tests",
        "requires_network": False,
        "execution_status": "preprocessed",
    }


def _write_daily_dataset(path: Path, *, project_variable: str, year: int, value_offset: float = 0.0) -> None:
    """Write a tiny 1D-in-time daily-product NetCDF for synthetic testing."""
    times = np.array(
        [
            f"{year}-01-15",
            f"{year}-07-15",
            f"{year}-12-15",
        ],
        dtype="datetime64[ns]",
    )
    # Stable, distinct values per (variable, year): vary slightly so each index
    # has a non-trivial annual reduction.
    base = {
        "tmax": [25.0, 35.0, 22.0],
        "tmin": [18.0, 24.0, 12.0],
        "tmean": [21.0, 29.0, 17.0],
    }[project_variable]
    values = np.array([v + value_offset for v in base], dtype="float32")
    da = xr.DataArray(
        values,
        dims=("time",),
        coords={"time": times},
        name=project_variable,
        attrs={"units": "degC"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(path, engine="h5netcdf")


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def test_load_preprocessing_manifest_rejects_missing(tmp_path: Path):
    with pytest.raises(IndexManifestError, match="not found"):
        load_preprocessing_manifest(tmp_path / "missing.json")


def test_load_preprocessing_manifest_rejects_wrong_type(tmp_path: Path):
    bad = _stub_preprocessing_manifest(_stub_preprocessing_result("tmax", 2000))
    bad["manifest_type"] = "something_else"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(IndexManifestError, match="manifest_type"):
        load_preprocessing_manifest(p)


def test_load_preprocessing_manifest_roundtrips_canonical():
    data = load_preprocessing_manifest(CANONICAL_PREPROCESSING_MANIFEST)
    assert data["manifest_type"] == MANIFEST_TYPE_PREPROCESSING
    assert data["region_id"] == "rbmn"


def test_compute_manifest_hash_is_deterministic(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text("hi", encoding="utf-8")
    assert compute_manifest_hash(p) == compute_manifest_hash(p)
    assert compute_manifest_hash(p).startswith("sha256:")


# ---------------------------------------------------------------------------
# Grouping + selection
# ---------------------------------------------------------------------------


def test_group_preprocessing_results_by_variable_sorts_by_year():
    manifest = _stub_preprocessing_manifest(
        _stub_preprocessing_result("tmax", 2001),
        _stub_preprocessing_result("tmax", 2000),
        _stub_preprocessing_result("tmin", 2000),
    )
    grouped = group_preprocessing_results_by_variable(manifest)
    assert sorted(grouped) == ["tmax", "tmin"]
    assert [r["year"] for r in grouped["tmax"]] == [2000, 2001]


def test_group_preprocessing_results_require_executed_filters_planned():
    manifest = _stub_preprocessing_manifest(
        _stub_preprocessing_result("tmax", 2000, status="planned"),
        _stub_preprocessing_result("tmax", 2001, status="preprocessed"),
    )
    grouped = group_preprocessing_results_by_variable(manifest, require_executed=True)
    assert [r["year"] for r in grouped["tmax"]] == [2001]


def test_select_index_specs_filters_and_limits():
    by_id = select_index_specs(TEMPERATURE_INDEX_SPECS, index_ids=["TXx", "SU"])
    assert sorted(s.index_id for s in by_id) == ["SU", "TXx"]
    limited = select_index_specs(TEMPERATURE_INDEX_SPECS, limit=3)
    assert len(limited) == 3
    with pytest.raises(IndexManifestError, match="positive int"):
        select_index_specs(TEMPERATURE_INDEX_SPECS, limit=0)
    with pytest.raises(IndexManifestError, match="unknown index_id"):
        select_index_specs(TEMPERATURE_INDEX_SPECS, index_ids=["ZZZ"])


def test_index_output_path_format():
    assert index_output_path("TXx") == "derived/indices/TXx.nc"


# ---------------------------------------------------------------------------
# Dry-run planning
# ---------------------------------------------------------------------------


def test_plan_index_results_records_missing_variable_reason(tmp_path: Path):
    manifest = _stub_preprocessing_manifest(_stub_preprocessing_result("tmax", 2000))
    results = plan_index_results(
        list(TEMPERATURE_INDEX_SPECS),
        preprocessing=manifest,
        output_root=tmp_path,
    )
    by_id = {r.index_id: r for r in results}
    assert by_id["TXx"].status == STATUS_PLANNED
    assert by_id["TXx"].reason is None
    assert by_id["Tmx"].status == STATUS_PLANNED
    assert by_id["Tmx"].reason and REASON_REQUIRED_VARIABLE_NOT_IN_PLAN in by_id["Tmx"].reason
    assert "tmean" in by_id["Tmx"].reason


def test_plan_index_results_includes_all_source_paths(tmp_path: Path):
    manifest = _stub_preprocessing_manifest(
        _stub_preprocessing_result("tmax", 2000),
        _stub_preprocessing_result("tmax", 2001),
    )
    results = plan_index_results(
        list(TEMPERATURE_INDEX_SPECS),
        preprocessing=manifest,
        output_root=tmp_path,
    )
    txx = {r.index_id: r for r in results}["TXx"]
    assert "intermediate/daily/tmax/2000.nc" in txx.source_paths
    assert "intermediate/daily/tmax/2001.nc" in txx.source_paths


# ---------------------------------------------------------------------------
# Execute mode
# ---------------------------------------------------------------------------


def test_execute_index_results_writes_TXx_netcdf(tmp_path: Path):
    output_root = tmp_path / "run"
    daily_path_a = output_root / "intermediate" / "daily" / "tmax" / "2000.nc"
    daily_path_b = output_root / "intermediate" / "daily" / "tmax" / "2001.nc"
    _write_daily_dataset(daily_path_a, project_variable="tmax", year=2000)
    _write_daily_dataset(daily_path_b, project_variable="tmax", year=2001, value_offset=1.0)
    manifest = _stub_preprocessing_manifest(
        _stub_preprocessing_result(
            "tmax", 2000, status="preprocessed",
            output_path=str(daily_path_a).replace("\\", "/"),
        ),
        _stub_preprocessing_result(
            "tmax", 2001, status="preprocessed",
            output_path=str(daily_path_b).replace("\\", "/"),
        ),
    )
    txx_spec = [s for s in TEMPERATURE_INDEX_SPECS if s.index_id == "TXx"]
    results = execute_index_results(
        txx_spec,
        preprocessing=manifest,
        output_root=output_root,
    )
    assert results[0].status == STATUS_COMPUTED
    target = output_root / "derived" / "indices" / "TXx.nc"
    assert target.exists()
    with xr.open_dataset(target) as ds_out:
        ds_out.load()
    assert "TXx" in ds_out.data_vars
    # 2000 tmax peak = 35.0; 2001 tmax peak = 36.0 (value_offset=1).
    assert float(ds_out["TXx"].isel(time=0)) == pytest.approx(35.0)
    assert float(ds_out["TXx"].isel(time=1)) == pytest.approx(36.0)


def test_execute_index_results_reports_missing_input_for_unpreprocessed_variable(tmp_path: Path):
    # tmin is required for TNn but only tmax is preprocessed.
    output_root = tmp_path / "run"
    daily_path = output_root / "intermediate" / "daily" / "tmax" / "2000.nc"
    _write_daily_dataset(daily_path, project_variable="tmax", year=2000)
    manifest = _stub_preprocessing_manifest(
        _stub_preprocessing_result(
            "tmax", 2000, status="preprocessed",
            output_path=str(daily_path).replace("\\", "/"),
        )
    )
    tnn_spec = [s for s in TEMPERATURE_INDEX_SPECS if s.index_id == "TNn"]
    results = execute_index_results(
        tnn_spec,
        preprocessing=manifest,
        output_root=output_root,
    )
    assert results[0].status == STATUS_MISSING_INPUT
    assert "tmin" in (results[0].reason or "")


def test_execute_index_results_skips_existing_output_unless_overwrite(tmp_path: Path):
    output_root = tmp_path / "run"
    daily_path = output_root / "intermediate" / "daily" / "tmax" / "2000.nc"
    _write_daily_dataset(daily_path, project_variable="tmax", year=2000)
    manifest = _stub_preprocessing_manifest(
        _stub_preprocessing_result(
            "tmax", 2000, status="preprocessed",
            output_path=str(daily_path).replace("\\", "/"),
        )
    )
    target = output_root / "derived" / "indices" / "TXx.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"prior")
    txx_spec = [s for s in TEMPERATURE_INDEX_SPECS if s.index_id == "TXx"]
    results = execute_index_results(txx_spec, preprocessing=manifest, output_root=output_root)
    assert results[0].status == STATUS_SKIPPED
    assert target.read_bytes() == b"prior"
    results_ow = execute_index_results(
        txx_spec, preprocessing=manifest, output_root=output_root, overwrite=True
    )
    assert results_ow[0].status == STATUS_COMPUTED
    assert target.read_bytes() != b"prior"


def test_execute_index_results_records_failure_on_corrupt_input(tmp_path: Path):
    output_root = tmp_path / "run"
    daily_path = output_root / "intermediate" / "daily" / "tmax" / "2000.nc"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily_path.write_bytes(b"not a netcdf file")
    manifest = _stub_preprocessing_manifest(
        _stub_preprocessing_result(
            "tmax", 2000, status="preprocessed",
            output_path=str(daily_path).replace("\\", "/"),
        )
    )
    txx_spec = [s for s in TEMPERATURE_INDEX_SPECS if s.index_id == "TXx"]
    results = execute_index_results(txx_spec, preprocessing=manifest, output_root=output_root)
    assert results[0].status == STATUS_FAILED
    assert results[0].error


# ---------------------------------------------------------------------------
# Manifest build / write
# ---------------------------------------------------------------------------


def test_build_index_manifest_dry_run_counts(tmp_path: Path):
    manifest = _stub_preprocessing_manifest(
        _stub_preprocessing_result("tmax", 2000),
    )
    results = plan_index_results(
        list(TEMPERATURE_INDEX_SPECS),
        preprocessing=manifest,
        output_root=tmp_path,
    )
    out = build_index_manifest(
        preprocessing_manifest=manifest,
        preprocessing_manifest_path=tmp_path / "p.json",
        preprocessing_manifest_hash="sha256:" + "f" * 64,
        mode=MODE_DRY_RUN,
        output_root=tmp_path,
        results=results,
        created_by="tests",
    )
    assert out["manifest_type"] == MANIFEST_TYPE_INDEX
    assert out["mode"] == MODE_DRY_RUN
    assert out["requires_network"] is False
    assert out["execution_status"] == EXECUTION_STATUS_PLANNED
    assert out["index_count"] == len(TEMPERATURE_INDEX_SPECS)
    assert out["planned_count"] == len(TEMPERATURE_INDEX_SPECS)


def test_write_index_manifest_is_deterministic(tmp_path: Path):
    manifest = _stub_preprocessing_manifest(_stub_preprocessing_result("tmax", 2000))
    results = plan_index_results(
        list(TEMPERATURE_INDEX_SPECS),
        preprocessing=manifest,
        output_root=tmp_path,
    )
    out = build_index_manifest(
        preprocessing_manifest=manifest,
        preprocessing_manifest_path=tmp_path / "p.json",
        preprocessing_manifest_hash="sha256:" + "f" * 64,
        mode=MODE_DRY_RUN,
        output_root=tmp_path,
        results=results,
        created_by="tests",
    )
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_index_manifest(a, out)
    write_index_manifest(b, out)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_text(encoding="utf-8").endswith("\n")


def test_derive_execution_status_branches():
    def _r(status):
        return IndexResult(
            index_id="X",
            required_variables=("tmax",),
            source_paths=(),
            output_path="y",
            status=status,
        )

    assert derive_execution_status(MODE_DRY_RUN, []) == EXECUTION_STATUS_PLANNED
    assert derive_execution_status(MODE_EXECUTE, [_r(STATUS_COMPUTED)]) == EXECUTION_STATUS_COMPUTED
    assert derive_execution_status(MODE_EXECUTE, [_r(STATUS_SKIPPED)]) == EXECUTION_STATUS_COMPLETE_EXISTING
    assert derive_execution_status(MODE_EXECUTE, [_r(STATUS_FAILED)]) == EXECUTION_STATUS_FAILED
    assert derive_execution_status(MODE_EXECUTE, [_r(STATUS_MISSING_INPUT)]) == EXECUTION_STATUS_FAILED
    assert derive_execution_status(MODE_EXECUTE, [_r(STATUS_COMPUTED), _r(STATUS_FAILED)]) == "partial"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "04_compute_indices.py"
    spec = importlib.util.spec_from_file_location("compute_indices_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_script_main_dry_run_on_canonical_preprocessing_manifest(tmp_path: Path, capsys):
    main = _load_script_main()
    output = tmp_path / "index_manifest.json"
    rc = main(
        [
            "--preprocessing-manifest",
            str(CANONICAL_PREPROCESSING_MANIFEST),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "run_root"),
            "--mode",
            "dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote index manifest" in captured.out
    assert "mode=dry-run" in captured.out
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["manifest_type"] == MANIFEST_TYPE_INDEX
    assert loaded["mode"] == MODE_DRY_RUN
    assert loaded["requires_network"] is False
    assert loaded["index_count"] == len(TEMPERATURE_INDEX_SPECS)
    assert loaded["execution_status"] == EXECUTION_STATUS_PLANNED
    # No derived NetCDF in dry-run.
    derived_dir = tmp_path / "run_root" / "derived"
    if derived_dir.exists():
        assert not list(derived_dir.rglob("*.nc"))


def test_script_main_returns_2_for_missing_preprocessing_manifest(tmp_path: Path, capsys):
    main = _load_script_main()
    rc = main(
        [
            "--preprocessing-manifest",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "out.json"),
            "--output-root",
            str(tmp_path),
            "--mode",
            "dry-run",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "index computation failed" in captured.err


def test_script_main_filter_by_index_id(tmp_path: Path, capsys):
    main = _load_script_main()
    output = tmp_path / "im.json"
    rc = main(
        [
            "--preprocessing-manifest",
            str(CANONICAL_PREPROCESSING_MANIFEST),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "rr"),
            "--mode",
            "dry-run",
            "--index-id",
            "TXx",
            "--index-id",
            "SU",
        ]
    )
    assert rc == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["index_count"] == 2
    assert {r["index_id"] for r in loaded["results"]} == {"TXx", "SU"}


def test_script_main_execute_against_synthetic_fixture(tmp_path: Path, capsys):
    main = _load_script_main()
    output_root = tmp_path / "run"
    daily_path = output_root / "intermediate" / "daily" / "tmax" / "2000.nc"
    _write_daily_dataset(daily_path, project_variable="tmax", year=2000)
    manifest = _stub_preprocessing_manifest(
        _stub_preprocessing_result(
            "tmax", 2000, status="preprocessed",
            output_path=str(daily_path).replace("\\", "/"),
        )
    )
    manifest_path = tmp_path / "preprocessing.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    output = tmp_path / "index.json"
    rc = main(
        [
            "--preprocessing-manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--output-root",
            str(output_root),
            "--mode",
            "execute",
            "--index-id",
            "TXx",
        ]
    )
    assert rc == 0
    derived = output_root / "derived" / "indices" / "TXx.nc"
    assert derived.exists()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["mode"] == MODE_EXECUTE
    assert loaded["computed_count"] == 1


def test_index_manifest_module_does_not_eagerly_import_heavy_deps():
    """``import lib.index_manifest`` must not pull in numpy/xarray at module load."""
    import sys

    # If a prior test has already loaded these, this assertion is informational
    # only. We at least make sure the module file itself doesn't `import xarray`
    # at module scope.
    import lib.index_manifest as m
    import inspect

    src = inspect.getsource(m)
    # No top-level `import xarray` / `import numpy`. Heavy imports must be
    # inside function bodies, indented.
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import xarray") or stripped.startswith("from xarray"):
            assert line != stripped, f"top-level import detected: {line!r}"
        if stripped.startswith("import numpy") or stripped.startswith("from numpy"):
            assert line != stripped, f"top-level import detected: {line!r}"
