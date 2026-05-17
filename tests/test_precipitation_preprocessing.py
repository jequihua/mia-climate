"""Tests for ``lib.precipitation`` and the precipitation path in
``lib.preprocessing`` / ``scripts/03_preprocess_daily.py``.

Synthetic NetCDF fixtures keep the suite offline. No CDS, no
committed `.nc` files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from lib.precipitation import (
    DAILY_UNITS_MM,
    LEGACY_UTC_OFFSET_HOURS,
    METERS_TO_MILLIMETERS,
    PR_PROJECT_VARIABLE,
    PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7,
    SUPPORTED_PRECIPITATION_POLICIES,
    PrecipitationError,
    aggregate_to_daily_mm,
    apply_utc_offset,
    deaccumulate_hourly_tp,
    preprocess_precipitation_dataset,
    rename_to_pr,
)
from lib.preprocessing import (
    MODE_DRY_RUN,
    MODE_EXECUTE,
    REASON_PRECIPITATION_OPEN,
    REASON_SOURCE_NOT_FOUND,
    REASON_UNSUPPORTED_PRECIPITATION_POLICY,
    STATUS_DEFERRED,
    STATUS_MISSING_INPUT,
    STATUS_PLANNED,
    STATUS_PREPROCESSED,
    execute_results,
    plan_results,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RBMN_GEOJSON = REPO_ROOT / "01_data" / "case_studies" / "rbmn.geojson"


# ---------------------------------------------------------------------------
# Synthetic NetCDF helpers
# ---------------------------------------------------------------------------


def _hourly_accumulated_tp_dataset(
    *,
    year: int = 2000,
    hours: int = 24,
    increment_mm_per_hour: float = 1.0,
    source_name: str = "tp",
    time_dim: str = "valid_time",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    lat_values: list[float] | None = None,
    lon_values: list[float] | None = None,
) -> xr.Dataset:
    """Build a tiny hourly-accumulated tp series in meters.

    Each hour ``h`` (0-indexed) carries the cumulative precipitation in
    meters since the start of the synthetic day, i.e.
    ``tp[h] = h * (increment_mm_per_hour / 1000)``.
    """
    lat = lat_values or [22.0, 21.5]
    lon = lon_values or [-105.6, -105.4]
    times = np.array(
        [f"{year}-01-01T{h:02d}:00:00" for h in range(hours)],
        dtype="datetime64[ns]",
    )
    step_m = increment_mm_per_hour / METERS_TO_MILLIMETERS
    accum = np.arange(hours, dtype="float32") * step_m
    grid_size = len(lat) * len(lon)
    values = np.tile(accum[:, None, None], (1, len(lat), len(lon)))
    da = xr.DataArray(
        values,
        dims=(time_dim, lat_dim, lon_dim),
        coords={time_dim: times, lat_dim: lat, lon_dim: lon},
        name=source_name,
        attrs={"units": "m"},
    )
    return da.to_dataset()


def _write_nc(path: Path, dataset: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path, engine="h5netcdf")


def _square_polygon_geojson(west: float, south: float, east: float, north: float) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [west, south],
                            [east, south],
                            [east, north],
                            [west, north],
                            [west, south],
                        ]
                    ],
                },
            }
        ],
    }


def _stub_pr_request(chunk_id: str, *, year: int = 2000) -> dict[str, Any]:
    months_h1 = ["01", "02", "03", "04", "05", "06"]
    months_h2 = ["07", "08", "09", "10", "11", "12"]
    months = months_h1 if chunk_id == "H1" else months_h2
    return {
        "request_id": f"era5_hourly_pr__{year}_{chunk_id}",
        "request_kind": "hourly_precipitation",
        "dataset": "reanalysis-era5-land",
        "project_variables": ["pr"],
        "cds_variables": ["total_precipitation"],
        "year": year,
        "chunk_id": chunk_id,
        "output_path": f"raw/era5_land/hourly_precipitation/{year}_{chunk_id}.nc",
        "payload": {"variable": ["total_precipitation"], "year": str(year), "month": months},
    }


def _stub_pr_acquisition_result(chunk_id: str, *, year: int = 2000, target_path: str, status: str = "downloaded") -> dict[str, Any]:
    return {
        "request_id": f"era5_hourly_pr__{year}_{chunk_id}",
        "dataset": "reanalysis-era5-land",
        "output_path": f"raw/era5_land/hourly_precipitation/{year}_{chunk_id}.nc",
        "target_path": target_path,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Unit-level pipeline helpers
# ---------------------------------------------------------------------------


def test_rename_to_pr_infers_tp():
    ds = _hourly_accumulated_tp_dataset(source_name="tp")
    out = rename_to_pr(ds)
    assert PR_PROJECT_VARIABLE in out.data_vars
    assert "tp" not in out.data_vars


def test_rename_to_pr_infers_total_precipitation():
    ds = _hourly_accumulated_tp_dataset(source_name="total_precipitation")
    out = rename_to_pr(ds)
    assert PR_PROJECT_VARIABLE in out.data_vars
    assert "total_precipitation" not in out.data_vars


def test_rename_to_pr_rejects_missing_explicit_source():
    ds = _hourly_accumulated_tp_dataset(source_name="tp")
    with pytest.raises(PrecipitationError, match="not found"):
        rename_to_pr(ds, source_variable="something_else")


def test_apply_utc_offset_shifts_time_by_minus_seven_hours():
    ds = _hourly_accumulated_tp_dataset(time_dim="time")
    original_times = ds["time"].values.copy()
    shifted = apply_utc_offset(ds, offset_hours=LEGACY_UTC_OFFSET_HOURS)
    delta = shifted["time"].values - original_times
    assert (delta == np.timedelta64(LEGACY_UTC_OFFSET_HOURS, "h")).all()


def test_deaccumulate_hourly_tp_diff_with_clamp():
    """24h accumulated, 1 mm/h step. Diff yields 23 increments of 1mm."""
    ds = _hourly_accumulated_tp_dataset(time_dim="time")
    increments = deaccumulate_hourly_tp(ds["tp"])
    assert increments.sizes["time"] == 23
    expected_m = 1.0 / METERS_TO_MILLIMETERS
    assert float(increments.isel(time=0, latitude=0, longitude=0)) == pytest.approx(expected_m)


def test_deaccumulate_hourly_tp_clamps_negatives():
    """A synthetic reset where tp drops produces a clamped 0 increment."""
    times = np.array(
        ["2000-01-01T00:00", "2000-01-01T01:00", "2000-01-01T02:00", "2000-01-01T03:00"],
        dtype="datetime64[ns]",
    )
    # Accumulated, then a reset between hour 02 and 03 (value drops to 0).
    values = np.array([0.0, 0.001, 0.002, 0.0], dtype="float32")
    da = xr.DataArray(values, dims=("time",), coords={"time": times}, name="tp")
    incs = deaccumulate_hourly_tp(da)
    assert list(incs.values) == [pytest.approx(0.001), pytest.approx(0.001), pytest.approx(0.0)]


def test_aggregate_to_daily_mm_converts_meters_to_mm_per_day():
    """A constant 1mm/hour stream should produce one daily total of 24 mm."""
    times = np.array(
        [f"2000-01-01T{h:02d}:00" for h in range(24)], dtype="datetime64[ns]"
    )
    incs_m = xr.DataArray(
        np.full(24, 1.0 / METERS_TO_MILLIMETERS, dtype="float32"),
        dims=("time",),
        coords={"time": times},
        name="pr",
    )
    daily = aggregate_to_daily_mm(incs_m)
    assert daily.attrs["units"] == DAILY_UNITS_MM
    assert daily.attrs["preprocessing_conversion_factor"] == METERS_TO_MILLIMETERS
    assert daily.attrs["source_units"] == "m"
    assert float(daily.values[0]) == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# End-to-end pipeline on a one-UTC-day fixture
# ---------------------------------------------------------------------------


def test_preprocess_precipitation_dataset_legacy_utc_minus_7_splits_one_utc_day_into_two_local_days():
    """
    With 24h accumulated UTC data (1mm/h) and a -7h shift:
      - 23 hourly diffs of 1mm each (hour 0 is lost to diff).
      - Shifted: UTC hours 1..23 -> local hours -6..16 (= prev day 18..23 and same day 0..16).
      - Daily totals: 6 mm on the previous local day, 17 mm on the same local day.
    """
    ds = _hourly_accumulated_tp_dataset(year=2000, hours=24, increment_mm_per_hour=1.0)
    polygon = _square_polygon_geojson(-105.8, 21.4, -105.3, 22.1)
    out = preprocess_precipitation_dataset(
        ds,
        policy=PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7,
        region_geojson=polygon,
    )
    assert PR_PROJECT_VARIABLE in out.data_vars
    pr = out[PR_PROJECT_VARIABLE]
    assert pr.attrs["units"] == DAILY_UNITS_MM
    assert pr.attrs["precipitation_policy"] == PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7
    # Two local days appear after the shift.
    assert pr.sizes["time"] == 2
    daily_vals = pr.isel(lat=0, lon=0).values
    assert float(daily_vals[0]) == pytest.approx(6.0)
    assert float(daily_vals[1]) == pytest.approx(17.0)


def test_preprocess_precipitation_dataset_rejects_unsupported_policy():
    ds = _hourly_accumulated_tp_dataset()
    with pytest.raises(PrecipitationError, match="unsupported precipitation_policy"):
        preprocess_precipitation_dataset(ds, policy="utc_native", region_geojson=None)


def test_preprocess_precipitation_dataset_polygon_mask_zeroes_outside_cells():
    """A polygon covering only the western half of the 2x2 grid should NaN the east column."""
    ds = _hourly_accumulated_tp_dataset()
    west_only = _square_polygon_geojson(-105.8, 21.4, -105.5, 22.1)
    out = preprocess_precipitation_dataset(
        ds,
        policy=PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7,
        region_geojson=west_only,
    )
    pr = out[PR_PROJECT_VARIABLE]
    west_col = pr.isel(time=0, lon=0).values
    east_col = pr.isel(time=0, lon=1).values
    assert not np.isnan(west_col).any()
    assert np.isnan(east_col).all()


# ---------------------------------------------------------------------------
# Preprocessing-layer integration: plan + execute
# ---------------------------------------------------------------------------


def test_plan_results_collapses_h1_and_h2_into_one_pr_result(tmp_path: Path):
    chunks = []
    for chunk_id in ("H1", "H2"):
        target = tmp_path / "raw" / "era5_land" / "hourly_precipitation" / f"2000_{chunk_id}.nc"
        _write_nc(target, _hourly_accumulated_tp_dataset())
        chunks.append(
            {
                "result": _stub_pr_acquisition_result(chunk_id, target_path=str(target).replace("\\", "/")),
                "request": _stub_pr_request(chunk_id),
            }
        )
    results = plan_results(
        chunks,
        output_root=tmp_path / "run",
        precipitation_policy=PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7,
    )
    assert len(results) == 1
    rec = results[0].to_dict()
    assert rec["status"] == STATUS_PLANNED
    assert rec["project_variable"] == "pr"
    assert rec["year"] == 2000
    assert rec["request_id"] == "era5_hourly_pr__2000"
    assert rec["output_path"].endswith("intermediate/daily/pr/2000.nc")
    # Both chunks present in source_chunks, sorted by chunk_id.
    assert [c["chunk_id"] for c in rec["source_chunks"]] == ["H1", "H2"]


def test_plan_results_without_policy_keeps_per_chunk_deferral(tmp_path: Path):
    """M004 behavior preserved: no policy -> each precipitation chunk deferred individually."""
    chunks = []
    for chunk_id in ("H1", "H2"):
        chunks.append(
            {
                "result": _stub_pr_acquisition_result(chunk_id, target_path=""),
                "request": _stub_pr_request(chunk_id),
            }
        )
    results = plan_results(chunks, output_root=tmp_path)
    assert len(results) == 2
    for r in results:
        rec = r.to_dict()
        assert rec["status"] == STATUS_DEFERRED
        assert rec["reason"] == REASON_PRECIPITATION_OPEN


def test_plan_results_rejects_unsupported_precipitation_policy(tmp_path: Path):
    joined = [
        {
            "result": _stub_pr_acquisition_result("H1", target_path=""),
            "request": _stub_pr_request("H1"),
        }
    ]
    results = plan_results(
        joined, output_root=tmp_path, precipitation_policy="utc_native_zero"
    )
    rec = results[0].to_dict()
    assert rec["status"] == STATUS_DEFERRED
    assert REASON_UNSUPPORTED_PRECIPITATION_POLICY in rec["reason"]


def test_plan_results_reports_missing_chunk_on_disk(tmp_path: Path):
    """If any chunk's target_path does not exist, the plan records a reason."""
    chunks = []
    for chunk_id, exists in (("H1", True), ("H2", False)):
        if exists:
            target = tmp_path / f"2000_{chunk_id}.nc"
            _write_nc(target, _hourly_accumulated_tp_dataset())
            target_path = str(target).replace("\\", "/")
        else:
            target_path = str(tmp_path / f"missing_{chunk_id}.nc").replace("\\", "/")
        chunks.append(
            {
                "result": _stub_pr_acquisition_result(chunk_id, target_path=target_path),
                "request": _stub_pr_request(chunk_id),
            }
        )
    results = plan_results(
        chunks,
        output_root=tmp_path / "run",
        precipitation_policy=PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7,
    )
    rec = results[0].to_dict()
    assert rec["reason"] == REASON_SOURCE_NOT_FOUND


def test_execute_results_writes_daily_pr_from_two_chunks(tmp_path: Path):
    """Two chunks concatenated along time produce one daily pr NetCDF in mm/day."""
    chunks = []
    for chunk_id, year, hour_start in (("H1", 2000, 0), ("H2", 2000, 12)):
        # Build a 12-hour chunk picking up where the previous chunk left off,
        # so concat along time gives 24 monotonically-increasing accumulated values.
        times = np.array(
            [f"{year}-01-01T{h:02d}:00" for h in range(hour_start, hour_start + 12)],
            dtype="datetime64[ns]",
        )
        accum = np.arange(hour_start, hour_start + 12, dtype="float32") / METERS_TO_MILLIMETERS
        lat = [22.0, 21.5]
        lon = [-105.6, -105.4]
        values = np.tile(accum[:, None, None], (1, len(lat), len(lon)))
        da = xr.DataArray(
            values,
            dims=("valid_time", "latitude", "longitude"),
            coords={"valid_time": times, "latitude": lat, "longitude": lon},
            name="tp",
            attrs={"units": "m"},
        )
        target = tmp_path / "raw" / "era5_land" / "hourly_precipitation" / f"{year}_{chunk_id}.nc"
        _write_nc(target, da.to_dataset())
        chunks.append(
            {
                "result": _stub_pr_acquisition_result(chunk_id, target_path=str(target).replace("\\", "/")),
                "request": _stub_pr_request(chunk_id),
            }
        )
    run_root = tmp_path / "run"
    polygon = _square_polygon_geojson(-105.8, 21.4, -105.3, 22.1)
    results = execute_results(
        chunks,
        output_root=run_root,
        region_geojson=polygon,
        precipitation_policy=PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7,
    )
    assert len(results) == 1
    assert results[0].status == STATUS_PREPROCESSED
    out_path = run_root / "intermediate" / "daily" / "pr" / "2000.nc"
    assert out_path.exists()
    with xr.open_dataset(out_path) as ds_out:
        ds_out.load()
    assert PR_PROJECT_VARIABLE in ds_out.data_vars
    pr = ds_out[PR_PROJECT_VARIABLE]
    assert pr.attrs["units"] == DAILY_UNITS_MM
    assert pr.attrs["precipitation_policy"] == PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7
    # Same accounting as the single-day fixture test: 6 mm on prev local day, 17 mm on same local day.
    daily_vals = pr.isel(lat=0, lon=0).values
    assert float(daily_vals[0]) == pytest.approx(6.0)
    assert float(daily_vals[1]) == pytest.approx(17.0)


def test_execute_results_records_missing_input_when_a_chunk_is_missing(tmp_path: Path):
    """If H2 is missing on disk, no daily output is written and the group is missing_input."""
    chunks = []
    h1_target = tmp_path / "2000_H1.nc"
    _write_nc(h1_target, _hourly_accumulated_tp_dataset())
    chunks.append(
        {
            "result": _stub_pr_acquisition_result("H1", target_path=str(h1_target).replace("\\", "/")),
            "request": _stub_pr_request("H1"),
        }
    )
    chunks.append(
        {
            "result": _stub_pr_acquisition_result("H2", target_path=str(tmp_path / "no_such_file.nc").replace("\\", "/")),
            "request": _stub_pr_request("H2"),
        }
    )
    run_root = tmp_path / "run"
    polygon = _square_polygon_geojson(-105.8, 21.4, -105.3, 22.1)
    results = execute_results(
        chunks,
        output_root=run_root,
        region_geojson=polygon,
        precipitation_policy=PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7,
    )
    assert len(results) == 1
    assert results[0].status == STATUS_MISSING_INPUT
    assert not (run_root / "intermediate" / "daily" / "pr").exists()


# ---------------------------------------------------------------------------
# Import-safety
# ---------------------------------------------------------------------------


def test_precipitation_module_does_not_eagerly_import_heavy_deps():
    import inspect

    import lib.precipitation as p

    src = inspect.getsource(p)
    for line in src.splitlines():
        stripped = line.lstrip()
        if line == stripped:  # top-level, not indented
            for forbidden in ("import numpy", "from numpy", "import xarray", "from xarray",
                              "import pandas", "from pandas", "import shapely", "from shapely",
                              "import dask", "from dask"):
                assert not stripped.startswith(forbidden), f"eager import: {line!r}"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "03_preprocess_daily.py"
    spec = importlib.util.spec_from_file_location("preprocess_script_pr", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _make_precipitation_only_acquisition_manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write minimal acquisition / download / region manifests pointing at precipitation chunks."""
    geometry_hash = "sha256:" + "0" * 64
    region = {
        "region_id": "rbmn",
        "geometry_path": str(RBMN_GEOJSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "geometry_type": "MultiPolygon",
        "feature_count": 1,
        "source_crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
        "normalized_crs": "EPSG:4326",
        "bbox_west_south_east_north": [-105.7, 21.7, -105.3, 22.5],
        "bbox_north_west_south_east": [22.5, -105.7, 21.7, -105.3],
        "clip_policy": "polygon",
        "geometry_hash": geometry_hash,
        "created_by": "tests",
    }
    # Substitute the canonical hash so load_region_manifest_for_preprocessing accepts it.
    canonical = json.loads(
        (REPO_ROOT / "runs" / "dev_region" / "region_manifest.json").read_text("utf-8")
    )
    region["geometry_hash"] = canonical["geometry_hash"]
    region["bbox_west_south_east_north"] = canonical["bbox_west_south_east_north"]
    region["bbox_north_west_south_east"] = canonical["bbox_north_west_south_east"]
    region_path = tmp_path / "region.json"
    region_path.write_text(json.dumps(region, sort_keys=True), encoding="utf-8")

    download = {
        "manifest_type": "era5_land_download_plan",
        "region_id": "rbmn",
        "region_manifest_path": str(region_path).replace("\\", "/"),
        "region_geometry_hash": region["geometry_hash"],
        "bbox_north_west_south_east": region["bbox_north_west_south_east"],
        "start_year": 2000,
        "end_year": 2000,
        "datasets": {},
        "requests": [_stub_pr_request("H1"), _stub_pr_request("H2")],
        "requires_network": False,
        "download_execution_status": "planned_only",
        "created_by": "tests",
    }
    download_path = tmp_path / "download.json"
    download_path.write_text(json.dumps(download, sort_keys=True), encoding="utf-8")

    acquisition = {
        "manifest_type": "era5_land_acquisition_run",
        "region_id": "rbmn",
        "region_geometry_hash": region["geometry_hash"],
        "mode": "dry-run",
        "output_root": "runs/dev_region",
        "request_count": 2,
        "planned_count": 2,
        "downloaded_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "results": [
            _stub_pr_acquisition_result("H1", target_path="runs/dev_region/raw/era5_land/hourly_precipitation/2000_H1.nc", status="planned"),
            _stub_pr_acquisition_result("H2", target_path="runs/dev_region/raw/era5_land/hourly_precipitation/2000_H2.nc", status="planned"),
        ],
        "created_by": "tests",
        "requires_network": False,
        "execution_status": "planned_only",
    }
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition, sort_keys=True), encoding="utf-8")

    return acquisition_path, download_path, region_path


def test_script_dry_run_with_precipitation_policy_emits_one_pr_planned_result(tmp_path: Path, capsys):
    main = _load_script_main()
    acquisition_path, download_path, region_path = _make_precipitation_only_acquisition_manifest(tmp_path)
    output = tmp_path / "preprocessing_manifest_pr.json"
    rc = main(
        [
            "--acquisition-manifest",
            str(acquisition_path),
            "--download-manifest",
            str(download_path),
            "--region-manifest",
            str(region_path),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "run_root"),
            "--mode",
            "dry-run",
            "--precipitation-policy",
            PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7,
        ]
    )
    assert rc == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["precipitation_policy"] == PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7
    assert loaded["mode"] == MODE_DRY_RUN
    assert loaded["requires_network"] is False
    assert loaded["request_count"] == 1
    assert loaded["planned_count"] == 1
    [result] = loaded["results"]
    assert result["request_id"] == "era5_hourly_pr__2000"
    assert result["project_variable"] == "pr"
    assert result["output_path"].endswith("intermediate/daily/pr/2000.nc")
    assert {c["chunk_id"] for c in result["source_chunks"]} == {"H1", "H2"}
    # No NetCDF side effects.
    assert not any((tmp_path / "run_root").rglob("*.nc")) if (tmp_path / "run_root").exists() else True


def test_script_dry_run_without_policy_still_defers_precipitation(tmp_path: Path):
    """M004 backward-compat path: without --precipitation-policy, precipitation requests defer per chunk."""
    main = _load_script_main()
    acquisition_path, download_path, region_path = _make_precipitation_only_acquisition_manifest(tmp_path)
    output = tmp_path / "preprocessing_manifest_default.json"
    rc = main(
        [
            "--acquisition-manifest",
            str(acquisition_path),
            "--download-manifest",
            str(download_path),
            "--region-manifest",
            str(region_path),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "run_root"),
            "--mode",
            "dry-run",
        ]
    )
    assert rc == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert "precipitation_policy" not in loaded
    assert loaded["deferred_count"] == 2
    for r in loaded["results"]:
        assert r["status"] == STATUS_DEFERRED
        assert r["reason"] == REASON_PRECIPITATION_OPEN


def test_supported_precipitation_policies_contains_legacy_utc_minus_7():
    assert PRECIPITATION_POLICY_LEGACY_UTC_MINUS_7 in SUPPORTED_PRECIPITATION_POLICIES
