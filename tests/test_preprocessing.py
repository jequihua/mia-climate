"""Tests for ``lib.preprocessing`` and ``scripts/03_preprocess_daily.py``.

Synthetic NetCDF fixtures (h5netcdf engine) keep the suite offline,
small, and CDS-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from lib.preprocessing import (
    EXECUTION_STATUS_COMPLETE_EXISTING,
    EXECUTION_STATUS_FAILED,
    EXECUTION_STATUS_PLANNED,
    EXECUTION_STATUS_PREPROCESSED,
    KELVIN_TO_CELSIUS,
    MANIFEST_TYPE_PREPROCESSING,
    MODE_DRY_RUN,
    MODE_EXECUTE,
    REASON_PRECIPITATION_OPEN,
    REASON_SOURCE_NOT_FOUND,
    STATUS_DEFERRED,
    STATUS_FAILED,
    STATUS_MISSING_INPUT,
    STATUS_PLANNED,
    STATUS_PREPROCESSED,
    STATUS_SKIPPED,
    PreprocessingError,
    apply_region_mask,
    assert_provenance_consistency,
    build_preprocessing_manifest,
    compute_manifest_hash,
    convert_temperature_if_needed,
    daily_output_path,
    derive_execution_status,
    execute_results,
    join_results_to_requests,
    load_acquisition_manifest,
    load_download_manifest,
    normalize_dimensions,
    plan_results,
    preprocess_dataset,
    rename_to_project_variable,
    select_joined_records,
    write_preprocessing_manifest,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RBMN_REGION_MANIFEST = REPO_ROOT / "runs" / "dev_region" / "region_manifest.json"
RBMN_DOWNLOAD_MANIFEST = REPO_ROOT / "runs" / "dev_region" / "download_manifest.json"
RBMN_ACQUISITION_MANIFEST = REPO_ROOT / "runs" / "dev_region" / "acquisition_manifest.json"
RBMN_GEOJSON = REPO_ROOT / "01_data" / "case_studies" / "rbmn.geojson"


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _synthetic_temperature_dataset(
    *,
    source_name: str = "t2m",
    time_dim: str = "valid_time",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    base_kelvin: float = 295.15,
    lat_values: list[float] | None = None,
    lon_values: list[float] | None = None,
) -> xr.Dataset:
    """Build a tiny 2x2-grid daily-statistics-style dataset in Kelvin."""
    lat = lat_values or [22.0, 21.5]
    lon = lon_values or [-105.6, -105.4]
    times = np.array(
        ["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[ns]"
    )
    values = np.array(
        [
            [[base_kelvin, base_kelvin + 1.0], [base_kelvin - 0.5, base_kelvin + 0.5]],
            [[base_kelvin + 0.2, base_kelvin + 0.8], [base_kelvin + 0.1, base_kelvin + 0.3]],
            [[base_kelvin + 0.5, base_kelvin + 1.5], [base_kelvin + 0.2, base_kelvin + 0.7]],
        ],
        dtype="float32",
    )
    da = xr.DataArray(
        values,
        dims=(time_dim, lat_dim, lon_dim),
        coords={time_dim: times, lat_dim: lat, lon_dim: lon},
        name=source_name,
        attrs={"units": "K"},
    )
    return da.to_dataset()


def _write_synthetic_netcdf(path: Path, dataset: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path, engine="h5netcdf")


def _rbmn_polygon_geojson() -> dict[str, Any]:
    """A tiny polygon roughly covering the synthetic 2x2 grid centers."""
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
                            [-105.8, 21.4],
                            [-105.3, 21.4],
                            [-105.3, 22.1],
                            [-105.8, 22.1],
                            [-105.8, 21.4],
                        ]
                    ],
                },
            }
        ],
    }


def _half_polygon_geojson() -> dict[str, Any]:
    """Polygon that only covers the western half of the 2x2 grid (lon=-105.6)."""
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
                            [-105.8, 21.4],
                            [-105.5, 21.4],
                            [-105.5, 22.1],
                            [-105.8, 22.1],
                            [-105.8, 21.4],
                        ]
                    ],
                },
            }
        ],
    }


def _stub_download_request(
    request_id: str,
    *,
    project_variable: str = "tmax",
    year: int = 2000,
    request_kind: str = "daily_statistics",
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "request_kind": request_kind,
        "dataset": "derived-era5-land-daily-statistics",
        "project_variables": [project_variable],
        "cds_variables": ["2m_temperature"],
        "year": year,
        "output_path": f"raw/era5_land/daily_statistics/{project_variable}/{year}.nc",
        "payload": {"variable": ["2m_temperature"]},
    }


def _stub_acquisition_result(request_id: str, *, target_path: str, status: str = "downloaded") -> dict[str, Any]:
    return {
        "request_id": request_id,
        "dataset": "derived-era5-land-daily-statistics",
        "output_path": f"raw/era5_land/daily_statistics/x/{request_id}.nc",
        "target_path": target_path,
        "status": status,
    }


def _stub_acquisition_manifest(*results: dict[str, Any], region_id: str = "rbmn") -> dict[str, Any]:
    return {
        "manifest_type": "era5_land_acquisition_run",
        "region_id": region_id,
        "region_geometry_hash": "sha256:" + "0" * 64,
        "mode": "execute",
        "output_root": "runs/dev_region",
        "request_count": len(results),
        "planned_count": 0,
        "downloaded_count": len(results),
        "skipped_count": 0,
        "failed_count": 0,
        "results": list(results),
        "created_by": "tests",
        "requires_network": True,
        "execution_status": "downloaded",
    }


def _stub_download_manifest(*requests: dict[str, Any], region_id: str = "rbmn") -> dict[str, Any]:
    return {
        "manifest_type": "era5_land_download_plan",
        "region_id": region_id,
        "region_manifest_path": "runs/dev_region/region_manifest.json",
        "region_geometry_hash": "sha256:" + "0" * 64,
        "bbox_north_west_south_east": [22.5, -105.7, 21.7, -105.3],
        "start_year": 2000,
        "end_year": 2000,
        "datasets": {},
        "requests": list(requests),
        "requires_network": False,
        "download_execution_status": "planned_only",
        "created_by": "tests",
    }


def _stub_region_manifest_dict(*, region_id: str = "rbmn") -> dict[str, Any]:
    return {
        "region_id": region_id,
        "geometry_path": "01_data/case_studies/rbmn.geojson",
        "geometry_type": "Polygon",
        "feature_count": 1,
        "source_crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
        "normalized_crs": "EPSG:4326",
        "bbox_west_south_east_north": [-105.7, 21.7, -105.3, 22.5],
        "bbox_north_west_south_east": [22.5, -105.7, 21.7, -105.3],
        "clip_policy": "polygon",
        "geometry_hash": "sha256:" + "0" * 64,
        "created_by": "tests",
    }


# ---------------------------------------------------------------------------
# Coordinate / variable normalization
# ---------------------------------------------------------------------------


def test_normalize_dimensions_renames_valid_time_and_lat_lon():
    ds = _synthetic_temperature_dataset()
    normalized = normalize_dimensions(ds)
    assert "time" in normalized.coords or "time" in normalized.dims
    assert "lat" in normalized.coords
    assert "lon" in normalized.coords
    assert "valid_time" not in normalized.coords
    assert "latitude" not in normalized.coords
    assert "longitude" not in normalized.coords


def test_normalize_dimensions_accepts_already_normalized_dataset():
    ds = _synthetic_temperature_dataset(time_dim="time", lat_dim="lat", lon_dim="lon")
    out = normalize_dimensions(ds)
    assert set(out.dims) == {"time", "lat", "lon"}


def test_normalize_dimensions_rejects_conflicting_existing_target():
    ds = _synthetic_temperature_dataset(time_dim="valid_time", lat_dim="lat", lon_dim="longitude")
    ds = ds.assign_coords(time=("valid_time", ds["valid_time"].values))
    with pytest.raises(PreprocessingError, match="both already exist"):
        normalize_dimensions(ds)


def test_rename_to_project_variable_explicit_source_and_inferred():
    ds = _synthetic_temperature_dataset()
    out = rename_to_project_variable(ds, project_variable="tmax", source_variable="t2m")
    assert "tmax" in out.data_vars and "t2m" not in out.data_vars
    inferred = rename_to_project_variable(ds, project_variable="tmin")
    assert "tmin" in inferred.data_vars


def test_rename_to_project_variable_rejects_unsupported_project_variable():
    ds = _synthetic_temperature_dataset()
    with pytest.raises(PreprocessingError, match="unsupported project_variable"):
        rename_to_project_variable(ds, project_variable="pr")


def test_rename_to_project_variable_rejects_missing_source():
    ds = _synthetic_temperature_dataset(source_name="something_else")
    with pytest.raises(PreprocessingError, match="not found in dataset"):
        rename_to_project_variable(ds, project_variable="tmax", source_variable="t2m")


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def test_convert_temperature_kelvin_to_celsius_for_temperature_vars():
    ds = _synthetic_temperature_dataset(base_kelvin=300.15)
    renamed = rename_to_project_variable(ds, project_variable="tmax", source_variable="t2m")
    converted = convert_temperature_if_needed(renamed, project_variable="tmax")
    # 300.15 K - 273.15 == 27.0 C; first sample at (0, 0, 0) was base_kelvin exactly.
    assert float(converted["tmax"].isel(valid_time=0, latitude=0, longitude=0)) == pytest.approx(27.0)
    assert converted["tmax"].attrs["units"] == "degC"
    assert converted["tmax"].attrs["preprocessing_offset_kelvin"] == KELVIN_TO_CELSIUS


def test_convert_temperature_is_noop_for_wind():
    ds = _synthetic_temperature_dataset(source_name="u10")
    renamed = rename_to_project_variable(ds, project_variable="u10m", source_variable="u10")
    out = convert_temperature_if_needed(renamed, project_variable="u10m")
    # No unit change, raw values preserved.
    assert "units" not in out["u10m"].attrs or out["u10m"].attrs["units"] == "K"
    assert float(out["u10m"].isel(valid_time=0, latitude=0, longitude=0)) == pytest.approx(295.15)


# ---------------------------------------------------------------------------
# Polygon masking
# ---------------------------------------------------------------------------


def test_apply_region_mask_sets_outside_cells_to_nan():
    ds = _synthetic_temperature_dataset()
    normalized = normalize_dimensions(ds)
    renamed = rename_to_project_variable(normalized, project_variable="tmax", source_variable="t2m")
    masked = apply_region_mask(renamed, region_geojson=_half_polygon_geojson())
    arr = masked["tmax"].isel(time=0).values
    # West column (lon=-105.6) inside polygon, east column (lon=-105.4) outside.
    assert not np.isnan(arr[:, 0]).any()
    assert np.isnan(arr[:, 1]).all()


def test_apply_region_mask_keeps_all_cells_for_covering_polygon():
    ds = _synthetic_temperature_dataset()
    normalized = normalize_dimensions(ds)
    renamed = rename_to_project_variable(normalized, project_variable="tmax", source_variable="t2m")
    masked = apply_region_mask(renamed, region_geojson=_rbmn_polygon_geojson())
    assert not np.isnan(masked["tmax"].values).any()


def test_preprocess_dataset_full_pipeline():
    ds = _synthetic_temperature_dataset(base_kelvin=298.15)
    out = preprocess_dataset(
        ds,
        project_variable="tmean",
        region_geojson=_rbmn_polygon_geojson(),
    )
    assert "tmean" in out.data_vars
    assert "time" in out.dims and "lat" in out.dims and "lon" in out.dims
    # 298.15 K -> 25.0 C
    assert float(out["tmean"].isel(time=0, lat=0, lon=0)) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Manifest loading + joining
# ---------------------------------------------------------------------------


def test_load_acquisition_manifest_rejects_wrong_type(tmp_path: Path):
    bad = _stub_acquisition_manifest(_stub_acquisition_result("a", target_path="x"))
    bad["manifest_type"] = "something_else"
    p = tmp_path / "acq.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(PreprocessingError, match="manifest_type"):
        load_acquisition_manifest(p)


def test_load_download_manifest_rejects_wrong_type(tmp_path: Path):
    bad = _stub_download_manifest(_stub_download_request("a"))
    bad["manifest_type"] = "something_else"
    p = tmp_path / "dl.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(PreprocessingError, match="manifest_type"):
        load_download_manifest(p)


def test_assert_provenance_consistency_detects_hash_mismatch():
    acq = _stub_acquisition_manifest(_stub_acquisition_result("a", target_path="x"))
    dl = _stub_download_manifest(_stub_download_request("a"))
    region = _stub_region_manifest_dict()
    # Mutate one hash so the chain disagrees.
    acq["region_geometry_hash"] = "sha256:" + "9" * 64
    with pytest.raises(PreprocessingError, match="region_geometry_hash"):
        assert_provenance_consistency(acquisition=acq, download=dl, region=region)


def test_assert_provenance_consistency_rejects_acquisition_only_region_id_mismatch():
    acq = _stub_acquisition_manifest(_stub_acquisition_result("a", target_path="x"), region_id="other")
    dl = _stub_download_manifest(_stub_download_request("a"), region_id="rbmn")
    region = _stub_region_manifest_dict(region_id="rbmn")
    with pytest.raises(PreprocessingError, match="region_id disagrees"):
        assert_provenance_consistency(acquisition=acq, download=dl, region=region)


def test_assert_provenance_consistency_rejects_download_only_region_id_mismatch():
    acq = _stub_acquisition_manifest(_stub_acquisition_result("a", target_path="x"), region_id="rbmn")
    dl = _stub_download_manifest(_stub_download_request("a"), region_id="other")
    region = _stub_region_manifest_dict(region_id="rbmn")
    with pytest.raises(PreprocessingError, match="region_id disagrees"):
        assert_provenance_consistency(acquisition=acq, download=dl, region=region)


def test_assert_provenance_consistency_rejects_region_only_region_id_mismatch():
    """The chained `a != b != c` form misses this case; the set-based check catches it."""
    acq = _stub_acquisition_manifest(_stub_acquisition_result("a", target_path="x"), region_id="rbmn")
    dl = _stub_download_manifest(_stub_download_request("a"), region_id="rbmn")
    region = _stub_region_manifest_dict(region_id="other")
    with pytest.raises(PreprocessingError, match="region_id disagrees"):
        assert_provenance_consistency(acquisition=acq, download=dl, region=region)


def test_assert_provenance_consistency_accepts_all_three_agreeing():
    acq = _stub_acquisition_manifest(_stub_acquisition_result("a", target_path="x"))
    dl = _stub_download_manifest(_stub_download_request("a"))
    region = _stub_region_manifest_dict()
    assert_provenance_consistency(acquisition=acq, download=dl, region=region)


def test_join_results_to_requests_raises_when_request_missing():
    acq = _stub_acquisition_manifest(_stub_acquisition_result("ghost", target_path="x"))
    dl = _stub_download_manifest(_stub_download_request("other"))
    with pytest.raises(PreprocessingError, match="no matching"):
        join_results_to_requests(acquisition=acq, download=dl)


def test_select_joined_records_filters_and_limits():
    joined = [
        {"result": _stub_acquisition_result(rid, target_path=f"x/{rid}"), "request": _stub_download_request(rid)}
        for rid in ("a", "b", "c")
    ]
    by_id = select_joined_records(joined, request_ids=["c", "a"])
    assert [e["result"]["request_id"] for e in by_id] == ["a", "c"]
    limited = select_joined_records(joined, limit=2)
    assert [e["result"]["request_id"] for e in limited] == ["a", "b"]
    with pytest.raises(PreprocessingError, match="positive int"):
        select_joined_records(joined, limit=0)
    with pytest.raises(PreprocessingError, match="unknown request_id"):
        select_joined_records(joined, request_ids=["zz"])


# ---------------------------------------------------------------------------
# Dry-run planning
# ---------------------------------------------------------------------------


def test_plan_results_marks_daily_stats_as_planned(tmp_path: Path):
    joined = [
        {
            "result": _stub_acquisition_result("a", target_path=str(tmp_path / "src.nc")),
            "request": _stub_download_request("a"),
        }
    ]
    results = plan_results(joined, output_root=tmp_path / "run")
    assert results[0].status == STATUS_PLANNED
    assert results[0].project_variable == "tmax"
    assert results[0].output_path.endswith("intermediate/daily/tmax/2000.nc")


def test_plan_results_defers_precipitation_request(tmp_path: Path):
    joined = [
        {
            "result": _stub_acquisition_result("pr-2000", target_path="x"),
            "request": _stub_download_request(
                "pr-2000", project_variable="pr", request_kind="hourly_precipitation"
            ),
        }
    ]
    results = plan_results(joined, output_root=tmp_path)
    assert results[0].status == STATUS_DEFERRED
    assert results[0].reason == REASON_PRECIPITATION_OPEN
    assert results[0].output_path == ""


def test_plan_results_records_source_not_found_reason(tmp_path: Path):
    joined = [
        {
            "result": _stub_acquisition_result("a", target_path=str(tmp_path / "missing.nc")),
            "request": _stub_download_request("a"),
        }
    ]
    results = plan_results(joined, output_root=tmp_path)
    assert results[0].reason == REASON_SOURCE_NOT_FOUND


# ---------------------------------------------------------------------------
# Execute mode
# ---------------------------------------------------------------------------


def test_execute_results_writes_daily_product(tmp_path: Path):
    source = tmp_path / "src.nc"
    _write_synthetic_netcdf(source, _synthetic_temperature_dataset(base_kelvin=298.15))
    joined = [
        {
            "result": _stub_acquisition_result("a", target_path=str(source), status="downloaded"),
            "request": _stub_download_request("a"),
        }
    ]
    results = execute_results(
        joined,
        output_root=tmp_path / "run",
        region_geojson=_rbmn_polygon_geojson(),
    )
    assert results[0].status == STATUS_PREPROCESSED
    output = tmp_path / "run" / "intermediate" / "daily" / "tmax" / "2000.nc"
    assert output.exists()
    with xr.open_dataset(output) as ds_out:
        ds_out.load()
    assert "tmax" in ds_out.data_vars
    assert set(ds_out.dims) == {"time", "lat", "lon"}
    assert float(ds_out["tmax"].isel(time=0, lat=0, lon=0)) == pytest.approx(25.0)


def test_execute_results_skips_existing_output_unless_overwrite(tmp_path: Path):
    source = tmp_path / "src.nc"
    _write_synthetic_netcdf(source, _synthetic_temperature_dataset())
    target = tmp_path / "run" / "intermediate" / "daily" / "tmax" / "2000.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"prior")
    joined = [
        {
            "result": _stub_acquisition_result("a", target_path=str(source)),
            "request": _stub_download_request("a"),
        }
    ]
    results = execute_results(joined, output_root=tmp_path / "run", region_geojson=_rbmn_polygon_geojson())
    assert results[0].status == STATUS_SKIPPED
    assert target.read_bytes() == b"prior"
    results_ow = execute_results(
        joined,
        output_root=tmp_path / "run",
        region_geojson=_rbmn_polygon_geojson(),
        overwrite=True,
    )
    assert results_ow[0].status == STATUS_PREPROCESSED
    assert target.read_bytes() != b"prior"


def test_execute_results_records_missing_input(tmp_path: Path):
    joined = [
        {
            "result": _stub_acquisition_result("a", target_path=str(tmp_path / "nope.nc")),
            "request": _stub_download_request("a"),
        }
    ]
    results = execute_results(
        joined,
        output_root=tmp_path / "run",
        region_geojson=_rbmn_polygon_geojson(),
    )
    assert results[0].status == STATUS_MISSING_INPUT


def test_execute_results_records_failure_on_bad_source(tmp_path: Path):
    bad_source = tmp_path / "garbage.nc"
    bad_source.write_bytes(b"not a netcdf file")
    joined = [
        {
            "result": _stub_acquisition_result("a", target_path=str(bad_source)),
            "request": _stub_download_request("a"),
        }
    ]
    results = execute_results(
        joined,
        output_root=tmp_path / "run",
        region_geojson=_rbmn_polygon_geojson(),
    )
    assert results[0].status == STATUS_FAILED
    assert results[0].error


def test_execute_results_defers_precipitation(tmp_path: Path):
    joined = [
        {
            "result": _stub_acquisition_result("p", target_path="anywhere"),
            "request": _stub_download_request("p", project_variable="pr", request_kind="hourly_precipitation"),
        }
    ]
    results = execute_results(
        joined,
        output_root=tmp_path / "run",
        region_geojson=_rbmn_polygon_geojson(),
    )
    assert results[0].status == STATUS_DEFERRED
    assert results[0].reason == REASON_PRECIPITATION_OPEN
    assert not any((tmp_path / "run").rglob("*.nc")) if (tmp_path / "run").exists() else True


def test_derive_execution_status_branches(tmp_path: Path):
    from lib.preprocessing import PreprocessingResult

    def _r(status, project="tmax", year=2000):
        return PreprocessingResult(
            request_id=f"r-{status}",
            project_variable=project,
            year=year,
            source_path="x",
            output_path="y",
            status=status,
        )

    assert derive_execution_status(MODE_DRY_RUN, []) == EXECUTION_STATUS_PLANNED
    assert derive_execution_status(MODE_EXECUTE, [_r(STATUS_PREPROCESSED)]) == EXECUTION_STATUS_PREPROCESSED
    assert derive_execution_status(MODE_EXECUTE, [_r(STATUS_SKIPPED)]) == EXECUTION_STATUS_COMPLETE_EXISTING
    assert derive_execution_status(MODE_EXECUTE, [_r(STATUS_FAILED)]) == EXECUTION_STATUS_FAILED
    assert derive_execution_status(
        MODE_EXECUTE, [_r(STATUS_PREPROCESSED), _r(STATUS_FAILED)]
    ) == "partial"


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------


def test_build_and_write_preprocessing_manifest_is_deterministic(tmp_path: Path):
    acq = _stub_acquisition_manifest(_stub_acquisition_result("a", target_path="x"))
    dl = _stub_download_manifest(_stub_download_request("a"))
    region = _stub_region_manifest_dict()
    joined = join_results_to_requests(acquisition=acq, download=dl)
    results = plan_results(joined, output_root=tmp_path)
    manifest = build_preprocessing_manifest(
        acquisition_manifest=acq,
        acquisition_manifest_path=tmp_path / "acq.json",
        acquisition_manifest_hash="sha256:" + "a" * 64,
        download_manifest_path=tmp_path / "dl.json",
        download_manifest_hash="sha256:" + "b" * 64,
        region_manifest=region,
        region_manifest_path=tmp_path / "rm.json",
        region_manifest_hash="sha256:" + "c" * 64,
        mode=MODE_DRY_RUN,
        output_root=tmp_path,
        results=results,
        created_by="tests",
    )
    assert manifest["manifest_type"] == MANIFEST_TYPE_PREPROCESSING
    assert manifest["mode"] == MODE_DRY_RUN
    assert manifest["requires_network"] is False
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    write_preprocessing_manifest(out_a, manifest)
    write_preprocessing_manifest(out_b, manifest)
    assert out_a.read_bytes() == out_b.read_bytes()


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "03_preprocess_daily.py"
    spec = importlib.util.spec_from_file_location("preprocess_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_script_main_dry_run_on_canonical_manifests(tmp_path: Path, capsys):
    main = _load_script_main()
    output = tmp_path / "preprocessing_manifest.json"
    rc = main(
        [
            "--acquisition-manifest",
            str(RBMN_ACQUISITION_MANIFEST),
            "--download-manifest",
            str(RBMN_DOWNLOAD_MANIFEST),
            "--region-manifest",
            str(RBMN_REGION_MANIFEST),
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
    assert "wrote preprocessing manifest" in captured.out
    assert "mode=dry-run" in captured.out
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["manifest_type"] == MANIFEST_TYPE_PREPROCESSING
    assert loaded["mode"] == MODE_DRY_RUN
    assert loaded["requires_network"] is False
    assert loaded["planned_count"] == 3
    assert not any((tmp_path / "run_root").rglob("*.nc")) if (tmp_path / "run_root").exists() else True


def test_script_main_returns_2_for_missing_acquisition_manifest(tmp_path: Path, capsys):
    main = _load_script_main()
    rc = main(
        [
            "--acquisition-manifest",
            str(tmp_path / "missing.json"),
            "--download-manifest",
            str(RBMN_DOWNLOAD_MANIFEST),
            "--region-manifest",
            str(RBMN_REGION_MANIFEST),
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
    assert "preprocessing failed" in captured.err


def test_script_main_execute_against_synthetic_fixture(tmp_path: Path, capsys):
    main = _load_script_main()
    # Build a self-contained acquisition+download+region setup pointing at synthetic NetCDF.
    run_root = tmp_path / "run"
    source = run_root / "raw" / "era5_land" / "daily_statistics" / "tmax" / "2000.nc"
    _write_synthetic_netcdf(source, _synthetic_temperature_dataset(base_kelvin=298.15))
    # Use the real rbmn.geojson so the region manifest validates without surprises.
    region = _stub_region_manifest_dict()
    region["geometry_path"] = str(RBMN_GEOJSON.relative_to(REPO_ROOT)).replace("\\", "/")
    # Substitute the canonical geometry hash so all three manifests agree.
    real_region = json.loads(RBMN_REGION_MANIFEST.read_text(encoding="utf-8"))
    region["geometry_hash"] = real_region["geometry_hash"]
    region["bbox_west_south_east_north"] = real_region["bbox_west_south_east_north"]
    region["bbox_north_west_south_east"] = real_region["bbox_north_west_south_east"]
    region["geometry_type"] = real_region["geometry_type"]
    region_path = tmp_path / "region.json"
    region_path.write_text(json.dumps(region, sort_keys=True), encoding="utf-8")
    dl = _stub_download_manifest(_stub_download_request("a"))
    dl["region_geometry_hash"] = region["geometry_hash"]
    dl_path = tmp_path / "dl.json"
    dl_path.write_text(json.dumps(dl, sort_keys=True), encoding="utf-8")
    acq = _stub_acquisition_manifest(
        _stub_acquisition_result("a", target_path=str(source), status="downloaded")
    )
    acq["region_geometry_hash"] = region["geometry_hash"]
    acq_path = tmp_path / "acq.json"
    acq_path.write_text(json.dumps(acq, sort_keys=True), encoding="utf-8")
    output = tmp_path / "preproc.json"
    rc = main(
        [
            "--acquisition-manifest",
            str(acq_path),
            "--download-manifest",
            str(dl_path),
            "--region-manifest",
            str(region_path),
            "--output",
            str(output),
            "--output-root",
            str(run_root),
            "--mode",
            "execute",
        ]
    )
    assert rc == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["mode"] == MODE_EXECUTE
    assert loaded["preprocessed_count"] == 1
    daily = run_root / "intermediate" / "daily" / "tmax" / "2000.nc"
    assert daily.exists()


def test_compute_manifest_hash_is_deterministic(tmp_path: Path):
    p = tmp_path / "x.json"
    p.write_text("hello", encoding="utf-8")
    assert compute_manifest_hash(p) == compute_manifest_hash(p)
    assert compute_manifest_hash(p).startswith("sha256:")


def test_daily_output_path_format():
    assert daily_output_path("tmax", 2024) == "intermediate/daily/tmax/2024.nc"
