"""Tests for ``lib.download_plan`` and ``scripts/01_plan_downloads.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.download_plan import (
    DAILY_STATISTICS_DATASET,
    DAILY_STATISTICS_FREQUENCY,
    DAILY_STATISTICS_TIME_ZONE,
    DAILY_STATISTIC_VARIABLES,
    DAYS,
    HOURLY_PRECIPITATION_CDS_VARIABLE,
    HOURLY_PRECIPITATION_DATASET,
    HOURLY_PRECIPITATION_DATA_FORMAT,
    HOURLY_PRECIPITATION_DOWNLOAD_FORMAT,
    HOURLY_PRECIPITATION_PROJECT_VARIABLE,
    HOURS,
    MONTHS,
    DownloadPlanError,
    build_daily_statistics_requests,
    build_download_manifest,
    build_hourly_precipitation_requests,
    plan_downloads,
    validate_year_range,
    write_download_manifest,
)
from lib.regions import RegionValidationError, load_region_manifest


REPO_ROOT = Path(__file__).resolve().parent.parent
RBMN_REGION_MANIFEST = REPO_ROOT / "runs" / "dev_region" / "region_manifest.json"

EXPECTED_RBMN_BBOX_NWSE = [
    22.573077683788142,
    -105.70053541497737,
    21.741759030754736,
    -105.2937762317876,
]
EXPECTED_RBMN_GEOMETRY_HASH = (
    "sha256:832f37d3d07aef95203f7b03b21b6f4bc8b2bcc8ee0c290f8088bc9bf82748cb"
)


def _stub_region_manifest(**overrides) -> dict:
    base = {
        "region_id": "rbmn",
        "geometry_path": "01_data/case_studies/rbmn.geojson",
        "geometry_type": "MultiPolygon",
        "feature_count": 1,
        "source_crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
        "normalized_crs": "EPSG:4326",
        "bbox_west_south_east_north": [
            -105.70053541497737,
            21.741759030754736,
            -105.2937762317876,
            22.573077683788142,
        ],
        "bbox_north_west_south_east": list(EXPECTED_RBMN_BBOX_NWSE),
        "clip_policy": "polygon",
        "geometry_hash": EXPECTED_RBMN_GEOMETRY_HASH,
        "created_by": "tests",
    }
    base.update(overrides)
    return base


# --- year range -----------------------------------------------------------


def test_validate_year_range_inclusive_ok():
    assert validate_year_range(2000, 2024) == (2000, 2024)
    assert validate_year_range(2000, 2000) == (2000, 2000)


def test_validate_year_range_rejects_inverted_range():
    with pytest.raises(DownloadPlanError, match="must be <= end_year"):
        validate_year_range(2024, 2000)


def test_validate_year_range_rejects_out_of_window():
    with pytest.raises(DownloadPlanError, match="outside the plannable window"):
        validate_year_range(1800, 2024)
    with pytest.raises(DownloadPlanError, match="outside the plannable window"):
        validate_year_range(2000, 2200)


def test_validate_year_range_rejects_booleans():
    with pytest.raises(DownloadPlanError, match="start_year must be an int"):
        validate_year_range(True, 2024)
    with pytest.raises(DownloadPlanError, match="end_year must be an int"):
        validate_year_range(2000, False)


# --- region manifest loading ---------------------------------------------


def test_load_region_manifest_round_trips_canonical_fields():
    data = load_region_manifest(RBMN_REGION_MANIFEST)
    assert data["region_id"] == "rbmn"
    assert data["bbox_north_west_south_east"] == EXPECTED_RBMN_BBOX_NWSE
    assert data["geometry_hash"] == EXPECTED_RBMN_GEOMETRY_HASH


def test_load_region_manifest_rejects_missing_fields(tmp_path: Path):
    bad = tmp_path / "incomplete.json"
    bad.write_text(json.dumps({"region_id": "x"}), encoding="utf-8")
    with pytest.raises(RegionValidationError, match="missing required fields"):
        load_region_manifest(bad)


def test_load_region_manifest_rejects_missing_file(tmp_path: Path):
    with pytest.raises(RegionValidationError, match="not found"):
        load_region_manifest(tmp_path / "missing.json")


def test_load_region_manifest_rejects_malformed_bbox(tmp_path: Path):
    manifest = _stub_region_manifest(bbox_north_west_south_east=[1.0, 2.0, 3.0])
    bad = tmp_path / "bad_bbox.json"
    bad.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RegionValidationError, match="4-element list"):
        load_region_manifest(bad)


# --- daily statistics requests -------------------------------------------


def test_daily_statistics_request_count_is_five_variables_per_year():
    manifest = _stub_region_manifest()
    requests = build_daily_statistics_requests(
        region_manifest=manifest, start_year=2000, end_year=2002
    )
    assert len(requests) == len(DAILY_STATISTIC_VARIABLES) * 3
    assert {r["dataset"] for r in requests} == {DAILY_STATISTICS_DATASET}
    assert {r["request_kind"] for r in requests} == {"daily_statistics"}
    request_ids = {r["request_id"] for r in requests}
    assert "era5_daily_stats__tmax__2000" in request_ids
    assert "era5_daily_stats__v10m__2002" in request_ids


def test_daily_statistics_payload_carries_legacy_metadata_and_bbox():
    manifest = _stub_region_manifest()
    requests = build_daily_statistics_requests(
        region_manifest=manifest, start_year=2000, end_year=2000
    )
    payloads_by_id = {r["request_id"]: r["payload"] for r in requests}
    tmax_payload = payloads_by_id["era5_daily_stats__tmax__2000"]
    assert tmax_payload["variable"] == ["2m_temperature"]
    assert tmax_payload["daily_statistic"] == "daily_maximum"
    assert tmax_payload["year"] == "2000"
    assert tmax_payload["month"] == list(MONTHS)
    assert tmax_payload["day"] == list(DAYS)
    assert tmax_payload["frequency"] == DAILY_STATISTICS_FREQUENCY
    assert tmax_payload["time_zone"] == DAILY_STATISTICS_TIME_ZONE
    assert tmax_payload["area"] == EXPECTED_RBMN_BBOX_NWSE
    tmean_payload = payloads_by_id["era5_daily_stats__tmean__2000"]
    assert tmean_payload["daily_statistic"] == "daily_mean"
    u_payload = payloads_by_id["era5_daily_stats__u10m__2000"]
    assert u_payload["variable"] == ["10m_u_component_of_wind"]
    assert u_payload["daily_statistic"] == "daily_mean"


def test_daily_statistics_output_paths_are_per_variable_per_year():
    manifest = _stub_region_manifest()
    requests = build_daily_statistics_requests(
        region_manifest=manifest, start_year=2000, end_year=2000
    )
    output_paths = {r["output_path"] for r in requests}
    assert "raw/era5_land/daily_statistics/tmax/2000.nc" in output_paths
    assert "raw/era5_land/daily_statistics/v10m/2000.nc" in output_paths


def test_daily_statistics_rejects_incomplete_region_manifest():
    bad = {"region_id": "rbmn"}
    with pytest.raises(DownloadPlanError, match="missing required fields"):
        build_daily_statistics_requests(
            region_manifest=bad, start_year=2000, end_year=2000
        )


# --- hourly precipitation requests ---------------------------------------


def test_hourly_precipitation_request_count_is_two_per_year():
    manifest = _stub_region_manifest()
    requests = build_hourly_precipitation_requests(
        region_manifest=manifest, start_year=2000, end_year=2002
    )
    assert len(requests) == 3 * 2
    assert {r["dataset"] for r in requests} == {HOURLY_PRECIPITATION_DATASET}
    assert {r["request_kind"] for r in requests} == {"hourly_precipitation"}
    chunk_ids = {r["chunk_id"] for r in requests}
    assert chunk_ids == {"H1", "H2"}


def test_hourly_precipitation_payload_uses_full_hour_grid_and_bbox():
    manifest = _stub_region_manifest()
    requests = build_hourly_precipitation_requests(
        region_manifest=manifest, start_year=2000, end_year=2000
    )
    by_id = {r["request_id"]: r for r in requests}
    h1 = by_id["era5_hourly_pr__2000_H1"]
    assert h1["project_variables"] == [HOURLY_PRECIPITATION_PROJECT_VARIABLE]
    assert h1["cds_variables"] == [HOURLY_PRECIPITATION_CDS_VARIABLE]
    assert h1["payload"]["variable"] == [HOURLY_PRECIPITATION_CDS_VARIABLE]
    assert h1["payload"]["year"] == "2000"
    assert h1["payload"]["month"] == ["01", "02", "03", "04", "05", "06"]
    assert h1["payload"]["day"] == list(DAYS)
    assert h1["payload"]["time"] == list(HOURS)
    assert h1["payload"]["data_format"] == HOURLY_PRECIPITATION_DATA_FORMAT
    assert h1["payload"]["download_format"] == HOURLY_PRECIPITATION_DOWNLOAD_FORMAT
    assert h1["payload"]["area"] == EXPECTED_RBMN_BBOX_NWSE
    h2 = by_id["era5_hourly_pr__2000_H2"]
    assert h2["payload"]["month"] == ["07", "08", "09", "10", "11", "12"]
    assert h2["output_path"] == "raw/era5_land/hourly_precipitation/2000_H2.nc"


# --- whole download manifest ---------------------------------------------


def test_build_download_manifest_carries_region_provenance():
    manifest = build_download_manifest(
        region_manifest=_stub_region_manifest(),
        region_manifest_path=Path("runs/dev_region/region_manifest.json"),
        start_year=2000,
        end_year=2000,
        created_by="tests",
    )
    assert manifest["manifest_type"] == "era5_land_download_plan"
    assert manifest["region_id"] == "rbmn"
    assert manifest["region_geometry_hash"] == EXPECTED_RBMN_GEOMETRY_HASH
    assert manifest["bbox_north_west_south_east"] == EXPECTED_RBMN_BBOX_NWSE
    assert manifest["start_year"] == 2000
    assert manifest["end_year"] == 2000
    assert manifest["requires_network"] is False
    assert manifest["download_execution_status"] == "planned_only"
    assert manifest["region_manifest_path"] == "runs/dev_region/region_manifest.json"
    assert manifest["created_by"] == "tests"
    daily = manifest["datasets"][DAILY_STATISTICS_DATASET]
    precip = manifest["datasets"][HOURLY_PRECIPITATION_DATASET]
    assert daily["request_count"] == len(DAILY_STATISTIC_VARIABLES)
    assert precip["request_count"] == 2


def test_canonical_rbmn_download_manifest_matches_region_manifest_bbox():
    manifest = plan_downloads(
        region_manifest_path=RBMN_REGION_MANIFEST,
        output_path=Path("runs/dev_region/download_manifest.json"),
        start_year=2000,
        end_year=2024,
        created_by="tests",
    )
    assert manifest["bbox_north_west_south_east"] == EXPECTED_RBMN_BBOX_NWSE
    assert manifest["region_geometry_hash"] == EXPECTED_RBMN_GEOMETRY_HASH
    assert manifest["start_year"] == 2000 and manifest["end_year"] == 2024
    assert manifest["datasets"][DAILY_STATISTICS_DATASET]["request_count"] == (
        len(DAILY_STATISTIC_VARIABLES) * 25
    )
    assert manifest["datasets"][HOURLY_PRECIPITATION_DATASET]["request_count"] == 25 * 2
    project_variables = {
        v
        for request in manifest["requests"]
        for v in request["project_variables"]
    }
    assert project_variables == {"tmax", "tmin", "tmean", "u10m", "v10m", "pr"}


def test_write_download_manifest_is_deterministic(tmp_path: Path):
    manifest = build_download_manifest(
        region_manifest=_stub_region_manifest(),
        region_manifest_path=Path("runs/dev_region/region_manifest.json"),
        start_year=2000,
        end_year=2001,
        created_by="tests",
    )
    a = tmp_path / "a" / "download_manifest.json"
    b = tmp_path / "b" / "download_manifest.json"
    write_download_manifest(a, manifest)
    write_download_manifest(b, manifest)
    assert a.read_bytes() == b.read_bytes()
    loaded = json.loads(a.read_text(encoding="utf-8"))
    assert loaded["download_execution_status"] == "planned_only"


# --- script smoke --------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "01_plan_downloads.py"
    spec = importlib.util.spec_from_file_location("plan_downloads_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_script_main_writes_canonical_download_manifest(tmp_path: Path, capsys):
    main = _load_script_main()
    output = tmp_path / "download_manifest.json"
    rc = main(
        [
            "--region-manifest",
            str(RBMN_REGION_MANIFEST),
            "--output",
            str(output),
            "--start-year",
            "2000",
            "--end-year",
            "2024",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote download plan" in captured.out
    assert "planned_only" in captured.out
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["bbox_north_west_south_east"] == EXPECTED_RBMN_BBOX_NWSE
    assert loaded["region_geometry_hash"] == EXPECTED_RBMN_GEOMETRY_HASH
    assert loaded["requires_network"] is False
    assert loaded["download_execution_status"] == "planned_only"


def test_script_main_returns_error_for_missing_region_manifest(tmp_path: Path, capsys):
    main = _load_script_main()
    rc = main(
        [
            "--region-manifest",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "out.json"),
            "--start-year",
            "2000",
            "--end-year",
            "2024",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "download planning failed" in captured.err


def test_script_main_returns_error_for_inverted_year_range(tmp_path: Path, capsys):
    main = _load_script_main()
    rc = main(
        [
            "--region-manifest",
            str(RBMN_REGION_MANIFEST),
            "--output",
            str(tmp_path / "out.json"),
            "--start-year",
            "2024",
            "--end-year",
            "2000",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "download planning failed" in captured.err
