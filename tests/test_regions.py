"""Tests for ``lib.regions`` and ``scripts/00_validate_region.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.regions import (
    BoundingBox,
    NORMALIZED_CRS,
    RegionValidationError,
    build_region_manifest,
    compute_bbox,
    compute_geometry_hash,
    detect_source_crs,
    iter_features,
    load_geojson,
    load_region_manifest,
    normalize_crs,
    summarize_geometry,
    validate_region,
    write_manifest,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RBMN_GEOJSON = REPO_ROOT / "01_data" / "case_studies" / "rbmn.geojson"

EXPECTED_RBMN_BBOX_WSEN = [
    -105.70053541497737,
    21.741759030754736,
    -105.2937762317876,
    22.573077683788142,
]
EXPECTED_RBMN_BBOX_NWSE = [
    22.573077683788142,
    -105.70053541497737,
    21.741759030754736,
    -105.2937762317876,
]

EXPECTED_RBMN_GEOMETRY_HASH = (
    "sha256:832f37d3d07aef95203f7b03b21b6f4bc8b2bcc8ee0c290f8088bc9bf82748cb"
)


def _square_polygon_feature(west: float, south: float, east: float, north: float) -> dict:
    return {
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


def _square_feature_collection(west: float, south: float, east: float, north: float) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [_square_polygon_feature(west, south, east, north)],
    }


# --- compute_bbox ---------------------------------------------------------


def test_compute_bbox_simple_polygon_feature_collection():
    geojson = _square_feature_collection(west=-1.0, south=-2.0, east=3.0, north=4.0)
    bbox = compute_bbox(geojson)
    assert bbox == BoundingBox(west=-1.0, south=-2.0, east=3.0, north=4.0)
    assert bbox.as_west_south_east_north() == [-1.0, -2.0, 3.0, 4.0]
    assert bbox.as_north_west_south_east() == [4.0, -1.0, -2.0, 3.0]


def test_compute_bbox_multipolygon():
    geojson = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                [[[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0], [5.0, 5.0]]],
            ],
        },
    }
    bbox = compute_bbox(geojson)
    assert bbox.as_west_south_east_north() == [0.0, 0.0, 6.0, 6.0]


def test_compute_bbox_bare_geometry_object():
    geometry = {
        "type": "Polygon",
        "coordinates": [[[10.0, 20.0], [11.0, 20.0], [11.0, 21.0], [10.0, 21.0], [10.0, 20.0]]],
    }
    assert compute_bbox(geometry).as_west_south_east_north() == [10.0, 20.0, 11.0, 21.0]


def test_compute_bbox_ignores_third_ordinate_for_3d_positions():
    geojson = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [0.0, 0.0, 100.0],
                    [1.0, 0.0, 200.0],
                    [1.0, 1.0, 300.0],
                    [0.0, 1.0, 400.0],
                    [0.0, 0.0, 100.0],
                ]
            ],
        },
    }
    assert compute_bbox(geojson).as_west_south_east_north() == [0.0, 0.0, 1.0, 1.0]


def test_compute_bbox_mixed_polygon_and_multipolygon_features():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            _square_polygon_feature(0.0, 0.0, 1.0, 1.0),
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0], [5.0, 5.0]]],
                        [[[-2.0, -3.0], [-1.0, -3.0], [-1.0, -1.0], [-2.0, -1.0], [-2.0, -3.0]]],
                    ],
                },
            },
        ],
    }
    bbox = compute_bbox(geojson)
    assert bbox.as_west_south_east_north() == [-2.0, -3.0, 6.0, 6.0]
    geometry_type, feature_count = summarize_geometry(geojson)
    assert geometry_type == "MultiPolygon+Polygon"
    assert feature_count == 2


def test_compute_bbox_rejects_boolean_coordinate_values():
    geojson = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [True, False],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [True, False],
                ]
            ],
        },
    }
    with pytest.raises(RegionValidationError, match="not a list of positions"):
        compute_bbox(geojson)


def test_compute_bbox_multiple_features():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            _square_polygon_feature(0.0, 0.0, 1.0, 1.0),
            _square_polygon_feature(-2.0, -3.0, -1.0, -1.0),
        ],
    }
    assert compute_bbox(geojson).as_west_south_east_north() == [-2.0, -3.0, 1.0, 1.0]


# --- canonical rbmn polygon -----------------------------------------------


def test_rbmn_geojson_bbox_matches_expected():
    geojson = load_geojson(RBMN_GEOJSON)
    bbox = compute_bbox(geojson)
    assert bbox.as_west_south_east_north() == EXPECTED_RBMN_BBOX_WSEN
    assert bbox.as_north_west_south_east() == EXPECTED_RBMN_BBOX_NWSE


def test_rbmn_geojson_summary_is_single_multipolygon():
    geojson = load_geojson(RBMN_GEOJSON)
    geometry_type, feature_count = summarize_geometry(geojson)
    assert geometry_type == "MultiPolygon"
    assert feature_count == 1


def test_rbmn_geojson_source_crs_is_crs84():
    geojson = load_geojson(RBMN_GEOJSON)
    source = detect_source_crs(geojson)
    assert source == "urn:ogc:def:crs:OGC:1.3:CRS84"
    assert normalize_crs(source) == NORMALIZED_CRS


# --- invalid inputs -------------------------------------------------------


def test_load_geojson_missing_file(tmp_path: Path):
    with pytest.raises(RegionValidationError, match="not found"):
        load_geojson(tmp_path / "missing.geojson")


def test_load_geojson_malformed_json(tmp_path: Path):
    bad = tmp_path / "bad.geojson"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RegionValidationError, match="not valid JSON"):
        load_geojson(bad)


def test_compute_bbox_empty_feature_collection_rejected():
    with pytest.raises(RegionValidationError, match="no features"):
        compute_bbox({"type": "FeatureCollection", "features": []})


def test_compute_bbox_unsupported_geometry_type_rejected():
    geojson = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
    }
    with pytest.raises(RegionValidationError, match="unsupported geometry type"):
        compute_bbox(geojson)


def test_compute_bbox_missing_geometry_rejected():
    geojson = {"type": "Feature", "properties": {}, "geometry": None}
    with pytest.raises(RegionValidationError, match="no geometry"):
        compute_bbox(geojson)


def test_compute_bbox_unsupported_top_level_type_rejected():
    with pytest.raises(RegionValidationError, match="unsupported GeoJSON top-level type"):
        compute_bbox({"type": "GeometryCollection", "geometries": []})


def test_normalize_crs_rejects_projected_crs():
    with pytest.raises(RegionValidationError, match="unsupported CRS"):
        normalize_crs("EPSG:3857")


def test_iter_features_rejects_non_feature_in_collection():
    geojson = {"type": "FeatureCollection", "features": [{"type": "Polygon", "coordinates": []}]}
    with pytest.raises(RegionValidationError, match="non-Feature entry"):
        list(iter_features(geojson))


# --- manifest writing -----------------------------------------------------


def test_write_manifest_is_deterministic(tmp_path: Path):
    geojson = _square_feature_collection(0.0, 0.0, 1.0, 1.0)
    geometry_path = tmp_path / "square.geojson"
    geometry_path.write_text(json.dumps(geojson), encoding="utf-8")
    manifest = build_region_manifest(
        region_id="square",
        geometry_path=geometry_path,
        geojson=geojson,
        bbox=compute_bbox(geojson),
        source_crs="urn:ogc:def:crs:OGC:1.3:CRS84",
        normalized_crs=NORMALIZED_CRS,
        geometry_hash=compute_geometry_hash(geometry_path),
        clip_policy="polygon",
        created_by="tests",
    )
    out_a = tmp_path / "a" / "manifest.json"
    out_b = tmp_path / "b" / "manifest.json"
    write_manifest(out_a, manifest)
    write_manifest(out_b, manifest)
    assert out_a.read_bytes() == out_b.read_bytes()
    loaded = json.loads(out_a.read_text(encoding="utf-8"))
    assert loaded["region_id"] == "square"
    assert loaded["bbox_west_south_east_north"] == [0.0, 0.0, 1.0, 1.0]
    assert loaded["bbox_north_west_south_east"] == [1.0, 0.0, 0.0, 1.0]
    assert loaded["geometry_type"] == "Polygon"
    assert loaded["feature_count"] == 1
    assert loaded["normalized_crs"] == NORMALIZED_CRS
    assert loaded["clip_policy"] == "polygon"
    assert loaded["geometry_hash"].startswith("sha256:")


def test_validate_region_round_trip_against_rbmn(tmp_path: Path):
    manifest = validate_region(
        region_id="rbmn",
        geometry_path=RBMN_GEOJSON,
        clip_policy="polygon",
        created_by="tests",
    )
    output = tmp_path / "region_manifest.json"
    write_manifest(output, manifest)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["region_id"] == "rbmn"
    assert loaded["bbox_north_west_south_east"] == EXPECTED_RBMN_BBOX_NWSE
    assert loaded["geometry_type"] == "MultiPolygon"
    assert loaded["feature_count"] == 1
    assert loaded["normalized_crs"] == NORMALIZED_CRS
    assert loaded["geometry_hash"] == EXPECTED_RBMN_GEOMETRY_HASH


def test_validate_region_rejects_unknown_clip_policy():
    with pytest.raises(RegionValidationError, match="unsupported clip_policy"):
        validate_region(
            region_id="rbmn",
            geometry_path=RBMN_GEOJSON,
            clip_policy="exact",
        )


def test_validate_region_rejects_empty_region_id():
    with pytest.raises(RegionValidationError, match="non-empty"):
        validate_region(region_id="", geometry_path=RBMN_GEOJSON)


# --- load_region_manifest strict validation -------------------------------


def _valid_region_manifest_dict() -> dict:
    return {
        "region_id": "rbmn",
        "geometry_path": "01_data/case_studies/rbmn.geojson",
        "geometry_type": "MultiPolygon",
        "feature_count": 1,
        "source_crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
        "normalized_crs": NORMALIZED_CRS,
        "bbox_west_south_east_north": [-105.7, 21.7, -105.3, 22.5],
        "bbox_north_west_south_east": [22.5, -105.7, 21.7, -105.3],
        "clip_policy": "polygon",
        "geometry_hash": "sha256:" + "0" * 64,
        "created_by": "tests",
    }


def _write_manifest_dict(tmp_path: Path, data: dict) -> Path:
    out = tmp_path / "region_manifest.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


def test_load_region_manifest_accepts_valid_dict(tmp_path: Path):
    path = _write_manifest_dict(tmp_path, _valid_region_manifest_dict())
    data = load_region_manifest(path)
    assert data["clip_policy"] == "polygon"


def test_load_region_manifest_rejects_tampered_geometry_hash(tmp_path: Path):
    bad = _valid_region_manifest_dict()
    bad["geometry_hash"] = "md5:abcd"
    with pytest.raises(RegionValidationError, match="geometry_hash"):
        load_region_manifest(_write_manifest_dict(tmp_path, bad))
    short = _valid_region_manifest_dict()
    short["geometry_hash"] = "sha256:dead"
    with pytest.raises(RegionValidationError, match="64 lowercase hex"):
        load_region_manifest(_write_manifest_dict(tmp_path, short))
    upper = _valid_region_manifest_dict()
    upper["geometry_hash"] = "sha256:" + "A" * 64
    with pytest.raises(RegionValidationError, match="64 lowercase hex"):
        load_region_manifest(_write_manifest_dict(tmp_path, upper))


def test_load_region_manifest_rejects_unsupported_clip_policy(tmp_path: Path):
    bad = _valid_region_manifest_dict()
    bad["clip_policy"] = "exact"
    with pytest.raises(RegionValidationError, match="clip_policy"):
        load_region_manifest(_write_manifest_dict(tmp_path, bad))


def test_load_region_manifest_rejects_bad_normalized_crs(tmp_path: Path):
    bad = _valid_region_manifest_dict()
    bad["normalized_crs"] = "EPSG:3857"
    with pytest.raises(RegionValidationError, match="normalized_crs"):
        load_region_manifest(_write_manifest_dict(tmp_path, bad))


def test_load_region_manifest_rejects_inconsistent_bbox_orderings(tmp_path: Path):
    bad = _valid_region_manifest_dict()
    # Rotate one ordering by changing a coordinate so the two no longer match.
    bad["bbox_north_west_south_east"] = [22.5, -105.7, 21.7, -999.0]
    with pytest.raises(RegionValidationError, match="bbox orderings disagree"):
        load_region_manifest(_write_manifest_dict(tmp_path, bad))


# --- script smoke test ----------------------------------------------------


def test_script_main_writes_expected_manifest(tmp_path: Path, capsys):
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "00_validate_region.py"
    spec = importlib.util.spec_from_file_location("validate_region_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "region_manifest.json"
    rc = module.main(
        [
            "--region-id",
            "rbmn",
            "--geometry",
            str(RBMN_GEOJSON),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote region manifest" in captured.out
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["bbox_north_west_south_east"] == EXPECTED_RBMN_BBOX_NWSE


def test_script_main_returns_error_for_missing_geometry(tmp_path: Path, capsys):
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "00_validate_region.py"
    spec = importlib.util.spec_from_file_location("validate_region_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rc = module.main(
        [
            "--region-id",
            "rbmn",
            "--geometry",
            str(tmp_path / "missing.geojson"),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "region validation failed" in captured.err
