"""Region-first helpers for the climate pipeline.

Pure helpers used by ``scripts/00_validate_region.py``. The functions here
operate on plain JSON-like dicts and standard library types only, so the same
code can run locally, inside a container, or against synced cloud paths.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

NORMALIZED_CRS = "EPSG:4326"

_LONLAT_CRS_NAMES = frozenset(
    {
        "epsg:4326",
        "urn:ogc:def:crs:epsg::4326",
        "urn:ogc:def:crs:ogc:1.3:crs84",
        "urn:ogc:def:crs:ogc::crs84",
        "ogc:crs84",
        "crs84",
        "wgs84",
        "wgs 84",
    }
)

_SUPPORTED_GEOMETRY_TYPES = frozenset({"Polygon", "MultiPolygon"})


class RegionValidationError(ValueError):
    """Raised when a region input fails validation."""


@dataclass(frozen=True)
class BoundingBox:
    west: float
    south: float
    east: float
    north: float

    def as_west_south_east_north(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    def as_north_west_south_east(self) -> list[float]:
        return [self.north, self.west, self.south, self.east]


def load_geojson(path: Path) -> dict[str, Any]:
    """Read a GeoJSON file from disk and return the parsed object."""
    if not path.exists():
        raise RegionValidationError(f"geometry file not found: {path}")
    if not path.is_file():
        raise RegionValidationError(f"geometry path is not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegionValidationError(f"could not read geometry file {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegionValidationError(f"geometry file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegionValidationError(
            f"geometry file must contain a JSON object, got {type(data).__name__}: {path}"
        )
    return data


def detect_source_crs(geojson: dict[str, Any]) -> str:
    """Return the CRS string declared in the GeoJSON, or the WGS84 default."""
    crs = geojson.get("crs")
    if not crs:
        return "urn:ogc:def:crs:OGC:1.3:CRS84"
    if isinstance(crs, dict):
        properties = crs.get("properties")
        if isinstance(properties, dict):
            name = properties.get("name")
            if isinstance(name, str) and name:
                return name
    raise RegionValidationError(f"unrecognized GeoJSON crs member: {crs!r}")


def normalize_crs(source_crs: str) -> str:
    """Confirm the source CRS is longitude/latitude and return ``EPSG:4326``."""
    if source_crs.strip().lower() in _LONLAT_CRS_NAMES:
        return NORMALIZED_CRS
    raise RegionValidationError(
        f"unsupported CRS {source_crs!r}; expected longitude/latitude (EPSG:4326 or OGC:CRS84)"
    )


def iter_features(geojson: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield Feature objects from a FeatureCollection / Feature / geometry input."""
    obj_type = geojson.get("type")
    if obj_type == "FeatureCollection":
        features = geojson.get("features")
        if not isinstance(features, list) or not features:
            raise RegionValidationError("FeatureCollection has no features")
        for feature in features:
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise RegionValidationError("FeatureCollection contains a non-Feature entry")
            yield feature
    elif obj_type == "Feature":
        yield geojson
    elif obj_type in _SUPPORTED_GEOMETRY_TYPES:
        yield {"type": "Feature", "properties": {}, "geometry": geojson}
    elif obj_type is None:
        raise RegionValidationError("GeoJSON object is missing 'type' member")
    else:
        raise RegionValidationError(
            f"unsupported GeoJSON top-level type {obj_type!r}; "
            f"expected FeatureCollection, Feature, Polygon, or MultiPolygon"
        )


def _validate_geometry(geometry: Any) -> dict[str, Any]:
    if not isinstance(geometry, dict):
        raise RegionValidationError("Feature has no geometry object")
    geom_type = geometry.get("type")
    if geom_type not in _SUPPORTED_GEOMETRY_TYPES:
        raise RegionValidationError(
            f"unsupported geometry type {geom_type!r}; expected one of "
            f"{sorted(_SUPPORTED_GEOMETRY_TYPES)}"
        )
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        raise RegionValidationError(f"{geom_type} geometry has empty coordinates")
    return geometry


def _is_numeric_coordinate(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _iter_positions(coordinates: Any) -> Iterator[tuple[float, float]]:
    """Walk arbitrarily nested coordinate arrays and yield (lon, lat) pairs."""
    if (
        isinstance(coordinates, list)
        and len(coordinates) >= 2
        and all(_is_numeric_coordinate(v) for v in coordinates[:2])
    ):
        yield float(coordinates[0]), float(coordinates[1])
        return
    if not isinstance(coordinates, list):
        raise RegionValidationError("coordinate element is not a list of positions")
    for item in coordinates:
        yield from _iter_positions(item)


def compute_bbox(geojson: dict[str, Any]) -> BoundingBox:
    """Derive the (west, south, east, north) bbox from a GeoJSON object."""
    west = float("inf")
    south = float("inf")
    east = float("-inf")
    north = float("-inf")
    feature_count = 0
    geometry_types: set[str] = set()
    for feature in iter_features(geojson):
        geometry = _validate_geometry(feature.get("geometry"))
        geometry_types.add(geometry["type"])
        feature_count += 1
        for lon, lat in _iter_positions(geometry["coordinates"]):
            if lon < west:
                west = lon
            if lon > east:
                east = lon
            if lat < south:
                south = lat
            if lat > north:
                north = lat
    if feature_count == 0:
        raise RegionValidationError("no features found while computing bbox")
    if west == float("inf"):
        raise RegionValidationError("no coordinates found while computing bbox")
    if not (west <= east and south <= north):
        raise RegionValidationError(
            f"degenerate bbox derived: west={west}, south={south}, east={east}, north={north}"
        )
    return BoundingBox(west=west, south=south, east=east, north=north)


def summarize_geometry(geojson: dict[str, Any]) -> tuple[str, int]:
    """Return ``(geometry_type_summary, feature_count)`` for the manifest."""
    types: list[str] = []
    count = 0
    for feature in iter_features(geojson):
        geometry = _validate_geometry(feature.get("geometry"))
        types.append(geometry["type"])
        count += 1
    unique_types = sorted(set(types))
    summary = unique_types[0] if len(unique_types) == 1 else "+".join(unique_types)
    return summary, count


def compute_geometry_hash(path: Path) -> str:
    """Return a stable ``sha256:<hex>`` hash of the geometry file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build_region_manifest(
    *,
    region_id: str,
    geometry_path: Path,
    geojson: dict[str, Any],
    bbox: BoundingBox,
    source_crs: str,
    normalized_crs: str,
    geometry_hash: str,
    clip_policy: str,
    created_by: str,
) -> dict[str, Any]:
    """Assemble the deterministic region manifest dict."""
    geometry_type, feature_count = summarize_geometry(geojson)
    manifest = {
        "region_id": region_id,
        "geometry_path": _as_posix(geometry_path),
        "geometry_type": geometry_type,
        "feature_count": feature_count,
        "source_crs": source_crs,
        "normalized_crs": normalized_crs,
        "bbox_west_south_east_north": bbox.as_west_south_east_north(),
        "bbox_north_west_south_east": bbox.as_north_west_south_east(),
        "clip_policy": clip_policy,
        "geometry_hash": geometry_hash,
        "created_by": created_by,
    }
    return manifest


def write_manifest(output_path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest as deterministic JSON with a trailing newline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")


def _as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def validate_region(
    *,
    region_id: str,
    geometry_path: Path,
    clip_policy: str = "polygon",
    created_by: str = "scripts/00_validate_region.py",
) -> dict[str, Any]:
    """End-to-end validation: load, validate, derive bbox, build manifest."""
    if not region_id:
        raise RegionValidationError("region_id must be a non-empty string")
    if clip_policy not in {"polygon", "bbox", "polygon+bbox"}:
        raise RegionValidationError(
            f"unsupported clip_policy {clip_policy!r}; expected polygon, bbox, or polygon+bbox"
        )
    geojson = load_geojson(geometry_path)
    source_crs = detect_source_crs(geojson)
    normalized_crs = normalize_crs(source_crs)
    bbox = compute_bbox(geojson)
    geometry_hash = compute_geometry_hash(geometry_path)
    return build_region_manifest(
        region_id=region_id,
        geometry_path=geometry_path,
        geojson=geojson,
        bbox=bbox,
        source_crs=source_crs,
        normalized_crs=normalized_crs,
        geometry_hash=geometry_hash,
        clip_policy=clip_policy,
        created_by=created_by,
    )


REGION_MANIFEST_REQUIRED_FIELDS = (
    "region_id",
    "geometry_path",
    "geometry_type",
    "feature_count",
    "source_crs",
    "normalized_crs",
    "bbox_west_south_east_north",
    "bbox_north_west_south_east",
    "clip_policy",
    "geometry_hash",
    "created_by",
)


_VALID_CLIP_POLICIES = frozenset({"polygon", "bbox", "polygon+bbox"})
_GEOMETRY_HASH_PREFIX = "sha256:"
_GEOMETRY_HASH_HEX_LEN = 64


def load_region_manifest(path: Path) -> dict[str, Any]:
    """Read and validate a region manifest produced by milestone 001.

    Validation is stricter than a presence check because downstream milestones
    (download planning, live acquisition) build live requests from these
    fields. Bad values caught here cannot become bad CDS requests later.
    """
    if not path.exists():
        raise RegionValidationError(f"region manifest not found: {path}")
    if not path.is_file():
        raise RegionValidationError(f"region manifest path is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegionValidationError(f"region manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegionValidationError(
            f"region manifest must be a JSON object, got {type(data).__name__}: {path}"
        )
    missing = [f for f in REGION_MANIFEST_REQUIRED_FIELDS if f not in data]
    if missing:
        raise RegionValidationError(
            f"region manifest {path} is missing required fields: {missing}"
        )
    bbox_nwse = data["bbox_north_west_south_east"]
    bbox_wsen = data["bbox_west_south_east_north"]
    for label, bbox in (("bbox_north_west_south_east", bbox_nwse), ("bbox_west_south_east_north", bbox_wsen)):
        if not (isinstance(bbox, list) and len(bbox) == 4):
            raise RegionValidationError(
                f"region manifest {label} must be a 4-element list, got {bbox!r}"
            )
        if not all(_is_numeric_coordinate(v) for v in bbox):
            raise RegionValidationError(
                f"region manifest {label} must contain only numeric values: {bbox!r}"
            )
    n, w, s, e = bbox_nwse
    if [w, s, e, n] != list(bbox_wsen):
        raise RegionValidationError(
            f"region manifest bbox orderings disagree: "
            f"bbox_north_west_south_east={bbox_nwse!r}, "
            f"bbox_west_south_east_north={bbox_wsen!r}"
        )
    if not (w <= e and s <= n):
        raise RegionValidationError(
            f"region manifest bbox is degenerate: west={w}, south={s}, east={e}, north={n}"
        )
    normalized_crs = data["normalized_crs"]
    if normalized_crs != NORMALIZED_CRS:
        raise RegionValidationError(
            f"region manifest normalized_crs must be {NORMALIZED_CRS!r}, got {normalized_crs!r}"
        )
    clip_policy = data["clip_policy"]
    if clip_policy not in _VALID_CLIP_POLICIES:
        raise RegionValidationError(
            f"region manifest clip_policy {clip_policy!r} is not one of {sorted(_VALID_CLIP_POLICIES)}"
        )
    geometry_hash = data["geometry_hash"]
    if not (isinstance(geometry_hash, str) and geometry_hash.startswith(_GEOMETRY_HASH_PREFIX)):
        raise RegionValidationError(
            f"region manifest geometry_hash must start with {_GEOMETRY_HASH_PREFIX!r}: {geometry_hash!r}"
        )
    hex_part = geometry_hash[len(_GEOMETRY_HASH_PREFIX):]
    if len(hex_part) != _GEOMETRY_HASH_HEX_LEN or any(c not in "0123456789abcdef" for c in hex_part):
        raise RegionValidationError(
            f"region manifest geometry_hash must be {_GEOMETRY_HASH_HEX_LEN} lowercase hex chars: {geometry_hash!r}"
        )
    return data


__all__ = [
    "BoundingBox",
    "NORMALIZED_CRS",
    "REGION_MANIFEST_REQUIRED_FIELDS",
    "RegionValidationError",
    "build_region_manifest",
    "compute_bbox",
    "compute_geometry_hash",
    "detect_source_crs",
    "iter_features",
    "load_geojson",
    "load_region_manifest",
    "normalize_crs",
    "summarize_geometry",
    "validate_region",
    "write_manifest",
]
