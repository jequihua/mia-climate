"""Validate a region polygon and write a deterministic region manifest.

Run from the repository root, for example:

    python scripts/00_validate_region.py \\
        --region-id rbmn \\
        --geometry 01_data/case_studies/rbmn.geojson \\
        --output runs/dev_region/region_manifest.json

PowerShell-friendly equivalent (Windows):

    python scripts/00_validate_region.py `
        --region-id rbmn `
        --geometry 01_data/case_studies/rbmn.geojson `
        --output runs/dev_region/region_manifest.json

This script does not download climate data and does not contact Copernicus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.regions import RegionValidationError, validate_region, write_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a region polygon (GeoJSON) and write a region manifest with "
            "deterministic bbox derivations for downstream pipeline steps."
        )
    )
    parser.add_argument(
        "--region-id",
        required=True,
        help="Stable region identifier, e.g. 'rbmn'.",
    )
    parser.add_argument(
        "--geometry",
        required=True,
        type=Path,
        help="Path to a GeoJSON file (FeatureCollection, Feature, or geometry object).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the region manifest JSON (parent directories are created).",
    )
    parser.add_argument(
        "--clip-policy",
        default="polygon",
        choices=("polygon", "bbox", "polygon+bbox"),
        help="Recorded clip policy for downstream preprocessing (default: polygon).",
    )
    parser.add_argument(
        "--created-by",
        default="scripts/00_validate_region.py",
        help="Free-form identifier recorded in the manifest's 'created_by' field.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = validate_region(
            region_id=args.region_id,
            geometry_path=args.geometry,
            clip_policy=args.clip_policy,
            created_by=args.created_by,
        )
    except RegionValidationError as exc:
        print(f"region validation failed: {exc}", file=sys.stderr)
        return 2
    write_manifest(args.output, manifest)
    bbox_nwse = manifest["bbox_north_west_south_east"]
    print(f"wrote region manifest: {args.output}")
    print(f"bbox_north_west_south_east: {bbox_nwse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
