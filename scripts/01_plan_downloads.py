"""Generate a deterministic ERA5-Land download plan manifest.

Run from the repository root, for example:

    python scripts/01_plan_downloads.py \\
        --region-manifest runs/dev_region/region_manifest.json \\
        --output runs/dev_region/download_manifest.json \\
        --start-year 2000 \\
        --end-year 2024

PowerShell-friendly equivalent (Windows):

    .\\.venv\\Scripts\\python.exe scripts\\01_plan_downloads.py `
        --region-manifest runs/dev_region/region_manifest.json `
        --output runs/dev_region/download_manifest.json `
        --start-year 2000 `
        --end-year 2024

This script does not contact Copernicus, does not require ``cdsapi``, and
does not read or write NetCDF files. It produces a planning artifact only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.download_plan import (
    DownloadPlanError,
    plan_downloads,
    write_download_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic ERA5-Land download plan manifest from a "
            "region manifest. Does not contact Copernicus."
        )
    )
    parser.add_argument(
        "--region-manifest",
        required=True,
        type=Path,
        help="Path to a region manifest produced by scripts/00_validate_region.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the download plan manifest JSON (parent dirs are created).",
    )
    parser.add_argument(
        "--start-year",
        required=True,
        type=int,
        help="Inclusive start year (e.g. 2000).",
    )
    parser.add_argument(
        "--end-year",
        required=True,
        type=int,
        help="Inclusive end year (e.g. 2024).",
    )
    parser.add_argument(
        "--created-by",
        default="scripts/01_plan_downloads.py",
        help="Free-form identifier recorded in the manifest's 'created_by' field.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = plan_downloads(
            region_manifest_path=args.region_manifest,
            output_path=args.output,
            start_year=args.start_year,
            end_year=args.end_year,
            created_by=args.created_by,
        )
    except DownloadPlanError as exc:
        print(f"download planning failed: {exc}", file=sys.stderr)
        return 2
    write_download_manifest(args.output, manifest)
    print(f"wrote download plan: {args.output}")
    print(
        f"requests planned: {len(manifest['requests'])} "
        f"(daily_statistics + hourly_precipitation, "
        f"years {manifest['start_year']}-{manifest['end_year']})"
    )
    print(f"download_execution_status: {manifest['download_execution_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
