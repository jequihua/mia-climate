"""Run the ERA5-Land acquisition adapter against an M002 download manifest.

Default behavior is dry-run: validate the download manifest, plan request
targets, and write an acquisition manifest without contacting Copernicus.

Live downloads require ``--mode execute`` and the ``cdsapi`` package; CDS
credentials are read from the normal ``~/.cdsapirc`` / environment chain.

Reference dry-run command (PowerShell, Windows):

    .\\.venv\\Scripts\\python.exe scripts\\02_download_era5_land.py `
        --download-manifest runs/dev_region/download_manifest.json `
        --output runs/dev_region/acquisition_manifest.json `
        --output-root runs/dev_region `
        --mode dry-run `
        --limit 3

No NetCDF library is imported here; raw outputs are written by ``cdsapi``
only when ``--mode execute`` is selected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.acquisition import (
    AcquisitionError,
    MODE_DRY_RUN,
    MODE_EXECUTE,
    VALID_MODES,
    build_acquisition_manifest,
    compute_manifest_hash,
    execute_results,
    load_cdsapi_client,
    load_download_manifest,
    plan_results,
    select_requests,
    write_acquisition_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute ERA5-Land downloads from an M002 download "
            "manifest. Defaults to dry-run; live downloads require "
            "--mode execute and a cdsapi installation."
        )
    )
    parser.add_argument(
        "--download-manifest",
        required=True,
        type=Path,
        help="Path to a download manifest produced by scripts/01_plan_downloads.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the acquisition manifest JSON (parent dirs are created).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory under which request output_path values are resolved.",
    )
    parser.add_argument(
        "--mode",
        default=MODE_DRY_RUN,
        choices=sorted(VALID_MODES),
        help="dry-run (default) plans requests without I/O; execute calls cdsapi.",
    )
    parser.add_argument(
        "--request-id",
        action="append",
        default=None,
        dest="request_ids",
        help="Restrict to this request_id; repeatable. Defaults to all requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many requests (after --request-id filtering).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="In execute mode, overwrite existing targets instead of skipping them.",
    )
    parser.add_argument(
        "--created-by",
        default="scripts/02_download_era5_land.py",
        help="Free-form identifier recorded in the manifest's 'created_by' field.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        download_manifest = load_download_manifest(args.download_manifest)
        download_manifest_hash = compute_manifest_hash(args.download_manifest)
        filtered = select_requests(
            download_manifest["requests"],
            request_ids=args.request_ids,
            limit=args.limit,
        )
        if args.mode == MODE_DRY_RUN:
            results = plan_results(filtered, output_root=args.output_root)
        else:
            client = load_cdsapi_client()
            results = execute_results(
                filtered,
                client=client,
                output_root=args.output_root,
                overwrite=args.overwrite,
            )
        manifest = build_acquisition_manifest(
            download_manifest=download_manifest,
            download_manifest_path=args.download_manifest,
            download_manifest_hash=download_manifest_hash,
            mode=args.mode,
            output_root=args.output_root,
            results=results,
            created_by=args.created_by,
        )
    except AcquisitionError as exc:
        print(f"acquisition failed: {exc}", file=sys.stderr)
        return 2
    write_acquisition_manifest(args.output, manifest)
    print(f"wrote acquisition manifest: {args.output}")
    print(
        f"mode={manifest['mode']} "
        f"planned={manifest['planned_count']} "
        f"skipped={manifest['skipped_count']} "
        f"downloaded={manifest['downloaded_count']} "
        f"failed={manifest['failed_count']}"
    )
    print(f"execution_status: {manifest['execution_status']}")
    if args.mode == MODE_EXECUTE and manifest["failed_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
