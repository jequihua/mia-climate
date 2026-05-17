"""Preprocess ERA5-Land hourly/daily NetCDF files into daily standard products.

Consumes the M003 acquisition manifest plus the M002 download manifest
and the M001 region manifest. Default mode is dry-run: no NetCDF is
opened or written. Execute mode opens local raw files referenced by
the acquisition manifest, normalizes coordinates/variables/units,
masks cells outside the region polygon, and writes daily products
under ``runs/{run_id}/intermediate/daily/{project_variable}/{year}.nc``.

Daily-statistics requests (``tmax`` / ``tmin`` / ``tmean`` / ``u10m``
/ ``v10m``) are always processed (M004 behavior). Hourly-precipitation
requests are gated by ``--precipitation-policy``:

- without the flag, precipitation chunks are recorded as
  ``status = deferred`` and no daily ``pr`` product is written (M004
  default preserved);
- with ``--precipitation-policy legacy_utc_minus_7``, H1 + H2 chunks
  for the same year collapse into one planning unit and produce a
  daily ``intermediate/daily/pr/{year}.nc`` via ``lib.precipitation``
  (M006). ``legacy_utc_minus_7`` is a compatibility policy matching
  the legacy fixed-offset ``Etc/GMT+7`` workflow; the final UTC vs
  region-specific civil-time decision is still open in
  ``90_legacy_review/migration_decision_log.md``.

Reference dry-run, daily statistics only (PowerShell, Windows):

    .\\.venv\\Scripts\\python.exe scripts\\03_preprocess_daily.py `
        --acquisition-manifest runs/dev_region/acquisition_manifest.json `
        --download-manifest runs/dev_region/download_manifest.json `
        --region-manifest runs/dev_region/region_manifest.json `
        --output runs/dev_region/preprocessing_manifest.json `
        --output-root runs/dev_region `
        --mode dry-run

Reference dry-run, precipitation under the legacy policy:

    .\\.venv\\Scripts\\python.exe scripts\\03_preprocess_daily.py `
        --acquisition-manifest runs/dev_region/acquisition_manifest_precipitation_dry_run.json `
        --download-manifest runs/dev_region/download_manifest.json `
        --region-manifest runs/dev_region/region_manifest.json `
        --output runs/dev_region/preprocessing_manifest_precipitation.json `
        --output-root runs/dev_region `
        --mode dry-run `
        --precipitation-policy legacy_utc_minus_7

This script does not call Copernicus, does not implement precipitation
indices, and does not implement a DST-aware or region-specific
civil-time precipitation policy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.preprocessing import (
    MODE_DRY_RUN,
    MODE_EXECUTE,
    SUPPORTED_PRECIPITATION_POLICIES,
    VALID_MODES,
    PreprocessingError,
    assert_provenance_consistency,
    build_preprocessing_manifest,
    compute_manifest_hash,
    execute_results,
    join_results_to_requests,
    load_acquisition_manifest,
    load_download_manifest,
    load_region_geojson,
    load_region_manifest_for_preprocessing,
    plan_results,
    select_joined_records,
    write_preprocessing_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute ERA5-Land daily-statistics preprocessing. "
            "Default is dry-run; execute opens local NetCDF files and "
            "writes daily standard products."
        )
    )
    parser.add_argument(
        "--acquisition-manifest",
        required=True,
        type=Path,
        help="Path to an M003 acquisition manifest.",
    )
    parser.add_argument(
        "--download-manifest",
        required=True,
        type=Path,
        help="Path to the M002 download manifest the acquisition manifest references.",
    )
    parser.add_argument(
        "--region-manifest",
        required=True,
        type=Path,
        help="Path to the M001 region manifest.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the preprocessing manifest JSON (parent dirs are created).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory under which daily product output paths are resolved.",
    )
    parser.add_argument(
        "--mode",
        default=MODE_DRY_RUN,
        choices=sorted(VALID_MODES),
        help="dry-run (default) plans only; execute opens NetCDF and writes products.",
    )
    parser.add_argument(
        "--request-id",
        action="append",
        default=None,
        dest="request_ids",
        help="Restrict to this request_id; repeatable. Defaults to all acquisition results.",
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
        help="In execute mode, overwrite existing daily product files.",
    )
    parser.add_argument(
        "--precipitation-policy",
        default=None,
        choices=sorted(SUPPORTED_PRECIPITATION_POLICIES),
        help=(
            "Enable precipitation preprocessing under this policy. "
            "Default: omitted, in which case hourly_precipitation requests are "
            "deferred (M004 behavior preserved)."
        ),
    )
    parser.add_argument(
        "--created-by",
        default="scripts/03_preprocess_daily.py",
        help="Free-form identifier recorded in the manifest's 'created_by' field.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        acquisition = load_acquisition_manifest(args.acquisition_manifest)
        download = load_download_manifest(args.download_manifest)
        region = load_region_manifest_for_preprocessing(args.region_manifest)
        assert_provenance_consistency(
            acquisition=acquisition, download=download, region=region
        )
        acquisition_hash = compute_manifest_hash(args.acquisition_manifest)
        download_hash = compute_manifest_hash(args.download_manifest)
        region_hash = compute_manifest_hash(args.region_manifest)
        joined = join_results_to_requests(acquisition=acquisition, download=download)
        joined = select_joined_records(
            joined, request_ids=args.request_ids, limit=args.limit
        )
        if args.mode == MODE_DRY_RUN:
            results = plan_results(
                joined,
                output_root=args.output_root,
                precipitation_policy=args.precipitation_policy,
            )
        else:
            region_geojson = load_region_geojson(region, repo_root=REPO_ROOT)
            results = execute_results(
                joined,
                output_root=args.output_root,
                region_geojson=region_geojson,
                overwrite=args.overwrite,
                precipitation_policy=args.precipitation_policy,
            )
        manifest = build_preprocessing_manifest(
            acquisition_manifest=acquisition,
            acquisition_manifest_path=args.acquisition_manifest,
            acquisition_manifest_hash=acquisition_hash,
            download_manifest_path=args.download_manifest,
            download_manifest_hash=download_hash,
            region_manifest=region,
            region_manifest_path=args.region_manifest,
            region_manifest_hash=region_hash,
            mode=args.mode,
            output_root=args.output_root,
            results=results,
            created_by=args.created_by,
            precipitation_policy=args.precipitation_policy,
        )
    except PreprocessingError as exc:
        print(f"preprocessing failed: {exc}", file=sys.stderr)
        return 2
    write_preprocessing_manifest(args.output, manifest)
    print(f"wrote preprocessing manifest: {args.output}")
    print(
        f"mode={manifest['mode']} "
        f"planned={manifest['planned_count']} "
        f"preprocessed={manifest['preprocessed_count']} "
        f"skipped={manifest['skipped_count']} "
        f"failed={manifest['failed_count']} "
        f"missing_input={manifest['missing_input_count']} "
        f"deferred={manifest['deferred_count']}"
    )
    print(f"execution_status: {manifest['execution_status']}")
    if args.mode == MODE_EXECUTE and (
        manifest["failed_count"] > 0 or manifest["missing_input_count"] > 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
