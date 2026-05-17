"""Plan or compute annual climate indices for M005 (temperature) and M007 (precipitation).

Consumes a preprocessing manifest (M004 daily-statistics for the
temperature family, or M006 daily precipitation for the precipitation
family) and emits a deterministic ``index_manifest.json``. Dry-run
(default) writes only the manifest; execute mode opens local daily
standard products and writes annual index NetCDF outputs under
``runs/{run_id}/derived/indices/``.

The family is selected with ``--index-family``:

- ``temperature`` (default, M005): the seven simple temperature
  indices over ``tmax`` / ``tmin`` / ``tmean``. Default is
  temperature so the M005 reference command produces a
  byte-identical manifest.
- ``precipitation`` (M007): the seven ETCCDI-style precipitation
  indices over daily ``pr``. Use against the M006 preprocessing
  manifest ``runs/dev_region/preprocessing_manifest_precipitation.json``.
- ``all``: temperature first, then precipitation.

M005 reference dry-run (PowerShell, Windows):

    .\\.venv\\Scripts\\python.exe scripts\\04_compute_indices.py `
        --preprocessing-manifest runs/dev_region/preprocessing_manifest.json `
        --output runs/dev_region/index_manifest.json `
        --output-root runs/dev_region `
        --mode dry-run

M007 reference dry-run (precipitation):

    .\\.venv\\Scripts\\python.exe scripts\\04_compute_indices.py `
        --preprocessing-manifest runs/dev_region/preprocessing_manifest_precipitation.json `
        --output runs/dev_region/index_manifest_precipitation.json `
        --output-root runs/dev_region `
        --mode dry-run `
        --index-family precipitation

This script does not read raw ERA5-Land files, does not call Copernicus,
and does not compute wind, percentile (``TX90p`` / ``TN10p``), or
spell-duration (``WSDI``) indices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.index_manifest import (
    INDEX_FAMILY_TEMPERATURE,
    MODE_DRY_RUN,
    MODE_EXECUTE,
    SUPPORTED_INDEX_FAMILIES,
    VALID_MODES,
    IndexManifestError,
    build_index_manifest,
    compute_manifest_hash,
    execute_index_results,
    load_preprocessing_manifest,
    plan_index_results,
    select_index_specs,
    specs_for_family,
    write_index_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or compute annual climate indices from a preprocessing "
            "manifest. Default --index-family is 'temperature' (M005, "
            "against an M004 preprocessing manifest); pass "
            "--index-family precipitation to compute the seven ETCCDI-style "
            "precipitation indices (M007, against an M006 preprocessing "
            "manifest), or --index-family all for both. Default mode is "
            "dry-run; execute opens local daily NetCDF products and writes "
            "annual index outputs."
        )
    )
    parser.add_argument(
        "--preprocessing-manifest",
        required=True,
        type=Path,
        help="Path to an M004 preprocessing manifest.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the index manifest JSON (parent dirs are created).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory under which annual index output paths are resolved.",
    )
    parser.add_argument(
        "--mode",
        default=MODE_DRY_RUN,
        choices=sorted(VALID_MODES),
        help=(
            "dry-run (default) plans only; execute opens daily NetCDF "
            "products and writes annual index outputs under "
            "runs/{run_id}/derived/indices/."
        ),
    )
    parser.add_argument(
        "--index-family",
        default=INDEX_FAMILY_TEMPERATURE,
        choices=sorted(SUPPORTED_INDEX_FAMILIES),
        help=(
            "Index family to plan/compute. Default is 'temperature' (M005) "
            "so the M005 reference command produces a byte-identical manifest. "
            "Use 'precipitation' for the M007 family against an M006 "
            "preprocessing manifest, or 'all' to plan both families."
        ),
    )
    parser.add_argument(
        "--index-id",
        action="append",
        default=None,
        dest="index_ids",
        help="Restrict to this index_id; repeatable. Defaults to every spec in the selected family.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many indices (after --index-id filtering).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="In execute mode, overwrite existing index NetCDF outputs.",
    )
    parser.add_argument(
        "--created-by",
        default="scripts/04_compute_indices.py",
        help="Free-form identifier recorded in the manifest's 'created_by' field.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        preprocessing = load_preprocessing_manifest(args.preprocessing_manifest)
        preprocessing_hash = compute_manifest_hash(args.preprocessing_manifest)
        specs = select_index_specs(
            specs_for_family(args.index_family),
            index_ids=args.index_ids,
            limit=args.limit,
        )
        if args.mode == MODE_DRY_RUN:
            results = plan_index_results(
                specs,
                preprocessing=preprocessing,
                output_root=args.output_root,
            )
        else:
            results = execute_index_results(
                specs,
                preprocessing=preprocessing,
                output_root=args.output_root,
                overwrite=args.overwrite,
            )
        manifest = build_index_manifest(
            preprocessing_manifest=preprocessing,
            preprocessing_manifest_path=args.preprocessing_manifest,
            preprocessing_manifest_hash=preprocessing_hash,
            mode=args.mode,
            output_root=args.output_root,
            results=results,
            created_by=args.created_by,
        )
    except IndexManifestError as exc:
        print(f"index computation failed: {exc}", file=sys.stderr)
        return 2
    write_index_manifest(args.output, manifest)
    print(f"wrote index manifest: {args.output}")
    print(
        f"mode={manifest['mode']} "
        f"planned={manifest['planned_count']} "
        f"computed={manifest['computed_count']} "
        f"skipped={manifest['skipped_count']} "
        f"failed={manifest['failed_count']} "
        f"missing_input={manifest['missing_input_count']}"
    )
    print(f"execution_status: {manifest['execution_status']}")
    if args.mode == MODE_EXECUTE and (
        manifest["failed_count"] > 0 or manifest["missing_input_count"] > 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
