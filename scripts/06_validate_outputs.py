"""Local validation / regression-foundation runner for milestone 009.

Reads the M008 pipeline manifest and the M001-M007 downstream manifests
it references, runs a stable ordered set of graph and side-effect
checks, and writes a deterministic ``validation_report.json``.

Reference command (PowerShell, Windows):

    .\\.venv\\Scripts\\python.exe scripts\\06_validate_outputs.py `
        --pipeline-manifest runs/dev_region/pipeline_manifest.json `
        --output runs/dev_region/validation_report.json `
        --output-root runs/dev_region `
        --mode dry-run

M009 dry-run validates the manifest graph and the dry-run side-effect
policy. NetCDF product schemas are checked by ``lib.validation``'s
``validate_daily_product`` / ``validate_index_product`` helpers, which
the canonical dry-run records as ``skipped`` (no products on disk
yet). Live CDS, Docker, cloud, and legacy-output numeric comparison
are out of scope for M009.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.validation import (
    MODE_DRY_RUN,
    SUPPORTED_MODES,
    ValidationError,
    build_validation_report,
    compute_file_hash,
    load_pipeline_manifest,
    run_validation,
    write_validation_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the M008 pipeline manifest graph + dry-run "
            "side-effect policy and write a deterministic "
            "validation_report.json. NetCDF product schemas are checked "
            "per-file by lib.validation helpers; the canonical dry-run "
            "records product checks as skipped since no products exist yet."
        )
    )
    parser.add_argument(
        "--pipeline-manifest",
        required=True,
        type=Path,
        help="Path to the M008 pipeline manifest (e.g. runs/dev_region/pipeline_manifest.json).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the validation report JSON (parent dirs are created).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Run output root scanned for side-effect violations (NetCDF / raw / intermediate / derived).",
    )
    parser.add_argument(
        "--mode",
        default=MODE_DRY_RUN,
        choices=sorted(SUPPORTED_MODES),
        help="Validation mode. M009 only supports dry-run.",
    )
    parser.add_argument(
        "--created-by",
        default="scripts/06_validate_outputs.py",
        help="Free-form identifier recorded in the report's 'created_by' field.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        pipeline_manifest_hash: str | None = None
        pipeline_manifest = None
        if args.pipeline_manifest.exists():
            pipeline_manifest_hash = compute_file_hash(args.pipeline_manifest)
            pipeline_manifest = load_pipeline_manifest(args.pipeline_manifest)
        results = run_validation(
            pipeline_manifest_path=args.pipeline_manifest,
            output_root=args.output_root,
            mode=args.mode,
        )
        report = build_validation_report(
            pipeline_manifest=pipeline_manifest,
            pipeline_manifest_path=args.pipeline_manifest,
            pipeline_manifest_hash=pipeline_manifest_hash,
            mode=args.mode,
            output_root=args.output_root,
            results=results,
            created_by=args.created_by,
        )
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2
    write_validation_report(args.output, report)
    print(f"wrote validation report: {args.output}")
    print(
        f"mode={report['mode']} "
        f"passed={report['passed_count']} "
        f"warning={report['warning_count']} "
        f"failed={report['failed_count']} "
        f"skipped={report['skipped_count']}"
    )
    print(f"execution_status: {report['execution_status']}")
    if report["execution_status"] == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
