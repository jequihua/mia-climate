"""Local dry-run pipeline runner for milestone 008.

Orchestrates the reviewed M001-M007 script boundaries in order against
a single config (default ``configs/rbmn_local.json``) and writes a
deterministic ``pipeline_manifest.json`` summarizing each step's
command argv, exit code, output path, and SHA-256 hash.

Reference dry-run (PowerShell, Windows):

    .\\.venv\\Scripts\\python.exe scripts\\05_run_pipeline.py `
        --config configs/rbmn_local.json `
        --mode dry-run `
        --output runs/dev_region/pipeline_manifest.json

This script does not call Copernicus, does not read or write NetCDF,
and does not run any milestone's execute mode. It is local orchestration
only: an auditable shim around the existing M00x scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.pipeline_runner import (
    MODE_DRY_RUN,
    SUPPORTED_MODES,
    PipelineRunnerError,
    build_pipeline_manifest,
    compute_file_hash,
    derive_execution_status,
    load_config,
    run_pipeline,
    write_pipeline_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical M001-M007 dry-run pipeline in order and "
            "write a deterministic pipeline_manifest.json summarizing each "
            "step. Dry-run only in M008; live CDS execution remains a "
            "separate owner-authorized step."
        )
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "rbmn_local.json"),
        type=Path,
        help="Path to the pipeline config JSON. Default: configs/rbmn_local.json.",
    )
    parser.add_argument(
        "--mode",
        default=MODE_DRY_RUN,
        choices=sorted(SUPPORTED_MODES),
        help="Pipeline mode. M008 only supports dry-run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the pipeline manifest JSON. Defaults to "
            "config.run.pipeline_manifest_path."
        ),
    )
    parser.add_argument(
        "--created-by",
        default="scripts/05_run_pipeline.py",
        help="Free-form identifier recorded in the manifest's 'created_by' field.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        config_hash = compute_file_hash(args.config)
        output_path = args.output
        if output_path is None:
            pipeline_path = config["run"].get("pipeline_manifest_path")
            if not pipeline_path:
                raise PipelineRunnerError(
                    "no --output given and config.run.pipeline_manifest_path is not set"
                )
            output_path = Path(pipeline_path)
        results = run_pipeline(config, repo_root=REPO_ROOT, mode=args.mode)
        manifest = build_pipeline_manifest(
            config=config,
            config_path=args.config,
            config_hash=config_hash,
            mode=args.mode,
            results=results,
            created_by=args.created_by,
        )
    except PipelineRunnerError as exc:
        print(f"pipeline runner failed: {exc}", file=sys.stderr)
        return 2
    write_pipeline_manifest(output_path, manifest)
    print(f"wrote pipeline manifest: {output_path}")
    print(
        f"mode={manifest['mode']} "
        f"succeeded={manifest['succeeded_count']} "
        f"failed={manifest['failed_count']} "
        f"skipped={manifest['skipped_count']}"
    )
    print(f"execution_status: {manifest['execution_status']}")
    if manifest["execution_status"] != derive_execution_status(results):
        # Defensive consistency check.
        return 2
    return 0 if manifest["execution_status"] == "completed_dry_run" else 1


if __name__ == "__main__":
    raise SystemExit(main())
