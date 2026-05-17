"""Live smoke audit runner for milestone 011.

Two modes:

- ``preflight`` (default): no network, no NetCDF reads. Hashes the M010
  preflight plan and the M010 safety-corrections review, confirms the
  expected scratch ``--expected-output-root`` is not the canonical
  ``runs/dev_region`` root, and writes a deterministic plan to
  ``runs/dev_region/live_smoke_audit_plan.json``.

- ``audit``: reads an M010 execute report under the scratch root,
  hashes the four expected product files, runs the M009 per-file
  validators on the daily and index NetCDFs, and writes a
  deterministic audit report.

Canonical preflight command (PowerShell, Windows):

    .\\.venv\\Scripts\\python.exe scripts\\08_audit_live_smoke.py `
        --config configs/rbmn_local.json `
        --mode preflight `
        --output runs/dev_region/live_smoke_audit_plan.json `
        --expected-output-root runs/live_smoke_tmax_2000

Audit command (owner runs after a successful M010 execute):

    .\\.venv\\Scripts\\python.exe scripts\\08_audit_live_smoke.py `
        --config configs/rbmn_local.json `
        --mode audit `
        --live-report runs/live_smoke_tmax_2000/live_smoke_report.json `
        --output runs/live_smoke_tmax_2000/live_smoke_audit_report.json `
        --expected-output-root runs/live_smoke_tmax_2000

The script never imports ``cdsapi`` at module scope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.live_smoke_audit import (
    AUDIT_EXECUTION_STATUS_PASSED,
    AUDIT_EXECUTION_STATUS_PASSED_WITH_WARNINGS,
    DEFAULT_AUDIT_REPORT_NAME,
    DEFAULT_EXPECTED_OUTPUT_ROOT,
    DEFAULT_LIVE_REPORT_NAME,
    DEFAULT_REQUEST_ID,
    MODE_AUDIT,
    MODE_PREFLIGHT,
    PREFLIGHT_EXECUTION_STATUS_READY,
    SUPPORTED_MODES,
    LiveSmokeAuditError,
    load_config,
    run_audit,
    run_preflight,
    write_plan,
)


DEFAULT_M010_PLAN_REL = "runs/dev_region/live_smoke_plan.json"
DEFAULT_M010_SAFETY_REVIEW_REL = (
    "05_governance/reviews/review_m010_live_smoke_safety_corrections.md"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M011 live smoke audit runner. Default mode is preflight "
            "(no network, no NetCDF). Audit mode reads an existing M010 "
            "execute report under the scratch output root."
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
        default=MODE_PREFLIGHT,
        choices=sorted(SUPPORTED_MODES),
        help="preflight (default) plans the audit; audit reads the live report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the audit plan/report JSON. Defaults to "
            "config.run.live_smoke_audit_plan_path in preflight mode and to "
            "<expected-output-root>/live_smoke_audit_report.json in audit mode."
        ),
    )
    parser.add_argument(
        "--expected-output-root",
        type=Path,
        default=Path(DEFAULT_EXPECTED_OUTPUT_ROOT),
        help=(
            f"Scratch directory the M010 execute run targeted. Default: "
            f"{DEFAULT_EXPECTED_OUTPUT_ROOT}. Must not resolve to runs/dev_region."
        ),
    )
    parser.add_argument(
        "--live-report",
        type=Path,
        default=None,
        help=(
            "Path to the M010 execute report. Defaults to "
            f"<expected-output-root>/{DEFAULT_LIVE_REPORT_NAME} in audit mode."
        ),
    )
    parser.add_argument(
        "--m010-plan",
        type=Path,
        default=Path(DEFAULT_M010_PLAN_REL),
        help=f"Path to the M010 preflight plan. Default: {DEFAULT_M010_PLAN_REL}.",
    )
    parser.add_argument(
        "--m010-safety-review",
        type=Path,
        default=Path(DEFAULT_M010_SAFETY_REVIEW_REL),
        help=(
            f"Path to the M010 safety-corrections review. "
            f"Default: {DEFAULT_M010_SAFETY_REVIEW_REL}."
        ),
    )
    parser.add_argument(
        "--request-id",
        default=DEFAULT_REQUEST_ID,
        help=f"Live smoke request id to audit. Default: {DEFAULT_REQUEST_ID}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        output_path = args.output
        if args.mode == MODE_PREFLIGHT:
            if output_path is None:
                cfg_path = config["run"].get("live_smoke_audit_plan_path")
                if not cfg_path:
                    raise LiveSmokeAuditError(
                        "no --output given and "
                        "config.run.live_smoke_audit_plan_path is not set"
                    )
                output_path = Path(cfg_path)
            plan = run_preflight(
                m010_plan_path=args.m010_plan,
                m010_safety_review_path=args.m010_safety_review,
                expected_output_root=args.expected_output_root,
                request_id=args.request_id,
            )
        else:
            live_report = args.live_report
            if live_report is None:
                live_report = args.expected_output_root / DEFAULT_LIVE_REPORT_NAME
            if output_path is None:
                output_path = args.expected_output_root / DEFAULT_AUDIT_REPORT_NAME
            plan = run_audit(
                expected_output_root=args.expected_output_root,
                live_report_path=live_report,
                request_id=args.request_id,
            )
    except LiveSmokeAuditError as exc:
        print(f"live smoke audit failed: {exc}", file=sys.stderr)
        return 2
    write_plan(output_path, plan)
    print(f"wrote live-smoke audit: {output_path}")
    print(
        f"mode={plan['mode']} "
        f"execution_status={plan['execution_status']} "
        f"request_id={plan['request_id']}"
    )
    if plan["mode"] == MODE_PREFLIGHT:
        return 0 if plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_READY else 1
    return 0 if plan["execution_status"] in (
        AUDIT_EXECUTION_STATUS_PASSED,
        AUDIT_EXECUTION_STATUS_PASSED_WITH_WARNINGS,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
