"""Owner-authorized live smoke-test runner for milestone 010.

Two modes:

- ``preflight`` (default): no network, no NetCDF writes. Validates
  the canonical M001-M009 manifest graph, looks up the configured
  smoke ``request_id`` in the M002 download manifest, and writes a
  deterministic plan to
  ``runs/dev_region/live_smoke_plan.json``.

- ``execute``: live ERA5-Land acquisition for **one** request +
  preprocessing + tmax-compatible indices + product validation.
  Requires an explicit confirmation token and a scratch
  ``--output-root`` other than ``runs/dev_region``.

Canonical preflight command (PowerShell, Windows):

    .\\.venv\\Scripts\\python.exe scripts\\07_run_live_smoke.py `
        --config configs/rbmn_local.json `
        --mode preflight `
        --output runs/dev_region/live_smoke_plan.json `
        --output-root runs/live_smoke_tmax_2000

Execute (owner-only; only run when authorized + CDS credentials in
``~/.cdsapirc`` or env):

    .\\.venv\\Scripts\\python.exe scripts\\07_run_live_smoke.py `
        --config configs/rbmn_local.json `
        --mode execute `
        --output runs/live_smoke_tmax_2000/live_smoke_run.json `
        --output-root runs/live_smoke_tmax_2000 `
        --confirm-live I_UNDERSTAND_THIS_USES_CDS

The script never imports ``cdsapi`` at module scope; live CDS calls
go through the existing M003 acquisition path, which loads
``cdsapi`` lazily only on its execute branch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.live_smoke import (
    CANONICAL_DEV_OUTPUT_ROOT,
    CONFIRMATION_TOKEN,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REQUEST_ID,
    MODE_EXECUTE,
    MODE_PREFLIGHT,
    PREFLIGHT_EXECUTION_STATUS_READY,
    SUPPORTED_MODES,
    LiveSmokeError,
    _resolves_to_canonical_dev_root,
    load_config,
    run_execute,
    run_preflight,
    write_plan,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Owner-authorized live smoke-test runner. Default mode is "
            "preflight (no network, no NetCDF). Execute mode runs one "
            "real ERA5-Land request + downstream products and requires "
            f"--confirm-live {CONFIRMATION_TOKEN!r} plus a scratch "
            "--output-root."
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
        help="preflight (default) plans without network; execute runs the real CDS call.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the live-smoke plan/report JSON. Defaults to "
            "config.run.live_smoke_plan_path."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
        help=(
            f"Scratch directory for the smoke run. Default: {DEFAULT_OUTPUT_ROOT}. "
            "Execute mode rejects runs/dev_region to protect canonical artifacts."
        ),
    )
    parser.add_argument(
        "--request-id",
        default=DEFAULT_REQUEST_ID,
        help=f"Request id to smoke. Default: {DEFAULT_REQUEST_ID}.",
    )
    parser.add_argument(
        "--confirm-live",
        default=None,
        help=(
            "Required for --mode execute. Pass the exact string "
            f"{CONFIRMATION_TOKEN!r} to acknowledge a real CDS call."
        ),
    )
    parser.add_argument(
        "--created-by",
        default="scripts/07_run_live_smoke.py",
        help="Free-form identifier recorded in the plan's 'created_by' field.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        output_path = args.output
        if output_path is None:
            cfg_path = config["run"].get("live_smoke_plan_path")
            if not cfg_path:
                raise LiveSmokeError(
                    "no --output given and config.run.live_smoke_plan_path is not set"
                )
            output_path = Path(cfg_path)
        if args.mode == MODE_EXECUTE:
            if args.confirm_live != CONFIRMATION_TOKEN:
                print(
                    f"live smoke refused: --mode execute requires "
                    f"--confirm-live {CONFIRMATION_TOKEN!r}",
                    file=sys.stderr,
                )
                return 2
            relative_match = (
                str(args.output_root).replace("\\", "/").rstrip("/")
                == CANONICAL_DEV_OUTPUT_ROOT
            )
            if relative_match or _resolves_to_canonical_dev_root(args.output_root):
                print(
                    f"live smoke refused: --output-root must not resolve to the "
                    f"canonical {CANONICAL_DEV_OUTPUT_ROOT!r} scratch root for execute mode",
                    file=sys.stderr,
                )
                return 2
            plan = run_execute(
                config,
                request_id=args.request_id,
                output_root=args.output_root,
                repo_root=REPO_ROOT,
                confirm_live=args.confirm_live,
            )
        else:
            plan = run_preflight(
                config,
                request_id=args.request_id,
                output_root=args.output_root,
            )
    except LiveSmokeError as exc:
        print(f"live smoke failed: {exc}", file=sys.stderr)
        return 2
    write_plan(output_path, plan)
    print(f"wrote live-smoke plan: {output_path}")
    print(
        f"mode={plan['mode']} "
        f"execution_status={plan['execution_status']} "
        f"request_id={plan['request_id']}"
    )
    if plan["mode"] == MODE_PREFLIGHT:
        return 0 if plan["execution_status"] == PREFLIGHT_EXECUTION_STATUS_READY else 1
    return 0 if plan["execution_status"] == "completed_live_smoke" else 1


if __name__ == "__main__":
    raise SystemExit(main())
