"""ERA5-Land acquisition helpers for milestone 003.

Consumes the M002 download manifest produced by ``lib.download_plan`` and
orchestrates either a dry-run plan or a live execution through a passed-in
client object that implements ``retrieve(dataset, payload, target)``.

``cdsapi`` is imported only inside the live-execution helper so this module
remains importable (and testable) without it. Automated tests inject fake
clients; live execution is gated behind the script's ``--mode execute``
flag.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Protocol

MANIFEST_TYPE_DOWNLOAD_PLAN = "era5_land_download_plan"
MANIFEST_TYPE_ACQUISITION = "era5_land_acquisition_run"

MODE_DRY_RUN = "dry-run"
MODE_EXECUTE = "execute"
VALID_MODES = frozenset({MODE_DRY_RUN, MODE_EXECUTE})

STATUS_PLANNED = "planned"
STATUS_SKIPPED = "skipped_existing"
STATUS_DOWNLOADED = "downloaded"
STATUS_FAILED = "failed"

EXECUTION_STATUS_PLANNED = "planned_only"
EXECUTION_STATUS_PARTIAL = "partial"
EXECUTION_STATUS_DOWNLOADED = "downloaded"
EXECUTION_STATUS_COMPLETE_EXISTING = "complete_existing"
EXECUTION_STATUS_FAILED = "failed"

DOWNLOAD_MANIFEST_REQUIRED_FIELDS = (
    "manifest_type",
    "region_id",
    "region_geometry_hash",
    "bbox_north_west_south_east",
    "start_year",
    "end_year",
    "datasets",
    "requests",
    "requires_network",
    "download_execution_status",
)

REQUEST_REQUIRED_FIELDS = (
    "request_id",
    "dataset",
    "output_path",
    "payload",
)


class AcquisitionError(ValueError):
    """Raised when an acquisition run cannot be planned or executed."""


class RetrieveClient(Protocol):
    """Minimal interface the acquisition orchestrator depends on.

    ``cdsapi.Client`` already satisfies this shape. Tests provide a fake.
    """

    def retrieve(self, dataset: str, payload: dict[str, Any], target: str) -> Any:
        ...


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    dataset: str
    output_path: str
    target_path: str
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "request_id": self.request_id,
            "dataset": self.dataset,
            "output_path": self.output_path,
            "target_path": self.target_path,
            "status": self.status,
        }
        if self.error is not None:
            record["error"] = self.error
        return record


def compute_manifest_hash(path: Path) -> str:
    """Return a stable ``sha256:<hex>`` hash of the manifest file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_download_manifest(path: Path) -> dict[str, Any]:
    """Read and validate a download manifest produced by milestone 002."""
    if not path.exists():
        raise AcquisitionError(f"download manifest not found: {path}")
    if not path.is_file():
        raise AcquisitionError(f"download manifest path is not a file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AcquisitionError(f"download manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AcquisitionError(
            f"download manifest must be a JSON object, got {type(data).__name__}: {path}"
        )
    missing = [f for f in DOWNLOAD_MANIFEST_REQUIRED_FIELDS if f not in data]
    if missing:
        raise AcquisitionError(
            f"download manifest {path} is missing required fields: {missing}"
        )
    if data["manifest_type"] != MANIFEST_TYPE_DOWNLOAD_PLAN:
        raise AcquisitionError(
            f"download manifest manifest_type must be {MANIFEST_TYPE_DOWNLOAD_PLAN!r}, "
            f"got {data['manifest_type']!r}"
        )
    requests = data["requests"]
    if not isinstance(requests, list) or not requests:
        raise AcquisitionError(
            f"download manifest 'requests' must be a non-empty list, got {type(requests).__name__}"
        )
    for index, request in enumerate(requests):
        _validate_request_record(request, index=index)
    return data


def _validate_request_record(request: Any, *, index: int) -> None:
    if not isinstance(request, dict):
        raise AcquisitionError(
            f"request at index {index} must be an object, got {type(request).__name__}"
        )
    missing = [f for f in REQUEST_REQUIRED_FIELDS if f not in request]
    if missing:
        raise AcquisitionError(
            f"request at index {index} is missing required fields: {missing}"
        )
    if not isinstance(request["request_id"], str) or not request["request_id"]:
        raise AcquisitionError(f"request at index {index} has empty/non-string request_id")
    if not isinstance(request["dataset"], str) or not request["dataset"]:
        raise AcquisitionError(
            f"request {request['request_id']!r} has empty/non-string dataset"
        )
    output_path = request["output_path"]
    if not isinstance(output_path, str) or not output_path:
        raise AcquisitionError(
            f"request {request['request_id']!r} has empty/non-string output_path"
        )
    if not isinstance(request["payload"], dict):
        raise AcquisitionError(
            f"request {request['request_id']!r} payload must be an object, "
            f"got {type(request['payload']).__name__}"
        )


def select_requests(
    requests: list[dict[str, Any]],
    *,
    request_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter ``requests`` by an optional id allow-list and/or count limit."""
    selected = list(requests)
    if request_ids is not None:
        wanted = list(request_ids)
        if not wanted:
            raise AcquisitionError("--request-id filter provided but list is empty")
        wanted_set = set(wanted)
        available = {r["request_id"] for r in selected}
        unknown = sorted(wanted_set - available)
        if unknown:
            raise AcquisitionError(f"unknown request_id(s): {unknown}")
        selected = [r for r in selected if r["request_id"] in wanted_set]
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise AcquisitionError(f"limit must be a positive int, got {type(limit).__name__}")
        if limit <= 0:
            raise AcquisitionError(f"limit must be a positive int, got {limit}")
        selected = selected[:limit]
    if not selected:
        raise AcquisitionError("no requests remain after filtering")
    return selected


def resolve_target_path(output_root: Path, output_path: str) -> Path:
    """Join a manifest-relative ``output_path`` under ``output_root`` safely.

    Rejects absolute paths, parent-traversal (``..``) segments, Windows
    drive-relative paths (``C:foo/bar.nc``), drive-qualified paths
    (``C:/foo/bar.nc``), and UNC paths (``//server/share/x.nc``) so a
    malformed download manifest cannot write outside the chosen run root.
    Windows-style escapes are checked even on POSIX hosts, because the
    same manifest may later be executed on Windows.
    """
    if not isinstance(output_path, str) or not output_path:
        raise AcquisitionError(f"output_path must be a non-empty string, got {output_path!r}")
    candidate = Path(output_path)
    win_candidate = PureWindowsPath(output_path)
    if candidate.is_absolute() or win_candidate.is_absolute():
        raise AcquisitionError(f"request output_path must be relative: {output_path!r}")
    if candidate.drive or win_candidate.drive:
        raise AcquisitionError(
            f"request output_path must not carry a drive or UNC prefix: {output_path!r}"
        )
    if any(part == ".." for part in candidate.parts) or any(
        part == ".." for part in win_candidate.parts
    ):
        raise AcquisitionError(
            f"request output_path must not contain '..' segments: {output_path!r}"
        )
    return output_root / candidate


def plan_results(
    requests: list[dict[str, Any]],
    *,
    output_root: Path,
) -> list[RequestResult]:
    """Build dry-run result records: each request marked ``planned``, no I/O."""
    results: list[RequestResult] = []
    for request in requests:
        target = resolve_target_path(output_root, request["output_path"])
        results.append(
            RequestResult(
                request_id=request["request_id"],
                dataset=request["dataset"],
                output_path=request["output_path"],
                target_path=_as_posix(target),
                status=STATUS_PLANNED,
            )
        )
    return results


def execute_results(
    requests: list[dict[str, Any]],
    *,
    client: RetrieveClient,
    output_root: Path,
    overwrite: bool = False,
) -> list[RequestResult]:
    """Call ``client.retrieve`` for each request, recording the outcome."""
    results: list[RequestResult] = []
    for request in requests:
        target = resolve_target_path(output_root, request["output_path"])
        target_posix = _as_posix(target)
        if target.exists() and not overwrite:
            results.append(
                RequestResult(
                    request_id=request["request_id"],
                    dataset=request["dataset"],
                    output_path=request["output_path"],
                    target_path=target_posix,
                    status=STATUS_SKIPPED,
                )
            )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.retrieve(request["dataset"], dict(request["payload"]), str(target))
        except Exception as exc:  # surface upstream library failures as a record
            results.append(
                RequestResult(
                    request_id=request["request_id"],
                    dataset=request["dataset"],
                    output_path=request["output_path"],
                    target_path=target_posix,
                    status=STATUS_FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(
            RequestResult(
                request_id=request["request_id"],
                dataset=request["dataset"],
                output_path=request["output_path"],
                target_path=target_posix,
                status=STATUS_DOWNLOADED,
            )
        )
    return results


def derive_execution_status(mode: str, results: list[RequestResult]) -> str:
    """Summarize per-request outcomes into a manifest-level execution_status.

    Status semantics in execute mode:

    - ``downloaded``: at least one request actually downloaded; any others
      were skipped because their targets already existed. The run did work.
    - ``complete_existing``: every selected request was skipped because the
      target already existed. No new files were fetched; the requested set
      is locally complete. Distinct from ``downloaded`` so a future
      orchestrator can detect "nothing to do" without re-parsing counts.
    - ``failed``: every selected request failed.
    - ``partial``: at least one failure mixed with at least one success or
      skip.
    """
    if mode == MODE_DRY_RUN:
        return EXECUTION_STATUS_PLANNED
    statuses = {r.status for r in results}
    has_failure = STATUS_FAILED in statuses
    has_success = STATUS_DOWNLOADED in statuses
    has_skipped = STATUS_SKIPPED in statuses
    if statuses == {STATUS_FAILED}:
        return EXECUTION_STATUS_FAILED
    if has_failure:
        return EXECUTION_STATUS_PARTIAL
    if statuses == {STATUS_SKIPPED}:
        return EXECUTION_STATUS_COMPLETE_EXISTING
    if has_success:
        return EXECUTION_STATUS_DOWNLOADED
    return EXECUTION_STATUS_PARTIAL


def build_acquisition_manifest(
    *,
    download_manifest: dict[str, Any],
    download_manifest_path: Path,
    download_manifest_hash: str,
    mode: str,
    output_root: Path,
    results: list[RequestResult],
    created_by: str,
) -> dict[str, Any]:
    """Assemble the deterministic acquisition manifest dict."""
    if mode not in VALID_MODES:
        raise AcquisitionError(f"mode {mode!r} is not one of {sorted(VALID_MODES)}")
    counts = {
        STATUS_PLANNED: 0,
        STATUS_SKIPPED: 0,
        STATUS_DOWNLOADED: 0,
        STATUS_FAILED: 0,
    }
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    manifest: dict[str, Any] = {
        "manifest_type": MANIFEST_TYPE_ACQUISITION,
        "download_manifest_path": _as_posix(download_manifest_path),
        "download_manifest_hash": download_manifest_hash,
        "region_id": download_manifest["region_id"],
        "region_geometry_hash": download_manifest["region_geometry_hash"],
        "mode": mode,
        "output_root": _as_posix(output_root),
        "request_count": len(results),
        "planned_count": counts[STATUS_PLANNED],
        "skipped_count": counts[STATUS_SKIPPED],
        "downloaded_count": counts[STATUS_DOWNLOADED],
        "failed_count": counts[STATUS_FAILED],
        "results": [r.to_dict() for r in results],
        "created_by": created_by,
        "requires_network": mode == MODE_EXECUTE,
        "execution_status": derive_execution_status(mode, results),
    }
    return manifest


def write_acquisition_manifest(output_path: Path, manifest: dict[str, Any]) -> None:
    """Write the acquisition manifest as deterministic JSON with a trailing newline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    output_path.write_text(text + "\n", encoding="utf-8")


def load_cdsapi_client() -> RetrieveClient:
    """Import ``cdsapi`` lazily and return a default ``cdsapi.Client``.

    Credentials come from the normal CDS discovery chain (``~/.cdsapirc`` or
    the ``CDSAPI_URL`` / ``CDSAPI_KEY`` environment variables). This module
    does not read or write secrets in repo files.
    """
    try:
        import cdsapi  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AcquisitionError(
            "cdsapi is required for --mode execute but is not installed. "
            "Install with: .venv/Scripts/python.exe -m pip install cdsapi"
        ) from exc
    return cdsapi.Client()


def _as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


__all__ = [
    "AcquisitionError",
    "DOWNLOAD_MANIFEST_REQUIRED_FIELDS",
    "EXECUTION_STATUS_COMPLETE_EXISTING",
    "EXECUTION_STATUS_DOWNLOADED",
    "EXECUTION_STATUS_FAILED",
    "EXECUTION_STATUS_PARTIAL",
    "EXECUTION_STATUS_PLANNED",
    "MANIFEST_TYPE_ACQUISITION",
    "MANIFEST_TYPE_DOWNLOAD_PLAN",
    "MODE_DRY_RUN",
    "MODE_EXECUTE",
    "REQUEST_REQUIRED_FIELDS",
    "RequestResult",
    "RetrieveClient",
    "STATUS_DOWNLOADED",
    "STATUS_FAILED",
    "STATUS_PLANNED",
    "STATUS_SKIPPED",
    "VALID_MODES",
    "build_acquisition_manifest",
    "compute_manifest_hash",
    "derive_execution_status",
    "execute_results",
    "load_cdsapi_client",
    "load_download_manifest",
    "plan_results",
    "resolve_target_path",
    "select_requests",
    "write_acquisition_manifest",
]
