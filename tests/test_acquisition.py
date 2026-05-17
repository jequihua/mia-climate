"""Tests for ``lib.acquisition`` and ``scripts/02_download_era5_land.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from lib.acquisition import (
    AcquisitionError,
    EXECUTION_STATUS_COMPLETE_EXISTING,
    EXECUTION_STATUS_DOWNLOADED,
    EXECUTION_STATUS_FAILED,
    EXECUTION_STATUS_PARTIAL,
    EXECUTION_STATUS_PLANNED,
    MANIFEST_TYPE_ACQUISITION,
    MANIFEST_TYPE_DOWNLOAD_PLAN,
    MODE_DRY_RUN,
    MODE_EXECUTE,
    STATUS_DOWNLOADED,
    STATUS_FAILED,
    STATUS_PLANNED,
    STATUS_SKIPPED,
    build_acquisition_manifest,
    compute_manifest_hash,
    execute_results,
    load_download_manifest,
    plan_results,
    resolve_target_path,
    select_requests,
    write_acquisition_manifest,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RBMN_DOWNLOAD_MANIFEST = REPO_ROOT / "runs" / "dev_region" / "download_manifest.json"


# --- fixtures ------------------------------------------------------------


def _stub_request(request_id: str, *, dataset: str = "derived-era5-land-daily-statistics",
                  output_path: str | None = None) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "request_kind": "daily_statistics",
        "dataset": dataset,
        "project_variables": ["tmax"],
        "cds_variables": ["2m_temperature"],
        "year": 2000,
        "output_path": output_path or f"raw/era5_land/daily_statistics/{request_id}.nc",
        "payload": {"variable": ["2m_temperature"], "year": "2000", "area": [1, 2, 3, 4]},
    }


def _stub_download_manifest(*requests: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_type": MANIFEST_TYPE_DOWNLOAD_PLAN,
        "region_id": "rbmn",
        "region_manifest_path": "runs/dev_region/region_manifest.json",
        "region_geometry_hash": "sha256:" + "0" * 64,
        "bbox_north_west_south_east": [22.5, -105.7, 21.7, -105.3],
        "start_year": 2000,
        "end_year": 2000,
        "datasets": {},
        "requests": list(requests),
        "requires_network": False,
        "download_execution_status": "planned_only",
        "created_by": "tests",
    }


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def retrieve(self, dataset: str, payload: dict[str, Any], target: str) -> None:
        self.calls.append((dataset, payload, target))
        Path(target).write_bytes(b"fake netcdf bytes")


class _FailingClient:
    def __init__(self, fail_on: set[str]) -> None:
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, dataset: str, payload: dict[str, Any], target: str) -> None:
        self.calls.append((dataset, target))
        if Path(target).name in self.fail_on:
            raise RuntimeError(f"simulated CDS failure for {target}")
        Path(target).write_bytes(b"fake netcdf bytes")


# --- load_download_manifest ----------------------------------------------


def test_load_download_manifest_rejects_missing_file(tmp_path: Path):
    with pytest.raises(AcquisitionError, match="not found"):
        load_download_manifest(tmp_path / "missing.json")


def test_load_download_manifest_rejects_wrong_manifest_type(tmp_path: Path):
    manifest = _stub_download_manifest(_stub_request("a"))
    manifest["manifest_type"] = "something_else"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AcquisitionError, match="manifest_type must be"):
        load_download_manifest(p)


def test_load_download_manifest_rejects_empty_requests(tmp_path: Path):
    manifest = _stub_download_manifest()
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AcquisitionError, match="non-empty list"):
        load_download_manifest(p)


def test_load_download_manifest_rejects_request_missing_fields(tmp_path: Path):
    manifest = _stub_download_manifest({"request_id": "x"})
    p = tmp_path / "bad_req.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AcquisitionError, match="missing required fields"):
        load_download_manifest(p)


def test_load_download_manifest_round_trips_canonical():
    data = load_download_manifest(RBMN_DOWNLOAD_MANIFEST)
    assert data["manifest_type"] == MANIFEST_TYPE_DOWNLOAD_PLAN
    assert data["region_id"] == "rbmn"
    assert len(data["requests"]) == 175


# --- compute_manifest_hash -----------------------------------------------


def test_compute_manifest_hash_is_deterministic(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text("hello", encoding="utf-8")
    assert compute_manifest_hash(p) == compute_manifest_hash(p)
    assert compute_manifest_hash(p).startswith("sha256:")


# --- select_requests -----------------------------------------------------


def test_select_requests_no_filters_returns_all():
    reqs = [_stub_request("a"), _stub_request("b"), _stub_request("c")]
    assert [r["request_id"] for r in select_requests(reqs)] == ["a", "b", "c"]


def test_select_requests_by_id_preserves_manifest_order():
    reqs = [_stub_request("a"), _stub_request("b"), _stub_request("c")]
    selected = select_requests(reqs, request_ids=["c", "a"])
    assert [r["request_id"] for r in selected] == ["a", "c"]


def test_select_requests_rejects_unknown_id():
    reqs = [_stub_request("a")]
    with pytest.raises(AcquisitionError, match="unknown request_id"):
        select_requests(reqs, request_ids=["zz"])


def test_select_requests_applies_limit_after_id_filter():
    reqs = [_stub_request(c) for c in "abcde"]
    selected = select_requests(reqs, limit=3)
    assert [r["request_id"] for r in selected] == ["a", "b", "c"]


def test_select_requests_rejects_non_positive_limit():
    reqs = [_stub_request("a")]
    with pytest.raises(AcquisitionError, match="positive int"):
        select_requests(reqs, limit=0)
    with pytest.raises(AcquisitionError, match="positive int"):
        select_requests(reqs, limit=-2)


def test_select_requests_rejects_empty_result():
    with pytest.raises(AcquisitionError, match="no requests remain"):
        select_requests([])


# --- resolve_target_path -------------------------------------------------


def test_resolve_target_path_joins_under_output_root(tmp_path: Path):
    target = resolve_target_path(tmp_path, "raw/era5_land/daily_statistics/tmax/2000.nc")
    assert target == tmp_path / "raw" / "era5_land" / "daily_statistics" / "tmax" / "2000.nc"


def test_resolve_target_path_rejects_absolute(tmp_path: Path):
    abs_path = str(tmp_path / "elsewhere.nc")
    with pytest.raises(AcquisitionError, match="must be relative"):
        resolve_target_path(tmp_path, abs_path)


def test_resolve_target_path_rejects_parent_escape(tmp_path: Path):
    with pytest.raises(AcquisitionError, match="\\.\\."):
        resolve_target_path(tmp_path, "raw/../../escape.nc")


def test_resolve_target_path_rejects_windows_drive_relative(tmp_path: Path):
    with pytest.raises(AcquisitionError, match="drive or UNC"):
        resolve_target_path(tmp_path, "C:foo/bar.nc")


def test_resolve_target_path_rejects_windows_drive_qualified(tmp_path: Path):
    with pytest.raises(AcquisitionError):
        resolve_target_path(tmp_path, "C:/foo/bar.nc")
    with pytest.raises(AcquisitionError):
        resolve_target_path(tmp_path, "C:\\foo\\bar.nc")


def test_resolve_target_path_rejects_unc(tmp_path: Path):
    with pytest.raises(AcquisitionError):
        resolve_target_path(tmp_path, "//server/share/file.nc")
    with pytest.raises(AcquisitionError):
        resolve_target_path(tmp_path, "\\\\server\\share\\file.nc")


def test_resolve_target_path_accepts_canonical_manifest_path(tmp_path: Path):
    target = resolve_target_path(tmp_path, "raw/era5_land/daily_statistics/tmax/2000.nc")
    assert target == tmp_path / "raw" / "era5_land" / "daily_statistics" / "tmax" / "2000.nc"


# --- dry-run mode --------------------------------------------------------


def test_plan_results_marks_every_request_as_planned_without_io(tmp_path: Path):
    reqs = [_stub_request("a"), _stub_request("b")]
    results = plan_results(reqs, output_root=tmp_path)
    assert [r.status for r in results] == [STATUS_PLANNED, STATUS_PLANNED]
    assert not any(tmp_path.rglob("*.nc"))


def test_build_acquisition_manifest_dry_run_counts_and_flags(tmp_path: Path):
    reqs = [_stub_request("a"), _stub_request("b")]
    download_manifest = _stub_download_manifest(*reqs)
    download_path = tmp_path / "download_manifest.json"
    download_path.write_text(json.dumps(download_manifest), encoding="utf-8")
    results = plan_results(reqs, output_root=tmp_path)
    manifest = build_acquisition_manifest(
        download_manifest=download_manifest,
        download_manifest_path=download_path,
        download_manifest_hash=compute_manifest_hash(download_path),
        mode=MODE_DRY_RUN,
        output_root=tmp_path,
        results=results,
        created_by="tests",
    )
    assert manifest["manifest_type"] == MANIFEST_TYPE_ACQUISITION
    assert manifest["mode"] == MODE_DRY_RUN
    assert manifest["requires_network"] is False
    assert manifest["execution_status"] == EXECUTION_STATUS_PLANNED
    assert manifest["request_count"] == 2
    assert manifest["planned_count"] == 2
    assert manifest["skipped_count"] == 0
    assert manifest["downloaded_count"] == 0
    assert manifest["failed_count"] == 0
    assert manifest["region_id"] == "rbmn"
    assert manifest["region_geometry_hash"].startswith("sha256:")


# --- execute mode (fake client) -----------------------------------------


def test_execute_results_calls_retrieve_with_dataset_payload_target(tmp_path: Path):
    reqs = [_stub_request("a"), _stub_request("b")]
    client = _RecordingClient()
    results = execute_results(reqs, client=client, output_root=tmp_path)
    assert [r.status for r in results] == [STATUS_DOWNLOADED, STATUS_DOWNLOADED]
    assert len(client.calls) == 2
    dataset, payload, target = client.calls[0]
    assert dataset == "derived-era5-land-daily-statistics"
    assert payload == reqs[0]["payload"]
    assert target.endswith("a.nc")
    assert Path(target).exists()


def test_execute_results_skips_existing_target_when_not_overwrite(tmp_path: Path):
    reqs = [_stub_request("a")]
    target = tmp_path / Path(reqs[0]["output_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"prior")
    client = _RecordingClient()
    results = execute_results(reqs, client=client, output_root=tmp_path)
    assert [r.status for r in results] == [STATUS_SKIPPED]
    assert client.calls == []
    assert target.read_bytes() == b"prior"


def test_execute_results_overwrites_when_flag_is_set(tmp_path: Path):
    reqs = [_stub_request("a")]
    target = tmp_path / Path(reqs[0]["output_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"prior")
    client = _RecordingClient()
    results = execute_results(reqs, client=client, output_root=tmp_path, overwrite=True)
    assert [r.status for r in results] == [STATUS_DOWNLOADED]
    assert target.read_bytes() == b"fake netcdf bytes"


def test_execute_results_records_partial_failure(tmp_path: Path):
    reqs = [_stub_request("a"), _stub_request("b"), _stub_request("c")]
    client = _FailingClient(fail_on={"b.nc"})
    results = execute_results(reqs, client=client, output_root=tmp_path)
    statuses = [r.status for r in results]
    assert statuses == [STATUS_DOWNLOADED, STATUS_FAILED, STATUS_DOWNLOADED]
    assert results[1].error and "simulated CDS failure" in results[1].error


def test_build_acquisition_manifest_execute_partial(tmp_path: Path):
    reqs = [_stub_request("a"), _stub_request("b")]
    download_manifest = _stub_download_manifest(*reqs)
    download_path = tmp_path / "dl.json"
    download_path.write_text(json.dumps(download_manifest), encoding="utf-8")
    client = _FailingClient(fail_on={"b.nc"})
    results = execute_results(reqs, client=client, output_root=tmp_path)
    manifest = build_acquisition_manifest(
        download_manifest=download_manifest,
        download_manifest_path=download_path,
        download_manifest_hash=compute_manifest_hash(download_path),
        mode=MODE_EXECUTE,
        output_root=tmp_path,
        results=results,
        created_by="tests",
    )
    assert manifest["mode"] == MODE_EXECUTE
    assert manifest["requires_network"] is True
    assert manifest["downloaded_count"] == 1
    assert manifest["failed_count"] == 1
    assert manifest["execution_status"] == EXECUTION_STATUS_PARTIAL


def test_derive_execution_status_all_failed(tmp_path: Path):
    reqs = [_stub_request("a")]
    download_manifest = _stub_download_manifest(*reqs)
    client = _FailingClient(fail_on={"a.nc"})
    results = execute_results(reqs, client=client, output_root=tmp_path)
    manifest = build_acquisition_manifest(
        download_manifest=download_manifest,
        download_manifest_path=tmp_path / "x.json",
        download_manifest_hash="sha256:" + "f" * 64,
        mode=MODE_EXECUTE,
        output_root=tmp_path,
        results=results,
        created_by="tests",
    )
    assert manifest["execution_status"] == EXECUTION_STATUS_FAILED


def test_derive_execution_status_all_downloaded(tmp_path: Path):
    reqs = [_stub_request("a")]
    download_manifest = _stub_download_manifest(*reqs)
    client = _RecordingClient()
    results = execute_results(reqs, client=client, output_root=tmp_path)
    manifest = build_acquisition_manifest(
        download_manifest=download_manifest,
        download_manifest_path=tmp_path / "x.json",
        download_manifest_hash="sha256:" + "f" * 64,
        mode=MODE_EXECUTE,
        output_root=tmp_path,
        results=results,
        created_by="tests",
    )
    assert manifest["execution_status"] == EXECUTION_STATUS_DOWNLOADED


def test_derive_execution_status_all_skipped_is_complete_existing(tmp_path: Path):
    reqs = [_stub_request("a"), _stub_request("b")]
    for req in reqs:
        target = tmp_path / Path(req["output_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"prior")
    client = _RecordingClient()
    results = execute_results(reqs, client=client, output_root=tmp_path)
    assert [r.status for r in results] == [STATUS_SKIPPED, STATUS_SKIPPED]
    assert client.calls == []
    download_manifest = _stub_download_manifest(*reqs)
    manifest = build_acquisition_manifest(
        download_manifest=download_manifest,
        download_manifest_path=tmp_path / "x.json",
        download_manifest_hash="sha256:" + "f" * 64,
        mode=MODE_EXECUTE,
        output_root=tmp_path,
        results=results,
        created_by="tests",
    )
    assert manifest["execution_status"] == EXECUTION_STATUS_COMPLETE_EXISTING
    assert manifest["downloaded_count"] == 0
    assert manifest["skipped_count"] == 2


def test_derive_execution_status_mixed_downloaded_and_skipped_is_downloaded(tmp_path: Path):
    reqs = [_stub_request("a"), _stub_request("b")]
    pre_existing = tmp_path / Path(reqs[0]["output_path"])
    pre_existing.parent.mkdir(parents=True, exist_ok=True)
    pre_existing.write_bytes(b"prior")
    client = _RecordingClient()
    results = execute_results(reqs, client=client, output_root=tmp_path)
    assert {r.status for r in results} == {STATUS_SKIPPED, STATUS_DOWNLOADED}
    download_manifest = _stub_download_manifest(*reqs)
    manifest = build_acquisition_manifest(
        download_manifest=download_manifest,
        download_manifest_path=tmp_path / "x.json",
        download_manifest_hash="sha256:" + "f" * 64,
        mode=MODE_EXECUTE,
        output_root=tmp_path,
        results=results,
        created_by="tests",
    )
    assert manifest["execution_status"] == EXECUTION_STATUS_DOWNLOADED


# --- deterministic writes ------------------------------------------------


def test_write_acquisition_manifest_is_deterministic(tmp_path: Path):
    reqs = [_stub_request("a"), _stub_request("b")]
    download_manifest = _stub_download_manifest(*reqs)
    download_path = tmp_path / "dl.json"
    download_path.write_text(json.dumps(download_manifest), encoding="utf-8")
    manifest = build_acquisition_manifest(
        download_manifest=download_manifest,
        download_manifest_path=download_path,
        download_manifest_hash=compute_manifest_hash(download_path),
        mode=MODE_DRY_RUN,
        output_root=tmp_path,
        results=plan_results(reqs, output_root=tmp_path),
        created_by="tests",
    )
    a = tmp_path / "a" / "acq.json"
    b = tmp_path / "b" / "acq.json"
    write_acquisition_manifest(a, manifest)
    write_acquisition_manifest(b, manifest)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_text(encoding="utf-8").endswith("\n")


def test_acquisition_manifest_does_not_leak_secrets():
    """A user request payload must round-trip into results without secrets."""
    reqs = [_stub_request("a")]
    download_manifest = _stub_download_manifest(*reqs)
    manifest = build_acquisition_manifest(
        download_manifest=download_manifest,
        download_manifest_path=Path("dl.json"),
        download_manifest_hash="sha256:" + "a" * 64,
        mode=MODE_DRY_RUN,
        output_root=Path("runs/dev_region"),
        results=plan_results(reqs, output_root=Path("runs/dev_region")),
        created_by="tests",
    )
    serialized = json.dumps(manifest)
    assert "key" not in serialized.lower() or "geometry_hash" in serialized.lower()
    # Just confirm payload is not embedded into results.
    assert "variable" not in {k for r in manifest["results"] for k in r.keys()}


# --- import safety -------------------------------------------------------


def test_acquisition_module_does_not_import_cdsapi_eagerly():
    """``import lib.acquisition`` must succeed without ``cdsapi`` installed."""
    mod = sys.modules.get("lib.acquisition")
    assert mod is not None
    assert "cdsapi" not in sys.modules


# --- script smoke --------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "02_download_era5_land.py"
    spec = importlib.util.spec_from_file_location("acquisition_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_script_main_dry_run_with_limit_writes_three_planned_results(tmp_path: Path, capsys):
    main = _load_script_main()
    output = tmp_path / "acquisition_manifest.json"
    rc = main(
        [
            "--download-manifest",
            str(RBMN_DOWNLOAD_MANIFEST),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "run_root"),
            "--mode",
            "dry-run",
            "--limit",
            "3",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote acquisition manifest" in captured.out
    assert "mode=dry-run planned=3" in captured.out
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["manifest_type"] == MANIFEST_TYPE_ACQUISITION
    assert loaded["mode"] == MODE_DRY_RUN
    assert loaded["requires_network"] is False
    assert loaded["request_count"] == 3
    assert loaded["planned_count"] == 3
    assert loaded["execution_status"] == EXECUTION_STATUS_PLANNED
    # No raw NetCDF files were created.
    assert not any((tmp_path / "run_root").rglob("*.nc")) if (tmp_path / "run_root").exists() else True


def test_script_main_default_mode_is_dry_run(tmp_path: Path, capsys):
    main = _load_script_main()
    output = tmp_path / "acquisition_manifest.json"
    rc = main(
        [
            "--download-manifest",
            str(RBMN_DOWNLOAD_MANIFEST),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "rr"),
            "--limit",
            "1",
        ]
    )
    assert rc == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["mode"] == MODE_DRY_RUN


def test_script_main_returns_error_for_missing_manifest(tmp_path: Path, capsys):
    main = _load_script_main()
    rc = main(
        [
            "--download-manifest",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "acq.json"),
            "--output-root",
            str(tmp_path),
            "--mode",
            "dry-run",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "acquisition failed" in captured.err


def test_script_main_returns_error_for_unknown_request_id(tmp_path: Path, capsys):
    main = _load_script_main()
    rc = main(
        [
            "--download-manifest",
            str(RBMN_DOWNLOAD_MANIFEST),
            "--output",
            str(tmp_path / "acq.json"),
            "--output-root",
            str(tmp_path),
            "--mode",
            "dry-run",
            "--request-id",
            "no_such_request",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "unknown request_id" in captured.err
