"""Tests for ``lib.indices_precipitation`` and the M007 wiring in
``lib.index_manifest`` / ``scripts/04_compute_indices.py``.

Synthetic 3-year fixtures pin the exact annual values for each
precipitation index. Run-length tests for CDD / CWD use small
hand-built boolean traces. M005 backward-compat is checked via the
canonical M004 preprocessing manifest at the CLI level.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from lib.index_manifest import (
    INDEX_FAMILY_ALL,
    INDEX_FAMILY_PRECIPITATION,
    INDEX_FAMILY_TEMPERATURE,
    MANIFEST_TYPE_INDEX,
    MANIFEST_TYPE_PREPROCESSING,
    MODE_DRY_RUN,
    PRECIPITATION_INDEX_SPECS,
    SUPPORTED_INDEX_FAMILIES,
    TEMPERATURE_INDEX_SPECS,
    IndexManifestError,
    plan_index_results,
    specs_for_family,
)
from lib.indices_precipitation import (
    INDEX_CDD,
    INDEX_CWD,
    INDEX_PRCPTOT,
    INDEX_R10MM,
    INDEX_R20MM,
    INDEX_R95P,
    INDEX_RX1DAY,
    R10_THRESHOLD_MM,
    R20_THRESHOLD_MM,
    R95P_BASELINE_POLICY,
    R95P_PERCENTILE,
    WET_DAY_THRESHOLD_MM,
    PrecipitationIndexError,
    compute_CDD,
    compute_CWD,
    compute_PRCPTOT,
    compute_R10mm,
    compute_R20mm,
    compute_R95p,
    compute_RX1day,
    compute_precipitation_index,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_M004_PREPROCESSING = (
    REPO_ROOT / "runs" / "dev_region" / "preprocessing_manifest.json"
)
CANONICAL_M005_INDEX = REPO_ROOT / "runs" / "dev_region" / "index_manifest.json"
CANONICAL_M006_PREPROCESSING = (
    REPO_ROOT / "runs" / "dev_region" / "preprocessing_manifest_precipitation.json"
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _daily_pr_dataarray(values_per_year: list[list[float]], *, name: str = "pr") -> xr.DataArray:
    """Build a 1D-in-time pr DataArray, ``daily`` cadence, ``mm/day`` units.

    ``values_per_year`` is a list of yearly slices; each slice is a list
    of daily mm values for that year, starting on Jan 1.
    """
    times: list[np.datetime64] = []
    flat: list[float] = []
    for year_index, year_values in enumerate(values_per_year):
        year = 2000 + year_index
        start = np.datetime64(f"{year}-01-01", "ns")
        for day in range(len(year_values)):
            times.append(start + np.timedelta64(day, "D"))
            flat.append(year_values[day])
    da = xr.DataArray(
        np.array(flat, dtype="float32"),
        dims=("time",),
        coords={"time": np.array(times)},
        name=name,
        attrs={"units": "mm/day"},
    )
    return da


# Pre-baked two-year fixture used by several tests.
# Year 2000 daily pr (mm/day): 5 mm * 4 days, 0.5 mm * 3 days, 15 mm * 1, 25 mm * 1, 0 mm * 2.
# Year 2001 daily pr (mm/day): 0 mm * 5, 12 mm * 2, 21 mm * 2, 0.5 mm * 2.
_YEAR_2000 = [5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 15.0, 25.0, 0.0, 0.0]
_YEAR_2001 = [0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 12.0, 21.0, 21.0, 0.5, 0.5]


def _two_year_fixture() -> dict[str, xr.DataArray]:
    return {"pr": _daily_pr_dataarray([_YEAR_2000, _YEAR_2001])}


# ---------------------------------------------------------------------------
# Pure-math index tests with exact expected values
# ---------------------------------------------------------------------------


def test_compute_PRCPTOT_annual_wet_day_sum():
    """Wet-day threshold is >= 1.0 mm. 2000 wet-day sum = 5+5+5+5+15+25 = 60.
    2001 wet-day sum = 12+12+21+21 = 66."""
    out = compute_PRCPTOT(_two_year_fixture())
    assert out.name == INDEX_PRCPTOT
    assert out.attrs["units"] == "mm"
    assert out.attrs["wet_day_threshold_mm"] == WET_DAY_THRESHOLD_MM
    assert list(out.values) == [pytest.approx(60.0), pytest.approx(66.0)]


def test_compute_RX1day_annual_max_daily():
    """2000 max = 25.0; 2001 max = 21.0."""
    out = compute_RX1day(_two_year_fixture())
    assert out.name == INDEX_RX1DAY
    assert out.attrs["units"] == "mm"
    assert list(out.values) == [pytest.approx(25.0), pytest.approx(21.0)]


def test_compute_R95p_uses_period_wide_wet_day_p95_baseline():
    """Wet days across both years: [5, 5, 5, 5, 15, 25, 12, 12, 21, 21].
    p95 of those 10 values ~= 25.0 (quantile at 0.95 is between 21 and 25,
    closer to 25 with linear interpolation). The legacy formula keeps days
    where pr > p95, so the count depends on the exact p95.

    Using numpy's default linear interpolation:
      sorted = [5,5,5,5,12,12,15,21,21,25]
      p95 index = 0.95 * (10 - 1) = 8.55
      => 21 + 0.55 * (25 - 21) = 23.2
    Days with pr > 23.2: only 2000-01-09 (25.0). 2001 has nothing > 23.2.
    So 2000 sum = 25.0; 2001 sum = 0.0.
    """
    out = compute_R95p(_two_year_fixture())
    assert out.name == INDEX_R95P
    assert out.attrs["units"] == "mm"
    assert out.attrs["percentile"] == R95P_PERCENTILE
    assert out.attrs["baseline_period_policy"] == R95P_BASELINE_POLICY
    assert out.attrs["wet_day_threshold_mm"] == WET_DAY_THRESHOLD_MM
    assert list(out.values) == [pytest.approx(25.0), pytest.approx(0.0)]


def test_compute_R10mm_counts_days_per_year():
    """2000 days with pr >= 10: only 15.0 and 25.0 -> 2.
    2001 days with pr >= 10: 12, 12, 21, 21 -> 4."""
    out = compute_R10mm(_two_year_fixture())
    assert out.name == INDEX_R10MM
    assert out.attrs["units"] == "days"
    assert out.attrs["threshold_mm"] == R10_THRESHOLD_MM
    assert list(out.values) == [2, 4]


def test_compute_R20mm_counts_days_per_year():
    """2000 days with pr >= 20: 25.0 -> 1.
    2001 days with pr >= 20: 21, 21 -> 2."""
    out = compute_R20mm(_two_year_fixture())
    assert out.name == INDEX_R20MM
    assert out.attrs["units"] == "days"
    assert out.attrs["threshold_mm"] == R20_THRESHOLD_MM
    assert list(out.values) == [1, 2]


def test_compute_CDD_max_consecutive_dry_days_per_year():
    """Year 2000 daily pr: [5,5,5,5,0.5,0.5,0.5,15,25,0,0].
       Dry mask (< 1.0):  [F,F,F,F,T,T,T,F,F,T,T]. Max run = 3.
    Year 2001 daily pr: [0,0,0,0,0,12,12,21,21,0.5,0.5].
       Dry mask: [T,T,T,T,T,F,F,F,F,T,T]. Max run = 5."""
    out = compute_CDD(_two_year_fixture())
    assert out.name == INDEX_CDD
    assert out.attrs["units"] == "days"
    assert out.attrs["wet_day_threshold_mm"] == WET_DAY_THRESHOLD_MM
    assert list(int(v) for v in out.values) == [3, 5]


def test_compute_CWD_max_consecutive_wet_days_per_year():
    """Year 2000 wet mask: [T,T,T,T,F,F,F,T,T,F,F]. Max run = 4.
    Year 2001 wet mask:    [F,F,F,F,F,T,T,T,T,F,F]. Max run = 4."""
    out = compute_CWD(_two_year_fixture())
    assert out.name == INDEX_CWD
    assert out.attrs["units"] == "days"
    assert list(int(v) for v in out.values) == [4, 4]


def test_CDD_handles_all_dry_year():
    """A year that is entirely dry should report CDD == year-length."""
    fixture = {"pr": _daily_pr_dataarray([[0.0] * 7])}
    out = compute_CDD(fixture)
    assert int(out.values[0]) == 7


def test_CWD_handles_all_wet_year():
    """A year that is entirely wet should report CWD == year-length."""
    fixture = {"pr": _daily_pr_dataarray([[5.0] * 6])}
    out = compute_CWD(fixture)
    assert int(out.values[0]) == 6


# ---------------------------------------------------------------------------
# Dispatch + spec wiring
# ---------------------------------------------------------------------------


def test_compute_precipitation_index_dispatches_each_spec():
    fixture = _two_year_fixture()
    for spec in PRECIPITATION_INDEX_SPECS:
        out = compute_precipitation_index(spec.index_id, fixture)
        assert out.name == spec.index_id
        assert out.sizes["time"] == 2


def test_compute_precipitation_index_rejects_unknown():
    with pytest.raises(PrecipitationIndexError, match="unknown precipitation index_id"):
        compute_precipitation_index("ZZZ", _two_year_fixture())


def test_specs_for_family_temperature_unchanged():
    """Critical for M005 byte-identity: temperature spec list and order
    must match what M005 originally pinned."""
    assert [s.index_id for s in specs_for_family(INDEX_FAMILY_TEMPERATURE)] == [
        s.index_id for s in TEMPERATURE_INDEX_SPECS
    ]


def test_specs_for_family_precipitation_lists_seven_ids():
    assert [s.index_id for s in specs_for_family(INDEX_FAMILY_PRECIPITATION)] == [
        "PRCPTOT", "RX1day", "R95p", "CDD", "CWD", "R10mm", "R20mm",
    ]


def test_specs_for_family_all_concatenates_temperature_then_precipitation():
    all_ids = [s.index_id for s in specs_for_family(INDEX_FAMILY_ALL)]
    assert all_ids[: len(TEMPERATURE_INDEX_SPECS)] == [s.index_id for s in TEMPERATURE_INDEX_SPECS]
    assert all_ids[len(TEMPERATURE_INDEX_SPECS) :] == [s.index_id for s in PRECIPITATION_INDEX_SPECS]


def test_specs_for_family_rejects_unknown_family():
    with pytest.raises(IndexManifestError, match="unknown index family"):
        specs_for_family("snow")


def test_supported_index_families_constant():
    assert SUPPORTED_INDEX_FAMILIES == {
        INDEX_FAMILY_TEMPERATURE,
        INDEX_FAMILY_PRECIPITATION,
        INDEX_FAMILY_ALL,
    }


# ---------------------------------------------------------------------------
# Manifest integration: planning against the canonical M006 manifest
# ---------------------------------------------------------------------------


def test_plan_index_results_against_canonical_m006_precipitation_manifest(tmp_path: Path):
    preprocessing = json.loads(CANONICAL_M006_PREPROCESSING.read_text(encoding="utf-8"))
    results = plan_index_results(
        list(specs_for_family(INDEX_FAMILY_PRECIPITATION)),
        preprocessing=preprocessing,
        output_root=tmp_path,
    )
    by_id = {r.index_id: r for r in results}
    assert sorted(by_id) == sorted(
        ["PRCPTOT", "RX1day", "R95p", "CDD", "CWD", "R10mm", "R20mm"]
    )
    for spec_id, result in by_id.items():
        assert result.required_variables == ("pr",)
        assert result.reason is None, f"{spec_id} should plan cleanly: reason={result.reason!r}"
        assert result.output_path.endswith(f"derived/indices/{spec_id}.nc")


def test_plan_index_results_temperature_against_canonical_m004_unchanged(tmp_path: Path):
    """Temperature planning against the M004 manifest still produces seven
    planned indices, with two clean (TXx, SU) and five with the
    required-variable-not-in-plan reason. This pins the M005 behavior the
    canonical reference manifest carries."""
    preprocessing = json.loads(CANONICAL_M004_PREPROCESSING.read_text(encoding="utf-8"))
    results = plan_index_results(
        list(specs_for_family(INDEX_FAMILY_TEMPERATURE)),
        preprocessing=preprocessing,
        output_root=tmp_path,
    )
    clean = [r.index_id for r in results if r.reason is None]
    assert clean == ["TXx", "SU"]
    with_reason = [r.index_id for r in results if r.reason is not None]
    assert sorted(with_reason) == sorted(["Tmx", "Tmn", "TNn", "DTR", "TR"])


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _load_script_main():
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "04_compute_indices.py"
    spec = importlib.util.spec_from_file_location("compute_indices_script_pr", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_cli_default_family_is_temperature_against_m004(tmp_path: Path):
    main = _load_script_main()
    output = tmp_path / "index_manifest.json"
    rc = main(
        [
            "--preprocessing-manifest",
            str(CANONICAL_M004_PREPROCESSING),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "run_root"),
            "--mode",
            "dry-run",
        ]
    )
    assert rc == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    # Same content as the canonical M005 reference, by construction (no
    # output_root substring or hash differs except output_root).
    canonical = json.loads(CANONICAL_M005_INDEX.read_text(encoding="utf-8"))
    assert loaded["manifest_type"] == canonical["manifest_type"]
    assert loaded["index_count"] == canonical["index_count"] == 7
    assert [r["index_id"] for r in loaded["results"]] == [
        r["index_id"] for r in canonical["results"]
    ]
    assert "index_family" not in loaded, "schema must not gain new fields"


def test_cli_precipitation_family_against_m006(tmp_path: Path):
    main = _load_script_main()
    output = tmp_path / "im_pr.json"
    rc = main(
        [
            "--preprocessing-manifest",
            str(CANONICAL_M006_PREPROCESSING),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "run_root"),
            "--mode",
            "dry-run",
            "--index-family",
            "precipitation",
        ]
    )
    assert rc == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["manifest_type"] == MANIFEST_TYPE_INDEX
    assert loaded["mode"] == MODE_DRY_RUN
    assert loaded["requires_network"] is False
    assert loaded["index_count"] == 7
    assert loaded["planned_count"] == 7
    assert loaded["computed_count"] == 0
    assert loaded["region_id"] == "rbmn"
    result_ids = [r["index_id"] for r in loaded["results"]]
    assert result_ids == ["PRCPTOT", "RX1day", "R95p", "CDD", "CWD", "R10mm", "R20mm"]
    for r in loaded["results"]:
        assert r["required_variables"] == ["pr"]
        assert "reason" not in r  # all plan cleanly against M006
        assert r["output_path"].endswith(f"derived/indices/{r['index_id']}.nc")


def test_cli_rejects_unknown_index_family(tmp_path: Path):
    main = _load_script_main()
    # argparse choices enforcement: parse_args will exit with SystemExit.
    with pytest.raises(SystemExit):
        main(
            [
                "--preprocessing-manifest",
                str(CANONICAL_M004_PREPROCESSING),
                "--output",
                str(tmp_path / "out.json"),
                "--output-root",
                str(tmp_path),
                "--mode",
                "dry-run",
                "--index-family",
                "snow",
            ]
        )


# ---------------------------------------------------------------------------
# Execute mode against a synthetic NetCDF
# ---------------------------------------------------------------------------


def test_execute_index_results_writes_pr_index_netcdfs(tmp_path: Path):
    """Build a fake M006 preprocessing manifest pointing at one synthetic
    daily pr NetCDF, then run execute mode for the precipitation family.
    Verify every index NetCDF appears under ``derived/indices/``."""
    from lib.index_manifest import execute_index_results

    output_root = tmp_path / "run"
    daily_path = output_root / "intermediate" / "daily" / "pr" / "2000.nc"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    # 11-day stretch with 5 wet days >= 1.0 mm and one day >= 10 mm.
    da = _daily_pr_dataarray([_YEAR_2000])
    da.to_dataset(name="pr").to_netcdf(daily_path, engine="h5netcdf")
    manifest = {
        "manifest_type": MANIFEST_TYPE_PREPROCESSING,
        "region_id": "rbmn",
        "region_geometry_hash": "sha256:" + "0" * 64,
        "results": [
            {
                "request_id": "era5_hourly_pr__2000",
                "project_variable": "pr",
                "year": 2000,
                "source_path": "",
                "output_path": str(daily_path).replace("\\", "/"),
                "status": "preprocessed",
            }
        ],
    }
    results = execute_index_results(
        list(specs_for_family(INDEX_FAMILY_PRECIPITATION)),
        preprocessing=manifest,
        output_root=output_root,
    )
    statuses = [r.status for r in results]
    assert statuses == ["computed"] * len(PRECIPITATION_INDEX_SPECS)
    for spec in PRECIPITATION_INDEX_SPECS:
        target = output_root / "derived" / "indices" / f"{spec.index_id}.nc"
        assert target.exists(), f"missing {target}"
        with xr.open_dataset(target) as ds_out:
            ds_out.load()
        assert spec.index_id in ds_out.data_vars


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2D mask-preservation regression tests
# ---------------------------------------------------------------------------


def _two_cell_lat_lon_pr_fixture() -> dict[str, xr.DataArray]:
    """Build a (time, lat, lon) pr fixture with one all-NaN cell and one valid cell.

    Grid: 2 cells along ``lon`` (lat is degenerate, size 1). The
    western cell (lon=-105.6) is fully masked (NaN at every timestamp);
    the eastern cell (lon=-105.4) carries the two-year daily pr
    pattern from ``_YEAR_2000`` + ``_YEAR_2001``.
    """
    time = []
    valid_series = []
    for year_index, year_values in enumerate([_YEAR_2000, _YEAR_2001]):
        year = 2000 + year_index
        start = np.datetime64(f"{year}-01-01", "ns")
        for day in range(len(year_values)):
            time.append(start + np.timedelta64(day, "D"))
            valid_series.append(year_values[day])
    time_arr = np.array(time)
    valid = np.array(valid_series, dtype="float32")
    # Stack into shape (time, lat=1, lon=2): masked column first, valid column second.
    masked_column = np.full_like(valid, np.nan)
    values = np.stack([masked_column, valid], axis=-1)[:, None, :]
    da = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time_arr, "lat": [22.0], "lon": [-105.6, -105.4]},
        name="pr",
        attrs={"units": "mm/day"},
    )
    return {"pr": da}


_MASK_REGRESSION_EXPECTED = {
    INDEX_PRCPTOT: [60.0, 66.0],
    INDEX_RX1DAY: [25.0, 21.0],
    INDEX_R95P: [25.0, 0.0],
    INDEX_R10MM: [2, 4],
    INDEX_R20MM: [1, 2],
    INDEX_CDD: [3, 5],
    INDEX_CWD: [4, 4],
}


@pytest.mark.parametrize("index_id, expected_valid", list(_MASK_REGRESSION_EXPECTED.items()))
def test_index_preserves_masked_cell_nan(index_id, expected_valid):
    """Every precipitation index must leave fully-NaN cells as NaN in the output
    while keeping the valid cell's annual values intact.

    This regression would have failed on the unfixed M007 implementation:
    sum-style indices coerced NaN-only cells to 0 via ``pr.where(cond, 0.0)``,
    threshold-count indices got 0 from ``(NaN >= 10)``, and CDD/CWD scored
    a run length of 0 because the predicate was uniformly False.
    """
    out = compute_precipitation_index(index_id, _two_cell_lat_lon_pr_fixture())
    # Sanity: annual time dim has two years, lat is degenerate, lon has two cells.
    assert out.sizes["time"] == 2
    assert out.sizes["lat"] == 1
    assert out.sizes["lon"] == 2
    masked_cell = out.isel(lat=0, lon=0)
    valid_cell = out.isel(lat=0, lon=1)
    # Masked cell must be NaN in every year.
    assert bool(np.isnan(masked_cell.values).all()), (
        f"{index_id}: masked cell did not stay NaN; got {masked_cell.values!r}"
    )
    # Valid cell must match the 1D-fixture expected values.
    valid_values = [float(v) for v in valid_cell.values]
    for got, want in zip(valid_values, expected_valid):
        assert got == pytest.approx(float(want)), (
            f"{index_id}: valid cell year value {got!r} != expected {want!r}"
        )


def test_R95p_masked_cell_stays_nan_even_when_other_cells_have_no_extreme_days():
    """R95p quantile is computed per cell. A fully-masked cell has NaN
    quantile, but the cell must still stay NaN -- not get coerced to 0 by
    the ``pr > NaN`` branch that the unfixed code took."""
    out = compute_R95p(_two_cell_lat_lon_pr_fixture())
    masked_cell = out.isel(lat=0, lon=0)
    assert bool(np.isnan(masked_cell.values).all())


def test_threshold_count_masked_cell_does_not_become_zero():
    """R10mm/R20mm specifically must not silently report 0 days for a
    fully-masked cell (the symptom of the (NaN >= threshold) -> False bug)."""
    fixture = _two_cell_lat_lon_pr_fixture()
    for index_id in (INDEX_R10MM, INDEX_R20MM):
        out = compute_precipitation_index(index_id, fixture)
        masked = out.isel(lat=0, lon=0)
        valid = out.isel(lat=0, lon=1)
        assert bool(np.isnan(masked.values).all()), f"{index_id}: masked cell became 0"
        # Valid cell must still report the per-year counts.
        expected = _MASK_REGRESSION_EXPECTED[index_id]
        assert [int(v) for v in valid.values] == list(expected)


def test_run_length_masked_cell_does_not_become_zero():
    """CDD/CWD specifically must distinguish 'fully masked' from 'zero
    qualifying days'. The unfixed code returned 0 for both."""
    fixture = _two_cell_lat_lon_pr_fixture()
    for index_id in (INDEX_CDD, INDEX_CWD):
        out = compute_precipitation_index(index_id, fixture)
        masked = out.isel(lat=0, lon=0)
        assert bool(np.isnan(masked.values).all()), f"{index_id}: masked cell became 0"


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_indices_precipitation_does_not_eagerly_import_heavy_deps():
    import inspect

    import lib.indices_precipitation as p

    src = inspect.getsource(p)
    for line in src.splitlines():
        stripped = line.lstrip()
        if line == stripped:
            for forbidden in (
                "import numpy", "from numpy",
                "import xarray", "from xarray",
                "import dask", "from dask",
                "import icclim", "from icclim",
                "import xclim", "from xclim",
                "import geopandas", "from geopandas",
                "import rioxarray", "from rioxarray",
            ):
                assert not stripped.startswith(forbidden), f"eager import: {line!r}"
