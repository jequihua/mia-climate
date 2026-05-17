"""Tests for ``lib.indices_temperature``.

Synthetic two-year fixtures keep the suite offline. Each test pins the
exact expected value from the index formula rather than approximating.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from lib.indices_temperature import (
    INDEX_DTR,
    INDEX_SU,
    INDEX_TMN,
    INDEX_TMX,
    INDEX_TNN,
    INDEX_TR,
    INDEX_TXX,
    SUMMER_DAY_THRESHOLD_DEGC,
    TEMPERATURE_INDEX_SPECS,
    TROPICAL_NIGHT_THRESHOLD_DEGC,
    TemperatureIndexError,
    compute_DTR,
    compute_SU,
    compute_TNn,
    compute_TR,
    compute_TXx,
    compute_Tmn,
    compute_Tmx,
    compute_index,
)


def _two_year_time() -> np.ndarray:
    return np.concatenate(
        [
            np.array(["2000-01-01", "2000-07-01", "2000-12-31"], dtype="datetime64[ns]"),
            np.array(["2001-01-01", "2001-07-01", "2001-12-31"], dtype="datetime64[ns]"),
        ]
    )


def _scalar_dataarray(values: list[float], *, name: str, units: str = "degC") -> xr.DataArray:
    """1D time-only DataArray (no lat/lon) so per-year reductions are exact scalars."""
    return xr.DataArray(
        np.array(values, dtype="float32"),
        dims=("time",),
        coords={"time": _two_year_time()},
        name=name,
        attrs={"units": units},
    )


# --- intensity indices ----------------------------------------------------


def test_compute_Tmx_returns_annual_max_of_tmean():
    daily = {"tmean": _scalar_dataarray([20.0, 32.0, 18.0, 22.0, 28.0, 19.0], name="tmean")}
    out = compute_Tmx(daily)
    assert out.name == INDEX_TMX
    assert out.attrs["units"] == "degC"
    assert list(out.values) == [pytest.approx(32.0), pytest.approx(28.0)]


def test_compute_Tmn_returns_annual_min_of_tmean():
    daily = {"tmean": _scalar_dataarray([20.0, 32.0, 18.0, 22.0, 28.0, 19.0], name="tmean")}
    out = compute_Tmn(daily)
    assert out.name == INDEX_TMN
    assert list(out.values) == [pytest.approx(18.0), pytest.approx(19.0)]


def test_compute_TXx_returns_annual_max_of_tmax():
    daily = {"tmax": _scalar_dataarray([25.0, 38.0, 22.0, 27.0, 35.0, 24.0], name="tmax")}
    out = compute_TXx(daily)
    assert out.name == INDEX_TXX
    assert list(out.values) == [pytest.approx(38.0), pytest.approx(35.0)]


def test_compute_TNn_returns_annual_min_of_tmin():
    daily = {"tmin": _scalar_dataarray([15.0, 25.0, 12.0, 17.0, 23.0, 14.0], name="tmin")}
    out = compute_TNn(daily)
    assert out.name == INDEX_TNN
    assert list(out.values) == [pytest.approx(12.0), pytest.approx(14.0)]


def test_compute_DTR_uses_tmax_minus_tmin_and_records_degC():
    daily = {
        "tmax": _scalar_dataarray([30.0, 35.0, 25.0, 32.0, 33.0, 22.0], name="tmax"),
        "tmin": _scalar_dataarray([20.0, 22.0, 17.0, 19.0, 21.0, 15.0], name="tmin"),
    }
    out = compute_DTR(daily)
    assert out.name == INDEX_DTR
    assert out.attrs["units"] == "degC"
    # 2000: mean(10, 13, 8) = 10.333...; 2001: mean(13, 12, 7) = 10.666...
    assert float(out.values[0]) == pytest.approx((10.0 + 13.0 + 8.0) / 3.0)
    assert float(out.values[1]) == pytest.approx((13.0 + 12.0 + 7.0) / 3.0)


def test_compute_DTR_requires_both_tmax_and_tmin():
    with pytest.raises(TemperatureIndexError, match="requires variable 'tmin'"):
        compute_DTR({"tmax": _scalar_dataarray([1.0] * 6, name="tmax")})


# --- frequency indices ----------------------------------------------------


def test_compute_SU_counts_days_above_30_per_year():
    # 2000: three values > 30 (31, 35, 40); 2001: two values > 30 (33, 31).
    daily = {"tmax": _scalar_dataarray([31.0, 35.0, 40.0, 33.0, 31.0, 29.5], name="tmax")}
    out = compute_SU(daily)
    assert out.name == INDEX_SU
    assert out.attrs["units"] == "days"
    assert out.attrs["threshold_degc"] == SUMMER_DAY_THRESHOLD_DEGC
    assert list(out.values) == [3, 2]


def test_compute_SU_threshold_strictly_greater_than_30():
    # Exactly-30 sample should NOT count (legacy uses ``Tmax > 30``).
    daily = {"tmax": _scalar_dataarray([30.0, 30.0, 30.0, 30.0, 30.0, 30.0], name="tmax")}
    out = compute_SU(daily)
    assert list(out.values) == [0, 0]


def test_compute_TR_counts_nights_above_20_per_year():
    # 2000: two values > 20 (21, 22); 2001: three values > 20 (21, 23, 25).
    daily = {"tmin": _scalar_dataarray([21.0, 22.0, 19.5, 21.0, 23.0, 25.0], name="tmin")}
    out = compute_TR(daily)
    assert out.name == INDEX_TR
    assert out.attrs["units"] == "days"
    assert out.attrs["threshold_degc"] == TROPICAL_NIGHT_THRESHOLD_DEGC
    assert list(out.values) == [2, 3]


def test_compute_TR_threshold_strictly_greater_than_20():
    daily = {"tmin": _scalar_dataarray([20.0] * 6, name="tmin")}
    out = compute_TR(daily)
    assert list(out.values) == [0, 0]


# --- dispatch + spec wiring ----------------------------------------------


def test_compute_index_dispatches_each_id():
    base_tmax = _scalar_dataarray([25.0, 31.0, 18.0, 24.0, 32.0, 19.0], name="tmax")
    base_tmin = _scalar_dataarray([10.0, 22.0, 5.0, 11.0, 23.0, 6.0], name="tmin")
    base_tmean = _scalar_dataarray([17.0, 26.5, 11.5, 17.5, 27.5, 12.5], name="tmean")
    daily = {"tmax": base_tmax, "tmin": base_tmin, "tmean": base_tmean}
    for spec in TEMPERATURE_INDEX_SPECS:
        out = compute_index(spec.index_id, daily)
        assert out.name == spec.index_id
        # Two-year synthetic data should always yield two annual values.
        assert out.sizes["time"] == 2


def test_compute_index_rejects_unknown_index_id():
    with pytest.raises(TemperatureIndexError, match="unknown temperature index_id"):
        compute_index("XX99", {"tmean": _scalar_dataarray([1.0] * 6, name="tmean")})
