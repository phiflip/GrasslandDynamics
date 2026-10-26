"""Run with: python -m pytest tests -q"""
import math

import numpy as np
import pandas as pd

from scripts.modvege import load_parameters, simulate
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def synthetic_weather(year):
    """Synthetic forcing for software tests only, NOT data shown in the app."""
    n = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    doy = np.arange(1, n+1)
    seasonal = np.sin(2 * np.pi * (doy - 90) / n)
    return pd.DataFrame({
        "DOY": doy,
        "Ta": 8 + 12 * seasonal,
        "PAR": np.clip(8 + 7 * seasonal, 1, None),
        "Precip": np.full(n, 1.6),
        "ETc": np.clip(2 + 2 * seasonal, 0.1, None),
    })


def test_manual_cuts_and_units():
    p, ic = load_parameters(ROOT)
    weather = synthetic_weather(2024)
    df = simulate(weather, 2024, [145, 200, 253], p, ic)
    assert len(df) == 366
    assert df["DOY"].tolist() == list(range(1, 367))
    assert np.isfinite(df["BM"]).all()
    assert np.isfinite(df["GRO"]).all()
    assert np.isfinite(df["LAI"]).all()
    assert (df["BM"] >= 0).all()
    assert (df["LAI"] >= 0).all()
    assert df["hvBM"].is_monotonic_increasing
    assert set(df.loc[df["CUT"] == 1, "DOY"]) == {145, 200, 253}
    assert df.loc[df["DOY"].isin([145, 200, 253]), "cut_yield"].gt(0).all()
    assert math.isclose(df.loc[df["DOY"] == 200, "f_abs"].iat[0],
                        1-math.exp(-.6*df.loc[df["DOY"] == 200, "LAI"].iat[0]))


def test_no_cuts_zero_harvest():
    p, ic = load_parameters(ROOT)
    df = simulate(synthetic_weather(2023), 2023, [], p, ic)
    assert df["CUT"].sum() == 0
    assert df["hvBM"].iat[-1] == 0
    assert len(df) == 365


def test_reject_missing_days():
    p, ic = load_parameters(ROOT)
    weather = synthetic_weather(2024).iloc[:-1]
    try:
        simulate(weather, 2024, [], p, ic)
    except ValueError as err:
        assert "Incomplete" in str(err)
    else:
        raise AssertionError("Missing weather day not detected")
