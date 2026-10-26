"""Minimal checks for DOY parsing and manual cut behavior (synthetic weather)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.modvege import load_parameters, simulate  # noqa: E402


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.initial = load_parameters(ROOT)

    def weather(self, n=365):
        doy = np.arange(1, n + 1)
        seasonal = np.sin(2 * np.pi * (doy - 93) / 365)
        return pd.DataFrame({
            "DOY": doy,
            "Ta": 10 + 10 * seasonal,
            "PAR": 6 + 4 * np.maximum(0, seasonal),
            "Precip": np.full(n, 3.0),
            "ETc": np.full(n, 2.2),
        })

    def simulate(self, days, n=365, year=2024, temperature=0.0):
        return simulate(self.weather(n), year, days, self.model, self.initial,
                        temp_offset=temperature)

    def test_leapyear_input_365(self):
        out = self.simulate([150, 220])
        self.assertEqual(len(out), 365)
        self.assertEqual(int(out["DOY"].iloc[-1]), 365)
        self.assertEqual(int(out["CUT"].sum()), 2)
        self.assertGreater(out["hvBM"].iloc[-1], 0)

    def test_366_and_normal_year(self):
        self.assertEqual(len(self.simulate([], n=366)), 366)
        self.assertEqual(len(self.simulate([], year=2023)), 365)

    def test_no_cuts_and_finite_values(self):
        out = self.simulate([])
        self.assertEqual(out["CUT"].sum(), 0)
        self.assertEqual(float(out["hvBM"].iloc[-1]), 0.0)
        for col in ("LAI", "BM", "GRO", "ST", "f_abs", "SEA"):
            self.assertTrue(np.isfinite(out[col]).all(), col)

    def test_biomass_drops_after_cut(self):
        out = self.simulate([170])
        self.assertEqual(int(out.loc[169, "CUT"]), 1)
        self.assertLess(out.loc[169, "BM"], out.loc[168, "BM"])
        self.assertGreater(float(out.loc[169, "cut_yield"]), 0.0)

    def test_fixed_intensive_nitrogen(self):
        out = simulate(self.weather(), 2024, [145, 200, 253],
                       self.model, self.initial, ni=1.2, whc=80.0)
        self.assertEqual(int(out["CUT"].sum()), 3)
        self.assertTrue(np.isfinite(out[["GRO", "LAI", "BM", "hvBM"]]).all().all())
        self.assertGreaterEqual(out["LAI"].min(), 0)
        self.assertGreater(out["hvBM"].iloc[-1], 0)

    def test_gts_growth_start_matches_original_threshold(self):
        out = self.simulate([])
        self.assertTrue((out["GTS"].diff().dropna() >= 0).all())
        start = int(out["SGS"].iloc[0])
        self.assertGreater(out.iloc[start - 1]["GTS"], 200)
        self.assertLessEqual(out.iloc[start - 2]["GTS"], 200)
        self.assertTrue((out.iloc[:start - 1]["GRO"] == 0).all())

    def test_temperature_scenario_changes_gts_and_start(self):
        reference = self.simulate([], temperature=0)
        warmer = self.simulate([], temperature=2)
        self.assertLessEqual(int(warmer["SGS"].iloc[0]), int(reference["SGS"].iloc[0]))
        self.assertGreater(warmer.iloc[100]["GTS"], reference.iloc[100]["GTS"])

    def test_gts_unaffected_by_cut_regime(self):
        no_cut = self.simulate([])
        with_cut = self.simulate([145, 200, 253])
        np.testing.assert_allclose(no_cut["GTS"], with_cut["GTS"])
        self.assertEqual(int(no_cut["SGS"].iloc[0]), int(with_cut["SGS"].iloc[0]))

    def test_bad_doy_is_rejected(self):
        bad = self.weather().drop(index=14)
        with self.assertRaisesRegex(ValueError, "Incomplete DOY series"):
            simulate(bad, 2024, [], self.model, self.initial)


if __name__ == "__main__":
    unittest.main()
