"""Saved CSVs, README claims and self-checks stay consistent with the code."""
import contextlib
import csv
import io
from pathlib import Path
import unittest

import numpy as np

import analysis
import model
import solvers
from model import ZETAS, x_exact
from solvers import solve

DATA = Path(__file__).resolve().parents[1] / "data"


def _rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class SavedMetricsTests(unittest.TestCase):
    def test_metrics_csv_matches_recomputed_errors(self):
        rows = _rows("metrics.csv")
        self.assertEqual(len(rows), 45)
        for row in rows:
            method, zeta, h = row["method"], float(row["zeta"]), float(row["h"])
            with self.subTest(method=method, zeta=zeta, h=h):
                t, x, _ = solve(method, zeta, h, analysis.T_END)
                err = x - x_exact(t, zeta)
                # RK4 errors near 1e-10 differ in the 8th digit between
                # NumPy builds (summation order); stale data differs by >20%.
                np.testing.assert_allclose(float(row["E_inf"]), np.max(np.abs(err)), rtol=1e-6, atol=1e-14)
                np.testing.assert_allclose(float(row["E_RMS"]), np.sqrt(np.mean(err ** 2)), rtol=1e-6, atol=1e-14)

    def test_readme_convergence_order_claims(self):
        # README section 6: Euler 1.0-1.3, Verlet-type 2.0-2.2, RK4 4.0-4.2.
        rows = _rows("metrics.csv")
        bounds = {"euler": (0.95, 1.3), "verlet": (1.95, 2.25), "rk4": (3.9, 4.25)}
        for zeta in ZETAS:
            for method, (lo, hi) in bounds.items():
                sel = [r for r in rows if r["method"] == method and float(r["zeta"]) == zeta]
                h = np.array([float(r["h"]) for r in sel])
                e = np.array([float(r["E_inf"]) for r in sel])
                slope = np.polyfit(np.log(h), np.log(e), 1)[0]
                with self.subTest(method=method, zeta=zeta):
                    self.assertGreaterEqual(slope, lo)
                    self.assertLessEqual(slope, hi)

    def test_stability_scan_euler_neutral_at_h_equals_2zeta(self):
        # README: Euler envelope is neutral at h = 2*zeta = 0.2 (100% decay-rate error).
        rows = {(r["method"], float(r["h"])): float(r["zeta_fit_err"])
                for r in _rows("stability_scan.csv")}
        self.assertAlmostEqual(rows[("euler", 0.2)], 1.0, places=4)
        self.assertLess(rows[("euler", 0.19)], 1.0)
        self.assertGreater(rows[("euler", 0.21)], 1.0)


class FullMatrixTests(unittest.TestCase):
    def test_full_matrix_structure_on_small_grid(self):
        results = analysis.full_matrix(t_end=2.0, h_values=(0.1, 0.05))
        self.assertEqual(set(results), set(ZETAS))
        for zeta, by_method in results.items():
            self.assertEqual(set(by_method), set(solvers.SOLVERS))
            for metrics_by_h in by_method.values():
                self.assertEqual(set(metrics_by_h), {0.1, 0.05})
                for metrics in metrics_by_h.values():
                    self.assertTrue(np.isfinite(metrics["E_inf"]))
                    self.assertEqual("phase_err" in metrics, zeta < 1)


class LocalMaximaTests(unittest.TestCase):
    def test_flat_top_is_skipped_not_divided_by_zero(self):
        # x[k-1] - 2x[k] + x[k+1] == 0 cannot occur at a strict maximum on
        # exact data, but a degenerate parabola must be skipped cleanly.
        t = np.arange(5.0)
        x = np.array([0.0, 1.0, 3.0, 1.0, 0.0])
        t_p, x_p = analysis._local_maxima(t, x)
        self.assertEqual(len(t_p), 1)
        self.assertAlmostEqual(t_p[0], 2.0)
        self.assertGreaterEqual(x_p[0], 3.0)


class SelfCheckTests(unittest.TestCase):
    def test_module_self_checks_pass(self):
        for module in (model, solvers):
            with self.subTest(module=module.__name__):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    module._self_check()
                self.assertIn("ALL PASSED", out.getvalue())
