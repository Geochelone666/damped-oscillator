"""Exact linear stability theory vs. the implemented solvers.

stability.py derives each method's 2x2 amplification matrix from the
documented one-step recurrences. The strongest check here is equivalence
with the solver loops: M**k @ (1, 0) must reproduce (x_k, v_k) from
solvers.solve() exactly, so the theory cannot drift from the code.
"""

import csv
from pathlib import Path
import unittest

import numpy as np

import stability
from solvers import SOLVERS, solve

DATA = Path(__file__).resolve().parents[1] / "data"


class AmplificationMatrixTests(unittest.TestCase):
    def test_matrix_powers_match_solver_trajectories(self):
        # The derivation is only meaningful if M**k @ y0 equals the loop.
        y0 = np.array([1.0, 0.0])  # the solvers' hard-coded initial state
        for method in SOLVERS:
            for zeta, h in ((0.0, 0.05), (0.1, 0.1), (1.0, 0.05), (2.0, 0.02), (-0.3, 0.05)):
                with self.subTest(method=method, zeta=zeta, h=h):
                    t, x, v = solve(method, zeta, h, 100 * h)
                    M = stability.amplification_matrix(method, zeta, h)
                    y = y0
                    for k in range(101):
                        np.testing.assert_allclose(
                            [x[k], v[k]], y, rtol=1e-10, atol=1e-12,
                            err_msg=f"step {k}",
                        )
                        y = M @ y

    def test_euler_matrix_in_closed_form(self):
        # M = [[1, h], [-h, 1 - 2*zeta*h]] from one explicit-Euler step.
        for zeta, h in ((0.1, 0.1), (2.0, 0.03)):
            with self.subTest(zeta=zeta, h=h):
                M = stability.amplification_matrix("euler", zeta, h)
                np.testing.assert_allclose(
                    M, [[1.0, h], [-h, 1.0 - 2.0 * zeta * h]],
                    rtol=0, atol=1e-15,
                )

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            stability.amplification_matrix("rk5", 0.1, 0.1)

    def test_invalid_steps_rejected(self):
        for h in (0.0, -0.1, np.nan, np.inf):
            with self.subTest(h=h):
                with self.assertRaises(ValueError):
                    stability.amplification_matrix("euler", 0.1, h)


class EnergyGrowthTests(unittest.TestCase):
    def test_euler_undamped_energy_multiplier_is_one_plus_h_squared(self):
        # experiments/energy_drift.py quotes E_n/E_0 = (1+h^2)^n; the
        # per-step multiplier is det(M) = 1 + h^2 (complex eigenpair).
        for h in (0.05, 0.1, 0.2, 0.5):
            with self.subTest(h=h):
                self.assertAlmostEqual(
                    stability.energy_growth_factor("euler", 0.0, h), 1.0 + h * h,
                )

    def test_euler_energy_law_matches_long_solver_run(self):
        # Direct end-to-end check of the quoted law against solve_euler.
        h = 0.1
        t, x, v = solve("euler", 0.0, h, 100.0)
        energy_ratio = (x[-1] ** 2 + v[-1] ** 2) / (x[0] ** 2 + v[0] ** 2)
        n = len(t) - 1
        np.testing.assert_allclose(energy_ratio, (1.0 + h * h) ** n, rtol=1e-9)

    def test_rk4_undamped_energy_multiplier_matches_documented_series(self):
        # data/README.md quotes RK4's per-step energy multiplier as
        # 1 - h^6/72 + h^8/576. det(M) = |R(i*h)|^2 for the RK4 stability
        # polynomial R, which expands to exactly that series.
        for h in (0.1, 0.5, 1.0, 2.0):
            with self.subTest(h=h):
                self.assertAlmostEqual(
                    stability.energy_growth_factor("rk4", 0.0, h),
                    1.0 - h ** 6 / 72.0 + h ** 8 / 576.0,
                )

    def test_verlet_undamped_preserves_quadratic_form(self):
        # det(M) = 1 for the symplectic zeta=0 limit: bounded energy error.
        for h in (0.1, 0.5, 1.0, 1.9):
            with self.subTest(h=h):
                self.assertAlmostEqual(
                    stability.energy_growth_factor("verlet", 0.0, h), 1.0,
                )


class BoundaryTests(unittest.TestCase):
    def test_euler_boundary_is_two_zeta_underdamped(self):
        # det(M) = 1 - 2*zeta*h + h^2 < 1 iff h < 2*zeta (complex pair).
        for zeta in (0.05, 0.1, 0.3, 0.5, 0.9):
            with self.subTest(zeta=zeta):
                h_star = stability.stability_boundary("euler", zeta)
                self.assertAlmostEqual(h_star, 2.0 * zeta, places=8)

    def test_euler_neutral_exactly_at_saved_scan_point(self):
        # data/stability_scan.csv: zeta=0.1, h=0.2 gives 100% envelope error
        # (neutral); h=0.19 decays, h=0.21 grows. The radii must agree.
        self.assertAlmostEqual(stability.spectral_radius("euler", 0.1, 0.2), 1.0, places=12)
        self.assertLess(stability.spectral_radius("euler", 0.1, 0.19), 1.0)
        self.assertGreater(stability.spectral_radius("euler", 0.1, 0.21), 1.0)

    def test_euler_undamped_unstable_for_every_h(self):
        self.assertTrue(np.isnan(stability.stability_boundary("euler", 0.0)))

    def test_verlet_undamped_boundary_is_two(self):
        # Standard velocity-Verlet limit for the harmonic oscillator.
        self.assertAlmostEqual(stability.stability_boundary("verlet", 0.0), 2.0, places=8)

    def test_rk4_undamped_boundary_is_two_root_two(self):
        # |R(i*h)| <= 1 for the RK4 stability polynomial gives h < 2*sqrt(2).
        self.assertAlmostEqual(
            stability.stability_boundary("rk4", 0.0), 2.0 * np.sqrt(2.0), places=8,
        )

    def test_boundary_bracket_failures_return_nan(self):
        # Verlet stays stable up to h ~ 2 for zeta=0.1, so h_hi=1.5 leaves
        # no bracket; Euler at zeta=0 has no stable h at all.
        self.assertTrue(np.isnan(stability.stability_boundary("verlet", 0.1, h_hi=1.5)))
        self.assertTrue(np.isnan(stability.stability_boundary("euler", 0.0)))

    def test_boundaries_are_not_monotone_in_zeta(self):
        # A naive guess says damping only widens the stable range. The
        # amplification matrices say otherwise for Verlet-type and RK4:
        # moderate damping shrinks their boundaries before they recover.
        # These pinned values (2026-10 bisection, tol 1e-10) guard that
        # documented finding.
        pinned = {
            "euler": {0.1: 0.2, 0.3: 0.6, 0.5: 1.0, 1.0: 2.0},
            "verlet": {0.0: 2.0, 0.1: 1.8435120420560085, 0.3: 1.732501170391315,
                       0.5: 2.0, 1.0: 1.0},
            "rk4": {0.0: 2.0 * np.sqrt(2.0), 0.1: 2.9508529575358367,
                    0.3: 2.8353001025873352, 0.5: 2.6225424918043103,
                    1.0: 2.7852935632256877},
        }
        for method, table in pinned.items():
            for zeta, expected in table.items():
                with self.subTest(method=method, zeta=zeta):
                    self.assertAlmostEqual(
                        stability.stability_boundary(method, zeta), expected, places=8,
                    )

    def test_radius_classification_matches_solver_boundedness(self):
        # The theory is only useful if rho <= 1 predicts bounded runs.
        # Damped case: an unstable run must end far above the decaying exact
        # envelope; a stable run stays near it.
        from model import x_exact

        cases = [
            ("euler", 0.1, 0.19, True),
            ("euler", 0.1, 0.21, False),
            ("verlet", 0.0, 1.9, True),
            ("verlet", 0.0, 2.1, False),
            ("rk4", 0.0, 2.8, True),
            ("rk4", 0.0, 2.9, False),
        ]
        # t_end = 600 lets even the slowest unstable case here (Euler at
        # h = 0.21, rho - 1 ~ 1e-3 per step) grow to ~20x its physical
        # envelope, well past the 10x ceiling.
        for method, zeta, h, expect_bounded in cases:
            with self.subTest(method=method, zeta=zeta, h=h):
                t, x, _ = solve(method, zeta, h, 600.0)
                ceiling = 10.0 * max(1.0, float(np.max(np.abs(x_exact(t, zeta)))))
                self.assertEqual(bool(np.all(np.abs(x) <= ceiling)), expect_bounded)


class DispersionTests(unittest.TestCase):
    """Predicted phase drift vs. the peak-fit metric from analysis.py."""

    def test_predicted_drift_matches_measured_for_euler_and_verlet(self):
        # analysis.phase_error fits peak-time drift from only ~3 positive
        # peaks on [0, 20], so the comparison tolerance must cover peak-fit
        # noise; agreement is nonetheless tight for the low-order methods.
        from analysis import phase_error

        tolerances = {"euler": 1e-3, "verlet": 4e-4}
        for method, tol in tolerances.items():
            for h in (0.1, 0.05, 0.025):
                with self.subTest(method=method, h=h):
                    t, x, _ = solve(method, 0.1, h, 20.0)
                    measured = phase_error(t, x, 0.1)
                    predicted = stability.period_drift(method, 0.1, h)
                    self.assertAlmostEqual(measured, predicted, delta=tol)

    def test_rk4_predicted_drift_below_peak_fit_noise(self):
        # Measured RK4 drift on [0, 20] is O(1e-4) and dominated by peak
        # interpolation; the prediction must be smaller than that noise.
        for h in (0.2, 0.1, 0.05):
            with self.subTest(h=h):
                self.assertLess(abs(stability.period_drift("rk4", 0.1, h)), 1e-4)

    def test_euler_frequency_error_is_first_order_in_h(self):
        # The damping term couples into Euler's angle at O(h*zeta), making
        # its dispersion error O(h) despite the O(h^2) argument of 1+h*lambda.
        wd = np.sqrt(1.0 - 0.1 ** 2)
        errs = [abs(stability.numerical_frequency("euler", 0.1, h) - wd) / wd
                for h in (0.01, 0.001)]
        ratio = errs[0] / errs[1]
        self.assertGreater(ratio, 5.0)
        self.assertLess(ratio, 20.0)

    def test_frequency_convergence_hierarchy(self):
        # At h=1e-3: Euler O(h) ~ 1e-4, Verlet-type O(h^2) ~ 4e-8, RK4 O(h^4).
        wd = np.sqrt(1.0 - 0.1 ** 2)
        bounds = {"euler": 2e-4, "verlet": 1e-7, "rk4": 1e-11}
        for method, bound in bounds.items():
            with self.subTest(method=method):
                err = abs(stability.numerical_frequency(method, 0.1, 1e-3) - wd) / wd
                self.assertLess(err, bound)

    def test_euler_frequency_identity_at_h_equals_zeta(self):
        # metrics.csv shows identical euler phase_slope at h=0.1 and h=0.2
        # for zeta=0.1. That is not stale data: arg(1+h*lambda)/h at h=zeta
        # equals the value at h=2*zeta exactly (tan double-angle identity).
        for zeta in (0.05, 0.1, 0.3):
            with self.subTest(zeta=zeta):
                self.assertAlmostEqual(
                    stability.numerical_frequency("euler", zeta, zeta),
                    stability.numerical_frequency("euler", zeta, 2.0 * zeta),
                    places=12,
                )

    def test_non_oscillatory_zeta_rejected(self):
        for zeta in (1.0, 2.0, -1.5):
            with self.subTest(zeta=zeta):
                with self.assertRaises(ValueError):
                    stability.numerical_frequency("euler", zeta, 0.1)


class SavedBoundariesTests(unittest.TestCase):
    """data/stability_boundaries.csv must match the theory it records."""

    def test_csv_matches_recomputed_boundaries(self):
        with (DATA / "stability_boundaries.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3 * 5)  # methods x zetas in the experiment
        for row in rows:
            method, zeta = row["method"], float(row["zeta"])
            with self.subTest(method=method, zeta=zeta):
                h_star = stability.stability_boundary(method, zeta)
                if row["h_boundary"] == "":
                    self.assertTrue(np.isnan(h_star))
                    continue
                self.assertAlmostEqual(float(row["h_boundary"]), h_star, places=8)
                # Recorded empirical flags: bounded 5% below, unbounded 5% above.
                self.assertEqual(row["bounded_5pct_below"], "True")
                self.assertEqual(row["bounded_5pct_above"], "False")
                if method == "euler" and 0.0 < zeta <= 1.0:
                    self.assertAlmostEqual(float(row["euler_boundary_2zeta"]), 2.0 * zeta)


class SavedDispersionTests(unittest.TestCase):
    """data/dispersion_check.csv must match recomputed theory and metric."""

    def test_csv_matches_recomputed_drift(self):
        from analysis import phase_error

        with (DATA / "dispersion_check.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 15)  # 3 methods x 5 step sizes
        for row in rows:
            method, zeta, h = row["method"], float(row["zeta"]), float(row["h"])
            with self.subTest(method=method, h=h):
                predicted = stability.period_drift(method, zeta, h)
                self.assertAlmostEqual(
                    float(row["predicted_drift_s_per_period"]), predicted, places=8,
                )
                t, x, _ = solve(method, zeta, h, 20.0)
                measured = phase_error(t, x, zeta)
                self.assertAlmostEqual(
                    float(row["measured_drift_s_per_period"]), measured, places=8,
                )


if __name__ == "__main__":
    unittest.main()
