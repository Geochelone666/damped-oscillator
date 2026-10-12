"""Regression checks for the zeta = 0 boundary contract."""

import unittest

import numpy as np

from analysis import phase_error, run_case
from model import v_exact, x_exact
from solvers import solve


class UndampedLimitTests(unittest.TestCase):
    def test_exact_solution_is_harmonic(self):
        t = np.linspace(0.0, 20.0, 2001)
        for exact, expected in ((x_exact, np.cos(t)), (v_exact, -np.sin(t))):
            with self.subTest(exact=exact.__name__):
                np.testing.assert_allclose(exact(t, 0.0), expected, rtol=0, atol=1e-14)

    def test_exact_solution_is_continuous_at_zero(self):
        t = np.linspace(0.0, 20.0, 2001)
        # Measured maximum deviation: 1.89e-11.
        self.assertLess(np.max(np.abs(x_exact(t, 1e-12) - np.cos(t))), 1e-9)

    def test_solver_convergence_orders(self):
        hs = (0.04, 0.02, 0.01, 0.005)
        # Measured slopes: 1.01746, 2.00005, 4.00274; all are asymptotic
        # enough for +/-0.1. A short interval limits Euler's energy growth.
        for method, expected in (("euler", 1.0), ("verlet", 2.0), ("rk4", 4.0)):
            with self.subTest(method=method):
                errors = []
                for h in hs:
                    t, x, _ = solve(method, 0.0, h, t_end=5.0)
                    errors.append(np.max(np.abs(x - np.cos(t))))
                slope, _ = np.polyfit(np.log(hs), np.log(errors), 1)
                self.assertAlmostEqual(slope, expected, delta=0.1)

    def test_rk4_metrics(self):
        metrics = run_case("rk4", 0.0, 0.01)
        # Observed |zeta_fit|=8.18e-12, |phase_err|=1.03e-8,
        # E_inf=1.52e-9; allow margin for interpolation and roundoff.
        self.assertLess(abs(metrics["zeta_fit"]), 1e-9)
        self.assertTrue(np.isnan(metrics["envelope_rel_err"]))
        self.assertTrue(np.isfinite(metrics["phase_err"]))
        self.assertLess(abs(metrics["phase_err"]), 1e-7)
        self.assertLess(metrics["E_inf"], 1e-8)

    def test_euler_reports_envelope_growth(self):
        metrics = run_case("euler", 0.0, 0.1)
        # Observed -0.04975, consistent with -log(1+h**2)/(2*h).
        self.assertTrue(np.isfinite(metrics["zeta_fit"]))
        self.assertLess(metrics["zeta_fit"], 0.0)
        self.assertTrue(np.isnan(metrics["envelope_rel_err"]))

    def test_verlet_relative_envelope_error_is_undefined(self):
        metrics = run_case("verlet", 0.0, 0.01)
        self.assertTrue(np.isnan(metrics["envelope_rel_err"]))

    def test_exact_cosine_has_negligible_phase_error(self):
        t = np.linspace(0.0, 20.0, 2001)
        # Observed 9.73e-9: parabolic peak interpolation leaves a residual
        # even for exact samples, so do not demand machine precision.
        self.assertLess(abs(phase_error(t, np.cos(t), 0.0)), 1e-7)
