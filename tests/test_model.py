"""Regression checks for exact-solution parameter validation."""

import unittest

import numpy as np

from model import v_exact, x_exact


class ZetaValidationTests(unittest.TestCase):
    def test_non_finite_zeta_is_rejected(self):
        for exact in (x_exact, v_exact):
            for zeta in (np.nan, np.inf, -np.inf):
                with self.subTest(exact=exact.__name__, zeta=zeta):
                    with self.assertRaisesRegex(ValueError, "zeta must be finite") as caught:
                        exact([0.0, 1.0], zeta)
                    self.assertIn(repr(zeta), str(caught.exception))

    def test_negative_zeta_stays_legal(self):
        t = np.linspace(0.0, 1.0, 11)
        x, v = x_exact(t, -0.1), v_exact(t, -0.1)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertEqual(x[0], 1.0)
        self.assertEqual(v[0], 0.0)
        self.assertGreater(x[-1] ** 2 + v[-1] ** 2, 1.0)


class AntiDampingExactSolutionTests(unittest.TestCase):
    def test_exact_solution_finite_and_satisfies_ode_for_negative_zeta(self):
        # zeta <= -1 used to fall into the underdamped branch and return NaN.
        t = np.linspace(0.0, 3.0, 3001)
        for zeta in (-0.5, -1.0, -1.5, -3.0):
            with self.subTest(zeta=zeta):
                x = x_exact(t, zeta)
                v = v_exact(t, zeta)
                self.assertTrue(np.all(np.isfinite(x)))
                self.assertAlmostEqual(float(x[0]), 1.0, places=12)
                self.assertAlmostEqual(float(v[0]), 0.0, places=12)
                # v is dx/dt and x'' + 2 zeta x' + x = 0 (central differences).
                dt = t[1] - t[0]
                dxdt = np.gradient(x, dt)
                np.testing.assert_allclose(dxdt[1:-1], v[1:-1], rtol=1e-4, atol=1e-4)
                a = np.gradient(v, dt)
                resid = a + 2 * zeta * v + x
                scale = np.max(np.abs(x))
                self.assertLess(np.max(np.abs(resid[1:-1])) / scale, 1e-3)

    def test_rk4_matches_exact_for_anti_damping(self):
        from solvers import solve_rk4
        for zeta in (-0.5, -1.0, -2.0):
            with self.subTest(zeta=zeta):
                t, x, _ = solve_rk4(zeta, 0.001, 2.0)
                ref = x_exact(t, zeta)
                self.assertLess(np.max(np.abs(x - ref)) / np.max(np.abs(ref)), 1e-9)

    def test_near_critical_continuity(self):
        # The three closed forms must join smoothly at |zeta| = 1.
        t = np.linspace(0.0, 10.0, 101)
        for z in (1.0, -1.0):
            with self.subTest(zeta=z):
                for eps in (1e-6, -1e-6):
                    np.testing.assert_allclose(
                        x_exact(t, z + eps), x_exact(t, z), rtol=1e-4, atol=1e-6)


class LargeDampingAccuracyTests(unittest.TestCase):
    def test_overdamped_closed_form_is_accurate_for_huge_zeta(self):
        # References from the closed form in 50-digit arithmetic (mpmath).
        # The naive r1 = -zeta + sqrt(zeta^2 - 1) lost ~6 digits at 1e7.
        cases = [
            (1e5, 1e4, 0.9512294245233057),
            (1e7, 1e4, 0.9995001249791717),
            (1e7, 1.0, 0.9999999500000037),
        ]
        for zeta, t, expected in cases:
            with self.subTest(zeta=zeta, t=t):
                self.assertAlmostEqual(float(x_exact(t, zeta)) / expected, 1.0, places=13)

    def test_overdamped_velocity_matches_slow_root_tail(self):
        zeta = 1e6
        r1 = -1.0 / (zeta + np.sqrt(zeta**2 - 1))
        t = 1e5
        self.assertAlmostEqual(float(v_exact(t, zeta) / x_exact(t, zeta)) / r1, 1.0, places=10)
