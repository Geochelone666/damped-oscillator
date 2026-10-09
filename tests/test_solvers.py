"""Regression checks for the documented damped Verlet-type scheme."""

import unittest

import numpy as np

from solvers import SOLVERS, solve_verlet


class VerletTests(unittest.TestCase):
    def test_next_step_uses_corrected_velocity_for_acceleration(self):
        # Hand evaluation of the documented recurrence for zeta=1, h=0.1:
        # x1=0.995, v1=-0.08975, so a1=-0.8155 (not -0.795,
        # the acceleration evaluated using the first predictor v_pred=-0.1).
        t, x, v = solve_verlet(zeta=1.0, h=0.1, t_end=0.2)
        np.testing.assert_allclose(t, [0.0, 0.1, 0.2], rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            x, [1.0, 0.995, 0.9819475], rtol=0, atol=1e-15,
        )
        np.testing.assert_allclose(
            v, [0.0, -0.08975, -0.162492375], rtol=0, atol=1e-15,
        )


class TimeGridTests(unittest.TestCase):
    def test_invalid_steps_are_rejected_even_for_zero_duration(self):
        for method, solver in SOLVERS.items():
            for h in (0.0, -0.1, np.nan, np.inf, -np.inf):
                for t_end in (0.0, 1.0):
                    with self.subTest(method=method, h=h, t_end=t_end):
                        with self.assertRaisesRegex(ValueError, "h must be finite and positive"):
                            solver(0.1, h, t_end)

    def test_invalid_end_times_are_rejected(self):
        for method, solver in SOLVERS.items():
            for t_end in (-0.01, np.nan, np.inf, -np.inf):
                with self.subTest(method=method, t_end=t_end):
                    with self.assertRaisesRegex(ValueError, "t_end must be finite and nonnegative"):
                        solver(0.1, 0.1, t_end)

    def test_zero_duration_preserves_initial_conditions(self):
        for method, solver in SOLVERS.items():
            with self.subTest(method=method):
                t, x, v = solver(0.1, 0.1, 0.0)
                np.testing.assert_array_equal(t, [0.0])
                np.testing.assert_array_equal(x, [1.0])
                np.testing.assert_array_equal(v, [0.0])


class ZetaValidationTests(unittest.TestCase):
    def test_non_finite_zeta_is_rejected(self):
        # NaN/inf zeta silently produced partially-NaN (x, v) arrays
        # (e.g. euler gave x=[1, 1, nan, ...]); fail fast instead.
        for method, solver in SOLVERS.items():
            for zeta in (np.nan, np.inf, -np.inf):
                with self.subTest(method=method, zeta=zeta):
                    with self.assertRaisesRegex(ValueError, "zeta must be finite"):
                        solver(zeta, 0.1, 1.0)

    def test_negative_zeta_stays_legal(self):
        # Negative zeta is physically meaningful (anti-damping), so only
        # non-finite values are rejected.
        for method, solver in SOLVERS.items():
            with self.subTest(method=method):
                t, x, v = solver(-0.1, 0.1, 1.0)
                self.assertTrue(np.all(np.isfinite(x)))
                self.assertTrue(np.all(np.isfinite(v)))
