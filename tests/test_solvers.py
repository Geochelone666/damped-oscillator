"""Regression checks for the documented damped Verlet-type scheme."""

import unittest

import numpy as np

from solvers import SOLVERS, solve, solve_verlet


class MethodValidationTests(unittest.TestCase):
    def test_unknown_method_lists_valid_methods(self):
        with self.assertRaises(ValueError) as caught:
            solve("euler2", 0.1, 0.01, 5)
        message = str(caught.exception)
        self.assertIn("euler2", message)
        self.assertIn("valid methods", message)
        for method in SOLVERS:
            with self.subTest(method=method):
                self.assertIn(method, message)

    def test_negative_zeta_stays_legal_through_dispatch(self):
        for method in SOLVERS:
            with self.subTest(method=method):
                _, x, v = solve(method, -0.1, 0.1, 1.0)
                self.assertTrue(np.all(np.isfinite(x)))
                self.assertTrue(np.all(np.isfinite(v)))


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


class GridEndpointTests(unittest.TestCase):
    def test_grid_never_overshoots_t_end(self):
        # round(t_end/h) used to integrate past t_end, e.g. h=0.19 on
        # [0, 40] ended at t=40.09. The grid must stop at the last
        # multiple of h that does not exceed t_end.
        for method, solver in SOLVERS.items():
            for h, t_end in ((0.19, 40.0), (0.22, 40.0), (0.3, 1.0), (0.7, 1.0)):
                with self.subTest(method=method, h=h, t_end=t_end):
                    t, x, v = solver(0.1, h, t_end)
                    self.assertLessEqual(t[-1], t_end + 1e-12)
                    self.assertGreater(t[-1] + h, t_end)
                    self.assertEqual(len(t), len(x))
                    self.assertEqual(len(t), len(v))

    def test_exact_multiples_reach_t_end(self):
        # Binary floating point makes e.g. 20/0.1 slightly below 200; the
        # endpoint must still be reached exactly.
        for h, t_end, n in ((0.1, 20.0, 200), (0.0125, 20.0, 1600),
                            (0.1, 0.3, 3), (0.05, 100.0, 2000)):
            with self.subTest(h=h, t_end=t_end):
                t, _, _ = solve("euler", 0.1, h, t_end)
                self.assertEqual(len(t), n + 1)
                self.assertAlmostEqual(t[-1], t_end, places=12)

    def test_step_larger_than_interval_returns_initial_state(self):
        t, x, v = solve("rk4", 0.1, 1.0, 0.4)
        np.testing.assert_array_equal(t, [0.0])
        np.testing.assert_array_equal(x, [1.0])
        np.testing.assert_array_equal(v, [0.0])


class ConvergenceOrderTests(unittest.TestCase):
    def test_global_order_from_step_halving(self):
        # Halving h must cut the endpoint error by ~2**p for a p-th order
        # method; checks the claimed orders 1/2/4 in every damping regime.
        from model import v_exact, x_exact

        expected = {"euler": 1.0, "verlet": 2.0, "rk4": 4.0}
        for method, p in expected.items():
            for zeta in (0.1, 1.0, 2.0):
                with self.subTest(method=method, zeta=zeta):
                    errs = []
                    for h in (0.02, 0.01):
                        t, x, v = solve(method, zeta, h, 2.0)
                        errs.append(np.hypot(x[-1] - x_exact(t[-1], zeta),
                                             v[-1] - v_exact(t[-1], zeta)))
                    observed = np.log2(errs[0] / errs[1])
                    self.assertAlmostEqual(observed, p, delta=0.15)


class StepCountOverflowTests(unittest.TestCase):
    def test_overflowing_step_count_raises_value_error(self):
        for method in SOLVERS:
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "overflows"):
                    solve(method, 0.1, 1e-300, 1e300)
