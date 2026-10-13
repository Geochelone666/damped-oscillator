"""Lock S05 dispatch/validation and S12 undamped behavior without changing them."""

import unittest
import warnings

import numpy as np

from analysis import convergence_order, measure_cpu, run_case
from model import v_exact, x_exact
from solvers import SOLVERS, get_solver, solve


class KeyBehaviorRegressionTests(unittest.TestCase):
    def test_get_solver_returns_registered_methods(self):
        for method, solver in SOLVERS.items():
            with self.subTest(method=method):
                # Dispatch returns the registered callable without wrapping it.
                self.assertIs(get_solver(method), solver)

    def test_get_solver_rejects_unknown_and_uppercase_names(self):
        for method in ("euler2", "EULER", "VERLET", "RK4"):
            with self.subTest(method=method):
                # Existing dictionary lookup is case-sensitive; no normalization.
                with self.assertRaises(ValueError) as caught:
                    get_solver(method)
                message = str(caught.exception)
                # Diagnostics identify the bad input and every supported name.
                self.assertIn(method, message)
                for valid in ("euler", "verlet", "rk4"):
                    self.assertIn(valid, message)

    def test_undamped_rk4_metrics(self):
        metrics = run_case("rk4", 0.0, 0.01)
        # Measured -8.1783e-12; a flat physical envelope permits small fit noise.
        self.assertLess(abs(metrics["zeta_fit"]), 1e-9)
        # Measured 1.0245e-8 s/period relative to T=2*pi; allow peak-fit noise.
        self.assertLess(abs(metrics["phase_err"]), 1e-7)
        # Measured 1.5159e-9 displacement error; allow RK4/roundoff headroom.
        self.assertLess(metrics["E_inf"], 1e-8)
        # A maximum absolute error cannot be negative, even if the cap passes.
        self.assertGreaterEqual(metrics["E_inf"], 0.0)

    def test_undamped_euler_reports_growth(self):
        rate = run_case("euler", 0.0, 0.1)["zeta_fit"]
        # Measured -0.04975175: amplitude grows by sqrt(1+h**2) per step,
        # giving decay rate -log(1+h**2)/(2*h), strictly negative and finite.
        self.assertTrue(np.isfinite(rate))
        self.assertLess(rate, 0.0)

    def test_undamped_relative_envelope_error_is_nan(self):
        for method in SOLVERS:
            for t_end in (1.0, 20.0):
                with self.subTest(method=method, t_end=t_end):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        value = run_case(method, 0.0, 0.1, t_end)["envelope_rel_err"]
                    # Relative error against zero is undefined with or without
                    # enough peaks; NaN (which excludes inf) is the contract.
                    self.assertTrue(np.isnan(value))

    def test_exact_solution_is_continuous_at_zero(self):
        t = np.linspace(0.0, 20.0, 2001)
        error = np.max(np.abs(x_exact(t, 1e-12) - x_exact(t, 0.0)))
        # Measured 1.88495e-11, consistent with O(zeta*t) envelope change;
        # 1e-9 leaves roundoff margin while excluding a discontinuous branch.
        self.assertLess(error, 1e-9)

    def test_numerical_solution_is_continuous_at_zero(self):
        t, x, _ = solve("rk4", 1e-12, 0.001, t_end=20.0)
        error = np.max(np.abs(x - x_exact(t, 0.0)))
        # Measured 1.88535e-11; h=0.001 suppresses RK4 truncation below the
        # O(zeta*t) perturbation, with a generous combined error cap of 1e-9.
        self.assertLess(error, 1e-9)

    def test_short_interval_convergence_order(self):
        for method, expected in (("euler", 1.0), ("verlet", 2.0), ("rk4", 4.0)):
            with self.subTest(method=method):
                slope = convergence_order(
                    0.0, method, h_values=(0.04, 0.02, 0.01, 0.005), t_end=5.0,
                )
                # Measured 1.017458 / 2.000053 / 4.002745. +/-0.1 around
                # theoretical order allows pre-asymptotic/roundoff effects;
                # t_end=5 limits Euler growth. This also checks the public API.
                self.assertAlmostEqual(slope, expected, delta=0.1)

    def test_exact_undamped_displacement_and_velocity(self):
        t = np.linspace(0.0, 20.0, 2001)
        for exact, expected in ((x_exact, np.cos(t)), (v_exact, -np.sin(t))):
            with self.subTest(exact=exact.__name__):
                # Both measured errors are zero; 1e-14 permits float roundoff
                # in the harmonic identities without relative slack at zeros.
                np.testing.assert_allclose(exact(t, 0.0), expected, rtol=0, atol=1e-14)

    def test_x_exact_rejects_nonfinite_zeta_at_entry(self):
        for zeta in (np.nan, np.inf, -np.inf):
            with self.subTest(zeta=zeta):
                # Invalid t cannot be converted to float: the zeta-specific
                # error proves validation occurs before even processing t.
                with self.assertRaisesRegex(ValueError, "zeta must be finite"):
                    x_exact(object(), zeta)

    def test_measure_cpu_returns_finite_positive_seconds(self):
        for method in SOLVERS:
            with self.subTest(method=method):
                seconds = measure_cpu(method, 0.0, 0.01, t_end=5.0)
                # Measured ~0.0005--0.0042 s; only finite, positive elapsed
                # time is contractual, not machine-dependent speed bounds.
                self.assertTrue(np.isfinite(seconds))
                self.assertGreater(seconds, 0.0)
