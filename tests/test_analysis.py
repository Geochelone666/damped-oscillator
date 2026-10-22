"""Regression checks for per-period phase drift."""

import math
import unittest
import warnings

import numpy as np

from analysis import _local_maxima, convergence_order, measure_cpu, phase_error, run_case
from model import x_exact
from solvers import SOLVERS


class MethodValidationTests(unittest.TestCase):
    def test_unknown_method_lists_valid_methods(self):
        for entry, kwargs in (
            (run_case, {"h": 0.1}),
            (measure_cpu, {"h": 0.1}),
            (convergence_order, {}),
        ):
            with self.subTest(entry=entry.__name__):
                with self.assertRaises(ValueError) as caught:
                    entry(method="euler2", zeta=0.1, **kwargs)
                message = str(caught.exception)
                self.assertIn("euler2", message)
                self.assertIn("valid methods", message)
                for method in SOLVERS:
                    self.assertIn(method, message)

    def test_negative_zeta_stays_legal(self):
        for method in SOLVERS:
            with self.subTest(method=method):
                metrics = run_case(method, -0.1, 0.1, t_end=1.0)
                self.assertTrue(np.isfinite(metrics["E_inf"]))
                self.assertTrue(np.isfinite(measure_cpu(method, -0.1, 0.1, t_end=1.0)))
                order = convergence_order(
                    -0.1, method, h_values=(0.1, 0.05, 0.025), t_end=1.0,
                )
                self.assertTrue(np.isfinite(order))


class PhaseErrorTests(unittest.TestCase):
    def test_exact_maxima_occur_at_integer_periods(self):
        zeta = 0.2
        period = 2.0 * np.pi / np.sqrt(1.0 - zeta**2)
        h = 0.01
        t = np.arange(0.0, 4.5 * period, h)
        t_peaks, _ = _local_maxima(t, x_exact(t, zeta))

        # The initial maximum is an endpoint; interior maxima recur every
        # full period, while the intervening half-period extrema are minima.
        self.assertEqual(len(t_peaks), 4)
        np.testing.assert_allclose(
            t_peaks, np.arange(1, 5) * period, atol=h, rtol=0.0,
        )

    def test_period_drift_is_independent_of_initial_peak_offset(self):
        # Symmetric sampled peaks have exact parabolic vertices. Their period
        # is 6 seconds, while their initial offset varies independently.
        t = np.arange(25.0)
        for first_peak in (5, 6, 7):
            with self.subTest(first_peak=first_peak):
                x = np.zeros_like(t)
                x[[first_peak, first_peak + 6, first_peak + 12]] = 1.0
                for zeta in (0.0, 0.1, 0.5):
                    with self.subTest(zeta=zeta):
                        exact_period = 2.0 * np.pi / np.sqrt(1.0 - zeta**2)
                        self.assertAlmostEqual(
                            phase_error(t, x, zeta), 6.0 - exact_period,
                            places=12,
                        )

    def test_insufficient_peaks_return_nan(self):
        t = np.arange(10.0)
        for peaks in ([], [5]):
            with self.subTest(peaks=peaks):
                x = np.zeros_like(t)
                x[peaks] = 1.0
                self.assertTrue(np.isnan(phase_error(t, x, 0.1)))


class EnvelopeRelErrTests(unittest.TestCase):
    def test_zero_damping_reports_nan_not_inf(self):
        # zeta = 0 is a legitimate input (the undamped oscillator), but a
        # relative error against a zero reference is undefined. run_case must
        # report NaN instead of inf, without emitting a RuntimeWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = run_case("rk4", 0.0, 0.1)
        self.assertTrue(math.isnan(m["envelope_rel_err"]))
        self.assertAlmostEqual(m["zeta_fit"], 0.0, places=3)

    def test_positive_damping_still_reports_finite_rel_err(self):
        m = run_case("rk4", 0.1, 0.05)
        self.assertTrue(np.isfinite(m["envelope_rel_err"]))
        self.assertLess(m["envelope_rel_err"], 0.05)


class AntiDampingMetricsTests(unittest.TestCase):
    def test_negative_damping_reports_finite_rel_err(self):
        # Negative zeta is legal anti-damping (solvers accept it); the fitted
        # decay rate is meaningful there, so the relative error must be a
        # finite number normalized by |zeta|, not NaN.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = run_case("rk4", -0.1, 0.05)
        self.assertTrue(np.isfinite(m["envelope_rel_err"]))
        self.assertLess(m["envelope_rel_err"], 1e-3)
        self.assertAlmostEqual(m["zeta_fit"], -0.1, places=3)

    def test_negative_damping_rel_err_is_nonnegative(self):
        # Guard against a sign-flipped denominator: a relative error must
        # never come out negative, whatever the damping sign.
        m = run_case("euler", -0.1, 0.05)
        self.assertTrue(np.isfinite(m["envelope_rel_err"]))
        self.assertGreaterEqual(m["envelope_rel_err"], 0.0)


if __name__ == "__main__":
    unittest.main()


class PhaseErrorDomainTests(unittest.TestCase):
    def test_phase_error_rejects_non_oscillatory_damping(self):
        t = np.linspace(0.0, 20.0, 201)
        for zeta in (1.0, 2.0, -1.0, -1.5, np.nan):
            with self.subTest(zeta=zeta):
                with self.assertRaisesRegex(ValueError, "underdamped"):
                    phase_error(t, np.cos(t), zeta)

    def test_strong_anti_damping_skips_oscillatory_metrics(self):
        metrics = run_case("rk4", -1.5, 0.1, t_end=2.0)
        self.assertNotIn("phase_err", metrics)
        self.assertNotIn("envelope_rel_err", metrics)
        self.assertTrue(np.isfinite(metrics["E_inf"]))
