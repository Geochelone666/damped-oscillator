"""Regression checks for per-period phase drift."""

import unittest

import numpy as np

from analysis import phase_error


class PhaseErrorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
