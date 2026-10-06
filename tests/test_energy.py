"""Regression checks for undamped Euler energy growth."""

import unittest

import numpy as np

from solvers import solve_euler


class EnergyTests(unittest.TestCase):
    def test_euler_energy_growth_matches_closed_form(self):
        for h in (0.2, 0.1, 0.05):
            with self.subTest(h=h):
                t, x, v = solve_euler(zeta=0.0, h=h, t_end=100.0)
                energy = 0.5 * (x**2 + v**2)
                steps = np.arange(len(t))
                np.testing.assert_allclose(
                    energy / energy[0], (1.0 + h**2)**steps,
                    rtol=1e-9, atol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
