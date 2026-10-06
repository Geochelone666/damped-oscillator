"""Regression checks for the documented damped Verlet-type scheme."""

import unittest

import numpy as np

from solvers import solve_verlet


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
