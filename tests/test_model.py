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
