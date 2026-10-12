"""
Analytical (exact) solutions of the dimensionless damped harmonic oscillator

    x'' + 2*zeta*x' + x = 0,   x(0) = 1, v(0) = 0

where zeta is the damping ratio and v = x'. Three regimes:

  * underdamped  (zeta < 1): oscillation with exponentially decaying envelope
  * critical     (zeta = 1): fastest non-oscillatory return to equilibrium
  * overdamped   (zeta > 1): slow non-oscillatory decay (two real time scales)

These exact solutions serve as the ground truth against which the numerical
integrators in solvers.py are benchmarked.
"""

import numpy as np

from solvers import _validate_zeta

# Damping ratios used in the experiment matrix (task specification).
ZETAS = (0.1, 1.0, 2.0)


def x_exact(t, zeta):
    """Exact displacement x(t) for the given damping ratio.

    Derived from the characteristic equation r^2 + 2*zeta*r + 1 = 0 with
    x(0) = 1 and v(0) = 0:

    - zeta < 1: x = exp(-zeta*t) * (cos(wd*t) + (zeta/wd)*sin(wd*t)),
      wd = sqrt(1 - zeta^2) is the damped angular frequency.
    - zeta = 1: x = (1 + t) * exp(-t).
    - zeta > 1: x = ((zeta+s)*exp(r1*t) - (zeta-s)*exp(r2*t)) / (2*s),
      s = sqrt(zeta^2 - 1), r1 = -zeta + s, r2 = -zeta - s.
    """
    _validate_zeta(zeta)
    t = np.asarray(t, dtype=float)
    if zeta < 1.0:
        wd = np.sqrt(1.0 - zeta ** 2)
        return np.exp(-zeta * t) * (np.cos(wd * t) + (zeta / wd) * np.sin(wd * t))
    elif zeta == 1.0:
        return (1.0 + t) * np.exp(-t)
    else:
        s = np.sqrt(zeta ** 2 - 1.0)
        r1, r2 = -zeta + s, -zeta - s
        return ((zeta + s) * np.exp(r1 * t) - (zeta - s) * np.exp(r2 * t)) / (2.0 * s)


def v_exact(t, zeta):
    """Exact velocity v(t) = dx/dt, obtained by differentiating x_exact."""
    _validate_zeta(zeta)
    t = np.asarray(t, dtype=float)
    if zeta < 1.0:
        # Closed form: differentiating x(t) and using zeta^2 + wd^2 = 1 gives
        #   v(t) = -(1/wd) * exp(-zeta*t) * sin(wd*t).
        wd = np.sqrt(1.0 - zeta ** 2)
        return -(1.0 / wd) * np.exp(-zeta * t) * np.sin(wd * t)
    elif zeta == 1.0:
        # d/dt [(1+t)e^{-t}] = e^{-t} - (1+t)e^{-t} = -t*e^{-t}
        return -t * np.exp(-t)
    else:
        s = np.sqrt(zeta ** 2 - 1.0)
        r1, r2 = -zeta + s, -zeta - s
        return ((zeta + s) * r1 * np.exp(r1 * t) - (zeta - s) * r2 * np.exp(r2 * t)) / (2.0 * s)


def _self_check():
    """Sanity checks of the analytical formulas."""
    # 1. zeta = 0 must reduce to the undamped oscillator x = cos(t).
    t = np.linspace(0, 20, 401)
    assert np.allclose(x_exact(t, 0.0), np.cos(t), atol=1e-12), "zeta=0 must give cos(t)"

    # 2. Initial conditions x(0) = 1, v(0) = 0 for every regime.
    for zeta in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0):
        assert abs(x_exact(0.0, zeta) - 1.0) < 1e-12, f"x(0) != 1 for zeta={zeta}"
        assert abs(v_exact(0.0, zeta)) < 1e-12, f"v(0) != 0 for zeta={zeta}"

    # 3. v_exact must be the numerical derivative of x_exact. np.gradient on a
    #    0.01 grid has ~1e-4 truncation error, so keep the tolerance loose here;
    #    the exactness of v_exact is already guaranteed by the closed forms.
    rng_t = np.linspace(0, 10, 1001)
    for zeta in ZETAS:
        dxdt = np.gradient(x_exact(rng_t, zeta), rng_t)
        # Skip the two endpoints: np.gradient is only first-order there.
        assert np.allclose(dxdt[1:-1], v_exact(rng_t, zeta)[1:-1], atol=5e-4), (
            f"v_exact inconsistent with d(x_exact)/dt for zeta={zeta}"
        )

    # 4. Spot values against hand-computed references.
    #    zeta = 1: x(1) = 2/e
    assert abs(x_exact(1.0, 1.0) - 2.0 / np.e) < 1e-12
    #    zeta = 0.1: x(0) envelope is 1; at t = pi/wd the first minimum sits
    #    near -exp(-zeta*pi/wd); just check sign and magnitude range.
    wd = np.sqrt(1 - 0.1 ** 2)
    xmin = x_exact(np.pi / wd, 0.1)
    assert -1.0 < xmin < -0.5, f"unexpected first minimum {xmin}"

    # 5. Overdamped long-time behaviour: dominated by the slow root r1 = -zeta + s.
    s = np.sqrt(2.0 ** 2 - 1.0)
    r1 = -2.0 + s
    t_long = np.array([15.0, 18.0])
    ratio = x_exact(t_long, 2.0)[1] / x_exact(t_long, 2.0)[0]
    assert abs(ratio - np.exp(r1 * 3.0)) < 1e-6, "overdamped tail must follow slow root"

    print("model.py self-checks: ALL PASSED")


if __name__ == "__main__":
    _self_check()
