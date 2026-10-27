"""
Exact linear stability analysis for the three integrators in solvers.py.

Every scheme in this repository is linear in the state y = (x, v), because
the acceleration a(x, v) = -x - 2*zeta*v is linear. One step is therefore a
constant 2x2 amplification matrix M:

    y_{n+1} = M(zeta, h) @ y_n

The long-time behaviour of a run is governed by the spectral radius
rho(M) = max |eig(M)|:

  * rho < 1: every numerical mode decays; the run is stable.
  * rho = 1: neutral (e.g. explicit Euler at h = 2*zeta, complex pair on the
    unit circle); envelopes neither decay nor grow.
  * rho > 1: a growing mode exists; the run diverges on long intervals.

The matrices are derived here by applying each documented one-step
recurrence to the basis vectors (1, 0) and (0, 1): column j of M is the
state after one step from e_j. tests/test_stability.py then verifies
M**k @ (1, 0) against the actual solver loops in solvers.py, so the
derivation cannot silently drift away from the implementation.

Known closed-form anchors (all verified in tests):

  * Euler: M = [[1, h], [-h, 1 - 2*zeta*h]], det(M) = 1 - 2*zeta*h + h^2.
    For 0 < zeta < 1 the eigenvalues are a complex pair with
    |lambda|^2 = det(M), so the boundary is exactly h = 2*zeta (0 < zeta <= 1).
    For zeta > 1 the eigenvalues are real and the boundary is instead
    h* = 2*(zeta - sqrt(zeta^2 - 1)) (e.g. 0.5359 at zeta = 2), which falls
    as zeta grows. Stability here is the non-strict test rho <= 1 (within
    1e-12); neutral cases such as zeta = 0 for Verlet-type count as stable
    even though energy is only conserved, not damped. At zeta = 0
    the energy multiplier per step is det(M) = 1 + h^2, the growth law used
    by experiments/energy_drift.py.
  * Verlet-type, zeta = 0: reduces to standard velocity Verlet for the
    harmonic oscillator, stable for h < 2.
  * RK4, zeta = 0: stability requires |R(i*h)| <= 1 with R the RK4
    stability polynomial, giving h < 2*sqrt(2) = 2.828427...

Only zeta >= 0 is analysed here: for anti-damping (zeta < 0) the exact
solution itself grows, so a one-mode spectral-radius test no longer
separates "method unstable" from "physical growth".
"""

import numpy as np

from solvers import SOLVERS, _accel, _validate_zeta


def _step_euler(x, v, zeta, h):
    """One explicit-Euler step of the recurrence in solvers.solve_euler."""
    return x + h * v, v + h * _accel(x, v, zeta)


def _step_verlet(x, v, zeta, h):
    """One step of the velocity-Verlet-type recurrence in solvers.solve_verlet."""
    a = _accel(x, v, zeta)
    x_new = x + h * v + 0.5 * h * h * a
    v_pred = v + h * a
    a_new = _accel(x_new, v_pred, zeta)
    v_new = v + 0.5 * h * (a + a_new)
    return x_new, v_new


def _step_rk4(x, v, zeta, h):
    """One classical-RK4 step on y' = (v, -x - 2*zeta*v)."""

    def rhs(xx, vv):
        return vv, _accel(xx, vv, zeta)

    k1x, k1v = rhs(x, v)
    k2x, k2v = rhs(x + 0.5 * h * k1x, v + 0.5 * h * k1v)
    k3x, k3v = rhs(x + 0.5 * h * k2x, v + 0.5 * h * k2v)
    k4x, k4v = rhs(x + h * k3x, v + h * k3v)
    return (
        x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x),
        v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v),
    )


_STEPS = {
    "euler": _step_euler,
    "verlet": _step_verlet,
    "rk4": _step_rk4,
}


def amplification_matrix(method, zeta, h):
    """Return the 2x2 one-step amplification matrix M for a method.

    Column j of M is the state after one step from basis vector e_j, using
    the same recurrence as the corresponding solver loop. M is real for
    every argument; complex eigenvalues appear only in np.linalg.eigvals.
    """
    if method not in _STEPS:
        raise ValueError(
            f"Unknown method {method!r}; valid methods: {', '.join(_STEPS)}"
        )
    _validate_zeta(zeta)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    step = _STEPS[method]
    col0 = step(1.0, 0.0, zeta, h)
    col1 = step(0.0, 1.0, zeta, h)
    return np.array([[col0[0], col1[0]], [col0[1], col1[1]]])


def spectral_radius(method, zeta, h):
    """rho(M) = max |eig(M)| for the method's one-step matrix."""
    return float(np.max(np.abs(np.linalg.eigvals(amplification_matrix(method, zeta, h)))))


def energy_growth_factor(method, zeta, h):
    """Per-step multiplier of the quadratic form det(M) for complex pairs.

    For an underdamped complex eigenpair, |lambda|^2 = det(M), so det(M) is
    exactly the per-step multiplier of x^2 + v^2 in the oscillatory mode.
    Explicit Euler at zeta = 0 gives 1 + h^2, the growth law quoted by
    experiments/energy_drift.py (E_n/E_0 = (1+h^2)**n).
    """
    return float(np.linalg.det(amplification_matrix(method, zeta, h)))


# Symplectic schemes sit exactly on rho = 1 in their stable range, and
# np.linalg.eigvals returns |lambda| = 1 + O(1e-16) there. Classify with a
# small tolerance so rounding cannot move the boundary; the jump of rho
# above a true boundary is orders of magnitude larger (e.g. rho = 1.065 at
# h = 2.001 for Verlet at zeta = 0).
_RHO_TOL = 1e-12


def _is_stable(method, zeta, h):
    return spectral_radius(method, zeta, h) <= 1.0 + _RHO_TOL


def stability_boundary(method, zeta, h_lo=1e-3, h_hi=10.0, tol=1e-10):
    # h_lo must be large enough that a genuinely unstable method exceeds the
    # rounding tolerance there: Euler at zeta = 0 has rho - 1 ~ h^2/2, which
    # only rises above _RHO_TOL for h >> 1e-6. A true boundary below h_lo is
    # reported as NaN (no bracket), so callers needing h* < 1e-3 must narrow
    # the range explicitly.
    """Largest h with rho(M) <= 1, found by bisection.

    Returns NaN when no bracket exists: rho(h_lo) > 1 means the method is
    unstable for every h in range (e.g. Euler at zeta = 0), rho(h_hi) < 1
    means the boundary lies above h_hi.
    """
    if not _is_stable(method, zeta, h_lo):
        return np.nan
    if _is_stable(method, zeta, h_hi):
        return np.nan
    for _ in range(200):
        h_mid = 0.5 * (h_lo + h_hi)
        if h_hi - h_lo <= tol:
            break
        if _is_stable(method, zeta, h_mid):
            h_lo = h_mid
        else:
            h_hi = h_mid
    return 0.5 * (h_lo + h_hi)


def numerical_frequency(method, zeta, h):
    """Angular frequency of the numerical oscillation mode (underdamped).

    The amplification matrix of an oscillatory run has a complex eigenpair
    lambda = rho * exp(+-i*theta); the numerical scheme advances the phase by
    theta per step, so its frequency is theta / h. The phase drift per period
    measured by analysis.phase_error is (2*pi / w_num - 2*pi / w_exact),
    i.e. the difference between numerical and exact periods.

    Returns NaN when the eigenvalues are real (no oscillation, e.g. Euler far
    above its stability boundary) or for |zeta| >= 1.
    """
    if not np.isfinite(zeta) or abs(zeta) >= 1.0:
        raise ValueError(f"numerical_frequency needs |zeta| < 1 (underdamped); got {zeta!r}")
    eigs = np.linalg.eigvals(amplification_matrix(method, zeta, h))
    osc = eigs[np.abs(eigs.imag) > 1e-15]
    if len(osc) == 0:
        return np.nan
    return float(abs(np.angle(osc[0])) / h)


def period_drift(method, zeta, h):
    """Predicted peak-time drift per oscillation period, in seconds.

    The exact period is T = 2*pi / wd with wd = sqrt(1 - zeta^2); the
    numerical period is 2*pi / w_num. A positive value means numerical peaks
    fall behind the exact ones (numerical frequency too low).
    """
    w_num = numerical_frequency(method, zeta, h)
    if not np.isfinite(w_num):
        return np.nan
    wd = np.sqrt(1.0 - zeta ** 2)
    return 2.0 * np.pi / w_num - 2.0 * np.pi / wd


def numerical_decay_rate(method, zeta, h):
    """Decay rate of the numerical oscillation mode (underdamped).

    For the complex eigenpair lambda = rho * exp(+-i*theta), peak amplitudes
    shrink by rho per period, so the fitted envelope follows exp(-zeta_num*t)
    with zeta_num = -ln(rho) / h. This predicts the zeta_fit values that
    analysis.envelope_decay_rate measures from peak amplitudes.

    Returns NaN when the eigenvalues are real (no oscillation); raises for
    |zeta| >= 1, matching numerical_frequency.
    """
    if not np.isfinite(zeta) or abs(zeta) >= 1.0:
        raise ValueError(f"numerical_decay_rate needs |zeta| < 1 (underdamped); got {zeta!r}")
    eigs = np.linalg.eigvals(amplification_matrix(method, zeta, h))
    osc = eigs[np.abs(eigs.imag) > 1e-15]
    if len(osc) == 0:
        return np.nan
    return float(-np.log(abs(osc[0])) / h)


def envelope_rel_err_prediction(method, zeta, h):
    """Predicted |zeta_num - zeta| / |zeta|; NaN at zeta = 0 by definition."""
    zeta_num = numerical_decay_rate(method, zeta, h)
    if zeta == 0 or not np.isfinite(zeta_num):
        return np.nan
    return abs(zeta_num - zeta) / abs(zeta)


def all_boundaries(zetas=(0.0, 0.1, 0.5, 1.0, 2.0)):
    """Boundary h for every (method, zeta); NaN where no bracket exists."""
    return {
        (method, zeta): stability_boundary(method, zeta)
        for method in SOLVERS
        for zeta in zetas
    }


def _self_check():
    """Spot checks against closed-form anchors."""
    # Euler boundary is exactly 2*zeta for 0 < zeta < 1.
    for zeta in (0.1, 0.5, 0.9):
        h_star = stability_boundary("euler", zeta)
        assert abs(h_star - 2 * zeta) < 1e-8, (zeta, h_star)
    # Euler at zeta=0: energy multiplier 1 + h^2, unstable for all h > 0.
    assert abs(energy_growth_factor("euler", 0.0, 0.3) - 1.09) < 1e-14
    assert np.isnan(stability_boundary("euler", 0.0))
    # Verlet at zeta=0: h* = 2; RK4 at zeta=0: h* = 2*sqrt(2).
    assert abs(stability_boundary("verlet", 0.0) - 2.0) < 1e-8
    assert abs(stability_boundary("rk4", 0.0) - 2.0 * np.sqrt(2.0)) < 1e-8
    # Numerical frequency approaches the exact wd as h -> 0, at rates
    # matching each method's dispersion order: Euler's relative error is
    # O(h) (the damping term couples in at first order), Verlet-type O(h^2),
    # RK4 O(h^4).
    wd = np.sqrt(1.0 - 0.1 ** 2)
    bounds = {"euler": 2e-4, "verlet": 1e-7, "rk4": 1e-11}
    for method, bound in bounds.items():
        err = abs(numerical_frequency(method, 0.1, 1e-3) - wd) / wd
        assert err < bound, (method, err)
    # Euler's decay rate vanishes exactly at h = 2*zeta (det M = 1), the
    # neutral point already recorded in stability_scan.csv.
    assert abs(numerical_decay_rate("euler", 0.1, 0.2)) < 1e-14
    print("stability.py self-checks: ALL PASSED")


if __name__ == "__main__":
    _self_check()
