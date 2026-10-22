"""
Numerical integrators for  x'' + 2*zeta*x' + x = 0,  x(0) = 1, v(0) = 0.

Three methods with a unified interface  solve(zeta, h, t_end) -> (t, x, v):

  * euler      -- explicit (forward) Euler on the first-order system y = (x, v).
  * verlet     -- velocity-Verlet-TYPE scheme adapted to velocity-dependent
                  damping. NOTE: the textbook velocity-Verlet is derived for
                  accelerations that depend on position only, a = a(x). The
                  damping term 2*zeta*v makes a = a(x, v), so the standard
                  scheme does not apply directly. The variant used here
                  predicts the velocity to evaluate the new acceleration:

                      a_n       = -x_n - 2*zeta*v_n
                      x_{n+1}   = x_n + h*v_n + 0.5*h^2*a_n
                      v_pred    = v_n + h*a_n
                      a_{n+1}   = -x_{n+1} - 2*zeta*v_pred
                      v_{n+1}   = v_n + 0.5*h*(a_n + a_{n+1})

                  We label it "velocity-Verlet-type" (not plain Verlet) and
                  document its limitation in README.md: the predictor step is
                  only first-order accurate in v, so the scheme is formally
                  second-order overall but with a larger error constant than
                  true velocity-Verlet on conservative systems.
  * rk4        -- classical 4th-order Runge-Kutta on the state vector y=(x, v).

Only numpy is used (no scipy ODE solvers), so every step is explicit here.
"""

import numpy as np


def _accel(x, v, zeta):
    """Right-hand side of v' = a(x, v) = -x - 2*zeta*v (unit mass, unit stiffness)."""
    return -x - 2.0 * zeta * v


def _step_count(h, t_end):
    """Validate the time interval before constructing a constant-step grid."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not np.isfinite(t_end) or t_end < 0:
        raise ValueError("t_end must be finite and nonnegative")
    # Largest n with n*h <= t_end (up to a relative rounding slack, so that
    # e.g. 20/0.1 = 199.99999999999997 still yields 200 steps). Plain
    # round() silently integrated *past* t_end by up to h/2 whenever t_end
    # was not a multiple of h (e.g. h=0.19, t_end=40 ended at t=40.09).
    return int(np.floor(t_end / h * (1.0 + 1e-12) + 1e-12))


def _validate_zeta(zeta):
    """Reject non-finite damping ratios before they silently poison a run.

    NaN/inf zeta makes _accel return NaN and the solvers hand back
    partially-NaN (x, v) arrays with no error. Negative zeta stays legal:
    it is physically meaningful (anti-damping / energy injection).
    """
    if not np.isfinite(zeta):
        raise ValueError(f"zeta must be finite; got {zeta!r}")


def solve_euler(zeta, h, t_end):
    """Explicit Euler: x_{n+1} = x_n + h*v_n ; v_{n+1} = v_n + h*a(x_n, v_n)."""
    _validate_zeta(zeta)
    n = _step_count(h, t_end)
    t = np.linspace(0.0, n * h, n + 1)
    x = np.empty(n + 1)
    v = np.empty(n + 1)
    x[0], v[0] = 1.0, 0.0
    for k in range(n):
        x[k + 1] = x[k] + h * v[k]
        v[k + 1] = v[k] + h * _accel(x[k], v[k], zeta)
    return t, x, v


def solve_verlet(zeta, h, t_end):
    """Velocity-Verlet-type scheme for velocity-dependent damping (see module docstring)."""
    _validate_zeta(zeta)
    n = _step_count(h, t_end)
    t = np.linspace(0.0, n * h, n + 1)
    x = np.empty(n + 1)
    v = np.empty(n + 1)
    x[0], v[0] = 1.0, 0.0
    a = _accel(x[0], v[0], zeta)
    for k in range(n):
        x[k + 1] = x[k] + h * v[k] + 0.5 * h * h * a
        v_pred = v[k] + h * a          # predictor for the damping term
        a_new = _accel(x[k + 1], v_pred, zeta)
        v[k + 1] = v[k] + 0.5 * h * (a + a_new)
        # The next step starts from the corrected velocity, not v_pred.
        a = _accel(x[k + 1], v[k + 1], zeta)
    return t, x, v


def solve_rk4(zeta, h, t_end):
    """Classical RK4 on y = (x, v) with y' = (v, -x - 2*zeta*v)."""
    _validate_zeta(zeta)
    n = _step_count(h, t_end)
    t = np.linspace(0.0, n * h, n + 1)
    x = np.empty(n + 1)
    v = np.empty(n + 1)
    x[0], v[0] = 1.0, 0.0

    def rhs(xx, vv):
        return vv, _accel(xx, vv, zeta)

    for k in range(n):
        k1x, k1v = rhs(x[k], v[k])
        k2x, k2v = rhs(x[k] + 0.5 * h * k1x, v[k] + 0.5 * h * k1v)
        k3x, k3v = rhs(x[k] + 0.5 * h * k2x, v[k] + 0.5 * h * k2v)
        k4x, k4v = rhs(x[k] + h * k3x, v[k] + h * k3v)
        x[k + 1] = x[k] + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v[k + 1] = v[k] + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return t, x, v


SOLVERS = {
    "euler": solve_euler,
    "verlet": solve_verlet,
    "rk4": solve_rk4,
}

# Short display labels used in figures and tables.
SOLVER_LABELS = {
    "euler": "Euler (1st order)",
    "verlet": "Velocity-Verlet-type (2nd order)",
    "rk4": "RK4 (4th order)",
}


def get_solver(method):
    """Return a solver by name, rejecting unknown methods explicitly."""
    try:
        return SOLVERS[method]
    except KeyError:
        raise ValueError(
            f"Unknown method {method!r}; valid methods: {', '.join(SOLVERS)}"
        ) from None


def solve(method, zeta, h, t_end):
    """Unified entry point: solve(method, zeta, h, t_end) -> (t, x, v)."""
    return get_solver(method)(zeta, h, t_end)


def _self_check():
    """Sanity checks: undamped limit, exact initial data, interface contract."""
    from model import x_exact

    # 1. zeta = 0 must reproduce the undamped oscillator x = cos(t) to the
    #    method's expected accuracy at small h. (Short interval: explicit Euler
    #    slowly pumps energy into undamped oscillations, so a long interval
    #    would fail this check for reasons unrelated to correctness.)
    for method in SOLVERS:
        t, x, v = solve(method, 0.0, 0.01, 5.0)
        err = np.max(np.abs(x - np.cos(t)))
        assert err < 5e-2, f"{method}: zeta=0 error {err} too large"

    # 2. Initial values and output shapes / grid consistency.
    for method in SOLVERS:
        t, x, v = solve(method, 0.1, 0.1, 20.0)
        assert len(t) == len(x) == len(v) == 201
        assert t[0] == 0.0 and abs(t[-1] - 20.0) < 1e-12
        assert x[0] == 1.0 and v[0] == 0.0

    # 3. A fine RK4 run must match the analytical solution closely
    #    (validates both the solver and model.py against each other).
    t, x, _ = solve("rk4", 0.1, 0.005, 20.0)
    assert np.max(np.abs(x - x_exact(t, 0.1))) < 1e-7

    print("solvers.py self-checks: ALL PASSED")


if __name__ == "__main__":
    _self_check()
