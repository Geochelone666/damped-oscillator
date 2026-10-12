"""
Error metrics and convergence analysis for the damped-oscillator benchmark.

Metrics per (method, zeta, h):
  * E_inf  = max |x_n - x(t_n)|            (worst-point displacement error)
  * E_RMS  = rms  |x_n - x(t_n)|            (average displacement error)
  * phase_err (underdamped only): peak-time drift per oscillation period
      relative to the exact solution (sub-grid refined).
  * envelope_err (underdamped only): |zeta_fit - zeta| / |zeta|, where
      zeta_fit is the decay rate from a least-squares fit of ln(peak
      amplitudes). NaN for zeta = 0, where a relative error is undefined;
      negative zeta (anti-damping) is normalized by |zeta| so the error
      keeps its usual sign-independent meaning.
  * cpu_s: wall time of solve() (minimum of 3 repeats).

Convergence order: least-squares slope of log(E_inf) vs log(h); expected
slopes are ~1 (Euler), ~2 (Verlet-type), ~4 (RK4).
"""

import time

import numpy as np

from model import ZETAS, x_exact
from solvers import SOLVERS, get_solver

T_END = 20.0
H_VALUES = (0.2, 0.1, 0.05, 0.025, 0.0125)


def _local_maxima(t, x):
    """Indices of strict local maxima, refined by parabolic interpolation.

    Returns (t_peaks, x_peaks) with sub-grid peak positions.
    """
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    t_p, x_p = [], []
    for k in idx:
        # Parabolic refinement through (t[k-1],x[k-1]), (t[k],x[k]), (t[k+1],x[k+1]).
        denom = x[k - 1] - 2 * x[k] + x[k + 1]
        if denom == 0:
            continue
        shift = 0.5 * (x[k - 1] - x[k + 1]) / denom  # in units of grid spacing
        dt = t[1] - t[0]
        t_p.append(t[k] + shift * dt)
        x_p.append(x[k] - 0.25 * (x[k - 1] - x[k + 1]) * shift)
        # (vertex value of the interpolating parabola)
    return np.array(t_p), np.array(x_p)


def phase_error(t_num, x_num, zeta):
    """Dispersion (frequency) error per oscillation period (underdamped only).

    With z = zeta and wd = sqrt(1 - z**2), differentiating the exact solution
    gives v(t) = -(wd + z**2/wd) e^{-zt} sin(wd t). Thus v(t) = 0 at
    t = k*T/2, with T = 2*pi/wd. The initial point is a maximum since
    x(0) = 1, v(0) = 0 and x''(0) = -1. Extrema alternate, so maxima
    occur at even half-periods, t_k = k*T.
    We use the maxima with integer k >= 1. A frequency error makes numerical
    peak times drift linearly with k, so we fit
        t_num,k - k*T = a + b*k
    and return b, the drift per period in seconds. (Using only the first
    peak's offset would conflate one period of drift with an initial
    transient offset, which partially cancel for some methods.)
    """
    wd = np.sqrt(1.0 - zeta ** 2)
    T = 2.0 * np.pi / wd
    t_peaks, _ = _local_maxima(t_num, x_num)
    t_peaks = t_peaks[t_peaks > 1.0]  # skip the t = 0 initial peak
    if len(t_peaks) < 2:
        return np.nan
    k = np.arange(1, len(t_peaks) + 1)
    b, _ = np.polyfit(k, t_peaks - k * T, 1)
    return b


def envelope_decay_rate(t_num, x_num):
    """Decay rate fitted from ln(peak amplitude) vs t (underdamped only).

    For x(t) ~ A e^{-zeta t} cos(...), peak amplitudes satisfy
    ln A_k = ln A_0 - zeta * t_k. Least-squares slope gives -zeta_fit.
    """
    t_peaks, x_peaks = _local_maxima(t_num, x_num)
    pos = x_peaks > 0
    t_peaks, x_peaks = t_peaks[pos], x_peaks[pos]
    if len(t_peaks) < 2:
        return np.nan
    slope, _ = np.polyfit(t_peaks, np.log(x_peaks), 1)
    return -slope


def measure_cpu(method, zeta, h, t_end=T_END, repeats=3):
    """Wall-clock time of one solve(); minimum over `repeats` runs."""
    solver = get_solver(method)
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        solver(zeta, h, t_end)
        best = min(best, time.perf_counter() - start)
    return best


def run_case(method, zeta, h, t_end=T_END):
    """Full metric set for one (method, zeta, h) combination."""
    t, x, v = get_solver(method)(zeta, h, t_end)
    x_ref = x_exact(t, zeta)
    err = x - x_ref
    metrics = {
        "E_inf": float(np.max(np.abs(err))),
        "E_RMS": float(np.sqrt(np.mean(err ** 2))),
        "cpu_s": measure_cpu(method, zeta, h, t_end),
    }
    if zeta < 1.0:
        metrics["phase_err"] = float(phase_error(t, x, zeta))
        zeta_fit = envelope_decay_rate(t, x)
        metrics["zeta_fit"] = float(zeta_fit) if np.isfinite(zeta_fit) else np.nan
        if zeta != 0 and np.isfinite(zeta_fit):
            # Negative zeta is legal anti-damping (see _validate_zeta); a
            # relative error against it is meaningful only when normalized
            # by |zeta|, not zeta (which would flip the sign).
            metrics["envelope_rel_err"] = float(abs(zeta_fit - zeta) / abs(zeta))
        else:
            # A relative error is meaningless against a zero reference:
            # report NaN instead of dividing by zero (which yielded inf).
            metrics["envelope_rel_err"] = np.nan
    return metrics


def convergence_order(zeta, method, h_values=H_VALUES, t_end=T_END):
    """Least-squares slope of log(E_inf) vs log(h); the empirical order."""
    solver = get_solver(method)
    hs, errs = [], []
    for h in h_values:
        t, x, _ = solver(zeta, h, t_end)
        errs.append(np.max(np.abs(x - x_exact(t, zeta))))
        hs.append(h)
    slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
    return float(slope)


def full_matrix(t_end=T_END, h_values=H_VALUES):
    """Run every (zeta, h, method): returns nested dict results[zeta][method][h]."""
    results = {}
    for zeta in ZETAS:
        results[zeta] = {}
        for method in SOLVERS:
            results[zeta][method] = {}
            for h in h_values:
                results[zeta][method][h] = run_case(method, zeta, h, t_end)
    return results


if __name__ == "__main__":
    # Quick convergence-order check across the whole matrix.
    expected = {"euler": 1.0, "verlet": 2.0, "rk4": 4.0}
    print(f"{'zeta':>6} {'method':>8} {'order':>7} {'expected':>9}")
    for zeta in ZETAS:
        for method in SOLVERS:
            order = convergence_order(zeta, method)
            print(f"{zeta:>6} {method:>8} {order:>7.2f} {expected[method]:>9.1f}")
