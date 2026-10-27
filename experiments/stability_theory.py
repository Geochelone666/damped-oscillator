"""Predicted stability boundaries vs. direct solver runs; independent of run_all.py.

For each (method, zeta) the amplification matrix in stability.py predicts the
boundary h* (largest step with spectral radius <= 1). Each prediction is then
checked against the actual solver: a run 5% below h* must stay bounded while a
run 5% above must outgrow ten times the exact solution's peak by t = 600.

Writes only data/stability_boundaries.csv. Euler at zeta = 0 is unstable for
every h > 0 (energy multiplier 1 + h^2 per step), so its boundary row is left
empty rather than reported as zero.
"""

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stability
from analysis import phase_error
from model import x_exact
from solvers import SOLVERS

ZETAS = (0.0, 0.1, 0.5, 1.0, 2.0)
T_END = 600.0
COLUMNS = ("method", "zeta", "h_boundary", "euler_boundary_2zeta",
           "bounded_5pct_below", "bounded_5pct_above")
DISPERSION_ZETA = 0.1
DISPERSION_H = (0.2, 0.1, 0.05, 0.025, 0.0125)
DISPERSION_COLUMNS = ("method", "zeta", "h", "predicted_drift_s_per_period",
                      "measured_drift_s_per_period")


def _is_bounded(method, zeta, h):
    t, x, _ = SOLVERS[method](zeta, h, T_END)
    ceiling = 10.0 * max(1.0, float(np.max(np.abs(x_exact(t, zeta)))))
    return bool(np.all(np.abs(x) <= ceiling))


def main():
    rows = []
    for method in SOLVERS:
        for zeta in ZETAS:
            h_star = stability.stability_boundary(method, zeta)
            analytic = 2.0 * zeta if method == "euler" and 0.0 < zeta <= 1.0 else ""
            if np.isfinite(h_star):
                below = _is_bounded(method, zeta, 0.95 * h_star)
                above = _is_bounded(method, zeta, 1.05 * h_star)
                print(f"{method:>6} zeta={zeta:g} h*={h_star:.6f} "
                      f"bounded(0.95h*)={below} bounded(1.05h*)={above}")
                rows.append((method, zeta, f"{h_star:.10f}", analytic, below, above))
            else:
                print(f"{method:>6} zeta={zeta:g} h*=none (unstable for all h)")
                rows.append((method, zeta, "", analytic, "", ""))

    path = ROOT / "data" / "stability_boundaries.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")

    # Dispersion cross-check: predicted per-period peak drift vs. the
    # peak-fit metric on the same [0, 20] grid used by metrics.csv.
    dispersion = []
    for method in SOLVERS:
        for h in DISPERSION_H:
            t, x, _ = SOLVERS[method](DISPERSION_ZETA, h, 20.0)
            predicted = stability.period_drift(method, DISPERSION_ZETA, h)
            measured = phase_error(t, x, DISPERSION_ZETA)
            dispersion.append((method, DISPERSION_ZETA, h,
                               f"{predicted:.9g}", f"{measured:.9g}"))
            print(f"{method:>6} h={h:g} predicted_drift={predicted:.6g} "
                  f"measured_drift={measured:.6g}")
    path = ROOT / "data" / "dispersion_check.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(DISPERSION_COLUMNS)
        writer.writerows(dispersion)
    print(f"Saved {len(dispersion)} rows to {path}")


if __name__ == "__main__":
    main()
