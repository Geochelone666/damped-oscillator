"""Scan the explicit-Euler stability boundary at zeta=0.1.

Run independently of run_all.py; writes only data/stability_scan.csv.
"""

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import envelope_decay_rate
from model import x_exact
from solvers import SOLVERS

ZETA = 0.1
T_END = 40.0
H_VALUES = (0.16, 0.18, 0.19, 0.20, 0.21, 0.22, 0.25)
COLUMNS = ("method", "h", "zeta_fit_err", "E_inf")


def main():
    rows = []
    for method, solver in SOLVERS.items():
        for h in H_VALUES:
            t, x, _ = solver(ZETA, h, T_END)
            # Keep the constant-h grid inside [0, 40]; no shortened last step.
            mask = t <= T_END
            t, x = t[mask], x[mask]
            zeta_fit = envelope_decay_rate(t, x)
            if not np.isfinite(zeta_fit):
                raise ValueError(f"Envelope fit failed for {method}, h={h}")
            error = float(abs(zeta_fit - ZETA) / ZETA)
            e_inf = float(np.max(np.abs(x - x_exact(t, ZETA))))
            rows.append((method, h, error, e_inf))
            print(f"{method:>6} h={h:.2f} zeta_fit={zeta_fit:.9g} "
                  f"zeta_fit_err={error:.9g} E_inf={e_inf:.9g}")

    path = ROOT / "data" / "stability_scan.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
