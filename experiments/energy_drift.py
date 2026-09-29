"""Undamped long-term energy test; run independently of run_all.py.

Metrics use every integration step; curves retain 201 grid samples per run.
Euler obeys E_n/E_0 = (1+h*h)**n, so growth is only initially linear.
"""

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solvers import SOLVERS

ZETA = 0.0
T_END = 100.0
H_VALUES = (0.2, 0.1, 0.05)
N_SAMPLES = 201


def main():
    metrics, curves = [], []
    for method, solver in SOLVERS.items():
        for h in H_VALUES:
            t, x, v = solver(ZETA, h, T_END)
            energy = (x*x + v*v) / 2.0
            e0 = energy[0]
            if e0 != 0.5 or not np.all(np.isfinite(energy)):
                raise ValueError(f"Invalid energy for {method}, h={h}")
            rel_error = np.abs(energy - e0) / e0
            metrics.append((method, h, float(rel_error.max()), float(rel_error[-1])))
            # Nearest integration indices, including both endpoints; no interpolation.
            indices = np.rint(np.linspace(0, len(t) - 1, N_SAMPLES)).astype(int)
            curves.extend((method, h, t[i], energy[i]) for i in indices)
            print(f"{method:>6} h={h:g} E0={e0:g} "
                  f"max_rel_E_err={rel_error.max():.9g} "
                  f"final_rel_E_err={rel_error[-1]:.9g}")

    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name, columns, rows in (
        ("energy_drift.csv", ("method", "h", "max_rel_E_err", "final_rel_E_err"), metrics),
        ("energy_curves.csv", ("method", "h", "t", "E"), curves),
    ):
        with (data / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {data / name}")


if __name__ == "__main__":
    main()
