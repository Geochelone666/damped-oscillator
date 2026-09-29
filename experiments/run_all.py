"""Save the existing benchmark matrix and figure inputs as CSV files."""

import csv
from pathlib import Path
import platform
import sys

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import H_VALUES, T_END, full_matrix
from model import ZETAS, x_exact
from solvers import SOLVERS, SOLVER_LABELS

DATA = ROOT / "data"
COLUMNS = ("method", "zeta", "h", "E_inf", "E_RMS", "phase_slope",
           "zeta_fit_err", "cpu_time")


def write_csv(name, columns, rows):
    with (DATA / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    results = full_matrix()
    rows = []
    for zeta in ZETAS:
        for method in SOLVERS:
            for h in H_VALUES:
                metrics = results[zeta][method][h]
                rows.append((method, zeta, h, metrics["E_inf"], metrics["E_RMS"],
                             metrics.get("phase_err", np.nan),
                             metrics.get("envelope_rel_err", np.nan), metrics["cpu_s"]))
    write_csv("metrics.csv", COLUMNS, rows)

    # Only the trajectories already used by figs 1–3 and 6. The h=0.5 case
    # reproduces the existing figure; it is not a new scan or a metric row.
    trajectories = []
    for zeta, h in [(z, 0.2) for z in ZETAS] + [(0.1, 0.5)]:
        for method, solver in SOLVERS.items():
            t, x, v = solver(zeta, h, T_END)
            trajectories.extend((method, zeta, h, tt, xx, vv)
                                for tt, xx, vv in zip(t, x, v))
    write_csv("trajectories.csv", ("method", "zeta", "h", "t", "x", "v"), trajectories)

    t_fine = np.linspace(0, T_END, 2001)
    write_csv("exact.csv", ("zeta", "t", "x"),
              ((zeta, t, x) for zeta in ZETAS
               for t, x in zip(t_fine, x_exact(t_fine, zeta))))
    write_csv("methods.csv", ("method", "label"), SOLVER_LABELS.items())
    write_csv("environment.csv", ("component", "version"),
              (("python", platform.python_version()), ("numpy", np.__version__),
               ("matplotlib", matplotlib.__version__)))
    print(f"Saved {len(rows)} metric rows, {len(trajectories)} trajectory rows, "
          f"and {len(ZETAS) * len(t_fine)} exact samples to {DATA}")
    print("Convergence slopes: log(E_inf) vs log(h)")
    print(f"{'zeta':>6} {'method':>8} {'order':>8}")
    for zeta in ZETAS:
        for method in SOLVERS:
            errors = [results[zeta][method][h]["E_inf"] for h in H_VALUES]
            slope = np.polyfit(np.log(H_VALUES), np.log(errors), 1)[0]
            print(f"{zeta:6g} {method:>8} {slope:8.4f}")


if __name__ == "__main__":
    main()
