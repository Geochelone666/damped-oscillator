"""Print the convergence-order table for paper/main.tex from data/metrics.csv.

    python3 experiments/make_tables.py          # print the LaTeX rows
    python3 experiments/make_tables.py --check  # exit 1 if paper/main.tex differs

The order is the least-squares slope of log(E_inf) against log(h) over the five
saved step sizes, per method and damping ratio. Only this table is generated;
the other numbers in the paper are checked by tests/test_documented_numbers.py.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = (("euler", "Explicit Euler"), ("verlet", "Velocity-Verlet-type"), ("rk4", "RK4"))
ZETAS = (0.1, 1.0, 2.0)


def orders(metrics_path=ROOT / "data" / "metrics.csv"):
    with Path(metrics_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    out = {}
    for method, _ in METHODS:
        out[method] = []
        for zeta in ZETAS:
            sel = [r for r in rows if r["method"] == method and float(r["zeta"]) == zeta]
            slope = np.polyfit(np.log([float(r["h"]) for r in sel]),
                               np.log([float(r["E_inf"]) for r in sel]), 1)[0]
            out[method].append(float(slope))
    return out


def latex_rows(metrics_path=ROOT / "data" / "metrics.csv"):
    table = orders(metrics_path)
    return [f"{label} & " + " & ".join(f"{p:.2f}" for p in table[m]) + " \\\\" for m, label in METHODS]


def main(argv):
    rows = latex_rows()
    if "--check" in argv:
        tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
        missing = [r for r in rows if r not in tex]
        for r in missing:
            print("missing in paper/main.tex:", r)
        return 1 if missing else 0
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
