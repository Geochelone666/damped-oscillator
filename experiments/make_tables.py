"""Print the convergence-order table for paper/main.tex from data/metrics.csv.

    python3 experiments/make_tables.py          # print the LaTeX rows
    python3 experiments/make_tables.py --check  # exit 1 if paper/main.tex differs

Paths are relative to the repository, so any working directory works. The script never writes
files. Input must be 5 finite positive (h, E_inf) points per method and zeta, else ValueError.
This is not a general convergence estimator: it only reads the saved metrics table, whose contract
(data/README.md) is exactly 5 step sizes per method and zeta. The order is the least-squares slope of log(E_inf) against log(h) over the five
saved step sizes, per method and damping ratio. Only this table is generated;
the other numbers in the paper are checked by tests/test_documented_numbers.py.
"""
import csv
import math
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
            if len(sel) != 5:
                raise ValueError(f"{method} zeta={zeta}: expected 5 step sizes, found {len(sel)}")
            hs = [float(r["h"]) for r in sel]
            es = [float(r["E_inf"]) for r in sel]
            if not all(math.isfinite(v) and v > 0 for v in hs + es):
                raise ValueError(f"{method} zeta={zeta}: h and E_inf must be finite and > 0")
            if len(set(hs)) != 5:
                raise ValueError(f"{method} zeta={zeta}: step sizes must be 5 distinct values")
            slope = np.polyfit(np.log(hs), np.log(es), 1)[0]
            out[method].append(float(slope))
    return out


def latex_rows(metrics_path=ROOT / "data" / "metrics.csv"):
    table = orders(metrics_path)
    return [f"{label} & " + " & ".join(f"{p:.2f}" for p in table[m]) + " \\\\" for m, label in METHODS]


def main(argv):
    rows = latex_rows()
    if "--check" in argv:
        tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
        start = tex.find("Method & $\\zeta=0.1$")
        end = tex.find("\\end{tabular}", start)
        block = [ln.strip() for ln in tex[start:end].splitlines()] if start >= 0 and end > start else []
        missing = [r for r in rows if r not in block]
        for r in missing:
            print("missing in paper/main.tex:", r)
        return 1 if missing else 0
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
