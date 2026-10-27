"""
Paper figures for the damped-oscillator integrator benchmark.

Generates paper_figures/:
  fig1_underdamped_trajectory.png  -- zeta=0.1, h=0.2: x(t) vs exact (+ zoom inset)
  fig2_critical_trajectory.png     -- zeta=1,   h=0.2: x(t) vs exact
  fig3_overdamped_trajectory.png   -- zeta=2,   h=0.2: x(t) vs exact
  fig4_convergence_loglog.png      -- E_inf vs h (log-log), 3 panels, slope 1/2/4 guides
  fig5_cost_vs_error.png           -- CPU time vs E_inf (log-log), 3 panels
  fig6_stability_h05.png           -- zeta=0.1, h=0.5: Euler blows up, others survive
  fig7_stability_boundary.png      -- zeta=0.1: envelope error near h=0.2
  fig8_energy_drift.png            -- zeta=0, h=0.1: relative energy error vs t
  fig9_stability_curves.png        -- predicted boundary h*(zeta) per method,
                                       with the solver-validated points from
                                       stability_boundaries.csv overlaid

Run:  python3 figures/make_figures.py (reads only saved CSV data)
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTDIR = ROOT / "paper_figures"


def _read_csv(name):
    path = DATA / name
    if not path.is_file():
        runner = ("stability_scan.py" if name == "stability_scan.csv" else "run_all.py")
        if name in ("energy_drift.csv", "energy_curves.csv"):
            runner = "energy_drift.py"
        if name == "stability_boundaries.csv":
            runner = "stability_theory.py"
        raise FileNotFoundError(
            f"Missing {path}. Run python3 experiments/{runner} first.")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _load_inputs():
    labels = {row["method"]: row["label"] for row in _read_csv("methods.csv")}
    results = {}
    for row in _read_csv("metrics.csv"):
        zeta, h = float(row["zeta"]), float(row["h"])
        methods = results.setdefault(zeta, {})
        cases = methods.setdefault(row["method"], {})
        if h in cases:
            raise ValueError(f"Duplicate metric row: {row['method']}, {zeta}, {h}")
        cases[h] = {key: float(value) for key, value in row.items()
                    if key not in ("method", "zeta", "h")}
    zetas = tuple(sorted(results))
    hs = tuple(sorted({h for methods in results.values()
                       for cases in methods.values() for h in cases}, reverse=True))
    for zeta in zetas:
        if set(results[zeta]) != set(labels):
            raise ValueError(f"Incomplete methods for zeta={zeta}")
        for method in labels:
            if set(results[zeta][method]) != set(hs):
                raise ValueError(f"Incomplete step sizes for {method}, zeta={zeta}")

    trajectories = {}
    for row in _read_csv("trajectories.csv"):
        key = (row["method"], float(row["zeta"]), float(row["h"]))
        trajectories.setdefault(key, []).append((float(row["t"]), float(row["x"])))
    exact = {}
    for row in _read_csv("exact.csv"):
        exact.setdefault(float(row["zeta"]), []).append((float(row["t"]), float(row["x"])))
    trajectories = {key: np.array(sorted(rows)).T for key, rows in trajectories.items()}
    exact = {key: np.array(sorted(rows)).T for key, rows in exact.items()}
    return results, labels, zetas, hs, trajectories, exact


COLORS = {"euler": "#d62728", "verlet": "#1f77b4", "rk4": "#2ca02c"}
ZETA_NAMES = {0.1: "underdamped ($\\zeta=0.1$)", 1.0: "critical ($\\zeta=1$)",
              2.0: "overdamped ($\\zeta=2$)"}

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150})


def _trajectory(zeta, h, fname, title, labels, trajectories, exact, inset=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t_fine, x_fine = exact[zeta]
    ax.plot(t_fine, x_fine, "k-", lw=1.8, label="exact")
    for method in labels:
        t, x = trajectories[(method, zeta, h)]
        ax.plot(t, x, "--", color=COLORS[method], lw=1.2,
                label=f"{labels[method]}, h={h}")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper right")
    if inset:
        ax_in = ax.inset_axes([0.08, 0.12, 0.38, 0.38])
        t0, t1 = inset
        mask_f = (t_fine >= t0) & (t_fine <= t1)
        ax_in.plot(t_fine[mask_f], x_fine[mask_f], "k-", lw=1.5)
        for method in labels:
            t, x = trajectories[(method, zeta, h)]
            m = (t >= t0) & (t <= t1)
            ax_in.plot(t[m], x[m], "--", color=COLORS[method], lw=1.2)
        ax_in.set_xlim(t0, t1)
        ax_in.tick_params(labelsize=8)
        ax.indicate_inset_zoom(ax_in, edgecolor="0.5")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}")
    plt.close(fig)


def _convergence(results, labels, zetas, h_values):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    hs = np.array(h_values)
    for ax, zeta in zip(axes, zetas):
        for method in labels:
            errs = [results[zeta][method][h]["E_inf"] for h in h_values]
            ax.loglog(hs, errs, "o-", color=COLORS[method], label=labels[method])
        # Reference slope guides anchored near the data (RK4's finest point).
        anchor_h, anchor_e = hs[-1], min(
            results[zeta][m][hs[-1]]["E_inf"] for m in labels) * 30
        for order, style in ((1, ":"), (2, "--"), (4, "-.")):
            ax.loglog(hs, anchor_e * (hs / anchor_h) ** order, "k" + style,
                      lw=1, label=f"slope {order}" if zeta == zetas[0] else None)
        ax.set_xlabel("step size h")
        ax.set_title(ZETA_NAMES[zeta])
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    axes[0].set_ylabel(r"$E_\infty = \max|x_n - x(t_n)|$")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Convergence: max displacement error vs step size")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig4_convergence_loglog.png")
    plt.close(fig)


def _cost_vs_error(results, labels, zetas, h_values):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, zeta in zip(axes, zetas):
        for method in labels:
            cpu = [results[zeta][method][h]["cpu_time"] for h in h_values]
            errs = [results[zeta][method][h]["E_inf"] for h in h_values]
            ax.loglog(cpu, errs, "o-", color=COLORS[method], label=labels[method])
        ax.set_xlabel("CPU time per run (s)")
        ax.set_title(ZETA_NAMES[zeta])
    axes[0].set_ylabel(r"$E_\infty$")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Cost vs accuracy (lower-left is better)")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig5_cost_vs_error.png")
    plt.close(fig)


def _stability(labels, trajectories, exact):
    # h = 0.5: forward Euler is unstable for zeta=0.1 (|1 + h*lambda| > 1),
    # while the Verlet-type and RK4 schemes stay bounded but visibly distorted.
    zeta, h = 0.1, 0.5
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t_fine, x_fine = exact[zeta]
    ax.plot(t_fine, x_fine, "k-", lw=1.8, label="exact")
    for method in labels:
        t, x = trajectories[(method, zeta, h)]
        ax.plot(t, x, "--", color=COLORS[method], lw=1.2,
                label=f"{labels[method]} (max|x|={np.max(np.abs(x)):.1e})")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)  [symlog scale]")
    ax.set_title("Large-step stability: $\\zeta=0.1$, h=0.5\n"
                 "forward Euler is linearly unstable here (|1+hλ|>1) and blows up")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig6_stability_h05.png")
    plt.close(fig)


def _stability_boundary():
    """Render fig7 using only the standalone stability scan CSV."""
    methods = {}
    for row in _read_csv("stability_scan.csv"):
        methods.setdefault(row["method"], []).append(
            (float(row["h"]), float(row["zeta_fit_err"])))
    labels = {"euler": "Euler", "verlet": "Velocity-Verlet-type", "rk4": "RK4"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, rows in methods.items():
        h, error = np.array(sorted(rows)).T
        ax.semilogy(h, error, "o-", color=COLORS[method], label=labels[method])
    ax.axvline(0.2, color="0.3", linestyle="--", label="theory: h=2ζ")
    ax.set_xlabel("step size h")
    ax.set_ylabel(r"Envelope decay error $|\zeta_{\mathrm{fit}}-\zeta|/\zeta$")
    ax.set_title(r"Euler stability boundary: $\zeta=0.1$, $0\leq t\leq 40$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig7_stability_boundary.png")
    plt.close(fig)


def _energy_drift():
    """Render fig8 from the two standalone energy CSVs only."""
    maxima = {row["method"]: float(row["max_rel_E_err"])
              for row in _read_csv("energy_drift.csv") if float(row["h"]) == 0.1}
    curves = {}
    for row in _read_csv("energy_curves.csv"):
        if float(row["h"]) == 0.1:
            curves.setdefault(row["method"], []).append((float(row["t"]), float(row["E"])))
    labels = {"euler": "Euler", "verlet": "Velocity-Verlet-type", "rk4": "RK4"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, label in labels.items():
        t, energy = np.array(sorted(curves[method])).T
        e0 = energy[0]
        if t[0] != 0.0 or e0 != 0.5:
            raise ValueError(f"Invalid initial energy sample for {method}")
        error = np.abs(energy - e0) / e0
        # Zero error (including t=0) is undefined on a logarithmic axis.
        error = np.ma.masked_less_equal(error, 0.0)
        ax.semilogy(t, error, color=COLORS[method], lw=1.2,
                    label=f"{label} (max={maxima[method]:.2e})")
    ax.set_xlabel("t")
    ax.set_ylabel(r"Relative energy error $|E(t)-E_0|/E_0$")
    ax.set_title(r"Undamped energy conservation: $\zeta=0$, $h=0.1$, $E_0=0.5$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig8_energy_drift.png")
    plt.close(fig)


def _stability_curves():
    """Render fig9: predicted h*(zeta) curves vs solver-validated points.

    The dense curves come from stability.stability_boundary (eigenvalue
    computations only, no solver runs, per the figures-do-not-run-solvers
    rule); the overlaid markers are the solver-validated rows of
    stability_boundaries.csv.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    import stability

    validated = {}
    for row in _read_csv("stability_boundaries.csv"):
        if row["h_boundary"]:
            validated.setdefault(row["method"], []).append(
                (float(row["zeta"]), float(row["h_boundary"])))
    labels = {"euler": "Euler", "verlet": "Velocity-Verlet-type", "rk4": "RK4"}
    zeta_grid = np.linspace(0.01, 2.0, 60)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, label in labels.items():
        curve = [stability.stability_boundary(method, float(z)) for z in zeta_grid]
        ax.plot(zeta_grid, curve, color=COLORS[method], lw=1.4, label=label)
        pts = np.array(sorted(validated[method])).T
        ax.plot(pts[0], pts[1], "o", color=COLORS[method], ms=5, mfc="none")
    ax.plot([], [], "o", color="0.3", ms=5, mfc="none",
            label="solver-validated (data/stability_boundaries.csv)")
    ax.set_xlabel(r"damping ratio $\zeta$")
    ax.set_ylabel(r"stability boundary $h^*$")
    ax.set_title("Predicted stability boundaries (amplification-matrix spectral radius)\n"
                 "boundaries are not monotone in $\zeta$ for Verlet-type / RK4")
    ax.set_ylim(0, 3.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig9_stability_curves.png")
    plt.close(fig)


def main():
    results, labels, zetas, h_values, trajectories, exact = _load_inputs()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures from saved CSV data...")
    _trajectory(0.1, 0.2, "fig1_underdamped_trajectory.png",
                "Underdamped: numerical vs exact displacement (h=0.2)",
                labels, trajectories, exact, inset=(5.0, 8.0))
    _trajectory(1.0, 0.2, "fig2_critical_trajectory.png",
                "Critically damped: numerical vs exact displacement (h=0.2)",
                labels, trajectories, exact)
    _trajectory(2.0, 0.2, "fig3_overdamped_trajectory.png",
                "Overdamped: numerical vs exact displacement (h=0.2)",
                labels, trajectories, exact)
    _convergence(results, labels, zetas, h_values)
    _cost_vs_error(results, labels, zetas, h_values)
    _stability(labels, trajectories, exact)
    _stability_boundary()
    _energy_drift()
    _stability_curves()
    print("Done. Figures in paper_figures/:")

    import os
    for f in sorted(os.listdir(OUTDIR)):
        print("  ", f)


if __name__ == "__main__":
    main()
