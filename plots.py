"""
Paper figures for the damped-oscillator integrator benchmark.

Generates paper_figures/:
  fig1_underdamped_trajectory.png  -- zeta=0.1, h=0.2: x(t) vs exact (+ zoom inset)
  fig2_critical_trajectory.png     -- zeta=1,   h=0.2: x(t) vs exact
  fig3_overdamped_trajectory.png   -- zeta=2,   h=0.2: x(t) vs exact
  fig4_convergence_loglog.png      -- E_inf vs h (log-log), 3 panels, slope 1/2/4 guides
  fig5_cost_vs_error.png           -- CPU time vs E_inf (log-log), 3 panels
  fig6_stability_h05.png           -- zeta=0.1, h=0.5: Euler blows up, others survive

Run:  python3 plots.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from model import ZETAS, x_exact
from solvers import SOLVERS, SOLVER_LABELS
from analysis import H_VALUES, T_END, full_matrix

OUTDIR = "paper_figures"
COLORS = {"euler": "#d62728", "verlet": "#1f77b4", "rk4": "#2ca02c"}
ZETA_NAMES = {0.1: "underdamped ($\\zeta=0.1$)", 1.0: "critical ($\\zeta=1$)",
              2.0: "overdamped ($\\zeta=2$)"}

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150})


def _trajectory(zeta, h, fname, title, inset=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t_fine = np.linspace(0, T_END, 2001)
    ax.plot(t_fine, x_exact(t_fine, zeta), "k-", lw=1.8, label="exact")
    for method in SOLVERS:
        t, x, _ = SOLVERS[method](zeta, h, T_END)
        ax.plot(t, x, "--", color=COLORS[method], lw=1.2,
                label=f"{SOLVER_LABELS[method]}, h={h}")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper right")
    if inset:
        ax_in = ax.inset_axes([0.08, 0.12, 0.38, 0.38])
        t0, t1 = inset
        mask_f = (t_fine >= t0) & (t_fine <= t1)
        ax_in.plot(t_fine[mask_f], x_exact(t_fine[mask_f], zeta), "k-", lw=1.5)
        for method in SOLVERS:
            t, x, _ = SOLVERS[method](zeta, h, T_END)
            m = (t >= t0) & (t <= t1)
            ax_in.plot(t[m], x[m], "--", color=COLORS[method], lw=1.2)
        ax_in.set_xlim(t0, t1)
        ax_in.tick_params(labelsize=8)
        ax.indicate_inset_zoom(ax_in, edgecolor="0.5")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}")
    plt.close(fig)


def _convergence(results):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    hs = np.array(H_VALUES)
    for ax, zeta in zip(axes, ZETAS):
        for method in SOLVERS:
            errs = [results[zeta][method][h]["E_inf"] for h in H_VALUES]
            ax.loglog(hs, errs, "o-", color=COLORS[method], label=SOLVER_LABELS[method])
        # Reference slope guides anchored near the data (RK4's finest point).
        anchor_h, anchor_e = hs[-1], min(
            results[zeta][m][hs[-1]]["E_inf"] for m in SOLVERS) * 30
        for order, style in ((1, ":"), (2, "--"), (4, "-.")):
            ax.loglog(hs, anchor_e * (hs / anchor_h) ** order, "k" + style,
                      lw=1, label=f"slope {order}" if zeta == ZETAS[0] else None)
        ax.set_xlabel("step size h")
        ax.set_title(ZETA_NAMES[zeta])
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    axes[0].set_ylabel(r"$E_\infty = \max|x_n - x(t_n)|$")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Convergence: max displacement error vs step size")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig4_convergence_loglog.png")
    plt.close(fig)


def _cost_vs_error(results):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, zeta in zip(axes, ZETAS):
        for method in SOLVERS:
            cpu = [results[zeta][method][h]["cpu_s"] for h in H_VALUES]
            errs = [results[zeta][method][h]["E_inf"] for h in H_VALUES]
            ax.loglog(cpu, errs, "o-", color=COLORS[method], label=SOLVER_LABELS[method])
        ax.set_xlabel("CPU time per run (s)")
        ax.set_title(ZETA_NAMES[zeta])
    axes[0].set_ylabel(r"$E_\infty$")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Cost vs accuracy (lower-left is better)")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig5_cost_vs_error.png")
    plt.close(fig)


def _stability():
    # h = 0.5: forward Euler is unstable for zeta=0.1 (|1 + h*lambda| > 1),
    # while the Verlet-type and RK4 schemes stay bounded but visibly distorted.
    zeta, h = 0.1, 0.5
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t_fine = np.linspace(0, T_END, 2001)
    ax.plot(t_fine, x_exact(t_fine, zeta), "k-", lw=1.8, label="exact")
    for method in SOLVERS:
        t, x, _ = SOLVERS[method](zeta, h, T_END)
        ax.plot(t, x, "--", color=COLORS[method], lw=1.2,
                label=f"{SOLVER_LABELS[method]} (max|x|={np.max(np.abs(x)):.1e})")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)  [symlog scale]")
    ax.set_title("Large-step stability: $\\zeta=0.1$, h=0.5\n"
                 "forward Euler is linearly unstable here (|1+hλ|>1) and blows up")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig6_stability_h05.png")
    plt.close(fig)


def main():
    print("Running full experiment matrix (3 zeta x 5 h x 3 methods)...")
    results = full_matrix()
    print("Generating figures...")
    _trajectory(0.1, 0.2, "fig1_underdamped_trajectory.png",
                "Underdamped: numerical vs exact displacement (h=0.2)",
                inset=(5.0, 8.0))
    _trajectory(1.0, 0.2, "fig2_critical_trajectory.png",
                "Critically damped: numerical vs exact displacement (h=0.2)")
    _trajectory(2.0, 0.2, "fig3_overdamped_trajectory.png",
                "Overdamped: numerical vs exact displacement (h=0.2)")
    _convergence(results)
    _cost_vs_error(results)
    _stability()
    print("Done. Figures in paper_figures/:")

    import os
    for f in sorted(os.listdir(OUTDIR)):
        print("  ", f)


if __name__ == "__main__":
    main()
