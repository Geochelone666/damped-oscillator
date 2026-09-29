# Saved benchmark data

Regenerate with `python3 experiments/run_all.py` from the repository root.
The runner uses `analysis.full_matrix()` unchanged. All times and displacements
use the original dimensionless model: x(0)=1, v(0)=0, t in [0,20].

## metrics.csv

Exactly 45 rows: three methods (`euler`, `verlet`, `rk4`), damping ratios
zeta in {0.1, 1, 2}, and steps h in {0.2, 0.1, 0.05, 0.025, 0.0125}.

| Column | Meaning |
| --- | --- |
| method | Solver key; `verlet` is the velocity-Verlet-type damping variant. |
| zeta | Dimensionless damping ratio. |
| h | Integration time step. |
| E_inf | Maximum absolute displacement error against the analytic solution on the solver grid. |
| E_RMS | Square root of the mean squared displacement error on that grid. |
| phase_slope | Preserved `analysis.run_case()["phase_err"]` value; see the existing implementation caveat below. |
| zeta_fit_err | Relative envelope decay error, abs(zeta_fit-zeta)/zeta, from `envelope_rel_err`. |
| cpu_time | Wall-clock seconds per solve (`cpu_s`), minimum of three repeats using `time.perf_counter`. |

Phase and envelope values are `nan` for critical/overdamped cases, where these
oscillatory metrics do not apply, or when a fit cannot be computed.

**Existing phase-metric caveat:** `analysis.phase_error()` documents the slope
of numerical peak-time drift versus peak index, but its `_, b = np.polyfit(...)`
assignment returns the fitted intercept. The requested `phase_slope` column
retains that existing result verbatim; it must not be interpreted as a measured
slope. No science code was changed in this pipeline refactor.

## Figure inputs

The aggregate metrics cannot reconstruct trajectories, so the runner also saves:

- `trajectories.csv`: `method,zeta,h,t,x,v`, one row per solver sample. Includes
  h=0.2 for all three damping ratios (figs 1–3) and the existing zeta=0.1,
  h=0.5 example (fig6), each with all three methods. This preserves the old
  figure inputs; it adds no stability scan or extra rows to the metric matrix.
- `exact.csv`: `zeta,t,x`, 2001 equally spaced analytic displacement samples
  over [0,20] for each damping ratio, matching the original plotting grid.
- `methods.csv`: `method,label`, the original solver display labels in order.
- `environment.csv`: `component,version`, Python, NumPy, and Matplotlib versions
  captured on every export.

`figures/make_figures.py` reads only these CSV inputs; it never calls the model,
solvers, or analysis. CSV floating-point values retain Python's round-trip
precision. Timing measurements vary across runs and machines, so fig5 changes
when experiments are rerun; figure generation from saved CSVs is repeatable.

## Export environment

Python 3.12.3; NumPy 1.26.4; Matplotlib 3.6.3.
The authoritative versions for the latest export are in `environment.csv`.
