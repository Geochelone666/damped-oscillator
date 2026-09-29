# Saved benchmark data

Regenerate with `python3 experiments/run_all.py` from the repository root.
The runner uses `analysis.full_matrix()` unchanged. For the original benchmark, all times and displacements
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

## stability_scan.csv

Regenerate separately with `python3 experiments/stability_scan.py`, then run
`python3 figures/make_figures.py` to render fig7 from this CSV alone.
This scan does not run the other planned experiments or change metrics.csv.

Exactly 21 rows: three methods (`euler`, `verlet`, `rk4`) at zeta=0.1,
h in {0.16, 0.18, 0.19, 0.20, 0.21, 0.22, 0.25}, x(0)=1, v(0)=0.
Samples use a constant-h grid inside [0,40]; when 40/h is not integral,
the last sample is floor(40/h)*h (no shortened final step).

| Column | Meaning |
| --- | --- |
| method | Existing solver key; `verlet` is the velocity-Verlet-type variant. |
| h | Integration time step. |
| zeta_fit_err | abs(zeta_fit-zeta)/zeta using `analysis.envelope_decay_rate`, with parabolically refined positive peaks. |
| E_inf | Maximum absolute displacement error against the analytic solution on the retained grid. |

Euler's amplification modulus is sqrt(1-2*zeta*h+h*h), giving the discrete
decay rate -log(1-2*zeta*h+h*h)/(2*h). Thus its relative decay error varies
continuously through 1 (100%) at h=0.2: the numerical envelope is neutral
there and grows for h>0.2. This metric does **not** jump by orders of magnitude
at the boundary. The measured errors run from 0.799355 to 1.248420 and agree
with the amplification-factor prediction to within 4e-5 absolute error.
Verlet-type errors increase smoothly from 0.012788 to 0.031739; RK4 errors
remain below 8e-5, with small nonmonotonic variations from the peak fit.

## energy_drift.csv and energy_curves.csv

Regenerate separately with `python3 experiments/energy_drift.py`, then run
`python3 figures/make_figures.py` to render fig8 from these two CSVs only.
This experiment uses zeta=0, x(0)=1, v(0)=0, t in [0,100], all three methods,
and h in {0.2, 0.1, 0.05}. The exact conserved energy is
E=(x*x+v*v)/2=E0=0.5. It does not run the zeta=0.5 experiment.

`energy_drift.csv` contains nine rows, one per (method,h):

| Column | Meaning |
| --- | --- |
| method | Solver key: euler, verlet, or rk4. |
| h | Integration time step. |
| max_rel_E_err | max(abs(E-E0)/E0) over every integration step, before subsampling. |
| final_rel_E_err | abs(E(100)-E0)/E0 (unsigned). |

`energy_curves.csv` has columns `method,h,t,E` and 1809 rows: 201 samples
per run, including t=0 and t=100. Samples use rounded, evenly spaced solver
indices without interpolation (spacing can alternate at h=0.2).
Fig8 shows abs(E-E0)/E0 for h=0.1 on a logarithmic y-axis; exact zeros,
including the initial point, are masked rather than assigned an artificial floor.
The legend reports maxima from the full integration grids.

Euler has E_n/E0=(1+h*h)^n: secular growth is exponential over long runs,
with linear approximation (E_n-E0)/E0 approximately t*h only when t*h is
small. At zeta=0 the Verlet-type scheme reduces to velocity Verlet, with
bounded oscillatory relative energy error no larger than h*h/4. RK4 has
small secular energy loss; its per-step energy multiplier is
1-h^6/72+h^8/576. Both have errors orders of magnitude below Euler here.
