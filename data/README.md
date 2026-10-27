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

**Phase metric:** `analysis.phase_error()` returns the fitted slope of
numerical peak-time drift versus peak index (seconds per period). The saved
`phase_slope` values have been refreshed using that implementation. With only
three positive peaks over [0,20], small drifts can be dominated by peak-fit noise.

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
Verlet-type errors increase smoothly from 0.006453 to 0.015991; RK4 errors
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

## stability_boundaries.csv

Regenerate separately with `python3 experiments/stability_theory.py`. For each
of the three methods and zeta in {0, 0.1, 0.5, 1, 2}, `stability.py` predicts
the stability boundary h* (largest step with spectral radius of the one-step
amplification matrix <= 1) and the runner then checks the prediction against
direct solver runs at 0.95*h* and 1.05*h* over t in [0, 600].

| Column | Meaning |
| --- | --- |
| method | Solver key: euler, verlet, or rk4. |
| zeta | Dimensionless damping ratio. |
| h_boundary | Predicted boundary; empty when the method is unstable for every h (Euler at zeta=0). |
| euler_boundary_2zeta | Closed form 2*zeta for Euler with 0 < zeta <= 1; empty otherwise. |
| bounded_5pct_below | True when the solver run at 0.95*h* stays within 10x the exact solution's peak. |
| bounded_5pct_above | False when the run at 1.05*h* outgrows that ceiling (i.e. the predicted boundary is real). |

Notable values: Euler matches 2*zeta exactly through the underdamped range;
Verlet-type and RK4 boundaries are not monotone in zeta (Verlet-type dips to
about 1.73 near zeta=0.3 before recovering to 2.0 at zeta=0.5; RK4 peaks
around 2.95 at zeta=0.1 and falls to 2.62 at zeta=0.5).

## dispersion_check.csv

Written by `python3 experiments/stability_theory.py` together with
stability_boundaries.csv. For zeta=0.1 and the five matrix step sizes it
compares the per-period peak drift predicted from the amplification matrix
(`stability.period_drift`: numerical period 2*pi/w_num from the eigenvalue
angle minus exact period 2*pi/wd) against the peak-fit drift measured by
`analysis.phase_error` on the same [0, 20] grid used by metrics.csv.

| Column | Meaning |
| --- | --- |
| method | Solver key: euler, verlet, or rk4. |
| zeta | Damping ratio (fixed 0.1). |
| h | Integration time step. |
| predicted_drift_s_per_period | Theory prediction, seconds per period. |
| measured_drift_s_per_period | Peak-fit measurement, seconds per period. |

Prediction and measurement agree to ~1e-6 for Euler and ~1e-4 for
Verlet-type; RK4's prediction (<= 8e-5) is below the peak-fit noise floor,
so only an upper bound is meaningful there.

Note: the identical Euler phase_slope values at h=0.1 and h=0.2 in
metrics.csv are not stale data. For explicit Euler, arg(1+h*lambda)/h at
h=zeta equals the value at h=2*zeta exactly (tan double-angle identity), so
zeta=0.1 gives the same numerical frequency at h=0.1 and h=0.2. Guarded by
DispersionTests.test_euler_frequency_identity_at_h_equals_zeta.
