# Changelog

## 2026-10-26

### Fixed
- `data/README.md` still quoted the pre-fix Verlet-type stability-scan range (0.012788 to 0.031739); it now matches `data/stability_scan.csv` (0.006453 to 0.015991).

### Added
- `tests/test_documented_numbers.py` recomputes the convergence-order table, stability-scan ranges, Euler deviation from amplification theory, energy-drift and cost examples from the saved CSVs and checks the text in `paper/main.tex` and `data/README.md` contains them. The `.docx` draft is not checked.

## 2026-10-22

### Fixed
- Saved Verlet-type results (`data/*.csv`, `paper_figures/`) were from before the corrected-velocity fix in `solvers.py`; regenerated. Verlet-type fitted orders are now 2.02 / 2.10 / 2.20 (ζ = 0.1 / 1 / 2) and the ζ = 0.1, h = 0.2 phase drift is −0.0097 s/period. README and `paper/main.tex` updated; `paper/damped-oscillator-draft-v01.docx` now matches (Verlet-type orders in the abstract and table, phase drift -0.0097 s/period).
- `x_exact` / `v_exact` returned NaN for ζ ≤ −1 (strong anti-damping).
- The overdamped closed form lost accuracy for very large ζ (cancellation in the slow root).
- Solvers no longer integrate past `t_end` when it is not a multiple of `h`: the grid ends at the last multiple of `h` that is ≤ `t_end`.
- `analysis.phase_error` raises `ValueError` for |ζ| ≥ 1 instead of returning NaN with a warning.
- Step counts that overflow (e.g. `h=1e-300, t_end=1e300`) raise `ValueError`.

### Added
- `requirements.txt` (Python 3.11–3.13); CI runs 3.11, 3.12 and 3.13 and checks that the figures rebuild from the CSVs.
- Tests: step-halving convergence orders, saved CSVs and README claims against the code, anti-damping and large-ζ exact solutions.
