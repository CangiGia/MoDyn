# Validation

Cross-checks of MoDyn against analytical solutions, PMD and ADAMS.

- `mass-spring-damper/`, `double compound pendulum/`, `four-bar linkage/` —
  the planar models already validated in PMD, re-simulated in MoDyn and
  compared with PMD and ADAMS (each folder holds the ADAMS `.cmd` model
  definitions and the exported results).
- `analytical/` — spatial benchmarks with an independent reference solution:
  torque-free asymmetric rigid body (Euler's equations) and heavy symmetric top
  (Lagrange top), including the step-size convergence of the solver.
- `benchmarks/` — reserved for standard multibody benchmarks.
- `plot_style.py` — shared Matplotlib style used by all comparison plots.

## Reproducing the comparisons

```powershell
python run_modyn.py fbl|msd|dcp          # simulate in MoDyn, write <model>/MoDyn/ (PMD text format, SI units)
python run_pmd.py dcp                    # PMD reference where no PMD export is provided
python compare_3way.py fbl|msd|dcp       # figures <PREFIX>_3way_*.pdf and metrics_<PREFIX>.csv
python analytical/spatial_benchmarks.py --convergence
```

The models are defined in `../examples`. Notes on the data:

- ADAMS reports the reaction on the body of marker *I*, in global axes; MoDyn
  uses the same convention (`model.reaction(result, joint, side="i")`).
- The double compound pendulum is chaotic: ADAMS/PMD/MoDyn trajectories separate
  after about 5 s, so the three-way comparison is restricted to that window and
  the whole 35 s run is checked through the conservation of energy instead.
- ADAMS results and PMD joint reactions are not provided for the double
  compound pendulum: it is compared with PMD (kinematics) only.
- PMD accelerations in the provided exports contain numerical spikes; they are
  despiked before plotting but remain noisy, which dominates the PMD errors on
  that channel.
