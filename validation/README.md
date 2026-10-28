# Validation

Cross-checks of MoDyn against analytical solutions, PMD and ADAMS.

- `mass-spring-damper/`, `double compound pendulum/`, `four-bar linkage/` —
  the planar models already validated in PMD, re-simulated in MoDyn and
  compared with PMD and ADAMS (each folder holds the ADAMS `.cmd` model
  definitions and the exported results).
- `analytical/` — closed-form sanity checks (e.g. spherical pendulum period,
  rigid-body free precession, energy conservation).
- `benchmarks/` — reproduction of standard multibody benchmarks.
- `plot_style.py` — shared Matplotlib style used by all comparison plots.
