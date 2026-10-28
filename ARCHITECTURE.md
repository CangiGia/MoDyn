# MoDyn — Architecture & Design Decisions

> 3D rigid multibody simulation in Python, spatial evolution of
> [PMD](https://github.com/CangiGia/PMD) (planar). Work in progress.

This document fixes the foundational design choices. Deviations require an
explicit update here.

---

## 1. Scope

Implemented:
- 3D rigid bodies with absolute coordinates.
- Lower-pair joints (Spherical, Revolute, Prismatic, Cylindrical, Planar,
  Universal), primitive constraints (coincidence, dot-1, dot-2) and
  rotational / prismatic drivers.
- Applied forces and torques, gravity, TSDA, user-defined forces (numeric
  solver only).
- Hybrid assembly: BFS spanning tree, placement from joint coordinates `q0`,
  minimum-norm Newton loop closure.
- Automatic detection of redundant constraint rows (planar mechanisms in 3D).
- Dynamic analysis with two solvers (Radau IIA, generalized-α), static
  equilibrium, joint reactions.
- Validation against PMD, ADAMS and analytical references (`validation/`).

Architectural placeholders (modules reserved, not implemented): contact /
collision (`contact/`), flexible bodies (`model/flex_body.py`), sensors,
actuators, control loops, GUI (`gui/`), exporters and serialization (`io/`).

Out of scope: real-time, GPU, distributed solvers.

---

## 2. Reference textbook

**Primary:** A. A. Shabana — *Computational Dynamics* (3rd ed.) and *Dynamics
of Multibody Systems* (4th ed.). Notation and equation numbers cited in
docstrings refer to Shabana unless stated otherwise.

---

## 3. Orientation and velocity variables

Orientation: **Euler parameters** (unit quaternions), `p = [e0, e1, e2, e3]ᵀ`,
`pᵀp = 1`. No singularities. Euler/Cardan angles exist only as input/output
utilities in `math/rotations.py`, never as state variables.

- `A(p) = (2e0² − 1) I + 2 (e eᵀ + e0 ẽ)` maps body to global axes.
- `ω' = 2 G(p) ṗ` (body frame), `ω = 2 E(p) ṗ` (global frame),
  `ṗ = ½ G(p)ᵀ ω'` on the unit sphere.

Positions per body: `q_i = [R_iᵀ, p_iᵀ]ᵀ ∈ ℝ⁷`.
Velocity variables (quasi-velocities): `v_i = [Ṙ_iᵀ, ω'_iᵀ]ᵀ ∈ ℝ⁶`, with
`q̇ = T(q) v`, `T_i = blockdiag(I₃, ½ G(p_i)ᵀ)`.

---

## 4. Equations of motion

Same structure as PMD (see the thesis chapters on the hybrid formulation):

```
q̇ = T(q) v + P(q) μ
M v̇ = h(q, v, t) − Dᵀ λ
0   = Φ(q, t)                 (joint and driver constraints)
0   = p_kᵀ p_k − 1             (Euler-parameter normalization, one per body)
```

- `M = diag(m I₃, J̄)` is **constant and block diagonal** (centroidal body
  coordinates, inertia in the body frame).
- `h` collects applied forces (`Tᵀ Q_e`) and the gyroscopic term
  `−ω' × J̄ ω'`.
- `D = Φ_q T`; `Φ_q`, `Φ_t` and the acceleration terms come from **automatic
  differentiation** (CasADi) of the symbolic constraints.
- `μ` multiplies the normalization; it vanishes on the exact solution and keeps
  the Euler parameters on the unit sphere without projection.

Redundant rows (planar mechanisms in 3D) are detected on the assembled
configuration by rank-revealing QR with column pivoting of `Dᵀ` and removed from
the DAE (`dynamics/redundancy.py`). Multipliers on all rows are recovered a
posteriori as the minimum-norm solution of `Dᵀ λ = h − M v̇`.

No Baumgarte stabilization: position-level constraints are integrated directly.

---

## 5. Solvers

Selected by `SpatialMultibodyModel.solve(method=...)`; both return a
`SimulationResult` (positions, quasi-velocities, quasi-accelerations,
multipliers on all rows).

| method              | formulation                                  | order | derivatives          |
|---------------------|----------------------------------------------|-------|----------------------|
| `radau` (default)   | quasi-velocities, constant `M`, DAE above    | 5     | automatic (CasADi)   |
| `generalized_alpha` | `q̇ = [Ṙ, ṗ]`, `M = diag(m I, 4 GᵀJ̄G)`       | 2     | hand-written Jacobians |

The generalized-α path keeps the original Euler-parameter form (normalization
as an explicit constraint with its own multiplier) and serves as an independent
check and as a base for future comparisons.

---

## 6. Package layout

`src/modyn/` — *src-layout*, installable via `pip install -e .`.

| package         | responsibility                                                   |
|-----------------|------------------------------------------------------------------|
| `core/`         | `Base` counter, `UnitSystem`, logging, generic utilities         |
| `math/`         | vectors, rotation matrices, quaternions, SE(3) transforms        |
| `model/`        | `Ground`, `RigidBody`, `Marker`, (future) shapes, flex bodies    |
| `constraints/`  | joints, primitives, drivers, `Function`                          |
| `forces/`       | gravity, springs, point force, torque, user forces               |
| `dynamics/`     | EoM assembly (Euler-parameter and quasi-velocity forms), redundancy |
| `symbolic/`     | CasADi expressions of constraints, forces and the DAE            |
| `builder/`      | BFS topology and hybrid assembly                                 |
| `solver/`       | model facade, Radau IIA, generalized-α, statics, reactions       |
| `io/`           | (placeholder) serialization, exporters                           |
| `contact/`, `sensors/`, `actuators/`, `control/` | (placeholders)                 |

Cross-package import rule (enforced by review):

```
math, core  →  (no internal deps)
model       → core, math
constraints, forces → core, math, model   (symbolic helpers imported lazily)
dynamics    → core, math, model, constraints, forces
symbolic    → core, math, model, dynamics
builder     → core, math, model, constraints
solver      → everything except gui / io
gui         → solver, io                  (one-way)
```

`solver.model.SpatialMultibodyModel` is the public facade.

Joints, drivers and forces implement `phi_sym` / `wrench_sym` to be usable by
the Radau solver; elements without a symbolic form (e.g. `UserForce`) work only
with the generalized-α solver. The joint coordinates `q0` of a tree joint give
the pose of the child marker frame relative to the parent marker frame (PMD
convention), independently of the marker order.

---

## 7. Testing and validation

- `tests/unit/` — one file per module; symbolic expressions are cross-checked
  against the hand-written Jacobians on random states.
- `tests/integration/` — redundant four-bar, solver agreement, spatial
  benchmarks with reference solutions (free asymmetric body, Lagrange top).
- `validation/` — three-way comparison with PMD and ADAMS (`run_modyn.py`,
  `compare_3way.py`), see its README. Plots use `validation/plot_style.py`.
- Markers: `unit`, `integration`, `regression`, `slow`.

Every dynamic check includes the quaternion-norm drift and the constraint
violation.

---

## 8. Coding conventions

- Python ≥ 3.10, type hints on public APIs, NumPy arrays at runtime.
- Docstrings: NumPy style; math notation follows Shabana.
- Logging via `logging.getLogger(__name__)`; no `print` in the library.
- Public symbols re-exported from each subpackage `__init__.py`.
- CasADi is a dependency of the Radau solver and the symbolic layer; the
  `*_sym` methods import it lazily.
