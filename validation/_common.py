"""Shared helpers for the ADAMS / PMD / MoDyn comparisons.

Each validation model is described by a :class:`ModelSpec` (folder, bodies,
joints, units). The text files of the three sources are loaded into SI units
(m, m/s, m/s^2, N) and compared on the time grid of the ADAMS output.

Data layout of a model folder (as provided with the ADAMS exports)::

    <folder>/ADAMS/[coordinates|velocities|accelerations|reactions/]ADAMS_<key>_cm_<quantity>.txt
    <folder>/PMD/...      PMD_<key>_cm_<quantity>.txt
    <folder>/MoDyn/...    MoDyn_<key>_cm_<quantity>.txt   (written by run_modyn.py)
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"

#: Colours of the three sources (ADAMS and PMD as in the thesis plots).
COLORS = {"ADAMS": "#2E7EC1", "PMD": "#D64B35", "MoDyn": "#2E9E5B"}
QUANTITIES = {
    "coordinates": ("Displacement", "m", 1.0),
    "velocities": ("Velocity", "m/s", 1.0),
    "accelerations": ("Acceleration", "m/s$^2$", 1.0),
}


@dataclass(frozen=True)
class Body:
    """A body of the model: plot label, file key and index in the model."""

    label: str
    key: str
    index: int


@dataclass(frozen=True)
class Reaction:
    """A joint reaction: label, file stem and index in ``parts['joints']``."""

    label: str
    stem: str
    joint: int


@dataclass(frozen=True)
class ModelSpec:
    """Description of a validation model."""

    name: str
    folder: str
    prefix: str
    example: str
    bodies: tuple[Body, ...]
    reactions: tuple[Reaction, ...] = ()
    scalars: tuple[tuple[str, str], ...] = ()  # (label, file stem) of scalar channels
    t_final: float = 1.0
    dt: float = 1e-3
    pmd_length_scale: float = 1.0  # PMD files -> metres
    plot_length: tuple[str, float] = ("m", 1.0)  # unit label and size of the unit in metres
    adams_scalar_sign: dict[str, float] = field(default_factory=dict)
    t_compare: float | None = None  # compare only up to this time (chaotic dynamics)

    @property
    def path(self) -> Path:
        return HERE / self.folder


SPECS: dict[str, ModelSpec] = {
    "fbl": ModelSpec(
        name="Four-bar linkage",
        folder="four-bar linkage",
        prefix="FBL",
        example="four_bar_linkage",
        bodies=(
            Body("Crankshaft", "crank", 0),
            Body("Coupler", "coupler", 1),
            Body("Follower", "follower", 2),
        ),
        reactions=(
            Reaction("Ground - Crank", "ground_crank_forces", 0),
            Reaction("Crank - Coupler", "crank_coupler_forces", 1),
            Reaction("Coupler - Follower", "coupler_follower_forces", 2),
            Reaction("Follower - Ground", "follower_ground_forces", 3),
        ),
        t_final=25.0,
        dt=1e-3,
    ),
    "msd": ModelSpec(
        name="Mass-spring-damper",
        folder="mass-spring-damper",
        prefix="MSD",
        example="mass_spring_damper",
        bodies=(Body("Mass", "mass", 0),),
        scalars=(("Spring force", "spring_force"),),
        t_final=15.0,
        dt=5e-3,
        pmd_length_scale=1e-3,
        plot_length=("mm", 1e-3),
        adams_scalar_sign={"spring_force": -1.0},
    ),
    "dcp": ModelSpec(
        name="Double compound pendulum",
        folder="double compound pendulum",
        prefix="DCP",
        example="double_compound_pendulum",
        bodies=(Body("Link 1", "link_1", 0), Body("Link 2", "link_2", 1)),
        reactions=(
            Reaction("Ground - Link 1", "ground_link_1_forces", 0),
            Reaction("Link 1 - Link 2", "link_1_link_2_forces", 1),
        ),
        t_final=35.0,
        dt=1e-3,
        plot_length=("mm", 1e-3),
        t_compare=5.0,  # the double pendulum is chaotic: trajectories separate after ~5 s
    ),
}


def load_example(spec: ModelSpec):
    """Import the example module that builds the MoDyn model of ``spec``."""
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module(spec.example)


# ─── Reading and writing text files ───────────────────────────────────


def _resolve(root: Path, source: str, subdir: str, stem: str, folder: str | None = None) -> Path | None:
    """Locate ``<source>_<stem>.txt`` flat or nested in ``subdir`` under ``root/<folder or source>``."""
    name = f"{source}_{stem}.txt"
    base = root / (folder or source)
    for candidate in (base / name, base / subdir / name):
        if candidate.exists():
            return candidate
    return None


def read_table(path: Path) -> pd.DataFrame:
    """Read an ADAMS/PMD/MoDyn text file (tab separated, optional pre-header lines)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = next(i for i, line in enumerate(lines) if "time" in line.lower())
    return pd.read_csv(path, sep=r"\t+", skiprows=header, quotechar='"', engine="python")


def load_xy(spec: ModelSpec, source: str, quantity: str, key: str, folder: str | None = None):
    """Time, x, y of one body in SI units, or ``None`` if the file is missing."""
    path = _resolve(spec.path, source, quantity, f"{key}_cm_{quantity}", folder)
    if path is None:
        return None
    df = read_table(path)
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.all(np.isfinite(arr[:, :3]), axis=1)]
    scale = {"ADAMS": 1e-3, "PMD": spec.pmd_length_scale, "MoDyn": 1.0}[source]
    t, x, y = arr[:, 0], arr[:, 1] * scale, arr[:, 2] * scale
    keep = t <= (spec.t_compare if spec.t_compare is not None else np.inf) + 1e-9
    return t[keep], x[keep], y[keep]


def load_force(spec: ModelSpec, source: str, stem: str, ncols: int = 2, folder: str | None = None):
    """Time and the first ``ncols`` force columns, or ``None`` if the file is missing."""
    path = _resolve(spec.path, source, "reactions", stem, folder)
    if path is None:
        return None
    df = read_table(path)
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.all(np.isfinite(arr[:, : 1 + ncols]), axis=1)]
    keep = arr[:, 0] <= (spec.t_compare if spec.t_compare is not None else np.inf) + 1e-9
    arr = arr[keep]
    return arr[:, 0], arr[:, 1 : 1 + ncols]


def write_table(path: Path, times: np.ndarray, columns: list[str], data: np.ndarray) -> None:
    """Write a PMD-style tab-separated table (header ``time_s`` and column names)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(["time_s", *columns])
    block = np.column_stack((times, data))
    np.savetxt(path, block, delimiter="\t", header=header, comments="", fmt="%.15g")


# ─── Signal processing and metrics ────────────────────────────────────


def despike(y: np.ndarray, window: int = 31, n_sigma: float = 6.0) -> np.ndarray:
    """Replace isolated outliers (rolling median + MAD) by linear interpolation."""
    y = np.asarray(y, dtype=float)
    if y.size < window:
        return y.copy()
    med = median_filter(y, size=window, mode="nearest")
    dev = np.abs(y - med)
    mad = median_filter(dev, size=window, mode="nearest")
    spike = dev > n_sigma * 1.4826 * np.maximum(mad, 1e-300)
    spike &= mad > 0.0
    if not spike.any():
        return y.copy()
    idx = np.arange(y.size)
    out = y.copy()
    out[spike] = np.interp(idx[spike], idx[~spike], y[~spike])
    return out


def on_grid(t_ref: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``y(t)`` on the part of ``t_ref`` covered by ``t``; returns (t_common, y)."""
    mask = (t_ref >= t[0] - 1e-12) & (t_ref <= t[-1] + 1e-12)
    return t_ref[mask], np.interp(t_ref[mask], t, y)


def error_metrics(t_ref, y_ref, t, y) -> dict[str, float]:
    """RMS and maximum error of ``y`` against ``y_ref``, as % of the peak of ``y_ref``."""
    tc, yi = on_grid(t_ref, t, y)
    ref = np.interp(tc, t_ref, y_ref)
    peak = float(np.max(np.abs(ref))) or 1.0
    err = yi - ref
    return {
        "rms_pct": 100.0 * float(np.sqrt(np.mean(err**2))) / peak,
        "max_pct": 100.0 * float(np.max(np.abs(err))) / peak,
    }


def best_sign(t_ref, y_ref, t, y) -> float:
    """+1 or -1 depending on the sign convention that best matches ``y`` to ``y_ref``."""
    tc, yi = on_grid(t_ref, t, y)
    ref = np.interp(tc, t_ref, y_ref)
    return 1.0 if np.sum(np.abs(yi - ref)) <= np.sum(np.abs(-yi - ref)) else -1.0
