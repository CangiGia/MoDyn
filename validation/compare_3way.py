"""Three-way comparison ADAMS / PMD / MoDyn of a validation model.

Usage::

    python compare_3way.py fbl|msd|dcp [--source MoDyn] [--no-show]

Loads the exported results of the three sources (``run_modyn.py`` produces the
MoDyn ones), writes the comparison figures in the style of the thesis
(``plot_style.py``) and a table of error metrics in the model folder:

- ``<PREFIX>_3way_*.pdf``: displacements, velocities, accelerations, joint
  reactions (or scalar channels);
- ``metrics_<PREFIX>.csv``: RMS and maximum error, as % of the peak of the
  reference signal, for MoDyn-ADAMS, PMD-ADAMS and MoDyn-PMD.

PMD results are despiked (isolated numerical spikes of the collocation output)
before plotting and before computing the metrics, as in the original
comparison scripts; MoDyn and ADAMS are used as they are. Sign conventions of
the reactions (force on body *i* or *j*) are detected against ADAMS.
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd
from _common import (
    COLORS,
    QUANTITIES,
    SPECS,
    best_sign,
    despike,
    error_metrics,
    load_force,
    load_xy,
)

SOURCES = ("ADAMS", "PMD", "MoDyn")
STYLE = {
    "ADAMS": {"color": COLORS["ADAMS"], "lw": 2.2, "ls": "-"},
    "PMD": {"color": COLORS["PMD"], "lw": 1.2, "ls": "--"},
    "MoDyn": {"color": COLORS["MoDyn"], "lw": 1.5, "ls": ":"},
}
NEGLIGIBLE = 1e-6  # components below this fraction of the peak are numerical zeros


def _load_all(spec, modyn_folder: str) -> dict:
    """Load every available channel of the three sources in SI units."""
    data: dict = {s: {"kin": {}, "force": {}} for s in SOURCES}
    for source in SOURCES:
        folder = modyn_folder if source == "MoDyn" else None
        for quantity in QUANTITIES:
            for body in spec.bodies:
                xy = load_xy(spec, source, quantity, body.key, folder)
                if xy is None:
                    continue
                t, x, y = xy
                if source == "PMD":
                    x, y = despike(x), despike(y)
                data[source]["kin"][(quantity, body.key)] = (t, np.column_stack((x, y)))
        for reaction in spec.reactions:
            fz = load_force(spec, source, reaction.stem, 2, folder)
            if fz is not None:
                t, f = fz
                if source == "PMD":
                    f = np.column_stack((despike(f[:, 0]), despike(f[:, 1])))
                data[source]["force"][reaction.stem] = (t, f)
        for _, stem in spec.scalars:
            fz = load_force(spec, source, stem, 1, folder)
            if fz is not None:
                data[source]["force"][stem] = fz
    return data


def _sign_conventions(spec, data) -> dict:
    """Sign that aligns each force channel of PMD/MoDyn to ADAMS (to MoDyn when ADAMS is absent)."""
    signs = {}
    for reaction in spec.reactions:
        for source in ("PMD", "MoDyn"):
            other = data[source]["force"].get(reaction.stem)
            ref = data["ADAMS"]["force"].get(reaction.stem)
            if ref is None and source == "PMD":
                ref = data["MoDyn"]["force"].get(reaction.stem)
            if ref is None or other is None or other is ref:
                continue
            for c in (0, 1):
                signs[(source, reaction.stem, c)] = best_sign(ref[0], ref[1][:, c], other[0], other[1][:, c])
    for _, stem in spec.scalars:
        signs[("ADAMS", stem, 0)] = spec.adams_scalar_sign.get(stem, 1.0)
    return signs


def _legend(plt, sources):
    return [plt.Line2D([0], [0], label=s, **STYLE[s]) for s in sources]


def _plot_kinematics(spec, data, plt, sources):
    unit, size = spec.plot_length
    suffix = {"coordinates": unit, "velocities": f"{unit}/s", "accelerations": f"{unit}/s$^2$"}
    legend = _legend(plt, sources)

    def draw(ax, quantity, body, comp):
        series = {s: data[s]["kin"].get((quantity, body.key)) for s in sources}
        peak = max(np.max(np.abs(v[1])) for v in series.values() if v is not None)
        for s, v in series.items():
            if v is None:
                continue
            sig = v[1][:, comp]
            if np.max(np.abs(sig)) < NEGLIGIBLE * peak:
                sig = np.zeros_like(sig)
            ax.plot(v[0], sig / size, **STYLE[s])
        label = QUANTITIES[quantity][0]
        ax.set_ylabel(f"{label} [{suffix[quantity]}]")
        ax.set_title(f"{body.label} - ${'xy'[comp]}_{{cm}}$")

    figs = {}
    if len(spec.bodies) == 1:
        fig, axes = plt.subplots(3, 2, figsize=(4.8, 6.0), constrained_layout=True)
        for r, quantity in enumerate(QUANTITIES):
            for c in range(2):
                draw(axes[r, c], quantity, spec.bodies[0], c)
        for ax in axes[-1]:
            ax.set_xlabel("Time [s]")
        figs["kinematics"] = fig
    else:
        n = len(spec.bodies)
        for quantity in QUANTITIES:
            fig, axes = plt.subplots(n, 2, figsize=(4.8, 2.0 * n), constrained_layout=True)
            for r, body in enumerate(spec.bodies):
                for c in range(2):
                    draw(axes[r, c], quantity, body, c)
            for ax in axes[-1]:
                ax.set_xlabel("Time [s]")
            figs[quantity] = fig
    for fig in figs.values():
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=len(legend), frameon=False)
    return figs


def _plot_forces(spec, data, signs, plt, sources):
    figs = {}
    legend = _legend(plt, sources)
    if spec.reactions:
        n = len(spec.reactions)
        fig, axes = plt.subplots(n, 2, figsize=(5.2, 1.85 * n), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for r, reaction in enumerate(spec.reactions):
            for c in range(2):
                ax = axes[r, c]
                for s in sources:
                    f = data[s]["force"].get(reaction.stem)
                    if f is not None:
                        ax.plot(f[0], signs.get((s, reaction.stem, c), 1.0) * f[1][:, c], **STYLE[s])
                ax.set_ylabel("Force [N]")
                ax.set_title(f"{reaction.label} - $F_{'xy'[c]}$")
        for ax in axes[-1]:
            ax.set_xlabel("Time [s]")
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(legend), frameon=False)
        figs["reactions"] = fig
    if spec.scalars:
        fig, axes = plt.subplots(len(spec.scalars), 1, figsize=(4.8, 2.4 * len(spec.scalars)), constrained_layout=True)
        for ax, (label, stem) in zip(np.atleast_1d(axes), spec.scalars, strict=True):
            for s in sources:
                f = data[s]["force"].get(stem)
                if f is not None:
                    ax.plot(f[0], signs.get((s, stem, 0), 1.0) * f[1][:, 0], **STYLE[s])
            ax.set_ylabel(f"{label} [N]")
            ax.set_xlabel("Time [s]")
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(legend), frameon=False)
        figs["scalars"] = fig
    return figs


def _metrics(spec, data, signs) -> pd.DataFrame:
    rows = []

    def add(channel, series):
        for other in ("MoDyn", "PMD"):
            if "ADAMS" in series and other in series:
                rows.append({"channel": channel, "pair": f"{other}-ADAMS", **error_metrics(*series["ADAMS"], *series[other])})
        if "PMD" in series and "MoDyn" in series:
            rows.append({"channel": channel, "pair": "MoDyn-PMD", **error_metrics(*series["PMD"], *series["MoDyn"])})

    for quantity in QUANTITIES:
        for body in spec.bodies:
            have = {s: data[s]["kin"].get((quantity, body.key)) for s in SOURCES}
            have = {s: v for s, v in have.items() if v is not None}
            ref = "ADAMS" if "ADAMS" in have else "PMD"
            if len(have) < 2 or ref not in have:
                continue
            peak = max(np.max(np.abs(v[1])) for v in have.values())
            for c in range(2):
                if np.max(np.abs(have[ref][1][:, c])) < NEGLIGIBLE * peak:
                    continue
                add(f"{quantity}/{body.label}/{'xy'[c]}", {s: (v[0], v[1][:, c]) for s, v in have.items()})
    for reaction in spec.reactions:
        for c in range(2):
            series = {
                s: (f[0], signs.get((s, reaction.stem, c), 1.0) * f[1][:, c])
                for s in SOURCES
                if (f := data[s]["force"].get(reaction.stem)) is not None
            }
            add(f"reaction/{reaction.label}/F{'xy'[c]}", series)
    for label, stem in spec.scalars:
        series = {
            s: (f[0], signs.get((s, stem, 0), 1.0) * f[1][:, 0])
            for s in SOURCES
            if (f := data[s]["force"].get(stem)) is not None
        }
        add(f"scalar/{label}", series)
    return pd.DataFrame(rows)


def compare(key: str, modyn_folder: str = "MoDyn", show: bool = True):
    """Plot and score the three-way comparison of the model ``key``."""
    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from plot_style import setup_thesis_plots

    spec = SPECS[key]
    setup_thesis_plots()
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    })

    data = _load_all(spec, modyn_folder)
    sources = [s for s in SOURCES if data[s]["kin"] or data[s]["force"]]
    missing = [s for s in SOURCES if s not in sources]
    if missing:
        print(f"warning: no data found for {missing}")
    if "ADAMS" not in sources:
        print("warning: ADAMS results not available, the metrics are computed against PMD only")
    signs = _sign_conventions(spec, data)

    figs = {**_plot_kinematics(spec, data, plt, sources), **_plot_forces(spec, data, signs, plt, sources)}
    for name, fig in figs.items():
        fig.savefig(spec.path / f"{spec.prefix}_3way_{name}.pdf")

    table = _metrics(spec, data, signs)
    table.to_csv(spec.path / f"metrics_{spec.prefix}.csv", index=False, float_format="%.6g")
    if not table.empty:
        table["quantity"] = table["channel"].str.split("/").str[0]
        summary = table.groupby(["quantity", "pair"]).agg(
            rms_mean=("rms_pct", "mean"), rms_max=("rms_pct", "max"), peak_max=("max_pct", "max")
        )
        print(f"\n{spec.name}: error metrics (% of the peak of the reference signal)")
        print(summary.round(4).to_string())
    flipped = sorted({(s, st, c) for (s, st, c), v in signs.items() if v < 0 and s != "ADAMS"})
    if flipped:
        print(f"sign convention flipped to match ADAMS: {sorted({(s, st) for s, st, _ in flipped})}")
    if show:
        plt.show()
    return table, figs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(SPECS))
    parser.add_argument("--source", default="MoDyn", help="folder with the MoDyn results")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    compare(args.model, args.source, show=not args.no_show)
