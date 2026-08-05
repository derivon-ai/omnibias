# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Figures for the min-square-cover study (lazy matplotlib, Agg backend).

Every function turns a list of :class:`~examples.min_square_cover.train.CoverResult` (or one
solved instance) into a committed ``figures/*.png``:

* :func:`plot_cover_counts` -- mean optimality ratio (``n_final / lower_bound``) per arm, with
  the greedy baseline drawn as a reference line (which optimiser gives the leanest cover);
* :func:`plot_energy_trajectory` -- the soft-coverage energy per step for each arm on one
  instance, with the ``beta`` annealing schedule on a twin axis;
* :func:`plot_certified_gap` -- the ``ceil(lower_bound) <= optimum <= K`` sandwich per shape;
* :func:`plot_cover_overlay` -- the binary image with the discrete squares drawn on top.

``matplotlib`` is imported lazily inside each function (Agg backend, no display needed), so the
rest of the example -- and its tests -- never depend on it.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from examples.min_square_cover.data import Instance, make_instance
from examples.min_square_cover.experiment import summarize
from examples.min_square_cover.train import CoverResult


def _pyplot():
    """Return ``matplotlib.pyplot`` on the non-interactive Agg backend (lazy import)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_cover_counts(results: list[CoverResult], path: str | Path) -> Path:
    """Bar chart of mean optimality ratio (``n_final / lower_bound``) per arm (lower is better).

    The greedy baseline's mean ratio is drawn as a dashed reference line, and the ideal ratio
    ``1.0`` (a cover that meets the certified lower bound) as a solid line.
    """
    plt = _pyplot()
    by_arm: dict[str, list[float]] = {}
    greedy_ratios: list[float] = []
    for r in results:
        by_arm.setdefault(r.arm, []).append(r.ratio_final)
        greedy_ratios.append(r.ratio_greedy)
    arms = sorted(by_arm, key=lambda a: statistics.fmean(by_arm[a]))
    means = [statistics.fmean(by_arm[a]) for a in arms]
    stds = [statistics.pstdev(by_arm[a]) if len(by_arm[a]) > 1 else 0.0 for a in arms]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(arms) + 3.0), 4.5))
    ax.bar(range(len(arms)), means, yerr=stds, capsize=3, color="#4C72B0", alpha=0.85)
    if greedy_ratios:
        g = statistics.fmean(greedy_ratios)
        ax.axhline(g, ls="--", color="#C44E52", label=f"greedy ({g:.2f})")
    ax.axhline(1.0, ls="-", color="#555555", lw=1.0, label="certified LB (1.00)")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=35, ha="right")
    ax.set_ylabel("n_final / lower_bound  (lower is better)")
    ax.set_title("Cover count vs optimiser (mean +/- std over shapes x seeds)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_energy_trajectory(
    results: list[CoverResult], path: str | Path, *, shape: str | None = None, seed: int | None = None
) -> Path:
    """Soft-coverage energy per step for each arm on one ``(shape, seed)``, with beta annealing.

    Picks the first available ``(shape, seed)`` when either is unset. Arms whose history is empty
    (e.g. a pure closed-form step that logged only endpoints) are skipped.
    """
    plt = _pyplot()
    with_hist = [r for r in results if r.history]
    if not with_hist:
        raise ValueError("no results carry a per-step history to plot")
    shape = shape if shape is not None else with_hist[0].shape
    seed = seed if seed is not None else with_hist[0].seed
    subset = [r for r in with_hist if r.shape == shape and r.seed == seed]
    if not subset:
        raise ValueError(f"no history for shape={shape!r} seed={seed}")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for r in sorted(subset, key=lambda r: r.arm):
        steps = [h["step"] for h in r.history]
        energy = [h["energy"] for h in r.history]
        ax.plot(steps, energy, label=r.arm, lw=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("soft-coverage energy")
    ax.set_title(f"Energy trajectory under annealing (shape={shape}, seed={seed})")
    ax.legend(fontsize=8, ncol=2)

    beta_ax = ax.twinx()
    betas = [h["beta"] for h in subset[0].history]
    steps0 = [h["step"] for h in subset[0].history]
    beta_ax.plot(steps0, betas, color="#999999", ls=":", lw=1.4)
    beta_ax.set_ylabel("beta (sharpness anneal)", color="#777777")
    beta_ax.tick_params(axis="y", labelcolor="#777777")
    fig.tight_layout()
    return _save(fig, path)


def plot_certified_gap(results: list[CoverResult], path: str | Path) -> Path:
    """Per-shape ``ceil(lower_bound) <= best K <= greedy`` sandwich as grouped bars."""
    plt = _pyplot()
    summary = summarize(results)
    shapes = sorted({s for s, _ in summary})
    best = []
    greedy = []
    lb = []
    for shape in shapes:
        finals = [summary[(s, a)]["final_mean"] for (s, a) in summary if s == shape]
        best.append(min(finals) if finals else float("nan"))
        any_key = next((k for k in summary if k[0] == shape), None)
        greedy.append(summary[any_key]["greedy_mean"] if any_key else float("nan"))
        lb.append(summary[any_key]["lower_bound"] if any_key else float("nan"))

    x = range(len(shapes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(shapes) + 3.0), 4.5))
    ax.bar([i - w for i in x], lb, width=w, label="certified LB", color="#55A868")
    ax.bar(list(x), best, width=w, label="best solver K", color="#4C72B0")
    ax.bar([i + w for i in x], greedy, width=w, label="greedy", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(shapes, rotation=20, ha="right")
    ax.set_ylabel("square count")
    ax.set_title("Certified sandwich: lower bound <= best solver <= greedy")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_anneal_ablation(results: list[CoverResult], path: str | Path) -> Path:
    """H3 ablation: mean final square count by ``beta_schedule``, grouped per arm.

    Bars for ``anneal`` vs each ``fixed@<beta>`` sitting within seed-noise error bars is the
    visual statement of the null result -- the round-then-complete-then-prune finaliser makes the
    final count robust to the sharpness schedule.
    """
    plt = _pyplot()
    by: dict[tuple[str, str], list[int]] = {}
    for r in results:
        by.setdefault((r.arm, r.beta_schedule), []).append(r.n_final)
    arms = sorted({a for a, _ in by})
    scheds = sorted({s for _, s in by})  # "anneal" sorts before "fixed@*"
    x = range(len(scheds))
    w = 0.8 / max(len(arms), 1)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(scheds) + 3.0), 4.5))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for j, arm in enumerate(arms):
        means = [statistics.fmean(by[(arm, s)]) if (arm, s) in by else 0.0 for s in scheds]
        stds = [
            statistics.pstdev(by[(arm, s)]) if by.get((arm, s), []) and len(by[(arm, s)]) > 1 else 0.0
            for s in scheds
        ]
        offs = [i + (j - (len(arms) - 1) / 2) * w for i in x]
        ax.bar(offs, means, width=w, yerr=stds, capsize=3, label=arm, color=palette[j % len(palette)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(scheds)
    ax.set_ylabel("mean final square count (lower is better)")
    ax.set_title("H3 ablation: annealed vs fixed beta (bars within noise => no benefit)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_cover_overlay(
    instance: Instance, squares: list[tuple[int, int]], side: int, path: str | Path
) -> Path:
    """Draw the binary image with the placed ``side x side`` squares overlaid."""
    plt = _pyplot()
    from matplotlib.patches import Rectangle

    img = instance.image.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(img, cmap="Greys", origin="upper", interpolation="nearest")
    for r, c in squares:
        # imshow puts column on x and row on y; a top-left (r, c) block spans [c-0.5, c+side-0.5].
        ax.add_patch(
            Rectangle(
                (c - 0.5, r - 0.5), side, side,
                fill=False, edgecolor="#DD8452", lw=2.0,
            )
        )
    ax.set_title(f"{instance.name}: {len(squares)} squares (side={side})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def make_all_figures(
    results: list[CoverResult], out_dir: str | Path, *, size: int = 24
) -> list[Path]:
    """Write the full figure set to ``out_dir`` and return the written paths.

    The cover overlay is drawn for the best (fewest-square) run of the first synthetic shape;
    its instance is rebuilt deterministically from ``(shape, side, seed)`` via
    :func:`~examples.min_square_cover.data.make_instance`.
    """
    out = Path(out_dir)
    paths = [
        plot_cover_counts(results, out / "cover_counts.png"),
        plot_certified_gap(results, out / "certified_gap.png"),
    ]
    if any(r.history for r in results):
        paths.append(plot_energy_trajectory(results, out / "energy_trajectory.png"))
    from examples.min_square_cover.data import SHAPES

    synthetic = [r for r in results if r.shape in SHAPES]
    if synthetic:
        best = min(synthetic, key=lambda r: r.n_final)
        instance = make_instance(best.shape, size=size, side=best.side, seed=best.seed)
        paths.append(
            plot_cover_overlay(instance, best.squares, best.side, out / "cover_overlay.png")
        )
    return paths


__all__ = [
    "make_all_figures",
    "plot_anneal_ablation",
    "plot_certified_gap",
    "plot_cover_counts",
    "plot_cover_overlay",
    "plot_energy_trajectory",
]
