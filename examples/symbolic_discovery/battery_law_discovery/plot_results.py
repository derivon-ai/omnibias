# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Plot helpers for the battery law-discovery demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .features import FeatureBundle, cell_indices
except ImportError:  # pragma: no cover
    from features import FeatureBundle, cell_indices


def plot_capacity_rollout(
    out: Path,
    test: FeatureBundle,
    omnibias_pred: np.ndarray,
    baseline_pred: np.ndarray | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib not installed; skipping capacity_rollout.png")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    cells = list(cell_indices(test.cell_id))
    n_show = min(4, len(cells))
    fig, axes = plt.subplots(n_show, 1, figsize=(8, 2.4 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    idx_by_cell = cell_indices(test.cell_id)
    for ax, cell in zip(axes, cells[:n_show], strict=False):
        idx = idx_by_cell[cell]
        ax.plot(test.cycle_norm[idx], test.y[idx], "k-", label="observed")
        ax.plot(test.cycle_norm[idx], omnibias_pred[idx], "C0--", label="omnibias law rollout")
        if baseline_pred is not None:
            ax.plot(test.cycle_norm[idx], baseline_pred[idx], "C1:", label="ridge")
        ax.set_ylabel("capacity / q0")
        ax.set_title(cell)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("normalized cycle index")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_baseline_comparison(out: Path, metrics: dict[str, dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib not installed; skipping baseline_comparison.png")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    names = [
        name
        for name, values in metrics.items()
        if "rmse_capacity" in values and "eol_mae_cycles" in values
    ]
    rmse = [metrics[name]["rmse_capacity"] for name in names]
    eol = [metrics[name]["eol_mae_cycles"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, rmse)
    axes[0].set_ylabel("capacity RMSE")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(names, eol)
    axes[1].set_ylabel("EOL MAE (cycles)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
