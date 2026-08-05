# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sweep arms x datasets x seeds and summarise the binary-vs-STE comparison.

:func:`run_sweep` trains every requested ``(dataset, arm, seed)`` combination
(sharing the exact same data per seed so only the backward differs), optionally
writing a full JSON log plus a flat CSV. :func:`summarize` / :func:`format_table`
reduce the runs to mean +/- std test accuracy for a quick read on whether the
omnibias surrogates beat the STE baseline.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from examples.binary_vs_ste.arms import ARMS, get_arm
from examples.binary_vs_ste.data import (
    DATASETS,
    get_spec,
    real_datasets,
    synthetic_datasets,
)
from examples.binary_vs_ste.train import RunResult, train_arm


def run_sweep(
    datasets: tuple[str, ...] = DATASETS,
    arms: tuple[str, ...] = ARMS,
    seeds: tuple[int, ...] = (0, 1, 2),
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    data_root: str = "data",
    download: bool = False,
    augment: bool = True,
    synthetic: bool = False,
    schedule: str = "exp",
    xnor: bool = False,
    lr_schedule: str = "constant",
    num_workers: int = 0,
    out_dir: str | None = None,
    log: bool = False,
) -> list[RunResult]:
    """Train all ``(dataset, arm, seed)`` combinations and return the run results."""
    for name in datasets:
        get_spec(name)  # validate early
    results: list[RunResult] = []
    for dataset in datasets:
        real_built: tuple[object, object, object] | None = None
        for seed in seeds:
            if synthetic:
                train_ds, test_ds, spec = synthetic_datasets(dataset, seed=seed)
            else:
                if real_built is None:
                    real_built = real_datasets(
                        dataset, data_root, download=download, augment=augment
                    )
                train_ds, test_ds, spec = real_built  # type: ignore[assignment]
            for arm_name in arms:
                result = train_arm(
                    get_arm(arm_name),
                    train_ds,
                    test_ds,
                    spec,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                    schedule=schedule,
                    seed=seed,
                    num_workers=num_workers,
                    xnor=xnor,
                    lr_schedule=lr_schedule,
                    log=log,
                )
                results.append(result)
    if out_dir is not None:
        write_results(results, out_dir)
    return results


def write_results(results: list[RunResult], out_dir: str) -> tuple[Path, Path]:
    """Write the full JSON log and a flat CSV; return both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    csv_path = out / "results.csv"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    fields = [
        "dataset",
        "arm",
        "seed",
        "epochs",
        "test_acc",
        "best_acc",
        "init_train_loss",
        "final_train_loss",
        "final_beta",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            writer.writerow({k: row[k] for k in fields})
    return json_path, csv_path


def summarize(results: list[RunResult]) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate to best-/final-epoch mean +/- std test accuracy per ``(dataset, arm)``.

    ``best_*`` uses each run's best epoch (the standard "best test accuracy"
    reporting, robust to late-training wobble); ``final_*`` keeps the last epoch for
    reference. ``mean``/``std`` alias the best-epoch numbers (the headline metric).
    """
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        grouped.setdefault((r.dataset, r.arm), []).append(r)
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, runs in grouped.items():
        finals = [r.test_acc for r in runs]
        bests = [r.best_acc for r in runs]
        best_mean = statistics.fmean(bests)
        best_std = statistics.pstdev(bests) if len(bests) > 1 else 0.0
        summary[key] = {
            "best_mean": best_mean,
            "best_std": best_std,
            "final_mean": statistics.fmean(finals),
            "final_std": statistics.pstdev(finals) if len(finals) > 1 else 0.0,
            "mean": best_mean,
            "std": best_std,
            "n": float(len(runs)),
        }
    return summary


def format_table(results: list[RunResult], *, metric: str = "best") -> str:
    """Render a ``dataset x arm`` grid of ``mean +/- std`` test accuracy.

    ``metric="best"`` (default) reports the best-epoch accuracy; ``metric="final"``
    reports the last epoch.
    """
    if metric not in ("best", "final"):
        raise ValueError(f"metric must be 'best' or 'final', got {metric!r}")
    mean_key, std_key = (f"{metric}_mean", f"{metric}_std")
    summary = summarize(results)
    datasets = sorted({d for d, _ in summary})
    arms = [a for a in ARMS if any((d, a) in summary for d in datasets)]
    width = max((len(a) for a in arms), default=4) + 2
    header = "dataset".ljust(15) + "".join(a.rjust(width + 8) for a in arms)
    lines = [header, "-" * len(header)]
    for dataset in datasets:
        cells = []
        for arm in arms:
            stats = summary.get((dataset, arm))
            if stats is None:
                cell = "n/a"
            else:
                cell = f"{stats[mean_key] * 100:5.2f}+/-{stats[std_key] * 100:4.2f}"
            cells.append(cell.rjust(width + 8))
        lines.append(dataset.ljust(15) + "".join(cells))
    lines.append("")
    label = "best-epoch" if metric == "best" else "final-epoch"
    lines.append(f"Cells are {label} top-1 test accuracy %, mean +/- std over seeds.")
    return "\n".join(lines)


__all__ = ["format_table", "run_sweep", "summarize", "write_results"]
