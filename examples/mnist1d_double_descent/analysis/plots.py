# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Figures for the double-descent study.

Reads the aggregated ``summary.json`` written by
:func:`examples.mnist1d_double_descent.experiment.write_summary` (mean +/- std test error
and mean curvature per ``(register, arm, width, label_noise)``) for the width-curve
figures, and an individual per-run JSON for the epoch-wise trajectory. Every figure
degrades gracefully when its slice is absent. Uses the headless ``Agg`` backend.

CLI::

    python -m examples.mnist1d_double_descent.analysis.plots \
        --summary examples/mnist1d_double_descent/results/summary.json \
        --out examples/mnist1d_double_descent/results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Row = dict[str, Any]

_SHARPNESS = ("adam", "sam_stochastic", "sam_exact", "sharpness_reg")
_SUBSPACE = ("jet_subspace_o2", "jet_subspace_o3")


def load_summary(path: str | Path) -> list[Row]:
    """Load the aggregate ``summary.json`` (a list of per-(register,arm,width,noise) rows)."""
    with Path(path).expanduser().open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"expected a list of aggregate rows in {path}")
    return data


# ---------------------------------------------------------------------------
# selection helpers
# ---------------------------------------------------------------------------


def _noises(rows: list[Row]) -> list[float]:
    return sorted({float(r["label_noise"]) for r in rows})


def _registers(rows: list[Row]) -> list[str]:
    return sorted({str(r["register"]) for r in rows})


def _arms(rows: list[Row], register: str) -> list[str]:
    return sorted({str(r["arm"]) for r in rows if r["register"] == register})


def _curve(
    rows: list[Row], *, register: str, arm: str, noise: float, key: str
) -> tuple[list[int], list[float], list[float]]:
    sel = [
        r
        for r in rows
        if r["register"] == register and r["arm"] == arm and abs(float(r["label_noise"]) - noise) < 1e-9
    ]
    sel.sort(key=lambda r: int(r["width"]))
    widths = [int(r["width"]) for r in sel]
    vals = [float(r.get(key, float("nan"))) for r in sel]
    std_key = key.replace("_mean", "_std")
    stds = [float(r.get(std_key, 0.0)) for r in sel]
    return widths, vals, stds


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def double_descent_figure(rows: list[Row], out_dir: Path, *, noise: float) -> Path | None:
    """Test error vs width, one line per optimizer arm, one panel per register (H2)."""
    registers = _registers(rows)
    if not registers:
        return None
    fig, axes = plt.subplots(1, len(registers), figsize=(7 * len(registers), 5), squeeze=False)
    for ax, register in zip(axes[0], registers, strict=True):
        for arm in _arms(rows, register):
            widths, means, stds = _curve(
                rows, register=register, arm=arm, noise=noise, key="test_err_mean"
            )
            if widths:
                ax.errorbar(widths, means, yerr=stds, marker="o", ms=3, capsize=2, label=arm)
        ax.set_xscale("log")
        ax.set_xlabel("hidden width")
        ax.set_ylabel("test error")
        ax.set_title(f"{register}  (noise={noise:g})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Model-wise double descent: test error vs width")
    return _save(fig, out_dir, f"double_descent_noise{noise:g}.png")


def curvature_overlay_figure(
    rows: list[Row], out_dir: Path, *, register: str, noise: float, arm: str = "adam"
) -> Path | None:
    """Overlay test error and mean lambda_max vs width for one arm (H1)."""
    widths, err, err_std = _curve(rows, register=register, arm=arm, noise=noise, key="test_err_mean")
    _, lam, _ = _curve(rows, register=register, arm=arm, noise=noise, key="lambda_max_mean")
    if not widths or all(v != v for v in lam):
        return None
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.errorbar(widths, err, yerr=err_std, color="C0", marker="o", ms=3, label="test error")
    ax1.set_xscale("log")
    ax1.set_xlabel("hidden width")
    ax1.set_ylabel("test error", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(widths, lam, color="C3", marker="s", ms=3, label="lambda_max(H)")
    ax2.set_ylabel("mean exact lambda_max(H)", color="C3")
    ax1.set_title(f"Curvature overlay: {register} / {arm} (noise={noise:g})")
    ax1.grid(True, alpha=0.3)
    return _save(fig, out_dir, f"curvature_overlay_{register}_{arm}_noise{noise:g}.png")


def register_comparison_figure(
    rows: list[Row], out_dir: Path, *, noise: float, arm: str = "adam"
) -> Path | None:
    """Compare the ce_relu and mse_tanh double-descent curves for one arm."""
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for register, color in (("ce_relu", "C0"), ("mse_tanh", "C1")):
        widths, means, stds = _curve(
            rows, register=register, arm=arm, noise=noise, key="test_err_mean"
        )
        if widths:
            ax.errorbar(widths, means, yerr=stds, color=color, marker="o", ms=3, capsize=2,
                        label=f"{register} ({arm})")
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_xlabel("hidden width")
    ax.set_ylabel("test error")
    ax.set_title(f"Register comparison (noise={noise:g})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, f"register_comparison_noise{noise:g}.png")


def _subset_figure(
    rows: list[Row], out_dir: Path, *, register: str, noise: float, arms: tuple[str, ...],
    title: str, fname: str,
) -> Path | None:
    present = [a for a in arms if a in _arms(rows, register)]
    if len(present) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm in present:
        widths, means, stds = _curve(
            rows, register=register, arm=arm, noise=noise, key="test_err_mean"
        )
        if widths:
            ax.errorbar(widths, means, yerr=stds, marker="o", ms=3, capsize=2, label=arm)
    ax.set_xscale("log")
    ax.set_xlabel("hidden width")
    ax.set_ylabel("test error")
    ax.set_title(f"{title}: {register} (noise={noise:g})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, fname)


def sharpness_figure(rows: list[Row], out_dir: Path, *, register: str, noise: float) -> Path | None:
    """Compare the sharpness-intervention arms against the Adam baseline (H4)."""
    return _subset_figure(
        rows, out_dir, register=register, noise=noise, arms=_SHARPNESS,
        title="Sharpness interventions", fname=f"sharpness_{register}_noise{noise:g}.png",
    )


def subspace_figure(rows: list[Row], out_dir: Path, *, register: str, noise: float) -> Path | None:
    """Compare JetSubspaceTensor order-2 vs order-3 near the threshold (H7)."""
    return _subset_figure(
        rows, out_dir, register=register, noise=noise, arms=_SUBSPACE,
        title="Subspace order 2 vs 3", fname=f"subspace_{register}_noise{noise:g}.png",
    )


def memory_efficiency_figure(
    rows: list[Row], out_dir: Path, *, register: str, noise: float
) -> Path | None:
    """Phase-1 memory / cost panel: optimizer-state footprint and the generalisation-vs-cost
    trade-off per arm (degrades gracefully when the telemetry columns are absent)."""
    sel = [
        r
        for r in rows
        if r["register"] == register and abs(float(r["label_noise"]) - noise) < 1e-9
        and float(r.get("opt_state_bytes", 0) or 0) > 0
    ]
    if not sel:
        return None
    max_w = max(int(r["width"]) for r in sel)
    at_w = sorted((r for r in sel if int(r["width"]) == max_w), key=lambda r: str(r["arm"]))
    if len(at_w) < 2:
        return None
    arms = [str(r["arm"]) for r in at_w]
    kib = [float(r["opt_state_bytes"]) / 1024.0 for r in at_w]
    errs = [float(r.get("test_err_mean", float("nan"))) for r in at_w]
    times = [float(r.get("update_time_s_mean", 0.0)) for r in at_w]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(arms, kib, color="C0")
    ax1.set_ylabel(f"optimizer state (KiB) at width={max_w}")
    ax1.set_title("Optimizer memory footprint")
    ax1.tick_params(axis="x", rotation=45)
    for arm, t, e in zip(arms, times, errs, strict=True):
        ax2.scatter([t], [e], s=40, label=arm)
    ax2.set_xlabel("mean update wall-clock (s)")
    ax2.set_ylabel("test error")
    ax2.set_title("Generalisation vs optimizer cost")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Phase-1 memory / cost: {register} (noise={noise:g})")
    return _save(fig, out_dir, f"memory_cost_{register}_noise{noise:g}.png")


def epochwise_figure(run_json: str | Path, out_dir: Path) -> Path | None:
    """Epoch-wise test error and lambda_max from one per-run JSON (H3)."""
    path = Path(run_json).expanduser()
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        run = json.load(fh)
    history = run.get("history") or []
    if not history:
        return None
    cfg = run.get("config", {})
    steps = [h["step"] for h in history]
    test_err = [h["test_err"] for h in history]
    curv_steps = [h["step"] for h in history if isinstance(h.get("curvature"), dict)]
    curv_lam = [h["curvature"]["lambda_max"] for h in history if isinstance(h.get("curvature"), dict)]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(steps, test_err, color="C0", label="test error")
    ax1.set_xlabel("step")
    ax1.set_ylabel("test error", color="C0")
    if curv_steps:
        ax2 = ax1.twinx()
        ax2.plot(curv_steps, curv_lam, color="C3", marker="s", ms=3, label="lambda_max(H)")
        ax2.set_ylabel("exact lambda_max(H)", color="C3")
    ax1.set_title(f"Epoch-wise: {cfg.get('register')} / {cfg.get('arm')} width={cfg.get('width')}")
    ax1.grid(True, alpha=0.3)
    return _save(fig, out_dir, f"epochwise_{cfg.get('register')}_{cfg.get('arm')}_w{cfg.get('width')}.png")


def certified_figure(certified_json: str | Path, out_dir: Path) -> Path | None:
    """Bar chart of certified Lipschitz / robustness vs width (reads a certified.json list)."""
    path = Path(certified_json).expanduser()
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    readouts = data if isinstance(data, list) else [data]
    if not readouts:
        return None
    readouts = sorted(readouts, key=lambda d: int(d["width"]))
    widths = [str(d["width"]) for d in readouts]
    lip = [float(d["lipschitz_inf"]) for d in readouts]
    robust = [float(d["robust_frac"]) for d in readouts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(widths, lip, color="C4")
    ax1.set_xlabel("hidden width")
    ax1.set_ylabel("certified Lipschitz (l-inf)")
    ax1.set_title("Certified Lipschitz upper bound")
    ax2.bar(widths, robust, color="C2")
    ax2.set_xlabel("hidden width")
    ax2.set_ylabel("certified-robust fraction")
    ax2.set_title("Certified robustness")
    fig.suptitle("Certified read-outs (sound enclosures)")
    return _save(fig, out_dir, "certified.png")


def make_all(
    summary_path: str | Path,
    out_dir: str | Path,
    *,
    epochwise_run: str | Path | None = None,
    certified_path: str | Path | None = None,
) -> list[Path]:
    """Generate every applicable figure from ``summary.json`` (+ optional per-run / certified)."""
    rows = load_summary(summary_path)
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path | None] = []
    for noise in _noises(rows):
        made.append(double_descent_figure(rows, out, noise=noise))
        made.append(register_comparison_figure(rows, out, noise=noise))
        for register in _registers(rows):
            made.append(curvature_overlay_figure(rows, out, register=register, noise=noise))
            made.append(sharpness_figure(rows, out, register=register, noise=noise))
            made.append(subspace_figure(rows, out, register=register, noise=noise))
            made.append(memory_efficiency_figure(rows, out, register=register, noise=noise))
    if epochwise_run is not None:
        made.append(epochwise_figure(epochwise_run, out))
    if certified_path is not None:
        made.append(certified_figure(certified_path, out))
    return [p for p in made if p is not None]


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the MNIST-1D double-descent results.")
    p.add_argument("--summary", required=True, help="path to summary.json (aggregate rows)")
    p.add_argument("--out", required=True, help="output figures directory")
    p.add_argument("--epochwise-run", default=None, help="optional per-run JSON for the trajectory")
    p.add_argument("--certified", default=None, help="optional certified.json path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    made = make_all(
        args.summary, args.out, epochwise_run=args.epochwise_run, certified_path=args.certified
    )
    print(f"Wrote {len(made)} figures to {args.out}:")
    for p in made:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()


__all__ = [
    "certified_figure",
    "curvature_overlay_figure",
    "double_descent_figure",
    "epochwise_figure",
    "load_summary",
    "make_all",
    "memory_efficiency_figure",
    "register_comparison_figure",
    "sharpness_figure",
    "subspace_figure",
]
