# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Edge-of-stability ablation: multi-seed robustness + the ``c`` and momentum knobs.

Follow-up to :mod:`~examples.mnist1d_double_descent.eos_experiment`. The single-seed
run showed the exact-``lambda_max`` controller pins ``lambda_max * eta`` on the edge
and ends flatter than Adam/SGD, but *under-fits* (its curvature-capped step shrinks as
the loss sharpens). This asks three questions across seeds at the threshold width:

* **robustness** -- does the eos-vs-Adam-vs-SGD picture hold across seeds?
* **the ``c`` knob** -- does sitting *on* (``c = 1``) or *past* (``c = 1.05``) the linear
  edge let eos fit more (lower train error) without blowing up?
* **momentum** -- does heavy-ball momentum (target widened to ``2c(1 + beta)``) recover
  the fitting speed while keeping the controlled sharpness?

Writes ``eos_ablation.png`` (mean test error, best-test bars, final-``lambda_max`` bars)
and ``eos_control_variants.png`` (each variant's ``lambda_max * eta`` locked at its own
``2c(1 + beta)`` target), plus an aggregated ``ablation_summary.json``.

CLI::

    python -m examples.mnist1d_double_descent.eos_ablation \
        --width 24 --steps 400 --noise 0.35 --seeds 0 1 2 \
        --scratch-dir artifacts/omnibias_mnist1d/eos_ablation \
        --out-dir examples/mnist1d_double_descent/results/eos
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn

from examples.mnist1d_double_descent.arms import OptimizerArm, get_arm
from examples.mnist1d_double_descent.data import Mnist1DConfig, load_mnist1d
from examples.mnist1d_double_descent.train import RunConfig, train_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Draw / report order (baselines first, then the eos family widening its edge).
ORDER = ("adam", "sgd", "eos", "eos_edge", "eos_past", "eos_mom")
EOS_VARIANTS = ("eos", "eos_edge", "eos_past", "eos_mom")
_COLORS = {
    "adam": "C0", "sgd": "C2",
    "eos": "C3", "eos_edge": "C1", "eos_past": "C4", "eos_mom": "C5",
}


def _eos_variant(name: str, *, c: float, momentum: float) -> OptimizerArm:
    """A custom eos arm (SGD carrier at ``momentum``, controller targeting ``2c(1+beta)``)."""

    def factory(model: nn.Module, lr: float) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    return OptimizerArm(
        name, "eos", factory, lr=1e-2,
        hypers={
            "c": c, "momentum": momentum, "eta_min": 1e-4, "eta_max": 1.0,
            "probe_iters": 10, "measure_every": 4, "ema": 0.5,
        },
    )


def build_arms() -> dict[str, OptimizerArm]:
    """The ablation arms: Adam / SGD baselines and four eos variants."""
    return {
        "adam": get_arm("adam"),
        "sgd": get_arm("sgd"),
        "eos": _eos_variant("eos", c=0.9, momentum=0.0),
        "eos_edge": _eos_variant("eos_edge", c=1.0, momentum=0.0),
        "eos_past": _eos_variant("eos_past", c=1.05, momentum=0.0),
        "eos_mom": _eos_variant("eos_mom", c=0.9, momentum=0.9),
    }


def _finite(x: float) -> float:
    return x if math.isfinite(x) else math.nan


def run_all(
    *, width: int, steps: int, noise: float, n_train: int, n_test: int,
    seeds: tuple[int, ...], scratch_dir: str,
) -> dict[str, dict]:
    """Train every arm across ``seeds`` at ``width``; return per-arm histories + scalars."""
    cfg_data = Mnist1DConfig(n_train=n_train, n_test=n_test)
    bundle = load_mnist1d(cfg_data, label_noise=noise, scratch_dir=scratch_dir)
    arms = build_arms()
    out: dict[str, dict] = {}
    for name in ORDER:
        arm = arms[name]
        histories: list[list[dict[str, object]]] = []
        scalars: list[dict[str, float]] = []
        for seed in seeds:
            cfg = RunConfig(
                register="ce_relu", arm=name, width=width, depth=1, seed=seed,
                label_noise=noise, steps=steps, lr=arm.lr, log_every=10,
                curvature=True, curvature_every=100, curv_batch=384,
                dense_max_params=0, curv_power_iters=12, curv_hutch=2,
            )
            try:
                res = train_run(bundle, arm, cfg, log=True)
            except Exception as exc:  # noqa: BLE001 -- a diverging variant must not abort the sweep
                print(f"  [{name} s={seed}] FAILED: {exc}")
                continue
            histories.append(res.history)
            cf = res.curvature_final if isinstance(res.curvature_final, dict) else {}
            scalars.append({
                "seed": float(seed),
                "best_test_err": _finite(res.best_test_err),
                "final_test_err": _finite(res.final_test_err),
                "final_train_err": _finite(res.final_train_err),
                "final_lambda_max": _finite(float(cf.get("lambda_max", math.nan))),
            })
        out[name] = {"histories": histories, "scalars": scalars}
    return out


def _mean_curve(histories: list[list[dict[str, object]]], key: str) -> tuple[list[int], np.ndarray]:
    series: list[list[float]] = []
    steps_ref: list[int] | None = None
    for hist in histories:
        steps = [int(e["step"]) for e in hist if key in e]
        vals = [float(e[key]) for e in hist if key in e]
        if steps_ref is None:
            steps_ref = steps
        if steps == steps_ref:
            series.append(vals)
    if steps_ref is None or not series:
        return [], np.empty((0, 0))
    return steps_ref, np.asarray(series)


def _agg(scalars: list[dict[str, float]], key: str) -> tuple[float, float]:
    vals = [s[key] for s in scalars if math.isfinite(s[key])]
    if not vals:
        return math.nan, 0.0
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std())


def make_figures(results: dict[str, dict], out_dir: Path) -> list[Path]:
    """Write the ablation summary figure and the per-variant control figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in ORDER:
        steps, arr = _mean_curve(results[name]["histories"], "test_err")
        if len(steps):
            ax1.plot(steps, arr.mean(0), color=_COLORS[name], label=name)
    ax1.set_xlabel("step")
    ax1.set_ylabel("test error (mean over seeds)")
    ax1.set_title("Test error vs step")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    xs = list(range(len(ORDER)))
    best_m = [_agg(results[n]["scalars"], "best_test_err")[0] for n in ORDER]
    best_s = [_agg(results[n]["scalars"], "best_test_err")[1] for n in ORDER]
    ax2.bar(xs, best_m, yerr=best_s, color=[_COLORS[n] for n in ORDER], capsize=3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(ORDER, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("best test error")
    ax2.set_title("Best test error (mean +/- std)")
    ax2.grid(True, axis="y", alpha=0.3)

    lam_m = [_agg(results[n]["scalars"], "final_lambda_max")[0] for n in ORDER]
    lam_s = [_agg(results[n]["scalars"], "final_lambda_max")[1] for n in ORDER]
    ax3.bar(xs, lam_m, yerr=lam_s, color=[_COLORS[n] for n in ORDER], capsize=3)
    ax3.set_xticks(xs)
    ax3.set_xticklabels(ORDER, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("final exact lambda_max(H)")
    ax3.set_title("Final sharpness (mean +/- std)")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Edge-of-stability ablation at the threshold (ce_relu, width 24)")
    fig.tight_layout()
    p1 = out_dir / "eos_ablation.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for name in EOS_VARIANTS:
        steps, arr = _mean_curve(results[name]["histories"], "eos_lambda_eta")
        _, tgt = _mean_curve(results[name]["histories"], "eos_target")
        if len(steps):
            ax.plot(steps, arr.mean(0), color=_COLORS[name], marker=".", ms=4, label=name)
            if tgt.size:
                ax.axhline(float(tgt.mean()), color=_COLORS[name], ls="--", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("lambda_max * eta")
    ax.set_title("EoS control: each variant holds its own 2c(1+beta) target")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / "eos_control_variants.png"
    fig2.savefig(p2, dpi=130)
    plt.close(fig2)
    paths.append(p2)
    return paths


def summarize(results: dict[str, dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in ORDER:
        sc = results[name]["scalars"]
        row: dict[str, object] = {"arm": name, "n_seeds": len(sc)}
        for key in ("best_test_err", "final_test_err", "final_train_err", "final_lambda_max"):
            m, s = _agg(sc, key)
            row[f"{key}_mean"] = m
            row[f"{key}_std"] = s
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Edge-of-stability ablation (c-sweep + momentum).")
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--noise", type=float, default=0.35)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--scratch-dir", default="artifacts/omnibias_mnist1d/eos_ablation")
    p.add_argument("--out-dir", default="examples/mnist1d_double_descent/results/eos")
    args = p.parse_args(argv)

    results = run_all(
        width=args.width, steps=args.steps, noise=args.noise,
        n_train=args.n_train, n_test=args.n_test, seeds=tuple(args.seeds),
        scratch_dir=args.scratch_dir,
    )
    out_dir = Path(args.out_dir).expanduser()
    figs = make_figures(results, out_dir / "figures")
    rows = summarize(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ablation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    for fig in figs:
        print(f"wrote {fig}")
    hdr = f"{'arm':>9s}  {'best_test':>16s}  {'final_test':>16s}  {'train_err':>16s}  {'lambda_max':>16s}"
    print(hdr)

    def _ms(row: dict[str, object], key: str) -> str:
        return f"{float(row[f'{key}_mean']):.3f}+/-{float(row[f'{key}_std']):.3f}"

    for r in rows:
        print(
            f"{str(r['arm']):>9s}  {_ms(r, 'best_test_err'):>16s}  {_ms(r, 'final_test_err'):>16s}  "
            f"{_ms(r, 'final_train_err'):>16s}  {_ms(r, 'final_lambda_max'):>16s}"
        )


if __name__ == "__main__":
    main()


__all__ = ["build_arms", "make_figures", "run_all", "summarize"]
