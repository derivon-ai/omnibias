# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Edge-of-stability experiment: does an exact-lambda_max LR controller help at the threshold?

Motivated by the double-descent study's own findings: (H1) the exact top Hessian
eigenvalue spikes at the interpolation threshold, and (H3) it climbs monotonically
epoch-wise (progressive sharpening). Edge-of-stability theory (Cohen et al., 2021)
says full-batch GD self-stabilises at ``lambda_max ~ 2/eta``; omnibias measures the
*exact* ``lambda_max`` cheaply, so we can instead *set* the step size to sit on that
edge (:class:`~examples.mnist1d_double_descent.eos.EdgeOfStabilityLR`).

This runs the ``eos`` arm against ``adam`` and ``sgd`` at the threshold width and
writes a three-panel figure: (1) test error vs step, (2) exact ``lambda_max(H)`` vs
step (flatness), (3) the EoS control product ``lambda_max * eta`` (held at ``2c``).

CLI::

    python -m examples.mnist1d_double_descent.eos_experiment \
        --width 24 --steps 600 --noise 0.35 --n-train 1000 --n-test 1000 \
        --scratch-dir artifacts/omnibias_mnist1d/eos \
        --out-dir examples/mnist1d_double_descent/results/eos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

from examples.mnist1d_double_descent.arms import get_arm
from examples.mnist1d_double_descent.data import Mnist1DConfig, load_mnist1d
from examples.mnist1d_double_descent.train import RunConfig, save_run, train_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS_TO_RUN = ("eos", "adam", "sgd")
_COLORS = {"eos": "C3", "adam": "C0", "sgd": "C2"}


def _series(history: list[dict[str, object]], key: str) -> tuple[list[int], list[float]]:
    steps = [int(h["step"]) for h in history if key in h]
    vals = [float(h[key]) for h in history if key in h]
    return steps, vals


def _curv_series(history: list[dict[str, object]], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    vals: list[float] = []
    for h in history:
        curv = h.get("curvature")
        if isinstance(curv, dict) and key in curv:
            steps.append(int(h["step"]))
            vals.append(float(curv[key]))
    return steps, vals


def run_all(
    *, width: int, steps: int, noise: float, n_train: int, n_test: int, seed: int,
    scratch_dir: str,
) -> dict[str, dict]:
    """Train every arm in :data:`ARMS_TO_RUN` at ``width`` and return their run dicts."""
    cfg_data = Mnist1DConfig(n_train=n_train, n_test=n_test)
    bundle = load_mnist1d(cfg_data, label_noise=noise, scratch_dir=scratch_dir)
    results: dict[str, dict] = {}
    for name in ARMS_TO_RUN:
        arm = get_arm(name)
        cfg = RunConfig(
            register="ce_relu", arm=name, width=width, depth=1, seed=seed,
            label_noise=noise, steps=steps, lr=arm.lr,
            log_every=10, curvature=True, curvature_every=50,
            curv_batch=384, dense_max_params=0, curv_power_iters=12, curv_hutch=2,
        )
        res = train_run(bundle, arm, cfg, log=True)
        save_run(res, scratch_dir)
        results[name] = {
            "history": res.history,
            "final_test_err": res.final_test_err,
            "best_test_err": res.best_test_err,
            "final_train_err": res.final_train_err,
            "interpolation_step": res.interpolation_step,
            "curvature_final": res.curvature_final,
        }
    return results


def make_figure(results: dict[str, dict], out_dir: Path, *, c: float) -> Path:
    """Three-panel figure: test error, exact lambda_max(H), and the EoS control product."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in ARMS_TO_RUN:
        hist = results[name]["history"]
        color = _COLORS[name]
        s, te = _series(hist, "test_err")
        ax1.plot(s, te, color=color, label=name)
        cs, lam = _curv_series(hist, "lambda_max")
        if cs:
            ax2.plot(cs, lam, color=color, marker="o", ms=3, label=name)
    ax1.set_xlabel("step")
    ax1.set_ylabel("test error")
    ax1.set_title("Test error vs step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("step")
    ax2.set_ylabel("exact lambda_max(H)")
    ax2.set_title("Sharpness (exact top eigenvalue) vs step")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    es, le = _series(results["eos"]["history"], "eos_lambda_eta")
    ax3.plot(es, le, color="C3", marker="s", ms=3, label="eos: lambda_max * eta")
    ax3.axhline(2.0 * c, color="k", ls="--", lw=1, label=f"edge target 2c = {2 * c:g}")
    ax3.set_xlabel("step")
    ax3.set_ylabel("lambda_max * eta")
    ax3.set_title("EoS control: step size holds the edge")
    ax3.set_ylim(0.0, max(2.5, 2.0 * c + 0.6))
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Exact edge-of-stability control at the interpolation threshold (ce_relu, width 24)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eos_control.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _summary(results: dict[str, dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in ARMS_TO_RUN:
        r = results[name]
        cf = r.get("curvature_final") or {}
        rows.append({
            "arm": name,
            "final_test_err": r["final_test_err"],
            "best_test_err": r["best_test_err"],
            "final_train_err": r["final_train_err"],
            "interpolation_step": r["interpolation_step"],
            "final_lambda_max": float(cf["lambda_max"]) if isinstance(cf, dict) and "lambda_max" in cf else None,
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Edge-of-stability controller experiment.")
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--noise", type=float, default=0.35)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scratch-dir", default="artifacts/omnibias_mnist1d/eos")
    p.add_argument("--out-dir", default="examples/mnist1d_double_descent/results/eos")
    args = p.parse_args(argv)

    results = run_all(
        width=args.width, steps=args.steps, noise=args.noise,
        n_train=args.n_train, n_test=args.n_test, seed=args.seed,
        scratch_dir=args.scratch_dir,
    )
    out_dir = Path(args.out_dir).expanduser()
    fig_path = make_figure(results, out_dir / "figures", c=float(get_arm("eos").hypers["c"]))
    rows = _summary(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {fig_path}")
    print(f"{'arm':>6s}  {'final_test':>10s}  {'best_test':>9s}  {'train_err':>9s}  {'interp':>6s}  {'lam_max':>8s}")
    for r in rows:
        lam = r["final_lambda_max"]
        lam_s = f"{lam:.3f}" if isinstance(lam, float) else "   n/a"
        print(
            f"{r['arm']:>6s}  {float(r['final_test_err']):>10.3f}  {float(r['best_test_err']):>9.3f}  "
            f"{float(r['final_train_err']):>9.3f}  {int(r['interpolation_step']):>6d}  {lam_s:>8s}"
        )


if __name__ == "__main__":
    main()


__all__ = ["make_figure", "run_all"]
