# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""CLI for the MNIST-1D double-descent / optimizer-axis sweep.

Examples::

    # Offline smoke test (synthetic data, tiny grid, CPU):
    python -m examples.mnist1d_double_descent.run_demo \
        --synthetic --arms adam cubic_newton --widths 4 16 64 --seeds 0 \
        --steps 30 --aggregate

    # P1: reproduce Fig. 5 with Adam + instrument curvature (real MNIST-1D):
    python -m examples.mnist1d_double_descent.run_demo \
        --register ce_relu --arms adam --noise 0.0 0.15 --seeds 0 1 2 3 4 \
        --steps 600 --scratch-dir artifacts/omnibias_mnist1d/p1 --aggregate

    # P2: one cluster job -- a slice of the optimizer-axis sweep:
    python -m examples.mnist1d_double_descent.run_demo \
        --register both --arms core --widths 30 40 50 60 75 --seeds 0 1 2 \
        --noise 0.15 --steps 600 --scratch-dir artifacts/omnibias_mnist1d/p2

Data is the frozen MNIST-1D feature set (``mnist1d`` pip package if installed,
else the vendored generator); only the label-noise mask and the model init vary
with ``--seeds``. Per-run JSONs go to ``--scratch-dir``; ``--aggregate`` reduces
them to CSV/JSON summaries under ``--out-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from examples.mnist1d_double_descent.arms import ARMS, CORE_ARMS, SHARPNESS_ARMS
from examples.mnist1d_double_descent.data import Mnist1DConfig
from examples.mnist1d_double_descent.experiment import (
    DEFAULT_WIDTHS,
    run_sweep,
    write_summary,
)
from examples.mnist1d_double_descent.models import REGISTERS, Register

_ARM_GROUPS = {"core": CORE_ARMS, "all": ARMS, "sharpness": SHARPNESS_ARMS}


def _resolve_arms(names: list[str]) -> tuple[str, ...]:
    if len(names) == 1 and names[0] in _ARM_GROUPS:
        return _ARM_GROUPS[names[0]]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {ARMS} or {list(_ARM_GROUPS)}")
    return tuple(names)


def _resolve_registers(choice: str) -> tuple[Register, ...]:
    if choice == "both":
        return REGISTERS
    if choice not in REGISTERS:
        raise SystemExit(f"unknown register {choice!r}; choose from {(*REGISTERS, 'both')}")
    return (choice,)  # type: ignore[return-value]


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    return choice


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST-1D double descent x omnibias optimizers.")
    p.add_argument("--register", default="both", help="'ce_relu', 'mse_tanh', or 'both'")
    p.add_argument("--arms", nargs="+", default=["core"], help="arm names or a group: core|all|sharpness")
    p.add_argument("--widths", nargs="+", type=int, default=list(DEFAULT_WIDTHS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.15])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--n-train", type=int, default=None, help="MNIST-1D train size (default 4000)")
    p.add_argument("--n-test", type=int, default=None, help="MNIST-1D test size (default 1000)")
    p.add_argument("--lr", type=float, default=None, help="override every arm's default lr")
    p.add_argument("--batch-size", type=int, default=None, help="minibatch (first-order arms only)")
    p.add_argument("--log-every", type=int, default=1, help="record full metrics every k steps")
    p.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'")
    p.add_argument("--scratch-dir", default="artifacts/omnibias_mnist1d/runs")
    p.add_argument("--dense-max-params", type=int, default=1500)
    p.add_argument("--curvature-every", type=int, default=0, help="0 -> ~5 evenly spaced snapshots")
    p.add_argument("--no-curvature", action="store_true", help="skip exact-curvature snapshots")
    p.add_argument("--curv-batch", type=int, default=0, help="Hessian subsample size (0 -> full train set)")
    p.add_argument("--curv-power-iters", type=int, default=40, help="matrix-free power iterations")
    p.add_argument("--curv-hutch", type=int, default=8, help="matrix-free Hutchinson samples")
    p.add_argument("--synthetic", action="store_true", help="offline synthetic data (smoke test)")
    p.add_argument("--no-pip", action="store_true", help="never use the mnist1d pip package")
    p.add_argument("--aggregate", action="store_true", help="write CSV/JSON summaries after the sweep")
    p.add_argument("--out-dir", default="examples/mnist1d_double_descent/results")
    p.add_argument("--log", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    device = _resolve_device(args.device)
    registers = _resolve_registers(args.register)
    arm_names = _resolve_arms(args.arms)
    scratch = Path(args.scratch_dir).expanduser()
    data_cfg: Mnist1DConfig | None = None
    if args.n_train is not None or args.n_test is not None:
        base = Mnist1DConfig()
        data_cfg = Mnist1DConfig(
            n_train=args.n_train if args.n_train is not None else base.n_train,
            n_test=args.n_test if args.n_test is not None else base.n_test,
        )
    print(
        f"MNIST-1D double descent: registers={registers} arms={arm_names} "
        f"widths={args.widths} seeds={args.seeds} noise={args.noise} steps={args.steps} "
        f"device={device} scratch={scratch} {'(synthetic)' if args.synthetic else ''}"
    )
    results = run_sweep(
        registers=registers,
        arm_names=arm_names,
        widths=tuple(args.widths),
        seeds=tuple(args.seeds),
        noise_levels=tuple(args.noise),
        steps=args.steps,
        depth=args.depth,
        lr_override=args.lr,
        batch_size=args.batch_size,
        log_every=args.log_every,
        device=device,
        scratch_dir=scratch,
        dense_max_params=args.dense_max_params,
        curvature=not args.no_curvature,
        curvature_every=args.curvature_every,
        curv_batch=args.curv_batch,
        curv_power_iters=args.curv_power_iters,
        curv_hutch=args.curv_hutch,
        synthetic=args.synthetic,
        cfg=data_cfg,
        allow_pip=not args.no_pip,
        log=args.log,
    )
    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"ran {len(results)} configs ({n_ok} ok, {len(results) - n_ok} errored); wrote to {scratch}")
    if args.aggregate:
        runs_csv, summary_csv, summary_json = write_summary(scratch, args.out_dir)
        print(f"wrote {runs_csv}\n      {summary_csv}\n      {summary_json}")


if __name__ == "__main__":
    main()
