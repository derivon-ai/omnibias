# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""CLI for the binary-vs-STE benchmark sweep.

Examples::

    # Offline smoke test (synthetic data, no download, CPU):
    python -m examples.binary_vs_ste.run_demo --synthetic --epochs 2 --seeds 0

    # Real MNIST + Fashion-MNIST on CPU/GPU (download once):
    python -m examples.binary_vs_ste.run_demo \
        --datasets mnist fashion_mnist --download --epochs 15 --seeds 0 1 2

    # CIFAR-10 on a GPU box:
    python -m examples.binary_vs_ste.run_demo \
        --datasets cifar10 --download --epochs 60 --seeds 0 1 2 --device cuda

The forward is always the exact hard quantizer; arms differ only in the backward,
so the table measures which surrogate gradient trains better -- not a claim that
the hard step is differentiable.
"""

from __future__ import annotations

import argparse

import torch

from examples.binary_vs_ste.arms import ARMS
from examples.binary_vs_ste.data import DATASETS
from examples.binary_vs_ste.experiment import format_table, run_sweep


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary NN: omnibias surrogate vs STE baseline.")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true", help="allow torchvision to download")
    p.add_argument("--no-augment", action="store_true", help="disable train-time augmentation")
    p.add_argument("--synthetic", action="store_true", help="offline synthetic data (smoke test)")
    p.add_argument(
        "--schedule",
        default="exp",
        choices=["linear", "exp", "cosine"],
        help="beta-annealing schedule (anneal arm only; range is per-arm)",
    )
    p.add_argument(
        "--xnor-scale",
        action="store_true",
        help="XNOR-Net per-filter weight scale alpha=mean|W| (lifts every arm)",
    )
    p.add_argument(
        "--lr-schedule",
        default="constant",
        choices=["constant", "cosine"],
        help="optimiser LR schedule (cosine decay reduces final-epoch noise)",
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default="examples/binary_vs_ste/results")
    return p.parse_args(argv)


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    return choice


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    device = _resolve_device(args.device)
    print(
        f"Binary-vs-STE sweep: datasets={args.datasets} arms={args.arms} "
        f"seeds={args.seeds} epochs={args.epochs} device={device} "
        f"xnor={args.xnor_scale} lr_schedule={args.lr_schedule} "
        f"{'(synthetic)' if args.synthetic else ''}"
    )
    results = run_sweep(
        datasets=tuple(args.datasets),
        arms=tuple(args.arms),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        data_root=args.data_root,
        download=args.download,
        augment=not args.no_augment,
        synthetic=args.synthetic,
        schedule=args.schedule,
        xnor=args.xnor_scale,
        lr_schedule=args.lr_schedule,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        log=True,
    )
    print("\n" + format_table(results, metric="best"))
    print("\n" + format_table(results, metric="final"))
    print(f"\nWrote results to {args.out_dir}/results.json and results.csv")


if __name__ == "__main__":
    main()
