# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""MNIST classification with :class:`CmbNet`.

Each conv layer carries an explicit operator role (gradient ->
Laplacian -> integral) so the learnt kernels prefer the corresponding
classical scale-space feature class.

By default the script trains on a 2,000-sample MNIST subset for a few
epochs purely as a smoke test. Pass ``--full`` to train on the full
training set.

Run::

    python examples/cmbnet_mnist.py            # smoke (subset)
    python examples/cmbnet_mnist.py --full     # full training set

Requires ``torchvision`` for the dataset; if unavailable, the script
falls back to a synthetic random dataset of the same shape so the
training loop still runs.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import torch
import torch.nn.functional as F
from omnibias.torch.architectures import CmbNet
from torch.utils.data import DataLoader, Subset, TensorDataset


def _make_synthetic(n: int = 2000, seed: int = 0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return TensorDataset(X, y)


def _make_real(train: bool, root: str = "./data") -> TensorDataset | None:
    try:
        import torchvision
        from torchvision import transforms
    except ImportError:
        return None
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return torchvision.datasets.MNIST(root=root, train=train, download=True, transform=tfm)


def main(
    full: bool = False,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    subset_size: int = 2000,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)

    train_ds = _make_real(train=True)
    test_ds = _make_real(train=False)

    if train_ds is None:
        print("torchvision unavailable; falling back to synthetic data.")
        train_ds = _make_synthetic(n=subset_size, seed=seed)
        test_ds = _make_synthetic(n=subset_size // 5, seed=seed + 1)
    elif not full:
        train_ds = Subset(train_ds, range(subset_size))
        test_ds = Subset(test_ds, range(subset_size // 5))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = CmbNet()
    print(f"CmbNet params: {sum(p.numel() for p in model.parameters())}")
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = perf_counter()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for X, y in train_loader:
            optim.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item() * X.size(0)
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_n += X.size(0)
        train_acc = total_correct / total_n
        train_loss = total_loss / total_n

        model.eval()
        with torch.no_grad():
            t_correct, t_n = 0, 0
            for X, y in test_loader:
                t_correct += (model(X).argmax(-1) == y).sum().item()
                t_n += X.size(0)
            test_acc = t_correct / t_n

        elapsed = perf_counter() - t0
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  ({elapsed:.1f}s)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the full MNIST training set")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        full=args.full,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset_size=args.subset_size,
        seed=args.seed,
    )
