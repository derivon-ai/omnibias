# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Dataset specs, real torchvision loaders, and an offline synthetic fallback.

Three image benchmarks are supported -- ``mnist``, ``fashion_mnist`` and
``cifar10``. :func:`real_datasets` builds the genuine torchvision datasets (with
download gated behind a flag, so CI never reaches for the network); for the
offline smoke test :func:`synthetic_datasets` fabricates a small, *learnable*
prototype-plus-noise task with the right tensor shape, so the training loop and
every arm exercise end-to-end without any download or :mod:`torchvision` import.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


@dataclass(frozen=True)
class DatasetSpec:
    """Static shape / normalization metadata for one image dataset."""

    name: str
    channels: int
    size: int
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec("mnist", 1, 28, 10, (0.1307,), (0.3081,)),
    "fashion_mnist": DatasetSpec("fashion_mnist", 1, 28, 10, (0.2860,), (0.3530,)),
    "cifar10": DatasetSpec(
        "cifar10", 3, 32, 10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    ),
}

#: Public dataset identifiers.
DATASETS: tuple[str, ...] = tuple(SPECS)


def get_spec(name: str) -> DatasetSpec:
    """Look up a :class:`DatasetSpec` by name (one of :data:`DATASETS`)."""
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}") from None


def real_datasets(
    name: str, root: str, *, download: bool = False, augment: bool = True
) -> tuple[Dataset, Dataset, DatasetSpec]:
    """Build the real torchvision train/test datasets for ``name``.

    ``torchvision`` is imported lazily so the synthetic path and a torch-only
    install keep working. ``download`` must be set explicitly to fetch the data.
    """
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

    spec = get_spec(name)
    normalize = transforms.Normalize(spec.mean, spec.std)
    eval_tfm = transforms.Compose([transforms.ToTensor(), normalize])
    if augment:
        crop = transforms.RandomCrop(spec.size, padding=4)
        flip = transforms.RandomHorizontalFlip()
        aug = [crop, flip] if name == "cifar10" else [crop]
        train_tfm = transforms.Compose([*aug, transforms.ToTensor(), normalize])
    else:
        train_tfm = eval_tfm

    cls = {"mnist": MNIST, "fashion_mnist": FashionMNIST, "cifar10": CIFAR10}[name]
    train = cls(root=root, train=True, transform=train_tfm, download=download)
    test = cls(root=root, train=False, transform=eval_tfm, download=download)
    return train, test, spec


def synthetic_datasets(
    name: str, *, n_train: int = 512, n_test: int = 256, seed: int = 0, noise: float = 0.6
) -> tuple[Dataset, Dataset, DatasetSpec]:
    """A tiny, learnable, offline stand-in shaped like ``name`` (for CI/smoke tests).

    Each class has a fixed random prototype tensor; samples are prototype + noise,
    so a small network can fit it and the surrogate arms can demonstrably reduce
    the loss -- without any download.
    """
    spec = get_spec(name)
    shape = (spec.channels, spec.size, spec.size)
    gen = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(spec.num_classes, *shape, generator=gen)

    def make(n: int, sub_seed: int) -> TensorDataset:
        g = torch.Generator().manual_seed(sub_seed)
        labels = torch.randint(0, spec.num_classes, (n,), generator=g)
        feats = prototypes[labels] + noise * torch.randn(n, *shape, generator=g)
        return TensorDataset(feats, labels)

    return make(n_train, seed + 1), make(n_test, seed + 2), spec


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap ``dataset`` in a :class:`DataLoader` with a seeded shuffle generator.

    Seeding the generator makes the batch order identical across arms for a given
    ``seed``, so the only difference between arms remains the backward.
    """
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )


__all__ = [
    "DATASETS",
    "DatasetSpec",
    "SPECS",
    "get_spec",
    "make_loader",
    "real_datasets",
    "synthetic_datasets",
]
