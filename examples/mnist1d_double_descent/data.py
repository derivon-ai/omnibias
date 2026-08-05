# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""MNIST-1D data: pip package (faithful) + a vendored generator + a synthetic mode.

Three ways to get data, in decreasing fidelity:

* :func:`load_mnist1d` first tries the ``mnist1d`` pip package
  (``pip install mnist1d``) and reproduces the frozen default dataset (seed 42),
  caching it to ``scratch_dir`` as an ``.npz``.
* If the package is missing it falls back to :func:`generate_mnist1d`, a
  self-contained re-implementation of the MNIST-1D *construction recipe*
  (ten hand-crafted length-12 templates -> pad -> shear -> circular translate ->
  Gaussian-correlate -> add correlated + iid noise -> downsample to 40). It is a
  faithful re-implementation of the recipe, **not** byte-identical to the pip
  dataset -- absolute accuracies differ, but the spatial structure and the
  double-descent behaviour under label noise are preserved.
* :func:`synthetic_mnist1d` fabricates a tiny, learnable prototype-plus-noise
  task with the right shape for offline CI smoke tests.

Label noise (the amplifier that makes double descent pronounced) is applied to
the **training** labels only, with a fixed per-seed mask (:func:`add_label_noise`).

Only NumPy + torch are used (no SciPy), so the vendored path runs anywhere torch
runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

NUM_CLASSES = 10
FINAL_SEQ_LEN = 40


@dataclass(frozen=True)
class Mnist1DConfig:
    """Generation parameters (defaults mirror the paper's Table 2)."""

    n_train: int = 4000
    n_test: int = 1000
    seed: int = 42
    template_len: int = 12
    padding: tuple[int, int] = (36, 60)
    max_translation: int = 48
    corr_noise_scale: float = 0.25
    iid_noise_scale: float = 0.02
    shear_scale: float = 0.75
    gaussian_filter_width: float = 2.0
    final_seq_len: int = FINAL_SEQ_LEN
    num_classes: int = NUM_CLASSES


@dataclass
class DataBundle:
    """A ready-to-train MNIST-1D split as float32 / int64 torch tensors.

    ``x_*`` are ``(N, final_seq_len)`` standardised features; ``y_*`` are
    ``(N,)`` int64 labels; ``y_train_clean`` keeps the pre-noise labels and
    ``noise_mask`` flags the corrupted training points.
    """

    x_train: Tensor
    y_train: Tensor
    x_test: Tensor
    y_test: Tensor
    y_train_clean: Tensor
    noise_mask: Tensor
    num_classes: int
    in_dim: int
    source: str
    label_noise: float
    meta: dict[str, float] = field(default_factory=dict)

    def to(self, device: str) -> DataBundle:
        """Move every tensor to ``device`` (returns a new bundle)."""
        return DataBundle(
            x_train=self.x_train.to(device),
            y_train=self.y_train.to(device),
            x_test=self.x_test.to(device),
            y_test=self.y_test.to(device),
            y_train_clean=self.y_train_clean.to(device),
            noise_mask=self.noise_mask.to(device),
            num_classes=self.num_classes,
            in_dim=self.in_dim,
            source=self.source,
            label_noise=self.label_noise,
            meta=dict(self.meta),
        )


# ---------------------------------------------------------------------------
# Vendored generator (faithful re-implementation of the MNIST-1D recipe)
# ---------------------------------------------------------------------------


def _templates() -> np.ndarray:
    """Ten hand-crafted length-12 digit-like patterns (rows 0..9)."""
    t = np.array(
        [
            [5, 6, 6.5, 7, 7, 7, 7, 7, 6.5, 6, 5, 5],  # 0: round bump
            [3, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8],  # 1: rising ramp
            [8, 8, 7, 6, 6, 7, 8, 7, 5, 4, 4, 4],  # 2: s-ish
            [8, 7, 6, 6, 7, 8, 7, 6, 6, 7, 8, 8],  # 3: double bump
            [3, 4, 5, 6, 7, 8, 8, 5, 4, 4, 4, 4],  # 4: spike then flat
            [8, 8, 7, 6, 5, 5, 6, 7, 6, 5, 4, 4],  # 5: wave down
            [8, 7, 6, 5, 4, 4, 5, 6, 6, 6, 5, 5],  # 6: valley
            [8, 8, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3],  # 7: falling ramp
            [5, 6, 7, 6, 5, 6, 7, 6, 5, 6, 7, 6],  # 8: oscillation
            [7, 8, 8, 7, 6, 5, 4, 4, 5, 6, 6, 5],  # 9: peak then dip
        ],
        dtype=np.float64,
    )
    return t - t.mean(axis=1, keepdims=True)


def _gaussian_kernel(width: float) -> np.ndarray:
    radius = max(int(round(3.0 * width)), 1)
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (xs / max(width, 1e-6)) ** 2)
    return k / k.sum()


def _smooth(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.convolve(x, kernel, mode="same")


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, num=len(x))
    dst = np.linspace(0.0, 1.0, num=n)
    return np.interp(dst, src, x)


def _make_one(template: np.ndarray, rng: np.random.RandomState, cfg: Mnist1DConfig,
              kernel: np.ndarray) -> np.ndarray:
    x = template.copy()
    pad_lo, pad_hi = cfg.padding
    pad = rng.randint(pad_lo, pad_hi + 1)
    x = np.concatenate([x, np.zeros(pad, dtype=np.float64)])
    slope = rng.randn() * cfg.shear_scale
    x = x + slope * np.linspace(-0.5, 0.5, num=len(x))
    shift = rng.randint(-cfg.max_translation, cfg.max_translation + 1)
    x = np.roll(x, shift)
    x = _smooth(x, kernel)
    corr = _smooth(rng.randn(len(x)), kernel) * cfg.corr_noise_scale
    iid = rng.randn(len(x)) * cfg.iid_noise_scale
    x = x + corr + iid
    return _resample(x, cfg.final_seq_len)


def generate_mnist1d(cfg: Mnist1DConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vendored MNIST-1D construction. Returns ``(x_train, y_train, x_test, y_test)`` numpy arrays."""
    rng = np.random.RandomState(cfg.seed)
    templates = _templates()
    kernel = _gaussian_kernel(cfg.gaussian_filter_width)
    n = cfg.n_train + cfg.n_test
    labels = rng.randint(0, cfg.num_classes, size=n)
    xs = np.stack([_make_one(templates[y], rng, cfg, kernel) for y in labels])
    # standardise features globally (per the usual MNIST-1D preprocessing)
    xs = (xs - xs.mean()) / (xs.std() + 1e-8)
    x_train, x_test = xs[: cfg.n_train], xs[cfg.n_train :]
    y_train, y_test = labels[: cfg.n_train], labels[cfg.n_train :]
    return (
        x_train.astype(np.float32),
        y_train.astype(np.int64),
        x_test.astype(np.float32),
        y_test.astype(np.int64),
    )


def _load_pip_mnist1d(cfg: Mnist1DConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the dataset from the ``mnist1d`` pip package (raises ImportError if absent)."""
    from mnist1d.data import get_dataset_args, make_dataset  # type: ignore[import-not-found]

    args = get_dataset_args()
    args.num_samples = cfg.n_train + cfg.n_test
    args.train_split = cfg.n_train / (cfg.n_train + cfg.n_test)
    args.seed = cfg.seed
    data = make_dataset(args)
    x_train = np.asarray(data["x"], dtype=np.float32)
    y_train = np.asarray(data["y"], dtype=np.int64)
    x_test = np.asarray(data["x_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.int64)
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Label noise / one-hot
# ---------------------------------------------------------------------------


def add_label_noise(
    y: Tensor, frac: float, *, num_classes: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Corrupt a fixed ``frac`` of labels to a *different* uniform class.

    Deterministic given ``seed``; returns ``(y_noisy, mask)`` where ``mask`` is a
    boolean of the flipped positions.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"label noise frac must be in [0, 1], got {frac}")
    y_noisy = y.clone()
    mask = torch.zeros_like(y, dtype=torch.bool)
    if frac == 0.0:
        return y_noisy, mask
    gen = torch.Generator().manual_seed(seed)
    n = int(y.numel())
    k = int(round(frac * n))
    idx = torch.randperm(n, generator=gen)[:k]
    offset = torch.randint(1, num_classes, (k,), generator=gen)
    y_noisy[idx] = (y[idx] + offset) % num_classes
    mask[idx] = True
    return y_noisy, mask


def one_hot(y: Tensor, num_classes: int) -> Tensor:
    """``(N,)`` int labels -> ``(N, num_classes)`` float32 one-hot targets."""
    return torch.nn.functional.one_hot(y, num_classes=num_classes).to(torch.float32)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_mnist1d(
    cfg: Mnist1DConfig | None = None,
    *,
    label_noise: float = 0.15,
    noise_seed: int = 0,
    scratch_dir: str | Path | None = None,
    allow_pip: bool = True,
) -> DataBundle:
    """Load MNIST-1D (pip if available, else vendored), apply train label noise, cache to scratch."""
    cfg = cfg or Mnist1DConfig()
    source = "vendored"
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    cache: Path | None = None
    if scratch_dir is not None:
        cache = (
            Path(scratch_dir).expanduser()
            / f"mnist1d_seed{cfg.seed}_n{cfg.n_train}_{cfg.n_test}.npz"
        )
        if cache.exists():
            npz = np.load(cache)
            arrays = (npz["x_train"], npz["y_train"], npz["x_test"], npz["y_test"])
            source = str(npz["source"]) if "source" in npz else "cache"

    if arrays is None:
        if allow_pip:
            try:
                arrays = _load_pip_mnist1d(cfg)
                source = "mnist1d-pip"
            except Exception:  # noqa: BLE001 -- any pip/import failure -> vendored fallback
                arrays = None
        if arrays is None:
            arrays = generate_mnist1d(cfg)
            source = "vendored"
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache,
                x_train=arrays[0], y_train=arrays[1],
                x_test=arrays[2], y_test=arrays[3], source=source,
            )

    x_train = torch.from_numpy(np.ascontiguousarray(arrays[0]))
    y_train = torch.from_numpy(np.ascontiguousarray(arrays[1]))
    x_test = torch.from_numpy(np.ascontiguousarray(arrays[2]))
    y_test = torch.from_numpy(np.ascontiguousarray(arrays[3]))

    y_noisy, mask = add_label_noise(
        y_train, label_noise, num_classes=cfg.num_classes, seed=noise_seed
    )
    return DataBundle(
        x_train=x_train,
        y_train=y_noisy,
        x_test=x_test,
        y_test=y_test,
        y_train_clean=y_train,
        noise_mask=mask,
        num_classes=cfg.num_classes,
        in_dim=int(x_train.shape[1]),
        source=source,
        label_noise=label_noise,
        meta={"n_train": float(x_train.shape[0]), "n_test": float(x_test.shape[0])},
    )


def synthetic_mnist1d(
    *,
    n_train: int = 256,
    n_test: int = 128,
    in_dim: int = FINAL_SEQ_LEN,
    num_classes: int = NUM_CLASSES,
    label_noise: float = 0.15,
    seed: int = 0,
    noise: float = 0.4,
) -> DataBundle:
    """A tiny, learnable, offline stand-in shaped like MNIST-1D (for CI/smoke tests)."""
    gen = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(num_classes, in_dim, generator=gen)

    def make(n: int, sub: int) -> tuple[Tensor, Tensor]:
        g = torch.Generator().manual_seed(sub)
        labels = torch.randint(0, num_classes, (n,), generator=g)
        feats = prototypes[labels] + noise * torch.randn(n, in_dim, generator=g)
        return feats, labels

    x_train, y_train = make(n_train, seed + 1)
    x_test, y_test = make(n_test, seed + 2)
    y_noisy, mask = add_label_noise(y_train, label_noise, num_classes=num_classes, seed=seed)
    return DataBundle(
        x_train=x_train,
        y_train=y_noisy,
        x_test=x_test,
        y_test=y_test,
        y_train_clean=y_train,
        noise_mask=mask,
        num_classes=num_classes,
        in_dim=in_dim,
        source="synthetic",
        label_noise=label_noise,
        meta={"n_train": float(n_train), "n_test": float(n_test)},
    )


__all__ = [
    "DataBundle",
    "FINAL_SEQ_LEN",
    "Mnist1DConfig",
    "NUM_CLASSES",
    "add_label_noise",
    "generate_mnist1d",
    "load_mnist1d",
    "one_hot",
    "synthetic_mnist1d",
]
