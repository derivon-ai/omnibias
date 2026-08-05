# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Backend-agnostic parameter container for a soft partition of unity.

:class:`PartitionParams` holds the raw numpy arrays that fully define a ``depth``-gate soft
partition of ``R^{n_features}``:

* ``W`` -- ``(depth, n_features)`` oblique split directions,
* ``t`` -- ``(depth,)`` split thresholds (the gate is ``sigmoid(beta (W.x - t))``).

The torch twin (:mod:`omnibias.partition.torch.weights`) and the jax twin
(:mod:`omnibias.partition.jax.weights`) both consume this container, so a partition trained
in one backend evaluates / certifies identically in the other.

Terminology: the gate hardens as ``beta -> inf`` -- the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.partition._core.config import PartitionConfig

FloatArray = NDArray[np.float64]


def _as_rng(seed: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


@dataclass
class PartitionParams:
    r"""The numpy arrays defining a soft partition (see the module docstring)."""

    config: PartitionConfig
    W: FloatArray
    t: FloatArray

    def __post_init__(self) -> None:
        cfg = self.config
        self.W = np.asarray(self.W, dtype=np.float64).reshape(cfg.depth, cfg.n_features)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(cfg.depth)

    @property
    def n_features(self) -> int:
        return int(self.config.n_features)

    @property
    def depth(self) -> int:
        return int(self.config.depth)

    @property
    def n_regions(self) -> int:
        return int(self.config.n_regions)

    def copy(self) -> PartitionParams:
        return PartitionParams(self.config, self.W.copy(), self.t.copy())


def region_code_matrix(depth: int) -> FloatArray:
    r"""``(2**depth, depth)`` matrix of region bit-codes in ``{0.0, 1.0}``.

    Row ``l`` is the base-2 expansion of the region index ``l`` (bit ``j`` = gate ``j``);
    ``1`` means "gate ``j`` fired" (``W.x > t``). Shared by every backend so the region
    ordering is identical.
    """
    L = 1 << depth
    codes = np.zeros((L, depth), dtype=np.float64)
    for region in range(L):
        for j in range(depth):
            codes[region, j] = float((region >> j) & 1)
    return codes


def init_params(
    config: PartitionConfig,
    rng: np.random.Generator | int | None = None,
    *,
    weight_scale: float | None = None,
) -> PartitionParams:
    r"""Small random initialisation of the ``depth`` split gates.

    ``oblique`` / ``sparse`` draw dense directions ``N(0, weight_scale**2)`` (default
    ``1/sqrt(n_features)``); ``axis`` assigns each gate a single feature with a unit
    direction (an interpretable ``x[f] > t`` split). Thresholds start at ``0``.
    """
    gen = _as_rng(rng if rng is not None else config.seed)
    D, d = config.depth, config.n_features
    ws = weight_scale if weight_scale is not None else 1.0 / np.sqrt(d)
    if config.split_kind == "axis":
        W = np.zeros((D, d), dtype=np.float64)
        for j in range(D):
            W[j, int(gen.integers(0, d))] = 1.0
    else:
        W = gen.standard_normal((D, d)) * ws
    t = np.zeros(D, dtype=np.float64)
    return PartitionParams(config, W, t)


__all__ = ["FloatArray", "PartitionParams", "init_params", "region_code_matrix"]
