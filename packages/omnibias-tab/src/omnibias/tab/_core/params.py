# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic parameter container for a soft decision-tree ensemble.

:class:`TabParams` holds the raw numpy arrays that fully define an ensemble of
``n_trees`` oblivious soft trees of ``depth`` gates:

* ``W`` -- ``(n_trees, depth, n_features)`` oblique split directions,
* ``t`` -- ``(n_trees, depth)`` split thresholds (the gate is ``sigmoid(beta (W.x - t))``),
* ``leaves`` -- ``(n_trees, 2**depth, n_outputs)`` leaf values,
* ``b0`` -- ``(n_outputs,)`` global output bias.

The torch module (:mod:`omnibias.tab.torch.model`) and the jax twin
(:mod:`omnibias.tab.jax.model`) both convert to / from this container, so a model
trained in one backend certifies / evaluates identically in the other.

Terminology: the gate hardens as ``beta -> inf`` -- the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.tab._core.config import SoftTreeConfig

FloatArray = NDArray[np.float64]


def _as_rng(seed: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


@dataclass
class TabParams:
    r"""The numpy arrays defining a soft-tree ensemble (see the module docstring)."""

    config: SoftTreeConfig
    W: FloatArray
    t: FloatArray
    leaves: FloatArray
    b0: FloatArray

    def __post_init__(self) -> None:
        cfg = self.config
        T, D, d, L, k = cfg.n_trees, cfg.depth, cfg.n_features, cfg.n_leaves, cfg.n_outputs
        self.W = np.asarray(self.W, dtype=np.float64).reshape(T, D, d)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(T, D)
        self.leaves = np.asarray(self.leaves, dtype=np.float64).reshape(T, L, k)
        self.b0 = np.asarray(self.b0, dtype=np.float64).reshape(k)

    # ----- convenience shape accessors ---------------------------------- #
    @property
    def n_features(self) -> int:
        return int(self.config.n_features)

    @property
    def n_outputs(self) -> int:
        return int(self.config.n_outputs)

    @property
    def depth(self) -> int:
        return int(self.config.depth)

    def copy(self) -> TabParams:
        return TabParams(
            self.config, self.W.copy(), self.t.copy(), self.leaves.copy(), self.b0.copy()
        )


def leaf_code_matrix(depth: int) -> FloatArray:
    r"""``(2**depth, depth)`` matrix of leaf bit-codes in ``{0.0, 1.0}``.

    Row ``l`` is the base-2 expansion of the leaf index ``l`` (bit ``j`` = gate ``j``);
    ``1`` means "the gate fired" (``W.x > t``). Shared by every backend so the leaf
    ordering is identical.
    """
    L = 1 << depth
    codes = np.zeros((L, depth), dtype=np.float64)
    for leaf in range(L):
        for j in range(depth):
            codes[leaf, j] = float((leaf >> j) & 1)
    return codes


def init_params(
    config: SoftTreeConfig,
    rng: np.random.Generator | int | None = None,
    *,
    weight_scale: float | None = None,
    leaf_scale: float = 0.1,
) -> TabParams:
    r"""Small random initialisation.

    ``W`` is drawn ``N(0, weight_scale**2)`` (default ``1/sqrt(n_features)`` so an oblique
    projection has unit-ish variance), thresholds start at ``0``, leaves are small so the
    initial logits sit near ``0`` (``p ~ 0.5`` for classification), and ``b0 = 0``.
    """
    gen = _as_rng(rng if rng is not None else config.seed)
    T, D, d, L, k = (
        config.n_trees,
        config.depth,
        config.n_features,
        config.n_leaves,
        config.n_outputs,
    )
    ws = weight_scale if weight_scale is not None else 1.0 / np.sqrt(d)
    W = gen.standard_normal((T, D, d)) * ws
    t = np.zeros((T, D), dtype=np.float64)
    leaves = gen.standard_normal((T, L, k)) * leaf_scale
    b0 = np.zeros(k, dtype=np.float64)
    return TabParams(config, W, t, leaves, b0)


__all__ = ["FloatArray", "TabParams", "init_params", "leaf_code_matrix"]
