# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Heterogeneous multi-pack Birkhoff support algebra (theory 01-01).

A pack of size ``K = order + 1`` at mean ``mu`` collapses under the founding
bias collapse ``delta -> 0`` to ``sigma^(order)(z + mu)``. Several packs at
distinct means give a scattered Birkhoff sample of ``sigma`` along one
transverse coordinate:

    F(z) -> sum_g c_g * sigma^(n_g)(z + mu_g)

This module is pure Python: specs, the Polya screen, the incidence matrix,
and a numerical rank test for poisedness. Tensor evaluation lives in the
torch / jax twins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True)
class PackSpec:
    """One pack: derivative order ``n``, mean ``mu``, outer weight ``c``."""

    order: int
    mean: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if int(self.order) < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        if not np.isfinite(self.mean):
            raise ValueError(f"mean must be finite, got {self.mean}")
        if not np.isfinite(self.weight):
            raise ValueError(f"weight must be finite, got {self.weight}")
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "weight", float(self.weight))

    @property
    def arity(self) -> int:
        """Number of biases in the pack: ``K = order + 1``."""
        return self.order + 1


@dataclass(frozen=True)
class MultiPackSpec:
    """An unordered collection of packs forming a Birkhoff support."""

    packs: tuple[PackSpec, ...]

    def __post_init__(self) -> None:
        packs = tuple(self.packs)
        if not packs:
            raise ValueError("MultiPackSpec requires at least one pack")
        object.__setattr__(self, "packs", packs)

    @classmethod
    def from_packs(cls, packs: Sequence[PackSpec]) -> MultiPackSpec:
        return cls(tuple(packs))

    @property
    def distinct_means(self) -> tuple[float, ...]:
        """Means in first-appearance order (exact float identity)."""
        seen: list[float] = []
        for p in self.packs:
            if p.mean not in seen:
                seen.append(p.mean)
        return tuple(seen)

    @property
    def max_order(self) -> int:
        return max(p.order for p in self.packs)

    @property
    def n_conditions(self) -> int:
        """Number of (mean, order) incidence ones (duplicate pairs count once)."""
        return sum(sum(row) for row in incidence_matrix(self))


def incidence_matrix(spec: MultiPackSpec) -> tuple[tuple[int, ...], ...]:
    """Incidence matrix ``E``: rows = distinct means, columns = orders.

    ``E[i][j] = 1`` iff some pack at mean ``i`` requests derivative order ``j``.
    Column count is ``max_order + 1``.
    """
    means = spec.distinct_means
    n_cols = spec.max_order + 1
    mean_index = {m: i for i, m in enumerate(means)}
    rows = [[0] * n_cols for _ in means]
    for p in spec.packs:
        rows[mean_index[p.mean]][p.order] = 1
    return tuple(tuple(r) for r in rows)


def polya_condition(spec: MultiPackSpec) -> bool:
    """Necessary Polya screen on the incidence matrix.

    For every ``j``, the sum over the first ``j + 1`` columns is at least
    ``j + 1``. Failure implies the scheme cannot be poised.
    """
    E = incidence_matrix(spec)
    if not E:
        return False
    n_cols = len(E[0])
    for j in range(n_cols):
        total = sum(E[i][k] for i in range(len(E)) for k in range(j + 1))
        if total < j + 1:
            return False
    return True


def _falling_factorial(j: int, n: int) -> float:
    """``j (j-1) ... (j-n+1)``; ``1`` when ``n == 0``."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return 1.0
    if j < n:
        return 0.0
    out = 1.0
    for t in range(n):
        out *= float(j - t)
    return out


def _birkhoff_vandermonde(spec: MultiPackSpec) -> np.ndarray:
    r"""Confluent / Birkhoff Vandermonde for the incidence conditions.

    One row per selected ``(mean, order)`` pair (row-major over means, then
    increasing order). Columns are monomial degrees ``0 .. m-1`` where ``m``
    is the number of conditions. Entry ``(r, j)`` is the ``n``-th derivative
    of ``x^j`` at ``mu``, i.e. ``j^{\underline{n}} mu^{j-n}`` when ``j >= n``.
    """
    E = incidence_matrix(spec)
    means = spec.distinct_means
    conditions: list[tuple[float, int]] = []
    for i, mu in enumerate(means):
        for n, flag in enumerate(E[i]):
            if flag:
                conditions.append((mu, n))
    m = len(conditions)
    if m == 0:
        return np.zeros((0, 0), dtype=np.float64)
    A = np.zeros((m, m), dtype=np.float64)
    for r, (mu, n) in enumerate(conditions):
        for j in range(m):
            if j < n:
                A[r, j] = 0.0
            else:
                A[r, j] = _falling_factorial(j, n) * (mu ** (j - n))
    return A


def is_poised(spec: MultiPackSpec, *, tol: float = 1e-10) -> bool | None:
    """Numerical poisedness test for the Birkhoff support.

    Returns
    -------
    bool | None
        ``False`` if Polya fails or the Vandermonde is clearly rank-deficient.
        ``True`` if the matrix is clearly full rank.
        ``None`` when the rank test is inconclusive (singular value near
        ``tol``); callers must not read ``None`` as ``True``.
    """
    if not polya_condition(spec):
        return False
    A = _birkhoff_vandermonde(spec)
    m = A.shape[0]
    if m == 0:
        return False
    # Exact 1x1 nonzero is poised; empty already handled.
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0:
        return False
    smax = float(s[0])
    smin = float(s[-1])
    if smax == 0.0:
        return False
    # Clear full rank.
    if smin > float(tol) * max(smax, 1.0):
        return True
    # Clear deficiency.
    if smin < float(tol) * float(tol) * max(smax, 1.0):
        return False
    # Near the threshold: inconclusive.
    return None


def central_stencil_weights(order: int, delta: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Central finite-difference biases and signs for a single pack.

    Biases: ``mu``-relative offsets ``(k - (K+1)/2) * delta`` for ``k = 1..K``.
    Signs: ``(-1)^{K-k} C(K-1, k-1) / delta^{K-1}`` with ``K = order + 1``.

    Returns offsets (add ``mu`` for absolute biases) and signs. Used by the
    G2 collapse check; the closed-form path never divides by ``delta``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    K = order + 1
    offsets = tuple((k - (K + 1) / 2.0) * float(delta) for k in range(1, K + 1))
    scale = float(delta) ** (K - 1)
    signs = tuple(
        ((-1.0) ** (K - k)) * float(comb(K - 1, k - 1)) / scale for k in range(1, K + 1)
    )
    return offsets, signs


__all__ = [
    "MultiPackSpec",
    "PackSpec",
    "central_stencil_weights",
    "incidence_matrix",
    "is_poised",
    "polya_condition",
]
