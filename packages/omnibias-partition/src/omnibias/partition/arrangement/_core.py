# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hyperplane arrangement geometry (theory 01-03).

A depth-``d`` oblique partition is a tree: ``2**d`` regions. An arrangement of
``n`` hyperplanes in ``R^D`` has up to ``sum_{k=0}^D C(n,k)`` cells. Sampling
returns a **lower bound**, never the complete face lattice.

``beta -> inf`` here is **temperature collapse** (soft indicators hardening),
not founding ``delta -> 0``. The gap is sound, not P vs NP, not theorem-prover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from numpy.typing import NDArray
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import sigmoid_iv
from omnibias.partition._core.weights import sigmoid_np

FloatArray = NDArray[np.float64]

_MAX_N = 12
_MAX_D = 4


def max_cells(n: int, dim: int) -> int:
    """Maximum number of cells of a simple arrangement of ``n`` hyperplanes in ``R^dim``."""
    if n < 0 or dim < 0:
        raise ValueError("n and dim must be non-negative")
    return int(sum(comb(n, k) for k in range(min(n, dim) + 1)))


@dataclass(frozen=True)
class Arrangement:
    """``n`` affine hyperplanes ``w_i . x = offset_i`` in ``R^D``."""

    normals: FloatArray
    offsets: FloatArray

    def __post_init__(self) -> None:
        normals = np.asarray(self.normals, dtype=np.float64)
        offsets = np.asarray(self.offsets, dtype=np.float64).reshape(-1)
        if normals.ndim != 2:
            raise ValueError("normals must have shape (n, D)")
        if offsets.shape[0] != normals.shape[0]:
            raise ValueError("offsets must have length n")
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "offsets", offsets)

    @property
    def n(self) -> int:
        return int(self.normals.shape[0])

    @property
    def dim(self) -> int:
        return int(self.normals.shape[1])


def affine_values(arr: Arrangement, x: FloatArray) -> FloatArray:
    """``W x - t``, shape ``(..., n)``."""
    xv = np.asarray(x, dtype=np.float64)
    if xv.ndim == 1:
        xv = xv.reshape(1, -1)
    return xv @ arr.normals.T - arr.offsets[None, :]


def sign_vector(arr: Arrangement, x: FloatArray, *, tol: float = 0.0) -> FloatArray:
    """Signs in ``{-1, 0, +1}``, shape ``(..., n)``."""
    z = affine_values(arr, x)
    s = np.sign(z)
    if tol > 0.0:
        s[np.abs(z) <= tol] = 0.0
    return s.astype(np.float64)


def realized_cells(
    arr: Arrangement, samples: FloatArray, *, tol: float = 1e-12
) -> tuple[tuple[int, ...], ...]:
    """Sign vectors observed at ``samples``. A lower bound, never completeness."""
    s = sign_vector(arr, samples, tol=tol)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    cells: set[tuple[int, ...]] = set()
    for row in s:
        if np.any(row == 0.0):
            continue
        cells.add(tuple(int(v) for v in row))
    return tuple(sorted(cells))


def tope_graph(cells: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Adjacency over the discovered cells only (Hamming distance 2 on ±1 vectors)."""
    listed = [tuple(int(v) for v in c) for c in cells]
    edges: list[tuple[int, int]] = []
    for i, a in enumerate(listed):
        for j, b in enumerate(listed):
            if j <= i:
                continue
            diffs = sum(1 for u, v in zip(a, b, strict=True) if u != v)
            if diffs == 1:
                edges.append((i, j))
    return tuple(edges)


def _sigmoid(z: FloatArray) -> FloatArray:
    return sigmoid_np(np.asarray(z, dtype=np.float64))


def soft_membership(
    arr: Arrangement,
    x: FloatArray,
    signs: Sequence[int],
    *,
    beta: float,
) -> FloatArray:
    """Soft indicator of one cell. ``beta -> inf`` is temperature collapse."""
    if beta <= 0.0:
        raise ValueError("beta must be > 0 (temperature collapse axis)")
    z = affine_values(arr, x)
    s = np.asarray(signs, dtype=np.float64).reshape(1, -1)
    return np.prod(_sigmoid(beta * s * z), axis=-1)


def margin(arr: Arrangement, x: FloatArray) -> FloatArray:
    """Per-sample minimum absolute affine value."""
    return np.min(np.abs(affine_values(arr, x)), axis=-1)


@dataclass(frozen=True)
class CellGapCertificate:
    """Sound bound on ``|soft - hard|``. Temperature collapse, not founding."""

    beta: float
    bound: float
    measured: float

    @property
    def is_sound(self) -> bool:
        return self.bound + 1e-12 >= self.measured


def certify_cell_gap(
    arr: Arrangement,
    x: FloatArray,
    signs: Sequence[int],
    *,
    beta: float,
) -> CellGapCertificate:
    """Bound ``|soft - 1[cell]|`` from the margin via ``n (1 - sigmoid(beta m))``.

    Each factor is in ``(0,1)``; the product is at least
    ``sigmoid(beta m)^n`` on the true cell and at most
    ``1 - sigmoid(beta m)`` off it (one opposing gate). The bound is
    ``n (1 - sigmoid_iv(beta * m_lo).lo)``, which only widens.
    """
    xv = np.asarray(x, dtype=np.float64)
    if xv.ndim == 1:
        xv = xv.reshape(1, -1)
    z = affine_values(arr, xv)
    s = np.asarray(signs, dtype=np.float64).reshape(1, -1)
    oriented = s * z
    hard = (oriented > 0.0).all(axis=-1).astype(np.float64)
    soft = np.prod(_sigmoid(beta * oriented), axis=-1)
    measured = float(np.max(np.abs(soft - hard)))
    m = np.min(np.abs(z), axis=-1)
    m_lo = float(np.min(m))
    # 1 - sigmoid(beta m) <= 1 / (1 + exp(beta m)); enclose with sigmoid_iv.
    one_minus = Interval.point(1.0) - sigmoid_iv(Interval.point(beta * m_lo))
    bound_iv = Interval.point(float(arr.n)) * one_minus
    return CellGapCertificate(beta=beta, bound=float(bound_iv.hi), measured=measured)


def tree_arrangement(W: FloatArray, t: FloatArray) -> Arrangement:
    """The ``depth`` global splits of an oblivious tree, as an arrangement."""
    return Arrangement(np.asarray(W, dtype=np.float64), np.asarray(t, dtype=np.float64))


def enumerate_cells_vertices(
    arr: Arrangement, *, eps: float = 1e-6
) -> tuple[tuple[int, ...], ...]:
    """Cells found by orthant-perturbing every simple vertex.

    At a vertex ``x = A^{-1} t`` of ``D`` independent planes, the ``2^D``
    adjacent cells are the points ``x + A^{-1} (eps * s)``. A sum-of-normals
    poke is the wrong cone. Several ``eps`` values are unioned so a nearby
    extra plane cannot hide a thin cell.

    Still a discovered set: parallel or concurrent degeneracies yield fewer
    cells than ``max_cells``.
    """
    n, d = arr.n, arr.dim
    if d < 1:
        return ()
    cells: set[tuple[int, ...]] = set()
    scales = (eps * 0.01, eps, eps * 100.0, eps * 1e4)
    for idx in combinations(range(n), d):
        a = arr.normals[list(idx)]
        b = arr.offsets[list(idx)]
        try:
            inv = np.linalg.inv(a)
        except np.linalg.LinAlgError:
            continue
        x = inv @ b
        for bits in range(1 << d):
            local = np.array(
                [1.0 if (bits >> k) & 1 else -1.0 for k in range(d)],
                dtype=np.float64,
            )
            for scale in scales:
                y = x + scale * (inv @ local)
                s = np.sign(y @ arr.normals.T - arr.offsets)
                if np.any(s == 0.0):
                    continue
                cells.add(tuple(int(v) for v in s))
    return tuple(sorted(cells))


def brute_force_cells(
    arr: Arrangement, box: float = 3.0, n_grid: int = 21
) -> tuple[tuple[int, ...], ...]:
    """Grid sampler used as a reference on tiny arrangements (not a completeness claim)."""
    if arr.dim > 3:
        raise ValueError("brute_force_cells is a small-D reference")
    axes = [np.linspace(-box, box, n_grid) for _ in range(arr.dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.reshape(-1) for m in mesh], axis=1)
    return realized_cells(arr, pts)


def general_position_normals(n: int, dim: int, rng: np.random.Generator) -> Arrangement:
    """Random normals, offsets; almost surely simple for the G1 sampler."""
    if n > _MAX_N or dim > _MAX_D:
        raise ValueError(f"G1 tooling caps n<= {_MAX_N}, D<= {_MAX_D}")
    W = rng.normal(size=(n, dim))
    W /= np.linalg.norm(W, axis=1, keepdims=True).clip(1e-12)
    t = rng.normal(size=n) * 0.3
    return Arrangement(W, t)


__all__ = [
    "Arrangement",
    "CellGapCertificate",
    "affine_values",
    "brute_force_cells",
    "certify_cell_gap",
    "enumerate_cells_vertices",
    "general_position_normals",
    "margin",
    "max_cells",
    "realized_cells",
    "sign_vector",
    "soft_membership",
    "tope_graph",
    "tree_arrangement",
]
