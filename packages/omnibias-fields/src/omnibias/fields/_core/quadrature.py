# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic quadrature rules for field integration.

A :class:`QuadratureSpec` bundles a set of collocation nodes and weights so a
field can be integrated over a box domain by evaluating it at the nodes and
contracting with the weights. The nodes and weights are generated once with
numpy in float64, so the torch and jax ``integrate`` ops produce *bit-identical*
results (they convert the same numpy arrays to their tensor type).

This module is pure Python + numpy (no torch / jax). numpy is treated as a
numerical primitive, not a deep-learning backend, consistent with the
``omnibias-fields`` package boundary (the stricter "no numpy" rule applies only
to ``omnibias-core``).

Rules
-----
- :func:`gauss_legendre` -- tensor-product Gauss-Legendre on a box; exact for
  polynomials up to degree ``2 n - 1`` per axis.
- :func:`gauss_hermite` -- Gauss-Hermite expectation under a (diagonal) Gaussian
  ``N(mean, scale**2)``; ``sum(weights) == 1``.
- :func:`monte_carlo` -- seeded uniform Monte-Carlo over a box.
- :func:`tensor_product` -- outer product of one-dimensional rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class QuadratureSpec:
    """A set of integration nodes and weights over a box domain.

    Parameters
    ----------
    rule
        Human-readable rule name (``"gauss_legendre"``, ``"gauss_hermite"``,
        ``"monte_carlo"``, ``"tensor_product"``).
    nodes
        Float64 array of shape ``(n_nodes, dim)`` -- the collocation points.
    weights
        Float64 array of shape ``(n_nodes,)`` -- the quadrature weights.
    bounds
        Optional ``(lo, hi)`` pair per axis recording the integration box.

    Notes
    -----
    The dataclass is frozen for hashable intent but uses ``eq=False`` because
    its fields are numpy arrays (elementwise ``==`` is not a valid ``__eq__``).
    """

    rule: str
    nodes: NDArray[np.float64]
    weights: NDArray[np.float64]
    bounds: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2:
            raise ValueError(
                f"nodes must be 2D (n_nodes, dim), got shape {self.nodes.shape}"
            )
        if self.weights.ndim != 1 or self.weights.shape[0] != self.nodes.shape[0]:
            raise ValueError(
                "weights must be 1D with one entry per node; "
                f"got weights {self.weights.shape}, nodes {self.nodes.shape}"
            )

    @property
    def dim(self) -> int:
        """Spatial dimension of the integration domain."""
        return int(self.nodes.shape[1])

    @property
    def n_nodes(self) -> int:
        """Number of collocation nodes."""
        return int(self.nodes.shape[0])

    def __repr__(self) -> str:
        return (
            f"QuadratureSpec(rule={self.rule!r}, dim={self.dim}, "
            f"n_nodes={self.n_nodes})"
        )


def _as_bounds(bounds: object, dim: int) -> tuple[tuple[float, float], ...]:
    seq = list(bounds)  # type: ignore[call-overload]
    if len(seq) != dim:
        raise ValueError(f"expected {dim} (lo, hi) bounds, got {len(seq)}")
    out: list[tuple[float, float]] = []
    for lo, hi in seq:
        lo_f, hi_f = float(lo), float(hi)
        if not (lo_f < hi_f):
            raise ValueError(f"each bound needs lo < hi, got ({lo_f}, {hi_f})")
        out.append((lo_f, hi_f))
    return tuple(out)


def _gl_1d(n: int, lo: float, hi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if n < 1:
        raise ValueError(f"points_per_axis entries must be >= 1, got {n}")
    t, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    x = half * t + mid
    return x.astype(np.float64), (half * w).astype(np.float64)


def _tensor(
    per_axis: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Outer-product 1D (nodes, weights) tuples into a full grid."""
    node_axes = [p[0] for p in per_axis]
    weight_axes = [p[1] for p in per_axis]
    grid = list(product(*node_axes))
    wgrid = list(product(*weight_axes))
    nodes = np.array(grid, dtype=np.float64).reshape(len(grid), len(per_axis))
    weights = np.array([float(np.prod(w)) for w in wgrid], dtype=np.float64)
    return nodes, weights


def gauss_legendre(
    bounds: object,
    points_per_axis: int | tuple[int, ...],
) -> QuadratureSpec:
    """Tensor-product Gauss-Legendre quadrature over a box.

    Parameters
    ----------
    bounds
        Sequence of ``(lo, hi)`` pairs, one per axis.
    points_per_axis
        Number of nodes per axis (an int broadcasts to every axis).

    Returns
    -------
    QuadratureSpec
        Exact for polynomials of degree ``<= 2 n - 1`` along each axis.
    """
    b = list(bounds)  # type: ignore[call-overload]
    dim = len(b)
    bnds = _as_bounds(b, dim)
    if isinstance(points_per_axis, int):
        pts = (points_per_axis,) * dim
    else:
        pts = tuple(points_per_axis)
    if len(pts) != dim:
        raise ValueError(f"points_per_axis must have length {dim}, got {len(pts)}")
    per_axis = [_gl_1d(pts[i], bnds[i][0], bnds[i][1]) for i in range(dim)]
    nodes, weights = _tensor(per_axis)
    return QuadratureSpec("gauss_legendre", nodes, weights, bnds)


def gauss_hermite(
    points_per_axis: int | tuple[int, ...],
    *,
    mean: float | tuple[float, ...] = 0.0,
    scale: float | tuple[float, ...] = 1.0,
    dim: int | None = None,
) -> QuadratureSpec:
    r"""Gauss-Hermite expectation rule under a diagonal Gaussian.

    The weights are normalised so that
    :math:`\sum_q w_q\,f(x_q) \approx \mathbb{E}_{x\sim\mathcal N(\mu,\Sigma)}[f]`
    with diagonal :math:`\Sigma = \mathrm{diag}(\text{scale}^2)`; in particular
    ``sum(weights) == 1``.

    Parameters
    ----------
    points_per_axis
        Number of Gauss-Hermite nodes per axis (int broadcasts).
    mean, scale
        Per-axis Gaussian mean and standard deviation (scalars broadcast).
    dim
        Dimension; inferred from the longest of the tuple arguments if omitted.
    """
    def _infer_dim() -> int:
        for v in (points_per_axis, mean, scale):
            if isinstance(v, tuple):
                return len(v)
        return 1

    d = dim if dim is not None else _infer_dim()
    pts = (points_per_axis,) * d if isinstance(points_per_axis, int) else tuple(points_per_axis)
    mus = (float(mean),) * d if not isinstance(mean, tuple) else tuple(float(m) for m in mean)
    sds = (float(scale),) * d if not isinstance(scale, tuple) else tuple(float(s) for s in scale)
    if not (len(pts) == len(mus) == len(sds) == d):
        raise ValueError("points_per_axis / mean / scale must agree in length")
    per_axis: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for i in range(d):
        if pts[i] < 1:
            raise ValueError(f"points_per_axis entries must be >= 1, got {pts[i]}")
        if sds[i] <= 0.0:
            raise ValueError(f"scale entries must be > 0, got {sds[i]}")
        t, w = np.polynomial.hermite.hermgauss(pts[i])
        x = mus[i] + np.sqrt(2.0) * sds[i] * t
        wn = w / np.sqrt(np.pi)
        per_axis.append((x.astype(np.float64), wn.astype(np.float64)))
    nodes, weights = _tensor(per_axis)
    return QuadratureSpec("gauss_hermite", nodes, weights, None)


def monte_carlo(
    bounds: object,
    n: int,
    *,
    seed: int = 0,
) -> QuadratureSpec:
    """Seeded uniform Monte-Carlo quadrature over a box.

    Weights are the constant box volume divided by ``n`` (so the estimate is the
    sample mean times the volume). Reproducible given ``seed``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    b = list(bounds)  # type: ignore[call-overload]
    dim = len(b)
    bnds = _as_bounds(b, dim)
    rng = np.random.default_rng(seed)
    los = np.array([lo for lo, _ in bnds], dtype=np.float64)
    his = np.array([hi for _, hi in bnds], dtype=np.float64)
    u = rng.random((n, dim))
    nodes = (los + u * (his - los)).astype(np.float64)
    volume = float(np.prod(his - los))
    weights = np.full((n,), volume / n, dtype=np.float64)
    return QuadratureSpec("monte_carlo", nodes, weights, bnds)


def tensor_product(*rules_1d: QuadratureSpec) -> QuadratureSpec:
    """Outer-product several one-dimensional rules into a multi-dim rule."""
    if not rules_1d:
        raise ValueError("tensor_product requires at least one rule")
    for r in rules_1d:
        if r.dim != 1:
            raise ValueError(f"every input rule must be 1D, got dim={r.dim}")
    per_axis = [(r.nodes[:, 0], r.weights) for r in rules_1d]
    nodes, weights = _tensor(per_axis)
    return QuadratureSpec("tensor_product", nodes, weights, None)


__all__ = [
    "QuadratureSpec",
    "gauss_hermite",
    "gauss_legendre",
    "monte_carlo",
    "tensor_product",
]
