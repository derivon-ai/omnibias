# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""1-D hierarchical pack tree (theory 02-07).

Offset axis only. Far-field is a truncation with a bound. ``eta = 0``
reduces to the dense sum bit-identically (same summands, same order).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.polynomials import tanh_polynomial_coeffs
from omnibias.core.verified.interval import Interval


@dataclass(frozen=True)
class Cluster:
    centre: float
    radius: float
    members: tuple[int, ...]
    children: tuple[Cluster, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children


def build_pack_tree(offsets: Sequence[float], *, leaf_size: int = 8) -> Cluster:
    """Binary tree on a sorted copy; ``members`` index the original sequence."""
    if leaf_size < 1:
        raise ValueError("leaf_size must be >= 1")
    offs = tuple(float(v) for v in offsets)
    order = tuple(sorted(range(len(offs)), key=lambda i: offs[i]))

    def rec(idx: tuple[int, ...]) -> Cluster:
        vals = [offs[i] for i in idx]
        lo, hi = min(vals), max(vals)
        centre = 0.5 * (lo + hi)
        radius = 0.5 * (hi - lo)
        if len(idx) <= leaf_size or radius == 0.0:
            return Cluster(centre, radius, idx)
        mid = len(idx) // 2
        left = rec(idx[:mid])
        right = rec(idx[mid:])
        return Cluster(centre, radius, idx, (left, right))

    return rec(order)


def _horner(coeffs: tuple[float, ...], t: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * t + c
    return acc


def sigma_n_tanh(order: int, z: float) -> float:
    t = math.tanh(z)
    return _horner(tanh_polynomial_coeffs(order), t)


def dense_scan(
    z: float,
    offsets: Sequence[float],
    weights: Sequence[float],
    orders: Sequence[int],
    *,
    base: str = "tanh",
) -> float:
    """Dense sum in **original offset order** (the eta=0 reference)."""
    if base != "tanh":
        raise ValueError("hierarchy G1 path is tanh-only")
    acc = 0.0
    for b, w, n in zip(offsets, weights, orders, strict=True):
        acc += float(w) * sigma_n_tanh(int(n), z - float(b))
    return acc


def multipole_moments(
    cluster: Cluster,
    offsets: Sequence[float],
    weights: Sequence[float],
    orders: Sequence[int],
    *,
    p: int,
) -> tuple[float, ...]:
    """``sum_j w_j (c - b_j)^k / k!`` for ``k = 0 .. p`` (order-0 source).

    Higher template orders are absorbed into the far-field bound, not the
    moment tensor, so the far evaluation is a truncated Taylor of
    ``sigma^{(n)}`` at the cluster centre.
    """
    if p < 0:
        raise ValueError("p must be >= 0")
    c = cluster.centre
    moms = [0.0] * (p + 1)
    fact = 1.0
    # mixed orders: expand each source as Taylor of sigma^{(n_j)}
    # moments here are geometric only; caller pairs with derivatives at centre.
    for j in cluster.members:
        db = c - float(offsets[j])
        pow_db = 1.0
        fact = 1.0
        w = float(weights[j])
        for k in range(p + 1):
            if k:
                fact *= k
            moms[k] += w * pow_db / fact
            pow_db *= db
    return tuple(moms)


def _deriv_bound(order: int) -> Interval:
    """Loose enclosure of ``sup |tanh^{(m)}|``; Euler numbers grow like m!."""
    m = int(order)
    if m < 0:
        raise ValueError("order must be >= 0")
    # |E_m| <= m! * 2^m  (crude, sound for the bound).
    cap = 1.0
    for i in range(1, m + 1):
        cap *= float(i) * 2.0
    return Interval(-cap, cap)


def truncation_bound(
    cluster: Cluster,
    *,
    distance: float,
    p: int,
    deriv_bound: Interval | None = None,
    n_members_weight: float = 1.0,
) -> Interval:
    """Lagrange remainder ``rho^{p+1}/(p+1)! * B * W``. Never undercovers."""
    if distance <= cluster.radius:
        raise ValueError("truncation_bound requires a far cluster (distance > radius)")
    rho = cluster.radius
    fact = 1.0
    for i in range(1, p + 2):
        fact *= float(i)
    scale = (rho ** (p + 1)) / fact
    b = deriv_bound if deriv_bound is not None else _deriv_bound(p + 1)
    w = abs(float(n_members_weight))
    cap = scale * max(abs(b.lo), abs(b.hi)) * w
    return Interval(-cap, cap)


def separation_for_accuracy(*, radius: float, p: int, target: float) -> float:
    """``eta`` such that ``(eta^{p+1})/(p+1)!`` meets ``target`` for unit bound."""
    if target <= 0.0:
        raise ValueError("target must be positive")
    fact = 1.0
    for i in range(1, p + 2):
        fact *= float(i)
    # (radius/R)^{p+1} / (p+1)! <= target  => R >= radius / (target * fact)^{1/(p+1)}
    root = (target * fact) ** (1.0 / (p + 1))
    if root <= 0.0:
        return float("inf")
    return float(radius / root)


def far_eval(
    z: float,
    cluster: Cluster,
    offsets: Sequence[float],
    weights: Sequence[float],
    orders: Sequence[int],
    *,
    p: int,
    base: str = "tanh",
) -> float:
    """Taylor of each source about ``cluster.centre``, truncated at ``p``."""
    if base != "tanh":
        raise ValueError("hierarchy far path is tanh-only")
    c = cluster.centre
    acc = 0.0
    for j in cluster.members:
        n0 = int(orders[j])
        w = float(weights[j])
        db = c - float(offsets[j])
        # sigma^{(n)}(z-b) = sum_{k=0}^p sigma^{(n+k)}(z-c) (c-b)^k / k!
        term = 0.0
        pow_db = 1.0
        fact = 1.0
        for k in range(p + 1):
            if k:
                fact *= k
            term += sigma_n_tanh(n0 + k, z - c) * pow_db / fact
            pow_db *= db
        acc += w * term
    return acc


def hierarchical_value(
    z: float,
    tree: Cluster,
    offsets: Sequence[float],
    weights: Sequence[float],
    orders: Sequence[int],
    *,
    p: int = 6,
    eta: float = 0.5,
    base: str = "tanh",
) -> float:
    """Near/far split. ``eta = 0`` is the dense path (original order)."""
    if eta <= 0.0:
        return dense_scan(z, offsets, weights, orders, base=base)

    def walk(node: Cluster) -> float:
        dist = abs(z - node.centre)
        if (not node.is_leaf) and node.radius <= eta * max(dist, 1e-18):
            return far_eval(z, node, offsets, weights, orders, p=p, base=base)
        if node.is_leaf:
            acc = 0.0
            for j in node.members:
                acc += float(weights[j]) * sigma_n_tanh(int(orders[j]), z - float(offsets[j]))
            return acc
        return sum(walk(ch) for ch in node.children)

    return walk(tree)


__all__ = [
    "Cluster",
    "build_pack_tree",
    "dense_scan",
    "far_eval",
    "hierarchical_value",
    "multipole_moments",
    "separation_for_accuracy",
    "sigma_n_tanh",
    "truncation_bound",
]
