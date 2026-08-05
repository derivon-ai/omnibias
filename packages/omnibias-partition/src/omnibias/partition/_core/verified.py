# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sound (outward-rounded) enclosures of the soft partition weights and their soft->hard gap.

These are the always-available certificate primitives -- they import only
:mod:`omnibias.core.verified` (no torch / jax), so they run in the core test job and
certify a :class:`~omnibias.partition._core.params.PartitionParams` of **any depth**. The
gate is enclosed by the rigorous :func:`omnibias.core.verified.transcend.sigmoid_iv`;
products and sums use outward-rounded :class:`~omnibias.core.verified.Interval` arithmetic,
so every bound genuinely encloses the true value (a looser bound only widens the certified
gap).

* :func:`interval_weight_bounds` -- an enclosure of each region weight over an input box.
* :func:`weight_rounding_gap` -- a sound per-sample bound on the L1 soft->hard partition
  gap ``sum_l |w_soft_l - w_hard_l|`` as ``beta -> inf`` (returned with the measured value
  so a caller can self-check ``bound >= measured``).
* :func:`gibbs_gap_bound` -- the closed-form ``log(n_regions)/beta`` Gibbs-to-Dirac scale.

Terminology: the ``beta -> inf`` gate hardening is the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import math

import numpy as np
from omnibias.core.verified import Interval
from omnibias.core.verified.transcend import sigmoid_iv
from omnibias.partition._core.params import FloatArray, PartitionParams, region_code_matrix
from omnibias.partition._core.weights import partition_weights, sigmoid_np


def normalize_box(box: object, d: int) -> list[Interval]:
    r"""Coerce a feature hyper-box to a list of ``d`` :class:`Interval`.

    Accepts a ``(2, d)`` ``[[lo...], [hi...]]`` array (as ``np.stack([X.min(0), X.max(0)])``
    yields), a ``(d, 2)`` array of ``[lo, hi]`` rows, or a sequence of ``(lo, hi)`` pairs.
    """
    if isinstance(box, list | tuple) and len(box) == d and all(
        hasattr(p, "__len__") and len(p) == 2 for p in box
    ):
        return [Interval(float(lo), float(hi)) for lo, hi in box]
    arr = np.asarray(box, dtype=np.float64)
    if arr.shape == (2, d):
        return [Interval(float(arr[0, f]), float(arr[1, f])) for f in range(d)]
    if arr.shape == (d, 2):
        return [Interval(float(arr[f, 0]), float(arr[f, 1])) for f in range(d)]
    raise ValueError(f"cannot interpret feature box of shape {arr.shape} for d={d}")


def _gate_intervals(params: PartitionParams, x_iv: list[Interval], beta: float) -> list[Interval]:
    r"""Gate enclosures ``g[j]`` over the box (one :class:`Interval` per gate)."""
    W, t = params.W, params.t
    gates: list[Interval] = []
    for j in range(params.depth):
        z = Interval.point(-float(t[j]))
        for f in range(params.n_features):
            z = z + float(W[j, f]) * x_iv[f]
        gates.append(sigmoid_iv(beta * z))
    return gates


def interval_weight_bounds(params: PartitionParams, box: object, beta: float) -> tuple[Interval, ...]:
    r"""Rigorous enclosure of each region weight ``w_l`` over the input hyper-box.

    ``bounds[l].lo`` is a certified lower bound on region ``l``'s membership over the box
    (e.g. ``>= 0.9`` proves the box lies essentially inside region ``l``).
    """
    d, D = params.n_features, params.depth
    x_iv = normalize_box(box, d)
    gates = _gate_intervals(params, x_iv, beta)
    codes = region_code_matrix(D)
    out: list[Interval] = []
    for region in range(1 << D):
        membership = Interval.point(1.0)
        for j in range(D):
            gj = gates[j]
            membership = membership * (gj if codes[region, j] > 0.5 else (1.0 - gj))
        out.append(membership)
    return tuple(out)


def weight_rounding_gap(
    params: PartitionParams, X: FloatArray, beta: float
) -> tuple[FloatArray, FloatArray]:
    r"""Sound per-sample bound on the L1 soft->hard partition gap, plus the measured value.

    The region routing is a product of independent Bernoulli gates. On the hard side each
    gate ``j`` disagrees with its crisp value with probability at most
    ``e_j = sigmoid(-beta |z_j|)``, so the soft mass on the crisp region is at least
    ``prod_j (1 - e_j) >= 1 - sum_j e_j``. The L1 distance to the one-hot hard weights is
    therefore ``2 (1 - w_soft[hard]) <= 2 min(1, sum_j e_j)`` -- a certified per-sample
    bound. Returned with the actually-measured L1 gap so callers can self-check
    ``bound >= measured``.
    """
    Xv = np.asarray(X, dtype=np.float64)
    z = np.abs(Xv @ params.W.T - params.t[None, :])  # (n, depth)
    gate_err = sigmoid_np(-beta * z)  # (n, depth) == e_j
    bound = 2.0 * np.minimum(1.0, np.sum(gate_err, axis=1))  # (n,)

    P_soft = partition_weights(params, Xv, beta)  # (n, L)
    from omnibias.partition._core.weights import hard_weights

    P_hard = hard_weights(params, Xv)  # (n, L)
    measured = np.sum(np.abs(P_soft - P_hard), axis=1)  # (n,)
    return bound, measured


def gibbs_gap_bound(n_regions: int, beta: float) -> float:
    r"""The closed-form ``log(n_regions) / beta`` Gibbs-to-Dirac collapse scale.

    A softmax-flavoured, split-geometry-independent reference for how fast the soft
    partition concentrates as ``beta -> inf`` (mirrors the ``log(N)/beta`` selection gap in
    :mod:`omnibias.struct`). Reported alongside the operative sound L1 bound.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    return math.log(max(n_regions, 1)) / beta


__all__ = [
    "gibbs_gap_bound",
    "interval_weight_bounds",
    "normalize_box",
    "weight_rounding_gap",
]
