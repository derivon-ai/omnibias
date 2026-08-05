# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Sound (outward-rounded) interval enclosures of the soft-tree forward and its Jacobian.

These are the always-available certificate primitives -- they import only
:mod:`omnibias.core.verified` (no torch / jax / verify), so they run in the core test job
and certify a :class:`~omnibias.tab._core.params.TabParams` of **any depth**. The gate is
enclosed by the rigorous :func:`omnibias.core.verified.transcend.sigmoid_iv`; products and
sums use outward-rounded :class:`~omnibias.core.verified.Interval` arithmetic, so every
bound genuinely encloses the true value (a looser bound only widens the certified gap).

* :func:`interval_output_bounds` -- an enclosure of each output over an input hyper-box.
* :func:`interval_jacobian` -- an enclosure of ``dF_k / dx_f`` (forward-mode) over the box,
  from which certified monotonicity and Lipschitz bounds follow.
* :func:`rounding_gap` -- a sound per-sample bound on ``|F_soft - F_hard|`` as
  ``beta -> inf`` (the certified train-soft / deploy-hard gap), via the total-variation
  subadditivity of the leaf-routing product distribution.

Terminology: the ``beta -> inf`` gate hardening is the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import math

import numpy as np
from omnibias.core.verified import Interval
from omnibias.core.verified.transcend import sigmoid_iv
from omnibias.tab._core.forward import sigmoid_np
from omnibias.tab._core.params import FloatArray, TabParams, leaf_code_matrix


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


def _gate_intervals(params: TabParams, x_iv: list[Interval], beta: float) -> list[list[Interval]]:
    r"""Gate enclosures ``g[m][j]`` over the box (one :class:`Interval` per tree/level)."""
    T, D = params.config.n_trees, params.depth
    W, t = params.W, params.t
    gates: list[list[Interval]] = []
    for m in range(T):
        row: list[Interval] = []
        for j in range(D):
            z = Interval.point(-float(t[m, j]))
            for f in range(params.n_features):
                z = z + float(W[m, j, f]) * x_iv[f]
            row.append(sigmoid_iv(beta * z))
        gates.append(row)
    return gates


def interval_output_bounds(params: TabParams, box: object, beta: float) -> tuple[Interval, ...]:
    r"""Rigorous enclosure of each output ``F_k`` over the input hyper-box."""
    d, T, D, k = params.n_features, params.config.n_trees, params.depth, params.n_outputs
    x_iv = normalize_box(box, d)
    gates = _gate_intervals(params, x_iv, beta)
    codes = leaf_code_matrix(D)
    out = [Interval.point(float(params.b0[c])) for c in range(k)]
    for m in range(T):
        for leaf in range(1 << D):
            membership = Interval.point(1.0)
            for j in range(D):
                gj = gates[m][j]
                membership = membership * (gj if codes[leaf, j] > 0.5 else (1.0 - gj))
            for c in range(k):
                out[c] = out[c] + membership * float(params.leaves[m, leaf, c])
    return tuple(out)


def interval_jacobian(params: TabParams, box: object, beta: float) -> list[list[Interval]]:
    r"""Enclosure of the Jacobian ``dF_k / dx_f`` over the box (forward-mode intervals).

    Returns ``jac[k][f]``; ``jac[k][f].lo >= 0`` certifies output ``k`` is monotone
    increasing in feature ``f`` over the box, ``hi <= 0`` decreasing.

    The soft tree is multilinear in its gates, so we differentiate **by gate** rather than
    by leaf: ``dF/dx_f = sum_j (dF/dg_j) (dg_j/dx_f)`` with ``dg_j/dx_f = beta g_j(1-g_j)
    W[m,j,f]`` and ``dF/dg_j`` a sum over leaves of the *other* gates only. This keeps the
    ``leaf_on - leaf_off`` difference correlated inside one interval product -- at depth 1 it
    is the exact ``(leaf_1 - leaf_0) beta g(1-g) W`` -- so a genuinely monotone model is
    certified instead of drowning in the interval dependency problem. (Regrouping a real
    expression never breaks soundness; it only tightens the enclosure.)
    """
    d, T, D, k = params.n_features, params.config.n_trees, params.depth, params.n_outputs
    x_iv = normalize_box(box, d)
    gates = _gate_intervals(params, x_iv, beta)
    codes = leaf_code_matrix(D)
    jac = [[Interval.point(0.0) for _ in range(d)] for _ in range(k)]

    for m in range(T):
        for j in range(D):
            # dF/dg_j = sum_leaf (+1 if gate j on else -1) * prod_{j'!=j} factor_{j'} * leaf
            dFdg = [Interval.point(0.0) for _ in range(k)]
            for leaf in range(1 << D):
                sign = 1.0 if codes[leaf, j] > 0.5 else -1.0
                prod_rest = Interval.point(1.0)
                for jj in range(D):
                    if jj != j:
                        gjj = gates[m][jj]
                        prod_rest = prod_rest * (gjj if codes[leaf, jj] > 0.5 else (1.0 - gjj))
                for c in range(k):
                    dFdg[c] = dFdg[c] + sign * prod_rest * float(params.leaves[m, leaf, c])
            gj = gates[m][j]
            gprime = gj * (1.0 - gj)  # gate slope, enclosed in [0, 1/4]
            for f in range(d):
                dgdx = beta * gprime * float(params.W[m, j, f])
                for c in range(k):
                    jac[c][f] = jac[c][f] + dFdg[c] * dgdx
    return jac


def lipschitz_from_jacobian(jac: list[list[Interval]], *, norm: str = "l2") -> tuple[float, ...]:
    r"""Per-output Lipschitz upper bound from a Jacobian enclosure (``l2`` / ``inf``)."""
    out: list[float] = []
    for row in jac:
        mags = [iv.mag for iv in row]
        if norm == "inf":
            out.append(math.fsum(mags))  # sup ||grad||_1 bounds the L-inf Lipschitz constant
        elif norm == "l2":
            acc = Interval.point(0.0)
            for mgn in mags:
                acc = acc + Interval.point(mgn) * Interval.point(mgn)
            out.append(acc.sqrt().hi)
        else:
            raise ValueError(f"unknown norm {norm!r}; choose 'l2' or 'inf'")
    return tuple(out)


def rounding_gap(params: TabParams, X: FloatArray, beta: float) -> tuple[FloatArray, FloatArray]:
    r"""Sound per-sample bound on ``|F_soft - F_hard|`` and the measured value.

    For tree ``m`` the leaf routing is a product of independent Bernoulli gates, so the
    total variation between the soft routing and its ``beta -> inf`` hardening is at most
    ``sum_j e_{m,j}`` with the per-gate error ``e = sigmoid(-beta |z|)``. A tree's
    contribution is an expectation over its leaves, so it moves by at most
    ``(max_l leaf - min_l leaf) * min(1, sum_j e)``. Summing over trees gives a certified
    per-sample, per-output bound (returned alongside the actually-measured gap so callers
    can self-check ``bound >= measured``).
    """
    from omnibias.tab._core.forward import forward_np, hard_forward_np

    Xv = np.asarray(X, dtype=np.float64)
    n = Xv.shape[0]
    k = params.n_outputs
    z = np.abs(np.einsum("nd,mjd->nmj", Xv, params.W) - params.t[None, :, :])  # (n, T, D)
    gate_err = sigmoid_np(-beta * z)  # (n, T, D)
    tv = np.minimum(1.0, np.sum(gate_err, axis=2))  # (n, T)
    leaf_range = params.leaves.max(axis=1) - params.leaves.min(axis=1)  # (T, k)
    bound = np.einsum("nm,mk->nk", tv, leaf_range)  # (n, k)
    measured = np.abs(forward_np(params, Xv, beta) - hard_forward_np(params, Xv))
    assert bound.shape == (n, k)
    return bound, measured


__all__ = [
    "interval_jacobian",
    "interval_output_bounds",
    "lipschitz_from_jacobian",
    "normalize_box",
    "rounding_gap",
]
