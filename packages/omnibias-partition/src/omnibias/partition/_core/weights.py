# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Numpy reference for the soft partition-of-unity weights (the parity ground truth).

A ``depth``-gate soft partition maps a batch ``X`` of shape ``(n, d)`` to region weights
``P`` of shape ``(n, 2**depth)``:

1. gate pre-activations ``Z[n, j] = W[j] . X[n] - t[j]``;
2. gates ``G = sigmoid(beta * Z) in (0, 1)`` (the ``beta -> inf`` limit is a hard split);
3. region weights ``P[n, l] = prod_j (G if bit_j(l) else 1 - G)`` -- a **soft AND** of the
   root-to-region path conditions.

For every ``x`` the weights are **non-negative and sum to one** (a genuine partition of
unity), and as ``beta -> inf`` they collapse to the crisp one-hot indicator of the region
picked by the hard gates. The torch / jax twins reproduce this bit-for-bit (float64, parity
``~1e-9``).

Terminology: the gate's ``beta -> inf`` hardening is the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import numpy as np
from omnibias.partition._core.params import FloatArray, PartitionParams, region_code_matrix


def sigmoid_np(z: FloatArray) -> FloatArray:
    r"""Numerically stable logistic ``1 / (1 + exp(-z))`` (float64)."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def gate_activations(params: PartitionParams, X: FloatArray, beta: float) -> FloatArray:
    r"""Soft gate activations ``G`` of shape ``(n, depth)`` in ``(0, 1)``."""
    Xv = np.asarray(X, dtype=np.float64)
    z = Xv @ params.W.T - params.t[None, :]  # (n, depth)
    return sigmoid_np(beta * z)


def _weights_from_gates(G: FloatArray, depth: int) -> FloatArray:
    r"""Region weights ``(n, 2**depth)`` from gates ``G`` ``(n, depth)`` (soft-AND product)."""
    codes = region_code_matrix(depth)  # (L, D)
    Gexp = G[:, None, :]  # (n, 1, D)
    B = codes[None, :, :]  # (1, L, D)
    factors = B * Gexp + (1.0 - B) * (1.0 - Gexp)  # (n, L, D)
    return np.prod(factors, axis=-1)  # (n, L)


def partition_weights(params: PartitionParams, X: FloatArray, beta: float) -> FloatArray:
    r"""Soft partition-of-unity weights ``P`` of shape ``(n, 2**depth)``.

    Each row is a probability vector over the regions (non-negative, sums to ``1``);
    ``P[i, l]`` is the soft-AND membership of sample ``i`` in region ``l``.
    """
    G = gate_activations(params, X, beta)
    return _weights_from_gates(G, params.depth)


def hard_assignment(params: PartitionParams, X: FloatArray) -> FloatArray:
    r"""Crisp region index ``(n,)`` from the ``beta -> inf`` hard gates ``1[W.x > t]``."""
    Xv = np.asarray(X, dtype=np.float64)
    z = Xv @ params.W.T - params.t[None, :]  # (n, depth)
    bits = (z > 0.0).astype(np.int64)  # (n, depth)
    weights = 1 << np.arange(params.depth, dtype=np.int64)  # bit j has value 2**j
    return (bits * weights[None, :]).sum(axis=1).astype(np.int64)


def hard_weights(params: PartitionParams, X: FloatArray) -> FloatArray:
    r"""One-hot crisp weights ``(n, 2**depth)`` -- the ``beta -> inf`` limit of the POU."""
    Xv = np.asarray(X, dtype=np.float64)
    idx = hard_assignment(params, Xv)
    out = np.zeros((Xv.shape[0], params.n_regions), dtype=np.float64)
    out[np.arange(Xv.shape[0]), idx] = 1.0
    return out


def _gate_rule(params: PartitionParams, j: int, *, tol: float = 1e-9) -> str:
    r"""Human-readable ``fires when ...`` condition for gate ``j`` (axis mode is single-feature)."""
    w, t = params.W[j], float(params.t[j])
    nz = np.flatnonzero(np.abs(w) > tol)
    if params.config.is_axis or nz.size == 1:
        f = int(nz[0]) if nz.size else 0
        sign = w[f] if nz.size else 1.0
        thr = t / sign if abs(sign) > tol else t
        rel = ">" if sign > 0 else "<"
        return f"x[{f}] {rel} {thr:.6g}"
    terms = " + ".join(f"{w[f]:.4g}*x[{f}]" for f in nz)
    return f"{terms} > {t:.6g}"


def hardened_rules(params: PartitionParams) -> list[str]:
    r"""The exported ``if ...`` split boundaries, one per gate (the ``beta -> inf`` rules)."""
    return [_gate_rule(params, j) for j in range(params.depth)]


def region_rule(params: PartitionParams, region: int) -> str:
    r"""The conjunction of gate conditions selecting ``region`` (a hardened if-then clause)."""
    codes = region_code_matrix(params.depth)
    parts: list[str] = []
    for j in range(params.depth):
        rule = _gate_rule(params, j)
        parts.append(rule if codes[region, j] > 0.5 else f"NOT ({rule})")
    return " AND ".join(parts)


__all__ = [
    "gate_activations",
    "hard_assignment",
    "hard_weights",
    "hardened_rules",
    "partition_weights",
    "region_rule",
    "sigmoid_np",
]
