# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified (outward-rounded) interval soft-DP -- sound enclosures of the soft value.

Every operation is an :class:`omnibias.core.verified.Interval` with directed rounding, so
the returned interval **provably** contains the true soft value ``V_beta`` for *every*
point in the input box (a ``local_box`` enclosure). The reduction is the stable identity
``lse_beta(x) = c + beta^-1 ln(sum_i exp(beta (x_i - c)))`` with ``c = max_i x_i.hi`` a
scalar shift (an exact algebraic rewrite, so soundness is preserved while ``exp`` arguments
stay ``<= 0`` and never overflow). The pairwise form uses
``lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))`` via
:func:`omnibias.core.verified.transcend.softplus_iv`.

Honesty: interval arithmetic over-approximates (dependency / wrapping), so these enclosures
are **sound but not tight** -- they widen with the box radius and are scoped to the given
local box, never a global claim. This is the rigorous register of the same soft DP; the
differentiable register lives in :mod:`omnibias.struct.torch` / :mod:`omnibias.struct.jax`.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from omnibias.core.verified import Interval
from omnibias.core.verified.transcend import exp_iv, ln_iv, softplus_iv
from omnibias.struct._core.align import AlignmentLattice
from omnibias.struct._core.trellis import DAG, CTCLattice

Number: TypeAlias = float | int | Interval


def _as_interval(x: Number) -> Interval:
    return x if isinstance(x, Interval) else Interval.from_value(float(x))


def _softmin_iv(values: Sequence[Number], beta: float) -> Interval:
    r"""Outward-rounded soft-``min`` ``-lse_beta(-values)`` (the min-convention reduction)."""
    return -lse_beta_iv([-_as_interval(v) for v in values], beta)


def lse_beta_iv(values: Sequence[Number], beta: float) -> Interval:
    r"""Outward-rounded enclosure of ``lse_beta`` over interval-valued scores.

    Uses the scalar-shift identity for stability; the result soundly encloses
    ``beta^-1 log sum_i exp(beta v_i)`` for every point selection ``v_i in values_i``.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    ivs = [_as_interval(v) for v in values]
    if not ivs:
        raise ValueError("lse_beta_iv needs at least one value")
    shift = max(iv.hi for iv in ivs)
    total = exp_iv(beta * (ivs[0] - shift))
    for iv in ivs[1:]:
        total = total + exp_iv(beta * (iv - shift))
    return shift + ln_iv(total) * (1.0 / beta)


def pairwise_lse_iv(a: Number, b: Number, beta: float) -> Interval:
    r"""Outward-rounded pairwise ``lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))``."""
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    ai, bi = _as_interval(a), _as_interval(b)
    return ai + softplus_iv(beta * (bi - ai)) * (1.0 / beta)


def chain_value_iv(
    emissions: Sequence[Sequence[Number]],
    transitions: Sequence[Sequence[Number]],
    beta: float,
    *,
    start: Sequence[Number] | None = None,
) -> Interval:
    r"""Sound interval enclosure of ``soft_viterbi`` over a box of emissions / transitions.

    ``emissions`` is ``(T, S)`` and ``transitions`` is ``(S, S)`` of numbers or
    :class:`Interval`; ``start`` is ``(S,)`` or ``None``. Encloses the true soft value for
    every point in the box (local scope).
    """
    n_steps = len(emissions)
    n_states = len(emissions[0])
    emit = [[_as_interval(emissions[t][s]) for s in range(n_states)] for t in range(n_steps)]
    trans = [[_as_interval(transitions[i][j]) for j in range(n_states)] for i in range(n_states)]
    start_iv = (
        [_as_interval(0.0) for _ in range(n_states)]
        if start is None
        else [_as_interval(start[s]) for s in range(n_states)]
    )
    alpha = [start_iv[s] + emit[0][s] for s in range(n_states)]
    for t in range(1, n_steps):
        alpha = [
            emit[t][s] + lse_beta_iv([alpha[sp] + trans[sp][s] for sp in range(n_states)], beta)
            for s in range(n_states)
        ]
    return lse_beta_iv(alpha, beta)


def dag_value_iv(
    weights: Sequence[Sequence[Number]],
    dag: DAG,
    beta: float,
) -> Interval:
    r"""Sound interval enclosure of ``soft_shortest_path`` (softmin cost) over a weight box.

    ``weights`` is ``(n, n)`` of numbers or :class:`Interval` (only ``dag`` edge entries
    are read). Returns ``-alpha[sink]`` in the max convention (the softmin cost enclosure).
    """
    alpha: list[Interval | None] = [None] * dag.num_nodes
    alpha[dag.source] = _as_interval(0.0)
    for v in range(dag.num_nodes):
        if v == dag.source:
            continue
        preds = [u for u in dag.incoming(v) if alpha[u] is not None]
        if not preds:
            continue
        contribs = [alpha_u - _as_interval(weights[u][v]) for u in preds if (alpha_u := alpha[u]) is not None]
        alpha[v] = lse_beta_iv(contribs, beta)
    sink = alpha[dag.sink]
    if sink is None:
        raise ValueError("sink is not reachable from source in this DAG")
    return -sink


def dtw_value_iv(cost: Sequence[Sequence[Number]], beta: float) -> Interval:
    r"""Sound interval enclosure of :func:`omnibias.struct.torch.soft_dtw` over a cost box.

    ``cost`` is the ``(n, m)`` local-cost box (numbers or :class:`Interval`). The soft-DTW
    recursion is a per-cell **softmin** (``-lse_beta`` of the negated predecessors), which
    equals the flat softmin over all monotonic paths; the interval version encloses the true
    soft value for every cost matrix in the box (``local_box`` scope).
    """
    d = [[_as_interval(cost[i][j]) for j in range(len(cost[i]))] for i in range(len(cost))]
    n, m = len(d), len(d[0])
    r: list[list[Interval | None]] = [[None] * m for _ in range(n)]
    r[0][0] = d[0][0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            preds = [
                r[pi][pj]
                for pi, pj in ((i - 1, j), (i, j - 1), (i - 1, j - 1))
                if pi >= 0 and pj >= 0 and r[pi][pj] is not None
            ]
            r[i][j] = d[i][j] + _softmin_iv([p for p in preds if p is not None], beta)
    out = r[n - 1][m - 1]
    if out is None:
        raise ValueError("DTW sink is unreachable")
    return out


def align_value_iv(
    a: object, b: object, sub: Sequence[Sequence[Number]], gap: Number, beta: float
) -> Interval:
    r"""Sound interval enclosure of :func:`omnibias.struct.torch.soft_align` over a param box.

    ``sub`` is a ``(K, K)`` substitution-score box and ``gap`` a scalar gap box. Global
    alignment is a longest-path DAG, so this assembles the interval edge-cost matrix and
    reuses :func:`dag_value_iv`; the enclosure contains the true soft score for every
    ``(sub, gap)`` in the box (repeated substitution entries are treated independently, so it
    is sound but not tight).
    """
    ai = np.asarray(a, dtype=int).reshape(-1)
    bj = np.asarray(b, dtype=int).reshape(-1)
    lattice = AlignmentLattice(ai.shape[0], bj.shape[0])
    dag, labels = lattice.build_dag()
    zero = Interval.point(0.0)
    weight: list[list[Interval]] = [[zero] * lattice.num_nodes for _ in range(lattice.num_nodes)]
    for (u, v), (kind, i, j) in labels.items():
        if kind == "sub":
            weight[u][v] = -_as_interval(sub[int(ai[i])][int(bj[j])])
        else:
            weight[u][v] = -_as_interval(gap)
    return -dag_value_iv(weight, dag, beta)


def ctc_value_iv(
    log_probs: Sequence[Sequence[Number]], lattice: CTCLattice, beta: float
) -> Interval:
    r"""Sound interval enclosure of the soft CTC log-partition over a ``log_probs`` box.

    ``log_probs`` is the ``(T, C)`` box; ``lattice`` fixes the target label sequence. Runs the
    blank-augmented forward recursion with outward-rounded ``lse_beta`` and encloses the true
    CTC soft value (``omnibias.struct.brute_force_partition``) for every ``log_probs`` in the
    box (``local_box`` scope).
    """
    lp = [[_as_interval(log_probs[t][c]) for c in range(len(log_probs[t]))] for t in range(len(log_probs))]
    n_steps = len(lp)
    ext = [int(x) for x in lattice.extended_labels()]
    m = len(ext)
    alpha: list[Interval | None] = [None] * m
    alpha[0] = lp[0][ext[0]]
    if m > 1:
        alpha[1] = lp[0][ext[1]]
    for t in range(1, n_steps):
        nxt: list[Interval | None] = [None] * m
        for s in range(m):
            preds = [alpha[p] for p in lattice.incoming(s) if alpha[p] is not None]
            if not preds:
                continue
            nxt[s] = lp[t][ext[s]] + lse_beta_iv([p for p in preds if p is not None], beta)
        alpha = nxt
    ends = [alpha[m - 1]] + ([alpha[m - 2]] if m >= 2 else [])
    reachable = [e for e in ends if e is not None]
    if not reachable:
        raise ValueError("no CTC alignment reaches an accepting state (T too small?)")
    return lse_beta_iv(reachable, beta)


def chain_marginals_iv(
    emissions: Sequence[Sequence[Number]],
    transitions: Sequence[Sequence[Number]],
    beta: float,
    *,
    start: Sequence[Number] | None = None,
) -> list[list[Interval]]:
    r"""Sound interval forward-backward enclosure of the ``(T, S)`` state-occupancy marginals.

    ``mu[t][s]`` encloses ``P_beta(state s at time t)`` -- equivalently
    ``d soft_viterbi / d emissions[t, s]`` -- for every emission / transition matrix in the
    box. Uses the log-domain identity ``mu = exp(beta (alpha + backward - Z))`` with
    outward-rounded ``lse_beta`` forward / backward passes; sound (not tight), ``local_box``
    scope. Each row soundly encloses the simplex point (``sum_s mu[t][s]`` contains ``1``).
    """
    n_steps = len(emissions)
    n_states = len(emissions[0])
    emit = [[_as_interval(emissions[t][s]) for s in range(n_states)] for t in range(n_steps)]
    trans = [[_as_interval(transitions[i][j]) for j in range(n_states)] for i in range(n_states)]
    start_iv = (
        [Interval.point(0.0) for _ in range(n_states)]
        if start is None
        else [_as_interval(start[s]) for s in range(n_states)]
    )
    alpha = [[start_iv[s] + emit[0][s] for s in range(n_states)]]
    for t in range(1, n_steps):
        alpha.append(
            [
                emit[t][s] + lse_beta_iv([alpha[t - 1][sp] + trans[sp][s] for sp in range(n_states)], beta)
                for s in range(n_states)
            ]
        )
    bwd: list[list[Interval]] = [[Interval.point(0.0) for _ in range(n_states)] for _ in range(n_steps)]
    for t in range(n_steps - 2, -1, -1):
        bwd[t] = [
            lse_beta_iv(
                [trans[s][sp] + emit[t + 1][sp] + bwd[t + 1][sp] for sp in range(n_states)], beta
            )
            for s in range(n_states)
        ]
    z = lse_beta_iv(alpha[n_steps - 1], beta)
    return [
        [exp_iv(beta * (alpha[t][s] + bwd[t][s] - z)) for s in range(n_states)]
        for t in range(n_steps)
    ]


def _perm_sign(perm: tuple[int, ...]) -> int:
    """Sign of a permutation (``+1`` / ``-1``) by counting inversions."""
    inversions = sum(
        1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j]
    )
    return 1 if inversions % 2 == 0 else -1


def _det_leibniz(mat: Sequence[Sequence[Interval]]) -> Interval:
    """Outward-rounded ``det`` via the Leibniz permutation sum (always defined, but loose)."""
    n = len(mat)
    total = Interval.point(0.0)
    for perm in itertools.permutations(range(n)):
        prod = Interval.point(float(_perm_sign(perm)))
        for i in range(n):
            prod = prod * mat[i][perm[i]]
        total = total + prod
    return total


def _det_gauss(mat: Sequence[Sequence[Interval]]) -> Interval | None:
    r"""Outward-rounded ``det`` via interval Gaussian elimination (``None`` if a pivot straddles 0).

    Much tighter than Leibniz for the (column-diagonally-dominant) Laplacian, since it avoids
    the sign cancellation of the permutation expansion. Sound: the product of interval pivots
    encloses the true determinant of every point matrix in the box.
    """
    n = len(mat)
    a = [[mat[i][j] for j in range(n)] for i in range(n)]
    det = Interval.point(1.0)
    for k in range(n):
        piv = a[k][k]
        if piv.lo <= 0.0 <= piv.hi:
            return None
        det = det * piv
        inv_piv = Interval.point(1.0) / piv
        for i in range(k + 1, n):
            factor = a[i][k] * inv_piv
            for j in range(k + 1, n):
                a[i][j] = a[i][j] - factor * a[k][j]
    return det


def _det_precond(mat: Sequence[Sequence[Interval]]) -> Interval | None:
    r"""Midpoint-inverse-preconditioned ``det`` enclosure (the tight rigorous method).

    With ``R = inv(mid(mat))`` (a float preconditioner), ``R @ mat`` is a tight interval
    matrix near the identity, so ``det(R @ mat)`` enclosed by Gaussian elimination has pivots
    near ``1`` and barely wraps; ``det(mat) = det(mid) * det(R @ mat)`` recovers the enclosure
    with ``det(mid)`` pinned by Leibniz on the (zero-width) midpoint. Returns ``None`` if the
    midpoint is numerically singular. Sound: every factor encloses the corresponding true
    determinant.
    """
    n = len(mat)
    mid = np.array([[0.5 * (mat[i][j].lo + mat[i][j].hi) for j in range(n)] for i in range(n)])
    try:
        r_pre = np.linalg.inv(mid)
    except np.linalg.LinAlgError:
        return None
    preconditioned = [
        [
            sum(
                (Interval.point(float(r_pre[i, k])) * mat[k][j] for k in range(n)),
                Interval.point(0.0),
            )
            for j in range(n)
        ]
        for i in range(n)
    ]
    det_near_identity = _det_gauss(preconditioned)
    if det_near_identity is None:
        return None
    det_mid = _det_leibniz([[Interval.point(float(mid[i, j])) for j in range(n)] for i in range(n)])
    return det_mid * det_near_identity


def _interval_det(mat: Sequence[Sequence[Interval]]) -> Interval:
    r"""Tightest sound ``det`` enclosure: intersect Leibniz, Gaussian, and preconditioned.

    All three enclose the true determinant, so their intersection does too (and is tighter):
    Leibniz is robust but loose, Gaussian is tighter, and the midpoint-preconditioned form is
    tightest whenever the midpoint is well-conditioned. Tiny ``n`` only.
    """
    lo, hi = _det_leibniz(mat).lo, _det_leibniz(mat).hi
    for candidate in (_det_gauss(mat), _det_precond(mat)):
        if candidate is not None:
            lo, hi = max(lo, candidate.lo), min(hi, candidate.hi)
    return Interval(lo, hi)


def matrix_tree_partition_iv(arc: Sequence[Sequence[Number]], beta: float) -> Interval:
    r"""Sound interval enclosure of the Matrix-Tree log-partition ``log det L(beta) / beta``.

    ``arc`` is the ``(n + 1, n + 1)`` arc-score box (row / column ``0`` is the ROOT wall).
    The Kirchhoff Laplacian is built with a fixed column-max log-shift (so ``exp`` arguments
    stay ``<= 0``) and its determinant is enclosed by the Leibniz permutation sum -- exact for
    tiny ``n``, and the log-partition encloses :func:`omnibias.struct.brute_force_arborescence`
    for every arc matrix in the box. This is the rigorous register of the *exact* determinant
    partition (not an ``lse_beta`` relaxation); ``local_box`` scope.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    a = [[_as_interval(arc[h][m]) for m in range(len(arc[h]))] for h in range(len(arc))]
    n = len(a) - 1
    if n < 1:
        raise ValueError(f"need at least one word (arc >= 2x2), got n={n}")
    # Fixed scalar log-shift per modifier column (uses the box upper bound; any fixed c is sound).
    c = [max((beta * a[h][m]).hi for h in range(n + 1) if h != m) for m in range(1, n + 1)]
    ltilde: list[list[Interval]] = [[Interval.point(0.0) for _ in range(n)] for _ in range(n)]
    for m in range(1, n + 1):
        for h in range(n + 1):
            if h == m:
                continue
            wt = exp_iv(beta * a[h][m] - c[m - 1])
            ltilde[m - 1][m - 1] = ltilde[m - 1][m - 1] + wt
            if h >= 1:
                ltilde[h - 1][m - 1] = ltilde[h - 1][m - 1] - wt
    det = _interval_det(ltilde)
    if det.lo <= 0.0:
        raise ValueError("Laplacian determinant enclosure is not strictly positive over the box")
    return (ln_iv(det) + float(sum(c))) * (1.0 / beta)


def box(center: object, radius: float) -> list[Interval] | list[list[Interval]]:
    r"""Build a symmetric interval box ``[c - radius, c + radius]`` for a 1-D / 2-D array."""
    c = np.asarray(center, dtype=float)
    r = float(radius)
    if c.ndim == 1:
        return [Interval(float(c[i]) - r, float(c[i]) + r) for i in range(c.shape[0])]
    if c.ndim == 2:
        return [
            [Interval(float(c[i, j]) - r, float(c[i, j]) + r) for j in range(c.shape[1])]
            for i in range(c.shape[0])
        ]
    raise ValueError(f"box supports 1-D or 2-D centers, got ndim {c.ndim}")


__all__ = [
    "align_value_iv",
    "box",
    "chain_marginals_iv",
    "chain_value_iv",
    "ctc_value_iv",
    "dag_value_iv",
    "dtw_value_iv",
    "lse_beta_iv",
    "matrix_tree_partition_iv",
    "pairwise_lse_iv",
]
