# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The Gilmore-Lawler bound: a sound, *scalable* lower bound on the QAP optimum.

The Gilmore-Lawler bound (Gilmore 1962; Lawler 1963) is the classic ``O(dim^3)`` lower
bound on the quadratic-assignment optimum. For the Koopmans-Beckmann objective
``sum_{i,k} F[i,k] D[pi(i),pi(k)]`` it builds a *leader* cost matrix

.. math::
    C_{ip} = F_{ii} D_{pp}
           + \bigl\langle \mathrm{sort}_\uparrow(F_{i,\cdot\neq i}),\;
                          \mathrm{sort}_\downarrow(D_{p,\cdot\neq p}) \bigr\rangle,

whose off-diagonal entry is the *minimum* scalar product of facility ``i``'s flows with
location ``p``'s distances (the rearrangement inequality: pairing an ascending vector with
a descending one minimises the dot product, so it is a rigorous lower bound on the
interaction contribution of ``i`` at ``p`` under *any* placement of the other facilities).
The linear-assignment minimum ``min_pi sum_i C_{i,pi(i)}`` is then a lower bound on the QAP
optimum -- the classic Gilmore-Lawler theorem.

Unlike the Lasserre / SOS bound (tight but confined to ``dim <= 4`` before the SDP blows
up) and the spectral bound (any size but very loose), GLB stays **sound and non-trivial at
realistic block counts** (``dim ~ 12-25``). It is still an NP-hard-honest bound: generally
**non-tight**, never asserted equal to the optimum.

Soundness. For integer ``F`` / ``D`` (the placement default) the whole computation is exact
integer arithmetic. For floating-point inputs the leader entries are lower-bounded with
outward-rounded :class:`omnibias.core.verified.Interval` arithmetic and the assignment is
solved on those lower bounds; since the linear-assignment optimum is monotone in its costs,
``LAP(C_lo) <= LAP(C) <= optimum`` -- so the returned value provably does not exceed the
true optimum.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omnibias.nphard._core.qap import QAPProblem

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _sorted_offdiagonals(
    matrix: FloatArray, *, descending: bool
) -> list[FloatArray]:
    """Per-row off-diagonal entries, sorted ascending (or descending)."""
    dim = int(matrix.shape[0])
    rows: list[FloatArray] = []
    for r in range(dim):
        off = np.delete(matrix[r], r)
        ordered = np.sort(off)
        rows.append(ordered[::-1] if descending else ordered)
    return rows


def gilmore_lawler_bound(problem: QAPProblem) -> tuple[float, bool]:
    r"""Sound Gilmore-Lawler lower bound on ``problem``'s QAP (permutation) optimum.

    Returns ``(lower_bound, sound)`` where ``lower_bound <= min_pi wirelength`` provably
    holds (``sound`` is ``True``: exact for integer instances, outward-rounded-interval for
    floating-point ones). The bound is generally **non-tight** -- QAP is NP-hard.
    """
    from scipy.optimize import linear_sum_assignment

    flow = np.asarray(problem.flow, dtype=float)
    distance = np.asarray(problem.distance, dtype=float)
    dim = int(flow.shape[0])
    if dim == 0:
        return 0.0, True

    diag_flow = np.diagonal(flow)
    diag_dist = np.diagonal(distance)
    flow_rows = _sorted_offdiagonals(flow, descending=False)  # ascending
    dist_rows = _sorted_offdiagonals(distance, descending=True)  # descending

    is_integer = bool(
        np.all(flow == np.round(flow)) and np.all(distance == np.round(distance))
    )

    if is_integer:
        # exact integer arithmetic -> the bound is exact and trivially sound
        fr: IntArray = np.stack([r.astype(np.int64) for r in flow_rows]) if dim > 1 else (
            np.zeros((1, 0), dtype=np.int64)
        )
        dr: IntArray = np.stack([r.astype(np.int64) for r in dist_rows]) if dim > 1 else (
            np.zeros((1, 0), dtype=np.int64)
        )
        leader = fr @ dr.T  # (dim, dim); leader[i, p] = <asc F_i, desc D_p>
        cost_i = np.outer(diag_flow.astype(np.int64), diag_dist.astype(np.int64)) + leader
        row, col = linear_sum_assignment(cost_i)
        return float(int(cost_i[row, col].sum())), True

    # floating-point: lower-bound every leader entry with outward-rounded intervals
    from omnibias.core.verified import Interval, sum_intervals

    cost = np.empty((dim, dim), dtype=float)
    for i in range(dim):
        f_i = flow_rows[i]
        lead_diag = Interval.point(float(diag_flow[i]))
        for p in range(dim):
            d_p = dist_rows[p]
            products = [
                Interval.point(float(f_i[t])) * Interval.point(float(d_p[t]))
                for t in range(dim - 1)
            ]
            entry = lead_diag * Interval.point(float(diag_dist[p]))
            if products:
                entry = entry + sum_intervals(products)
            cost[i, p] = entry.lo  # a rigorous lower bound on the true leader cost

    row, col = linear_sum_assignment(cost)
    # downward-rounded sum of the (already lower-bounding) selected entries
    acc = Interval.point(0.0)
    for i, p in zip(row, col, strict=True):
        acc = acc + Interval.point(float(cost[i, p]))
    return acc.lo, True


__all__ = ["gilmore_lawler_bound"]
