# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Learned / geometric warm starts for the interior-point LP/QP solver.

An LP/QP constraint row ``a_i^T x <= b_i`` is a hyperplane, and the optimum of a
non-degenerate LP sits at a *vertex* -- the intersection of ``n`` **active**
hyperplanes. A model that reads the problem can therefore predict the **active
set** (which constraints bind at the optimum) far more cheaply than running the
solver.

Terminology: the "active set" here is the LP/QP feasibility sense (a constraint
binding at the optimum); it is **not** the founding ``delta -> 0`` bias collapse
(the multi-bias derivative-generating limit). Do not conflate the two.

This module turns such a prediction into a strictly feasible starting point:

* :func:`predicted_vertex` -- solve the ``n`` highest-scored constraints for the
  predicted vertex (the active-constraint geometry).
* :func:`geometry_warm_start` -- pull any hint back to a strictly feasible
  interior point by backtracking toward a feasible anchor (a ratio test).
* :func:`active_set_warm_start` -- compose the two: predict the vertex from
  per-constraint activation scores, then back off to the interior.

The returned point is a plain :class:`numpy.ndarray`; pass it as ``x0`` to either
:func:`omnibias.convex.jax.solve_qp` or :func:`omnibias.convex.torch.solve_qp`.
A strictly feasible ``x0`` lets the solver **skip phase-1** entirely, which is
the measurable iteration saving (see ``ConvexSolution.newton_iterations``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "active_set_warm_start",
    "geometry_warm_start",
    "predicted_vertex",
]


def _strictly_feasible(A: NDArray[np.float64], b: NDArray[np.float64],
                       x: NDArray[np.float64], margin: float) -> bool:
    return bool(np.all(b - A @ x >= margin))


def predicted_vertex(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    active_scores: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    r"""Solve the ``n`` highest-scored constraints for the predicted vertex.

    Given per-constraint activation scores (high = the predictor expects that
    bias to collapse onto its hyperplane at the optimum), select the ``n`` rows
    with the largest scores and solve the square system ``A_act x = b_act``.

    Parameters
    ----------
    A, b:
        Constraint data ``A x <= b`` with shapes ``(m, n)`` and ``(m,)``.
    active_scores:
        Shape ``(m,)`` predicted-active scores; only the ranking matters.

    Returns
    -------
    The predicted vertex ``x``, or ``None`` if the selected rows are singular
    (degenerate / rank-deficient prediction). The result is *not* guaranteed to
    be feasible -- pass it through :func:`geometry_warm_start`.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scores = np.asarray(active_scores, dtype=np.float64)
    m, n = A.shape
    if scores.shape != (m,):
        raise ValueError(f"active_scores must have shape (m,) = ({m},), got {scores.shape}")
    if m < n:
        return None
    top = np.argsort(scores)[::-1][:n]
    a_act = A[top]
    b_act = b[top]
    try:
        x = np.linalg.solve(a_act, b_act)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(x)):
        return None
    return np.asarray(x, dtype=np.float64)


def geometry_warm_start(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x_hint: NDArray[np.float64],
    *,
    anchor: NDArray[np.float64] | None = None,
    margin: float = 1e-6,
    shrink: float = 0.99,
) -> NDArray[np.float64] | None:
    r"""Pull ``x_hint`` back to a strictly feasible interior point.

    If ``x_hint`` already satisfies ``A x < b`` with at least ``margin`` slack it
    is returned unchanged. Otherwise the segment from a strictly feasible
    ``anchor`` toward ``x_hint`` is walked as far as a ratio test allows while
    keeping every slack ``>= margin``, then scaled by ``shrink`` to stay strictly
    interior.

    Parameters
    ----------
    A, b:
        Constraint data ``A x <= b``.
    x_hint:
        Candidate point (e.g. :func:`predicted_vertex`).
    anchor:
        A strictly feasible interior point to back off toward. Defaults to the
        origin, which is feasible iff ``b > 0``; if neither the supplied anchor
        nor the origin is strictly feasible, ``None`` is returned (the caller
        should fall back to the solver's phase-1).
    margin:
        Minimum slack required of the returned point.
    shrink:
        Fraction of the feasible step length to take (``0 < shrink < 1``).

    Returns
    -------
    A strictly feasible point, or ``None`` if no feasible anchor is available.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x_hint = np.asarray(x_hint, dtype=np.float64)
    if _strictly_feasible(A, b, x_hint, margin):
        return x_hint

    if anchor is None:
        anchor = np.zeros(A.shape[1], dtype=np.float64)
    else:
        anchor = np.asarray(anchor, dtype=np.float64)
    if not _strictly_feasible(A, b, anchor, margin):
        return None

    # Largest theta in [0, 1] with  b - A(anchor + theta d) >= margin, d = x_hint - anchor.
    d = x_hint - anchor
    s_anchor = b - A @ anchor          # > 0 by the check above
    ad = A @ d
    theta = 1.0
    increasing = ad > 0.0              # slack shrinks where A d > 0
    if np.any(increasing):
        ratios = (s_anchor[increasing] - margin) / ad[increasing]
        theta = float(min(1.0, np.min(ratios)))
    theta = max(0.0, theta) * shrink
    x0 = anchor + theta * d
    return x0 if _strictly_feasible(A, b, x0, margin) else anchor


def active_set_warm_start(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    active_scores: NDArray[np.float64],
    *,
    anchor: NDArray[np.float64] | None = None,
    margin: float = 1e-6,
    shrink: float = 0.99,
) -> NDArray[np.float64] | None:
    r"""Predict the active-set vertex, then back off to a strictly feasible point.

    Convenience composition of :func:`predicted_vertex` and
    :func:`geometry_warm_start`. Returns ``None`` if no vertex can be predicted
    (rank-deficient selection) or no feasible anchor exists.
    """
    vertex = predicted_vertex(A, b, active_scores)
    if vertex is None:
        return None
    return geometry_warm_start(
        A, b, vertex, anchor=anchor, margin=margin, shrink=shrink
    )
