# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous lower bounds on the minimum QUBO energy (the certificate's lower side).

Two strengths, both **sound** -- they never exceed the true minimum:

* :func:`spectral_lower_bound` (any ``n``, cheap): shift the diagonal so
  ``Q' = Q + t I`` is positive definite (``t`` from a Gershgorin eigenvalue bound). On
  the cube ``x_i^2 = x_i`` so ``E(x) = x^T Q' x + (c - t 1)^T x + const``; over the box
  ``[0, 1]^n`` (which contains the cube) this convex box-QP's minimum is a lower bound,
  and :func:`omnibias.convex.certify_qp_optimum` seals it with outward-rounded interval
  arithmetic. Without ``omnibias-convex`` it degrades to the (still valid) float value
  with ``certified=False``. This is the quadratic-specific bound and stays in
  ``omnibias-qubo``.

* :func:`lasserre_lower_bound` (small / moderate ``n``, headline): the Lasserre /
  moment-SOS bound over the Boolean hypercube, re-exported from the ``omnibias-discrete``
  substrate (it is generic over any ``DiscreteProblem``). See
  :func:`omnibias.discrete.lasserre_lower_bound`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omnibias.discrete import gershgorin_min_eig_lower, lasserre_lower_bound
from omnibias.qubo.problem import QUBOProblem

FloatArray = NDArray[np.float64]


def _diag_shift(problem: QUBOProblem) -> tuple[FloatArray, FloatArray, float]:
    """Positive-definite diagonal shift ``Q' = Q + t I`` and the linear term ``c - t 1``."""
    q = np.asarray(problem.Q, dtype=float)
    c = np.asarray(problem.c, dtype=float)
    n = problem.n
    lam_lo = gershgorin_min_eig_lower(q)
    margin = 1e-6 * (1.0 + float(np.max(np.abs(q))) if q.size else 1.0)
    t = max(0.0, -lam_lo) + margin
    qp = q + t * np.eye(n)
    cp = c - t * np.ones(n)
    return qp, cp, t


def _solve_box_qp(qp: FloatArray, cp: FloatArray, *, iters: int = 800) -> FloatArray:
    """Projected-gradient minimizer of ``x^T Q' x + c'^T x`` over the box ``[0, 1]^n``."""
    n = qp.shape[0]
    lip = 2.0 * float(np.linalg.norm(qp, 2)) + 1e-30
    eta = 1.0 / lip
    x = 0.5 * np.ones(n)
    for _ in range(iters):
        grad = 2.0 * (qp @ x) + cp
        x = np.clip(x - eta * grad, 0.0, 1.0)
    return x


def spectral_lower_bound(problem: QUBOProblem) -> tuple[float, bool]:
    r"""A rigorous lower bound on the minimum energy via the convex box-QP relaxation.

    Returns ``(lower_bound, certified)``; ``certified`` is ``True`` when the bound is
    interval-sealed by :func:`omnibias.convex.certify_qp_optimum`, ``False`` when it
    degrades to the valid float value (``omnibias-convex`` absent or its Neumann
    conditioning check declines the ``Q'`` at hand).
    """
    qp, cp, _t = _diag_shift(problem)
    n = problem.n
    x = _solve_box_qp(qp, cp)
    grad = 2.0 * (qp @ x) + cp
    dual = np.concatenate([np.maximum(-grad, 0.0), np.maximum(grad, 0.0)])
    a_box = np.vstack([np.eye(n), -np.eye(n)])
    b_box = np.concatenate([np.ones(n), np.zeros(n)])
    try:
        from omnibias.convex import certify_qp_optimum

        # Weak duality: the enclosure's lower endpoint g(lambda) depends only on the
        # dual, so any strictly-interior primal satisfies the rigorous feasibility gate
        # (the solved boundary point rounds marginally infeasible) without loosening it.
        interior = 0.5 * np.ones(n)
        cert = certify_qp_optimum(2.0 * qp, cp, a_box, b_box, interior, dual)
        return float(cert.enclosure.lo) + problem.const, True
    except Exception:
        # Valid (unconstrained) float lower bound on the box-QP minimum: min_x f(x) =
        # -1/4 c'^T Q'^{-1} c' <= min_{box} f <= min_{cube} (E - const). Not sealed.
        qp_inv = np.linalg.inv(qp)
        lower = -0.25 * float(cp @ qp_inv @ cp) + problem.const
        return lower, False


__all__ = [
    "gershgorin_min_eig_lower",
    "lasserre_lower_bound",
    "spectral_lower_bound",
]
