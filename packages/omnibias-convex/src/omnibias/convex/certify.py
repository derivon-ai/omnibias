# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Verified (certified) a-posteriori optimality enclosure for convex QPs.

Given a *computed* primal ``x`` and dual ``lambda >= 0`` for the strictly convex
QP

.. math::
    \min_x \tfrac12 x^\top Q x + c^\top x \quad\text{s.t.}\quad A x \le b,
    \qquad Q \succ 0,

this returns a **rigorous interval enclosure of the optimal value** ``f*`` using
interval arithmetic from :mod:`omnibias.core.verified` -- independent of the
(floating-point) solver that produced ``x`` and ``lambda``.

* **Upper bound.** If ``x`` is (rigorously) primal feasible, ``f* <= f0(x)``;
  ``f0(x)`` is evaluated in outward-rounded interval arithmetic.
* **Lower bound.** Weak duality: for any ``lambda >= 0``,
  ``f* >= g(lambda) = -1/2 v^T Q^{-1} v - b^T lambda`` with ``v = c + A^T lambda``.
  A rigorous upper bound on ``v^T Q^{-1} v <= ||Q^{-1}||_inf ||v||_1 ||v||_inf``
  uses the Neumann certificate :func:`~omnibias.core.verified.neumann_inverse_norm_bound`
  for ``||Q^{-1}||_inf``.

The width of the returned interval is a rigorous bound on the suboptimality gap;
it shrinks to zero as ``(x, lambda)`` approach the true optimum (``v -> 0`` and the
primal/dual objectives meet). This is the certificate that distinguishes the
omnibias solver from a plain float solve.

For a **linear** program (``Q = 0``, optionally with equalities ``A_eq x = b_eq``)
the Neumann ``Q^{-1}`` machinery does not apply; :func:`lp_dual_lower_bound` /
:func:`certify_lp_optimum` instead use the **Neumaier-Shcherbina** verified LP
bound: given *any* multipliers ``lambda >= 0, nu`` (even the first-order solver's
approximate, dual-*infeasible* estimates) and a finite box ``x_lo <= x <= x_hi``
containing the feasible set,

.. math::
    f^* \ge \min_{x \in [x_{lo}, x_{hi}]}
        \bigl(c + A^\top\lambda + A_{eq}^\top\nu\bigr)^\top x
        - b^\top\lambda - b_{eq}^\top\nu

is a rigorous lower bound (the separable box-minimum of the dual-residual term
corrects for dual infeasibility), enclosed with outward-rounded intervals. Looser
multipliers only loosen -- never invalidate -- the bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.core.verified import Interval, neumann_inverse_norm_bound, sum_intervals
from omnibias.core.verified.linalg import inf_norm_vector, matvec, to_interval_matrix


class CertificationError(RuntimeError):
    """Raised when a rigorous certificate cannot be produced."""


@dataclass(frozen=True)
class Certificate:
    """Result of a verified optimality enclosure.

    Attributes
    ----------
    enclosure:
        Rigorous :class:`~omnibias.core.verified.Interval` containing ``f*``.
    primal_feasible:
        Whether ``x`` was rigorously verified to satisfy ``A x <= b``.
    dual_residual_inf:
        Rigorous upper bound on ``||c + A^T lambda||_inf`` (dual-feasibility
        residual; zero at a KKT point).
    gap:
        Certified suboptimality gap ``enclosure.width`` (upper - lower).
    """

    enclosure: Interval
    primal_feasible: bool
    dual_residual_inf: float
    gap: float


def _ivec(x: np.ndarray) -> list[Interval]:
    return [Interval.point(float(v)) for v in x]


def certify_qp_optimum(
    Q: Any, c: Any, A: Any, b: Any, x: Any, dual: Any
) -> Certificate:
    r"""Rigorous interval enclosure of the optimal value of a strictly convex QP.

    Parameters
    ----------
    Q, c, A, b:
        Problem data (``Q`` symmetric positive definite). Any array-like.
    x:
        Computed primal point.
    dual:
        Computed dual multipliers; clamped to ``max(dual, 0)`` so the enclosure is
        valid for the nearest sign-feasible dual.

    Returns
    -------
    :class:`Certificate` whose ``enclosure`` provably contains ``f*``.

    Raises
    ------
    CertificationError
        If ``x`` is not rigorously primal feasible, or ``||Q^{-1}||`` cannot be
        certified (Neumann ``kappa >= 1``; supply a better-conditioned ``Q``).
    """
    Qn = np.asarray(Q, dtype=float)
    cn = np.asarray(c, dtype=float)
    An = np.asarray(A, dtype=float)
    bn = np.asarray(b, dtype=float)
    xn = np.asarray(x, dtype=float)
    lam = np.maximum(np.asarray(dual, dtype=float), 0.0)
    n = cn.shape[0]
    m = bn.shape[0]

    A_iv = to_interval_matrix(An.tolist())
    Q_iv = to_interval_matrix(Qn.tolist())
    x_iv = _ivec(xn)
    b_iv = _ivec(bn)
    lam_iv = _ivec(lam)

    # --- primal feasibility: s = b - A x >= 0 (rigorous) ------------------- #
    Ax = matvec(A_iv, x_iv)
    slack = [b_iv[i] - Ax[i] for i in range(m)]
    primal_feasible = all(s.lo >= 0.0 for s in slack)
    if not primal_feasible:
        raise CertificationError(
            "x is not rigorously primal feasible (A x <= b violated within rounding); "
            "certify at a strictly interior point (e.g. a looser barrier gap)"
        )

    # --- dual residual v = c + A^T lambda --------------------------------- #
    v = [
        Interval.point(float(cn[j])) + sum_intervals([A_iv[i][j] * lam_iv[i] for i in range(m)])
        for j in range(n)
    ]
    v_l1 = sum_intervals([vi.abs() for vi in v])
    v_linf = inf_norm_vector(v)

    # --- ||Q^{-1}||_inf via the Neumann certificate ----------------------- #
    try:
        b_approx = np.linalg.inv(Qn)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - singular Q
        raise CertificationError("Q is singular; a positive-definite Q is required") from exc
    neumann = neumann_inverse_norm_bound(Qn.tolist(), b_approx.tolist())
    if not neumann["certified"]:
        raise CertificationError(
            f"could not certify ||Q^-1|| (Neumann kappa = {neumann['kappa']:.3e} >= 1)"
        )
    inv_norm = float(neumann["inverse_norm_bound"])

    # v^T Q^{-1} v in [0, ||Q^-1||_inf * ||v||_1 * ||v||_inf] (rigorous).
    vqv = Interval(0.0, (Interval.point(inv_norm) * v_l1 * Interval.point(v_linf)).hi)
    b_dot_lam = sum_intervals([b_iv[i] * lam_iv[i] for i in range(m)])
    g_lower = Interval.point(-0.5) * vqv - b_dot_lam  # encloses g(lambda) <= f*

    # f0(x) = 1/2 x^T Q x + c^T x  (upper bound, x feasible).
    Qx = matvec(Q_iv, x_iv)
    xQx = sum_intervals([x_iv[j] * Qx[j] for j in range(n)])
    cTx = sum_intervals([Interval.point(float(cn[j])) * x_iv[j] for j in range(n)])
    f0 = Interval.point(0.5) * xQx + cTx

    enclosure = Interval(g_lower.lo, f0.hi)
    return Certificate(
        enclosure=enclosure,
        primal_feasible=primal_feasible,
        dual_residual_inf=v_linf,
        gap=enclosure.width,
    )


def _broadcast_box(x_lower: Any, x_upper: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    xl = np.broadcast_to(np.asarray(x_lower, dtype=float), (n,)).astype(float)
    xu = np.broadcast_to(np.asarray(x_upper, dtype=float), (n,)).astype(float)
    if not (np.all(np.isfinite(xl)) and np.all(np.isfinite(xu))):
        raise CertificationError(
            "an LP certificate needs finite variable bounds x_lower <= x <= x_upper "
            "that contain the feasible set (the box corrects for dual infeasibility)"
        )
    if np.any(xl > xu):
        raise CertificationError("x_lower must be <= x_upper componentwise")
    return xl, xu


def lp_dual_lower_bound(
    c: Any,
    A: Any,
    b: Any,
    dual: Any,
    *,
    A_eq: Any = None,
    b_eq: Any = None,
    eq_dual: Any = None,
    x_lower: Any,
    x_upper: Any,
) -> Interval:
    r"""Rigorous Neumaier-Shcherbina lower bound on the LP optimum ``f*``.

    Encloses the Lagrangian dual value ``min_{x in box} (c + A^T lambda +
    A_eq^T nu)^T x - b^T lambda - b_eq^T nu`` in outward-rounded interval
    arithmetic. The returned interval's ``lo`` is a **guaranteed** lower bound on
    ``f*`` for *any* ``lambda >= 0`` (clamped) and ``nu`` (sign-free), provided the
    box ``[x_lower, x_upper]`` contains the feasible set -- so the first-order
    penalty solver's approximate ``dual`` / ``eq_dual`` estimates are admissible.
    """
    cn = np.asarray(c, dtype=float)
    An = np.asarray(A, dtype=float)
    bn = np.asarray(b, dtype=float)
    lam = np.maximum(np.asarray(dual, dtype=float), 0.0)
    n = cn.shape[0]
    m = bn.shape[0]
    xl, xu = _broadcast_box(x_lower, x_upper, n)

    A_iv = to_interval_matrix(An.tolist())
    lam_iv = _ivec(lam)
    b_iv = _ivec(bn)
    has_eq = A_eq is not None and b_eq is not None
    Aeq_iv: list[list[Interval]] = []
    nu_iv: list[Interval] = []
    beq_iv: list[Interval] = []
    m_eq = 0
    if has_eq:
        Aeqn = np.asarray(A_eq, dtype=float)
        beqn = np.asarray(b_eq, dtype=float)
        m_eq = beqn.shape[0]
        nu = np.zeros(m_eq) if eq_dual is None else np.asarray(eq_dual, dtype=float)
        Aeq_iv = to_interval_matrix(Aeqn.tolist())
        nu_iv = _ivec(nu)
        beq_iv = _ivec(beqn)

    # dual-residual r_j = c_j + sum_i A[i,j] lambda_i + sum_k A_eq[k,j] nu_k.
    box = [Interval(float(xl[j]), float(xu[j])) for j in range(n)]
    terms: list[Interval] = []
    for j in range(n):
        r_j = Interval.point(float(cn[j])) + sum_intervals(
            [A_iv[i][j] * lam_iv[i] for i in range(m)]
        )
        if has_eq:
            r_j = r_j + sum_intervals([Aeq_iv[k][j] * nu_iv[k] for k in range(m_eq)])
        terms.append(r_j * box[j])  # min over the box is (r_j * box_j).lo

    base = -sum_intervals([b_iv[i] * lam_iv[i] for i in range(m)])
    if has_eq:
        base = base - sum_intervals([beq_iv[k] * nu_iv[k] for k in range(m_eq)])
    return base + sum_intervals(terms)


def certify_lp_optimum(
    c: Any,
    A: Any,
    b: Any,
    x: Any,
    dual: Any,
    *,
    A_eq: Any = None,
    b_eq: Any = None,
    eq_dual: Any = None,
    x_lower: Any,
    x_upper: Any,
) -> Certificate:
    r"""Rigorous interval enclosure of an LP optimum ``f*`` (complements the QP one).

    Lower bound: the verified :func:`lp_dual_lower_bound` (Neumaier-Shcherbina, valid
    for the solver's approximate multipliers). Upper bound: ``c^T x`` at the supplied
    primal ``x``, which must be rigorously feasible (``A x <= b`` and, if given,
    ``A_eq x = b_eq`` within rounding). ``x_lower`` / ``x_upper`` are required finite
    variable bounds containing the feasible set.

    Raises
    ------
    CertificationError
        If the box is not finite, or ``x`` is not rigorously primal feasible (then
        ``c^T x`` is not a valid upper bound -- use :func:`lp_dual_lower_bound` for
        the lower bound alone, e.g. with a separately decoded feasible point).
    """
    cn = np.asarray(c, dtype=float)
    An = np.asarray(A, dtype=float)
    bn = np.asarray(b, dtype=float)
    xn = np.asarray(x, dtype=float)
    n = cn.shape[0]
    m = bn.shape[0]

    lower = lp_dual_lower_bound(
        c, A, b, dual, A_eq=A_eq, b_eq=b_eq, eq_dual=eq_dual,
        x_lower=x_lower, x_upper=x_upper,
    )

    A_iv = to_interval_matrix(An.tolist())
    x_iv = _ivec(xn)
    b_iv = _ivec(bn)
    Ax = matvec(A_iv, x_iv)
    primal_feasible = all((b_iv[i] - Ax[i]).lo >= 0.0 for i in range(m))
    has_eq = A_eq is not None and b_eq is not None
    if has_eq:
        Aeqn = np.asarray(A_eq, dtype=float)
        beqn = np.asarray(b_eq, dtype=float)
        Aeq_iv = to_interval_matrix(Aeqn.tolist())
        beq_iv = _ivec(beqn)
        Aeqx = matvec(Aeq_iv, x_iv)
        eq_ok = all(
            (beq_iv[k] - Aeqx[k]).lo <= 0.0 <= (beq_iv[k] - Aeqx[k]).hi
            for k in range(beqn.shape[0])
        )
        primal_feasible = primal_feasible and eq_ok
    if not primal_feasible:
        raise CertificationError(
            "x is not rigorously primal feasible (A x <= b / A_eq x = b_eq violated "
            "within rounding); c^T x is then not a valid upper bound"
        )

    resid = cn + An.T @ np.maximum(np.asarray(dual, dtype=float), 0.0)
    if has_eq:
        nu = (
            np.zeros(np.asarray(b_eq).shape[0])
            if eq_dual is None
            else np.asarray(eq_dual, dtype=float)
        )
        resid = resid + np.asarray(A_eq, dtype=float).T @ nu
    dual_residual_inf = float(np.max(np.abs(resid)))

    cTx = sum_intervals([Interval.point(float(cn[j])) * x_iv[j] for j in range(n)])
    enclosure = Interval(lower.lo, cTx.hi)
    return Certificate(
        enclosure=enclosure,
        primal_feasible=True,
        dual_residual_inf=dual_residual_inf,
        gap=enclosure.width,
    )


__all__ = [
    "Certificate",
    "CertificationError",
    "certify_lp_optimum",
    "certify_qp_optimum",
    "lp_dual_lower_bound",
]
