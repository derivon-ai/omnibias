# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""QR-Lohner validated flow: rigorous ODE integration without the wrapping blow-up.

Stepping a box through a rotation with plain interval arithmetic is a disaster:
each step re-encloses a rotated box in an axis-aligned one, and the
over-approximation compounds *exponentially* even though the true solution set
stays bounded -- the **wrapping effect**.  Lohner's trick is to carry the set in a
co-moving orthonormal frame,

.. math::

    S_j = x_j \;\oplus\; Q_j\, r_j,

with ``x_j`` a point, ``Q_j`` an orthogonal matrix (refreshed by a QR
factorisation each step) and ``r_j`` a small box.  Because ``r_j`` lives in the
rotating frame, the box does **not** grow under rotation.

The propagation needs the Jacobian of the time-``h`` flow.  This module encloses
it with a rigorous **interval matrix exponential** of ``h * DF([Z])`` over the
step's a-priori enclosure ``[Z]`` (exact for linear systems, a sound C^1 bound for
nonlinear ones), and reuses the validated Taylor step from
:mod:`omnibias.core.verified.ode` for the centre.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import factorial, sqrt

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.linalg import (
    IntervalMatrix,
    identity_matrix,
    inf_norm_matrix,
    matmul,
    matvec,
    to_interval_matrix,
)
from omnibias.core.verified.ode import TaylorSeries, VectorField, _apriori_enclosure, _step

#: A Jacobian enclosure: maps a box to an interval matrix bounding ``DF`` on it.
JacobianEnclosure = Callable[[Sequence[Interval]], IntervalMatrix]


# --------------------------------------------------------------------------- #
# Small float / interval matrix helpers (kept local to avoid widening linalg).
# --------------------------------------------------------------------------- #
def _mat_add(a: IntervalMatrix, b: IntervalMatrix) -> IntervalMatrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]


def _scale_matrix(a: IntervalMatrix, s: IntervalLike) -> IntervalMatrix:
    factor = Interval.from_value(s)
    return [[x * factor for x in row] for row in a]


def _midpoint(a: IntervalMatrix) -> list[list[float]]:
    return [[x.mid for x in row] for row in a]


def _transpose_float(q: Sequence[Sequence[float]]) -> list[list[float]]:
    n, m = len(q), len(q[0])
    return [[q[i][j] for i in range(n)] for j in range(m)]


def interval_matrix_exp(m: IntervalMatrix, order: int = 12) -> IntervalMatrix:
    r"""Rigorous enclosure of ``exp(M)`` for an interval matrix ``M``.

    Sums ``sum_{k=0}^{order} M^k / k!`` and adds an entrywise Lagrange remainder
    ``sum_{k>order} ||M||^k / k!`` (valid because ``|(M^k)_{ij}| <= ||M||_inf^k``).
    Requires ``||M||_inf < order + 2`` for a convergent tail bound.
    """
    n = len(m)
    acc = identity_matrix(n)
    term = identity_matrix(n)  # M^0 / 0!
    for k in range(1, order + 1):
        term = _scale_matrix(matmul(term, m), Interval.from_rational(Fraction(1, k)))
        acc = _mat_add(acc, term)
    norm = inf_norm_matrix(m)
    ratio = norm / (order + 2)
    if ratio >= 1.0:
        raise ValueError(f"||M||={norm!r} too large for order {order}; reduce the step size")
    tail = (
        Interval.point(norm).pow_int(order + 1)
        / Interval.from_rational(factorial(order + 1))
        / (Interval.point(1.0) - Interval.point(ratio))
    )
    t = tail.hi
    rem = Interval(-t, t)
    return [[acc[i][j] + rem for j in range(n)] for i in range(n)]


def qr_gram_schmidt(a: Sequence[Sequence[float]]) -> list[list[float]]:
    """Orthonormal ``Q`` (as ``Q[i][j]``) from modified Gram-Schmidt on ``a``'s columns.

    Falls back to the identity for a (numerically) rank-deficient input so the
    Lohner step always has a valid frame.

    Rigour caveat (recorded assumption, **not** a discharged theorem): ``Q`` is
    computed in plain ``float`` and is orthonormal only to working precision. The
    QR-Lohner step uses ``Qᵀ`` in place of the true inverse ``Q⁻¹`` (see
    :func:`lohner_step`), which is exact only for a perfectly orthogonal ``Q``. The
    residual non-orthogonality ``||QᵀQ - I||`` (~1e-15 for modified Gram-Schmidt)
    is **assumed negligible** here and is *not* separately interval-bounded. A
    fully rigorous implementation would either enclose ``Q⁻¹`` by verified Gaussian
    elimination or inflate the box ``r`` by a certified bound on that defect; long
    nonlinear integrations should be treated with that caveat in mind.
    """
    n = len(a)
    cols = [[float(a[i][j]) for i in range(n)] for j in range(n)]
    q_cols: list[list[float]] = []
    for j in range(n):
        v = cols[j][:]
        for q in q_cols:
            dot = sum(q[i] * v[i] for i in range(n))
            v = [v[i] - dot * q[i] for i in range(n)]
        norm = sqrt(sum(vi * vi for vi in v))
        if norm < 1e-12:
            return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        q_cols.append([vi / norm for vi in v])
    return [[q_cols[j][i] for j in range(n)] for i in range(n)]


@dataclass
class LohnerSet:
    r"""A set ``center + Q r``: point centre, orthogonal frame ``Q``, box ``r``."""

    center: list[Interval]
    q: list[list[float]]
    r: list[Interval]

    @classmethod
    def from_box(cls, box: Sequence[IntervalLike]) -> LohnerSet:
        ivs = [Interval.from_value(b) for b in box]
        n = len(ivs)
        center = [Interval.point(iv.mid) for iv in ivs]
        r = [iv - Interval.point(iv.mid) for iv in ivs]
        q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return cls(center, q, r)

    def to_box(self) -> list[Interval]:
        """Outward-rounded axis-aligned enclosure of the represented set."""
        qr = matvec(to_interval_matrix(self.q), self.r)
        return [self.center[i] + qr[i] for i in range(len(self.center))]

    def width(self) -> float:
        """The widest component of the enclosing box (a wrapping diagnostic)."""
        return max(iv.width for iv in self.to_box())


def lohner_step(
    field: VectorField,
    jac: JacobianEnclosure,
    state: LohnerSet,
    h: float,
    order: int,
) -> LohnerSet:
    """One QR-Lohner step of size ``h`` for ``y' = field(y)`` with Jacobian ``jac``."""
    n = len(state.center)
    box = state.to_box()
    z_encl = _apriori_enclosure(field, box, h)
    # Centre: validated Taylor flow; keep the point, push its spread into the box.
    c_next = _step(field, state.center, h, order)
    xc = [Interval.point(iv.mid) for iv in c_next]
    z_rem = [c_next[i] - xc[i] for i in range(n)]
    # Variational Jacobian of the step, enclosed via the interval matrix exp.
    j_step = interval_matrix_exp(_scale_matrix(jac(z_encl), h), order=max(order, 8))
    b = matmul(j_step, to_interval_matrix(state.q))
    q_new = qr_gram_schmidt(_midpoint(b))
    q_new_t = to_interval_matrix(_transpose_float(q_new))
    r_lin = matvec(matmul(q_new_t, b), state.r)
    r_rem = matvec(q_new_t, z_rem)
    r_new = [r_lin[i] + r_rem[i] for i in range(n)]
    return LohnerSet(xc, q_new, r_new)


def linear_field(a: Sequence[Sequence[float]]) -> VectorField:
    """The autonomous linear vector field ``y' = A y`` as a :data:`VectorField`."""
    a_iv = to_interval_matrix(a)

    def field(series: list[TaylorSeries]) -> list[TaylorSeries]:
        order = series[0].order
        out: list[TaylorSeries] = []
        for i in range(len(a_iv)):
            acc = TaylorSeries.constant(0.0, order)
            for j in range(len(a_iv)):
                acc = acc + series[j] * a_iv[i][j]
            out.append(acc)
        return out

    return field


def constant_jacobian(a: Sequence[Sequence[float]]) -> JacobianEnclosure:
    """Jacobian enclosure for a linear field (the constant matrix ``A``)."""
    a_iv = to_interval_matrix(a)

    def jac(_box: Sequence[Interval]) -> IntervalMatrix:
        return [row[:] for row in a_iv]

    return jac


def lohner_flow(
    field: VectorField,
    jac: JacobianEnclosure,
    y0: Sequence[IntervalLike],
    h: float,
    n_steps: int,
    order: int = 12,
) -> LohnerSet:
    """Integrate ``y' = field(y)`` for ``n_steps`` QR-Lohner steps; return the final set."""
    state = LohnerSet.from_box(y0)
    for _ in range(n_steps):
        state = lohner_step(field, jac, state, h, order)
    return state


def naive_interval_flow(
    a: Sequence[Sequence[float]],
    y0: Sequence[IntervalLike],
    h: float,
    n_steps: int,
    order: int = 12,
) -> list[Interval]:
    """Plain interval stepping of a *linear* flow -- the wrapping-prone baseline."""
    j_step = interval_matrix_exp(_scale_matrix(to_interval_matrix(a), h), order=max(order, 8))
    box = [Interval.from_value(v) for v in y0]
    for _ in range(n_steps):
        box = matvec(j_step, box)
    return box


__all__ = [
    "JacobianEnclosure",
    "LohnerSet",
    "constant_jacobian",
    "interval_matrix_exp",
    "linear_field",
    "lohner_flow",
    "lohner_step",
    "naive_interval_flow",
    "qr_gram_schmidt",
]
