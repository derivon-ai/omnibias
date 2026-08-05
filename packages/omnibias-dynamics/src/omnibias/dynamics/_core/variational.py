# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Validated variational / monodromy flow.

For an autonomous system :math:`y' = F(y)` the **fundamental** (variational)
matrix :math:`M(t) = \partial y(t)/\partial y(0)` solves the linear matrix ODE

.. math::

    M'(t) = DF(y(t))\, M(t), \qquad M(0) = I,

and its value after one period, the **monodromy matrix**, carries the Floquet
multipliers that decide a periodic orbit's stability.

This module propagates the state *and* its fundamental matrix rigorously, reusing
the QR-Lohner state flow (:mod:`omnibias.core.verified.lohner`) for the centre and
the **interval matrix exponential** for the per-step transition matrix.  The key
soundness fact is that for a *time-varying* generator :math:`A(t) \in [A]` (a
constant interval matrix) on a step, the time-:math:`h` solution operator of
:math:`M' = A(t) M` is enclosed by :math:`\exp([A]\,h)`: the Peano-Baker term of
order :math:`k` is an integral over a simplex of volume :math:`h^k/k!` of a
product :math:`A(t_1)\cdots A(t_k) \in [A]^k`, so it lies in
:math:`[A]^k h^k / k!`, and summing gives :math:`\exp([A] h)`.  We take
:math:`[A] = DF([Z])` over the step's a-priori enclosure :math:`[Z]`, so the
ordered product of the per-step enclosures encloses the true fundamental matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import nextafter

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.linalg import (
    IntervalMatrix,
    identity_matrix,
    inf_norm_matrix,
    matmul,
)
from omnibias.core.verified.lohner import (
    JacobianEnclosure,
    LohnerSet,
    interval_matrix_exp,
    lohner_step,
)
from omnibias.core.verified.ode import VectorField, _apriori_enclosure


def _scale(a: IntervalMatrix, h: float) -> IntervalMatrix:
    factor = Interval.point(h)
    return [[x * factor for x in row] for row in a]


def _transpose(a: IntervalMatrix) -> IntervalMatrix:
    rows, cols = len(a), len(a[0]) if a else 0
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def step_transition_matrix(
    field: VectorField,
    jac: JacobianEnclosure,
    state: LohnerSet,
    h: float,
    order: int,
) -> IntervalMatrix:
    r"""Rigorous enclosure of the step transition matrix ``exp(h DF([Z]))``.

    ``[Z]`` is the a-priori enclosure of the state over the step, so the returned
    interval matrix contains :math:`\partial y(t{+}h)/\partial y(t)` for the whole
    bundle of trajectories starting in ``state``.
    """
    box = state.to_box()
    z_encl = _apriori_enclosure(field, box, h)
    return interval_matrix_exp(_scale(jac(z_encl), h), order=max(order, 8))


@dataclass
class VariationalState:
    """A rigorously enclosed state together with its fundamental matrix ``M(t)``."""

    state: LohnerSet
    fundamental: IntervalMatrix
    time: float

    @classmethod
    def initial(cls, y0: Sequence[IntervalLike]) -> VariationalState:
        state = LohnerSet.from_box(y0)
        return cls(state, identity_matrix(len(state.center)), 0.0)

    def box(self) -> list[Interval]:
        """Outward-rounded axis-aligned enclosure of the state."""
        return self.state.to_box()


def variational_step(
    field: VectorField,
    jac: JacobianEnclosure,
    vstate: VariationalState,
    h: float,
    order: int,
) -> VariationalState:
    """One validated step of the coupled (state, fundamental-matrix) flow."""
    j_step = step_transition_matrix(field, jac, vstate.state, h, order)
    new_state = lohner_step(field, jac, vstate.state, h, order)
    new_fundamental = matmul(j_step, vstate.fundamental)
    return VariationalState(new_state, new_fundamental, vstate.time + h)


def variational_flow(
    field: VectorField,
    jac: JacobianEnclosure,
    y0: Sequence[IntervalLike],
    h: float,
    n_steps: int,
    order: int = 12,
) -> VariationalState:
    """Integrate the state and its fundamental matrix for ``n_steps`` Lohner steps."""
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    vstate = VariationalState.initial(y0)
    for _ in range(n_steps):
        vstate = variational_step(field, jac, vstate, h, order)
    return vstate


def monodromy_matrix(
    field: VectorField,
    jac: JacobianEnclosure,
    periodic_point: Sequence[IntervalLike],
    period: float,
    *,
    n_steps: int = 200,
    order: int = 12,
) -> IntervalMatrix:
    r"""Rigorous enclosure of the monodromy matrix ``M(T)`` over one ``period``.

    Integrates the variational flow from ``periodic_point`` for time ``period``
    and returns the fundamental matrix.  For a *true* periodic point the result
    encloses the monodromy whose eigenvalues are the Floquet multipliers; for an
    approximate one it still rigorously encloses ``M(T)`` of the flow started
    there (useful as the Jacobian of a shooting / return map).
    """
    if period <= 0.0:
        raise ValueError("period must be positive")
    h = period / n_steps
    return variational_flow(field, jac, periodic_point, h, n_steps, order).fundamental


# --------------------------------------------------------------------------- #
# Spectral read-outs of the monodromy (Floquet diagnostics).
# --------------------------------------------------------------------------- #
def monodromy_trace(m: IntervalMatrix) -> Interval:
    """Rigorous enclosure of ``trace(M)`` (sum of Floquet multipliers)."""
    acc = Interval.point(0.0)
    for i in range(len(m)):
        acc = acc + m[i][i]
    return acc


def monodromy_determinant(m: IntervalMatrix) -> Interval:
    r"""Rigorous enclosure of ``det(M)`` (product of Floquet multipliers).

    For a flow this equals :math:`\exp\!\int_0^T \operatorname{tr} DF(y(t))\,dt`
    (Liouville's formula), a sharp analytic oracle for tests.
    """
    n = len(m)
    if n == 0:
        return Interval.point(1.0)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    acc = Interval.point(0.0)
    for j in range(n):
        minor = [[m[i][k] for k in range(n) if k != j] for i in range(1, n)]
        term = m[0][j] * monodromy_determinant(minor)
        acc = acc + term if j % 2 == 0 else acc - term
    return acc


def spectral_radius_bound(m: IntervalMatrix) -> Interval:
    r"""Two-sided rigorous bracket of the spectral radius ``rho(M)``.

    Upper bound: ``rho <= min(||M||_1, ||M||_inf)`` (any induced norm dominates the
    spectral radius).  Lower bound: ``rho >= |det(M)|^{1/n}`` (the geometric mean
    of the eigenvalue moduli never exceeds the largest).  Both are sound for every
    matrix in the interval enclosure ``m``.

    The ``n``-th root needs care: ``float.__pow__`` rounds to nearest and the
    exponent ``1/n`` is itself inexact, so the naive ``base ** (1/n)`` can land an
    ulp *above* the true root -- which is the wrong direction for a lower bound and
    made this bracket miss the true spectral radius on roughly 1% of random point
    matrices. It is therefore stepped down until outward-rounded arithmetic proves
    ``root^n <= |det|``.
    """
    n = len(m)
    upper = min(inf_norm_matrix(m), inf_norm_matrix(_transpose(m)))
    det = monodromy_determinant(m).abs()
    if n <= 1:
        lower = det.lo
    else:
        base = det.lo
        lower = _nth_root_down(base, n) if base > 0.0 else 0.0
    return Interval(lower, upper)


def _nth_root_down(base: float, n: int) -> float:
    """The largest float ``r`` this can *prove* satisfies ``r**n <= base``."""
    root = base ** (1.0 / n)
    # A handful of steps at most; the seed is within a few ulp of the true root.
    for _ in range(64):
        if root <= 0.0 or Interval.point(root).pow_int(n).hi <= base:
            break
        root = nextafter(root, 0.0)
    return max(root, 0.0)


__all__ = [
    "VariationalState",
    "monodromy_determinant",
    "monodromy_matrix",
    "monodromy_trace",
    "spectral_radius_bound",
    "step_transition_matrix",
    "variational_flow",
    "variational_step",
]
