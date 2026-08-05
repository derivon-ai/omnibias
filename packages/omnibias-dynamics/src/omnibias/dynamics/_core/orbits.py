# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous periodic-orbit existence via the Krawczyk / radii-polynomial test.

A periodic point of period :math:`T` is a fixed point of the time-:math:`T` flow
map :math:`\Phi_T`, i.e. a zero of :math:`G(x) = \Phi_T(x) - x`.  We hand
:math:`G` and its Jacobian -- the **monodromy minus identity**, ``M(T) - I``,
both enclosed by the validated variational flow -- to the finite-dimensional
:func:`~omnibias.core.verified.kantorovich.krawczyk_certificate`.  If the
Krawczyk operator maps the trial ball strictly inside itself with contraction
factor :math:`< 1`, a **true** periodic point exists (and is unique) in that ball.

.. note::

   For an autonomous flow the time-:math:`T` map has the trivial Floquet
   multiplier :math:`1` along the flow direction, so in :math:`n \ge 2` dimensions
   periodic points of the *un-reduced* map are non-isolated and the test correctly
   declines (no contraction).  Apply it after a section reduction -- e.g. the
   scalar radial map of a rotationally-symmetric oscillator -- where the fixed
   point is isolated and the Jacobian ``M(T) - I`` is non-singular.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import KrawczykCertificate, krawczyk_certificate
from omnibias.core.verified.linalg import IntervalMatrix, identity_matrix, mat_sub
from omnibias.core.verified.lohner import JacobianEnclosure
from omnibias.core.verified.ode import VectorField
from omnibias.dynamics._core.variational import variational_flow


def _float_inverse(a: Sequence[Sequence[float]]) -> list[list[float]] | None:
    """Gauss-Jordan inverse of a small float matrix (``None`` if near-singular)."""
    n = len(a)
    aug = [[float(a[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]
    for col in range(n):
        pivot = col
        best = abs(aug[col][col])
        for r in range(col + 1, n):
            if abs(aug[r][col]) > best:
                best, pivot = abs(aug[r][col]), r
        if abs(aug[pivot][col]) < 1e-14:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [v / piv for v in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                aug[r] = [aug[r][k] - factor * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]


def _midpoint(m: IntervalMatrix) -> list[list[float]]:
    return [[x.mid for x in row] for row in m]


@dataclass(frozen=True)
class PeriodicOrbitCertificate:
    """A verified periodic point of the time-``period`` map (or a negative result)."""

    exists: bool
    period: float
    center: tuple[float, ...]
    enclosure: tuple[tuple[float, float], ...] | None
    krawczyk: KrawczykCertificate | None


def prove_periodic_orbit(
    field: VectorField,
    jac: JacobianEnclosure,
    x_bar: Sequence[float],
    period: float,
    *,
    n_steps: int = 200,
    order: int = 12,
    radii: Sequence[float] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2),
) -> PeriodicOrbitCertificate:
    r"""Prove a true periodic point of period ``period`` exists near ``x_bar``.

    Solves ``Phi_T(x) - x = 0`` with the Krawczyk test, the flow map and its
    Jacobian (``M(T) - I``) supplied by the validated variational flow.  Tries each
    radius in ``radii`` (ascending) and returns the first verified ball.
    """
    if period <= 0.0:
        raise ValueError("period must be positive")
    n = len(x_bar)
    h = period / n_steps

    def func(box: list[Interval]) -> list[Interval]:
        end = variational_flow(field, jac, box, h, n_steps, order).box()
        return [end[i] - box[i] for i in range(n)]

    def jacobian(box: list[Interval]) -> IntervalMatrix:
        m = variational_flow(field, jac, box, h, n_steps, order).fundamental
        return mat_sub(m, identity_matrix(n))

    # A diverging a-priori enclosure (over-large box, strongly unstable flow) raises
    # rather than returning -- treat any such failure as "not certified here".
    failure = (ValueError, RuntimeError, OverflowError, ZeroDivisionError)
    try:
        j_center = jacobian([Interval.point(v) for v in x_bar])
        a_inv = _float_inverse(_midpoint(j_center))
    except failure:
        a_inv = None
    if a_inv is None:
        return PeriodicOrbitCertificate(False, period, tuple(x_bar), None, None)

    for r in radii:
        try:
            cert = krawczyk_certificate(func, jacobian, list(x_bar), a_inv, r)
        except failure:
            continue
        if cert is not None:
            return PeriodicOrbitCertificate(
                exists=True,
                period=period,
                center=tuple(x_bar),
                enclosure=cert.enclosure,
                krawczyk=cert,
            )
    return PeriodicOrbitCertificate(False, period, tuple(x_bar), None, None)


__all__ = [
    "PeriodicOrbitCertificate",
    "prove_periodic_orbit",
]
