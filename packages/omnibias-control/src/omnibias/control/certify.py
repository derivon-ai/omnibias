# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous model-relative recoverable-set certificate for the CBF-QP filter.

The obstacle CBF row ``G_obs a <= h_obs`` is satisfiable by an *actuator-admissible*
action iff ``min_{||a||_inf <= a_max} G_obs a <= h_obs``, i.e. iff the recoverability
margin

.. math::
    \varphi(s) \;=\; h_{\text{obs}}(s) + a_{\max}\lVert G_{\text{obs}}(s)\rVert_1
    \;\ge\; 0 .

``phi(s) >= 0`` over a whole state box certifies the filter is feasible there -- the
**recoverable set**. This module builds a sound interval extension of ``phi`` for a
disc obstacle (closed form, so no interval NN propagation is needed) and runs
:func:`omnibias.verify.certified_minimize` (interval branch-and-bound) to obtain a
rigorous ``f_lower <= min phi``.

The certificate is **model-relative**: pass the (constant) model matrix ``g = M^{-1} B``
of the dynamics you are certifying -- the *learned* one for a learned-dynamics filter,
whose empirical error is reported separately. It is *not* a robustness guarantee against
the difference between the model and the true system.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from omnibias.control.problem import RecoverableCertificate
from omnibias.core.verified import Interval, sum_intervals

IntervalFn = Callable[[Sequence[Interval]], Interval]
IntervalGrad = Callable[[Sequence[Interval]], list[Interval]]


def _sign_iv(x: Interval) -> Interval:
    if x.lo > 0.0:
        return Interval.point(1.0)
    if x.hi < 0.0:
        return Interval.point(-1.0)
    return Interval(-1.0, 1.0)


def disc_obstacle_margin(
    center: Sequence[float],
    radius: float,
    gains: tuple[float, float],
    a_max: float,
    g: Any | None = None,
) -> tuple[IntervalFn, IntervalGrad]:
    r"""Interval extension (and interval gradient) of ``phi`` for a disc obstacle.

    For a relative-degree-2 disc barrier ``b(p) = ||p - c||^2 - r^2`` under
    control-affine dynamics ``qddot = g a`` (``g = M^{-1} B`` constant), the recoverable
    margin over the state ``s = [p, v]`` (each in ``R^d``) is

    .. math::
        \varphi(s) = 2\lVert v\rVert^2 + 2(\alpha_1+\alpha_2)(p-c)\!\cdot\!v
        + \alpha_1\alpha_2\,b + a_{\max}\sum_j\Big|2\sum_i (p_i - c_i) g_{ij}\Big|.

    Parameters
    ----------
    center, radius:
        Obstacle disc, ``center`` length ``d``.
    gains:
        Exponential-CBF class-K gains ``(alpha_1, alpha_2)``.
    a_max:
        Actuator box radius (``||a||_inf <= a_max``).
    g:
        Constant model matrix ``M^{-1} B`` (shape ``(d, m)``); ``None`` uses the
        identity ``I_d`` (a unit-mass double integrator, ``G_obs = 2 (p-c)``).

    Returns
    -------
    ``(phi, grad)`` closures over a state box ``[p_0, ..., p_{d-1}, v_0, ..., v_{d-1}]``
    of :class:`~omnibias.core.verified.Interval`; ``grad`` enables the monotonicity /
    mean-value accelerators of ``certified_minimize``.
    """
    if len(gains) != 2:
        raise ValueError("disc_obstacle_margin requires relative degree 2 (two gains)")
    cen = [float(c) for c in center]
    d = len(cen)
    g_mat = np.eye(d) if g is None else np.asarray(g, dtype=float)
    if g_mat.shape[0] != d:
        raise ValueError(f"g must have {d} rows to match center; got shape {g_mat.shape}")
    m = g_mat.shape[1]
    a1, a2 = float(gains[0]), float(gains[1])

    C = [Interval.point(c) for c in cen]
    Gp = [[Interval.point(float(g_mat[i, j])) for j in range(m)] for i in range(d)]
    two = Interval.point(2.0)
    four = Interval.point(4.0)
    k1 = Interval.point(2.0 * (a1 + a2))
    k2 = Interval.point(a1 * a2)
    amx = Interval.point(a_max)
    r2 = Interval.point(radius * radius)

    def _split(box: Sequence[Interval]) -> tuple[list[Interval], list[Interval]]:
        p, v = list(box[:d]), list(box[d:])
        dvec = [p[i] - C[i] for i in range(d)]
        return dvec, v

    def _u(dvec: list[Interval]) -> list[Interval]:
        return [two * sum_intervals([dvec[i] * Gp[i][j] for i in range(d)]) for j in range(m)]

    def phi(box: Sequence[Interval]) -> Interval:
        dvec, v = _split(box)
        b = sum_intervals([dvec[i] * dvec[i] for i in range(d)]) - r2
        dv = sum_intervals([dvec[i] * v[i] for i in range(d)])
        v2 = sum_intervals([v[i] * v[i] for i in range(d)])
        l1 = sum_intervals([uj.abs() for uj in _u(dvec)])
        return two * v2 + k1 * dv + k2 * b + amx * l1

    def grad(box: Sequence[Interval]) -> list[Interval]:
        dvec, v = _split(box)
        s = [_sign_iv(uj) for uj in _u(dvec)]
        d_p = [
            k1 * v[k]
            + (two * k2) * dvec[k]
            + (two * amx) * sum_intervals([s[j] * Gp[k][j] for j in range(m)])
            for k in range(d)
        ]
        d_v = [four * v[k] + k1 * dvec[k] for k in range(d)]
        return d_p + d_v

    return phi, grad


def certify_recoverable(
    phi: IntervalFn,
    box: Sequence[Interval],
    *,
    grad: IntervalGrad | None = None,
    tol: float = 1e-2,
    max_boxes: int = 40000,
) -> RecoverableCertificate | None:
    r"""Run interval branch-and-bound on ``phi`` over ``box`` (the recoverable-set proof).

    Returns a :class:`~omnibias.control.problem.RecoverableCertificate`, or ``None`` if
    :mod:`omnibias.verify` is not installed. ``result.certified`` (``f_lower >= 0``)
    proves the CBF-QP filter is feasible over the entire ``box``.
    """
    try:
        from omnibias.verify import certified_minimize
    except ImportError:
        return None
    res = certified_minimize(phi, list(box), tol=tol, max_boxes=max_boxes, grad=grad)
    return RecoverableCertificate(
        f_lower=res.f_lower,
        f_upper=res.f_upper,
        boxes_explored=res.boxes_explored,
        converged=res.converged,
    )


def certify_disc_recoverable(
    center: Sequence[float],
    radius: float,
    gains: tuple[float, float],
    a_max: float,
    position_ranges: Sequence[tuple[float, float]],
    vmax: float,
    *,
    g: Any | None = None,
    tol: float = 1e-2,
    max_boxes: int = 40000,
) -> RecoverableCertificate | None:
    r"""Convenience: certify a disc obstacle over ``position_ranges x [-vmax, vmax]^d``.

    Builds the :func:`disc_obstacle_margin` interval functions and the state box, then
    calls :func:`certify_recoverable`. ``position_ranges`` is a per-position-dim
    ``(lo, hi)`` sequence (length ``d``).
    """
    phi, grad = disc_obstacle_margin(center, radius, gains, a_max, g=g)
    d = len(center)
    if len(position_ranges) != d:
        raise ValueError(f"position_ranges must have {d} entries, got {len(position_ranges)}")
    box = [Interval(lo, hi) for (lo, hi) in position_ranges]
    box += [Interval(-vmax, vmax) for _ in range(d)]
    return certify_recoverable(phi, box, grad=grad, tol=tol, max_boxes=max_boxes)


__all__ = [
    "certify_disc_recoverable",
    "certify_recoverable",
    "disc_obstacle_margin",
]
