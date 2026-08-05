# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified finite-time Lyapunov-exponent bounds.

The leading **finite-time Lyapunov exponent** over :math:`[0, T]` is

.. math::

    \lambda(T) = \frac1T \ln \sigma_{\max}\big(M(T)\big),

the largest singular value of the fundamental matrix.  This module brackets it
rigorously from a validated variational flow (:mod:`.variational`):

* **upper bound** -- :math:`\sigma_{\max}(M) \le \sqrt{\|M\|_1\,\|M\|_\infty}`
  (the Holder interpolation bound on the spectral / 2-norm), so
  :math:`\lambda(T) \le \frac1T \ln\sqrt{\|M\|_1\|M\|_\infty}`;
* **lower bound** -- for *any* probe direction :math:`v`,
  :math:`\sigma_{\max}(M) \ge \|Mv\|_2/\|v\|_2`, so
  :math:`\lambda(T) \ge \frac1T \ln(\|Mv\|_2/\|v\|_2)`.

Both endpoints are computed in outward-rounded interval arithmetic, so the
returned bracket provably contains the true finite-time exponent of *every*
trajectory in the initial box.  On a periodic orbit, taking :math:`T` equal to
the period brackets the magnitude of the leading Floquet exponent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.linalg import IntervalMatrix, inf_norm_matrix, matvec
from omnibias.core.verified.lohner import JacobianEnclosure
from omnibias.core.verified.ode import VectorField
from omnibias.core.verified.transcend import ln_iv
from omnibias.dynamics._core.variational import _transpose, variational_flow


@dataclass(frozen=True)
class LyapunovBounds:
    """A rigorous two-sided bracket of the leading finite-time Lyapunov exponent."""

    lower: float
    upper: float
    time: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


def _two_norm_lower(vec: Sequence[Interval]) -> float:
    """Rigorous lower bound on the Euclidean norm of an interval vector."""
    acc = Interval.point(0.0)
    for x in vec:
        acc = acc + x.pow_int(2)
    return acc.sqrt().lo


def _sigma_max_upper(m: IntervalMatrix) -> Interval:
    """Upper enclosure of ``sigma_max(M)`` via ``sqrt(||M||_1 ||M||_inf)``."""
    one_norm = inf_norm_matrix(_transpose(m))
    inf_norm = inf_norm_matrix(m)
    return (Interval.point(one_norm) * Interval.point(inf_norm)).sqrt()


def certified_lyapunov_exponent(
    field: VectorField,
    jac: JacobianEnclosure,
    y0: Sequence[IntervalLike],
    *,
    time: float,
    n_steps: int = 200,
    order: int = 12,
    probe: Sequence[float] | None = None,
) -> LyapunovBounds:
    r"""Rigorously bracket ``(1/T) ln sigma_max(M(T))`` for the flow from ``y0``.

    ``probe`` is the direction used for the lower bound (default: the first
    coordinate axis).  A larger ``n_steps`` tightens the variational flow.
    """
    if time <= 0.0:
        raise ValueError("time must be positive")
    h = time / n_steps
    vf = variational_flow(field, jac, y0, h, n_steps, order)
    m = vf.fundamental
    n = len(m)
    inv_t = Interval.point(time).reciprocal()  # rigorous outward 1/T

    sigma_up = _sigma_max_upper(m)
    if sigma_up.lo <= 0.0:
        raise ValueError("degenerate fundamental matrix; cannot bound the exponent")
    upper = (ln_iv(Interval(max(sigma_up.lo, 1e-300), sigma_up.hi)) * inv_t).hi

    probe_vec = list(probe) if probe is not None else [1.0 if i == 0 else 0.0 for i in range(n)]
    if len(probe_vec) != n:
        raise ValueError(f"probe must have length {n}")
    mv = matvec(m, [Interval.point(c) for c in probe_vec])
    num_lo = _two_norm_lower(mv)
    den = Interval.point(0.0)
    for c in probe_vec:
        den = den + Interval.point(c).pow_int(2)
    den_hi = den.sqrt().hi
    if num_lo <= 0.0 or den_hi <= 0.0:
        lower = float("-inf")
    else:
        ratio = Interval.point(num_lo) * Interval.point(den_hi).reciprocal()
        lower = (ln_iv(Interval(max(ratio.lo, 1e-300), ratio.hi)) * inv_t).lo
    return LyapunovBounds(lower=lower, upper=upper, time=time)


__all__ = [
    "LyapunovBounds",
    "certified_lyapunov_exponent",
]
