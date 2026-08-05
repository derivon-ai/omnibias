# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified finite-difference -> derivative extraction (the founding collapse).

Three registers, honestly labelled:

* **closed-form** -- :func:`certified_derivative_enclosure` returns the rigorous
  interval enclosure of the whole tower ``sigma^(0..m)(z)`` from
  :func:`omnibias.core.verified.sigma.sigma_tower_interval`; the ``m``-th entry is
  the guaranteed enclosure of the true ``m``-th derivative.
* **numerical** -- :func:`finite_difference_estimate` evaluates the multi-bias
  stencil ``sum_j s_j sigma(z + b_j)`` in plain ``float`` (the estimate that
  suffers the ``1/delta^m`` cancellation the closed-form tower avoids).
* **certified sandwich** -- :func:`certified_fd_error` proves
  ``|estimate - sigma^(m)(z)| <= error_bound`` with a rigorous Taylor-remainder
  enclosure of ``sigma^(m+p)`` over the whole stencil box, so the numerical
  estimate is *certified* to collapse into the closed-form enclosure as
  ``delta -> 0`` (order ``p``: 1 forward, 2 central).

This is the ``delta -> 0`` founding bias collapse (a smooth ``sigma^(K-1)``
derivative), never the ``beta -> inf`` feasibility penalty; do not conflate the
two senses (see the ``omnibias-dev-core-concepts`` skill).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import cos, cosh, erf, exp, factorial, log1p, sin, sqrt, tanh

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.difference._core.stencil import (
    Stencil,
    accuracy_order,
    offsets_exact,
    signs_exact,
    stencil_offsets,
    stencil_signs,
)

_SQRT2 = sqrt(2.0)

#: A derivative-enclosure oracle: ``deriv_bound(k, box)`` returns a rigorous
#: :class:`Interval` enclosure of ``f^(k)`` over ``box``. This is the seam that
#: decouples the certified-remainder engine from the built-in activation
#: dictionary -- :func:`omnibias.core.verified.sigma.sigma_tower_interval` is one
#: provider, but *any* function with a sound derivative-tower / interval-jet
#: enclosure (a hand-written analytic tower, a composed ``compose_jet`` tower, a
#: differentiated Taylor model, ...) can be certified through the same math.
DerivBound = Callable[[int, Interval], Interval]


def _sigma_float(name: str, z: float) -> float:
    """Plain-``float`` value of a supported activation (the numerical register)."""
    if name == "tanh":
        return tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + exp(-z))
    if name == "gaussian":
        return exp(-0.5 * z * z)
    if name == "sin":
        return sin(z)
    if name == "cos":
        return cos(z)
    if name == "silu":
        return z / (1.0 + exp(-z))
    if name == "softplus":
        # numerically stable ln(1 + e^z)
        return max(z, 0.0) + log1p(exp(-abs(z)))
    if name == "gelu":
        return z * 0.5 * (1.0 + erf(z / _SQRT2))
    if name == "sech":
        return 1.0 / cosh(z)
    raise ValueError(f"unsupported activation {name!r}")


@dataclass(frozen=True)
class DerivativeEnclosure:
    """Closed-form rigorous enclosure of ``sigma^(0..order)(z)``."""

    name: str
    order: int
    argument: Interval
    tower: tuple[Interval, ...]
    value: Interval
    label: str = "closed-form"


def certified_derivative_enclosure(
    name: str, z: IntervalLike, order: int
) -> DerivativeEnclosure:
    """Closed-form interval enclosure of the derivative tower ``sigma^(0..order)(z)``.

    ``z`` may be a scalar (an exact point) or an :class:`Interval` (a box); the
    returned ``value`` is the guaranteed enclosure of the true ``order``-th
    derivative over that box.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z_iv = Interval.from_value(z)
    tower = sigma_tower_interval(name, z_iv, order)
    return DerivativeEnclosure(name, order, z_iv, tower, tower[order])


@dataclass(frozen=True)
class DifferenceEstimate:
    """Numerical finite-difference estimate of ``sigma^(order)(z)``."""

    name: str
    order: int
    z: float
    delta: float
    stencil: str
    estimate: float
    label: str = "numerical"


def finite_difference_estimate(
    name: str, z: float, order: int, delta: float, stencil: Stencil = "central"
) -> DifferenceEstimate:
    """Numerical multi-bias stencil ``sum_j s_j sigma(z + b_j)`` (the FD estimate).

    This is the *naive* estimate: the signs scale like ``1/delta^order``, so for a
    very small ``delta`` it loses precision to catastrophic cancellation -- exactly
    the failure the closed-form tower sidesteps.
    """
    signs = stencil_signs(order, delta, stencil)
    offsets = stencil_offsets(order, delta, stencil)
    estimate = sum(
        s * _sigma_float(name, float(z) + b) for s, b in zip(signs, offsets, strict=True)
    )
    return DifferenceEstimate(name, order, float(z), float(delta), stencil, estimate)


@dataclass(frozen=True)
class FiniteDifferenceCertificate:
    """A certified sandwich linking the numerical estimate to the closed-form tower.

    ``enclosure`` rigorously contains the true ``sigma^(order)(z)`` (closed-form),
    and ``|estimate - sigma^(order)(z)| <= error_bound`` is certified by a Taylor
    remainder enclosed over the whole stencil box. Hence the true derivative lies
    in ``[estimate - error_bound, estimate + error_bound]`` *and* the estimate lies
    in ``enclosure`` widened by ``error_bound`` -- a self-checking sandwich.
    """

    name: str
    order: int
    z: float
    delta: float
    stencil: str
    estimate: float
    enclosure: Interval
    error_bound: float
    accuracy_order: int
    label: str = "closed-form + numerical"

    @property
    def certified(self) -> bool:
        """Whether the numerical estimate is consistent with the certified sandwich."""
        return (
            self.enclosure.lo - self.error_bound
            <= self.estimate
            <= self.enclosure.hi + self.error_bound
        )

    @property
    def true_derivative_interval(self) -> Interval:
        """The estimate widened by the certified error: a rigorous bracket of the truth."""
        return Interval(self.estimate - self.error_bound, self.estimate + self.error_bound)


def certified_fd_error_general(
    f_float: Callable[[float], float],
    deriv_bound: DerivBound,
    z: float,
    order: int,
    delta: float,
    stencil: Stencil = "central",
    *,
    name: str = "f",
) -> FiniteDifferenceCertificate:
    r"""Certify ``|finite_difference - f^(order)(z)| <= error_bound`` for **any** ``f``.

    The activation-agnostic core of :func:`certified_fd_error`: it takes a plain
    ``f_float`` (for the numerical stencil evaluation) and a :data:`DerivBound`
    oracle ``deriv_bound(k, box)`` enclosing ``f^(k)`` over a box, so it works for
    any function with a sound derivative-tower / interval-jet enclosure -- not just
    the nine built-in activations.

    The stencil kills every Taylor moment below the accuracy order, so the exact
    remainder is ``sum_j s_j b_j^{m+p} f^(m+p)(xi_j) / (m+p)!`` with each ``xi_j``
    in the stencil box. Enclosing ``f^(m+p)`` over that box gives the rigorous bound

    .. math::

        |\text{error}| \le \frac{1}{(m+p)!}\Big(\sum_j |s_j|\,|b_j|^{m+p}\Big)\,
            \max_{\text{box}} |f^{(m+p)}|,

    which is ``O(delta^p)`` (``p = 1`` forward, ``p = 2`` central).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    p = accuracy_order(stencil)
    z_iv = Interval.from_value(float(z))
    enclosure = deriv_bound(order, z_iv)

    # Numerical stencil estimate in plain float (the cancellation-prone register).
    float_signs = stencil_signs(order, delta, stencil)
    float_offsets = stencil_offsets(order, delta, stencil)
    estimate = sum(
        s * f_float(float(z) + b)
        for s, b in zip(float_signs, float_offsets, strict=True)
    )

    signs = signs_exact(order, Fraction(delta))
    offsets = offsets_exact(order, Fraction(delta), stencil)
    mp = order + p

    # Rigorous (outward) coefficient sum_j |s_j| |b_j|^{mp} / (mp)!.
    coeff = Interval.point(0.0)
    for s, b in zip(signs, offsets, strict=True):
        coeff = coeff + Interval.from_rational(abs(s)) * (Interval.from_rational(abs(b)) ** mp)
    coeff = coeff * Interval.from_rational(Fraction(1, factorial(mp)))

    # Box enclosing the whole stencil support [z + min b_j, z + max b_j].
    span = Interval(Interval.from_rational(min(offsets)).lo, Interval.from_rational(max(offsets)).hi)
    box = z_iv + span
    fmp = deriv_bound(mp, box)

    error_bound = (coeff * Interval(-fmp.mag, fmp.mag)).mag
    return FiniteDifferenceCertificate(
        name, order, float(z), float(delta), stencil, estimate, enclosure, error_bound, p
    )


def certified_fd_error(
    name: str, z: float, order: int, delta: float, stencil: Stencil = "central"
) -> FiniteDifferenceCertificate:
    r"""Certify ``|finite_difference - sigma^(order)(z)| <= error_bound``.

    A thin adapter over :func:`certified_fd_error_general` that binds the built-in
    activation dictionary: the float value comes from :func:`_sigma_float` and the
    derivative-over-box enclosures from
    :func:`omnibias.core.verified.sigma.sigma_tower_interval`. See the general
    function for the remainder math.
    """
    return certified_fd_error_general(
        lambda x: _sigma_float(name, x),
        lambda k, box: sigma_tower_interval(name, box, k)[k],
        z,
        order,
        delta,
        stencil,
        name=name,
    )


def sigma_deriv_bound(name: str) -> DerivBound:
    """The :data:`DerivBound` oracle for a built-in activation (from ``sigma_tower_interval``)."""
    return lambda k, box: sigma_tower_interval(name, box, k)[k]


__all__ = [
    "DerivBound",
    "DerivativeEnclosure",
    "DifferenceEstimate",
    "FiniteDifferenceCertificate",
    "certified_derivative_enclosure",
    "certified_fd_error",
    "certified_fd_error_general",
    "finite_difference_estimate",
    "sigma_deriv_bound",
]
