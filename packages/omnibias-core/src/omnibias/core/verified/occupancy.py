# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified Fermi-Dirac occupancy and thermodynamic potentials (theory 04-03).

The rigorous twin of :mod:`omnibias.core.occupancy`. Every enclosure here is
built from exactly two verified primitives -- :func:`~omnibias.core.verified
.sigma.sigma_tower_interval` (the sigmoid derivative tower) and
:func:`~omnibias.core.verified.transcend.softplus_iv` -- composed with
:class:`~omnibias.core.verified.interval.Interval` arithmetic, so nothing here
introduces a heuristic fudge factor.

Two deliverables sit on top of the identity replays:

* :func:`electron_count_enclosure` -- a certified
  ``integral dos(e) f(e) de`` over a finite band, feeding
  :func:`~omnibias.core.verified.quadrature.trapezoid_integral` a
  **tower-derived** second-derivative bound (via the exact Leibniz
  combination of the polynomial density-of-states derivatives with the
  sigmoid tower) rather than asking the caller for one. The composite
  trapezoid rule is used in preference to a higher-order rule (Simpson,
  Gauss-Legendre) specifically because it needs only a *second*-derivative
  bound: the Kantorovich Lipschitz constant below needs ``d^2 N/d mu^2``,
  and capping the combined mixed-partial order at ``mu_order + 2`` (at most
  ``4``, since ``mu_order <= 2``) keeps every sigmoid-tower enclosure this
  module evaluates at an order where a modest sub-box partition
  (:func:`_occ_mixed_bound_over_box`) stays tight; a fourth-derivative
  bound (Simpson) would push that combined order to ``6`` and demand a
  materially finer, more expensive partition for the same tightness.
* :func:`certified_chemical_potential` -- Newton on
  ``N(mu) - N_target = 0`` gated by
  :func:`~omnibias.core.verified.kantorovich.kantorovich_accept_step`, with
  the required ``lipschitz_df`` bound on ``d^2 N / d mu^2`` likewise read
  straight off the tower over the trial ball. An empty ball is a
  **reported halt** (``reason="empty"``), never an exception and never a
  silent fallback.

Two collapse senses are named (mirroring :mod:`omnibias.core.occupancy`) and
must not be conflated: the founding bias collapse (``delta -> 0``) never
appears here, and the ``beta -> inf`` founding **temperature collapse**
(feasibility sense) is evaluated only as an external reference value, never
as a registry-seeking claim of this module. Do not conflate the two.

:func:`sommerfeld_moment_enclosure` reads the Sommerfeld coefficients off
:func:`~omnibias.core.verified.dirichlet.zeta_even`, which is defined only at
even integers ``2m >= 2`` -- comfortably inside the ``Re(s) > 1`` wall that
:func:`~omnibias.core.verified.dirichlet.zeta_enclosure` guards explicitly,
and never widened. ``zeta_even`` is used in preference to the general
tail-bounded Dirichlet series because ``zeta(2m)`` is an exact rational
multiple of ``pi^{2m}`` at these points: a strictly tighter closed form than
a truncated-series enclosure, not merely an equivalent one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.verified.dirichlet import zeta_even
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.kantorovich import KantorovichAccept, kantorovich_accept_step
from omnibias.core.verified.quadrature import trapezoid_integral
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.core.verified.transcend import softplus_iv


def _reduced_argument_enclosure(
    beta: float, mu: IntervalLike, energy: IntervalLike
) -> Interval:
    """``z = -beta * (energy - mu)``, rigorously."""
    beta_iv = Interval.point(float(beta))
    return -beta_iv * (Interval.from_value(energy) - Interval.from_value(mu))


def _occ_mixed_enclosure(
    beta: float,
    mu: IntervalLike,
    energy: IntervalLike,
    *,
    mu_order: int,
    e_order: int,
) -> Interval:
    r"""``d^(e_order)/de^(e_order) [ beta^(mu_order) sigma^(mu_order)(z) ]``.

    Both the ``mu``- and ``e``-derivatives of the occupancy act on the same
    ``z``-tower; only the prefactor sign differs (``+beta`` per unit
    ``mu``-order, ``-beta`` per unit ``e``-order), since ``dz/dmu = +beta``
    and ``dz/de = -beta``. One :func:`sigma_tower_interval` call to the
    combined order supplies every mixed partial needed anywhere in this
    module.
    """
    if mu_order < 0 or e_order < 0:
        raise ValueError("mu_order and e_order must both be >= 0")
    z = _reduced_argument_enclosure(beta, mu, energy)
    top = mu_order + e_order
    tower = sigma_tower_interval("sigmoid", z, top)
    beta_iv = Interval.point(float(beta))
    prefactor = beta_iv.pow_int(mu_order) * (-beta_iv).pow_int(e_order)
    return prefactor * tower[top]


#: Target ``z``-width per sub-box in :func:`_occ_mixed_bound_over_box`. A
#: high-order sigmoid-tower polynomial evaluated by naive interval Horner over
#: a *wide* ``s = sigmoid(z)`` box suffers the classic dependency-problem
#: overestimation (the same variable occurs at several powers, so interval
#: arithmetic drops the correlation between them); narrowing the box back
#: down to this native scale of the tower recovers a tight -- and still fully
#: rigorous -- bound. ``0.05`` keeps the overestimation factor within about
#: ``2x`` for every combined mixed-partial order this module evaluates
#: (at most ``4``: ``mu_order <= 2`` plus the trapezoid rule's own
#: second-``e``-derivative remainder order).
_BOUND_TARGET_Z_WIDTH: float = 0.05
#: Floor on the sub-box count so a degenerate (near-zero-width) box is still
#: hulled at least once.
_BOUND_MIN_SUBDIVISIONS: int = 8
#: Ceiling on the sub-box count so a pathologically wide caller-supplied
#: domain cannot make a single certified call unboundedly expensive.
_BOUND_MAX_SUBDIVISIONS: int = 4096


def _occ_mixed_bound_over_box(
    beta: float,
    mu: IntervalLike,
    box: Interval,
    *,
    mu_order: int,
    e_order: int,
) -> Interval:
    """Tight enclosure of :func:`_occ_mixed_enclosure` over the whole ``box``.

    Evaluating the sigmoid-tower polynomial directly on a wide ``box`` is
    still *sound* but can be extremely loose (see the dependency-problem
    note on :data:`_BOUND_TARGET_Z_WIDTH`). This instead partitions ``box``
    into narrow sub-boxes, encloses each one, and takes the hull -- a valid
    enclosure of the whole box is exactly the hull of valid enclosures of a
    partition of it, so this only *tightens* the result, never loses
    soundness.
    """
    lo, hi = box.lo, box.hi
    z_width = float(beta) * (hi - lo)
    n_sub = max(
        _BOUND_MIN_SUBDIVISIONS,
        min(_BOUND_MAX_SUBDIVISIONS, math.ceil(abs(z_width) / _BOUND_TARGET_Z_WIDTH) or 1),
    )
    h = (hi - lo) / n_sub
    pieces = [
        _occ_mixed_enclosure(
            beta, mu, Interval(lo + i * h, lo + (i + 1) * h), mu_order=mu_order, e_order=e_order
        )
        for i in range(n_sub)
    ]
    return Interval.hull(*pieces)


def occupancy_enclosure(
    beta: float, mu: IntervalLike, energy: IntervalLike, *, order: int = 0
) -> tuple[Interval, ...]:
    """Enclosure of ``(f(e), df/de, ..., d^order f / de^order)``."""
    return tuple(
        _occ_mixed_enclosure(beta, mu, energy, mu_order=0, e_order=n)
        for n in range(order + 1)
    )


def occupancy_mu_enclosure(
    beta: float, mu: IntervalLike, energy: IntervalLike, *, order: int = 0
) -> tuple[Interval, ...]:
    """Enclosure of ``(f(e), df/dmu, ..., d^order f / dmu^order)``."""
    return tuple(
        _occ_mixed_enclosure(beta, mu, energy, mu_order=n, e_order=0)
        for n in range(order + 1)
    )


def entropy_enclosure(
    beta: float, mu: IntervalLike, energy: IntervalLike, *, order: int = 0
) -> tuple[Interval, ...]:
    """Enclosure of ``(s(e), ds/de, ..., d^order s / de^order)``.

    Mirrors :func:`omnibias.core.occupancy._entropy_z_tower`, rigorously.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = _reduced_argument_enclosure(beta, mu, energy)
    sigma_tower = sigma_tower_interval("sigmoid", z, order)
    softplus_val = softplus_iv(z)
    s_tower = [softplus_val - z * sigma_tower[0]]
    for n in range(1, order + 1):
        s_tower.append(
            -(z * sigma_tower[n] + Interval.point(float(n - 1)) * sigma_tower[n - 1])
        )
    neg_beta = Interval.point(-float(beta))
    return tuple(neg_beta.pow_int(n) * s_tower[n] for n in range(order + 1))


def grand_potential_enclosure(
    beta: float, mu: IntervalLike, energy: IntervalLike
) -> Interval:
    """Enclosure of ``omega(z) = -(1/beta) softplus(z)``."""
    z = _reduced_argument_enclosure(beta, mu, energy)
    return (-softplus_iv(z)) / float(beta)


def occupancy_window_enclosure(
    beta: float, mu: IntervalLike, e_lo: IntervalLike, e_hi: IntervalLike
) -> Interval:
    """Enclosure of the exact constant-density-of-states electron count in
    ``[e_lo, e_hi]``; see :func:`omnibias.core.occupancy.occupancy_window`."""
    z_lo = _reduced_argument_enclosure(beta, mu, e_lo)
    z_hi = _reduced_argument_enclosure(beta, mu, e_hi)
    return (softplus_iv(z_lo) - softplus_iv(z_hi)) / float(beta)


@dataclass(frozen=True)
class PolynomialDensityOfStates:
    r"""A polynomial density of states ``g(e) = sum_k coefficients[k] e^k``.

    Kept polynomial (rather than an arbitrary caller-supplied derivative
    oracle) so every derivative used by :func:`electron_count_enclosure`
    and :func:`certified_chemical_potential` is **exact and closed form**
    -- zero above :attr:`degree` -- which is what lets those two functions
    derive their own quadrature error bound instead of asking the caller
    for one. A constant density of states is ``PolynomialDensityOfStates
    ((g0,))``.
    """

    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("coefficients must be non-empty")
        object.__setattr__(
            self, "coefficients", tuple(float(c) for c in self.coefficients)
        )

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def value_enclosure(self, energy: IntervalLike) -> Interval:
        """Enclosure of ``g(e)``."""
        e = Interval.from_value(energy)
        acc = Interval.point(self.coefficients[-1])
        for c in reversed(self.coefficients[:-1]):
            acc = acc * e + Interval.point(c)
        return acc

    def derivative_enclosure(self, order: int, energy: IntervalLike) -> Interval:
        """Enclosure of ``g^(order)(e)``; exactly ``0`` once ``order > degree``."""
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if order > self.degree:
            return Interval.point(0.0)
        deriv_coeffs = [
            self.coefficients[k] * math.perm(k, order)
            for k in range(order, self.degree + 1)
        ]
        e = Interval.from_value(energy)
        acc = Interval.point(deriv_coeffs[-1])
        for c in reversed(deriv_coeffs[:-1]):
            acc = acc * e + Interval.point(c)
        return acc


def constant_density_of_states(g0: float) -> PolynomialDensityOfStates:
    """``g(e) = g0`` for all ``e`` (``g0 > 0``)."""
    if not (float(g0) > 0.0):
        raise ValueError(f"g0 must be > 0, got {g0}")
    return PolynomialDensityOfStates((float(g0),))


def _mixed_integrand_e_derivative_enclosure(
    beta: float,
    mu: IntervalLike,
    dos: PolynomialDensityOfStates,
    *,
    mu_order: int,
    e_order: int,
    box: Interval,
) -> Interval:
    r"""``d^(e_order)/de^(e_order) [ dos(e) * beta^(mu_order) sigma^(mu_order)(z) ]``
    over the whole ``box``, via the exact Leibniz product rule.

    ``dos`` is polynomial, so its own derivative enclosure
    (:meth:`PolynomialDensityOfStates.derivative_enclosure`) is exact; the
    occupancy factor's ``e``-derivatives reuse the single mixed sigmoid
    tower in :func:`_occ_mixed_enclosure`, tightened over the (possibly
    wide) ``box`` by :func:`_occ_mixed_bound_over_box`.
    """
    terms = [
        Interval.point(float(math.comb(e_order, k)))
        * dos.derivative_enclosure(k, box)
        * _occ_mixed_bound_over_box(beta, mu, box, mu_order=mu_order, e_order=e_order - k)
        for k in range(e_order + 1)
    ]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _mu_weighted_integral_enclosure(
    beta: float,
    mu: IntervalLike,
    dos: PolynomialDensityOfStates,
    e_lo: float,
    e_hi: float,
    *,
    mu_order: int,
    panels: int = 32,
) -> Interval:
    r"""Certified ``integral_{e_lo}^{e_hi} dos(e) * beta^(mu_order)
    sigma^(mu_order)(z(e)) de``, e.g. ``N(mu)`` at ``mu_order=0`` and
    ``dN/dmu`` at ``mu_order=1``.

    Feeds :func:`~omnibias.core.verified.quadrature.trapezoid_integral` its
    own second-``e``-derivative bound over ``[e_lo, e_hi]``, derived in
    closed form via :func:`_mixed_integrand_e_derivative_enclosure` (see
    the module docstring for why the trapezoid rule, needing only a
    second-derivative bound, is used here rather than a higher-order rule).
    The default ``panels=32`` is chosen for tightness, not stability: the
    trapezoid *remainder* term already uses a rigorous tower-derived bound
    at any panel count, but that bound's own magnitude is fixed by
    :data:`_BOUND_TARGET_Z_WIDTH` independent of ``panels``, so the
    remainder shrinks as ``O(1/panels^2)`` while the (already-sound)
    ``panels=1`` enclosure can be needlessly wide; more panels only ever
    tighten the returned interval, never change its soundness.
    """
    lo, hi = float(e_lo), float(e_hi)
    if lo > hi:
        raise ValueError(f"requires e_lo <= e_hi, got e_lo={lo}, e_hi={hi}")
    n_panels = max(1, int(panels))
    box = Interval(lo, hi)
    second_bound = _mixed_integrand_e_derivative_enclosure(
        beta, mu, dos, mu_order=mu_order, e_order=2, box=box
    )
    h = (hi - lo) / n_panels
    node_values = []
    for k in range(n_panels + 1):
        node = Interval.point(lo + k * h)
        weight = _occ_mixed_enclosure(beta, mu, node, mu_order=mu_order, e_order=0)
        node_values.append(dos.value_enclosure(node) * weight)

    return trapezoid_integral(node_values, lo, hi, second_bound)


def electron_count_enclosure(
    beta: float,
    mu: IntervalLike,
    dos: PolynomialDensityOfStates,
    e_lo: float,
    e_hi: float,
    *,
    panels: int = 32,
) -> Interval:
    """Certified ``integral_{e_lo}^{e_hi} dos(e) f(e) de``.

    The second-derivative bound the certified quadrature needs is derived
    from the tower itself (see :func:`_mu_weighted_integral_enclosure`);
    the caller never supplies a heuristic bound.
    """
    return _mu_weighted_integral_enclosure(
        beta, mu, dos, e_lo, e_hi, mu_order=0, panels=panels
    )


def certified_chemical_potential(
    beta: float,
    dos: PolynomialDensityOfStates,
    e_lo: float,
    e_hi: float,
    n_target: float,
    mu_bar: float,
    *,
    r_max: float,
    panels: int = 32,
) -> KantorovichAccept:
    r"""Certify a unique chemical potential solving ``N(mu) = n_target``.

    Runs :func:`~omnibias.core.verified.kantorovich.kantorovich_accept_step`
    on ``F(mu) = N(mu) - n_target`` (:func:`electron_count_enclosure`) with
    the exact Jacobian ``dN/dmu`` (:func:`_mu_weighted_integral_enclosure`
    at ``mu_order=1``) and a ``lipschitz_df`` bound on ``d^2 N/d mu^2``
    enclosed over the whole trial ball ``[mu_bar - r_max, mu_bar + r_max]``
    (``mu_order=2``) -- both tower-derived, never caller-supplied fudge
    factors. Returns the :class:`KantorovichAccept` verdict verbatim: an
    empty ball (``reason="empty"``) is a **reported halt**, not a failure
    and not silently retried.
    """
    if r_max <= 0.0:
        raise ValueError(f"r_max must be > 0, got {r_max}")
    mu_bar = float(mu_bar)
    n_target = float(n_target)

    def func(xs: list[Interval]) -> list[Interval]:
        (mu_iv,) = xs
        return [
            _mu_weighted_integral_enclosure(
                beta, mu_iv, dos, e_lo, e_hi, mu_order=0, panels=panels
            )
            - Interval.point(n_target)
        ]

    def jacobian(xs: list[Interval]) -> list[list[Interval]]:
        (mu_iv,) = xs
        return [
            [
                _mu_weighted_integral_enclosure(
                    beta, mu_iv, dos, e_lo, e_hi, mu_order=1, panels=panels
                )
            ]
        ]

    d1_at_bar = jacobian([Interval.point(mu_bar)])[0][0].mid
    if d1_at_bar == 0.0:
        return KantorovichAccept(False, None, "bounds_failed")
    a_inv = [[1.0 / d1_at_bar]]

    mu_box = Interval(mu_bar - float(r_max), mu_bar + float(r_max))
    d2_box = _mu_weighted_integral_enclosure(
        beta, mu_box, dos, e_lo, e_hi, mu_order=2, panels=panels
    )
    lipschitz_df = d2_box.mag

    return kantorovich_accept_step(
        func,
        jacobian,
        a_inv,
        [mu_bar],
        lipschitz_df=lipschitz_df,
        r_max=float(r_max),
        claim=(
            "unique zero of N(mu) - n_target for a non-interacting Fermi "
            "occupancy over a finite band [e_lo, e_hi]; not a continuum "
            "thermodynamic-limit claim"
        ),
    )


def sommerfeld_moment_enclosure(order: int) -> Interval:
    """Enclosure of :func:`omnibias.core.occupancy.sommerfeld_moment`.

    Closed form via :func:`~omnibias.core.verified.dirichlet.zeta_even`
    for even ``order = 2n >= 2``; exactly ``0`` (odd) or ``1`` (``order=0``)
    otherwise -- see the module docstring for why ``zeta_even`` is used in
    preference to the general tail-bounded Dirichlet series.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        return Interval.point(1.0)
    if order % 2 == 1:
        return Interval.point(0.0)
    n = order // 2
    zeta_2n = zeta_even(n)
    eta_2n = (
        Interval.point(1.0) - Interval.from_rational(Fraction(1, 2 ** (2 * n - 1)))
    ) * zeta_2n
    return Interval.point(float(math.factorial(order)) * 2.0) * eta_2n


def sommerfeld_coefficient_enclosure(n: int) -> Interval:
    """Enclosure of :func:`omnibias.core.occupancy.sommerfeld_coefficient`."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return sommerfeld_moment_enclosure(2 * n) / float(math.factorial(2 * n))


__all__ = [
    "PolynomialDensityOfStates",
    "certified_chemical_potential",
    "constant_density_of_states",
    "electron_count_enclosure",
    "entropy_enclosure",
    "grand_potential_enclosure",
    "occupancy_enclosure",
    "occupancy_mu_enclosure",
    "occupancy_window_enclosure",
    "sommerfeld_coefficient_enclosure",
    "sommerfeld_moment_enclosure",
]
