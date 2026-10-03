# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""A fixed-box certified zero count for the genuine de Bruijn-Newman ``H_t``.

**Scope, read first.** This module produces exactly one kind of result: a
sound argument-principle zero *count* of the real de Bruijn-Newman function
``H_t`` inside one declared, finite complex rectangle, at one declared,
finite truncation and series-term budget. It does **not** produce a bound on
the de Bruijn-Newman constant ``Lambda``, and it never will by itself.
To prove ``Lambda <= t0``, it suffices to prove that ``H_t0`` has only real
zeros globally, at the single fixed parameter ``t = t0``. The proposed
finite-reduction route additionally needs a **far-field** theorem excluding
non-real zeros beyond the bounded region at that same parameter. This module
does not discharge that theorem; it is recorded as an external premise.
See ``theory/07-frontier/01-sub-obligation-ledger.md``'s RH row and
``docs/frontier-ledger.md`` (planned Lambda program; no implementation).
Never read a result from this module as evidence for or against the
Riemann Hypothesis.

Mathematics
-----------
The de Bruijn-Newman function (de Bruijn 1950; C. M. Newman, *Fourier
transforms with only real zeros*, Proc. AMS 61 (1976) 245-251; Csordas,
Norfolk, Varga 1987 onward; D.H.J. Polymath, *Effective approximation of heat
flow evolution of the Riemann xi function...*, Res. Math. Sci. 6 (2019),
arXiv:1904.12438) is

.. math::

    H_t(z) = \int_0^\infty \Phi(u)\, e^{t u^2}\, \cos(zu)\, du,
    \qquad
    \Phi(u) = \sum_{n=1}^\infty
        \bigl(2\pi^2 n^4 e^{9u} - 3\pi n^2 e^{5u}\bigr)
        \exp\!\bigl(-\pi n^2 e^{4u}\bigr).

``Phi`` is a real, super-exponentially-decaying series derived from the
Jacobi theta functional equation -- **not** the Riemann zeta function
evaluated anywhere. This module does not claim continuation of zeta into
the critical strip, and none is used or needed.
:mod:`omnibias.core.verified.dirichlet` (``Re(s) > 1`` only) is never
imported, called, or otherwise touched here; the two modules are
mathematically disjoint constructions of a shared classical object (the
Riemann xi function), not a dependency of one on the other.

**Normalization actually used, verified against known numbers, not merely
assumed.** With the exact ``Phi``/``H_t`` above (the one tabulated on
Wikipedia's "De Bruijn-Newman constant" page and matching de Bruijn 1950 /
Newman 1976's original construction), the classical identity is

.. math::

    H_0(z) = \tfrac18\, \Xi(z/2), \qquad \Xi(w) := \xi\!\left(\tfrac12 + iw\right),
    \qquad \xi(s) := \tfrac12\, s(s-1)\, \pi^{-s/2}\, \Gamma(s/2)\, \zeta(s),

the Riemann ``xi`` function in the classical (Titchmarsh) normalization; this
is also Dobner's explicit relation ``H_t(z) = (1/8) xi_t((1+iz)/2)`` at
``t=0``. This was independently re-derived and confirmed numerically here
(``mpmath``, 50 decimal digits) against ``mpmath.zetazero`` before writing any
certified code: away from a common zero, ``H_0(z) / (Xi(z/2)/8) = 1`` to
better than ``1e-49`` relative at ``z in {0, 10, 40}``, and *both* sides
vanish (to quadrature noise, ``~1e-55``) at ``z = 2*gamma_1``,
``z = 2*gamma_2`` for the first two published nontrivial zeta zeros
``gamma_1 = 14.134725...``, ``gamma_2 = 21.022040...``. **Consequently the
zero of H_0 associated with the first published Riemann zero sits at**
``z = 2 * 14.134725141734693790... = 28.269450283469387580...``, **not** at
``14.1347...`` itself -- the factor of two is the price of the ``z/2``
argument in ``Xi(z/2)``, is a fact about this classical identity, and is not
a free choice made in this module.

Why the finite pieces are tractable without a far-field theorem
-----------------------------------------------------------------
Two genuinely different rigorous ingredients are combined, each finite:

1. A **series tail bound** for ``Phi(u)`` at fixed ``u`` (this module,
   :func:`phi_series_tail_bound`): the ratio of consecutive terms
   ``n -> n+1`` in the majorant ``n**4 * q**(n**2)`` (``q = exp(-pi e^{4u})``)
   is *strictly decreasing* in ``n`` for every fixed ``u >= 0``, so its value
   at the first omitted index bounds every later ratio, giving a genuine
   geometric tail (the same ratio-test pattern already used by
   :func:`omnibias.core.verified.transcend.besseli_iv`'s mpmath-free fallback
   and by :func:`omnibias.core.verified.dirichlet.theta_enclosure`).
2. An **outer quadrature tail bound** for the omitted ``u > U`` part of the
   defining integral (:func:`debruijn_newman_outer_tail_bound`): the
   elementary inequality ``e^x >= 1 + x + x^2/2`` (``x >= 0``) applied to
   ``x = 4(u-U)`` gives a *quadratic-in-(u-U)* lower bound on the decay
   exponent ``pi e^{4u}``, which dominates both the linear growth from
   ``cos``/``Phi``'s polynomial prefactor and the ``t u^2`` heat-flow factor
   for any one fixed, finite ``(t, U)`` satisfying two explicit inequalities
   checked in :meth:`DeBruijnNewmanContract.__post_init__`. The result is an
   ordinary convergent exponential integral bound, in closed form.

Neither ingredient is a far-field theorem: both apply only inside one
declared, finite rectangle and one declared, finite truncation ``U``. The
**far-field premise** for the proposed ``Lambda <= t0`` reduction is control
of ``H_t0(x+iy)`` outside the bounded region, sufficient to exclude every
non-real zero there. Published Riemann-Siegel-type asymptotics (Polymath15,
arXiv:1904.12438, section on "zero-free regions") provide relevant estimates,
but this module does not discharge their hypotheses for the named attempt.
Together with a complete finite-region real-zero proof, such control at
``t = t0`` would suffice. Requiring only real zeros for every ``t in [0, t0]``
would be stronger: the assertion at ``t = 0`` already implies RH. The local
evaluators and counts here remain **finite** statements at their declared
fixed parameter.

Quadrature choice
------------------
The inner integral over ``u in [0, U]`` is a **Taylor product rule**. On
each panel the amplitude ``f(u) = Phi(u) exp(t u^2)`` is expanded about the
midpoint (jet of the partial series, plus a Cauchy tail ball). Powers of the
offset times ``cos(zu)`` and ``sin(zu)`` are integrated in closed form, so
the oscillation cancels inside the main term instead of inside an interval
sum. The Lagrange remainder uses a panel enclosure of ``f^{(P)}``. A plain
box rule cannot resolve a zero of ``H_0``: panel widths add, and the
absolute enclosure stays near ``∫ Phi`` while ``H_0`` near ``2 gamma_1`` is
many orders smaller.

When the argument ``z`` is a positive-width rectangle, ``H`` is expanded
about the midpoint through order 2. The holomorphic integral remainder is a
disk of radius ``|Δz|^3 / 6`` times a moment bound on ``|H'''|``. A
degenerate argument returns the order-0 integral. The outer ``u > U`` tail
is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import (
    PI_IV,
    cos_iv,
    exp_iv,
    require_rigorous_backend,
    sin_iv,
)

#: Default number of retained terms in the ``Phi`` series. ``Phi``'s terms
#: decay doubly-exponentially in ``n`` even at ``u = 0`` (the slowest-decay
#: point on the whole domain), so this is already far more than needed for
#: double-precision tightness; it is cheap to keep generous.
PHI_DEFAULT_TERMS: int = 8


def _exp_complex(z: ComplexInterval) -> ComplexInterval:
    """Enclose ``exp(z)`` from real interval transcendental primitives."""
    mag = exp_iv(z.re)
    return ComplexInterval(mag * cos_iv(z.im), mag * sin_iv(z.im))


def _cosh_hi(value: float) -> float:
    """Rigorous upper bound on ``cosh(value)`` for a real ``value``."""
    if value <= 0.0:
        return 1.0
    v = Interval.point(value)
    return ((exp_iv(v) + exp_iv(-v)) * Interval.point(0.5)).hi


#: Rigorous upper bound on ``q(0) = exp(-pi)``, the largest value the decay
#: factor ``q(u) = exp(-pi e^{4u})`` can take over ``u >= 0`` (``q`` is
#: strictly decreasing in ``u``). Used to build the universal safety factor
#: :data:`_PHI_TAIL_SAFETY` below.
_Q_AT_ZERO_HI: float = exp_iv(-PI_IV).hi

#: Universal (u-independent) inflation factor bounding
#: ``1 / (1 - 16 q(u)**3)`` for every ``u >= 0``, derived in
#: :func:`phi_series_tail_bound`'s docstring. ``16 * _Q_AT_ZERO_HI**3`` is of
#: order ``1.3e-3``, so this is a tiny, cheap safety margin.
_PHI_TAIL_SAFETY: float = 1.0 / (1.0 - 16.0 * _Q_AT_ZERO_HI**3)

#: The exact rigorous constant ``pi * (2*pi + 3)`` bounding
#: ``2 pi^2 n^4 + 3 pi n^2 <= pi(2pi+3) n^4`` for every integer ``n >= 1``
#: (proved in :func:`phi_series_tail_bound`'s docstring). Kept as an
#: :class:`Interval` so downstream products stay outward-rounded.
_PHI_MAJORANT_CONST = PI_IV * (Interval.point(2.0) * PI_IV + Interval.point(3.0))


def phi_enclosure(u: Interval, n_terms: int = PHI_DEFAULT_TERMS) -> Interval:
    r"""Rigorous enclosure of ``Phi(u)`` for ``u >= 0`` (any width interval).

    Sums the first ``n_terms`` series terms in :class:`Interval` arithmetic
    and adds the rigorous tail majorant from :func:`phi_series_tail_bound`.
    Valid for a *wide* interval ``u`` (e.g. one quadrature panel), not only a
    point: every primitive composed here (``exp_iv``, interval
    multiplication/subtraction) is a sound enclosure over the whole interval,
    and the tail bound is derived to be valid uniformly across ``u.lo`` to
    ``u.hi`` (see :func:`phi_series_tail_bound`).
    """
    if u.lo < 0.0:
        raise ValueError(f"phi_enclosure requires u >= 0, got u.lo={u.lo!r}")
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")
    e9u = exp_iv(Interval.point(9.0) * u)
    e5u = exp_iv(Interval.point(5.0) * u)
    e4u = exp_iv(Interval.point(4.0) * u)
    two_pi2 = Interval.point(2.0) * PI_IV.pow_int(2)
    three_pi = Interval.point(3.0) * PI_IV
    total = Interval.point(0.0)
    for n in range(1, n_terms + 1):
        n2 = Interval.from_value(n * n)
        n4 = Interval.from_value(n * n * n * n)
        decay = exp_iv(-PI_IV * n2 * e4u)
        poly = two_pi2 * n4 * e9u - three_pi * n2 * e5u
        total = total + poly * decay
    tail = phi_series_tail_bound(u, n_terms)
    return total + Interval(-tail, tail)


def phi_series_tail_bound(u: Interval, n_terms: int) -> float:
    r"""Rigorous upper bound on ``|sum_{n > n_terms} a_n(u')|`` over ``u' in u``.

    Derivation. Write ``a_n(u) = (2 pi^2 n^4 e^{9u} - 3 pi n^2 e^{5u}) q^{n^2}``
    with ``q = q(u) = exp(-pi e^{4u})``. For every ``n >= 1`` and ``u >= 0``,
    ``n^2 e^{4u} >= 1``, so ``3 pi n^2 e^{5u} <= 3 pi n^4 e^{9u}`` and hence

    .. math::

        |a_n(u)| \le b_n(u) := \pi(2\pi+3)\, n^4\, e^{9u}\, q(u)^{n^2}.

    ``e^{9u}`` is increasing and ``q(u)`` is decreasing in ``u``, so for
    *every* ``u`` in the interval, ``b_n(u) <= pi(2pi+3) n^4 e^{9 u.hi}
    q(u.lo)^{n^2}`` -- a single bound valid across the whole interval,
    independent of where the true maximum of ``b_n`` itself sits.

    The remaining task is bounding ``sum_{n > N} n^4 q^{n^2}`` for the fixed
    ``q = q(u.lo)``. The term ratio ``t_{n+1}/t_n = ((n+1)/n)^4 q^{2n+1}`` is
    a product of two positive sequences that are each strictly decreasing in
    ``n`` (for fixed ``q < 1``), hence itself strictly decreasing; its value
    at ``n = N+1`` therefore bounds every later ratio, giving the geometric
    tail ``sum_{n>N} t_n <= t_{N+1} / (1 - rho)`` with
    ``rho = ((N+2)/(N+1))^4 q^{2N+3}`` -- exactly the ratio-test pattern
    :func:`~omnibias.core.verified.transcend.besseli_iv`'s series fallback
    and :func:`~omnibias.core.verified.dirichlet.theta_enclosure` already
    use. This routine raises if ``rho >= 1`` (never triggered for ``u >= 0``
    given ``q(u) <= q(0) = e^{-pi} ~ 0.0432``, for which even ``N=0`` already
    gives ``rho = 16 q(0)^3 ~ 1.3e-3``).
    """
    if u.lo < 0.0:
        raise ValueError(f"phi_series_tail_bound requires u >= 0, got u.lo={u.lo!r}")
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")
    n1 = n_terms + 1
    growth_hi = exp_iv(Interval.point(9.0) * Interval.point(u.hi)).hi
    q_hi = exp_iv(-PI_IV * exp_iv(Interval.point(4.0) * Interval.point(u.lo))).hi
    q_hi_iv = Interval.point(q_hi)
    term_n1 = Interval.from_value(n1**4) * q_hi_iv.pow_int(n1 * n1)
    ratio = (
        Interval.from_value((n1 + 1) ** 4)
        / Interval.from_value(n1**4)
        * q_hi_iv.pow_int(2 * n1 + 1)
    )
    if ratio.hi >= 1.0:
        raise ValueError(
            "phi_series_tail_bound: term ratio bound is not < 1 "
            f"(got {ratio.hi!r}); this should never happen for u >= 0"
        )
    bound = (
        _PHI_MAJORANT_CONST * Interval.point(growth_hi) * term_n1 / (Interval.point(1.0) - ratio)
    ).hi
    return max(bound, 0.0)


@dataclass(frozen=True)
class DeBruijnNewmanContract:
    """Immutable finite contract: one rectangle, one truncation, one panel/term budget.

    ``t`` is the heat-flow parameter of ``H_t`` itself (unrelated to any
    de Bruijn-Newman constant bound -- there is no claim here that ``t`` is
    below or above ``Lambda``). ``truncation`` is the finite upper limit
    ``U`` of the retained ``u``-integral; ``phi_terms`` is the number ``N``
    of retained ``Phi`` series terms.
    """

    t: float
    center: complex
    half_width: float
    half_height: float
    truncation: float
    phi_terms: int = PHI_DEFAULT_TERMS
    panels: int = 512
    contour_segments: int = 64

    def __post_init__(self) -> None:
        if not math.isfinite(self.t):
            raise ValueError("t must be finite")
        if self.half_width <= 0.0 or not math.isfinite(self.half_width):
            raise ValueError("half_width must be finite and > 0")
        if self.half_height <= 0.0 or not math.isfinite(self.half_height):
            raise ValueError("half_height must be finite and > 0")
        if self.truncation <= 0.0 or not math.isfinite(self.truncation):
            raise ValueError("truncation must be finite and > 0")
        if self.phi_terms < 1:
            raise ValueError("phi_terms must be >= 1")
        if self.panels < 1:
            raise ValueError("panels must be >= 1")
        if self.contour_segments < 4 or self.contour_segments % 4:
            raise ValueError("contour_segments must be divisible by 4 and >= 4")
        b = self.max_imaginary_part
        kappa = self._kappa()
        if kappa.lo <= 0.0:
            raise ValueError(
                "truncation does not satisfy 4*pi*e^(4U) > (9+b) + 2*t*U "
                "(outer tail bound would not converge); increase truncation"
            )
        quad_margin = Interval.point(8.0) * PI_IV * self._e4U() - Interval.point(self.t)
        if quad_margin.lo <= 0.0:
            raise ValueError(
                "truncation does not satisfy 8*pi*e^(4U) > t "
                "(quadratic tangent-domination step is invalid); increase truncation"
            )
        del b  # validated via _kappa(); kept for readability of the check above

    @property
    def max_imaginary_part(self) -> float:
        """Maximum ``|Im z|`` across the declared rectangle."""
        return abs(self.center.imag) + self.half_height

    def _e4U(self) -> Interval:
        return exp_iv(Interval.point(4.0) * Interval.point(self.truncation))

    def _kappa(self) -> Interval:
        """The outer-tail exponential decay rate; must be strictly positive."""
        b = self.max_imaginary_part
        return (
            Interval.point(4.0) * PI_IV * self._e4U()
            - Interval.point(9.0 + b)
            - Interval.point(2.0) * Interval.point(self.t) * Interval.point(self.truncation)
        )


@dataclass(frozen=True)
class DeBruijnNewmanZeroCount:
    """Result of one fixed-box argument-principle computation for ``H_t``."""

    count: int | None
    winding: Interval | None
    certified: bool
    detail: str
    contract: DeBruijnNewmanContract


def debruijn_newman_outer_tail_bound(
    z: ComplexInterval, contract: DeBruijnNewmanContract
) -> Interval:
    r"""Rigorous upper bound on the omitted ``u > U`` integral's modulus.

    Derivation. Write ``b = max|Im z|`` over the declared rectangle, so
    ``|cos(zu)| <= e^{bu}`` for ``u >= 0``. From
    :func:`phi_series_tail_bound`'s ``N=0`` case, ``|Phi(u)| <= pi(2pi+3)
    e^{9u} q(u) / (1 - 16 q(u)^3) <= K pi(2pi+3) e^{9u} q(u)`` for every
    ``u >= 0``, where ``K`` (:data:`_PHI_TAIL_SAFETY`) is the ``u``-uniform
    safety factor bounding ``1/(1-16 q(u)^3) <= 1/(1-16 q(0)^3)``. Using
    ``e^x >= 1 + x + x^2/2`` (``x = 4(u-U) >= 0``) gives the quadratic-in-
    ``(u-U)`` lower bound ``pi e^{4u} >= pi e^{4U}(1 + 4w + 8w^2)``,
    ``w = u - U``. Substituting and collecting the exponent in ``w``:

    .. math::

        (9+b)u + t u^2 - \pi e^{4u}
        \le \bigl[(9+b)U + tU^2 - \pi e^{4U}\bigr]
            - \kappa w + (t - 8\pi e^{4U}) w^2,
        \qquad \kappa = 4\pi e^{4U} - (9+b) - 2tU,

    and the contract's ``__post_init__`` checks both ``kappa > 0`` and
    ``8 pi e^{4U} > t``, so the ``w^2`` coefficient is non-positive and the
    right side is bounded by its value at ``w^2 = 0``, giving a genuine
    exponential-in-``w`` majorant whose integral over ``w >= 0`` is
    ``1/kappa`` in closed form. The final bound is

    .. math::

        \Bigl|\int_U^\infty \Phi(u) e^{tu^2}\cos(zu)\,du\Bigr|
        \le \pi(2\pi+3)\,K\,
            \frac{\exp\bigl((9+b)U + tU^2 - \pi e^{4U}\bigr)}{\kappa}.
    """
    b = z.im.mag
    U_iv = Interval.point(contract.truncation)
    t_iv = Interval.point(contract.t)
    e4U = contract._e4U()
    kappa = (
        Interval.point(4.0) * PI_IV * e4U
        - Interval.point(9.0 + b)
        - Interval.point(2.0) * t_iv * U_iv
    )
    if kappa.lo <= 0.0:
        raise ValueError(
            "outer tail bound requires kappa > 0 on this image box; increase truncation U"
        )
    exponent = Interval.point(9.0 + b) * U_iv + t_iv * U_iv.pow_int(2) - PI_IV * e4U
    growth = exp_iv(exponent)
    return _PHI_MAJORANT_CONST * Interval.point(_PHI_TAIL_SAFETY) * growth / kappa


#: Amplitude Taylor order. Main terms use derivatives ``0 .. P-1``; the
#: panel remainder is controlled by ``f^{(P)}``.
_U_TAYLOR_ORDER = 4
#: Disk radius for the Cauchy tail of the omitted ``Phi`` terms.
_TAIL_CAUCHY_R = 0.1
#: Argument boxes thinner than this use a Lipschitz ball about the midpoint.
_TINY_ARGUMENT = 1e-8


def _panel_nodes(truncation: float, panels: int) -> tuple[float, ...]:
    """Monotone float nodes covering ``[0, truncation]`` exactly."""
    nodes = [0.0]
    span = Interval.point(truncation)
    for index in range(1, panels):
        bound = (span * Interval.from_rational(Fraction(index, panels))).lo
        if bound < nodes[-1]:
            bound = nodes[-1]
        if bound > truncation:
            bound = truncation
        nodes.append(bound)
    nodes.append(float(truncation))
    return tuple(nodes)


def _exp_jet(coeffs: Sequence[Interval]) -> list[Interval]:
    """Taylor coefficients of ``exp(F)`` from those of ``F`` (``c_k = F^{(k)}/k!``)."""
    out: list[Interval] = [exp_iv(coeffs[0])]
    for k in range(1, len(coeffs)):
        acc = Interval.point(0.0)
        for j in range(k):
            acc = acc + Interval.from_value(j + 1) * coeffs[j + 1] * out[k - 1 - j]
        out.append(acc / Interval.from_value(k))
    return out


def _scale_jet(factor: Interval, coeffs: Sequence[Interval]) -> list[Interval]:
    return [factor * coeff for coeff in coeffs]


def _add_jet(left: Sequence[Interval], right: Sequence[Interval]) -> list[Interval]:
    return [a + b for a, b in zip(left, right, strict=True)]


def _mul_jet(left: Sequence[Interval], right: Sequence[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for k in range(len(left)):
        acc = Interval.point(0.0)
        for i in range(k + 1):
            acc = acc + left[i] * right[k - i]
        out.append(acc)
    return out


def _identity_jet(u: Interval, order: int) -> list[Interval]:
    coeffs = [u]
    if order >= 1:
        coeffs.append(Interval.point(1.0))
    coeffs.extend(Interval.point(0.0) for _ in range(order - 1))
    return coeffs


def _phi_jet(u: Interval, n_terms: int, order: int) -> list[Interval]:
    """Taylor coefficients of the partial ``Phi`` sum, highest index ``order``."""
    total = [Interval.point(0.0) for _ in range(order + 1)]
    two_pi2 = Interval.point(2.0) * PI_IV.pow_int(2)
    three_pi = Interval.point(3.0) * PI_IV
    four = Interval.point(4.0)
    for n in range(1, n_terms + 1):
        n2 = float(n * n)
        n4 = n2 * n2
        amplitude = _add_jet(
            _scale_jet(two_pi2 * Interval.point(n4), _exp_jet(_scale_jet(Interval.point(9.0), _identity_jet(u, order)))),
            _scale_jet(-(three_pi * Interval.point(n2)), _exp_jet(_scale_jet(Interval.point(5.0), _identity_jet(u, order)))),
        )
        decay = _exp_jet(_scale_jet(-(PI_IV * Interval.point(n2)), _exp_jet(_scale_jet(four, _identity_jet(u, order)))))
        total = _add_jet(total, _mul_jet(amplitude, decay))
    return total


def _heat_jet(t: float, u: Interval, order: int) -> list[Interval]:
    if t == 0.0:
        coeffs = [Interval.point(1.0)]
        coeffs.extend(Interval.point(0.0) for _ in range(order))
        return coeffs
    u_jet = _identity_jet(u, order)
    return _exp_jet(_scale_jet(Interval.point(t), _mul_jet(u_jet, u_jet)))


def _decay_base_hi(x_lo: float, y_abs: float) -> float:
    """Upper bound on ``exp(-pi exp(4 x_lo) cos(4 y_abs))``."""
    rate = PI_IV * exp_iv(Interval.point(4.0) * Interval.point(x_lo)) * cos_iv(
        Interval.point(4.0) * Interval.point(y_abs)
    )
    if rate.lo <= 0.0:
        raise ValueError("Phi decay bound is not positive on this Cauchy rectangle")
    return exp_iv(-Interval.point(rate.lo)).hi


def _tail_modulus_hi(x_lo: float, x_hi: float, y_abs: float, n_terms: int) -> float:
    """Upper bound on the omitted ``|Phi|`` tail over a complex rectangle."""
    if x_lo > x_hi:
        x_lo, x_hi = x_hi, x_lo
    q_hi = _decay_base_hi(x_lo, y_abs)
    g9 = exp_iv(Interval.point(9.0) * Interval.point(x_hi)).hi
    g5 = exp_iv(Interval.point(5.0) * Interval.point(x_hi)).hi
    two_pi2 = (Interval.point(2.0) * PI_IV.pow_int(2)).hi
    three_pi = (Interval.point(3.0) * PI_IV).hi
    q_iv = Interval.point(q_hi)
    explicit = Interval.point(0.0)
    last = n_terms + 8
    for n in range(n_terms + 1, last + 1):
        n2 = n * n
        term = (
            Interval.point(two_pi2) * Interval.from_value(n2 * n2) * Interval.point(g9)
            + Interval.point(three_pi) * Interval.from_value(n2) * Interval.point(g5)
        ) * q_iv.pow_int(n2)
        explicit = explicit + term
    n1 = last + 1
    const = (
        Interval.point(two_pi2) * Interval.point(g9) + Interval.point(three_pi) * Interval.point(g5)
    )
    term_n1 = Interval.from_value(n1**4) * q_iv.pow_int(n1 * n1)
    ratio = (
        Interval.from_value((n1 + 1) ** 4)
        / Interval.from_value(n1**4)
        * q_iv.pow_int(2 * n1 + 1)
    )
    if ratio.hi >= 1.0:
        raise ValueError("Phi tail ratio is not < 1 on this Cauchy rectangle")
    geometric = const * term_n1 / (Interval.point(1.0) - ratio)
    return (explicit + geometric).hi


def _tail_coeff_radii(
    x_lo: float, x_hi: float, highest: int, n_terms: int
) -> tuple[float, ...]:
    """Cauchy bounds on ``|tail^{(k)}/k!|`` for ``k = 0 .. highest``."""
    modulus = _tail_modulus_hi(x_lo - _TAIL_CAUCHY_R, x_hi + _TAIL_CAUCHY_R, _TAIL_CAUCHY_R, n_terms)
    mod_iv = Interval.point(modulus)
    radius = Interval.point(_TAIL_CAUCHY_R)
    power = Interval.point(1.0)
    out: list[float] = []
    for _k in range(highest + 1):
        out.append((mod_iv / power).hi)
        power = power * radius
    return tuple(out)


def _widen_jet(coeffs: Sequence[Interval], radii: Sequence[float]) -> list[Interval]:
    return [coeff + Interval(-rad, rad) for coeff, rad in zip(coeffs, radii, strict=True)]


def _amplitude_jet(t: float, u: Interval, n_terms: int, order: int, radii: Sequence[float]) -> list[Interval]:
    phi = _widen_jet(_phi_jet(u, n_terms, order), radii)
    if t == 0.0:
        return phi
    return _mul_jet(phi, _heat_jet(t, u, order))


def _finite_jet(coeffs: Sequence[Interval]) -> bool:
    return all(math.isfinite(coeff.lo) and math.isfinite(coeff.hi) for coeff in coeffs)


@dataclass(frozen=True)
class _PanelQuad:
    a: float
    b: float
    m: float
    alpha: float
    beta: float
    jet: tuple[Interval, ...]
    deriv_mag: float
    box_only: bool
    amp_hi: float


@lru_cache(maxsize=16)
def _prepared_panels(
    t: float, truncation: float, phi_terms: int, panels: int
) -> tuple[_PanelQuad, ...]:
    """Panel jets for one ``(t, U, N, panels)`` contract. Independent of ``z``."""
    order = _U_TAYLOR_ORDER
    built: list[_PanelQuad] = []
    nodes = _panel_nodes(truncation, panels)
    for index in range(len(nodes) - 1):
        a = nodes[index]
        b = nodes[index + 1]
        if not b > a:
            continue
        u_panel = Interval(a, b)
        amp = phi_enclosure(u_panel, phi_terms).hi
        heat = exp_iv(Interval.point(t) * u_panel.pow_int(2)).hi
        amp_hi = (Interval.point(amp) * Interval.point(heat)).hi
        m = (a + b) * 0.5
        if m < a:
            m = a
        if m > b:
            m = b
        alpha = a - m
        beta = b - m
        point_radii = _tail_coeff_radii(m, m, order - 1, phi_terms)
        point_jet = _amplitude_jet(t, Interval.point(m), phi_terms, order - 1, point_radii)
        panel_radii = _tail_coeff_radii(a, b, order, phi_terms)
        panel_jet = _amplitude_jet(t, u_panel, phi_terms, order, panel_radii)
        box_only = amp_hi < 1e-14 or not _finite_jet(point_jet) or not _finite_jet(panel_jet)
        deriv_mag = 0.0 if box_only else math.factorial(order) * panel_jet[order].mag
        if not math.isfinite(deriv_mag):
            box_only = True
            deriv_mag = 0.0
        built.append(
            _PanelQuad(
                a=a,
                b=b,
                m=m,
                alpha=alpha,
                beta=beta,
                jet=tuple(point_jet),
                deriv_mag=deriv_mag,
                box_only=box_only,
                amp_hi=amp_hi,
            )
        )
    return tuple(built)


def _g_powers_series(
    s: ComplexInterval, alpha: float, beta: float, count: int
) -> list[ComplexInterval]:
    """``∫_alpha^beta w^k exp(s w) dw`` by power series, for ``s`` near ``0``."""
    n_terms = 10
    width = Interval.point(beta) - Interval.point(alpha)
    w_max = max(abs(alpha), abs(beta))
    s_mag = s.mag
    out: list[ComplexInterval] = []
    for k in range(count):
        acc = ComplexInterval.zero()
        s_power = ComplexInterval.one()
        fact = 1
        for n in range(n_terms):
            q = k + n
            integ = (
                Interval.point(beta).pow_int(q + 1) - Interval.point(alpha).pow_int(q + 1)
            ) / Interval.from_value(q + 1)
            acc = acc + s_power * (ComplexInterval.from_value(integ) / float(fact))
            s_power = s_power * s
            fact *= n + 1
        rem_iv = (
            width
            * Interval.point(s_mag).pow_int(n_terms)
            * Interval.point(w_max).pow_int(k + n_terms)
            / Interval.from_value(math.factorial(n_terms))
            * exp_iv(Interval.point(s_mag * w_max))
        )
        rad = max(rem_iv.hi, 0.0)
        out.append(ComplexInterval(acc.re + Interval(-rad, rad), acc.im + Interval(-rad, rad)))
    return out


def _g_powers(s: ComplexInterval, alpha: float, beta: float, count: int) -> list[ComplexInterval]:
    """``G_k = ∫_alpha^beta w^k exp(s w) dw`` for ``k = 0 .. count-1``."""
    if count < 1:
        return []
    modulus2 = s.re.pow_int(2) + s.im.pow_int(2)
    if modulus2.lo <= 1e-12:
        return _g_powers_series(s, alpha, beta, count)
    exp_b = _exp_complex(s * beta)
    exp_a = _exp_complex(s * alpha)
    inv_s = ComplexInterval.one() / s
    out = [(exp_b - exp_a) * inv_s]
    a_pow = Interval.point(1.0)
    b_pow = Interval.point(1.0)
    a_iv = Interval.point(alpha)
    b_iv = Interval.point(beta)
    for k in range(1, count):
        a_pow = a_pow * a_iv
        b_pow = b_pow * b_iv
        boundary = ComplexInterval.from_value(b_pow) * exp_b - ComplexInterval.from_value(a_pow) * exp_a
        out.append(boundary * inv_s - out[k - 1] * (float(k) * inv_s))
    return out


def _monomial_trig(
    z: ComplexInterval, alpha: float, beta: float, m: float, highest: int
) -> tuple[list[ComplexInterval], list[ComplexInterval]]:
    """``∫ w^k cos(z(w+m)) dw`` and the sine twin, ``k = 0 .. highest``."""
    iz = ComplexInterval.imag_unit() * z
    gp = _g_powers(iz, alpha, beta, highest + 1)
    gm = _g_powers(-iz, alpha, beta, highest + 1)
    phase_p = _exp_complex(iz * m)
    phase_m = _exp_complex((-iz) * m)
    half = Interval.point(0.5)
    inv_2i = ComplexInterval.one() / (ComplexInterval.imag_unit() * 2.0)
    cos_terms: list[ComplexInterval] = []
    sin_terms: list[ComplexInterval] = []
    for k in range(highest + 1):
        ep = phase_p * gp[k]
        em = phase_m * gm[k]
        cos_terms.append((ep + em) * half)
        sin_terms.append((ep - em) * inv_2i)
    return cos_terms, sin_terms


def _u_remainder(panel: _PanelQuad, q: int, y_abs: float) -> float:
    """Lagrange ball for ``∫ (f - T) u^q trig`` on one panel."""
    if panel.box_only:
        cosh = _cosh_hi(y_abs * panel.b)
        width = Interval.point(panel.b) - Interval.point(panel.a)
        bound = (
            Interval.point(panel.amp_hi)
            * Interval.point(panel.b).pow_int(q)
            * Interval.point(cosh)
            * width
        )
        return max(bound.hi, 0.0)
    half_span = max(abs(panel.alpha), abs(panel.beta))
    cosh = _cosh_hi(y_abs * panel.b)
    bound = (
        Interval.point(2.0)
        * Interval.point(half_span).pow_int(_U_TAYLOR_ORDER + 1)
        / Interval.from_value(_U_TAYLOR_ORDER + 1)
        * Interval.point(panel.deriv_mag)
        / Interval.from_value(math.factorial(_U_TAYLOR_ORDER))
        * Interval.point(panel.b).pow_int(q)
        * Interval.point(cosh)
    )
    return max(bound.hi, 0.0)


def _eval_moments(
    z0: complex, prepared: tuple[_PanelQuad, ...], max_q: int
) -> tuple[list[ComplexInterval], list[ComplexInterval]]:
    """Inner integrals ``∫_0^U f(u) u^q cos/sin(z0 u) du`` for ``q <= max_q``."""
    z = ComplexInterval.point(z0)
    y_abs = abs(z0.imag)
    icos = [ComplexInterval.zero() for _ in range(max_q + 1)]
    isin = [ComplexInterval.zero() for _ in range(max_q + 1)]
    radii = [0.0 for _ in range(max_q + 1)]
    highest = (_U_TAYLOR_ORDER - 1) + max_q
    for panel in prepared:
        if panel.box_only:
            for q in range(max_q + 1):
                radii[q] += _u_remainder(panel, q, y_abs)
            continue
        jcos, jsin = _monomial_trig(z, panel.alpha, panel.beta, panel.m, highest)
        for q in range(max_q + 1):
            acc_c = ComplexInterval.zero()
            acc_s = ComplexInterval.zero()
            for i in range(q + 1):
                scale = Interval.from_value(math.comb(q, i)) * Interval.point(panel.m).pow_int(q - i)
                for j in range(_U_TAYLOR_ORDER):
                    coeff = panel.jet[j] * scale
                    acc_c = acc_c + jcos[j + i] * coeff
                    acc_s = acc_s + jsin[j + i] * coeff
            icos[q] = icos[q] + acc_c
            isin[q] = isin[q] + acc_s
            radii[q] += _u_remainder(panel, q, y_abs)
    out_c: list[ComplexInterval] = []
    out_s: list[ComplexInterval] = []
    for q in range(max_q + 1):
        rad = radii[q]
        out_c.append(ComplexInterval(icos[q].re + Interval(-rad, rad), icos[q].im + Interval(-rad, rad)))
        out_s.append(ComplexInterval(isin[q].re + Interval(-rad, rad), isin[q].im + Interval(-rad, rad)))
    return out_c, out_s


def _outer_moment_tail(t: float, truncation: float, y_abs: float, n: int) -> float:
    """Upper bound on ``∫_U^∞ u^n |Phi| exp(t u^2) exp(y u) du``."""
    u_iv = Interval.point(truncation)
    e4u = exp_iv(Interval.point(4.0) * u_iv)
    kappa = (
        Interval.point(4.0) * PI_IV * e4u
        - Interval.point(9.0 + y_abs)
        - Interval.point(2.0) * Interval.point(t) * u_iv
    )
    if kappa.lo <= 0.0:
        raise ValueError("moment tail requires kappa > 0; increase truncation")
    exponent = (
        Interval.point(9.0 + y_abs) * u_iv
        + Interval.point(t) * u_iv.pow_int(2)
        - PI_IV * e4u
    )
    prefactor = _PHI_MAJORANT_CONST * Interval.point(_PHI_TAIL_SAFETY) * exp_iv(exponent)
    if n == 0:
        return (prefactor / kappa).hi
    rate = kappa - Interval.from_value(n) / u_iv
    if rate.lo <= 0.0:
        raise ValueError("moment tail needs kappa > n/U; increase truncation")
    # (U+w)^n <= U^n exp(n w / U), so the w-integral is at most U^n / (kappa - n/U).
    return (prefactor * u_iv.pow_int(n) / rate).hi


@lru_cache(maxsize=64)
def _moment_majorant(
    t: float, truncation: float, phi_terms: int, panels: int, n: int, y_abs: float
) -> float:
    """Upper bound on ``∫_0^∞ u^n |f(u)| exp(y |u|) du``."""
    total = Interval.point(0.0)
    nodes = _panel_nodes(truncation, panels)
    for index in range(len(nodes) - 1):
        a = nodes[index]
        b = nodes[index + 1]
        if not b > a:
            continue
        u_panel = Interval(a, b)
        phi_hi = phi_enclosure(u_panel, phi_terms).hi
        heat = exp_iv(Interval.point(t) * u_panel.pow_int(2))
        exp_y = exp_iv(Interval.point(y_abs) * u_panel)
        width = Interval.point(b) - Interval.point(a)
        total = total + Interval.point(phi_hi) * heat * exp_y * Interval.point(b).pow_int(n) * width
    outer = _outer_moment_tail(t, truncation, y_abs, n)
    return (total + Interval.point(outer)).hi


def _widen_complex(value: ComplexInterval, radius: float) -> ComplexInterval:
    ball = Interval(-radius, radius)
    return ComplexInterval(value.re + ball, value.im + ball)


def _derivatives_at(
    z0: complex, contract: DeBruijnNewmanContract, prepared: tuple[_PanelQuad, ...], max_order: int
) -> tuple[ComplexInterval, ...]:
    """``H, H', ...`` through ``max_order`` at the point ``z0``, tails included."""
    icos, isin = _eval_moments(z0, prepared, max_order)
    y_abs = abs(z0.imag)
    point = ComplexInterval.point(z0)
    out: list[ComplexInterval] = []
    for order in range(max_order + 1):
        if order % 4 == 0:
            main = icos[order]
        elif order % 4 == 1:
            main = -isin[order]
        elif order % 4 == 2:
            main = -icos[order]
        else:
            main = isin[order]
        if order == 0:
            tail = debruijn_newman_outer_tail_bound(point, contract).hi
        else:
            tail = _outer_moment_tail(contract.t, contract.truncation, y_abs, order)
        out.append(_widen_complex(main, tail))
    return tuple(out)


def debruijn_newman_enclosure(
    z: ComplexInterval, contract: DeBruijnNewmanContract
) -> ComplexInterval:
    """Enclose the genuine de Bruijn-Newman ``H_t(z)`` over a complex rectangle.

    Inner integral by the Taylor product rule in the module docstring; outer
    ``u > U`` tail via :func:`debruijn_newman_outer_tail_bound`. A wide ``z``
    is reduced to a point expansion plus a holomorphic third-derivative ball.
    """
    prepared = _prepared_panels(contract.t, contract.truncation, contract.phi_terms, contract.panels)
    z0 = complex(0.5 * (z.re.lo + z.re.hi), 0.5 * (z.im.lo + z.im.hi))
    delta = z - ComplexInterval.point(z0)
    y_box = z.im.mag
    if delta.mag <= _TINY_ARGUMENT:
        value = _derivatives_at(z0, contract, prepared, 0)[0]
        lip = _moment_majorant(
            contract.t, contract.truncation, contract.phi_terms, contract.panels, 1, y_box
        )
        radius = (Interval.point(delta.mag) * Interval.point(lip)).hi
        return _widen_complex(value, radius)
    value, first, second = _derivatives_at(z0, contract, prepared, 2)
    main = value + first * delta + second * delta * delta * Interval.point(0.5)
    third = _moment_majorant(
        contract.t, contract.truncation, contract.phi_terms, contract.panels, 3, y_box
    )
    radius = (
        Interval.point(delta.mag).pow_int(3) / Interval.from_value(6) * Interval.point(third)
    ).hi
    return _widen_complex(main, radius)


def count_debruijn_newman_zeros(
    contract: DeBruijnNewmanContract,
) -> DeBruijnNewmanZeroCount:
    """Certify the zeros of the genuine ``H_t`` inside one fixed rectangle.

    Returns a **local, finite** zero count via the same
    :mod:`omnibias.core.collapse.winding` argument-principle machinery
    :mod:`omnibias.core.verified.heat_kernel` already uses (no separate
    winding implementation). This is never a de Bruijn-Newman ``Lambda``
    bound and never a statement about the Riemann Hypothesis: see the module
    docstring's "Scope, read first" paragraph.
    """
    # Lazy: a top-level import of collapse.winding loads collapse/__init__.py
    # while omnibias.core.verified is still importing, which circularly
    # re-enters omnibias.core.proof.engine -> collapse.identity.
    from omnibias.core.collapse.winding import (
        contour_parameter_segments,
        integers_in,
        winding_enclosure_function,
    )

    require_rigorous_backend()
    domains = contour_parameter_segments(
        contract.center,
        contract.half_width,
        segments=contract.contour_segments,
        contour="rectangle",
        half_width=contract.half_width,
        half_height=contract.half_height,
    )
    winding = winding_enclosure_function(
        lambda z: debruijn_newman_enclosure(z, contract),
        contract.center,
        contract.half_width,
        segments=contract.contour_segments,
        contour="rectangle",
        half_width=contract.half_width,
        half_height=contract.half_height,
    )
    if winding is None:
        return DeBruijnNewmanZeroCount(
            None,
            None,
            False,
            "contour image may contain zero; fixed contract is inconclusive",
            contract,
        )
    hits = integers_in(winding)
    if len(hits) != 1:
        return DeBruijnNewmanZeroCount(
            None,
            winding,
            False,
            "winding enclosure did not isolate a unique integer",
            contract,
        )
    # Retaining this explicit cover guards future refactors from treating a
    # pointwise evaluator as a contour certificate.
    assert len(domains) == contract.contour_segments
    return DeBruijnNewmanZeroCount(
        hits[0],
        winding,
        True,
        "fixed-rectangle de Bruijn-Newman H_t zero count certified "
        "(local argument-principle fact; not a Lambda bound, not an RH claim)",
        contract,
    )


#: Imaginary part of the first published nontrivial zeta zero (Odlyzko).
FIRST_RIEMANN_ZERO_IMAG = 14.134725141734693790
#: Location of the matching ``H_0`` zero: ``z = 2 * gamma_1`` (see module docstring).
H0_FIRST_ZERO = 2.0 * FIRST_RIEMANN_ZERO_IMAG

#: Pre-registered target for a ``Lambda <= t0`` attempt.  Must stay ``< 0.22``.
PRE_REGISTERED_T0 = 0.2

FAR_FIELD_CITATION = (
    "D.H.J. Polymath, Effective approximation of heat flow evolution of the "
    "Riemann xi function and of the constant of de Bruijn--Newman, "
    "Res. Math. Sci. 6 (2019); T. Rodgers and T. Tao, The de Bruijn--Newman "
    "constant is non-negative, Forum of Mathematics, Pi 8 (2020)."
)


@dataclass(frozen=True)
class LambdaBoundAttempt:
    """An attempted ``Lambda <= t0`` certificate.  Unearned unless every piece closes.

    The far-field premise is recorded as an **external obligation** (cited,
    independently published).  This module never discharges it in Lean, never
    sets ``theorem_prover_verified``, and never infers the Riemann Hypothesis.
    ``rh_claim`` is frozen ``False``.
    """

    t0: float
    certified: bool
    finite_cover_certified: bool
    far_field_premise_discharged: bool
    first_zero_crosscheck: bool
    missing_piece: str
    far_field_citation: str
    rh_claim: bool = False

    def __post_init__(self) -> None:
        if self.rh_claim:
            raise ValueError("rh_claim must stay False (never infer RH from Lambda)")
        if self.t0 >= 0.22:
            raise ValueError(f"pre-registered t0 must be < 0.22, got {self.t0!r}")


def crosscheck_h0_at_first_zero(
    *,
    truncation: float = 2.0,
    phi_terms: int = 6,
    panels: int = 24,
) -> bool:
    """``True`` iff the ``H_0`` enclosure at ``z = 2 gamma_1`` contains 0.

    A local consistency check against the first published Riemann zero, not a
    winding count and not a ``Lambda`` bound.  Fat enclosures that contain 0
    vacuously still return ``True``; pair with :func:`phi_enclosure` tests.
    """
    contract = DeBruijnNewmanContract(
        t=0.0,
        center=complex(H0_FIRST_ZERO, 0.0),
        half_width=0.25,
        half_height=0.25,
        truncation=truncation,
        phi_terms=phi_terms,
        panels=panels,
        contour_segments=8,
    )
    enclosure = debruijn_newman_enclosure(
        ComplexInterval.point(complex(H0_FIRST_ZERO, 0.0)),
        contract,
    )
    return enclosure.contains(0.0)


def attempt_named_lambda_bound(
    *,
    t0: float = PRE_REGISTERED_T0,
    far_field_premise_discharged: bool = False,
) -> LambdaBoundAttempt:
    """Attempt ``Lambda <= t0`` with a cited far-field premise as an external obligation.

    This finite-reduction route needs (i) a complete contour cover proving
    that every relevant zero of ``H_t0`` in the bounded region is real and
    (ii) a far-field theorem excluding non-real zeros outside that region,
    both at the single parameter ``t = t0``. Global real-rootedness of
    ``H_t0`` suffices for ``Lambda <= t0``. This function **records** the two
    unresolved obligations. It
    does not run an unbounded cover, does not import
    :mod:`omnibias.core.verified.dirichlet`, and does not apply Padé / Borel
    to a Dirichlet series.

    ``certified`` is ``True`` only when the finite cover *and* the cited
    far-field premise are both discharged.  The far-field flag is a caller-
    supplied acknowledgement of an *external* theorem; this repository does
    not prove that theorem, so the default (and the only honest in-tree
    value) is ``False``.
    """
    if t0 >= 0.22 or t0 <= 0.0 or not math.isfinite(t0):
        raise ValueError(f"t0 must be finite, > 0, and < 0.22, got {t0!r}")
    first_zero_ok = False
    try:
        phi0 = phi_enclosure(Interval.point(0.0), PHI_DEFAULT_TERMS)
        first_zero_ok = phi0.lo > 0.0
    except (ValueError, ArithmeticError):
        first_zero_ok = False
    finite_cover = False
    missing: list[str] = []
    if not finite_cover:
        missing.append(
            "complete finite contour cover proving that every relevant H_t0 "
            f"zero in the bounded region is real at t0={t0} "
            "(local rectangles are not a complete cover)"
        )
    if not far_field_premise_discharged:
        missing.append(
            "cited far-field premise "
            f"(exclusion of non-real H_t0 zeros outside the bounded region at t0={t0}) "
            "is an external obligation, not discharged in this repository"
        )
    certified = finite_cover and far_field_premise_discharged
    return LambdaBoundAttempt(
        t0=float(t0),
        certified=certified,
        finite_cover_certified=finite_cover,
        far_field_premise_discharged=bool(far_field_premise_discharged),
        first_zero_crosscheck=first_zero_ok,
        missing_piece="; ".join(missing) if missing else "",
        far_field_citation=FAR_FIELD_CITATION,
        rh_claim=False,
    )


@dataclass(frozen=True)
class FiniteHtRectanglePack:
    """A declared finite list of real-axis ``H_t`` rectangles on a bounded interval.

    Local certified counts do **not** establish a whole-line real-zero
    theorem at the pack's fixed ``t``. ``finite_cover_certified`` is frozen ``False``.
    ``rh_claim`` is frozen ``False``.
    """

    t: float
    real_interval: tuple[float, float]
    counts: tuple[DeBruijnNewmanZeroCount, ...]
    n_certified: int
    n_blocked: int
    finite_cover_certified: bool = False
    rh_claim: bool = False

    def __post_init__(self) -> None:
        if self.rh_claim:
            raise ValueError("rh_claim must stay False (never infer RH from a pack)")
        if self.finite_cover_certified:
            raise ValueError(
                "finite_cover_certified must stay False "
                "(a local pack does not establish a whole-line real-zero theorem at its fixed t)"
            )


def finite_ht_rectangle_pack(
    *,
    t: float = 0.0,
    boxes: Sequence[tuple[float, float]] | None = None,
    half_height: float = 0.25,
    truncation: float = 1.0,
    phi_terms: int = 4,
    panels: int = 8,
    contour_segments: int = 8,
    real_lo: float = 0.0,
    real_hi: float = 32.0,
) -> FiniteHtRectanglePack:
    """Count ``H_t`` zeros on a declared finite pack of real-axis rectangles.

    Default boxes live in ``[0, 32]`` (the first ``H_0`` zero is at
    ``2 gamma_1 ≈ 28.27``). Each box reuses :func:`count_debruijn_newman_zeros`.
    BLOCKED boxes are first-class: a winding enclosure that does not isolate
    one integer stays ``certified=False`` with ``count is None``. The pack is
    not a whole-line cover. :func:`declared_local_ht_rectangle_pack` is a
    concrete ``t=0`` budget that isolates the first zero and one empty box.
    """
    declared = (
        tuple(boxes)
        if boxes is not None
        else (
            (4.0, 4.0),
            (16.0, 4.0),
            (H0_FIRST_ZERO, 2.0),
        )
    )
    counts: list[DeBruijnNewmanZeroCount] = []
    n_certified = 0
    n_blocked = 0
    for center_re, half_width in declared:
        contract = DeBruijnNewmanContract(
            t=t,
            center=complex(center_re, 0.0),
            half_width=half_width,
            half_height=half_height,
            truncation=truncation,
            phi_terms=phi_terms,
            panels=panels,
            contour_segments=contour_segments,
        )
        result = count_debruijn_newman_zeros(contract)
        counts.append(result)
        if result.certified:
            n_certified += 1
        else:
            n_blocked += 1
    return FiniteHtRectanglePack(
        t=float(t),
        real_interval=(float(real_lo), float(real_hi)),
        counts=tuple(counts),
        n_certified=n_certified,
        n_blocked=n_blocked,
        finite_cover_certified=False,
        rh_claim=False,
    )


def declared_local_ht_rectangle_pack() -> FiniteHtRectanglePack:
    """Local ``t = 0`` pack on ``[0, 32]`` that isolates one winding integer.

    Shared budgets: ``half_height=0.5``, ``truncation=0.55``, ``phi_terms=4``,
    ``panels=32``, ``contour_segments=8``. The rectangles are

    * center ``4``, half-width ``1`` — inside ``[0, 32]``, does not contain
      ``H0_FIRST_ZERO``;
    * center ``H0_FIRST_ZERO``, half-width ``1`` — that first ``H_0`` zero;
    * center ``H0_FIRST_ZERO - 1``, half-width ``1`` — the zero lies on the
      right edge, so the contour image contains zero and the count stays
      blocked.

    ``finite_cover_certified`` and ``rh_claim`` stay false. This is not a
    ``Lambda`` bound and not a statement about the Riemann hypothesis.
    """
    return finite_ht_rectangle_pack(
        t=0.0,
        boxes=(
            (4.0, 1.0),
            (H0_FIRST_ZERO, 1.0),
            (H0_FIRST_ZERO - 1.0, 1.0),
        ),
        half_height=0.5,
        truncation=0.55,
        phi_terms=4,
        panels=32,
        contour_segments=8,
        real_lo=0.0,
        real_hi=32.0,
    )


__all__ = [
    "DeBruijnNewmanContract",
    "DeBruijnNewmanZeroCount",
    "FAR_FIELD_CITATION",
    "FIRST_RIEMANN_ZERO_IMAG",
    "FiniteHtRectanglePack",
    "H0_FIRST_ZERO",
    "LambdaBoundAttempt",
    "PHI_DEFAULT_TERMS",
    "PRE_REGISTERED_T0",
    "attempt_named_lambda_bound",
    "count_debruijn_newman_zeros",
    "crosscheck_h0_at_first_zero",
    "declared_local_ht_rectangle_pack",
    "debruijn_newman_enclosure",
    "debruijn_newman_outer_tail_bound",
    "finite_ht_rectangle_pack",
    "phi_enclosure",
    "phi_series_tail_bound",
]
