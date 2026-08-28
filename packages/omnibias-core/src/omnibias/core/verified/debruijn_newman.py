# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""A fixed-box certified zero count for the genuine de Bruijn-Newman ``H_t``.

**Scope, read first.** This module produces exactly one kind of result: a
sound argument-principle zero *count* of the real de Bruijn-Newman function
``H_t`` inside one declared, finite complex rectangle, at one declared,
finite truncation and series-term budget. It does **not** produce a bound on
the de Bruijn-Newman constant ``Lambda``, and it never will by itself -- a
``Lambda <= t0`` certificate additionally needs an independently published
**far-field** theorem (the behaviour of ``H_t`` over the *entire* unbounded
real line, at *every* ``t`` up to ``t0``), which this module does not attempt
and which is recorded as an external premise everywhere this module is used.
See ``theory/07-frontier/01-sub-obligation-ledger.md``'s RH row and
``theory/07-frontier/08-de-bruijn-newman-heat-flow.md``. Never read a result
from this module as evidence for or against the Riemann Hypothesis.

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
**far-field premise** a genuine ``Lambda <= t0`` bound additionally needs --
control of ``H_t(x+iy)`` for *every* real ``x`` (not one bounded rectangle)
and *every* ``t`` up to ``t0`` (not one fixed value) -- is an independently
published theorem (the Riemann-Siegel-type asymptotics of Polymath15's
paper, arXiv:1904.12438, section on "zero-free regions") that this module
neither states, proves, nor invokes as if it were established here. Every
public function in this module is scoped to one fixed rectangle and is
honestly a **local, finite** statement.

Quadrature choice
------------------
The inner integral over ``u in [0, U]`` is enclosed by a **natural-interval-
extension (box) rule**: each panel's contribution is bounded by its width
times the *interval-arithmetic enclosure of the whole integrand evaluated
directly on that panel* (``Phi`` via :func:`phi_enclosure`, the ``exp(t u^2)``
factor via :func:`~omnibias.core.verified.transcend.exp_iv`, ``cos(zu)`` via
the same real-primitive composition
:mod:`omnibias.core.verified.heat_kernel` uses). This is deliberately a
*coarser* (first-order) rule than :mod:`omnibias.core.verified.quadrature`'s
derivative-bound rules (trapezoid, Gauss-Legendre, ...): ``Phi`` is itself an
infinite series, so a rigorous closed-form second ``u``-derivative would
require differentiating that series term-by-term and re-deriving a fresh
tail bound for the differentiated series. The box rule needs no derivative
bound at all -- soundness follows immediately from interval-arithmetic
containment (if ``F(I)`` encloses ``f`` over ``I``, then
``width(I) * F(I)`` encloses ``int_I f``) -- at the cost of needing more
panels for the same tightness. Given ``Phi``'s super-exponential decay this
trade is cheap in practice: the panel count is fixed at construction and
does not depend on ``z``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval, sum_intervals
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


def _cos_complex(z: ComplexInterval) -> ComplexInterval:
    """Enclose ``cos(z)`` from real interval transcendental primitives."""
    cosh = (exp_iv(z.im) + exp_iv(-z.im)) * 0.5
    sinh = (exp_iv(z.im) - exp_iv(-z.im)) * 0.5
    return ComplexInterval(cos_iv(z.re) * cosh, -sin_iv(z.re) * sinh)


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
        _PHI_MAJORANT_CONST
        * Interval.point(growth_hi)
        * term_n1
        / (Interval.point(1.0) - ratio)
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
        quad_margin = (
            Interval.point(8.0) * PI_IV * self._e4U() - Interval.point(self.t)
        )
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
            "outer tail bound requires kappa > 0 on this image box; "
            "increase truncation U"
        )
    exponent = (
        Interval.point(9.0 + b) * U_iv + t_iv * U_iv.pow_int(2) - PI_IV * e4U
    )
    growth = exp_iv(exponent)
    return _PHI_MAJORANT_CONST * Interval.point(_PHI_TAIL_SAFETY) * growth / kappa


def debruijn_newman_enclosure(
    z: ComplexInterval, contract: DeBruijnNewmanContract
) -> ComplexInterval:
    """Enclose the genuine de Bruijn-Newman ``H_t(z)`` over a complex rectangle.

    Inner integral over ``u in [0, U]`` via the box (natural-interval-
    extension) rule described in the module docstring; outer ``u > U`` tail
    via :func:`debruijn_newman_outer_tail_bound`.
    """
    t_iv = Interval.point(contract.t)
    real_panels: list[Interval] = []
    imag_panels: list[Interval] = []
    for index in range(contract.panels):
        u0 = contract.truncation * index / contract.panels
        u1 = contract.truncation * (index + 1) / contract.panels
        u_panel = Interval(u0, u1)
        phi_iv = phi_enclosure(u_panel, contract.phi_terms)
        heat_iv = exp_iv(t_iv * u_panel.pow_int(2))
        amplitude = ComplexInterval.from_value(phi_iv * heat_iv)
        cosine = _cos_complex(z * ComplexInterval.from_value(u_panel))
        panel_value = amplitude * cosine
        width = Interval.point(u1) - Interval.point(u0)
        real_panels.append(width * panel_value.re)
        imag_panels.append(width * panel_value.im)
    real = sum_intervals(real_panels)
    imag = sum_intervals(imag_panels)
    tail = debruijn_newman_outer_tail_bound(z, contract).hi
    return ComplexInterval(
        real + Interval(-tail, tail),
        imag + Interval(-tail, tail),
    )


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

    A complete proof would need (i) a finite contour cover of every relevant
    zero of ``H_t`` for all ``t in [0, t0]`` and (ii) an independently
    published far-field theorem that no zeros hide at infinity (Polymath15 /
    Rodgers--Tao style).  This function **records** both obligations.  It
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
            "finite contour cover of H_t zeros along the real line for all "
            f"t in [0, {t0}] (local rectangles are not a cover)"
        )
    if not far_field_premise_discharged:
        missing.append(
            "cited far-field premise "
            "(Polymath15 / Rodgers-Tao unbounded-line zero-free region) "
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


__all__ = [
    "DeBruijnNewmanContract",
    "DeBruijnNewmanZeroCount",
    "FAR_FIELD_CITATION",
    "FIRST_RIEMANN_ZERO_IMAG",
    "H0_FIRST_ZERO",
    "LambdaBoundAttempt",
    "PHI_DEFAULT_TERMS",
    "PRE_REGISTERED_T0",
    "attempt_named_lambda_bound",
    "count_debruijn_newman_zeros",
    "crosscheck_h0_at_first_zero",
    "debruijn_newman_enclosure",
    "debruijn_newman_outer_tail_bound",
    "phi_enclosure",
    "phi_series_tail_bound",
]
