# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified local series at a singular / branch point: Frobenius + Puiseux.

Two classical "expand around a singularity" constructions, both packaged on top
of :class:`~omnibias.core.verified.sequence_space.ValidatedSeries` so every
returned series carries an explicit, honestly-derived tail bound rather than a
bare truncated float sum.

**Part A -- Frobenius series of a 2nd-order linear ODE.** Given a regular
singular point at ``x = 0`` of

.. math::

    x^2 y'' + x\,p(x)\,y' + q(x)\,y = 0, \qquad
    p(x) = \sum_n p_n x^n, \quad q(x) = \sum_n q_n x^n,

the indicial equation ``r(r-1) + p_0 r + q_0 = 0`` fixes the two candidate
leading exponents. :func:`indicial_roots` solves it as a certified quadratic
directly with :meth:`Interval.sqrt`
(:mod:`omnibias.core.verified.rootfind` targets *general* nonlinear root
isolation via bisection / interval Newton; a closed-form quadratic has an exact
formula and does not need that machinery). Given one root ``r``,
:func:`frobenius_coefficients` computes the coefficients ``a_n`` of
``y = x^r \sum_n a_n x^n`` (``a_0 = 1``) via the standard three-term-per-step
recurrence, in :class:`Interval` arithmetic so genuine coefficient uncertainty
propagates soundly.

**Scope carve-out (Part A):** when the two indicial roots differ by a
non-negative integer (including the repeated-root case, difference ``0``), the
second solution needs a logarithmic term; that generalized construction is
*out of scope* here and every entry point raises :class:`ValueError` instead of
silently returning an unsound or incomplete series.

**Part B -- Puiseux branch of an algebraic curve.** Given
``F(x, y) = sum_{i,j} c_{ij} x^i y^j = 0`` and a *caller-supplied* ramification
index ``e`` (deriving the full Newton polygon automatically is out of scope --
see :func:`newton_polygon_leading_term`), a single Newton-polygon
leading-term step recovers the leading exponent/coefficient of a branch
``y = c_0 x^{k_0/e} + ...``. Substituting the local uniformizer ``x = t^e``
turns the rest of the branch into an ordinary (integer-power) series
``y(t) = t^{k_0} \sum_j d_j t^j``, whose coefficients
:func:`puiseux_coefficients` computes via a linearized order-by-order
recurrence (the classical technique of solving ``H(t, w) = F(t^e, t^{k_0} w) /
t^{m_0} = 0`` for ``w(t)``, valid once ``c_0`` is a *simple* root of the ``t=0``
edge polynomial ``\phi(w) = H(0, w)`` -- non-simple roots and the general
multi-edge Newton polygon are out of scope, and both raise ``ValueError`` /
``NotImplementedError`` rather than guessing).

**Shared tail contract.** Both parts hand their last retained coefficient and a
*caller-supplied* consecutive-ratio hypothesis (``|a_{n+1}/a_n| <= ratio`` from
the last retained index onward) to
:func:`consecutive_ratio_tail_bound`, a thin reindexing of
:func:`~omnibias.core.verified.sequence_space.geometric_tail_bound` into the
"from-the-last-kept-term" convention a truncated recurrence naturally produces.
Exactly as with :mod:`omnibias.core.verified.series`, the ratio is trusted, not
inferred: this module checks only that the resulting weighted ratio
``nu*ratio < 1`` is self-consistent (raising otherwise); an analytically wrong
``ratio`` yields a silently unsound (too tight) tail, which is why every
consumer of this contract must justify ``ratio`` from the recurrence's actual
asymptotics before trusting the result.

Both :class:`FrobeniusSolution` and :class:`PuiseuxBranch` are pure algebraic /
interval constructions -- **not** the closed-form sigma tower.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.sequence_space import ValidatedSeries, convolve, geometric_tail_bound
from omnibias.core.verified.transcend import exp_iv, ln_iv

#: ``F(x, y) = sum_{(i, j)} coeffs[(i, j)] * x**i * y**j``, finite support,
#: non-negative integer exponents.
BivariateCoeffs = Mapping[tuple[int, int], IntervalLike]

#: Above this many candidate integers, a root-gap or edge search refuses rather
#: than looping over an impractically (or unboundedly, for a very wide
#: uncertain interval) large integer range.
_MAX_INTEGER_SCAN = 4096


# --------------------------------------------------------------------------- #
# shared small helpers
# --------------------------------------------------------------------------- #
def _intervals_overlap(a: Interval, b: Interval) -> bool:
    return a.lo <= b.hi and b.lo <= a.hi


def _trunc_list(values: Sequence[Interval], length: int) -> list[Interval]:
    """Pad or cut ``values`` to exactly ``length`` entries (padding with 0)."""
    out = list(values[:length])
    out.extend(Interval.point(0.0) for _ in range(length - len(out)))
    return out


def _poly_mul_trunc(a: Sequence[Interval], b: Sequence[Interval], length: int) -> list[Interval]:
    """``(a * b) mod t**length`` via Cauchy convolution, truncated to ``length``."""
    return _trunc_list(convolve(list(a), list(b)), length)


def _poly_pow_trunc(base: Sequence[Interval], power: int, length: int) -> list[Interval]:
    """``base(t)**power mod t**length`` by repeated truncated multiplication."""
    result = _trunc_list([Interval.point(1.0)], length)
    factor = _trunc_list(base, length)
    for _ in range(power):
        result = _poly_mul_trunc(result, factor, length)
    return result


def _horner_kept(coeffs: Sequence[Interval], x: Interval) -> Interval:
    if not coeffs:
        return Interval.point(0.0)
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = acc * x + c
    return acc


def _validated_series_value(series: ValidatedSeries, x: IntervalLike) -> Interval:
    """Sound enclosure of ``sum_n a_n x**n`` (kept Horner + symmetric tail).

    Requires ``|x| <= series.nu``: past that radius the tail bound (a weighted
    ``ell^1_nu`` norm) no longer dominates the omitted terms at ``x``.
    """
    x_iv = Interval.from_value(x)
    if x_iv.mag > series.nu:
        raise ValueError(
            f"|x| up to {x_iv.mag!r} exceeds the validated radius nu={series.nu!r}"
        )
    kept = _horner_kept(series.coeffs, x_iv)
    radius = series.tail.hi
    return kept + Interval(-radius, radius)


def consecutive_ratio_tail_bound(
    last_term: IntervalLike, ratio: float, nu: float, n_trunc: int
) -> Interval:
    r"""Nu-weighted tail bound from a caller-supplied consecutive-ratio hypothesis.

    Given the hypothesis ``|a_k| <= |a_{n_trunc}| * ratio**(k - n_trunc)`` for
    every ``k > n_trunc`` -- i.e. ``ratio`` bounds every consecutive ratio
    ``|a_{n+1}/a_n|`` from ``n = n_trunc`` (the last *retained* index) onward --
    returns a sound bound on ``sum_{k>n_trunc} |a_k| nu**k`` by translating that
    hypothesis into :func:`~omnibias.core.verified.sequence_space.geometric_tail_bound`'s
    ``|a_k| <= coeff_bound*ratio**k`` convention (``coeff_bound =
    |a_{n_trunc}| / ratio**n_trunc``) and delegating to it.

    As with :func:`~omnibias.core.verified.series.certified_geometric_series_sum`,
    ``ratio`` is trusted, not inferred from the retained terms: this function
    only checks that the resulting weighted ratio ``nu*ratio < 1`` is
    self-consistent (:func:`geometric_tail_bound` raises ``ValueError``
    otherwise). A ``ratio`` that is analytically wrong for the true sequence
    silently yields a too-tight (unsound) bound -- callers must justify it.
    """
    if ratio <= 0.0:
        raise ValueError(f"ratio must be > 0, got {ratio!r}")
    if n_trunc < 0:
        raise ValueError(f"n_trunc must be >= 0, got {n_trunc}")
    last_mag = Interval.from_value(last_term).mag
    coeff_bound = last_mag / (ratio**n_trunc)
    bound = geometric_tail_bound(coeff_bound, ratio, nu, n_trunc)
    if bound.lo < 0.0:
        # A tail bound is provably a non-negative quantity (a bound on a sum of
        # weighted magnitudes); a negative `lo` can only be an outward-rounding
        # artifact of a product/subtraction that computed to exactly 0.0 (e.g.
        # `coeff_bound` underflowing through denormal range). Clamping to 0.0
        # only removes floating-point noise, never real information.
        bound = Interval(0.0, bound.hi)
    return bound


# --------------------------------------------------------------------------- #
# Part A -- Frobenius series of a regular singular point
# --------------------------------------------------------------------------- #
def indicial_roots(p0: IntervalLike, q0: IntervalLike) -> tuple[Interval, Interval]:
    r"""Certified real roots of the indicial equation ``r(r-1) + p0*r + q0 = 0``.

    Equivalent to ``r**2 + (p0-1)*r + q0 = 0``, solved by the certified
    quadratic formula ``r = (-(p0-1) +/- sqrt(disc)) / 2`` with
    ``disc = (p0-1)**2 - 4*q0``, using :meth:`Interval.sqrt` directly.
    :mod:`omnibias.core.verified.rootfind` (``interval_newton`` /
    ``bisection_bracket``) targets *general* nonlinear root isolation and has
    no closed-form-quadratic primitive to reuse; a bisection/Newton search
    would also be strictly weaker here since the quadratic formula is exact.

    Returns ``(r_plus, r_minus)`` for the ``+``/``-`` branches respectively.
    Raises ``ValueError`` if ``disc`` is certifiably negative (``disc.hi <
    0.0``): a complex-conjugate pair of exponents, out of scope for the
    real-exponent Frobenius series built here.

    A discriminant enclosure that straddles zero (``disc.lo < 0.0 <=
    disc.hi``) is *not* rejected: outward rounding of an exactly-zero
    subtraction (e.g. a genuine repeated root, ``disc == 0``) unavoidably
    produces a sign-straddling interval one ulp wide, so requiring
    ``disc.lo >= 0.0`` would spuriously reject every repeated-root case.
    Instead the lower bound is clamped to ``max(disc.lo, 0.0)`` before
    taking the square root -- sound whenever the true discriminant is
    genuinely non-negative (the only case a real Frobenius exponent
    exists), and this is the only case in which the returned roots are
    meaningful. If the *true* discriminant were actually negative while its
    enclosure still straddled zero (both endpoints from genuinely uncertain
    interval ``p0``/``q0`` inputs, not from rounding noise), this clamp
    would optimistically report a spurious real root pair rather than
    raising; callers passing genuinely wide interval coefficients should
    additionally check ``disc.hi`` themselves if that distinction matters.
    """
    p0_iv = Interval.from_value(p0)
    q0_iv = Interval.from_value(q0)
    b = p0_iv - 1
    disc = b.pow_int(2) - Interval.point(4.0) * q0_iv
    if disc.hi < 0.0:
        raise ValueError(
            f"indicial discriminant {disc!r} is certifiably negative; "
            "complex-conjugate indicial roots are out of scope"
        )
    sqrt_disc = Interval(max(disc.lo, 0.0), disc.hi).sqrt()
    r_plus = (-b + sqrt_disc) / 2.0
    r_minus = (-b - sqrt_disc) / 2.0
    return r_plus, r_minus


def roots_gap_blocks_second_solution(r1: Interval, r2: Interval) -> bool:
    """True iff ``r1 - r2`` is certifiably a non-negative integer (or this
    cannot be certified false), i.e. whichever of ``r1``/``r2`` is the
    *smaller* root would need the log-term second solution to reach past the
    resonant order -- see :func:`frobenius_coefficients`.

    Conservative: an interval difference that merely *might* contain a
    non-negative integer (genuine uncertainty, not yet excluded) is treated as
    blocking, matching the "raise rather than silently proceed" doctrine. This
    is informational only: :func:`frobenius_coefficients` does not gate on it
    directly (that would also reject the perfectly well-defined *larger*-root
    series), and instead detects the same degeneracy per recurrence step,
    exactly where it would actually bite.
    """
    diff = r1 - r2
    if diff.hi < 0.0:
        return False  # certifiably negative: cannot equal any n >= 0
    lo_n = max(0, math.ceil(diff.lo))
    hi_n = math.floor(diff.hi)
    if hi_n < lo_n:
        return False
    if hi_n - lo_n > _MAX_INTEGER_SCAN:
        raise ValueError(
            f"indicial root difference {diff!r} is too wide to certify against "
            "a non-negative-integer gap"
        )
    return any(diff.contains(float(n)) for n in range(lo_n, hi_n + 1))


def frobenius_coefficients(
    p: Sequence[IntervalLike],
    q: Sequence[IntervalLike],
    root: IntervalLike,
    *,
    num_terms: int,
) -> list[Interval]:
    r"""Certified Frobenius coefficients ``a_0, ..., a_{num_terms-1}`` (``a_0=1``).

    Solves ``x**2 y'' + x p(x) y' + q(x) y = 0`` for ``y = x**root * sum_n a_n
    x**n`` given the coefficient sequences ``p`` (``p(x) = sum p[n] x**n``) and
    ``q`` (analogously); indices beyond the supplied length are treated as
    exactly ``0``.

    ``root`` must coincide (in the sense of interval overlap) with one of the
    two indicial roots recomputed internally from ``p[0]``/``q[0]`` via
    :func:`indicial_roots`, or ``ValueError`` is raised. Whether ``root`` is
    the larger or the smaller of a pair that differs by a non-negative
    integer, the per-step indicial divisor ``I(root+m) = (root+m)(root+m-1) +
    p0*(root+m) + q0`` is checked at every ``m`` (``1 <= m < num_terms``): it
    can never vanish when ``root`` is the *larger* root (or the roots
    coincide), but for the *smaller* root of a gapped pair it becomes exactly
    ``0`` at ``m = r_larger - r_smaller`` -- precisely the classical resonance
    that forces a logarithmic term in the second solution. Reaching that step
    (``num_terms`` large enough) raises ``ValueError`` rather than attempting
    that log-term construction, which is out of scope; a caller who only
    wants the coefficients *before* the resonance (``num_terms <= gap``) is
    unaffected.
    """
    if num_terms < 1:
        raise ValueError(f"num_terms must be >= 1, got {num_terms}")
    p_iv = [Interval.from_value(c) for c in p]
    q_iv = [Interval.from_value(c) for c in q]

    def p_at(k: int) -> Interval:
        return p_iv[k] if 0 <= k < len(p_iv) else Interval.point(0.0)

    def q_at(k: int) -> Interval:
        return q_iv[k] if 0 <= k < len(q_iv) else Interval.point(0.0)

    p0, q0 = p_at(0), q_at(0)
    r = Interval.from_value(root)
    r_plus, r_minus = indicial_roots(p0, q0)
    if not (_intervals_overlap(r, r_plus) or _intervals_overlap(r, r_minus)):
        raise ValueError(
            f"root {r!r} does not match either certified indicial root "
            f"({r_plus!r}, {r_minus!r})"
        )

    a: list[Interval] = [Interval.point(1.0)]
    for m in range(1, num_terms):
        rm = r + m
        indicial_m = rm * (rm - 1) + p0 * rm + q0
        if indicial_m.contains_zero():
            raise ValueError(
                f"indicial divisor at root+{m} is not certifiably nonzero "
                f"({indicial_m!r}); the indicial roots differ by the "
                f"non-negative integer {m} and `root` is the smaller one, so "
                "the recurrence resonates here -- the log-term second "
                "solution needed to continue is out of scope"
            )
        acc = Interval.point(0.0)
        for n in range(m):
            rn = r + n
            acc = acc + (p_at(m - n) * rn + q_at(m - n)) * a[n]
        a.append(-(acc / indicial_m))
    return a


@dataclass
class FrobeniusSolution:
    r"""A certified Frobenius series ``y = x**root * sum_n a_n x**n``.

    Solves ``x**2 y'' + x p(x) y' + q(x) y = 0`` near the regular singular
    point ``x = 0`` for the caller-selected indicial root ``root``. ``series``
    is a :class:`ValidatedSeries` of the ``a_n`` (``a_0 = 1``) whose ``tail``
    bounds the ``nu``-weighted norm of everything past the retained
    coefficients under the hypothesis used to build it (see
    :func:`solve_frobenius`).
    """

    root: Interval
    series: ValidatedSeries

    def evaluate(self, x: IntervalLike) -> Interval:
        r"""Sound enclosure of ``y(x) = x**root * sum_n a_n x**n`` for real ``x > 0``.

        Requires ``0 < x <= series.nu``. ``x**root`` is evaluated as
        ``exp(root * ln(x))`` (:mod:`omnibias.core.verified.transcend`): a
        rigorous but *not closed-form* transcendental enclosure -- the
        founding sigma tower is not involved anywhere in this module.
        """
        x_iv = Interval.from_value(x)
        if x_iv.lo <= 0.0:
            raise ValueError("FrobeniusSolution.evaluate requires x > 0 (real x**root branch)")
        series_val = _validated_series_value(self.series, x_iv)
        x_pow_r = exp_iv(self.root * ln_iv(x_iv))
        return series_val * x_pow_r


def solve_frobenius(
    p: Sequence[IntervalLike],
    q: Sequence[IntervalLike],
    root: IntervalLike,
    *,
    num_terms: int,
    ratio: float,
    nu: float = 1.0,
) -> FrobeniusSolution:
    """Compute :func:`frobenius_coefficients` and package as a `FrobeniusSolution`.

    ``ratio``/``nu`` feed :func:`consecutive_ratio_tail_bound` on the last
    retained coefficient; see that function's docstring for exactly which
    hypothesis is being trusted.
    """
    coeffs = frobenius_coefficients(p, q, root, num_terms=num_terms)
    tail = consecutive_ratio_tail_bound(coeffs[-1], ratio, nu, num_terms - 1)
    return FrobeniusSolution(Interval.from_value(root), ValidatedSeries(coeffs, tail, nu))


# --------------------------------------------------------------------------- #
# Part B -- Puiseux branch of an algebraic curve
# --------------------------------------------------------------------------- #
def _support(coeffs: BivariateCoeffs) -> list[tuple[int, int, Interval]]:
    out: list[tuple[int, int, Interval]] = []
    for (i, j), raw in coeffs.items():
        if i < 0 or j < 0:
            raise ValueError(f"bivariate exponents must be non-negative, got ({i}, {j})")
        c = Interval.from_value(raw)
        if c.lo == 0.0 and c.hi == 0.0:
            continue
        out.append((i, j, c))
    return out


def _shifted_coefficients(
    support: Sequence[tuple[int, int, Interval]], e: int, k0: int
) -> tuple[int, dict[int, list[Interval]]]:
    r"""``(m0, Cj)``: the minimal weighted ``t``-degree and, for each ``y``-power
    ``j`` present, the polynomial-in-``t`` coefficient list of
    ``F(t**e, t**k0 * w) / t**m0`` (index ``0`` is the minimal weighted degree).
    """
    if not support:
        raise ValueError("F has no nonzero monomials")
    weighted = [(e * i + k0 * j, i, j, c) for i, j, c in support]
    m0 = min(w for w, _, _, _ in weighted)
    raw: dict[int, dict[int, Interval]] = {}
    for w, _i, j, c in weighted:
        shifted = w - m0
        row = raw.setdefault(j, {})
        row[shifted] = row.get(shifted, Interval.point(0.0)) + c
    out: dict[int, list[Interval]] = {}
    for j, entries in raw.items():
        length = max(entries) + 1
        lst = [Interval.point(0.0)] * length
        for shifted, c in entries.items():
            lst[shifted] = lst[shifted] + c
        out[j] = lst
    return m0, out


def newton_polygon_leading_term(
    coeffs: BivariateCoeffs, e: int
) -> tuple[int, tuple[Interval, ...]]:
    r"""A single Newton-polygon leading-term step for a branch ``y ~ c0 x**(k0/e)``.

    Given ``F(x, y) = sum_{i,j} coeffs[(i,j)] x**i y**j`` and a ramification
    index ``e`` (caller-supplied; deriving the Newton polygon automatically is
    out of scope), enumerates every candidate leading exponent ``k0 >= 0``
    forced by a pair of monomials (``e*i1 + k0*j1 = e*i2 + k0*j2``), keeps the
    ones for which that ``k0`` makes *at least two* monomials jointly attain
    the minimal weighted degree ``e*i + k0*j`` (a genuine Newton-polygon edge,
    not a one-term coincidence), and requires exactly one such ``k0``
    (raising ``ValueError`` if none or several balance -- the general
    multi-edge polygon is out of scope).

    The edge's two-term polynomial ``phi(c) = A*c**j_lo + B*c**j_hi`` (``A``,
    ``B`` both nonzero by construction) is then solved for nonzero ``c``:
    ``c**(j_hi-j_lo) = -A/B``. Only ``degree = j_hi - j_lo`` equal to ``1``
    (unique real root) or ``2`` (returned as both ``+`` and ``-`` real square
    roots, via :meth:`Interval.sqrt`) is supported; other degrees raise
    ``NotImplementedError``.

    Returns ``(k0, roots)`` with ``roots`` a 1- or 2-tuple of candidate ``c0``
    enclosures.
    """
    if e < 1:
        raise ValueError(f"ramification index e must be a positive integer, got {e}")
    support = _support(coeffs)
    if len(support) < 2:
        raise ValueError("F needs >= 2 nonzero monomials to form a Newton-polygon edge")

    candidates: set[int] = set()
    for a_idx in range(len(support)):
        for b_idx in range(a_idx + 1, len(support)):
            i1, j1, _ = support[a_idx]
            i2, j2, _ = support[b_idx]
            if j1 == j2:
                continue
            numerator = e * (i2 - i1)
            denominator = j1 - j2
            if numerator % denominator != 0:
                continue
            k0 = numerator // denominator
            if k0 >= 0:
                candidates.add(k0)
    if len(candidates) > _MAX_INTEGER_SCAN:
        raise ValueError("too many candidate leading exponents to certify uniqueness")

    valid: list[tuple[int, dict[int, Interval]]] = []
    for k0 in sorted(candidates):
        _m0, cj_lists = _shifted_coefficients(support, e, k0)
        edge = {j: lst[0] for j, lst in cj_lists.items() if not (lst[0].lo == 0.0 and lst[0].hi == 0.0)}
        if len(edge) >= 2:
            valid.append((k0, edge))

    if not valid:
        raise ValueError(
            f"no candidate leading exponent balances the Newton polygon at ramification e={e}"
        )
    if len(valid) > 1:
        raise ValueError(
            "ambiguous Newton-polygon leading term: candidates "
            f"{[k for k, _ in valid]} all balance at e={e}; automatic full-polygon "
            "resolution is out of scope -- disambiguate externally"
        )
    k0, edge = valid[0]
    powers = sorted(edge)
    if len(powers) != 2:
        raise NotImplementedError(
            f"edge polynomial touches {len(powers)} distinct y-powers {powers}; "
            "only two-term (linear or quadratic) edges are supported"
        )
    j_lo, j_hi = powers
    a_coef, b_coef = edge[j_lo], edge[j_hi]
    if b_coef.contains_zero():
        raise ValueError("edge leading coefficient is not certifiably nonzero")
    degree = j_hi - j_lo
    ratio = -(a_coef / b_coef)
    roots: tuple[Interval, ...]
    if degree == 1:
        roots = (ratio,)
    elif degree == 2:
        if ratio.lo < 0.0:
            raise ValueError(
                f"edge balance requires a non-negative c0**2={ratio!r} for a real root"
            )
        magnitude = ratio.sqrt()
        roots = (magnitude, -magnitude)
    else:
        raise NotImplementedError(
            f"edge power {degree} needs a degree-{degree} solve; only degree 1 or 2 "
            "edges are supported"
        )
    return k0, roots


def puiseux_coefficients(
    coeffs: BivariateCoeffs,
    e: int,
    k0: int,
    c0: IntervalLike,
    *,
    num_terms: int,
) -> list[Interval]:
    r"""Certified coefficients ``d_0=c0, d_1, ..., d_{num_terms-1}`` of ``w(t)``.

    ``y(t) = t**k0 * w(t)`` is a branch of ``F(t**e, y) = 0`` (the local
    uniformizer substitution ``x = t**e``). Writing
    ``H(t, w) = F(t**e, t**k0 * w) / t**m0`` (``m0`` the minimal weighted
    ``t``-degree, recomputed internally), ``phi(w) = H(0, w)`` is the
    Newton-polygon edge polynomial from :func:`newton_polygon_leading_term`.

    Requires ``phi(c0)`` to certifiably contain ``0`` (``c0`` really is a root
    of the edge polynomial for this ``(e, k0)``) and ``phi'(c0)`` to be
    certifiably nonzero (a *simple* root -- a repeated root would need a
    higher-order local expansion, out of scope). Given that, ``H(t,w(t)) = 0``
    is solved order by order: writing ``w(t) = c0 + d1 t + ... + d_n t**n +
    O(t**(n+1))``, the coefficient of ``t**n`` in ``H(t, w(t))`` is *affine* in
    the new unknown ``d_n`` -- ``R_n + phi'(c0) * d_n`` -- because ``d_n``
    only ever appears linearly (picking the ``t**n`` slot from exactly one of
    the ``j`` copies of ``w`` in a ``w**j`` term, the rest forced to their
    known ``t**0 = c0`` value). Hence ``d_n = -R_n / phi'(c0)``, computed via
    truncated interval power-series arithmetic (:func:`~omnibias.core.verified.sequence_space.convolve`).
    """
    if e < 1:
        raise ValueError(f"ramification index e must be a positive integer, got {e}")
    if k0 < 0:
        raise ValueError(f"leading exponent k0 must be >= 0, got {k0}")
    if num_terms < 1:
        raise ValueError(f"num_terms must be >= 1, got {num_terms}")
    support = _support(coeffs)
    _m0, cj_lists = _shifted_coefficients(support, e, k0)
    if not cj_lists:
        raise ValueError("F does not depend on y")

    c0_iv = Interval.from_value(c0)
    phi_c0 = Interval.point(0.0)
    phi_prime_c0 = Interval.point(0.0)
    for j, lst in cj_lists.items():
        coeff0 = lst[0]
        phi_c0 = phi_c0 + coeff0 * c0_iv.pow_int(j)
        if j >= 1:
            phi_prime_c0 = phi_prime_c0 + Interval.point(float(j)) * coeff0 * c0_iv.pow_int(j - 1)
    if not phi_c0.contains_zero():
        raise ValueError(
            f"c0={c0_iv!r} is not certifiably a root of the Newton-polygon edge "
            f"polynomial (phi(c0)={phi_c0!r}) for (e={e}, k0={k0})"
        )
    if phi_prime_c0.contains_zero():
        raise ValueError(
            f"edge-polynomial derivative at c0 is not certifiably nonzero "
            f"({phi_prime_c0!r}); c0 is a non-simple root, out of scope"
        )

    d: list[Interval] = [c0_iv]
    for n in range(1, num_terms):
        length = n + 1
        w_partial = _trunc_list(d, length)  # index n implicitly 0 (unknown d_n)
        h_trunc = [Interval.point(0.0) for _ in range(length)]
        for j, cj in cj_lists.items():
            wj = _poly_pow_trunc(w_partial, j, length)
            cj_trunc = _trunc_list(cj, length)
            prod = _poly_mul_trunc(cj_trunc, wj, length)
            for idx in range(length):
                h_trunc[idx] = h_trunc[idx] + prod[idx]
        r_n = h_trunc[n]
        d.append(-(r_n / phi_prime_c0))
    return d


@dataclass
class PuiseuxBranch:
    r"""A certified Puiseux branch ``y(t) = t**k0 * w(t)`` (``x = t**e``).

    ``series`` is a :class:`ValidatedSeries` of the ``w(t)`` coefficients
    ``d_j`` (``d_0 = c0``); ``e``/``k0`` record the ramification index and
    leading exponent so ``y ~ c0 * x**(k0/e)`` to leading order.
    """

    e: int
    k0: int
    series: ValidatedSeries

    def evaluate_t(self, t: IntervalLike) -> Interval:
        """Sound enclosure of ``y(t) = t**k0 * w(t)`` for the local uniformizer ``t``."""
        t_iv = Interval.from_value(t)
        series_val = _validated_series_value(self.series, t_iv)
        return series_val * t_iv.pow_int(self.k0)

    def evaluate_x(self, x: IntervalLike) -> Interval:
        r"""Sound enclosure of ``y(x)`` via the real principal root ``t = x**(1/e)``.

        Requires ``x >= 0`` (the real branch of ``t = x**(1/e)``); ``x**(1/e)``
        for ``x > 0`` is evaluated as ``exp(ln(x)/e)``
        (:mod:`omnibias.core.verified.transcend`), a rigorous but
        not-closed-form transcendental enclosure.
        """
        x_iv = Interval.from_value(x)
        if x_iv.lo < 0.0:
            raise ValueError("PuiseuxBranch.evaluate_x requires x >= 0 for the real branch")
        if x_iv.lo == 0.0 and x_iv.hi == 0.0:
            return Interval.point(0.0) if self.k0 > 0 else self.series.coeffs[0]
        t_iv = exp_iv(ln_iv(x_iv) / float(self.e))
        return self.evaluate_t(t_iv)


def solve_puiseux(
    coeffs: BivariateCoeffs,
    e: int,
    k0: int,
    c0: IntervalLike,
    *,
    num_terms: int,
    ratio: float,
    nu: float = 1.0,
) -> PuiseuxBranch:
    """Compute :func:`puiseux_coefficients` and package as a `PuiseuxBranch`.

    ``ratio``/``nu`` feed :func:`consecutive_ratio_tail_bound` on the last
    retained coefficient; see that function's docstring for exactly which
    hypothesis is being trusted.
    """
    d = puiseux_coefficients(coeffs, e, k0, c0, num_terms=num_terms)
    tail = consecutive_ratio_tail_bound(d[-1], ratio, nu, num_terms - 1)
    return PuiseuxBranch(e, k0, ValidatedSeries(d, tail, nu))


__all__ = [
    "BivariateCoeffs",
    "FrobeniusSolution",
    "PuiseuxBranch",
    "consecutive_ratio_tail_bound",
    "frobenius_coefficients",
    "indicial_roots",
    "newton_polygon_leading_term",
    "puiseux_coefficients",
    "roots_gap_blocks_second_solution",
    "solve_frobenius",
    "solve_puiseux",
]
