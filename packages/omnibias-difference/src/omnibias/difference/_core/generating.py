# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Analytic combinatorics: generating functions + **certified** asymptotic enclosures.

The bridge from the exact special-number coefficients (:mod:`~omnibias.difference._core.bernoulli`,
:mod:`~omnibias.difference._core.euler`, :mod:`~omnibias.difference._core.stirling`) to their
asymptotic behaviour, in two honesty registers:

* **numerical** -- ordinary / exponential generating-function helpers (exact rational
  power-series algebra), singularity-analysis growth rates, and the Moser--Wyman
  saddle-point asymptotic for the Bell numbers (now with the ``R = W(n+1)`` +
  Canfield ``q_n`` correction, ~90x tighter than the raw ``r = W(n)`` form).
* **closed-form / certified** -- rigorous outward-rounded :class:`Interval` *enclosures*
  of the numbers themselves, closing the "asymptotics have no error bars" gap:

  - :func:`bernoulli_enclosure` -- ``B_{2m} = (-1)^{m+1} 2 (2m)! zeta(2m) / (2 pi)^{2m}``,
    with a certified ``zeta(2m)`` (:func:`zeta_int_enclosure`) and the certified
    ``PI_IV``;
  - :func:`euler_enclosure` -- ``E_{2m} = (-1)^m 2^{2m+2} (2m)! beta(2m+1) / pi^{2m+1}``,
    with a certified Dirichlet ``beta`` (:func:`dirichlet_beta_odd_enclosure`);
  - :func:`bell_dobinski_enclosure` -- Dobinski ``B_n = e^{-1} sum_{k>=0} k^n / k!``
    with a rigorous geometric-tail bound and the certified ``E_IV``.

Each certified enclosure **provably contains** the true (exactly known) value, so the
tests verify containment on a grid, and its relative width shrinks with the index --
that is the error bar the float asymptotics lacked.

The **transfer theorem** layer (:func:`transfer_theorem`,
:func:`singular_template_coefficient`, :func:`dominant_pole_coefficient_asymptotic`)
maps an OGF singularity (pole / algebraic branch point) to its ``[z^n]`` coefficient
asymptotics -- the leading term is **numerical**, but the exact coefficient of the
singular template is an exact rational (**closed-form / certified**), so the classical
asymptotic ships with a rigorous error bar.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import exp, expm1, factorial, gamma, lgamma, log, pi

from omnibias.core.verified import E_IV, PI_IV
from omnibias.core.verified.interval import Interval
from omnibias.difference._core.asymptotics import _lambert_w

Rational = Fraction | int

# Direct :func:`bernoulli_enclosure` / :func:`euler_enclosure` overflow ``float``
# once ``(2m)!`` exceeds the double range (``171!`` overflows); beyond this a
# log-space form would be needed (out of scope for the enclosure demonstration).
_FACTORIAL_FLOAT_LIMIT = 170


# --------------------------------------------------------------------------- #
# Generating-function algebra (exact rational, numerical register)            #
# --------------------------------------------------------------------------- #
def exponential_generating_coeffs(seq: Sequence[Rational]) -> tuple[Fraction, ...]:
    r"""EGF coefficients ``a_n / n!`` of a sequence ``a_0, a_1, ...`` (exact)."""
    return tuple(Fraction(a) / factorial(n) for n, a in enumerate(seq))


def ordinary_from_exponential(egf_coeffs: Sequence[Rational]) -> tuple[Fraction, ...]:
    r"""Recover the sequence ``a_n = n! c_n`` from EGF coefficients ``c_n`` (inverse)."""
    return tuple(Fraction(c) * factorial(n) for n, c in enumerate(egf_coeffs))


def cauchy_product(a: Sequence[Rational], b: Sequence[Rational]) -> tuple[Fraction, ...]:
    r"""Coefficients of the OGF product ``(sum a_i x^i)(sum b_j x^j)`` (exact)."""
    af = [Fraction(x) for x in a]
    bf = [Fraction(x) for x in b]
    if not af or not bf:
        raise ValueError("cauchy_product needs non-empty coefficient lists")
    out: list[Fraction] = []
    for n in range(len(af) + len(bf) - 1):
        lo = max(0, n - len(bf) + 1)
        hi = min(n, len(af) - 1)
        out.append(sum((af[i] * bf[n - i] for i in range(lo, hi + 1)), Fraction(0)))
    return tuple(out)


def rational_ogf_coefficients(
    numer: Sequence[Rational], denom: Sequence[Rational], count: int
) -> tuple[Fraction, ...]:
    r"""Power-series coefficients ``[x^0..x^{count-1}]`` of ``numer(x) / denom(x)`` (exact).

    Both polynomials are given in **ascending** order (``[c_0, c_1, ...]``); the
    denominator's constant term must be non-zero. This is the exact linear-recurrence
    unrolling ``c_n = (num_n - sum_{k>=1} den_k c_{n-k}) / den_0`` -- e.g. the Fibonacci
    OGF ``x / (1 - x - x^2)`` (``numer=[0, 1]``, ``denom=[1, -1, -1]``) yields
    ``0, 1, 1, 2, 3, 5, ...``.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    num = [Fraction(x) for x in numer]
    den = [Fraction(x) for x in denom]
    if not den or den[0] == 0:
        raise ValueError("denom must be non-empty with a non-zero constant term")
    coeffs: list[Fraction] = []
    for n in range(count):
        acc = num[n] if n < len(num) else Fraction(0)
        for k in range(1, min(n, len(den) - 1) + 1):
            acc -= den[k] * coeffs[n - k]
        coeffs.append(acc / den[0])
    return tuple(coeffs)


def _polynomial_roots(coeffs: Sequence[complex]) -> list[complex]:
    """All complex roots of an ascending-order polynomial (Durand-Kerner iteration)."""
    c = [complex(x) for x in coeffs]
    while len(c) > 1 and c[-1] == 0:
        c = c[:-1]
    degree = len(c) - 1
    if degree < 1:
        return []
    lead = c[-1]
    monic = [ci / lead for ci in c]  # ascending, monic

    def peval(x: complex) -> complex:
        acc = 0j
        for ci in reversed(monic):
            acc = acc * x + ci
        return acc

    roots = [(0.4 + 0.9j) ** k for k in range(degree)]
    for _ in range(500):
        max_delta = 0.0
        updated = roots[:]
        for i in range(degree):
            xi = roots[i]
            denom = 1 + 0j
            for j in range(degree):
                if j != i:
                    denom *= xi - roots[j]
            if denom == 0:
                continue
            delta = peval(xi) / denom
            updated[i] = xi - delta
            max_delta = max(max_delta, abs(delta))
        roots = updated
        if max_delta < 1e-15:
            break
    return roots


def rational_ogf_growth_base(denom: Sequence[Rational]) -> float:
    r"""Exponential growth base ``1/|rho|`` from the dominant singularity of a rational OGF.

    ``rho`` is the smallest-modulus root of the (ascending-order) denominator; by
    singularity analysis the coefficients grow like ``rho^{-n}`` up to a subexponential
    factor. Numerical (root-finding). E.g. the Fibonacci denominator ``[1, -1, -1]``
    gives the golden ratio ``phi`` and ``[1, -2]`` (OGF ``1/(1-2x)``) gives ``2``.
    """
    roots = _polynomial_roots([complex(Fraction(x)) for x in denom])
    if not roots:
        raise ValueError("denom must be a non-constant polynomial")
    rho = min(abs(r) for r in roots)
    if rho == 0.0:
        raise ValueError("denom has a zero root: not an ordinary-generating-function singularity")
    return 1.0 / rho


def catalan_asymptotic(n: int) -> float:
    r"""Singularity-analysis asymptotic ``C_n ~ 4^n / (sqrt(pi) n^{3/2})`` (numerical).

    The OGF ``(1 - sqrt(1 - 4x)) / (2x)`` has its dominant singularity at ``x = 1/4``;
    the square-root branch point gives the ``n^{-3/2}`` sub-exponential factor.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return exp(n * log(4.0) - 0.5 * log(pi) - 1.5 * log(n))


# --------------------------------------------------------------------------- #
# Singularity analysis: the transfer theorem (Flajolet-Sedgewick)             #
# --------------------------------------------------------------------------- #
def singular_template_coefficient(exponent: Rational, n: int) -> Fraction:
    r"""Exact ``[z^n] (1 - z)^{-exponent}`` for any rational ``exponent`` (closed-form).

    The generalised binomial ``binom(n + exponent - 1, n) = prod_{j=0}^{n-1}
    (exponent + j) / (j + 1)``, a product of rationals -- so the coefficient is an
    **exact** :class:`~fractions.Fraction`, with no ``Gamma`` evaluation. This is the
    standard-function scale whose coefficients the Flajolet-Sedgewick transfer theorem
    maps a singularity onto: ``exponent = 1`` gives the geometric ``1``s (simple pole),
    ``exponent = -1/2`` gives ``1, -1/2, -1/8, ...`` (the square-root branch point behind
    Catalan / tree / lattice-path counts).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    alpha = Fraction(exponent)
    product = Fraction(1)
    for j in range(n):
        product *= (alpha + j) / Fraction(j + 1)
    return product


@dataclass(frozen=True)
class TransferEstimate:
    r"""The transfer theorem's coefficient asymptotic with a **certified** error bar.

    For a singular template ``f(z) = scale * (1 - z/radius)^{-exponent}`` the true
    ``[z^n] f`` is the exact rational :attr:`exact_coefficient` (certified); the
    classical leading asymptotic ``scale * radius^{-n} * n^{exponent-1} /
    Gamma(exponent)`` is :attr:`leading` (numerical). :attr:`abs_error` /
    :attr:`rel_error` rigorously bound how far the leading term is from the exact
    coefficient (``abs_error`` is the outward magnitude of ``exact - leading``).
    """

    n: int
    leading: float
    exact_coefficient: Interval
    abs_error: float
    rel_error: float
    label: str = "numerical asymptotic + certified exact-coefficient enclosure"


def transfer_theorem(
    scale: Rational | float, radius: Rational | float, exponent: Rational, n: int
) -> TransferEstimate:
    r"""Transfer an algebraic OGF singularity to its ``[z^n]`` asymptotics (certified error).

    Maps the singular element ``scale * (1 - z/radius)^{-exponent}`` -- a **pole**
    (``exponent`` a positive integer) or an **algebraic** branch point (``exponent`` a
    non-integer rational) at ``z = radius`` -- to the coefficient asymptotic

    .. math::

        [z^n] f \;\sim\; \text{scale}\;\text{radius}^{-n}\,
            \frac{n^{\text{exponent}-1}}{\Gamma(\text{exponent})},

    returning it (:attr:`~TransferEstimate.leading`, numerical) alongside the **exact**
    coefficient :func:`singular_template_coefficient` scaled by ``scale * radius^{-n}``
    (a certified :class:`Interval`) and the rigorous gap between them. ``exponent`` must
    not be a non-positive integer (where the template is a polynomial and ``Gamma`` has a
    pole); ``n >= 1``.
    """
    if n < 1:
        raise ValueError(f"transfer_theorem needs n >= 1, got {n}")
    alpha = Fraction(exponent)
    if alpha <= 0 and alpha.denominator == 1:
        raise ValueError(
            f"exponent must not be a non-positive integer (template is a polynomial), got {alpha}"
        )
    if radius == 0:
        raise ValueError("radius (the singularity location) must be non-zero")
    template = singular_template_coefficient(alpha, n)
    scale_iv = Interval.from_value(scale if isinstance(scale, float) else Fraction(scale))
    radius_iv = Interval.from_value(radius if isinstance(radius, float) else Fraction(radius))
    exact = scale_iv * radius_iv.pow_int(n).reciprocal() * Interval.from_rational(template)
    a = float(alpha)
    leading = float(scale) * float(radius) ** (-n) * n ** (a - 1.0) / gamma(a)
    abs_error = (exact - Interval.point(leading)).mag
    rel_error = abs_error / abs(leading) if leading != 0.0 else float("inf")
    return TransferEstimate(n, leading, exact, abs_error, rel_error)


def dominant_pole_coefficient_asymptotic(
    numer: Sequence[Rational], denom: Sequence[Rational], n: int
) -> float:
    r"""Leading coefficient asymptotic ``c * rho^{-n}`` of a rational OGF (numerical).

    For ``f = numer/denom`` (ascending order) with a **simple, real, dominant** pole
    ``rho`` (the smallest-modulus denominator root), singularity analysis gives
    ``[z^n] f ~ c rho^{-n}`` with residue-derived ``c = -numer(rho) / (rho
    denom'(rho))``. Numerical (the pole comes from :func:`_polynomial_roots`); raises if
    the dominant pole is complex (an oscillatory asymptotic needs the conjugate pair).
    The exact coefficients themselves are available -- exactly -- from
    :func:`rational_ogf_coefficients`; this is the closed-form *growth law*.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    roots = _polynomial_roots([complex(Fraction(x)) for x in denom])
    if not roots:
        raise ValueError("denom must be a non-constant polynomial")
    rho = min(roots, key=abs)
    if rho == 0:
        raise ValueError("denom has a zero root: not an ordinary-generating-function singularity")
    if abs(rho.imag) > 1e-9 * max(1.0, abs(rho.real)):
        raise ValueError("dominant pole is complex; the coefficient asymptotic is oscillatory")
    num_val = sum(complex(Fraction(numer[i])) * rho**i for i in range(len(numer)))
    dprime = sum(i * complex(Fraction(denom[i])) * rho ** (i - 1) for i in range(1, len(denom)))
    if dprime == 0:
        raise ValueError("dominant pole is not simple (denom'(rho) = 0)")
    c = -num_val / (rho * dprime)
    return c.real * rho.real ** (-n)


# --------------------------------------------------------------------------- #
# Bell saddle-point (refined, numerical register)                             #
# --------------------------------------------------------------------------- #
def log_bell_number_asymptotic_refined(n: int) -> float:
    r"""Refined log Bell-number saddle-point (``R = W(n+1)`` + Canfield ``q_n``).

    Uses the Moser--Wyman / Szekeres--Binet convention ``R e^R = n + 1`` (numerically
    superior to ``r e^r = n``):

    .. math::

        B_n \approx \frac{n!\,e^{e^R - 1}}{R^{n}\sqrt{2\pi(n+1)(R+1)}}\; q_n,

    with the Canfield correction ``q_n = 1 - e^{-R}/12 (1 - 3/(2R) - 10/R^2 - 9/R^3
    + 1/R^4)/(1 + 1/R)^3`` (``B_n / E_n = 1 + O(e^{-R/5})``). Roughly ~90x tighter
    than :func:`~omnibias.difference._core.asymptotics.log_bell_number_asymptotic`
    at ``n = 30`` (~1e-4 vs ~1e-2 relative error). Still **numerical** -- for a
    rigorous bound use :func:`bell_dobinski_enclosure`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    big_r = _lambert_w(float(n + 1))
    log_e = (
        lgamma(n + 1)
        + ((n + 1) / big_r - 1.0)
        - n * log(big_r)
        - 0.5 * log(2.0 * pi * (n + 1) * (big_r + 1.0))
    )
    e_neg_r = big_r / (n + 1.0)  # e^{-R} since R e^R = n + 1
    q = 1.0 - e_neg_r / 12.0 * (
        1.0 - 1.5 / big_r - 10.0 / big_r**2 - 9.0 / big_r**3 + 1.0 / big_r**4
    ) / (1.0 + 1.0 / big_r) ** 3
    return log_e + log(q)


def bell_number_asymptotic_refined(n: int) -> float:
    r"""Refined Moser--Wyman Bell asymptotic; see :func:`log_bell_number_asymptotic_refined`.

    Returns the value as a ``float``, which **overflows** for ``n`` beyond ``~150``
    (``B_n`` exceeds the double range). For large ``n`` -- or any relative-accuracy
    comparison -- use :func:`log_bell_number_asymptotic_refined` and stay in log space
    (see :func:`bell_asymptotic_relative_error`).
    """
    log_val = log_bell_number_asymptotic_refined(n)
    if log_val > 709.78:
        raise OverflowError(
            f"bell_number_asymptotic_refined({n}) overflows float "
            f"(log value {log_val:.1f} > 709.78); use log_bell_number_asymptotic_refined"
        )
    return exp(log_val)


# --------------------------------------------------------------------------- #
# Certified series enclosures (closed-form / verified register)               #
# --------------------------------------------------------------------------- #
def zeta_int_enclosure(s: int, *, terms: int = 64) -> Interval:
    r"""Certified enclosure of ``zeta(s) = sum_{n>=1} n^{-s}`` for integer ``s >= 2``.

    Exact rational partial sum over ``terms`` terms plus a **two-sided** integral-test
    tail bracket ``int_{N+1}^inf x^{-s} dx <= tail <= int_N^inf x^{-s} dx`` (``x^{-s}``
    is decreasing), i.e. ``(N+1)^{1-s}/(s-1) <= tail <= N^{1-s}/(s-1)``. The returned
    interval provably contains the true ``zeta(s)``.
    """
    if s < 2:
        raise ValueError(f"zeta_int_enclosure requires integer s >= 2, got {s}")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    partial = sum((Fraction(1, n**s) for n in range(1, terms + 1)), Fraction(0))
    tail_hi = Fraction(1, (s - 1) * terms ** (s - 1))
    tail_lo = Fraction(1, (s - 1) * (terms + 1) ** (s - 1))
    tail = Interval(Interval.from_rational(tail_lo).lo, Interval.from_rational(tail_hi).hi)
    return Interval.from_rational(partial) + tail


def dirichlet_beta_odd_enclosure(s: int, *, terms: int = 64) -> Interval:
    r"""Certified enclosure of the Dirichlet beta ``beta(s) = sum_k (-1)^k (2k+1)^{-s}``.

    For odd ``s >= 1`` (where the closed form ``beta(2m+1)`` is a rational multiple of
    ``pi^{2m+1}``). The series is alternating with terms decreasing to zero, so the true
    value lies between two consecutive partial sums -- the tight one-term bracket used
    here.
    """
    if s < 1 or s % 2 == 0:
        raise ValueError(f"dirichlet_beta_odd_enclosure requires odd s >= 1, got {s}")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    partial = sum(
        (Fraction((-1) ** k, (2 * k + 1) ** s) for k in range(terms)), Fraction(0)
    )
    nxt = partial + Fraction((-1) ** terms, (2 * terms + 1) ** s)
    lo, hi = (partial, nxt) if partial <= nxt else (nxt, partial)
    return Interval(Interval.from_rational(lo).lo, Interval.from_rational(hi).hi)


def bernoulli_enclosure(n: int) -> Interval:
    r"""Certified signed enclosure of the Bernoulli number ``B_n`` (even ``n >= 2``).

    ``B_{2m} = (-1)^{m+1} \, 2 (2m)! \, zeta(2m) / (2 pi)^{2m}``, composing the exact
    ``(2m)!``, the certified :func:`zeta_int_enclosure`, and the certified ``2 pi`` from
    ``PI_IV``. The enclosure provably contains the exact ``B_n``; its relative width
    ``-> 0`` as ``n`` grows (``zeta(2m) -> 1``) -- the error bar the float asymptotic
    :func:`~omnibias.difference._core.asymptotics.bernoulli_asymptotic` lacks.
    """
    if n < 2 or n % 2 == 1:
        raise ValueError(f"bernoulli_enclosure is defined for even n >= 2, got {n}")
    if n > _FACTORIAL_FLOAT_LIMIT:
        raise ValueError(
            f"bernoulli_enclosure overflows float for n > {_FACTORIAL_FLOAT_LIMIT}; "
            "a log-space enclosure is out of scope"
        )
    m = n // 2
    # Fold the tiny (2 pi)^{-n} factor in *before* the large (2m)! so no
    # intermediate overflows float even when the final B_n is representable.
    scale = Interval.from_rational(2) * (Interval.from_rational(2) * PI_IV).pow_int(n).reciprocal()
    magnitude = scale * Interval.from_rational(factorial(n)) * zeta_int_enclosure(n)
    return magnitude if m % 2 == 1 else -magnitude  # sign (-1)^{m+1}


def euler_enclosure(n: int) -> Interval:
    r"""Certified signed enclosure of the (secant) Euler number ``E_n`` (even ``n >= 2``).

    ``E_{2m} = (-1)^m \, 2^{2m+2} (2m)! \, beta(2m+1) / pi^{2m+1}``, composing the exact
    factors, the certified :func:`dirichlet_beta_odd_enclosure`, and ``PI_IV``. Provably
    contains the exact ``E_n`` with relative width ``-> 0`` (``beta(2m+1) -> 1``).
    """
    if n < 2 or n % 2 == 1:
        raise ValueError(f"euler_enclosure is defined for even n >= 2, got {n}")
    if n > _FACTORIAL_FLOAT_LIMIT:
        raise ValueError(
            f"euler_enclosure overflows float for n > {_FACTORIAL_FLOAT_LIMIT}; "
            "a log-space enclosure is out of scope"
        )
    m = n // 2
    # Fold the tiny 2^{2m+2} / pi^{2m+1} ratio in *before* the large (2m)! so no
    # intermediate overflows float (2^{n+2}*(2m)! alone overflows around n~130
    # even though the final E_n stays representable to n~180).
    scale = Interval.from_rational(2 ** (n + 2)) * PI_IV.pow_int(n + 1).reciprocal()
    magnitude = scale * Interval.from_rational(factorial(n)) * dirichlet_beta_odd_enclosure(n + 1)
    return magnitude if m % 2 == 0 else -magnitude  # sign (-1)^m


def bell_dobinski_enclosure(
    n: int, *, rel_tol: float = 1e-18, max_terms: int = 100_000
) -> Interval:
    r"""Certified enclosure of the Bell number ``B_n`` via Dobinski's formula (``n >= 1``).

    ``B_n = e^{-1} sum_{k>=0} k^n / k!``. The terms ``t_k = k^n/k!`` rise then fall with a
    ratio ``t_{k+1}/t_k = (1 + 1/k)^n / (k+1)`` that decreases monotonically in ``k``; once
    it drops below ``1`` (past the peak) the omitted tail obeys the geometric bound
    ``sum_{j>k} t_j <= t_k q / (1-q)`` with ``q = ratio(k) < 1`` (since later ratios are
    smaller). Summation continues until that certified tail is below ``rel_tol`` times the
    partial sum, so the enclosure is float-tight. The exact rational partial sum plus the
    rigorous tail, divided by the certified ``e`` (``E_IV``), provably contains ``B_n`` -- a
    genuine (non-asymptotic) error bar for the Bell numbers, complementing the saddle point.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if rel_tol <= 0.0:
        raise ValueError(f"rel_tol must be > 0, got {rel_tol}")
    tol = Fraction(rel_tol).limit_denominator(10**18)
    partial = Fraction(0)
    tail_hi = Fraction(0)
    k = 1
    while True:
        t_k = Fraction(k**n, factorial(k))
        partial += t_k
        ratio = Fraction((k + 1) ** n, k**n * (k + 1))  # t_{k+1}/t_k, decreasing in k
        if ratio < 1:
            # sum_{j>k} t_j <= t_k q/(1-q); tight once the peak is well behind us.
            tail_hi = t_k * ratio / (1 - ratio)
            if tail_hi <= partial * tol:
                break
        k += 1
        if k > max_terms:  # pragma: no cover - safety valve, never hit for sane n
            raise RuntimeError(f"Dobinski tail did not converge within {max_terms} terms")
    total = Interval(
        Interval.from_rational(partial).lo,
        Interval.from_rational(partial + tail_hi).hi,
    )
    return total * E_IV.reciprocal()  # multiply by e^{-1}


def bell_asymptotic_relative_error(n: int) -> float:
    r"""Relative error of the refined Bell saddle point vs the exact ``B_n`` (numerical).

    Computed in **log space** as ``|expm1(log_approx - log_exact)| = |approx/exact - 1|``,
    using :func:`math.log` on the exact big-integer :func:`~omnibias.difference._core.stirling.bell_number`
    (Python's ``log`` accepts arbitrary-precision ints). This avoids the ``float`` overflow
    that the value-space form :func:`bell_number_asymptotic_refined` hits past ``n ~ 150``,
    so the probe is valid for arbitrarily large ``n``. It is a *calibration* probe for
    :func:`recommended_bell_fallback_n`, not a hot path.

    Empirically the error decays like ``~n^{-3/2}`` in the large-``n`` regime but is
    **non-monotone** at small ``n``: it dips to ``~4e-5`` at ``n = 5`` (a near-cancellation
    fluke), rises back to a ``~2.8e-4`` local max near ``n = 10``, then decreases -- which is
    exactly why :func:`recommended_bell_fallback_n` uses a last-exceedance (not first-crossing)
    rule.
    """
    from omnibias.difference._core.stirling import bell_number

    log_exact = log(bell_number(n))
    log_approx = log_bell_number_asymptotic_refined(n)
    return abs(expm1(log_approx - log_exact))


def recommended_bell_fallback_n(rel_tol: float, *, n_max: int = 400) -> int:
    r"""Smallest ``n`` beyond which the refined saddle point stays within ``rel_tol``.

    Because :func:`bell_asymptotic_relative_error` is **non-monotone** at small ``n`` (it
    dips below ``~1e-4`` at ``n = 5`` then rises again through ``n ~ 10``), a naive
    first-crossing search is *unsound* -- it can return an ``n`` whose immediate successors
    exceed ``rel_tol``. Instead this scans the whole window ``2..n_max`` and returns
    ``last_exceed + 1``: the smallest ``n`` such that **every** measured ``m >= n`` in the
    window satisfies ``err(m) <= rel_tol``. That is the data-driven cutoff beyond which the
    fast asymptotic may replace the slow exact Bell computation.

    Raises ``ValueError`` if the tolerance is still violated at ``n_max`` (the window does
    not certify the cutoff -- widen ``n_max``); returns ``2`` if the tolerance already holds
    across the entire window.
    """
    if rel_tol <= 0.0:
        raise ValueError(f"rel_tol must be > 0, got {rel_tol}")
    errors = [(n, bell_asymptotic_relative_error(n)) for n in range(2, n_max + 1)]
    if errors[-1][1] > rel_tol:
        raise ValueError(
            f"refined Bell asymptotic still exceeds rel_tol {rel_tol} at n_max={n_max} "
            f"(err={errors[-1][1]:.3e}); widen n_max"
        )
    last_exceed = max((n for n, e in errors if e > rel_tol), default=None)
    return 2 if last_exceed is None else last_exceed + 1


__all__ = [
    "TransferEstimate",
    "bell_asymptotic_relative_error",
    "bell_dobinski_enclosure",
    "bell_number_asymptotic_refined",
    "bernoulli_enclosure",
    "catalan_asymptotic",
    "cauchy_product",
    "dirichlet_beta_odd_enclosure",
    "dominant_pole_coefficient_asymptotic",
    "euler_enclosure",
    "exponential_generating_coeffs",
    "log_bell_number_asymptotic_refined",
    "ordinary_from_exponential",
    "rational_ogf_coefficients",
    "rational_ogf_growth_base",
    "recommended_bell_fallback_n",
    "singular_template_coefficient",
    "transfer_theorem",
    "zeta_int_enclosure",
]
