# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Umbral / Sheffer finite-difference calculus (exact rational arithmetic).

The discrete counterpart of the derivative tower: the forward-difference operator
``Delta`` (the ``delta = 1`` finite difference), Newton's forward-difference
interpolation ``f(x) = sum_k Delta^k f(0) C(x, k)``, the binomial transform, the
monomial <-> falling-factorial change of basis (the Stirling transforms), and
Appell / Sheffer sequences.

Everything runs in exact :class:`~fractions.Fraction` arithmetic so the umbral
identities hold *exactly*, not up to rounding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial

from omnibias.difference._core.stirling import stirling_first_signed, stirling_second

Rational = Fraction | int


def _frac(v: Rational) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def forward_difference(values: Sequence[Rational], order: int = 1) -> tuple[Fraction, ...]:
    r"""Apply the forward-difference operator ``Delta f(i) = f(i+1) - f(i)`` ``order`` times.

    Returns a tuple shorter by ``order`` (``Delta^order`` of the value list).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order > len(values):
        raise ValueError(f"order {order} exceeds the number of values {len(values)}")
    seq = [_frac(v) for v in values]
    for _ in range(order):
        seq = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    return tuple(seq)


def newton_forward_coeffs(values: Sequence[Rational]) -> tuple[Fraction, ...]:
    r"""Leading finite differences ``Delta^k f(0)`` for ``k = 0 .. len-1``.

    ``Delta^k f(0) = sum_{j=0}^{k} (-1)^{k-j} C(k, j) f(j)`` -- the coefficients of
    Newton's forward-difference interpolation of the samples ``f(0), f(1), ...``.
    """
    vals = [_frac(v) for v in values]
    n = len(vals)
    return tuple(
        sum((Fraction((-1) ** (k - j) * comb(k, j)) * vals[j] for j in range(k + 1)), Fraction(0))
        for k in range(n)
    )


def binomial_coefficient(x: Rational, k: int) -> Fraction:
    r"""Generalised binomial ``C(x, k) = (x)_k / k!`` for a rational ``x``."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    xf = _frac(x)
    num = Fraction(1)
    for j in range(k):
        num *= xf - j
    return num / factorial(k)


def newton_forward_value(coeffs: Sequence[Rational], x: Rational) -> Fraction:
    r"""Evaluate Newton's forward interpolation ``sum_k coeffs[k] C(x, k)`` at ``x``.

    With ``coeffs = newton_forward_coeffs(f-samples)`` this reproduces the unique
    interpolating polynomial exactly at every integer node (and everywhere, if the
    samples came from a polynomial of degree ``< len(coeffs)``).
    """
    return sum(
        (_frac(c) * binomial_coefficient(x, k) for k, c in enumerate(coeffs)), Fraction(0)
    )


def binomial_transform(seq: Sequence[Rational]) -> tuple[Fraction, ...]:
    r"""Binomial transform ``b_n = sum_k C(n, k) a_k``."""
    a = [_frac(v) for v in seq]
    return tuple(
        sum((Fraction(comb(n, k)) * a[k] for k in range(n + 1)), Fraction(0))
        for n in range(len(a))
    )


def inverse_binomial_transform(seq: Sequence[Rational]) -> tuple[Fraction, ...]:
    r"""Inverse binomial transform ``a_n = sum_k (-1)^{n-k} C(n, k) b_k``."""
    b = [_frac(v) for v in seq]
    return tuple(
        sum(
            (Fraction((-1) ** (n - k) * comb(n, k)) * b[k] for k in range(n + 1)),
            Fraction(0),
        )
        for n in range(len(b))
    )


def monomial_to_falling(coeffs: Sequence[Rational]) -> tuple[Fraction, ...]:
    r"""Rewrite ``sum_j c_j x^j`` in the falling-factorial basis ``sum_k a_k (x)_k``.

    Uses ``x^j = sum_k S(j, k) (x)_k`` (Stirling second kind), so
    ``a_k = sum_j c_j S(j, k)``.
    """
    c = [_frac(v) for v in coeffs]
    d = len(c)
    return tuple(
        sum((c[j] * stirling_second(j, k) for j in range(k, d)), Fraction(0)) for k in range(d)
    )


def falling_to_monomial(coeffs: Sequence[Rational]) -> tuple[Fraction, ...]:
    r"""Rewrite ``sum_k a_k (x)_k`` in the monomial basis ``sum_j c_j x^j``.

    Uses ``(x)_k = sum_j s(k, j) x^j`` (signed Stirling first kind), so
    ``c_j = sum_k a_k s(k, j)``. Exact inverse of :func:`monomial_to_falling`.
    """
    a = [_frac(v) for v in coeffs]
    d = len(a)
    return tuple(
        sum((a[k] * stirling_first_signed(k, j) for k in range(j, d)), Fraction(0))
        for j in range(d)
    )


def appell_sequence(constants: Sequence[Rational]) -> list[tuple[Fraction, ...]]:
    r"""Appell sequence ``p_n(x) = sum_k C(n, k) a_{n-k} x^k`` from constants ``a_j = p_j(0)``.

    An Appell sequence satisfies ``p_n'(x) = n p_{n-1}(x)``; it is the ``h -> 0``
    (Sheffer) shift-invariant family whose evaluation functional is fixed by the
    moments ``a_j``. Bernoulli polynomials are the Appell sequence of the Bernoulli
    numbers; ``a = (1, 0, 0, ...)`` gives the monomials ``x^n``.
    """
    a = [_frac(v) for v in constants]
    return [
        tuple(Fraction(comb(n, k)) * a[n - k] for k in range(n + 1)) for n in range(len(a))
    ]


# --------------------------------------------------------------------------- #
# Formal power-series composition / inversion (exact, Riordan substrate)      #
# --------------------------------------------------------------------------- #
def compose_series(
    outer: Sequence[Rational], inner: Sequence[Rational], order: int
) -> tuple[Fraction, ...]:
    r"""Coefficients ``[t^0..t^order]`` of ``outer(inner(t))`` with ``inner(0) = 0`` (exact).

    Horner in the composition algebra: requires a *delta series* ``inner`` (zero
    constant term) so the composition is a well-defined formal power series.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    f = [_frac(v) for v in outer]
    g = [_frac(v) for v in inner]
    if g and g[0] != 0:
        raise ValueError("inner series must have zero constant term (inner(0) = 0)")
    result = [Fraction(0)] * (order + 1)
    if f:
        result[0] = f[0]
    gpow = [Fraction(0)] * (order + 1)
    gpow[0] = Fraction(1)  # inner^0 = 1
    for k in range(1, len(f)):
        nextpow = [Fraction(0)] * (order + 1)
        for i in range(order + 1):
            if gpow[i] == 0:
                continue
            for j in range(1, len(g)):
                if i + j > order:
                    break
                nextpow[i + j] += gpow[i] * g[j]
        gpow = nextpow
        for i in range(order + 1):
            if gpow[i] != 0:
                result[i] += f[k] * gpow[i]
    return tuple(result)


def compositional_inverse(series: Sequence[Rational], order: int) -> tuple[Fraction, ...]:
    r"""Compositional inverse ``sbar`` with ``series(sbar(t)) = t`` (exact, ``series(0)=0``).

    Requires a delta series with ``series[1] != 0``. Solves order by order using
    :func:`compose_series`; e.g. the inverse of ``t/(1-t)`` is ``t/(1+t)``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    h = [_frac(v) for v in series]
    if len(h) < 2 or h[0] != 0 or h[1] == 0:
        raise ValueError("series must be a delta series: series(0) = 0 and series'(0) != 0")
    hbar = [Fraction(0)] * (order + 1)
    if order >= 1:
        hbar[1] = Fraction(1) / h[1]
    for k in range(2, order + 1):
        current = compose_series(h, hbar, k)
        hbar[k] = -current[k] / h[1]
    return tuple(hbar)


def series_reciprocal(series: Sequence[Rational], order: int) -> tuple[Fraction, ...]:
    r"""Coefficients ``[t^0..t^order]`` of ``1 / series`` with ``series(0) != 0`` (exact)."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    a = [_frac(v) for v in series]
    if not a or a[0] == 0:
        raise ValueError("series must have a non-zero constant term")
    b = [Fraction(0)] * (order + 1)
    b[0] = Fraction(1) / a[0]
    for k in range(1, order + 1):
        acc = Fraction(0)
        for j in range(1, min(k, len(a) - 1) + 1):
            acc += a[j] * b[k - j]
        b[k] = -acc / a[0]
    return tuple(b)


# --------------------------------------------------------------------------- #
# Sheffer classification                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShefferClass:
    """Classification of a Sheffer sequence from its defining pair ``(g, f)``."""

    kind: str  # "appell" | "associated" | "sheffer"
    is_appell: bool
    is_associated: bool


def _is_identity_delta(coeffs: Sequence[Fraction]) -> bool:
    """Whether ``coeffs`` is the series ``t`` (``0, 1, 0, 0, ...``)."""
    return (
        len(coeffs) >= 2
        and coeffs[0] == 0
        and coeffs[1] == 1
        and all(c == 0 for c in coeffs[2:])
    )


def _is_unit_constant(coeffs: Sequence[Fraction]) -> bool:
    """Whether ``coeffs`` is the series ``1`` (``1, 0, 0, ...``)."""
    return len(coeffs) >= 1 and coeffs[0] == 1 and all(c == 0 for c in coeffs[1:])


def sheffer_classify(
    g_coeffs: Sequence[Rational], f_coeffs: Sequence[Rational]
) -> ShefferClass:
    r"""Classify the Sheffer sequence for the pair ``(g, f)`` (exact).

    A Sheffer sequence is defined by an invertible series ``g`` (``g(0) != 0``) and a
    delta series ``f`` (``f(0) = 0``, ``f'(0) != 0``). This classifies it as:

    * **appell** -- ``f(t) = t`` (the sequence satisfies ``p_n'(x) = n p_{n-1}(x)``;
      Bernoulli / Euler / Hermite polynomials),
    * **associated** (binomial type) -- ``g(t) = 1`` (``p_n(x+y) = sum_k C(n,k) p_k(x)
      p_{n-k}(y)``; falling factorials, Abel / Touchard polynomials),
    * **sheffer** -- the general case (e.g. Laguerre, actuarial polynomials).

    Raises if ``(g, f)`` is not a valid Sheffer pair.
    """
    g = [_frac(v) for v in g_coeffs]
    f = [_frac(v) for v in f_coeffs]
    if not g or g[0] == 0:
        raise ValueError("g must be invertible: g(0) != 0")
    if len(f) < 2 or f[0] != 0 or f[1] == 0:
        raise ValueError("f must be a delta series: f(0) = 0 and f'(0) != 0")
    is_appell = _is_identity_delta(f)
    is_associated = _is_unit_constant(g)
    if is_appell:
        kind = "appell"
    elif is_associated:
        kind = "associated"
    else:
        kind = "sheffer"
    return ShefferClass(kind=kind, is_appell=is_appell, is_associated=is_associated)


# --------------------------------------------------------------------------- #
# Riordan arrays (group under the Fundamental Theorem of Riordan Arrays)       #
# --------------------------------------------------------------------------- #
def riordan_array(
    d_coeffs: Sequence[Rational], h_coeffs: Sequence[Rational], size: int
) -> tuple[tuple[Fraction, ...], ...]:
    r"""Lower-triangular Riordan array ``T[n, k] = [t^n] d(t) h(t)^k`` (exact, ``size x size``).

    A proper Riordan array needs ``d(0) != 0`` and ``h(0) = 0``, ``h'(0) != 0`` (so it is
    lower-triangular with a non-zero diagonal). ``(d, h) = (1/(1-t), t/(1-t))`` gives
    Pascal's triangle; ``(1, t/(1-t))`` gives the binomial ``C(n-1, k-1)`` array.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    d = [_frac(v) for v in d_coeffs]
    h = [_frac(v) for v in h_coeffs]
    if not d or d[0] == 0:
        raise ValueError("Riordan array needs d(0) != 0")
    if len(h) < 2 or h[0] != 0 or h[1] == 0:
        raise ValueError("Riordan array needs h(0) = 0 and h'(0) != 0")

    def _mul(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        out = [Fraction(0)] * size
        for i in range(min(len(a), size)):
            if a[i] == 0:
                continue
            for j in range(min(len(b), size - i)):
                out[i + j] += a[i] * b[j]
        return out

    dd = ([*d] + [Fraction(0)] * size)[:size]
    hh = ([*h] + [Fraction(0)] * size)[:size]
    col = [Fraction(0)] * size
    col[0] = Fraction(1)  # h^0
    matrix = [[Fraction(0)] * size for _ in range(size)]
    for k in range(size):
        dcol = _mul(dd, col)  # d * h^k
        for n in range(size):
            matrix[n][k] = dcol[n]
        col = _mul(col, hh)
    return tuple(tuple(row) for row in matrix)


def riordan_product(
    left: tuple[Sequence[Rational], Sequence[Rational]],
    right: tuple[Sequence[Rational], Sequence[Rational]],
    order: int,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    r"""Riordan group product ``(d1, h1) * (d2, h2) = (d1 * (d2 o h1), h2 o h1)`` (exact).

    The Fundamental Theorem of Riordan Arrays as a group law on the defining series
    (matrix multiplication of the two arrays), truncated to ``order + 1`` terms.
    """
    d1, h1 = ([_frac(v) for v in left[0]], [_frac(v) for v in left[1]])
    d2, h2 = ([_frac(v) for v in right[0]], [_frac(v) for v in right[1]])
    d2_on_h1 = compose_series(d2, h1, order)
    prod = [Fraction(0)] * (order + 1)
    for i in range(min(len(d1), order + 1)):
        if d1[i] == 0:
            continue
        for j in range(order + 1 - i):
            prod[i + j] += d1[i] * d2_on_h1[j]
    d_out = tuple(prod)
    h_out = compose_series(h2, h1, order)
    return d_out, h_out


def riordan_inverse(
    d_coeffs: Sequence[Rational], h_coeffs: Sequence[Rational], order: int
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    r"""Riordan group inverse ``(d, h)^{-1} = (1 / (d o hbar), hbar)`` (exact).

    ``hbar`` is the compositional inverse of ``h``; the product of ``(d, h)`` with its
    inverse is the identity Riordan array ``(1, t)``.
    """
    hbar = compositional_inverse(h_coeffs, order)
    d_on_hbar = compose_series(d_coeffs, hbar, order)
    d_inv = series_reciprocal(d_on_hbar, order)
    return d_inv, hbar


# --------------------------------------------------------------------------- #
# Connection constants                                                         #
# --------------------------------------------------------------------------- #
def connection_constants(
    source: Sequence[Sequence[Rational]], target: Sequence[Sequence[Rational]]
) -> list[tuple[Fraction, ...]]:
    r"""Connection constants ``c[n][k]`` with ``source_n = sum_k c[n][k] target_k`` (exact).

    Both bases are graded polynomial sequences given as ascending coefficient lists with
    ``deg target_k = k`` (a triangular basis, e.g. monomials or falling factorials), so the
    change of basis is an exact back-substitution. E.g. with ``target`` the falling
    factorials this recovers the Stirling-second-kind connection ``x^n = sum_k S(n,k)(x)_k``.
    """
    src = [[_frac(v) for v in poly] for poly in source]
    tgt = [[_frac(v) for v in poly] for poly in target]
    for k, poly in enumerate(tgt):
        lead = poly[k] if k < len(poly) else Fraction(0)
        if lead == 0:
            raise ValueError(f"target[{k}] must have non-zero degree-{k} coefficient (graded basis)")

    def coeff(poly: list[Fraction], i: int) -> Fraction:
        return poly[i] if i < len(poly) else Fraction(0)

    out: list[tuple[Fraction, ...]] = []
    for poly in src:
        degree = len(poly) - 1
        residual = [coeff(poly, i) for i in range(degree + 1)]
        row = [Fraction(0)] * (degree + 1)
        for k in range(degree, -1, -1):
            if k >= len(tgt):
                raise ValueError("target basis is too short for the source degree")
            ck = residual[k] / coeff(tgt[k], k)
            row[k] = ck
            if ck != 0:
                for i in range(k + 1):
                    residual[i] -= ck * coeff(tgt[k], i)
        out.append(tuple(row))
    return out


__all__ = [
    "ShefferClass",
    "appell_sequence",
    "binomial_coefficient",
    "binomial_transform",
    "compose_series",
    "compositional_inverse",
    "connection_constants",
    "falling_to_monomial",
    "forward_difference",
    "inverse_binomial_transform",
    "monomial_to_falling",
    "newton_forward_coeffs",
    "newton_forward_value",
    "riordan_array",
    "riordan_inverse",
    "riordan_product",
    "series_reciprocal",
    "sheffer_classify",
]
