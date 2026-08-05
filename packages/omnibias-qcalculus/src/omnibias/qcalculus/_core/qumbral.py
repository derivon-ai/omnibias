# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""q-umbral calculus: q-transforms and the full q-Sheffer sequence layer (exact).

The q-deformation of :mod:`omnibias.difference.umbral`. Two layers, both exact
:class:`~fractions.Fraction` arithmetic at a numeric ``q``:

**Mirror layer** -- the q-analogs of the classical umbral transforms:

* :func:`q_binomial_transform` / :func:`q_inverse_binomial_transform` (Gaussian binomial,
  q-binomial inversion with the ``q^{C(n-k,2)}`` sign);
* :func:`q_monomial_to_falling` / :func:`q_falling_to_monomial` (the q-Stirling transforms);
* :func:`q_appell_sequence` (``p_n(x) = sum_k [n,k]_q a_{n-k} x^k``, satisfying the
  q-Appell property ``D_q p_n = [n]_q p_{n-1}``);
* :func:`q_newton_forward_coeffs` / :func:`q_newton_forward_value` (Gregory-Newton
  interpolation on the q-integer nodes ``[j]_q`` in the ``[x]_{k,q}/[k]_q!`` basis);
* :func:`q_sheffer_classify` (structural, q-independent).

**Full q-Sheffer layer** -- generated from the q-exponential generating function
``(1/g(fbar(t))) e_q(x fbar(t))`` with ``e_q(u) = sum_m u^m/[m]_q!``:

* :func:`q_sheffer_sequence` / :func:`q_associated_sequence`;
* :func:`q_delta_operator_apply` (the q-delta operator ``Q = f(D_q)``);
* :func:`q_pincherle_derivative`, :func:`q_umbral_composition`.

These carry the **exact** q-Sheffer recurrence ``Q s_n = [n]_q s_{n-1}`` (with ``Q = f(D_q)``;
proved from ``D_q e_q(lambda x) = lambda e_q(lambda x)`` and ``f(fbar(t)) = t``), so the layer
is closed-form, not merely ``q -> 1``-validated. Honesty note: the q-Pincherle derivative is
the *series-level* q-derivative of the indicator ``f``; because ``[D_q, X] = M_q`` (the
q-dilation) rather than the identity, the classical operator commutator is q-deformed -- the
function returns the series object and is validated by its ``q -> 1`` limit.

Everything reduces to :mod:`omnibias.difference.umbral` as ``q -> 1``. This is the
**distinct** ``q -> 1`` limit, never the ``delta -> 0`` founding bias collapse nor
``beta -> inf`` **temperature collapse**; same "collapse" word, different limits, never conflated.
The ordinary formal-power-series composition/inversion is q-independent and is reused from
:mod:`omnibias.difference` (the q enters only through the ``[n]_q!`` normalisation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb

from omnibias.difference import (
    compose_series,
    compositional_inverse,
    series_reciprocal,
    umbral_composition,
)
from omnibias.qcalculus._core.qderiv import q_derivative_poly
from omnibias.qcalculus._core.qnumbers import q_binomial, q_bracket, q_factorial
from omnibias.qcalculus._core.qstirling import (
    q_stirling_first_signed,
    q_stirling_second,
)

Rational = Fraction | int


def _frac(v: Rational) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def _trim(coeffs: list[Fraction]) -> tuple[Fraction, ...]:
    out = list(coeffs)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def _series_powers(series: Sequence[Rational], order: int) -> list[list[Fraction]]:
    """Powers ``series^0 .. series^order`` as ``t``-coefficient lists (delta series)."""
    base = [Fraction(0)] * (order + 1)
    for i, c in enumerate(series):
        if i > order:
            break
        base[i] = _frac(c)
    powers = [[Fraction(0)] * (order + 1) for _ in range(order + 1)]
    powers[0][0] = Fraction(1)
    for m in range(1, order + 1):
        prev, cur = powers[m - 1], powers[m]
        for i in range(order + 1):
            if prev[i] == 0:
                continue
            for k in range(order + 1 - i):
                if base[k]:
                    cur[i + k] += prev[i] * base[k]
    return powers


# --------------------------------------------------------------------------- #
# Mirror layer                                                                 #
# --------------------------------------------------------------------------- #
def q_binomial_transform(seq: Sequence[Rational], q: Rational) -> tuple[Fraction, ...]:
    r"""The q-binomial transform ``b_n = sum_k [n,k]_q a_k`` (exact)."""
    a = [_frac(v) for v in seq]
    return tuple(
        sum((q_binomial(n, k, q) * a[k] for k in range(n + 1)), Fraction(0))
        for n in range(len(a))
    )


def q_inverse_binomial_transform(
    seq: Sequence[Rational], q: Rational
) -> tuple[Fraction, ...]:
    r"""The inverse q-binomial transform ``a_n = sum_k (-1)^{n-k} q^{C(n-k,2)} [n,k]_q b_k`` (exact)."""
    b = [_frac(v) for v in seq]
    qq = _frac(q)
    out: list[Fraction] = []
    for n in range(len(b)):
        acc = Fraction(0)
        for k in range(n + 1):
            sign = Fraction((-1) ** (n - k))
            acc += sign * qq ** comb(n - k, 2) * q_binomial(n, k, q) * b[k]
        out.append(acc)
    return tuple(out)


def q_monomial_to_falling(
    coeffs: Sequence[Rational], q: Rational
) -> tuple[Fraction, ...]:
    r"""q-Stirling transform: coefficients of a polynomial in the q-falling-factorial basis.

    ``a_k = sum_j c_j S_q(j, k)`` where ``c`` are the monomial coefficients; the inverse of
    :func:`q_falling_to_monomial`. At ``q -> 1`` this is the classical
    :func:`omnibias.difference.monomial_to_falling`.
    """
    c = [_frac(v) for v in coeffs]
    n = len(c)
    return tuple(
        sum((c[j] * q_stirling_second(j, k, q) for j in range(n)), Fraction(0))
        for k in range(n)
    )


def q_falling_to_monomial(
    coeffs: Sequence[Rational], q: Rational
) -> tuple[Fraction, ...]:
    r"""Inverse q-Stirling transform: q-falling-factorial coordinates back to monomials.

    ``c_j = sum_k a_k s_q(k, j)`` (signed q-Stirling first kind); the inverse of
    :func:`q_monomial_to_falling`. At ``q -> 1`` this is the classical
    :func:`omnibias.difference.falling_to_monomial`.
    """
    a = [_frac(v) for v in coeffs]
    n = len(a)
    return tuple(
        sum((a[k] * q_stirling_first_signed(k, j, q) for k in range(n)), Fraction(0))
        for j in range(n)
    )


def q_appell_sequence(
    constants: Sequence[Rational], q: Rational
) -> list[tuple[Fraction, ...]]:
    r"""The q-Appell sequence ``p_n(x) = sum_k [n,k]_q a_{n-k} x^k`` (exact).

    ``constants`` are ``a_j = p_j(0)``. Satisfies the q-Appell property
    ``D_q p_n = [n]_q p_{n-1}`` (the ``f = t`` q-Sheffer case). At ``q -> 1`` this is the
    classical :func:`omnibias.difference.appell_sequence`.
    """
    a = [_frac(v) for v in constants]
    return [
        tuple(q_binomial(n, k, q) * a[n - k] for k in range(n + 1)) for n in range(len(a))
    ]


def _phi(k: int, x: Fraction, q: Fraction) -> Fraction:
    r"""The q-Newton basis polynomial ``phi_k(x) = [x]_{k,q}/[k]_q!`` evaluated at ``x``."""
    num = Fraction(1)
    for i in range(k):
        num *= x - q_bracket(i, q)
    return num / q_factorial(k, q)


def q_newton_forward_coeffs(
    samples: Sequence[Rational], q: Rational
) -> tuple[Fraction, ...]:
    r"""q-Gregory-Newton forward coefficients from samples at the q-integer nodes ``[j]_q``.

    ``samples[j]`` are ``f([0]_q), f([1]_q), ...``; returns the coefficients ``b_k`` of the
    interpolant ``f(x) = sum_k b_k [x]_{k,q}/[k]_q!`` (a triangular solve on the nodes, exact).
    At ``q -> 1`` the nodes become ``0, 1, 2, ...`` and ``b_k -> Delta^k f(0)``, recovering the
    classical :func:`omnibias.difference.newton_forward_coeffs`.
    """
    y = [_frac(v) for v in samples]
    qq = _frac(q)
    b: list[Fraction] = []
    for k in range(len(y)):
        node = q_bracket(k, qq)
        acc = sum((b[j] * _phi(j, node, qq) for j in range(k)), Fraction(0))
        b.append((y[k] - acc) / _phi(k, node, qq))
    return tuple(b)


def q_newton_forward_value(
    coeffs: Sequence[Rational], x: Rational, q: Rational
) -> Fraction:
    r"""Evaluate the q-Newton interpolation ``sum_k coeffs[k] [x]_{k,q}/[k]_q!`` at ``x`` (exact)."""
    qq = _frac(q)
    xf = _frac(x)
    return sum(
        (_frac(c) * _phi(k, xf, qq) for k, c in enumerate(coeffs)), Fraction(0)
    )


@dataclass(frozen=True)
class QShefferClass:
    """Classification of a q-Sheffer sequence from its pair ``(g, f)`` (structural)."""

    kind: str  # "appell" | "associated" | "sheffer"
    is_appell: bool
    is_associated: bool


def _is_identity_delta(coeffs: Sequence[Fraction]) -> bool:
    return (
        len(coeffs) >= 2
        and coeffs[0] == 0
        and coeffs[1] == 1
        and all(c == 0 for c in coeffs[2:])
    )


def _is_unit_constant(coeffs: Sequence[Fraction]) -> bool:
    return len(coeffs) >= 1 and coeffs[0] == 1 and all(c == 0 for c in coeffs[1:])


def q_sheffer_classify(
    g_coeffs: Sequence[Rational], f_coeffs: Sequence[Rational]
) -> QShefferClass:
    r"""Classify the q-Sheffer pair ``(g, f)`` (appell / associated / sheffer).

    ``g`` must be invertible (``g(0) != 0``) and ``f`` a delta series (``f(0) = 0``,
    ``f'(0) != 0``). The classification is *structural* -- identical to the classical
    :func:`omnibias.difference.sheffer_classify`, since it does not depend on ``q``.
    """
    g = [_frac(v) for v in g_coeffs]
    f = [_frac(v) for v in f_coeffs]
    if not g or g[0] == 0:
        raise ValueError("g must be invertible: g(0) != 0")
    if len(f) < 2 or f[0] != 0 or f[1] == 0:
        raise ValueError("f must be a delta series: f(0) = 0 and f'(0) != 0")
    is_appell = _is_identity_delta(f)
    is_associated = _is_unit_constant(g)
    kind = "appell" if is_appell else "associated" if is_associated else "sheffer"
    return QShefferClass(kind=kind, is_appell=is_appell, is_associated=is_associated)


# --------------------------------------------------------------------------- #
# Full q-Sheffer layer                                                         #
# --------------------------------------------------------------------------- #
def q_sheffer_sequence(
    g_coeffs: Sequence[Rational], f_coeffs: Sequence[Rational], n: int, q: Rational
) -> list[tuple[Fraction, ...]]:
    r"""The q-Sheffer sequence ``s_0(x) .. s_n(x)`` for the pair ``(g, f)`` (exact).

    Read off the q-exponential generating function

    .. math::

        \sum_{n\ge 0} s_n(x)\,\frac{t^n}{[n]_q!}
            = \frac{1}{g(\bar f(t))}\, e_q\!\big(x\,\bar f(t)\big),
        \qquad e_q(u) = \sum_m \frac{u^m}{[m]_q!},

    with ``fbar`` the (ordinary) compositional inverse of the delta series ``f``. Carries the
    exact q-Sheffer recurrence ``Q s_n = [n]_q s_{n-1}`` with ``Q = f(D_q)``
    (:func:`q_delta_operator_apply`). At ``q -> 1`` this is the classical
    :func:`omnibias.difference.sheffer_sequence`.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    q_sheffer_classify(g_coeffs, f_coeffs)  # validates the (g, f) pair
    qq = _frac(q)
    fbar = compositional_inverse(f_coeffs, n)
    a_series = series_reciprocal(compose_series(g_coeffs, fbar, n), n)  # 1 / g(fbar)
    a_coeffs = [a_series[j] if j < len(a_series) else Fraction(0) for j in range(n + 1)]
    fbar_pow = _series_powers(fbar, n)
    result: list[tuple[Fraction, ...]] = []
    for degree in range(n + 1):
        poly = [Fraction(0)] * (degree + 1)
        for m in range(degree + 1):
            acc = Fraction(0)
            for j in range(degree - m + 1):
                aj = a_coeffs[j]
                if aj:
                    acc += aj * fbar_pow[m][degree - j]
            poly[m] = acc / q_factorial(m, qq)
        fact = q_factorial(degree, qq)
        result.append(tuple(fact * c for c in poly))
    return result


def q_associated_sequence(
    f_coeffs: Sequence[Rational], n: int, q: Rational
) -> list[tuple[Fraction, ...]]:
    r"""The q-associated (binomial-type) sequence of the delta series ``f`` (``g = 1``, exact).

    ``f = t`` gives the monomials ``x^n``; general ``f`` gives the q-analog of the associated
    sequence, satisfying ``f(D_q) p_n = [n]_q p_{n-1}``.
    """
    return q_sheffer_sequence((1,), f_coeffs, n, q)


def q_delta_operator_apply(
    f_coeffs: Sequence[Rational], coeffs: Sequence[Rational], q: Rational
) -> tuple[Fraction, ...]:
    r"""Apply the q-delta operator ``Q = f(D_q) = sum_k f_k D_q^k`` to a polynomial (exact).

    ``D_q`` is the Jackson q-derivative (:func:`omnibias.qcalculus.q_derivative_poly`). With
    ``f = t`` this is ``D_q`` itself; for the q-associated sequence ``p_n`` of ``f`` it realises
    the exact recurrence ``Q p_n = [n]_q p_{n-1}``.
    """
    f = [_frac(v) for v in f_coeffs]
    qq = _frac(q)
    result = [Fraction(0)] * len(coeffs)
    deriv = [_frac(v) for v in coeffs]  # D_q^0 p
    for fk in f:
        if fk and deriv:
            for j, c in enumerate(deriv):
                result[j] += fk * c
        deriv = list(q_derivative_poly(deriv, qq))
    return _trim(result)


def q_pincherle_derivative(
    f_coeffs: Sequence[Rational], q: Rational
) -> tuple[Fraction, ...]:
    r"""The (series-level) q-Pincherle derivative of ``Q = f(D_q)``: the q-derivative of ``f``.

    Returns ``D_q f`` (as a coefficient series). This is the q-analog of the Pincherle
    derivative; it reduces to the classical ``f'`` as ``q -> 1``. Because ``[D_q, X] = M_q``
    (the q-dilation) the classical operator commutator identity is q-deformed, so this is the
    honest series-level object rather than a claimed ``QX - XQ`` operator equality.
    """
    return tuple(q_derivative_poly([_frac(v) for v in f_coeffs], _frac(q)))


def q_umbral_composition(
    s_seq: Sequence[Sequence[Rational]], r_seq: Sequence[Sequence[Rational]]
) -> list[tuple[Fraction, ...]]:
    r"""Umbral composition ``(s # r)_n(x) = sum_k s_{n,k} r_k(x)`` (exact, q-independent).

    The polynomial-level umbral composition does not depend on ``q``, so it delegates to the
    classical :func:`omnibias.difference.umbral_composition`.
    """
    return umbral_composition(s_seq, r_seq)


__all__ = [
    "QShefferClass",
    "q_appell_sequence",
    "q_associated_sequence",
    "q_binomial_transform",
    "q_delta_operator_apply",
    "q_falling_to_monomial",
    "q_inverse_binomial_transform",
    "q_monomial_to_falling",
    "q_newton_forward_coeffs",
    "q_newton_forward_value",
    "q_pincherle_derivative",
    "q_sheffer_classify",
    "q_sheffer_sequence",
    "q_umbral_composition",
]
