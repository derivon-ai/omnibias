# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous quadrature with a *derived* error term.

The certified-evidence stack currently bounds an integral with a user-supplied heuristic
padding (``interval_trapezoid_bound``).  This module replaces that with the
textbook composite-trapezoid error bound

.. math::

    \int_a^b f \;-\; T_n(f) \;=\; -\,\frac{(b-a)\,h^2}{12}\, f''(\xi),
    \qquad \xi \in [a, b],\; h = (b-a)/n,

so given a rigorous enclosure ``M2 \supseteq f''([a, b])`` the integral is bracketed
*without any tunable fudge factor*.  The remainder interval is genuinely derived
from the second-derivative enclosure -- exactly the kind of term a Taylor model
(:class:`~omnibias.core.verified.taylor_model.TaylorModel`) produces.

A composite midpoint variant is included for completeness.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.verified.coeffs import bernoulli_number_exact
from omnibias.core.verified.interval import Interval, sum_intervals
from omnibias.core.verified.transcend import PI_IV, cos_iv

#: A guaranteed enclosure oracle ``f_iv(x)`` bounding ``f`` over an interval ``x``.
IntervalFn = Callable[[Interval], Interval]
#: A derivative-enclosure oracle ``deriv(k, x)`` bounding ``f^{(k)}`` over ``x``.
DerivOracle = Callable[[int, Interval], Interval]


def trapezoid_sum(node_values: Sequence[Interval], h: Interval | float) -> Interval:
    """Composite-trapezoid weighted sum ``h*(f0/2 + f1 + ... + f_{n-1} + fn/2)``."""
    n_nodes = len(node_values)
    if n_nodes < 2:
        raise ValueError("trapezoid rule needs at least two nodes")
    hi = Interval.from_value(h)
    half = Interval.from_rational(Fraction(1, 2))
    terms = [node_values[0] * half]
    terms.extend(node_values[1:-1])
    terms.append(node_values[-1] * half)
    return sum_intervals(terms) * hi


def trapezoid_integral(
    node_values: Sequence[Interval],
    a: float,
    b: float,
    second_deriv_bound: Interval,
) -> Interval:
    """Rigorous enclosure of ``\\int_a^b f`` from equispaced node enclosures.

    ``node_values`` are enclosures of ``f`` at ``n+1`` equispaced nodes spanning
    ``[a, b]``; ``second_deriv_bound`` must enclose ``f''`` over all of ``[a, b]``.
    Requires ``a <= b`` (reversed limits raise :class:`ValueError`).
    """
    n = len(node_values) - 1
    if n < 1:
        raise ValueError("need at least one panel")
    if a > b:
        raise ValueError(
            f"trapezoid_integral requires a <= b (got a={a}, b={b}); the "
            "composite rule and its derived remainder assume nodes ordered "
            "left-to-right across [a, b]. Integrate over [b, a] and negate."
        )
    span = Interval.point(b) - Interval.point(a)
    h = span * Interval.from_rational(Fraction(1, n))
    main = trapezoid_sum(node_values, h)
    # error = -(b-a) h^2 / 12 * f''(xi), xi in [a,b]
    err = -(span * h.pow_int(2) * Interval.from_rational(Fraction(1, 12))) * second_deriv_bound
    return main + err


def midpoint_integral(
    midpoint_values: Sequence[Interval],
    a: float,
    b: float,
    second_deriv_bound: Interval,
) -> Interval:
    """Rigorous composite-midpoint enclosure with the ``+(b-a)h^2/24 f''`` term.

    Requires ``a <= b`` (reversed limits raise :class:`ValueError`).
    """
    n = len(midpoint_values)
    if n < 1:
        raise ValueError("need at least one panel")
    if a > b:
        raise ValueError(
            f"midpoint_integral requires a <= b (got a={a}, b={b}); the "
            "composite rule and its derived remainder assume nodes ordered "
            "left-to-right across [a, b]. Integrate over [b, a] and negate."
        )
    span = Interval.point(b) - Interval.point(a)
    h = span * Interval.from_rational(Fraction(1, n))
    main = sum_intervals(list(midpoint_values)) * h
    err = (span * h.pow_int(2) * Interval.from_rational(Fraction(1, 24))) * second_deriv_bound
    return main + err


def _equispaced_node_intervals(a: float, b: float, panels: int) -> list[Interval]:
    """The ``panels + 1`` node enclosures ``a + k (b-a)/panels`` (rigorous)."""
    a_iv = Interval.point(a)
    h = (Interval.point(b) - a_iv) * Interval.from_rational(Fraction(1, panels))
    return [a_iv + h * Interval.from_rational(k) for k in range(panels + 1)]


def simpson_integral(
    f: IntervalFn,
    a: float,
    b: float,
    *,
    panels: int,
    fourth_deriv_bound: Interval,
) -> Interval:
    r"""Rigorous composite Simpson enclosure with the ``-(b-a)h^4/180 f''''`` remainder.

    ``panels`` must be a positive **even** integer (Simpson pairs the panels). The
    quadrature evaluates ``f`` at the ``panels + 1`` equispaced node *intervals* via
    the enclosure oracle ``f``; the fourth-order Peano remainder
    ``-(b-a) h^4 / 180 \cdot f^{(4)}(\xi)`` is bounded by ``fourth_deriv_bound`` (a
    guaranteed enclosure of ``f^{(4)}`` over ``[a, b]``). Requires ``a <= b``.
    """
    if panels < 2 or panels % 2 != 0:
        raise ValueError(f"simpson_integral needs a positive even panel count, got {panels}")
    if a > b:
        raise ValueError(f"simpson_integral requires a <= b, got a={a}, b={b}")
    nodes = _equispaced_node_intervals(a, b, panels)
    vals = [f(node) for node in nodes]
    span = Interval.point(b) - Interval.point(a)
    h = span * Interval.from_rational(Fraction(1, panels))
    weighted = vals[0] + vals[-1]
    for k in range(1, panels):
        w = 4 if k % 2 == 1 else 2
        weighted = weighted + vals[k] * Interval.from_rational(w)
    main = h * Interval.from_rational(Fraction(1, 3)) * weighted
    err = -(span * h.pow_int(4) * Interval.from_rational(Fraction(1, 180))) * fourth_deriv_bound
    return main + err


#: Outward-rounded certified Gauss-Legendre nodes/weights on ``[-1, 1]``.  Each
#: entry ``n -> [((node_lo, node_hi), (weight_lo, weight_hi)), ...]`` is a
#: high-precision (``mpmath``, 60 dps) computation of the roots of the Legendre
#: polynomial ``P_n`` and the weights ``2 / ((1 - x_i^2) P_n'(x_i)^2)``, each
#: endpoint pushed two representable steps outward so the interval provably
#: brackets the exact algebraic value.  Data-only, so unconditionally rigorous
#: with no runtime ``mpmath`` dependency.
_GL_RULES: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = {
    1: [
        ((-5e-324, 5e-324), (1.9999999999999998, 2.0000000000000004)),
    ],
    2: [
        ((-0.577350269189626, -0.5773502691896256), (0.9999999999999998, 1.0000000000000002)),
        ((0.5773502691896256, 0.577350269189626), (0.9999999999999998, 1.0000000000000002)),
    ],
    3: [
        ((-0.7745966692414835, -0.7745966692414832), (0.5555555555555554, 0.5555555555555557)),
        ((-5e-324, 5e-324), (0.8888888888888887, 0.8888888888888891)),
        ((0.7745966692414832, 0.7745966692414835), (0.5555555555555554, 0.5555555555555557)),
    ],
    4: [
        ((-0.8611363115940528, -0.8611363115940525), (0.3478548451374538, 0.34785484513745396)),
        ((-0.33998104358485637, -0.3399810435848562), (0.652145154862546, 0.6521451548625463)),
        ((0.3399810435848562, 0.33998104358485637), (0.652145154862546, 0.6521451548625463)),
        ((0.8611363115940525, 0.8611363115940528), (0.3478548451374538, 0.34785484513745396)),
    ],
    5: [
        ((-0.9061798459386642, -0.9061798459386639), (0.23692688505618906, 0.23692688505618914)),
        ((-0.5384693101056832, -0.5384693101056829), (0.47862867049936636, 0.4786286704993665)),
        ((-5e-324, 5e-324), (0.5688888888888888, 0.5688888888888891)),
        ((0.5384693101056829, 0.5384693101056832), (0.47862867049936636, 0.4786286704993665)),
        ((0.9061798459386639, 0.9061798459386642), (0.23692688505618906, 0.23692688505618914)),
    ],
    6: [
        ((-0.9324695142031522, -0.9324695142031518), (0.1713244923791703, 0.17132449237917038)),
        ((-0.6612093864662647, -0.6612093864662644), (0.36076157304813855, 0.3607615730481387)),
        ((-0.23861918608319696, -0.23861918608319688), (0.467913934572691, 0.46791393457269115)),
        ((0.23861918608319688, 0.23861918608319696), (0.467913934572691, 0.46791393457269115)),
        ((0.6612093864662644, 0.6612093864662647), (0.36076157304813855, 0.3607615730481387)),
        ((0.9324695142031518, 0.9324695142031522), (0.1713244923791703, 0.17132449237917038)),
    ],
}

#: The largest ``n`` for which a certified Gauss-Legendre rule is tabulated.
GAUSS_LEGENDRE_MAX_N: int = max(_GL_RULES)


def gauss_legendre_nodes(n: int, a: float, b: float) -> list[Interval]:
    """The ``n`` certified Gauss-Legendre node enclosures mapped to ``[a, b]``.

    Useful when the caller wants to evaluate its own oracle at the exact nodes.
    """
    if n not in _GL_RULES:
        raise ValueError(f"no tabulated Gauss-Legendre rule for n={n} (have 1..{GAUSS_LEGENDRE_MAX_N})")
    mid = (Interval.point(a) + Interval.point(b)) * Interval.point(0.5)
    half = (Interval.point(b) - Interval.point(a)) * Interval.point(0.5)
    return [mid + half * Interval(nlo, nhi) for (nlo, nhi), _ in _GL_RULES[n]]


def gauss_legendre_integral(
    f: IntervalFn,
    a: float,
    b: float,
    *,
    n: int,
    deriv_2n_bound: Interval,
) -> Interval:
    r"""Rigorous ``n``-point Gauss-Legendre enclosure with the classical remainder.

    Maps the certified ``[-1, 1]`` nodes/weights to ``[a, b]``, sums
    ``(b-a)/2 \sum_i w_i f(t_i)`` with ``f`` the enclosure oracle, and adds the
    guaranteed Gauss error term

    .. math::
        E_n = \frac{(b-a)^{2n+1} (n!)^4}{(2n+1)\,[(2n)!]^3}\, f^{(2n)}(\xi),

    bounded by ``deriv_2n_bound`` (a guaranteed enclosure of ``f^{(2n)}`` over
    ``[a, b]``).  Exact for polynomials of degree ``<= 2n-1``.  Requires ``a <= b``
    and ``1 <= n <= GAUSS_LEGENDRE_MAX_N``.
    """
    if n not in _GL_RULES:
        raise ValueError(f"no tabulated Gauss-Legendre rule for n={n} (have 1..{GAUSS_LEGENDRE_MAX_N})")
    if a > b:
        raise ValueError(f"gauss_legendre_integral requires a <= b, got a={a}, b={b}")
    mid = (Interval.point(a) + Interval.point(b)) * Interval.point(0.5)
    half = (Interval.point(b) - Interval.point(a)) * Interval.point(0.5)
    acc = Interval.point(0.0)
    for (nlo, nhi), (wlo, whi) in _GL_RULES[n]:
        t = mid + half * Interval(nlo, nhi)
        acc = acc + Interval(wlo, whi) * f(t)
    main = half * acc
    span = Interval.point(b) - Interval.point(a)
    const = Fraction(math.factorial(n) ** 4, (2 * n + 1) * math.factorial(2 * n) ** 3)
    err = Interval.from_rational(const) * span.pow_int(2 * n + 1) * deriv_2n_bound
    return main + err


def euler_maclaurin_quadrature(
    f: IntervalFn,
    deriv: DerivOracle,
    a: float,
    b: float,
    *,
    panels: int,
    terms: int,
) -> Interval:
    r"""Rigorous high-order integral enclosure via the Euler-Maclaurin trapezoid.

    Corrects the composite trapezoid ``T_h`` by its Euler-Maclaurin endpoint terms,

    .. math::
        \int_a^b f = T_h
            - \sum_{k=1}^{p} \frac{B_{2k}}{(2k)!} h^{2k}
                \bigl(f^{(2k-1)}(b) - f^{(2k-1)}(a)\bigr) - R_p,

    with the classical remainder ``|R_p| \le |B_{2p}|/(2p)! \, h^{2p} (b-a)
    \max_{[a,b]} |f^{(2p)}|``.  ``f`` encloses ``f`` at the trapezoid nodes;
    ``deriv(k, x)`` encloses ``f^{(k)}``.  This is the certified Romberg-class rule:
    each extra Bernoulli pair raises the order by two.  Requires ``a <= b``.
    """
    if panels < 1:
        raise ValueError(f"euler_maclaurin_quadrature needs panels >= 1, got {panels}")
    if terms < 1:
        raise ValueError(f"terms (p) must be >= 1, got {terms}")
    if a > b:
        raise ValueError(f"euler_maclaurin_quadrature requires a <= b, got a={a}, b={b}")
    nodes = _equispaced_node_intervals(a, b, panels)
    vals = [f(node) for node in nodes]
    span = Interval.point(b) - Interval.point(a)
    h = span * Interval.from_rational(Fraction(1, panels))
    trap = (vals[0] + vals[-1]) * Interval.point(0.5)
    for k in range(1, panels):
        trap = trap + vals[k]
    total = h * trap

    a_iv = Interval.point(a)
    b_iv = Interval.point(b)
    for k in range(1, terms + 1):
        coeff = Interval.from_rational(bernoulli_number_exact(2 * k) / math.factorial(2 * k))
        total = total - coeff * h.pow_int(2 * k) * (deriv(2 * k - 1, b_iv) - deriv(2 * k - 1, a_iv))

    box = Interval(a, b)
    f2p = deriv(2 * terms, box)
    bound = (
        Interval.from_rational(abs(bernoulli_number_exact(2 * terms)) / math.factorial(2 * terms))
        * h.pow_int(2 * terms)
        * span
        * Interval.point(f2p.mag)
    ).hi
    return total + Interval(-bound, bound)


def romberg_integral(
    f: IntervalFn,
    deriv: DerivOracle,
    a: float,
    b: float,
    *,
    panels: int = 8,
    terms: int = 4,
) -> Interval:
    r"""Rigorous Romberg-class integral enclosure.

    Romberg integration is Richardson extrapolation of the composite trapezoid;
    because the trapezoid error has the *step-independent-coefficient*
    Euler-Maclaurin expansion, eliminating its first ``terms`` even powers is the
    same asymptotic correction as :func:`euler_maclaurin_quadrature`.  This routine
    is that certified correction (a genuine enclosure, unlike a bare Richardson
    tableau, which would only enclose the trapezoid *values*, not the integral).
    Requires ``a <= b``.
    """
    return euler_maclaurin_quadrature(f, deriv, a, b, panels=panels, terms=terms)


def _cc_weights(n: int) -> list[Interval]:
    r"""Clenshaw-Curtis weights for the ``n + 1`` Chebyshev-Lobatto nodes (intervals).

    Fejer/CC formula ``w_k = (c_k / n)[1 - sum_{j=1}^{floor(n/2)} (b_j/(4j^2-1))
    cos(2 j k pi / n)]`` with ``c_0 = c_n = 1`` else ``2`` and ``b_j = 1`` at
    ``j = n/2`` else ``2``.  The ``cos`` terms are enclosed with :func:`cos_iv`, so
    the weights are rigorous intervals summing to an enclosure of ``2``.
    """
    weights: list[Interval] = []
    half = n // 2
    for k in range(n + 1):
        inner = Interval.point(1.0)
        for j in range(1, half + 1):
            bj = 1 if (2 * j == n) else 2
            angle = PI_IV * Interval.from_rational(2 * j * k) * Interval.from_rational(Fraction(1, n))
            inner = inner - Interval.from_rational(Fraction(bj, 4 * j * j - 1)) * cos_iv(angle)
        ck = 1 if (k == 0 or k == n) else 2
        weights.append(Interval.from_rational(Fraction(ck, n)) * inner)
    return weights


def clenshaw_curtis_integral(
    f: IntervalFn,
    a: float,
    b: float,
    *,
    n: int,
    deriv_np1_bound: Interval,
) -> Interval:
    r"""Rigorous Clenshaw-Curtis enclosure (Chebyshev-Lobatto interpolatory rule).

    Integrates the degree-``n`` Chebyshev interpolant of ``f`` at the ``n + 1``
    Lobatto nodes ``x_k = cos(k pi / n)`` (enclosed with :func:`cos_iv`, weights via
    :func:`_cc_weights`).  The remainder is the *interpolation* error
    ``\int_a^b (f - p_n)`` bounded rigorously (if conservatively) by

    .. math::
        |E| \le (b-a) \frac{(b-a)^{n+1}}{(n+1)!}\, \max_{[a,b]} |f^{(n+1)}|,

    using ``\max |\omega_{n+1}| \le (b-a)^{n+1}``.  ``deriv_np1_bound`` encloses
    ``f^{(n+1)}`` over ``[a, b]``.  A weaker (looser) bound than Gauss for smooth
    ``f``, so it only ever widens the certified gap.  Requires ``a <= b``, ``n>=2``.
    """
    if n < 2:
        raise ValueError(f"clenshaw_curtis_integral needs n >= 2, got {n}")
    if a > b:
        raise ValueError(f"clenshaw_curtis_integral requires a <= b, got a={a}, b={b}")
    mid = (Interval.point(a) + Interval.point(b)) * Interval.point(0.5)
    half = (Interval.point(b) - Interval.point(a)) * Interval.point(0.5)
    weights = _cc_weights(n)
    acc = Interval.point(0.0)
    for k in range(n + 1):
        x_k = cos_iv(PI_IV * Interval.from_rational(k) * Interval.from_rational(Fraction(1, n)))
        t_k = mid + half * x_k
        acc = acc + weights[k] * f(t_k)
    main = half * acc
    span = Interval.point(b) - Interval.point(a)
    bound = (
        span
        * span.pow_int(n + 1)
        * Interval.from_rational(Fraction(1, math.factorial(n + 1)))
        * Interval.point(deriv_np1_bound.mag)
    ).hi
    return main + Interval(-bound, bound)


@dataclass(frozen=True)
class QuadEstimate:
    """A **numerical** quadrature estimate with an a-posteriori error indicator.

    Unlike the certified rules above, this is *not* a guaranteed enclosure: it is a
    high-accuracy estimate plus a heuristic error from the last refinement step,
    honestly labelled ``numerical``.  Use it for integrands where the certified
    rules do not apply (endpoint singularities), and prefer a certified rule when
    a guaranteed bound is required.
    """

    value: float
    error_estimate: float
    label: str = "numerical"


def tanh_sinh_estimate(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    level: int = 6,
) -> QuadEstimate:
    r"""Double-exponential (tanh-sinh) **numerical** estimate for ``\int_a^b f``.

    The substitution ``x = (a+b)/2 + (b-a)/2 \tanh(\tfrac{\pi}{2}\sinh t)`` clusters
    nodes toward the endpoints, so it handles integrable endpoint singularities
    that defeat the derivative-bound rules.  This returns a :class:`QuadEstimate`,
    **not** a certified interval: a rigorous tanh-sinh error needs a
    strip-analyticity bound on ``f`` (not a real-derivative bound), which is out of
    scope here (honestly labelled ``numerical``).  The error indicator is the
    change between the last two refinement levels.
    """
    if a > b:
        raise ValueError(f"tanh_sinh_estimate requires a <= b, got a={a}, b={b}")
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    half_pi = math.pi / 2.0

    def _sum_at(h: float) -> float:
        total = 0.0
        j = 0
        while True:
            t = j * h
            sh = half_pi * math.sinh(t)
            x = math.tanh(sh)
            w = half_pi * math.cosh(t) / (math.cosh(sh) ** 2)
            if 1.0 - abs(x) < 1e-15 and j > 0:
                break
            contrib = w * (f(mid + half * x) + (f(mid - half * x) if j > 0 else 0.0))
            total += contrib
            j += 1
            if j > 2000:
                break
        return total * h

    prev = half * _sum_at(1.0)
    est = prev
    err = abs(prev)
    for lv in range(1, level + 1):
        h = 1.0 / (2**lv)
        est = half * _sum_at(h)
        err = abs(est - prev)
        prev = est
    return QuadEstimate(value=est, error_estimate=err)


__all__ = [
    "DerivOracle",
    "GAUSS_LEGENDRE_MAX_N",
    "IntervalFn",
    "QuadEstimate",
    "clenshaw_curtis_integral",
    "euler_maclaurin_quadrature",
    "gauss_legendre_integral",
    "gauss_legendre_nodes",
    "midpoint_integral",
    "romberg_integral",
    "simpson_integral",
    "tanh_sinh_estimate",
    "trapezoid_integral",
    "trapezoid_sum",
]
