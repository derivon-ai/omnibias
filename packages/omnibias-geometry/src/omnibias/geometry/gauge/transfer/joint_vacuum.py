# SPDX-License-Identifier: Apache-2.0
"""Exact norm and tail arithmetic for the physical SU(2) theta vacuum.

The basis is B_(a,b,c)=Tr(P_c (D_a tensor D_b))/(c+1), with integer
doubled spins. Its product-representation coefficient has orthogonal
projector blocks, so its nuclear norm is the sum of the absolute scalar
basis coefficients. These helpers do not establish an analytic source,
identify a vacuum, set certificate flags, or claim a continuum theorem.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction as Q
from math import comb

from .character_algebra import su2_character_norm, su2_character_product

ThetaState = tuple[int, int, int]
Exact = int | Q


def _q(value: Exact) -> Q:
    if type(value) is not int and not isinstance(value, Q):
        raise TypeError("proof inputs must be exact integers or Fraction")
    return Q(value)


def _order(value: int, *, supported: bool = False) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("norm_order must be a nonnegative integer")
    if supported and value not in (0, 1, 2):
        raise ValueError("the rational tail supports norm_order 0, 1, or 2")


def _theta(values: Mapping[ThetaState, Exact]) -> dict[ThetaState, Q]:
    result: dict[ThetaState, Q] = {}
    for state, raw in values.items():
        if (
            not isinstance(state, tuple)
            or len(state) != 3
            or any(type(label) is not int or label < 0 for label in state)
        ):
            raise ValueError("theta keys must be triples of nonnegative doubled spins")
        a, b, c = state
        if abs(a - b) > c or c > a + b or (a + b + c) % 2:
            raise ValueError("theta spin triple is not admissible")
        value = _q(raw)
        if value:
            result[state] = value
    return result


def _characters(values: Mapping[int, Exact]) -> dict[int, Q]:
    result: dict[int, Q] = {}
    for label, raw in values.items():
        if type(label) is not int or label < 0:
            raise ValueError("character labels must be nonnegative doubled spins")
        value = _q(raw)
        if value:
            result[label] = value
    return result


def su2_theta_norm(
    values: Mapping[ThetaState, Exact], *, norm_order: int = 2
) -> Q:
    """Return the exact joint matrix Fourier norm, including the constant."""
    _order(norm_order)
    return sum(
        (
            (1 + a + b) ** norm_order * abs(value)
            for (a, b, _), value in _theta(values).items()
        ),
        Q(0),
    )


def su2_theta_product_characters(
    left: Mapping[int, Exact], right: Mapping[int, Exact]
) -> dict[ThetaState, Q]:
    """Express f(U)g(V) exactly in the theta basis; this is not convolution."""
    lhs, rhs = _characters(left), _characters(right)
    return {
        (a, b, c): x * y * (c + 1)
        for a, x in lhs.items()
        for b, y in rhs.items()
        for c in range(abs(a - b), a + b + 1, 2)
    }


def su2_theta_vacuum_tail(
    ratio: Exact,
    *,
    first_omitted: int,
    circle_norm: Exact,
    norm_order: int = 2,
) -> Q:
    """Rational support-lifted tail M sum_(n>=first) t^n (n+1)^s C(n+3,3).

    The caller must establish ||psi_n||_2 <= M*rho^(-n) and complete
    outer-spin support a+b <= n. The ratio is abs(g)/rho, not its square.
    """
    _order(norm_order, supported=True)
    if type(first_omitted) is not int or first_omitted < 0:
        raise ValueError("first_omitted must be a nonnegative integer")
    t, bound = _q(ratio), _q(circle_norm)
    if not 0 <= t < 1:
        raise ValueError("ratio must belong to [0, 1)")
    if bound < 0:
        raise ValueError("circle_norm must be nonnegative")
    numerator = (Q(1), 1 + 3 * t, 1 + 10 * t + 9 * t * t)[norm_order]
    total = numerator / (1 - t) ** (4 + norm_order)
    retained = sum(
        (t**n * (n + 1) ** norm_order * comb(n + 3, 3) for n in range(first_omitted)),
        Q(0),
    )
    return bound * (total - retained)


def _product_norm(
    left: Mapping[int, Q], right: Mapping[int, Q], norm_order: int
) -> Q:
    # Sum_c(c+1)=(a+1)(b+1), so no theta matrix needs to be materialized.
    return sum(
        (
            (1 + a + b) ** norm_order * (a + 1) * (b + 1) * abs(x * y)
            for a, x in left.items()
            for b, y in right.items()
        ),
        Q(0),
    )


def su2_theta_product_reference_bound(
    vacuum: Mapping[ThetaState, Exact],
    *,
    vacuum_norm_error: Exact,
    vacuum_l2_error: Exact,
    norm_order: int = 2,
) -> dict[str, Q]:
    """Bound actual density versus a product reference and its own marginals.

    The input is a real finite polynomial p with Haar mean one. The
    external analytic premise is that the actual real mean-one vacuum psi
    obeys the supplied A_s and L2 errors. Its product amplitude reference
    is q=(integral_V p)(integral_U p). The normalized q^2 is a comparison
    density, not a replacement for either actual marginal.

    The returned own-marginal bound uses disjoint Fourier-sector norm
    additivity. All conclusions are conditional arithmetic; no source
    verification or physical claim is inferred from an input polynomial.
    """
    _order(norm_order)
    p = _theta(vacuum)
    if p.get((0, 0, 0), Q(0)) != 1:
        raise ValueError("vacuum polynomial must have Haar mean exactly one")
    error, l2_error = _q(vacuum_norm_error), _q(vacuum_l2_error)
    if error < 0 or l2_error < 0:
        raise ValueError("vacuum errors must be nonnegative")
    left = {a: value / (a + 1) for (a, b, _), value in p.items() if b == 0}
    right = {b: value / (b + 1) for (a, b, _), value in p.items() if a == 0}
    reference = su2_theta_product_characters(left, right)
    difference = dict(p)
    for state, value in reference.items():
        difference[state] = difference.get(state, Q(0)) - value

    p_norm = su2_theta_norm(p, norm_order=norm_order)
    p_norm_zero = su2_theta_norm(p, norm_order=0)
    reference_norm = _product_norm(left, right, norm_order)
    p_square_norm = sum(
        (
            value * value / ((a + 1) * (b + 1) * (c + 1))
            for (a, b, c), value in p.items()
        ),
        Q(0),
    )
    left_square_norm = sum((value * value for value in left.values()), Q(0))
    right_square_norm = sum((value * value for value in right.values()), Q(0))
    reference_square_norm = left_square_norm * right_square_norm
    left_density = {
        label: value / left_square_norm
        for label, value in su2_character_product(left, left).items()
    }
    right_density = {
        label: value / right_square_norm
        for label, value in su2_character_product(right, right).items()
    }
    reference_density_norm = _product_norm(left_density, right_density, norm_order)
    left_centered = su2_character_norm(left_density, norm_order=norm_order) - 1
    right_centered = su2_character_norm(right_density, norm_order=norm_order) - 1
    polynomial_difference = su2_theta_norm(difference, norm_order=norm_order)
    amplitude_error = polynomial_difference + error
    raw_density_error = amplitude_error * (2 * reference_norm + amplitude_error)
    norm_error_vs_p = l2_error * (2 * (p_norm_zero - 1) + l2_error)
    norm_error_vs_reference = abs(p_square_norm - reference_square_norm) + norm_error_vs_p
    density_error = raw_density_error + norm_error_vs_reference * reference_density_norm

    # For h=rho-sigma, h=h_U+h_V+h_UV occupies disjoint nonconstant
    # Fourier sectors. If x+y+z<=delta and b=max(b_U,b_V), then
    # ||C(rho)|| <= z+b*(x+y)+x*y
    #             <= delta+(b-1)*t+t^2/4.
    # Convexity in 0<=t<=delta gives this exact endpoint maximum.
    centered_max = max(left_centered, right_centered)
    own_error = max(
        density_error, centered_max * density_error + density_error * density_error / 4
    )
    return {
        "vacuum_polynomial_norm": p_norm,
        "vacuum_polynomial_norm_zero": p_norm_zero,
        "vacuum_polynomial_l2_norm_squared": p_square_norm,
        "reference_amplitude_norm": reference_norm,
        "reference_amplitude_l2_norm_squared": reference_square_norm,
        "amplitude_difference_polynomial_norm": polynomial_difference,
        "actual_amplitude_to_reference_error": amplitude_error,
        "normalization_error_vs_polynomial": norm_error_vs_p,
        "normalization_error_vs_reference": norm_error_vs_reference,
        "raw_density_to_reference_error": raw_density_error,
        "reference_density_norm": reference_density_norm,
        "reference_left_centered_density_norm": left_centered,
        "reference_right_centered_density_norm": right_centered,
        "actual_joint_to_product_reference_error": density_error,
        "actual_joint_to_own_marginals_error": own_error,
    }
