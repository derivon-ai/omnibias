# SPDX-License-Identifier: Apache-2.0
"""Independent rational character, Haar, and derivative checks.

Grid/random checks below are regressions for the written whole-domain
arguments, not substitutes for those arguments.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.geometry.gauge.transfer.character_algebra import (
    su2_character_convolution,
    su2_character_even_tail,
    su2_character_laplacian,
    su2_character_log_bound,
    su2_character_norm,
    su2_character_potential_bound,
    su2_character_product,
    su2_character_round,
)


def _add(left: list[Q], right: list[Q], scale: Q = Q(1)) -> list[Q]:
    result = left + [Q(0)] * max(0, len(right) - len(left))
    for k, value in enumerate(right):
        result[k] += scale * value
    return result


def _chebyshev(spin: int) -> list[Q]:
    """chi_(spin/2)=U_spin(z), where z is the unit-quaternion scalar part."""
    previous, current = [Q(0)], [Q(1)]
    for _ in range(spin):
        previous, current = current, _add([Q(0)] + [2 * c for c in current], previous, Q(-1))
    return current


def _powers(characters: dict[int, Q]) -> list[Q]:
    result = [Q(0)]
    for spin, coefficient in characters.items():
        result = _add(result, _chebyshev(spin), coefficient)
    return result


def _value(polynomial: list[Q], z: Q) -> Q:
    result = Q(0)
    for coefficient in reversed(polynomial):
        result = result * z + coefficient
    return result


def _derivative(polynomial: list[Q]) -> list[Q]:
    return [degree * coefficient for degree, coefficient in enumerate(polynomial) if degree]


def _lap_value(polynomial: list[Q], z: Q) -> Q:
    first = _derivative(polynomial)
    return ((1 - z*z) * _value(_derivative(first), z) - 3*z*_value(first, z)) / 4


def _points() -> list[Q]:
    rng = random.Random(60491)
    return [Q(i, 50) for i in range(-50, 51)] + [Q(rng.randint(-10000, 10000), 10000) for _ in range(64)]


def _sphere_moment(a: int, b: int) -> Q:
    """Exact integral of z0**a*z1**b on the unit three-sphere."""
    if a % 2 or b % 2:
        return Q(0)
    j, k = a // 2, b // 2
    return Q(math.factorial(2*j) * math.factorial(2*k),
             4**(j+k) * math.factorial(j) * math.factorial(k) * math.factorial(j+k+1))


def _haar_convolution_at(left: list[Q], right: list[Q], a: Q, b: Q) -> Q:
    assert a*a + b*b == 1
    # For W=(a,b,0,0), Re(U^-1 W)=a*z0+b*z1 under Haar U.
    return sum((ci * cj * math.comb(j, k) * a**(j-k) * b**k * _sphere_moment(i+j-k, k)
                for i, ci in enumerate(left) for j, cj in enumerate(right)
                for k in range(j+1)), Q(0))


def test_known_fusion_and_laplacian_normalizations() -> None:
    assert su2_character_product({1: 1}, {1: 1}) == {0: Q(1), 2: Q(1)}
    assert su2_character_product({2: 1}, {3: 1}) == {1: Q(1), 3: Q(1), 5: Q(1)}
    assert su2_character_laplacian({0: 7, 1: 2, 2: -3}) == {1: Q(-3, 2), 2: Q(6)}
    assert su2_character_product({}, {1: 3}) == {}
    assert su2_character_convolution({}, {1: 3}) == {}


def test_dyadic_rounding_preserves_mean_and_floors_negative_coefficients() -> None:
    source = {0: Q(1), 1: Q(1, 10), 2: Q(-1, 10), 4: Q(2)}
    rounded, error = su2_character_round(source, bits=3)
    assert rounded == {0: Q(1), 2: Q(-1, 8), 4: Q(2)}
    assert error == Q(59, 40)
    assert source[1] == Q(1, 10)  # Compression does not mutate the input.
    assert rounded[0] == source[0]
    assert su2_character_round({0: 1, 1: -2, 2: 3}, bits=1) == (
        {0: Q(1), 1: Q(-2), 2: Q(3)}, Q(0)
    )


@pytest.mark.parametrize("bits", [1, 8, 128])
@pytest.mark.parametrize("norm_order", [0, 2, 4])
def test_dyadic_rounding_reports_actual_weighted_error(bits: int, norm_order: int) -> None:
    source = {0: Q(1), 1: Q(-7, 19), 2: Q(13, 31), 7: Q(-1, 10**50)}
    rounded, error = su2_character_round(source, bits=bits, norm_order=norm_order)
    step = Q(1, 1 << bits)
    differences = {spin: value-rounded.get(spin, Q(0)) for spin, value in source.items()}
    assert all(0 <= value < step for value in differences.values())
    assert all((value/step).denominator == 1 for value in rounded.values())
    assert error == sum((value*(spin+1)**(norm_order+1)
                         for spin, value in differences.items()), Q(0))


def test_fusion_agrees_with_independent_chebyshev_polynomials() -> None:
    points = _points()
    for a in range(7):
        for b in range(7):
            product = _powers(su2_character_product({a: 1}, {b: 1}))
            for z in points:
                assert _value(product, z) == _value(_chebyshev(a), z) * _value(_chebyshev(b), z)


@pytest.mark.parametrize("a,b", [(Q(1), Q(0)), (Q(-1), Q(0)), (Q(0), Q(1)),
                                  (Q(3, 5), Q(4, 5)), (Q(5, 13), Q(-12, 13))])
def test_convolution_matches_independent_sphere_haar_moments(a: Q, b: Q) -> None:
    left = {0: Q(1), 1: Q(2, 7), 2: Q(-1, 9), 4: Q(1, 17)}
    right = {0: Q(1), 1: Q(-1, 6), 2: Q(1, 8), 3: Q(3, 23)}
    convolution = su2_character_convolution(left, right)
    assert _value(_powers(convolution), a) == _haar_convolution_at(_powers(left), _powers(right), a, b)
    assert convolution == {0: Q(1), 1: Q(-1, 42), 2: Q(-1, 216)}


@pytest.mark.parametrize("norm_order", [0, 1, 2, 4])
def test_product_norm_and_centered_convolution_contractions(norm_order: int) -> None:
    rng = random.Random(421 + norm_order)
    for _ in range(32):
        left = {j: Q(rng.randint(-5, 5), 19) for j in range(7)}
        right = {j: Q(rng.randint(-5, 5), 23) for j in range(7)}
        a = su2_character_norm(left, norm_order=norm_order)
        b = su2_character_norm(right, norm_order=norm_order)
        assert su2_character_norm(su2_character_product(left, right), norm_order=norm_order) <= a*b
        left.pop(0)
        right.pop(0)
        a = su2_character_norm(left, norm_order=norm_order)
        b = su2_character_norm(right, norm_order=norm_order)
        assert su2_character_norm(su2_character_convolution(left, right), norm_order=norm_order) <= a*b / 2**(norm_order+2)


@pytest.mark.parametrize("norm_order", [0, 1, 2])
@pytest.mark.parametrize("ratio", [Q(0), Q(1, 8), Q(1, 2), Q(3, 4)])
def test_rational_tail_encloses_independent_positive_sum(norm_order: int, ratio: Q) -> None:
    start, stop, bound = 5, 80, Q(7, 3)
    x = ratio**2
    actual = su2_character_even_tail(ratio, first_omitted=start, circle_norm=bound, norm_order=norm_order)
    partial = bound * sum((x**m * sum(d**(norm_order+2) for d in range(1, m+2))
                           for m in range(start, stop)), Q(0))
    # P_s(m)<=(m+1)^(s+3); the ratio of this majorant decreases in m.
    geometric_ratio = x * Q(stop+2, stop+1)**(norm_order+3)
    assert geometric_ratio < 1
    rest = bound * x**stop * (stop+1)**(norm_order+3) / (1-geometric_ratio)
    assert partial <= actual <= partial + rest
    next_tail = su2_character_even_tail(ratio, first_omitted=start+1, circle_norm=bound, norm_order=norm_order)
    assert actual-next_tail == bound*x**start*sum(d**(norm_order+2) for d in range(1, start+2))


@pytest.mark.parametrize("norm_order,expected", [(0, Q(320, 81)), (1, Q(704, 81)),
                                                (2, Q(6080, 243))])
def test_known_full_tail_generating_function_values(norm_order: int, expected: Q) -> None:
    assert su2_character_even_tail(Q(1, 2), first_omitted=0, circle_norm=1, norm_order=norm_order) == expected
    assert su2_character_even_tail(0, first_omitted=0, circle_norm=3, norm_order=norm_order) == 3
    assert su2_character_even_tail(0, first_omitted=1, circle_norm=3, norm_order=norm_order) == 0
    assert su2_character_even_tail(Q(9, 10), first_omitted=0, circle_norm=0, norm_order=norm_order) == 0


def test_weighted_norm_controls_exact_radial_derivatives_on_grid_and_random_points() -> None:
    polynomial = {0: Q(2), 1: Q(-2, 7), 2: Q(1, 5), 5: Q(-1, 37)}
    powers = _powers(polynomial)
    laplacian = _powers(su2_character_laplacian(polynomial))
    norm1 = su2_character_norm(polynomial, norm_order=1)
    norm2 = su2_character_norm(polynomial)
    for z in _points():
        assert _lap_value(powers, z) == _value(laplacian, z)
        gradient_squared = (1-z*z) * _value(_derivative(powers), z)**2 / 4
        assert gradient_squared <= (norm1/2)**2
        assert abs(_lap_value(powers, z)) <= norm2/4


def test_log_error_controls_pointwise_and_derivative_errors() -> None:
    density = {0: Q(1), 1: Q(1, 128), 2: Q(-1, 1000)}
    delta = Q(1, 500)
    actual = dict(density)
    actual[3] = delta/4**3  # Its exact A2 distance from density is delta.
    logarithm, error = su2_character_log_bound(density, error=delta, order=3)
    q, p = _powers(actual), _powers(logarithm)
    dq, dp = _derivative(q), _derivative(p)
    for z in _points():
        value = _value(q, z)
        assert value > 0
        assert abs(math.log(float(value))-float(_value(p, z))) <= float(error)+1e-15
        derivative_error = _value(dq, z)/value-_value(dp, z)
        assert (1-z*z)*derivative_error**2/4 <= (error/2)**2
        actual_lap_log = _lap_value(q, z)/value-(1-z*z)*_value(dq, z)**2/(4*value**2)
        assert abs(actual_lap_log-_lap_value(p, z)) <= error/4


def test_log_identity_and_exact_first_order() -> None:
    assert su2_character_log_bound({0: 1}, error=0, order=8) == ({}, Q(0))
    polynomial, error = su2_character_log_bound({0: 1, 1: Q(1, 100)}, order=1)
    assert polynomial == {1: Q(1, 100)}
    size = Q(2, 25)
    assert error == size**2/(2*(1-size))


def test_potential_known_character_identity_and_scalar_invariance() -> None:
    coefficient, kinetic, delta = Q(2, 9), Q(7, 3), Q(1, 200)
    expected = {
        0: kinetic*3*coefficient**2/4,
        1: -kinetic*3*coefficient/4,
        2: -kinetic*coefficient**2/4,
    }
    polynomial, error = su2_character_potential_bound({1: coefficient}, error=delta, kinetic=kinetic)
    assert polynomial == expected
    assert error == kinetic*delta*(1+16*coefficient+delta)/4
    assert su2_character_potential_bound({0: Q(1000), 1: coefficient}, error=delta, kinetic=kinetic) == (polynomial, error)
    assert su2_character_potential_bound({0: Q(9)}, error=0) == ({}, Q(0))


def test_potential_enclosure_contains_exact_perturbed_values() -> None:
    log_vacuum = {0: Q(5), 1: Q(1, 17), 2: Q(-1, 83)}
    delta, kinetic = Q(1, 400), Q(57)
    actual = dict(log_vacuum)
    actual[4] = delta/5**3
    polynomial, error = su2_character_potential_bound(log_vacuum, error=delta, kinetic=kinetic)
    p, s = _powers(polynomial), _powers(actual)
    ds = _derivative(s)
    for z in _points():
        actual_potential = kinetic * (_lap_value(s, z)+(1-z*z)*_value(ds, z)**2/4)
        assert abs(actual_potential-_value(p, z)) <= error


@pytest.mark.parametrize("name,args,kwargs,error", [
    ("product", ({-1: 1}, {}), {}, ValueError),
    ("product", ({True: 1}, {}), {}, ValueError),
    ("product", ({1: 0.5}, {}), {}, TypeError),
    ("convolution", ({0: True}, {}), {}, TypeError),
    ("norm", ({0: 1},), {"norm_order": -1}, ValueError),
    ("norm", ({0: 1},), {"norm_order": True}, ValueError),
    ("round", ({0: 1},), {"bits": 0}, ValueError),
    ("round", ({0: 1},), {"bits": -1}, ValueError),
    ("round", ({0: 1},), {"bits": True}, ValueError),
    ("round", ({0: 1},), {"bits": 8.0}, ValueError),
    ("round", ({0: 1},), {"norm_order": -1}, ValueError),
    ("tail", (1,), {"first_omitted": 1, "circle_norm": 1}, ValueError),
    ("tail", (Q(-1, 3),), {"first_omitted": 1, "circle_norm": 1}, ValueError),
    ("tail", (0.1,), {"first_omitted": 1, "circle_norm": 1}, TypeError),
    ("tail", (Q(1, 2),), {"first_omitted": -1, "circle_norm": 1}, ValueError),
    ("tail", (Q(1, 2),), {"first_omitted": True, "circle_norm": 1}, ValueError),
    ("tail", (Q(1, 2),), {"first_omitted": 1, "circle_norm": -1}, ValueError),
    ("tail", (Q(1, 2),), {"first_omitted": 1, "circle_norm": 1, "norm_order": 3}, ValueError),
    ("log", ({0: 1},), {"error": -1}, ValueError),
    ("log", ({0: 1},), {"order": 0}, ValueError),
    ("log", ({0: 1},), {"order": True}, ValueError),
    ("log", ({0: 2},), {}, ValueError),
    ("log", ({0: 1, 1: Q(1, 8)},), {}, ValueError),
    ("log", ({0: 1},), {"error": 1}, ValueError),
    ("potential", ({1: 1},), {"kinetic": 0}, ValueError),
    ("potential", ({1: 1},), {"error": -1}, ValueError),
    ("potential", ({1: 1},), {"kinetic": 0.5}, TypeError),
])
def test_exact_input_and_analytic_domain_guards(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any], error: type[Exception],
) -> None:
    functions: dict[str, Any] = {
        "product": su2_character_product, "convolution": su2_character_convolution,
        "norm": su2_character_norm, "tail": su2_character_even_tail,
        "round": su2_character_round,
        "log": su2_character_log_bound, "potential": su2_character_potential_bound,
    }
    with pytest.raises(error):
        functions[name](*args, **kwargs)
