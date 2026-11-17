# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent exact polynomial and generating-function calculus checks."""

from fractions import Fraction as F
from math import factorial

import pytest
from omnibias.core._chebyshev_calculus import (
    chebyshev_derivative_norm,
    chebyshev_derivative_tail,
    differentiated_chebyshev_coefficients,
    geometric_power_tail,
)
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.ridge_system import RidgeBudget


def _arithmetic(**kwargs):
    return _Arithmetic(RidgeBudget(**kwargs))


def _basis(degree):
    """Expand T_n in the monomial basis for a separate derivative oracle."""
    polynomials = [[F(1)], [F(0), F(1)]]
    for _ in range(2, degree+1):
        row = [F(0)] + [2*value for value in polynomials[-1]]
        for j, value in enumerate(polynomials[-2]):
            row[j] -= value
        polynomials.append(row)
    return polynomials[:degree+1]


def _evaluate(rows, point):
    basis = _basis(len(rows)-1)
    return tuple(sum(row[axis]*sum(c*point**j for j, c in enumerate(polynomial))
                     for row, polynomial in zip(rows, basis, strict=True)) for axis in range(len(rows[0])))


@pytest.mark.parametrize("order", range(10))
def test_coefficients_match_independent_monomial_differentiation(order):
    rows = tuple((F((-1)**n, n+1), F(n+2, n+3)) for n in range(8))
    width = F(3, 7)
    differentiated = differentiated_chebyshev_coefficients(rows, order, width, _arithmetic())
    assert len(differentiated) == max(1, len(rows)-order)
    basis = _basis(len(rows)-1)
    for physical in (F(-3, 7), F(-2, 13), F(0), F(1, 5), F(3, 7)):
        expected = tuple(sum(row[axis]*sum(c*F(factorial(j), factorial(j-order))*physical**(j-order)/width**j
                                          for j, c in enumerate(polynomial) if j >= order)
                             for row, polynomial in zip(rows, basis, strict=True)) for axis in range(2))
        assert _evaluate(differentiated, physical/width) == expected


def test_zero_constant_and_endpoint_half_normalization():
    arithmetic = _arithmetic()
    assert differentiated_chebyshev_coefficients(((F(5),),), 1, F(2), arithmetic) == ((F(0),),)
    # T_3'=3*T_0+6*T_2, before the physical-coordinate scale.
    assert differentiated_chebyshev_coefficients(((F(0),),)*3+((F(1),),), 1, F(2), arithmetic) == (
        (F(3, 2),), (F(0),), (F(3),))
    assert differentiated_chebyshev_coefficients(((F(0),),)*5, 2, F(1), arithmetic) == ((F(0),),)*3


@pytest.mark.parametrize("degree", range(9))
def test_derivative_norm_matches_endpoint_polynomial_and_bounds_interval(degree):
    polynomial = _basis(degree)[degree]
    for order in range(10):
        norm = chebyshev_derivative_norm(degree, order, _arithmetic())
        derivative = [(j-order, c*F(factorial(j), factorial(j-order)))
                      for j, c in enumerate(polynomial) if j >= order]
        endpoint = sum(c for _, c in derivative)
        assert norm == endpoint
        for point in (F(-1), F(-4, 5), F(-1, 3), F(0), F(3, 7), F(1)):
            assert abs(sum(c*point**j for j, c in derivative)) <= norm


def _unshifted_moment(r, power):
    numerators = (F(1), r, r*(1+r), r*(1+4*r+r*r), r*(1+11*r+11*r*r+r**3))
    return numerators[power]/(1-r)**(power+1)


@pytest.mark.parametrize("r", [F(0), F(1, 7), F(2, 3), F(99, 100)])
@pytest.mark.parametrize("start", [0, 1, 7])
def test_geometric_moments_match_independent_generating_functions(r, start):
    for power in range(5):
        expected = _unshifted_moment(r, power)-sum(F(n**power)*r**n for n in range(start))
        assert geometric_power_tail(r, start, power, _arithmetic()) == expected


@pytest.mark.parametrize("r", [F(0), F(1, 5), F(3, 4)])
@pytest.mark.parametrize("start", [2, 4, 9])
def test_first_two_chebyshev_derivative_tails_match_closed_generating_functions(r, start):
    for order in (1, 2):
        full = _unshifted_moment(r, 2) if order == 1 else (_unshifted_moment(r, 4)-_unshifted_moment(r, 2))/3
        prefix = sum((F(n*n) if order == 1 else F(n**4-n*n, 3))*r**n for n in range(start))
        assert chebyshev_derivative_tail(r, start, order, _arithmetic()) == full-prefix


@pytest.mark.parametrize("order,start", [(3, 3), (4, 7), (8, 13)])
def test_higher_derivative_tail_agrees_with_high_precision_direct_series(order, start):
    mp = pytest.importorskip("mpmath")
    r = F(1, 4)
    exact = chebyshev_derivative_tail(r, start, order, _arithmetic())
    with mp.workdps(90):
        # Direct positive summation of the endpoint product, with hundreds of
        # terms beyond the orders that materially affect the 80-digit result.
        total = mp.mpf(0)
        for n in range(start, 400):
            term = mp.mpf(1)
            for j in range(order):
                term *= mp.mpf(n*n-j*j)/(2*j+1)
            total += term*mp.mpf("0.25")**n
        expected = mp.mpf(exact.numerator)/exact.denominator
        assert abs(total-expected) < mp.mpf("1e-80")*max(1, abs(expected))


def test_tail_differences_recover_each_endpoint_term_and_auxiliary_index_is_allowed():
    for order in range(1, 7):
        arithmetic = _arithmetic(max_order=order)
        r, start = F(2, 7), order+2
        left = chebyshev_derivative_tail(r, start, order, arithmetic)
        right = chebyshev_derivative_tail(r, start+1, order, arithmetic)
        assert left-right == chebyshev_derivative_norm(start, order, arithmetic)*r**start
    assert chebyshev_derivative_tail(F(0), 1000, 3, _arithmetic()) == 0
    assert geometric_power_tail(F(0), 0, 0, _arithmetic()) == 1


@pytest.mark.parametrize("bad", [True, F(1, 2), 1.0])
def test_integer_arguments_reject_ambiguous_types(bad):
    with pytest.raises(TypeError):
        chebyshev_derivative_norm(bad, 0, _arithmetic())
    with pytest.raises(TypeError):
        geometric_power_tail(F(1, 2), 0, bad, _arithmetic())
    with pytest.raises(TypeError):
        chebyshev_derivative_tail(F(1, 2), 2, bad, _arithmetic())


@pytest.mark.parametrize("r", [F(-1, 10), F(1), F(2)])
def test_nonconvergent_geometric_domains_raise(r):
    with pytest.raises(ValueError):
        geometric_power_tail(r, 0, 2, _arithmetic())
    with pytest.raises(ValueError):
        chebyshev_derivative_tail(r, 2, 2, _arithmetic())


def test_invalid_orders_and_malformed_coefficients_raise():
    for degree, order in ((-1, 0), (1, -1)):
        with pytest.raises(ValueError):
            chebyshev_derivative_norm(degree, order, _arithmetic())
    for start, order in ((0, 1), (2, 0), (-1, 1)):
        with pytest.raises(ValueError):
            chebyshev_derivative_tail(F(1, 2), start, order, _arithmetic())
    for rows in ((), ((),), ((F(1),), (F(1), F(2))), ([F(1)],)):
        with pytest.raises(ValueError):
            differentiated_chebyshev_coefficients(rows, 1, F(1), _arithmetic())
    for bad in (True, 0.5):
        with pytest.raises(TypeError):
            differentiated_chebyshev_coefficients(((bad,),), 1, F(1), _arithmetic())
        with pytest.raises(TypeError):
            geometric_power_tail(bad, 0, 1, _arithmetic())
    for width in (F(0), F(-1)):
        with pytest.raises(ValueError):
            differentiated_chebyshev_coefficients(((F(1),),), 0, width, _arithmetic())


def test_order_entries_work_bits_and_shared_cumulative_meter_are_enforced():
    with pytest.raises(ValueError, match="order|power"):
        geometric_power_tail(F(1, 2), 0, 3, _arithmetic(max_order=2))
    with pytest.raises(ValueError, match="order"):
        differentiated_chebyshev_coefficients(((F(1),),), 3, F(1), _arithmetic(max_order=2))
    with pytest.raises(ValueError, match="entries"):
        differentiated_chebyshev_coefficients(((F(1),),)*10, 1, F(1), _arithmetic(max_matrix_entries=50))
    with pytest.raises(ValueError, match="work"):
        geometric_power_tail(F(1, 2), 0, 10, _arithmetic(max_elimination_work=100))
    with pytest.raises(ValueError, match="bit"):
        chebyshev_derivative_tail(F(1, 2), 1000, 2, _arithmetic(max_coefficient_bits=64))
    with pytest.raises(ValueError, match="bit"):
        chebyshev_derivative_norm(2**100, 2, _arithmetic(max_coefficient_bits=64))
    arithmetic = _arithmetic(max_elimination_work=40)
    assert chebyshev_derivative_norm(3, 1, arithmetic) == 9
    used = arithmetic.used
    with pytest.raises(ValueError, match="work"):
        chebyshev_derivative_norm(3, 1, arithmetic)
    assert arithmetic.used > used
