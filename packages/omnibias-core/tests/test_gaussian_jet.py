# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact inverse coefficients checked by independent polynomial identities."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from math import factorial

import pytest
from omnibias.core import gaussian_jet
from omnibias.core._rational_linear import factor_positive_definite
from omnibias.core.gaussian_jet import GaussianInverseSeries, gaussian_inverse_series
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _product(left, right, order):
    return [sum((left[j]*right[n-j] for j in range(n+1)), F(0)) for n in range(order+1)]


def _check_formal_identity(series):
    # Independent finite power expansion of exp(g), not the incremental
    # exponential recurrence used by the inverse constructor.
    n, d = series.order, series.dimension
    partition = [F(0)]*(n+1)
    moments = [[F(0)]*(n+1) for _ in range(d)]
    for probability, center in zip(series.posterior, series.centers, strict=True):
        logit = [F(0)] + [sum(a*x for a,x in zip(center,row,strict=True))/(2*series.viscosity*series.time)
                         for row in series.coefficients[1:]]
        power = [F(1)] + [F(0)]*n
        exponential = power[:]
        for degree in range(1, n+1):
            power = _product(power, logit, n)
            exponential = [value+term/factorial(degree) for value,term in zip(exponential,power,strict=True)]
        for degree, value in enumerate(exponential):
            partition[degree] += probability*value
            for axis in range(d):
                moments[axis][degree] += probability*center[axis]*value
    for axis in range(d):
        numerator = [x[axis]-series.time*v[axis] for x,v in zip(series.coefficients,series.target_coefficients,strict=True)]
        assert _product(numerator, partition, n) == moments[axis]


def test_exact_scalar_inverse_coefficients_and_nonlinear_identity():
    series = gaussian_inverse_series(((-1,), (1,)), (F(1,2), F(1,2)), (0,),
                                     ((0,), (1,), *((0,),)*10))
    assert series.coefficients[:6] == ((F(0),), (F(2),), (F(0),), (F(-2,3),), (F(0),), (F(14,15),))
    assert all(series.coefficients[n] == (0,) for n in range(0,series.order+1,2))
    assert series.contraction_bound == F(1,2)
    _check_formal_identity(series)


def test_equal_logit_factory_binds_coupled_existing_model():
    model = GaussianHeatMap(((0,0), (1,0), (0,1)), (0,F(1,4),F(1,4)))
    series = GaussianInverseSeries.from_equal_logits(model, (0,0), ((1,2), (F(1,3),F(-1,5)), *((0,0),)*6))
    assert series.posterior == (F(1,3),)*3
    assert series.target_coefficients[0] == (F(-1,3),)*2
    assert series.contraction_bound == F(1,4)
    assert series.coefficients[1] == (F(84,85), F(186,85))
    _check_formal_identity(series)


def test_asymmetric_family_against_independent_high_precision_inverse():
    mp = pytest.importorskip("mpmath")
    target = ((F(1,15),), (F(1,2),), (F(1,3),), *((0,),)*4)
    series = gaussian_inverse_series(((-1,), (0,), (1,)), (F(1,5),F(2,5),F(2,5)), (F(1,3),),
                                     target, viscosity=F(3,2), time=2)
    _check_formal_identity(series)
    with mp.workdps(70):
        def inverse(t):
            def residual(x):
                weights = [p*mp.exp(a*(x-mp.mpf(1)/3)/6) for a,p in ((-1,mp.mpf(1)/5),(0,mp.mpf(2)/5),(1,mp.mpf(2)/5))]
                mean = (-weights[0]+weights[2])/sum(weights)
                return x-mean-2*(mp.mpf(1)/15+t/2+t*t/3)
            return mp.findroot(residual, mp.mpf(1)/3)
        expected = mp.taylor(inverse, 0, series.order)
        for (value,), reference in zip(series.coefficients, expected, strict=True):
            assert abs(mp.mpf(value.numerator)/value.denominator-reference) < mp.mpf("1e-60")


def test_factorization_is_shared_across_all_orders(monkeypatch):
    calls = 0
    original = gaussian_jet.factor_positive_definite
    def tracked(*args):
        nonlocal calls
        calls += 1
        return original(*args)
    monkeypatch.setattr(gaussian_jet, "factor_positive_definite", tracked)
    gaussian_inverse_series(((0,), (1,)), (F(1,2),)*2, (0,), ((F(-1,2),), (1,), *((0,),)*10))
    assert calls == 1
    gaussian_inverse_series(((0,), (1,)), (F(1,2),)*2, (0,), ((F(-1,2),),))
    assert calls == 1


def test_one_center_is_exact_affine_inverse_of_polynomial_target():
    series = gaussian_inverse_series(((2,-1),), (1,), (0,0), ((-1,F(1,2)), (1,2), (F(1,3),F(-1,5))), time=2)
    assert series.coefficients == ((0,0), (2,4), (F(2,3),F(-2,5)))
    assert series.evaluate(F(1,2)) == (F(7,6),F(19,10))
    assert series.contraction_bound == 0
    _check_formal_identity(series)


def test_immutable_copies_and_factory_only():
    centers = [[-1],[1]]
    path = [[0],[1]]
    series = gaussian_inverse_series(centers, [F(1,2)]*2, [0], path)
    centers[0][0], path[1][0] = 99, 99
    assert series.centers[0] == (-1,) and series.target_coefficients[1] == (1,)
    with pytest.raises(FrozenInstanceError):
        series.coefficients = ()
    with pytest.raises(TypeError, match="use"):
        GaussianInverseSeries()


def test_reusable_exact_ldlt_solves_independent_rhs_and_refuses_non_spd():
    budget = RidgeBudget()
    matrix = tuple(tuple(F(v) for v in row) for row in ((4,1,2),(1,3,0),(2,0,5)))
    factor = factor_positive_definite(matrix, budget)
    for rhs in ((F(1),F(2),F(3)), (F(2,3),F(-1,5),F(0))):
        solution = factor.solve(rhs, budget)
        assert tuple(sum(a*x for a,x in zip(row,solution,strict=True)) for row in matrix) == rhs
    for invalid in ((), ((F(0),),), ((F(-1),),), ((F(1),F(2)),(F(0),F(1)))):
        with pytest.raises(ValueError):
            factor_positive_definite(invalid, budget)
    with pytest.raises(ValueError):
        factor.solve((F(1),), budget)


@pytest.mark.parametrize("posterior", [(F(1,3),F(1,3)), (0,1), (-1,2)])
def test_posterior_must_be_exact_strict_probability_vector(posterior):
    with pytest.raises(ValueError, match="posterior"):
        gaussian_inverse_series(((-1,), (1,)), posterior, (0,), ((0,),))


@pytest.mark.parametrize("value", [0.5, True, "1/2"])
def test_inexact_metadata_refused(value):
    with pytest.raises(TypeError):
        gaussian_inverse_series(((-1,), (1,)), (value,F(1,2)), (0,), ((0,),))


@pytest.mark.parametrize("nu,time", [(0,1), (1,0), (-1,1), (1,F(1,2))])
def test_invalid_or_noncontractive_scales_refused(nu,time):
    with pytest.raises(ValueError):
        gaussian_inverse_series(((-1,), (1,)), (F(1,2),)*2, (0,), ((0,),), viscosity=nu,time=time)


def test_wrong_base_target_and_false_equal_logit_claim_refused():
    with pytest.raises(ValueError, match="constant target"):
        gaussian_inverse_series(((-1,), (1,)), (F(1,2),)*2, (0,), ((1,),))
    model = GaussianHeatMap(((-1,), (1,)), (0,1))
    with pytest.raises(ValueError, match="exactly equal"):
        GaussianInverseSeries.from_equal_logits(model, (0,), ((1,),))


@pytest.mark.parametrize("budget", [RidgeBudget(max_order=1), RidgeBudget(max_matrix_entries=8),
                                    RidgeBudget(max_elimination_work=10), RidgeBudget(max_coefficient_bits=3)])
def test_exact_resource_limits_cover_orders_storage_work_and_bits(budget):
    with pytest.raises(ValueError, match="budget"):
        gaussian_inverse_series(((-1,), (1,)), (F(1,2),)*2, (0,), ((0,),(1,),(0,),(0,),(0,),(0,)), budget=budget)


@pytest.mark.parametrize("centers,posterior,point,path", [((),(),(0,),((0,),)), (((0,),),(1,),(),((0,),)),
    (((0,),),(1,),(0,),()), (((0,1),),(1,),(0,),((0,),)), (((0,),),(1,),(0,),((0,1),))])
def test_empty_and_mismatched_shapes_refused(centers,posterior,point,path):
    with pytest.raises(ValueError):
        gaussian_inverse_series(centers,posterior,point,path)
