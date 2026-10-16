# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact eventual-face path evidence against independent original KKT laws."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)
from omnibias.core.verified.gaussian_polyhedral_path import (
    compile_gaussian_polyhedral_path,
)


def _bind(source, inequalities, upper, active, target, point, *, equalities=(), rhs=()):
    base = restrict_gaussian_heat(source, equalities, rhs)
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, inequalities, upper, active)
    assert face is not None
    equilibrium = certify_gaussian_polyhedral_equilibrium(face, target, point)
    assert equilibrium is not None
    return equilibrium


def _scalar(active=()):
    source = GaussianHeatMap(((-1,), (1,)), (F(1, 4), F(1, 4)))
    return _bind(source, ((1,),), (0,), active, (0,), (0,))


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _evaluate(mp, coefficients, parameter):
    return tuple(sum(_mp(mp, row[i])*parameter**n for n, row in enumerate(coefficients))
                 for i in range(len(coefficients[0])))


@pytest.mark.parametrize("order", [1, 2, 3, 5])
@pytest.mark.parametrize("active", [(), (0,)])
def test_cubic_departure_is_inconclusive_at_lower_orders_then_resolves_the_candidate_face(order, active):
    target = ((-1,),)+((0,),)*(order-1)
    bias = ((0, 2),)+((0, 0),)*(order-1)
    equilibrium = _scalar(active)
    path = compile_gaussian_polyhedral_path(equilibrium, target, bias)
    assert path is not None
    assert path.target_coefficients == ((0,),)+target
    assert path.bias_coefficients == ((F(1, 4), F(1, 4)),)+bias
    assert path.coefficients[0] == (0,) and path.posterior_coefficients[0] == (F(1, 2), F(1, 2))
    assert path.equality_multiplier_coefficients == ((),)*(order+1)

    # Independently expand x-tanh(x/2+t)=-t. On the free branch,
    # x=-2*t^3/3+14*t^5/15+O(t^7), and mean a=x+t. On the bound
    # branch x=0, mean a=tanh(t), and mu=-t+tanh(t).
    free_x = {3: F(-2, 3), 5: F(14, 15)}
    bound_mean = {1: F(1), 3: F(-1, 3), 5: F(2, 15)}
    expected_x = tuple((F(0) if active else free_x.get(n, F(0)),) for n in range(order+1))
    mean = tuple(bound_mean.get(n, F(0)) if active else free_x.get(n, F(0))+int(n == 1)
                 for n in range(order+1))
    posterior = ((F(1, 2), F(1, 2)),)+tuple((-value/2, value/2) for value in mean[1:])
    multipliers = tuple((bound_mean.get(n, F(0))-int(n == 1) if active else F(0),)
                        for n in range(order+1))
    assert path.coefficients == expected_x
    assert path.posterior_coefficients == posterior
    assert path.inequality_multiplier_coefficients == multipliers
    assert path.slack_coefficients == tuple((-row[0],) for row in expected_x)
    assert path.status == ("inconclusive" if order < 3 else "invalid" if active else "valid")
    assert path.eventual_active_indices == (() if path.status == "valid" else None)
    sign = next(test for test in path.sign_tests if test.quantity == ("multiplier" if active else "slack"))
    assert sign.index == 0
    assert sign.coefficients == tuple(row[0] for row in (multipliers if active else path.slack_coefficients))
    if order < 3:
        assert sign.verdict == "inconclusive" and sign.leading_order is None and sign.leading_coefficient is None
    else:
        assert sign.leading_order == 3
        assert sign.leading_coefficient == (F(-1, 3) if active else F(2, 3))
        assert sign.verdict == ("negative" if active else "positive")


def test_valid_cubic_free_path_matches_original_nonlinear_root_and_posterior_through_order_five():
    mp = pytest.importorskip("mpmath")
    path = compile_gaussian_polyhedral_path(_scalar(), ((-1,),)+((0,),)*4, ((0, 2),)+((0, 0),)*4)
    assert path is not None and path.status == "valid"
    with mp.workdps(100):
        errors = []
        for denominator in (8, 16, 32):
            t = mp.mpf(1)/denominator
            root = mp.findroot(lambda x, t=t: x-mp.tanh(x/2+t)+t, (-t**3, 0))
            assert root < 0 and -t+mp.tanh(t) < 0
            original_weights = (mp.exp(mp.mpf(1)/4-(root+1)**2/4), mp.exp(mp.mpf(1)/4+2*t-(root-1)**2/4))
            posterior = tuple(weight/sum(original_weights) for weight in original_weights)
            wanted = (root,)+posterior
            polynomial = _evaluate(mp, path.coefficients, t)+_evaluate(mp, path.posterior_coefficients, t)
            errors.append(max(abs(a-b) for a, b in zip(polynomial, wanted, strict=True)))
        assert errors[1] < errors[0]/100 and errors[2] < errors[1]/100
        assert errors[-1] < mp.mpf("1e-9")


@pytest.mark.parametrize("order", [1, 2, 3])
def test_single_gaussian_exact_polynomial_path_resolves_two_distinct_contact_orders(order):
    age = F(3, 2)
    center = (F(1, 3), F(-2, 5))
    source = GaussianHeatMap((center,), (F(7, 11),), time=age)
    base_target = tuple(-a/age for a in center)
    equilibrium = _bind(source, ((1, 0), (0, 1)), (0, 0), (), base_target, (0, 0))
    target = ((0, 0), (-1, 0), (0, -1))[:order]
    bias = ((10**400,), (F(3, 7),), (-11,))[:order]
    path = compile_gaussian_polyhedral_path(equilibrium, target, bias)
    assert path is not None
    assert path.coefficients == ((0, 0),)+tuple(tuple(age*value for value in row) for row in target)
    assert path.posterior_coefficients == ((1,),)+((0,),)*order
    assert path.inequality_multiplier_coefficients == ((0, 0),)*(order+1)
    assert path.equality_multiplier_coefficients == ((),)*(order+1)
    assert path.slack_coefficients == tuple(tuple(-value for value in row) for row in path.coefficients)
    # At order one the supplied polynomial target is exactly constant and
    # the single bias is a gauge, so the new exact-identity check resolves
    # the entire path. The order-two input has one unresolved zero prefix.
    assert path.status == ("inconclusive" if order == 2 else "valid")
    assert path.eventual_active_indices == ((0, 1) if order == 1 else () if order == 3 else None)
    for sign in path.sign_tests:
        assert sign.quantity == "slack"
        leading = sign.index+2
        if order >= leading:
            assert sign.leading_order == leading and sign.leading_coefficient == age and sign.verdict == "positive"
        elif order == 1:
            assert sign.verdict == "zero" and sign.leading_order is None and sign.identity is not None
        else:
            assert sign.verdict == "inconclusive" and sign.leading_order is None


def test_coupled_active_path_matches_original_kkt_root_multiplier_posterior_and_slack_expansions():
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((-1, 0), (1, 0)), (F(1, 4), F(1, 4)))
    inequalities, upper = ((1, 1), (-1, 0), (0, -1)), (0, 2, 3)
    equilibrium = _bind(source, inequalities, upper, (0,), (0, 0), (0, 0))
    target = ((1, 0), (F(1, 3), F(-1, 5)), (F(-2, 11), F(1, 13)), (0, 0), (0, 0))
    bias = ((0, 1), (F(1, 7), F(-1, 9)), (0, 0), (0, 0), (0, 0))
    path = compile_gaussian_polyhedral_path(equilibrium, target, bias)
    assert path is not None and path.status == "valid"
    assert path.eventual_active_indices == (0,)
    assert path.coefficients[1] == (1, -1)
    assert path.inequality_multiplier_coefficients[1][0] == 1
    assert all(row[0]+row[1] == 0 for row in path.coefficients)
    assert all(row[0] == 0 for row in path.slack_coefficients)
    assert all(row[1:] == (0, 0) for row in path.inequality_multiplier_coefficients)
    assert tuple(sum(row) for row in path.posterior_coefficients) == (1, 0, 0, 0, 0, 0)
    with mp.workdps(110):
        errors = []
        for denominator in (8, 16, 32):
            t = mp.mpf(1)/denominator
            vx, vy = _evaluate(mp, path.target_coefficients, t)
            bminus, bplus = _evaluate(mp, path.bias_coefficients, t)
            contrast = bplus-bminus
            x = mp.findroot(lambda x, contrast=contrast, vx=vx, vy=vy: 2*x-mp.tanh((x+contrast)/2)-vx+vy, (0, t))
            point, multiplier = (x, -x), vy+x
            weights = (mp.exp(bminus-((x+1)**2+x*x)/4), mp.exp(bplus-((x-1)**2+x*x)/4))
            posterior = tuple(weight/sum(weights) for weight in weights)
            velocity = (x-(posterior[1]-posterior[0]), -x)
            assert multiplier > 0 and x > 0
            assert max(abs(u-v+multiplier) for u, v in zip(velocity, (vx, vy), strict=True)) < mp.mpf("1e-95")
            slacks = (mp.mpf(0), 2+x, 3-x)
            assert min(slacks) >= 0
            wanted = point+posterior+(multiplier, mp.mpf(0), mp.mpf(0))+slacks
            polynomial = (_evaluate(mp, path.coefficients, t)+_evaluate(mp, path.posterior_coefficients, t)
                          +_evaluate(mp, path.inequality_multiplier_coefficients, t)+_evaluate(mp, path.slack_coefficients, t))
            errors.append(max(abs(a-b) for a, b in zip(polynomial, wanted, strict=True)))
        assert errors[1] < errors[0]/40 and errors[2] < errors[1]/40
        assert errors[-1] < mp.mpf("1e-7")


def test_original_normal_multiplier_path_is_preserved_with_indefinite_ambient_curvature():
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)))
    equilibrium = _bind(source, ((1, 0),), (0,), (), (0, 0), (0, 1), equalities=((0, 1),), rhs=(1,))
    assert source.contraction_bound == F(5, 2)
    assert equilibrium.face.base_restriction.model.contraction_bound == F(1, 2)
    assert equilibrium.hessian[1][1] == -1
    path = compile_gaussian_polyhedral_path(equilibrium, ((-1, 0),)+((0, 0),)*4, ((0, 2),)+((0, 0),)*4)
    assert path is not None and path.status == "valid" and path.eventual_active_indices == ()
    assert path.coefficients == ((0, 1), (0, 0), (0, 0), (F(-2, 3), 0), (0, 0), (F(14, 15), 0))
    # Along y=1 the original Gaussian mean is (mu,2*mu), where mu=x+t.
    # Original normal stationarity therefore gives lambda=-1+2*(x+t).
    assert path.equality_multiplier_coefficients == ((-1,), (2,), (0,), (F(-4, 3),), (0,), (F(28, 15),))
    with mp.workdps(100):
        t = mp.mpf(1)/32
        x = mp.findroot(lambda x: x-mp.tanh(x/2+t)+t, (-t**3, 0))
        weights = (mp.exp(mp.mpf(9)/4-((x+1)**2+9)/4), mp.exp(mp.mpf(1)/4+2*t-((x-1)**2+1)/4))
        mean = (weights[1]-weights[0])/sum(weights)
        multiplier = -1+2*mean
        assert x < 0 and abs(x-mean+t) < mp.mpf("1e-90")
        assert abs(1-2*mean+multiplier) < mp.mpf("1e-90")
        assert abs(_evaluate(mp, path.equality_multiplier_coefficients, t)[0]-multiplier) < mp.mpf("1e-9")


@pytest.mark.parametrize("active,expected_status", [((0,), "invalid"), ((1,), "valid")])
def test_invalid_selected_multiplier_representation_does_not_reject_the_true_singleton_optimizer(active, expected_status):
    source = GaussianHeatMap(((0,),), (0,))
    equilibrium = _bind(source, ((1,), (-1,)), (0, 0), active, (0,), (0,))
    path = compile_gaussian_polyhedral_path(equilibrium, ((-1,), (F(-1, 2),), (0,)), ((7,), (0,), (-3,)))
    assert path is not None and path.status == expected_status
    assert path.eventual_active_indices == ((0, 1) if expected_status == "valid" else None)
    # The two inequalities force x=0 for every target and bias path. A
    # negative chosen multiplier therefore rejects only this KKT witness.
    assert path.coefficients == ((0,),)*4
    assert path.slack_coefficients == ((0, 0),)*4
    assert path.posterior_coefficients == ((1,), (0,), (0,), (0,))
    selected = next(sign for sign in path.sign_tests if sign.quantity == "multiplier")
    omitted = next(sign for sign in path.sign_tests if sign.quantity == "slack")
    assert selected.index == active[0] and selected.leading_order == 1
    assert selected.leading_coefficient == (-1 if active == (0,) else 1)
    assert selected.verdict == ("negative" if active == (0,) else "positive")
    assert omitted.index == 1-active[0] and omitted.verdict == "zero"
    assert omitted.leading_order is None and omitted.leading_coefficient is None
