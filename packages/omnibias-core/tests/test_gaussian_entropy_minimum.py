# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent Gibbs energies and complete entropy-search contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from itertools import product
from math import isfinite

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import certificate_is_unconditional, verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_entropy_minimum import (
    GaussianEntropyMinimum,
    _EntropyOracle,
    certify_gaussian_entropy_minimum,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _domain(*, centers=((-1, 0), (1, 0), (0, 1)), biases=(0, 0, 0), viscosity=1,
            time=1, rows=(), bounds=(), target=None, budget=None):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time, budget=budget)
    projector = compile_polyhedral_projector(model.dimension, rows, bounds, budget=budget)
    assert projector is not None
    result = certify_gaussian_posterior_domain(model, projector, (0,)*model.dimension if target is None else target)
    assert result is not None
    return result


def _minimum(domain=None, **kwargs):
    result = certify_gaussian_entropy_minimum(_domain() if domain is None else domain, **kwargs)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _contains(mp, interval, value):
    return _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _weights(parameters, count):
    weights, remainder = [], 1
    for value in parameters[:count-1]:
        weights.append(remainder*value)
        remainder *= 1-value
    return (*weights, remainder)


def _original_energy(mp, domain, point):
    model = domain.model
    logits = [_mp(mp, bias-model.biases[0])-sum((x-_mp(mp, coordinate))**2 for x, coordinate in zip(point, center, strict=True))
              /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
              for center, bias in zip(model.centers, model.biases, strict=True)]
    peak = max(logits)
    energy = -2*_mp(mp, model.viscosity)*(peak+mp.log(sum(mp.exp(value-peak) for value in logits)))
    return energy-sum(_mp(mp, v)*(x-_mp(mp, c)) for v, x, c in zip(domain.target, point, model.centers[0], strict=True))


def _entropy(mp, domain, parameters, projection):
    """Independent joint Gaussian free energy, before minimization over p."""
    model = domain.model
    p = _weights(parameters, model.mode_count)
    source = tuple(_mp(mp, model.time*v)+sum(weight*_mp(mp, center[j]) for weight, center in zip(p, model.centers, strict=True))
                   for j, v in enumerate(domain.target))
    point = projection(source)
    # E = sum p_j[|x-a_j|²/(2s)+2nu(log p_j-(b_j-b0))]
    #     - v.(x-a0). This form is independent of the completed-square code.
    energy = sum(weight*(sum((x-_mp(mp, coordinate))**2 for x, coordinate in zip(point, center, strict=True))
                         /(2*_mp(mp, model.time))+2*_mp(mp, model.viscosity)*(mp.log(weight)-_mp(mp, bias-model.biases[0])))
                 for weight, center, bias in zip(p, model.centers, model.biases, strict=True) if weight)
    return energy-sum(_mp(mp, v)*(x-_mp(mp, c)) for v, x, c in zip(domain.target, point, model.centers[0], strict=True))


def _gradient(mp, domain, parameters, projection):
    """Full simplex derivative, including the final dependent probability."""
    model, weights = domain.model, _weights(parameters, domain.model.mode_count)
    source = tuple(_mp(mp, model.time*v)+sum(p*_mp(mp, center[j]) for p, center in zip(weights, model.centers, strict=True))
                   for j, v in enumerate(domain.target))
    point = projection(source)
    partial = tuple(sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))/(2*_mp(mp, model.time))
                    +2*_mp(mp, model.viscosity)*(mp.log(p)+1-_mp(mp, bias-model.biases[0]))
                    for p, center, bias in zip(weights, model.centers, model.biases, strict=True))
    result, prefix = [], mp.mpf(1)
    for j, u in enumerate(parameters[:model.mode_count-1]):
        derivative = prefix*partial[j]-sum(weights[k]*partial[k]/(1-u) for k in range(j+1, model.mode_count))
        result.append(derivative)
        prefix *= 1-u
    return tuple(result)


@pytest.mark.parametrize("parameters", [(F(1, 4), F(2, 3)), (F(3, 4), F(1, 8)), (F(1, 2), F(1, 2))])
def test_value_and_full_stick_gradient_match_independent_joint_energy(parameters):
    mp = pytest.importorskip("mpmath")
    domain = _domain(biases=(F(1, 7), F(-2, 9), F(1, 3)), viscosity=F(3, 4), time=F(5, 4),
                     target=(F(1, 5), F(-1, 7)))
    oracle = _EntropyOracle(domain, _Arithmetic(domain.model.budget))
    box = tuple(Interval.point(float(value)) for value in parameters)
    with _strict(), mp.workdps(100):
        u = tuple(_mp(mp, value.lo) for value in box)
        value, gradient = oracle.value(box), oracle.gradient(box)
        assert _contains(mp, value, _entropy(mp, domain, u, lambda point: point))
        expected = _gradient(mp, domain, u, lambda point: point)
        for j, (enclosure, exact) in enumerate(zip(gradient, expected, strict=True)):
            assert _contains(mp, enclosure, exact)
            derivative = mp.diff(lambda t, index=j: _entropy(mp, domain, (*u[:index], t, *u[index+1:]), lambda point: point), u[j])
            assert abs(derivative-exact) < mp.mpf("1e-85")


def test_entropy_gradient_is_continuous_at_projection_active_set_switch():
    mp = pytest.importorskip("mpmath")
    domain = _domain(centers=((-1,), (1,)), biases=(F(1, 4),)*2, rows=((-1,),), bounds=(0,))
    oracle = _EntropyOracle(domain, _Arithmetic(domain.model.budget))
    def projection(point):
        return (max(mp.mpf(0), point[0]),)
    with _strict(), mp.workdps(100):
        wide = oracle.gradient((Interval(7/16, 9/16),))[0]
        for parameter in (F(7, 16), F(1, 2), F(9, 16)):
            u = _mp(mp, parameter)
            box = (Interval.point(float(parameter)),)
            expected = _gradient(mp, domain, (u,), projection)[0]
            assert _contains(mp, oracle.gradient(box)[0], expected)
            assert _contains(mp, wide, expected)
            assert _contains(mp, oracle.value(box), _entropy(mp, domain, (u,), projection))
        assert _contains(mp, oracle.gradient((Interval.point(.5),))[0], mp.mpf(0))
        epsilon = mp.mpf("1e-40")
        left = _gradient(mp, domain, (mp.mpf(".5")-epsilon,), projection)[0]
        right = _gradient(mp, domain, (mp.mpf(".5")+epsilon,), projection)[0]
        assert abs(left)+abs(right) < mp.mpf("1e-37")


def test_gradient_avoids_log_of_underflowed_probability_product():
    mp = pytest.importorskip("mpmath")
    domain = _domain(centers=((0,),)*65, biases=(0,)*65)
    assert domain.gradient_safe
    oracle = _EntropyOracle(domain, _Arithmetic(domain.model.budget))
    coordinate = 1-2**-32
    box = (Interval.point(coordinate),)*64
    with _strict(), mp.workdps(100):
        gradient = oracle.gradient(box)
        assert all(isfinite(iv.lo) and isfinite(iv.hi) for iv in gradient)
        prefix = mp.power(2, -32*63)
        u = _mp(mp, coordinate)
        exact_last = 2*prefix*mp.log(u/(1-u))
        assert 0 < exact_last < mp.mpf("1e-324")
        assert _contains(mp, gradient[-1], exact_last)


def test_boundary_entropy_contains_zero_log_zero_and_interior_minimum():
    mp = pytest.importorskip("mpmath")
    domain = _domain(centers=((0,),)*3, biases=(0,)*3)
    oracle = _EntropyOracle(domain, _Arithmetic(domain.model.budget))
    with _strict(), mp.workdps(90):
        enclosure = oracle.value((Interval(0.0, 1.0),)*2)
        assert _contains(mp, enclosure, -2*mp.log(3))
        assert _contains(mp, enclosure, mp.mpf(0))
        for parameters in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
            at_vertex = oracle.value(tuple(Interval.point(u) for u in parameters))
            assert _contains(mp, at_vertex, mp.mpf(0))


def test_exact_noncollinear_uniform_posterior_root_and_sample_gibbs_gap():
    mp = pytest.importorskip("mpmath")
    domain = _domain(biases=(F(5, 18), F(5, 18), F(1, 9)))
    result = _minimum(domain, max_boxes=63)
    assert domain.gradient_safe and domain.model.contraction_bound < 1
    with mp.workdps(90):
        root = (mp.mpf(0), mp.mpf(1)/3)
        assert _contains(mp, result.minimum, _original_energy(mp, domain, root))
        assert any(all(_contains(mp, iv, x) for iv, x in zip(box, root, strict=True)) for box in result.candidate_boxes)
        sample = tuple(_mp(mp, x) for x in result.sample_point)
        original = _original_energy(mp, domain, sample)
        parameters = tuple(_mp(mp, x) for x in result.parameter_result.point)
        entropy = _entropy(mp, domain, parameters, lambda point: point)
        assert _contains(mp, result.sample_value, original)
        assert _contains(mp, result.sample_entropy, entropy)
        assert entropy >= original-mp.mpf("1e-80")
    assert result.parameter_result.gradient_evaluations > 0
    assert result.gradient_used


def test_public_original_upper_bound_is_strictly_tighter_than_sample_entropy():
    result = _minimum(_domain(biases=(0, F(1, 3), -1)), max_boxes=1)
    assert result.sample_value.hi < result.sample_entropy.lo-1e-3
    assert result.minimum.hi == result.sample_value.hi
    assert result.parameter_result.minimum.hi == result.sample_entropy.hi
    assert result.gap_upper < result.parameter_result.gap_upper


@pytest.mark.parametrize("boxes", [1, 31, 127])
def test_all_four_noncollinear_physical_global_wells_survive(boxes):
    mp = pytest.importorskip("mpmath")
    domain = _domain(centers=((-3, -3), (-3, 3), (3, -3), (3, 3)), biases=(0,)*4, viscosity=F(1, 4))
    result = _minimum(domain, max_boxes=boxes)
    assert domain.model.contraction_bound > 1
    with mp.workdps(90):
        root = mp.findroot(lambda x: x-3*mp.tanh(6*x), (mp.mpf("2.9"), mp.mpf("3.1")))
        for x, y in product((-root, root), repeat=2):
            assert _contains(mp, result.minimum, _original_energy(mp, domain, (x, y)))
            assert any(_contains(mp, box[0], x) and _contains(mp, box[1], y) for box in result.candidate_boxes)
    assert result.parameter_result.evaluated_boxes <= boxes
    if result.status == "value_tolerance":
        assert result.gap_upper <= result.value_tolerance


def test_interior_entropy_gradient_retains_four_tied_nonconvex_wells():
    mp = pytest.importorskip("mpmath")
    domain = _domain(centers=((-1, -1), (-1, 1), (1, -1), (1, 1)), biases=(0,)*4, viscosity=F(1, 4))
    assert domain.gradient_safe and domain.model.contraction_bound > 1
    result = _minimum(domain, max_boxes=63)
    assert result.parameter_result.gradient_evaluations > 0
    with mp.workdps(90):
        root = mp.findroot(lambda x: x-mp.tanh(2*x), (mp.mpf(".9"), mp.mpf(1)))
        for x, y in product((-root, root), repeat=2):
            assert _contains(mp, result.minimum, _original_energy(mp, domain, (x, y)))
            assert any(_contains(mp, box[0], x) and _contains(mp, box[1], y) for box in result.candidate_boxes)


def test_equality_face_samples_are_exactly_feasible():
    mp = pytest.importorskip("mpmath")
    domain = _domain(rows=((1, 1), (-1, -1)), bounds=(F(1, 3), F(-1, 3)))
    result = _minimum(domain, max_boxes=31)
    assert sum(result.sample_point) == F(1, 3)
    with mp.workdps(80):
        original = _original_energy(mp, domain, tuple(_mp(mp, x) for x in result.sample_point))
        assert _contains(mp, result.sample_value, original)


def test_single_mode_entropy_equals_physical_quadratic_minimum():
    mp = pytest.importorskip("mpmath")
    domain = _domain(centers=((-2, F(1, 3)),), biases=(F(4, 7),), time=2,
                     rows=((-1, 0),), bounds=(0,), target=(F(1, 2), F(-1, 5)))
    result = _minimum(domain, max_boxes=1)
    assert result.sample_point == (F(0), F(-1, 15))
    assert result.gap_upper < 1e-12
    with mp.workdps(80):
        value = _original_energy(mp, domain, tuple(_mp(mp, x) for x in result.sample_point))
        assert _contains(mp, result.minimum, value)
        assert _contains(mp, result.sample_value, value)
        assert _contains(mp, result.sample_entropy, value)


def test_underflowed_posterior_domain_uses_continuous_boundary_value_only(monkeypatch):
    domain = _domain(centers=((0,), (1,), (2,)), biases=(-1000, 0, 0))
    assert not domain.gradient_safe
    module = import_module("omnibias.core.verified.gaussian_entropy_minimum")
    original = module.minimize_box

    def no_gradient(*args, **kwargs):
        assert kwargs.get("gradient") is None
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "minimize_box", no_gradient)
    result = _minimum(domain, max_boxes=7)
    assert result.parameter_result.gradient_evaluations == 0
    assert not result.gradient_used
    assert result.minimum.lo <= result.minimum.hi


def test_translation_and_common_bias_gauges_preserve_objective_and_search():
    first = _domain(target=(F(1, 4), F(-1, 8)))
    shift = (F(10**200), F(-10**200))
    centers = tuple(tuple(c+s for c, s in zip(row, shift, strict=True)) for row in first.model.centers)
    other = _domain(centers=centers, biases=(10**400,)*3, target=first.target)
    a, b = _minimum(first, max_boxes=7), _minimum(other, max_boxes=7)
    assert a.minimum == b.minimum and a.sample_entropy == b.sample_entropy and a.sample_value == b.sample_value
    assert a.parameter_result == b.parameter_result
    assert b.sample_point == tuple(x+s for x, s in zip(a.sample_point, shift, strict=True))


def test_nested_certificate_snapshot_and_frozen_record():
    result = _minimum(max_boxes=1)
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    result.domain.certificate["payload"]["type"] = "damaged"
    assert verify_certificate_digest(result.certificate)
    with pytest.raises(FrozenInstanceError):
        result.status = "box_budget"
    with pytest.raises(TypeError):
        GaussianEntropyMinimum()


@pytest.mark.parametrize("nested", [False, True])
def test_damaged_dependency_refuses_before_search(nested, monkeypatch):
    domain = deepcopy(_domain())
    damaged = domain.projector.certificate if nested else domain.certificate
    damaged["payload"]["type"] = "damaged"
    module = import_module("omnibias.core.verified.gaussian_entropy_minimum")
    monkeypatch.setattr(module, "minimize_box", lambda *_a, **_k: pytest.fail("damaged source started search"))
    assert certify_gaussian_entropy_minimum(domain, max_boxes=1) is None


def test_overlarge_search_refuses_before_callback(monkeypatch):
    domain = _domain()
    module = import_module("omnibias.core.verified.gaussian_entropy_minimum")
    monkeypatch.setattr(module, "minimize_box", lambda *_a, **_k: pytest.fail("unreserved search"))
    assert certify_gaussian_entropy_minimum(domain, max_boxes=10**1000) is None


def test_aggregate_search_budget_precedes_dependency_scan(monkeypatch):
    domain = _domain(budget=RidgeBudget(max_elimination_work=200_000))
    module = import_module("omnibias.core.verified.gaussian_entropy_minimum")
    monkeypatch.setattr(module, "_domain_intact", lambda *_a: pytest.fail("unreserved source scan"))
    assert certify_gaussian_entropy_minimum(domain) is None


def test_exact_bit_exhaustion_after_valid_domain_returns_no_certificate():
    domain = _domain(budget=RidgeBudget(max_coefficient_bits=64))
    assert verify_certificate_digest(domain.certificate)
    assert certify_gaussian_entropy_minimum(domain, max_boxes=1) is None


def test_mid_search_resource_failure_never_returns_partial_global_cover(monkeypatch):
    domain = _domain(rows=((-1, 0),), bounds=(0,))
    module = import_module("omnibias.core.verified.gaussian_entropy_minimum")
    projection = import_module("omnibias.core.verified.gaussian_projection_minimum")
    original = module._project_exact
    calls = 0

    def bounded_projection(*args):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise ValueError("simulated cumulative exact resource exhaustion")
        return original(*args)

    for owner in (module, projection):
        monkeypatch.setattr(owner, "_project_exact", bounded_projection)
    assert certify_gaussian_entropy_minimum(domain, max_boxes=7) is None
    assert calls == 5


def test_compiled_projection_is_reused_without_factorization_or_query_seals(monkeypatch):
    domain = _domain(rows=((-1, 0),), bounds=(0,))
    projection = import_module("omnibias.core.verified.polyhedral_projection")
    linear = import_module("omnibias.core._rational_linear")
    for owner in (projection, linear):
        for name in ("solve_affine", "factor_positive_definite"):
            monkeypatch.setattr(owner, name, lambda *_a, **_k: pytest.fail("query compiled a new linear system"))
    monkeypatch.setattr(type(domain.projector), "project", lambda *_a, **_k: pytest.fail("query reset a projection budget"))
    _minimum(domain, max_boxes=7)


@pytest.mark.parametrize("strict", [False, True])
def test_strict_and_sticky_provenance_is_preserved(strict, monkeypatch):
    domain = _domain()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    _minimum(domain, max_boxes=1)
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_entropy_minimum(domain, max_boxes=1)
    assert transcend.libm_fallback_used() and transcend.strict_backend() is strict


def test_missing_backend_refuses_explicitly(monkeypatch):
    domain = _domain()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_minimum(domain, max_boxes=1)


@pytest.mark.parametrize("keyword,value,error", [("max_boxes", True, TypeError), ("max_boxes", 1.5, TypeError),
                                                ("max_boxes", 0, ValueError), ("value_tolerance", True, TypeError),
                                                ("value_tolerance", "1", TypeError), ("value_tolerance", -1, ValueError),
                                                ("value_tolerance", float("inf"), ValueError),
                                                ("value_tolerance", float("nan"), ValueError)])
def test_invalid_controls_raise(keyword, value, error):
    with pytest.raises(error):
        certify_gaussian_entropy_minimum(_domain(), **{keyword: value})


def test_invalid_domain_type_raises():
    with pytest.raises(TypeError):
        certify_gaussian_entropy_minimum(object())
