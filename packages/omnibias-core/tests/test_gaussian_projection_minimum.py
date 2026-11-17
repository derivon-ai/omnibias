# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent original energies and contracts for projected-image search."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from itertools import product

import pytest
from omnibias.core.proof.certificate import certificate_is_unconditional, verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_projection_minimum import (
    GaussianProjectionMinimum,
    certify_gaussian_projection_minimum,
)
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _problem(*, centers=((-1, 0), (1, 0), (0, 1)), biases=(0, 0, 0), viscosity=1,
             time=1, rows=(), bounds=(), budget=None):
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time, budget=budget)
    projector = compile_polyhedral_projector(model.dimension, rows, bounds, budget=budget)
    assert projector is not None
    return model, projector


def _minimum(problem=None, target=None, **kwargs):
    pytest.importorskip("mpmath")
    model, projector = _problem() if problem is None else problem
    result = certify_gaussian_projection_minimum(model, projector, (0,)*model.dimension if target is None else target,
                                                **kwargs)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _contains(mp, interval, value):
    return _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _energy(mp, result, point):
    """Direct squared-distance Gaussian density, with the documented gauge."""
    model = result.model
    logits = [_mp(mp, b-model.biases[0])-sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
              /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
              for center, b in zip(model.centers, model.biases, strict=True)]
    peak = max(logits)
    value = -2*_mp(mp, model.viscosity)*(peak+mp.log(sum(mp.exp(logit-peak) for logit in logits)))
    return value-sum(_mp(mp, v)*(x-_mp(mp, a)) for v, x, a in zip(result.target, point, model.centers[0], strict=True))


def _assert_feasible(result):
    assert all(sum(c*x for c, x in zip(row, result.sample_point, strict=True)) <= bound
               for row, bound in zip(result.projector.inequalities, result.projector.upper_bounds, strict=True))


def _stick_point(model, target, parameters):
    """Independent stick-breaking definition, including its final remainder."""
    weights, remainder = [], F(1)
    for parameter in parameters[:model.mode_count-1]:
        weights.append(remainder*parameter)
        remainder *= 1-parameter
    weights.append(remainder)
    return tuple(model.time*v+sum(weight*center[j] for weight, center in zip(weights, model.centers, strict=True))
                 for j, v in enumerate(target))


@pytest.mark.parametrize("boxes", [1, 31, 127])
def test_four_noncollinear_global_wells_are_all_retained(boxes):
    mp = pytest.importorskip("mpmath")
    problem = _problem(centers=((-3, -3), (-3, 3), (3, -3), (3, 3)), biases=(0,)*4, viscosity=F(1, 4))
    result = _minimum(problem, max_boxes=boxes)
    assert result.model.contraction_bound > 1
    with mp.workdps(80):
        root = mp.findroot(lambda x: x-3*mp.tanh(6*x), (mp.mpf("2.9"), mp.mpf("3.1")))
        for x, y in product((-root, root), repeat=2):
            assert _contains(mp, result.minimum, _energy(mp, result, (x, y)))
            assert any(_contains(mp, box[0], x) and _contains(mp, box[1], y) for box in result.candidate_boxes)
    assert result.parameter_result.evaluated_boxes <= boxes
    assert result.status in {"value_tolerance", "box_budget", "resolution_limit"}
    if result.status == "value_tolerance":
        assert result.gap_upper <= result.value_tolerance


def test_projected_convex_hull_is_not_convex_hull_of_projected_centers():
    mp = pytest.importorskip("mpmath")
    problem = _problem(centers=((-1, 2), (2, -1), (1, 1)), biases=(F(41, 360), F(41, 360), F(1, 180)),
                       viscosity=10, rows=((-1, 0), (0, -1)), bounds=(0, 0))
    model, projector = problem
    projected_centers = tuple(projector.project(center).point for center in model.centers)
    assert all(sum(point) == 2 for point in projected_centers)
    root = (F(2, 3), F(2, 3))
    assert sum(root) != 2 and model.contraction_bound < 1
    logits = tuple(b-sum((x-a)**2 for x, a in zip(root, center, strict=True))/(4*model.viscosity*model.time)
                   for center, b in zip(model.centers, model.biases, strict=True))
    assert logits == (F(0),)*3  # Uniform posterior gives the exact stationary point.
    result = _minimum(problem, max_boxes=127)
    with mp.workdps(80):
        exact = tuple(_mp(mp, x) for x in root)
        assert _contains(mp, result.minimum, _energy(mp, result, exact))
        assert any(all(_contains(mp, iv, x) for iv, x in zip(box, exact, strict=True)) for box in result.candidate_boxes)
    _assert_feasible(result)


@pytest.mark.parametrize("target", [(F(1, 7), F(-2, 9)), (0, 0)])
def test_samples_and_candidate_images_remain_sound_across_projection_switches(target):
    mp = pytest.importorskip("mpmath")
    problem = _problem(centers=((-2, 1), (1, -2), (2, 2)), biases=(0, F(-1, 7), F(1, 3)),
                       rows=((-1, 0), (0, -1)), bounds=(0, 0))
    result = _minimum(problem, target, max_boxes=31)
    _assert_feasible(result)
    with mp.workdps(80):
        assert _contains(mp, result.sample_value, _energy(mp, result, tuple(_mp(mp, x) for x in result.sample_point)))
    assert len(result.candidate_parameters) == len(result.candidate_boxes)
    for parameters, physical in zip(result.candidate_parameters, result.candidate_boxes, strict=True):
        for corner in product(*((F(iv.lo), (F(iv.lo)+F(iv.hi))/2, F(iv.hi)) for iv in parameters)):
            point = tuple(max(F(0), x) for x in _stick_point(result.model, result.target, corner))
            assert all(F(iv.lo) <= x <= F(iv.hi) for iv, x in zip(physical, point, strict=True))


def test_lower_dimensional_rational_equality_uses_feasible_sample():
    mp = pytest.importorskip("mpmath")
    problem = _problem(rows=((1, 1), (-1, -1)), bounds=(F(1, 3), F(-1, 3)))
    result = _minimum(problem, max_boxes=31)
    assert sum(result.sample_point) == F(1, 3)
    _assert_feasible(result)
    with mp.workdps(70):
        sample = tuple(_mp(mp, value) for value in result.sample_point)
        assert _contains(mp, result.sample_value, _energy(mp, result, sample))


def test_noncollinear_boundary_minimum_with_nonzero_normal_gradient_is_covered():
    mp = pytest.importorskip("mpmath")
    # At zero all logits vanish, the mean is (2/3,0), and -grad F=(2/3,0)
    # is the outward normal for x<=0. q<1 makes this the unique global minimum.
    problem = _problem(centers=((0, 1), (0, -1), (2, 0)), biases=(F(1, 4), F(1, 4), 1),
                       rows=((1, 0),), bounds=(0,))
    result = _minimum(problem, max_boxes=31)
    assert result.model.contraction_bound < 1
    assert any(all(iv.lo <= 0 <= iv.hi for iv in box) for box in result.candidate_boxes)
    _assert_feasible(result)
    with mp.workdps(80):
        assert _contains(mp, result.minimum, _energy(mp, result, (mp.mpf(0), mp.mpf(0))))


def test_single_mode_dummy_domain_is_exact_projected_quadratic_minimum():
    mp = pytest.importorskip("mpmath")
    problem = _problem(centers=((-2, F(1, 3)),), biases=(F(2, 7),), time=2,
                       rows=((-1, 0),), bounds=(0,))
    result = _minimum(problem, (F(1, 2), F(-1, 5)), max_boxes=1)
    exact = (F(0), F(-1, 15))
    assert result.sample_point == exact
    assert result.parameter_result.evaluated_boxes == 1
    assert all(iv.lo == iv.hi == 0 for iv in result.parameter_result.domain)
    with mp.workdps(70):
        value = _energy(mp, result, tuple(_mp(mp, x) for x in exact))
        assert _contains(mp, result.minimum, value)
        assert _contains(mp, result.sample_value, value)
    assert result.gap_upper < 1e-12


def test_duplicate_coincident_modes_have_zero_image_radius():
    result = _minimum(_problem(centers=((F(1, 3), F(-1, 5)),)*3, biases=(0, 1, -1)), max_boxes=1)
    assert result.sample_point == (F(1, 3), F(-1, 5))
    assert result.gap_upper < 1e-12
    assert all(all(iv.hi-iv.lo < 1e-14 for iv in box) for box in result.candidate_boxes)


def test_exact_translation_and_common_bias_cancel_before_numeric_evaluation():
    reference = _problem()
    offset = (F(10**200), F(-10**200))
    translated = tuple(tuple(x+shift for x, shift in zip(center, offset, strict=True))
                       for center in reference[0].centers)
    shifted = _problem(centers=translated, biases=(10**400,)*3)
    target = (F(1, 4), F(-1, 8))
    first, second = _minimum(reference, target, max_boxes=7), _minimum(shifted, target, max_boxes=7)
    assert first.minimum == second.minimum
    assert first.sample_value == second.sample_value
    assert first.parameter_result == second.parameter_result
    assert second.sample_point == tuple(x+shift for x, shift in zip(first.sample_point, offset, strict=True))


def test_singleton_polyhedron_and_infeasible_polyhedron():
    problem = _problem(rows=((1, 0), (-1, 0), (0, 1), (0, -1)),
                       bounds=(F(1, 3), F(-1, 3), F(-2, 5), F(2, 5)))
    result = _minimum(problem, max_boxes=1)
    assert result.sample_point == (F(1, 3), F(-2, 5))
    _assert_feasible(result)
    model, projector = _problem(rows=((1, 0), (-1, 0)), bounds=(0, -1))
    assert certify_gaussian_projection_minimum(model, projector, (0, 0), max_boxes=1) is None


def test_search_uses_no_gradient_across_nonsmooth_projection(monkeypatch):
    module = import_module("omnibias.core.verified.gaussian_projection_minimum")
    original = module.minimize_box
    calls = []

    def value_only(*args, **kwargs):
        assert kwargs.get("gradient") is None
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "minimize_box", value_only)
    _minimum(max_boxes=1)
    assert calls == [True]


def test_search_reuses_compiled_projection_without_factorization_or_public_seals(monkeypatch):
    problem = _problem(rows=((-1, 0), (0, -1)), bounds=(0, 0))
    module = import_module("omnibias.core.verified.polyhedral_projection")
    linear = import_module("omnibias.core._rational_linear")
    for owner in (module, linear):
        for name in ("factor_positive_definite", "solve_affine"):
            monkeypatch.setattr(owner, name, lambda *_a, **_k: pytest.fail("query compiled a new linear system"))
    monkeypatch.setattr(type(problem[1]), "project", lambda *_a, **_k: pytest.fail("query reset projection budgets"))
    _assert_feasible(_minimum(problem, max_boxes=7))


def test_factory_record_and_dependency_snapshot_are_immutable():
    result = _minimum(max_boxes=1)
    assert verify_certificate_digest(result.certificate)
    assert certificate_is_unconditional(result.certificate)
    result.projector.certificate["payload"]["dimension"] = 99
    assert verify_certificate_digest(result.certificate)
    with pytest.raises(FrozenInstanceError):
        result.status = "box_budget"
    with pytest.raises(TypeError):
        GaussianProjectionMinimum()


def test_damaged_dependency_refuses_before_search(monkeypatch):
    model, projector = _problem()
    damaged = deepcopy(projector)
    damaged.certificate["payload"]["dimension"] = 99
    module = import_module("omnibias.core.verified.gaussian_projection_minimum")
    monkeypatch.setattr(module, "minimize_box", lambda *_a, **_k: pytest.fail("damaged source started search"))
    assert certify_gaussian_projection_minimum(model, damaged, (0, 0), max_boxes=1) is None


def test_mid_search_projection_resource_failure_never_returns_partial_cover(monkeypatch):
    problem = _problem(rows=((-1, 0),), bounds=(0,))
    module = import_module("omnibias.core.verified.gaussian_projection_minimum")
    original = module._project_exact
    calls = 0

    def bounded_projection(*args):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise ValueError("simulated cumulative projection bit budget exhaustion")
        return original(*args)

    monkeypatch.setattr(module, "_project_exact", bounded_projection)
    assert certify_gaussian_projection_minimum(*problem, (0, 0), max_boxes=7) is None
    assert calls == 4


def test_real_exact_bit_exhaustion_returns_no_partial_certificate():
    problem = _problem(budget=RidgeBudget(max_coefficient_bits=8))
    assert _minimum(problem, max_boxes=1).status == "box_budget"
    assert certify_gaussian_projection_minimum(*problem, (0, 0), max_boxes=7) is None


@pytest.mark.parametrize("limited_source", ["model", "projector"])
def test_both_source_budgets_apply_before_search(limited_source, monkeypatch):
    limits = RidgeBudget(max_elimination_work=20_000)
    model = GaussianHeatMap(((-1, 0), (1, 0), (0, 1)), (0, 0, 0),
                            budget=limits if limited_source == "model" else None)
    projector = compile_polyhedral_projector(2, (), (), budget=limits if limited_source == "projector" else None)
    assert projector is not None
    module = import_module("omnibias.core.verified.gaussian_projection_minimum")
    monkeypatch.setattr(module, "minimize_box", lambda *_a, **_k: pytest.fail("unreserved complete search"))
    assert certify_gaussian_projection_minimum(model, projector, (0, 0)) is None


def test_overlarge_box_request_refuses_before_search(monkeypatch):
    problem = _problem()
    module = import_module("omnibias.core.verified.gaussian_projection_minimum")
    monkeypatch.setattr(module, "minimize_box", lambda *_a, **_k: pytest.fail("unreserved complete search"))
    assert certify_gaussian_projection_minimum(*problem, (0, 0), max_boxes=10**1000) is None


def test_source_reservation_precedes_digest_scans(monkeypatch):
    limits = RidgeBudget(max_elimination_work=20_000)
    problem = _problem(budget=limits)
    module = import_module("omnibias.core.verified.gaussian_projection_minimum")
    monkeypatch.setattr(module, "_intact", lambda *_a: pytest.fail("unreserved source scan"))
    assert certify_gaussian_projection_minimum(*problem, (0, 0)) is None


def test_invalid_types_remain_errors_before_preflight(monkeypatch):
    problem = _problem(budget=RidgeBudget(max_elimination_work=20_000))
    module = import_module("omnibias.core.verified.gaussian_projection_minimum")
    monkeypatch.setattr(module, "_source_slots", lambda *_a: pytest.fail("invalid input reached resource preflight"))
    with pytest.raises(TypeError):
        certify_gaussian_projection_minimum(*problem, (0.0, 0))
    with pytest.raises(ValueError):
        certify_gaussian_projection_minimum(*problem, (0, 0), max_boxes=0)


def test_numeric_objective_overflow_refuses_without_false_global_bounds():
    pytest.importorskip("mpmath")
    assert certify_gaussian_projection_minimum(*_problem(), (10**300, 0), max_boxes=1) is None


@pytest.mark.parametrize("strict", [False, True])
def test_strict_and_sticky_backend_history_is_preserved(strict, monkeypatch):
    problem = _problem()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    _minimum(problem, max_boxes=1)
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_projection_minimum(*problem, (0, 0), max_boxes=1)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


def test_missing_transcendental_backend_is_explicit(monkeypatch):
    problem = _problem()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_projection_minimum(*problem, (0, 0), max_boxes=1)


@pytest.mark.parametrize("keyword,value,error", [("max_boxes", True, TypeError), ("max_boxes", 1.5, TypeError),
                                                ("max_boxes", 0, ValueError), ("value_tolerance", True, TypeError),
                                                ("value_tolerance", "1", TypeError), ("value_tolerance", -1, ValueError),
                                                ("value_tolerance", float("inf"), ValueError),
                                                ("value_tolerance", float("nan"), ValueError)])
def test_invalid_controls_raise(keyword, value, error):
    with pytest.raises(error):
        certify_gaussian_projection_minimum(*_problem(), (0, 0), **{keyword: value})


@pytest.mark.parametrize("target,error", [((True, 0), TypeError), ((0.0, 0), TypeError), ((0,), ValueError),
                                          ((0, 0, 0), ValueError), (("0", 0), TypeError)])
def test_invalid_exact_target_raises(target, error):
    with pytest.raises(error):
        certify_gaussian_projection_minimum(*_problem(), target, max_boxes=1)


def test_invalid_source_types_and_dimension_raise():
    model, projector = _problem()
    with pytest.raises(TypeError):
        certify_gaussian_projection_minimum(object(), projector, (0, 0))
    with pytest.raises(TypeError):
        certify_gaussian_projection_minimum(model, object(), (0, 0))
    other = compile_polyhedral_projector(1, (), ())
    with pytest.raises(ValueError):
        certify_gaussian_projection_minimum(model, other, (0, 0))
