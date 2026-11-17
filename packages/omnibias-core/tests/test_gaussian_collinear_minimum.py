# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original mixture energies, competing wells and complete enclosure contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import certificate_is_unconditional, verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_collinear_minimum import (
    GaussianCollinearMinimum,
    _PieceOracle,
    certify_gaussian_collinear_minimum,
)
from omnibias.core.verified.gaussian_collinear_reduction import compile_gaussian_collinear_reduction
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _reduction(*, centers=((-4,), (0,), (4,)), biases=(0, -1, 0), viscosity=F(1, 4),
               target=(0,), rows=(), bounds=(), budget=None):
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, budget=budget)
    result = compile_gaussian_collinear_reduction(model, rows, bounds, target)
    assert result is not None
    return result


def _minimum(reduction=None, **kwargs):
    pytest.importorskip("mpmath")
    result = certify_gaussian_collinear_minimum(_reduction() if reduction is None else reduction, **kwargs)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _velocity(mp, model, point):
    logits = [_mp(mp, bias-model.biases[0])-sum((value-_mp(mp, coordinate))**2 for value, coordinate in zip(point, center, strict=True))
              /(4*_mp(mp, model.viscosity)*_mp(mp, model.time)) for center, bias in zip(model.centers, model.biases, strict=True)]
    weights = [mp.exp(value-max(logits)) for value in logits]
    total = sum(weights)
    return tuple((point[j]-sum(weight*_mp(mp, center[j]) for weight, center in zip(weights, model.centers, strict=True))/total)
                 /_mp(mp, model.time) for j in range(model.dimension))


def _energy(mp, reduction, point):
    model = reduction.model
    logits = [_mp(mp, bias-model.biases[0])-sum((value-_mp(mp, coordinate))**2 for value, coordinate in zip(point, center, strict=True))
              /(4*_mp(mp, model.viscosity)*_mp(mp, model.time)) for center, bias in zip(model.centers, model.biases, strict=True)]
    peak = max(logits)
    original = -2*_mp(mp, model.viscosity)*(peak+mp.log(sum(mp.exp(value-peak) for value in logits)))
    original -= sum(_mp(mp, v)*x for v, x in zip(reduction.target, point, strict=True))
    projected = tuple(sum(c*v for c, v in zip(row, reduction.target, strict=True)) for row in reduction.projector)
    constant = sum(v*x for v, x in zip(reduction.target, reduction.origin, strict=True))
    constant += reduction.model.time*sum(v*v for v in projected)/2
    return original+_mp(mp, constant)


def _contains(mp, interval, value):
    return _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.mark.parametrize("boxes", [1, 63, 255])
def test_three_well_mixture_retains_both_tied_global_wells(boxes):
    mp = pytest.importorskip("mpmath")
    reduction = _reduction()
    result = _minimum(reduction, max_boxes_per_piece=boxes)
    with mp.workdps(90):
        roots = tuple(mp.findroot(lambda x: _velocity(mp, reduction.model, (x,))[0], (seed-mp.mpf(".05"), seed+mp.mpf(".05")))
                      for seed in (-4, 0, 4))
        energies = tuple(_energy(mp, reduction, (root,)) for root in roots)
        assert abs(energies[0]-energies[2]) < mp.mpf("1e-80")
        assert energies[0] < energies[1]-mp.mpf(".4")
        for point, energy in ((roots[0], energies[0]), (roots[2], energies[2])):
            assert _contains(mp, result.minimum, energy)
            assert any(_contains(mp, box[0], point) for box in result.candidate_boxes)
        if boxes == 255:
            assert not any(box[0].lo <= 0 <= box[0].hi for box in result.candidate_boxes)
    assert all(part.evaluated_boxes <= boxes for part in result.piece_results)
    if result.status == "value_tolerance":
        assert result.gap_upper <= result.value_tolerance


@pytest.mark.parametrize("target", [F(-1, 20), F(1, 20)])
def test_target_tilt_selects_correct_original_outer_well(target):
    mp = pytest.importorskip("mpmath")
    reduction = _reduction(target=(target,))
    result = _minimum(reduction)
    with mp.workdps(80):
        seed = -4 if target < 0 else 4
        root = mp.findroot(lambda x: _velocity(mp, reduction.model, (x,))[0]-_mp(mp, target), (seed, seed+mp.mpf(".1")))
        assert _contains(mp, result.minimum, _energy(mp, reduction, (root,)))
        assert any(_contains(mp, box[0], root) for box in result.candidate_boxes)
    assert all(box[0].hi < 0 if target < 0 else box[0].lo > 0 for box in result.candidate_boxes)


def test_constrained_central_local_well_can_be_global_on_the_polyhedron():
    mp = pytest.importorskip("mpmath")
    reduction = _reduction(rows=((1,), (-1,)), bounds=(1, 1))
    result = _minimum(reduction)
    with mp.workdps(80):
        energy = _energy(mp, reduction, (mp.mpf(0),))
        assert _contains(mp, result.minimum, energy)
        assert energy < min(_energy(mp, reduction, (mp.mpf(-1),)), _energy(mp, reduction, (mp.mpf(1),)))
    assert any(box[0].lo <= 0 <= box[0].hi for box in result.candidate_boxes)
    assert -1 <= result.sample_point[0] <= 1


def test_duplicate_modes_and_giant_common_bias_remain_bound_to_original_family():
    mp = pytest.importorskip("mpmath")
    reduction = _reduction(centers=((-1,), (0,), (0,), (1,)), biases=(0, 0, 0, 0), viscosity=1)
    shifted = _reduction(centers=((-1,), (0,), (0,), (1,)), biases=(10**400,)*4, viscosity=1)
    result, other = _minimum(reduction), _minimum(shifted)
    for name in ("minimum", "sample_point", "candidate_boxes", "piece_results"):
        assert getattr(result, name) == getattr(other, name)
    with mp.workdps(70):
        assert _contains(mp, result.minimum, _energy(mp, reduction, (mp.mpf(0),)))
    assert any(box[0].lo <= 0 <= box[0].hi for box in result.candidate_boxes)


def test_breakpoint_and_rational_singleton_lifts_stay_exactly_feasible():
    reduction = _reduction(centers=((-1, 0), (0, 0), (1, 0)), biases=(0, 0, 0), viscosity=1,
                           rows=((1, -1), (-1, -1)), bounds=(-1, -1), target=(0, 0))
    result = _minimum(reduction, max_boxes_per_piece=63)
    assert result.sample_point == (F(0), F(1))
    assert all(box[0].lo <= 0 <= box[0].hi and box[1].lo <= 1 <= box[1].hi for box in result.candidate_boxes)
    reduction = _reduction(rows=((1,), (-1,)), bounds=(F(1, 3), F(-1, 3)))
    result = _minimum(reduction)
    assert result.sample_point == (F(1, 3),)
    assert all(F(box[0].lo) <= F(1, 3) <= F(box[0].hi) for box in result.candidate_boxes)


def test_rotated_translated_geometry_matches_original_energy_and_feasible_witness():
    mp = pytest.importorskip("mpmath")
    origin, direction = (F(1, 3), F(-2, 5)), (F(3, 5), F(4, 5))
    centers = tuple(tuple(o+t*a for o, a in zip(origin, direction, strict=True)) for t in (-2, 0, 1, 3))
    rows, bounds = ((1, 2), (-1, 0)), (1, 3)
    reduction = _reduction(centers=centers, biases=(0, F(-1, 7), F(2, 5), 0), viscosity=F(3, 4),
                           target=(F(1, 4), F(-1, 5)), rows=rows, bounds=bounds)
    result = _minimum(reduction, max_boxes_per_piece=63)
    assert all(sum(c*x for c, x in zip(row, result.sample_point, strict=True)) <= bound for row, bound in zip(rows, bounds, strict=True))
    with mp.workdps(90):
        sample = tuple(_mp(mp, value) for value in result.sample_point)
        assert _contains(mp, result.sample_value, _energy(mp, reduction, sample))
    for index, parameter, box in zip(result.candidate_piece_indices, result.candidate_parameters, result.candidate_boxes, strict=True):
        piece = reduction.pieces[index]
        left, right = piece.search_interval
        for endpoint in (parameter.lo, parameter.hi):
            point = reduction.lift(index, left+(right-left)*F(endpoint))
            assert all(F(iv.lo) <= value <= F(iv.hi) for iv, value in zip(box, point, strict=True))


def test_affine_logits_cancel_exactly_before_binary64_conversion():
    reduction = _reduction(centers=((0,), (1,), (2,)), biases=(0, -10**400, -2*10**400), viscosity=1)
    oracle = _PieceOracle(reduction, reduction.pieces[0], _Arithmetic(reduction.model.budget))
    logits = oracle.logits(F(10**400), F(10**400))
    assert all(interval.lo == interval.hi == value for interval, value in zip(logits, (0, -.25, -1), strict=True))


def test_underflowed_small_mode_does_not_poison_other_mode_enclosures():
    reduction = _reduction(centers=((-1,), (0,), (1,)), biases=(0, -10**300, 0), viscosity=1)
    result = _minimum(reduction)
    assert any(box[0].lo <= 0 <= box[0].hi for box in result.candidate_boxes)
    assert result.minimum.lo <= result.minimum.hi


def test_posterior_mean_endpoint_enclosures_are_monotone_and_support_bounded():
    mp = pytest.importorskip("mpmath")
    reduction = _reduction()
    oracle = _PieceOracle(reduction, reduction.pieces[0], _Arithmetic(reduction.model.budget))
    with mp.workdps(80):
        for point in (F(-100), F(0), F(17), F(32), F(64), F(100)):
            enclosure = oracle.mean(point)
            logits = [_mp(mp, t*point+b) for t, b in zip(reduction.center_coordinates, reduction.logit_offsets, strict=True)]
            weights = [mp.exp(x-max(logits)) for x in logits]
            mean = sum(weight*_mp(mp, t) for weight, t in zip(weights, reduction.center_coordinates, strict=True))/sum(weights)
            assert _contains(mp, enclosure, mean)
            assert enclosure.lo >= float(reduction.mean_bounds[0]) and enclosure.hi <= float(reduction.mean_bounds[1])


def test_nested_certificate_copy_and_frozen_result():
    result = _minimum()
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    result.reduction.certificate["payload"]["field"] = [9, 1]
    assert verify_certificate_digest(result.certificate)
    with pytest.raises(FrozenInstanceError):
        result.status = "box_budget"
    with pytest.raises(TypeError):
        GaussianCollinearMinimum()


def test_damaged_or_infeasible_source_refuses_before_callback(monkeypatch):
    module = import_module("omnibias.core.verified.gaussian_collinear_minimum")
    monkeypatch.setattr(module, "minimize_box", lambda *_a, **_k: pytest.fail("source refusal started search"))
    damaged = deepcopy(_reduction())
    damaged.certificate["payload"]["field"] = [9, 1]
    assert certify_gaussian_collinear_minimum(damaged) is None
    infeasible = _reduction(rows=((1,), (-1,)), bounds=(0, -1))
    assert not infeasible.feasible and certify_gaussian_collinear_minimum(infeasible) is None


def test_mode_dependent_work_preflight_precedes_all_search(monkeypatch):
    reduction = _reduction(budget=RidgeBudget(max_elimination_work=1_000_000))
    module = import_module("omnibias.core.verified.gaussian_collinear_minimum")
    monkeypatch.setattr(module, "minimize_box", lambda *_a, **_k: pytest.fail("unreserved mode work"))
    assert certify_gaussian_collinear_minimum(reduction) is None


def test_large_source_metadata_preflight_precedes_certificate_scan(monkeypatch):
    reduction = _reduction(rows=((1,),)*1000, bounds=(0,)*1000, budget=RidgeBudget(max_matrix_entries=100_000))
    assert len(reduction.pieces) == 1
    module = import_module("omnibias.core.verified.gaussian_collinear_minimum")
    monkeypatch.setattr(module, "_intact", lambda *_: pytest.fail("unreserved certificate scan"))
    assert certify_gaussian_collinear_minimum(reduction) is None


def test_exact_bit_exhaustion_never_returns_a_partial_search_certificate():
    pytest.importorskip("mpmath")
    reduction = _reduction(budget=RidgeBudget(max_coefficient_bits=16))
    coarse = certify_gaussian_collinear_minimum(reduction, max_boxes_per_piece=1)
    assert coarse is not None and coarse.status == "box_budget"
    assert certify_gaussian_collinear_minimum(reduction, max_boxes_per_piece=255) is None


@pytest.mark.parametrize("strict", [False, True])
def test_strict_and_sticky_provenance_are_preserved(strict, monkeypatch):
    reduction = _reduction()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    _minimum(reduction)
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_collinear_minimum(reduction)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


def test_missing_backend_and_numeric_overflow_refuse_explicitly(monkeypatch):
    reduction = _reduction()
    with monkeypatch.context() as patch:
        patch.setattr(transcend, "_mpmath", lambda: None)
        patch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
        with pytest.raises(RuntimeError, match="mpmath"):
            certify_gaussian_collinear_minimum(reduction)
    pytest.importorskip("mpmath")
    assert certify_gaussian_collinear_minimum(_reduction(target=(10**300,))) is None


@pytest.mark.parametrize("keyword,value,error", [("max_boxes_per_piece", True, TypeError), ("max_boxes_per_piece", 1.5, TypeError),
                                                ("max_boxes_per_piece", 0, ValueError), ("value_tolerance", True, TypeError),
                                                ("value_tolerance", "1", TypeError), ("value_tolerance", -1, ValueError),
                                                ("value_tolerance", float("inf"), ValueError), ("value_tolerance", float("nan"), ValueError)])
def test_invalid_controls_raise(keyword, value, error):
    with pytest.raises(error):
        certify_gaussian_collinear_minimum(_reduction(), **{keyword: value})


def test_invalid_source_type_raises():
    with pytest.raises(TypeError):
        certify_gaussian_collinear_minimum(object())
