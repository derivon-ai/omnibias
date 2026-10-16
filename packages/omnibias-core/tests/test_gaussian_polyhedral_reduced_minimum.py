# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original-potential minima, feasible witnesses, and bounded proof contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.proof.certificate import certificate_is_unconditional, verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_reduced_minimum import (
    GaussianPolyhedralReducedMinimum,
    certify_gaussian_polyhedral_reduced_minimum,
)
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    compile_gaussian_polyhedral_reduction,
)


def _reduction(*, nu=F(1, 4), target=F(0), rows=(), bounds=(), gauge=0, budget=None):
    model = GaussianHeatMap(((-1,), (1,)), (gauge, gauge), viscosity=nu, budget=budget)
    result = compile_gaussian_polyhedral_reduction(model, rows, bounds, (target,))
    assert result is not None
    return result


def _minimum(reduction=None, **kwargs):
    pytest.importorskip("mpmath")
    result = certify_gaussian_polyhedral_reduced_minimum(_reduction() if reduction is None else reduction, **kwargs)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _original_scalar(mp, nu, target, seed):
    viscosity, wanted = _mp(mp, nu), _mp(mp, target)

    def velocity(point):
        left = mp.exp(-(point+1)**2/(4*viscosity))
        right = mp.exp(-(point-1)**2/(4*viscosity))
        return point-(right-left)/(left+right)

    point = mp.findroot(lambda x: velocity(x)-wanted, (seed-mp.mpf(".05"), seed+mp.mpf(".05")))
    energy = point**2/2-2*viscosity*mp.log((mp.exp(point/(2*viscosity))+mp.exp(-point/(2*viscosity)))/2)-wanted*point
    return point, energy


@pytest.mark.parametrize("nu", [F(1, 4), F(1, 2), F(1)])
@pytest.mark.parametrize("boxes", [1, 255])
def test_all_q_regimes_retain_true_minima_and_sound_values(nu, boxes):
    mp = pytest.importorskip("mpmath")
    result = _minimum(_reduction(nu=nu), max_boxes_per_piece=boxes)
    with mp.workdps(90):
        if nu < F(1, 2):
            roots = [_original_scalar(mp, nu, 0, seed) for seed in (-1, 1)]
        else:
            roots = [(mp.mpf(0), mp.mpf(0))]
        for point, energy in roots:
            assert _mp(mp, result.minimum.lo) <= energy <= _mp(mp, result.minimum.hi)
            assert any(_mp(mp, box[0].lo) <= point <= _mp(mp, box[0].hi) for box in result.candidate_boxes)
    assert result.gap_upper >= result.minimum.hi-result.minimum.lo
    if result.status == "value_tolerance":
        assert result.gap_upper <= result.value_tolerance
    assert all(piece.evaluated_boxes <= boxes for piece in result.piece_results)


@pytest.mark.parametrize("target", [F(-1, 20), F(1, 20)])
def test_unique_winning_well_is_retained_and_losing_well_pruned(target):
    mp = pytest.importorskip("mpmath")
    result = _minimum(_reduction(target=target))
    with mp.workdps(90):
        point, energy = _original_scalar(mp, F(1, 4), target, -1 if target < 0 else 1)
        assert _mp(mp, result.minimum.lo) <= energy <= _mp(mp, result.minimum.hi)
        assert any(_mp(mp, box[0].lo) <= point <= _mp(mp, box[0].hi) for box in result.candidate_boxes)
    assert all(box[0].hi < 0 if target < 0 else box[0].lo > 0 for box in result.candidate_boxes)


def test_constrained_metastable_branch_wins_against_feasible_boundary():
    mp = pytest.importorskip("mpmath")
    result = _minimum(_reduction(target=F(1, 100), rows=((1,),), bounds=(0,)))
    with mp.workdps(90):
        point, energy = _original_scalar(mp, F(1, 4), F(1, 100), -1)
        assert energy < 0  # Boundary x=0 has normalized energy zero.
        assert _mp(mp, result.minimum.lo) <= energy <= _mp(mp, result.minimum.hi)
        assert any(_mp(mp, box[0].lo) <= point <= _mp(mp, box[0].hi) for box in result.candidate_boxes)
    assert result.sample_point[0] <= 0
    assert all(box[0].hi < 0 for box in result.candidate_boxes)


def test_breakpoint_minimum_keeps_both_overlapping_pieces():
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=F(1, 4))
    reduction = compile_gaussian_polyhedral_reduction(model, ((1, -1), (-1, -1)), (-1, -1), (0, 0))
    assert reduction is not None and len(reduction.pieces) == 2
    result = _minimum(reduction)
    assert result.sample_point == (F(0), F(1))
    assert set(result.candidate_piece_indices) == {0, 1}
    assert all(box[0].lo == box[0].hi == 0 and box[1].lo == box[1].hi == 1 for box in result.candidate_boxes)
    assert result.minimum.lo <= .5 <= result.minimum.hi


def test_nonbinary_rational_singleton_is_sampled_exactly_on_constraint():
    reduction = _reduction(rows=((1,), (-1,)), bounds=(F(1, 3), F(-1, 3)))
    result = _minimum(reduction)
    assert result.sample_point == (F(1, 3),)
    assert all(F(box[0].lo) <= F(1, 3) <= F(box[0].hi) for box in result.candidate_boxes)
    assert all(piece.point == (.5,) for piece in result.piece_results)


def test_exact_feasible_sample_survives_oblique_original_inequalities():
    model = GaussianHeatMap(((-F(3, 5), -F(4, 5)), (F(3, 5), F(4, 5))), (0, F(1, 3)), viscosity=F(1, 4))
    rows, bounds = ((1, 2), (-1, 0)), (F(1, 3), F(2))
    reduction = compile_gaussian_polyhedral_reduction(model, rows, bounds, (F(1, 7), F(-1, 11)))
    assert reduction is not None
    result = _minimum(reduction, max_boxes_per_piece=63)
    assert all(sum(c*x for c, x in zip(row, result.sample_point, strict=True)) <= bound
               for row, bound in zip(rows, bounds, strict=True))
    for index, parameter, box in zip(result.candidate_piece_indices, result.candidate_parameters, result.candidate_boxes, strict=True):
        piece = reduction.pieces[index]
        lower, upper = piece.search_interval
        for endpoint in (parameter.lo, parameter.hi):
            y = lower+(upper-lower)*F(endpoint)
            point = reduction.lift(index, y)
            assert all(F(interval.lo) <= value <= F(interval.hi) for interval, value in zip(box, point, strict=True))


def test_huge_common_bias_gauge_preserves_entire_numeric_answer():
    first, second = _minimum(), _minimum(_reduction(gauge=10**400))
    for name in ("minimum", "sample_point", "sample_value", "candidate_boxes", "candidate_piece_indices", "piece_results"):
        assert getattr(first, name) == getattr(second, name)


def test_certificate_binds_full_reduction_and_frozen_result():
    reduction = _reduction()
    result = _minimum(reduction)
    assert verify_certificate_digest(result.certificate)
    assert certificate_is_unconditional(result.certificate)
    assert result.certificate["payload"]["reduction_certificate"] == reduction.certificate
    reduction.certificate["payload"]["field"] = [8, 9]
    assert verify_certificate_digest(result.certificate)
    with pytest.raises(FrozenInstanceError):
        result.status = "box_budget"
    with pytest.raises(TypeError):
        GaussianPolyhedralReducedMinimum()


def test_damaged_or_infeasible_reduction_refuses_before_transcendentals(monkeypatch):
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduced_minimum")
    monkeypatch.setattr(module, "ln_iv", lambda *_: pytest.fail("unexpected evaluation"))
    damaged = deepcopy(_reduction())
    damaged.certificate["payload"]["field"] = [8, 9]
    assert certify_gaussian_polyhedral_reduced_minimum(damaged) is None
    infeasible = _reduction(rows=((1,), (-1,)), bounds=(0, -1))
    assert not infeasible.feasible
    assert certify_gaussian_polyhedral_reduced_minimum(infeasible) is None


@pytest.mark.parametrize("limit", ["work", "storage"])
def test_aggregate_piece_reservation_precedes_every_callback(limit, monkeypatch):
    limits = {"max_elimination_work": 10_000} if limit == "work" else {"max_matrix_entries": 10_000}
    reduction = _reduction(budget=RidgeBudget(**limits))
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduced_minimum")
    monkeypatch.setattr(module, "minimize_box", lambda *_a, **_k: pytest.fail("partial-piece search started"))
    assert certify_gaussian_polyhedral_reduced_minimum(reduction) is None


@pytest.mark.parametrize("limits", [{"max_matrix_entries": 100_000}, {"max_elimination_work": 600_000}])
def test_redundant_source_metadata_is_reserved_before_hashing_or_copying(limits, monkeypatch):
    reduction = _reduction(rows=((1,),)*1000, bounds=(0,)*1000, budget=RidgeBudget(**limits))
    assert len(reduction.pieces) == 1
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduced_minimum")
    monkeypatch.setattr(module, "_intact", lambda *_: pytest.fail("unreserved source certificate scan"))
    assert certify_gaussian_polyhedral_reduced_minimum(reduction) is None


@pytest.mark.parametrize("strict", [False, True])
def test_strict_and_sticky_provenance_restored_on_success_and_refusal(strict, monkeypatch):
    reduction = _reduction()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    _minimum(reduction)
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_polyhedral_reduced_minimum(reduction)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


def test_missing_backend_and_interval_overflow_are_explicit(monkeypatch):
    reduction = _reduction()
    with monkeypatch.context() as patch:
        patch.setattr(transcend, "_mpmath", lambda: None)
        patch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
        with pytest.raises(RuntimeError, match="mpmath"):
            certify_gaussian_polyhedral_reduced_minimum(reduction)
    pytest.importorskip("mpmath")
    huge = _reduction(target=10**300)
    assert certify_gaussian_polyhedral_reduced_minimum(huge) is None


@pytest.mark.parametrize("bad", [True, 1.5, "2"])
def test_invalid_box_count_types_raise(bad):
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_reduced_minimum(_reduction(), max_boxes_per_piece=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_box_count_raises(bad):
    with pytest.raises(ValueError):
        certify_gaussian_polyhedral_reduced_minimum(_reduction(), max_boxes_per_piece=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_tolerance_domains_raise(bad):
    with pytest.raises(ValueError):
        certify_gaussian_polyhedral_reduced_minimum(_reduction(), value_tolerance=bad)


@pytest.mark.parametrize("bad", [True, "0.1", complex(1)])
def test_invalid_tolerance_types_raise(bad):
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_reduced_minimum(_reduction(), value_tolerance=bad)


def test_wrong_record_type_raises():
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_reduced_minimum(object())
