# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Whole-hull uniqueness proofs, critical curvature, and dependency contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    seal_certificate,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_reduced_minimum import (
    certify_gaussian_polyhedral_reduced_minimum,
)
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    compile_gaussian_polyhedral_reduction,
)
from omnibias.core.verified.gaussian_polyhedral_unique import (
    GaussianPolyhedralUnique,
    certify_gaussian_polyhedral_unique,
)


def _minimum(*, nu=F(1, 2), target=0, rows=(), bounds=(), boxes=1):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=nu)
    reduction = compile_gaussian_polyhedral_reduction(model, rows, bounds, (target,))
    assert reduction is not None
    minimum = certify_gaussian_polyhedral_reduced_minimum(reduction, max_boxes_per_piece=boxes)
    assert minimum is not None
    return minimum


def _corner(bound, *, boxes=127):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=F(1, 4))
    reduction = compile_gaussian_polyhedral_reduction(model, ((1, -1), (-1, -1)), (bound, bound), (0, 0))
    assert reduction is not None
    minimum = certify_gaussian_polyhedral_reduced_minimum(reduction, max_boxes_per_piece=boxes, value_tolerance=1e-5)
    assert minimum is not None
    return minimum


def _hull(minimum):
    endpoints = []
    for index, interval in zip(minimum.candidate_piece_indices, minimum.candidate_parameters, strict=True):
        lo, hi = minimum.reduction.pieces[index].search_interval
        endpoints.extend(lo+(hi-lo)*F(value) for value in (interval.lo, interval.hi))
    return min(endpoints), max(endpoints)


def test_critical_unique_minimum_has_zero_curvature_bound_and_strict_convexity():
    minimum = _minimum()
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None and result.minimum is minimum
    assert result.parameter_hull == _hull(minimum) == (-1, 1)
    assert result.method == "strict_convexity" and result.curvature_lower_bound == 0
    assert result.checked_piece_indices == (0,)
    assert result.piece_parameter_intervals == ((F(-1), F(1)),)
    assert result.piece_curvature_lower_bounds == (0,)
    # At the unique root, original scalar curvature vanishes; strong convexity
    # is not needed because y-tanh(y) is strictly increasing through zero.
    assert minimum.reduction.pieces[0].scalar_curvature == 1


def test_strong_convexity_bound_uses_normalized_scalar_curvature():
    minimum = _minimum(nu=1)
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None and result.method == "strong_convexity"
    assert result.parameter_hull == (F(-1, 2), F(1, 2))
    assert result.curvature_lower_bound == 1  # Objective curvature is 2*nu times this.
    assert result.piece_curvature_lower_bounds == (1,)


def test_separated_tied_minima_do_not_prove_uniqueness_from_local_curvature():
    minimum = _minimum(nu=F(1, 4), boxes=255)
    assert any(box[0].hi < 0 for box in minimum.candidate_boxes)
    assert any(box[0].lo > 0 for box in minimum.candidate_boxes)
    assert all(not box[0].contains(0) for box in minimum.candidate_boxes)
    lower, upper = _hull(minimum)
    assert lower < 0 < upper
    assert certify_gaussian_polyhedral_unique(minimum) is None


def test_a_pruned_nonconvex_piece_in_the_hull_must_still_be_checked():
    minimum = _corner(F(1, 4))
    assert set(minimum.candidate_piece_indices) == {1, 2}
    central = minimum.reduction.pieces[0]
    assert central.scalar_curvature == F(1, 2) and (central.lower, central.upper) == (F(-1, 2), F(1, 2))
    assert all(minimum.reduction.pieces[index].scalar_curvature == 1 for index in minimum.candidate_piece_indices)
    assert certify_gaussian_polyhedral_unique(minimum) is None


def test_point_only_nonconvex_piece_does_not_block_critical_uniqueness():
    minimum = _corner(F(0), boxes=1)
    point_piece = minimum.reduction.pieces[0]
    assert point_piece.lower == point_piece.upper == 0 and point_piece.scalar_curvature == F(1, 2)
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None and result.method == "strict_convexity"
    assert result.curvature_lower_bound == 0 and result.parameter_hull[0] < 0 < result.parameter_hull[1]
    assert result.checked_piece_indices == (1, 2)
    assert result.piece_curvature_lower_bounds == (0, 0)
    assert result.piece_parameter_intervals[0][0] == result.piece_parameter_intervals[1][1] == 0


def test_unique_nonconvex_winner_gets_a_rational_positive_local_curvature_bound():
    minimum = _minimum(nu=F(1, 4), target=F(1, 20), boxes=255)
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None and result.method == "strong_convexity"
    assert result.parameter_hull == _hull(minimum)
    closest = result.parameter_hull[0]
    assert closest > 0
    expected = F(1, 2)-1/(1+closest*closest/2)**2
    assert expected > 0 and result.curvature_lower_bound == expected


def test_singleton_scalar_domain_is_unique_without_positive_curvature():
    minimum = _minimum(nu=F(1, 4), rows=((1,), (-1,)), bounds=(F(1, 3), F(-1, 3)))
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None and result.method == "singleton"
    assert result.parameter_hull == (F(2, 3), F(2, 3))
    assert result.curvature_lower_bound is None
    assert minimum.sample_point == (F(1, 3),)


def test_nonsmooth_breakpoint_singleton_is_also_unique():
    minimum = _corner(F(-1))
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None and result.method == "singleton"
    assert result.parameter_hull == (0, 0) and result.curvature_lower_bound is None
    assert minimum.sample_point == (0, 1)


def test_exact_evidence_covers_the_complete_hull():
    result = certify_gaussian_polyhedral_unique(_corner(F(0), boxes=1))
    assert result is not None
    intervals = sorted(result.piece_parameter_intervals)
    assert intervals[0][0] == result.parameter_hull[0] and intervals[-1][1] == result.parameter_hull[1]
    right = intervals[0][1]
    for left, end in intervals[1:]:
        assert left <= right
        right = max(right, end)
    assert right == result.parameter_hull[1]


def test_certificate_snapshots_complete_dependencies_and_record_is_frozen():
    minimum = _minimum()
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None
    saved = deepcopy(result.certificate)
    assert verify_certificate_digest(result.certificate) and schema_errors_v1(result.certificate) == []
    assert certificate_is_unconditional(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    with pytest.raises(TypeError):
        GaussianPolyhedralUnique()
    with pytest.raises(FrozenInstanceError):
        result.method = "singleton"
    minimum.certificate["payload"]["minimum"]["lo"] = "0x0.0p+0"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert certify_gaussian_polyhedral_unique(minimum) is None


@pytest.mark.parametrize("dependency", ["minimum", "reduction", "nested_reduction"])
def test_damaged_dependencies_refuse_before_exact_proof_arithmetic(dependency, monkeypatch):
    minimum = _minimum()
    if dependency == "minimum":
        minimum.certificate["payload"]["status"] = "altered"
    elif dependency == "reduction":
        minimum.reduction.certificate["payload"]["field"] = [99, 1]
    else:
        minimum.certificate["payload"]["reduction_certificate"]["payload"]["field"] = [99, 1]
        # Seal only the outer snapshot: the nested digest must still be checked.
        changed = seal_certificate(minimum.certificate)
        minimum.certificate.clear()
        minimum.certificate.update(changed)
        assert verify_certificate_digest(minimum.certificate)
    assert certify_gaussian_polyhedral_unique(minimum) is None


def test_individually_valid_but_mismatched_nested_dependency_is_refused():
    minimum, other = _minimum(), _minimum(nu=1)
    minimum.certificate["payload"]["reduction_certificate"] = deepcopy(other.reduction.certificate)
    changed = seal_certificate(minimum.certificate)
    minimum.certificate.clear()
    minimum.certificate.update(changed)
    assert verify_certificate_digest(minimum.certificate)
    assert verify_certificate_digest(minimum.certificate["payload"]["reduction_certificate"])
    assert certify_gaussian_polyhedral_unique(minimum) is None


@pytest.mark.parametrize("value", [None, 1, True, (), {}])
def test_bad_record_types_raise_before_resource_preflight(value, monkeypatch):
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_: pytest.fail("bad record reached preflight"))
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_unique(value)


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_aggregate_preflight_precedes_dependency_hashing(guard, monkeypatch):
    minimum = _minimum()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_unique")

    def exhausted(*args):
        raise ValueError("test uniqueness resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_minimum_intact", lambda *_: pytest.fail("failed preflight reached dependency hashing"))
    assert certify_gaussian_polyhedral_unique(minimum) is None


@pytest.mark.parametrize("strict,sticky", [(False, False), (False, True), (True, False)])
def test_cached_proof_uses_no_new_solver_or_transcendentals_and_preserves_provenance(strict, sticky, monkeypatch):
    minimum = _minimum()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduced_minimum")

    def fail(*args, **kwargs):
        pytest.fail("uniqueness query recomputed the minimum or evaluated a transcendental")

    monkeypatch.setattr(module, "minimize_box", fail)
    for name in ("exp_iv", "ln_iv", "tanh_iv"):
        monkeypatch.setattr(transcend, name, fail)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None
    assert result.certificate["meta"]["transcend_backend"] == minimum.certificate["meta"]["transcend_backend"]
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used() is sticky


def test_exact_followup_without_optional_backend_still_obeys_strict_sealing(monkeypatch):
    minimum = _minimum()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "mpmath"
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_polyhedral_unique(minimum)


def test_strict_sticky_policy_is_not_reset_by_exact_followup(monkeypatch):
    minimum = _minimum()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_polyhedral_unique(minimum)
    assert transcend.strict_backend() and transcend.libm_fallback_used()


def test_singleton_at_negative_actual_curvature_claims_no_curvature_bound():
    minimum = _minimum(nu=F(1, 4), rows=((1,), (-1,)), bounds=(0, 0))
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None and result.method == "singleton"
    assert result.parameter_hull == (0, 0) and result.curvature_lower_bound is None
    assert minimum.reduction.pieces[0].scalar_curvature-1 == F(-1, 2)
    assert result.certificate["payload"]["curvature_lower_bound"] is None


def test_dependency_reservations_include_original_constraints(monkeypatch):
    plain = _minimum(nu=1)
    large = _minimum(nu=1, rows=((1,),)*128, bounds=(10,)*128)
    module = import_module("omnibias.core.verified.gaussian_polyhedral_unique")
    entries, work = [], []
    original_entries, original_work = RidgeBudget._entries, RidgeBudget._work

    def reserve(self, count):
        entries.append(count)
        return original_entries(self, count)

    def charge(self, count):
        work.append(count)
        return original_work(self, count)

    monkeypatch.setattr(RidgeBudget, "_entries", reserve)
    monkeypatch.setattr(RidgeBudget, "_work", charge)
    module._minimum_preflight(plain, _Arithmetic(plain.reduction.model.budget))
    small_entries, small_work = entries[0], work[0]
    entries.clear()
    work.clear()
    module._minimum_preflight(large, _Arithmetic(large.reduction.model.budget))
    assert entries[0] > small_entries+128 and work[0] > small_work+128


def test_dependency_reservations_include_stored_box_diagnostics(monkeypatch):
    coarse, refined = _corner(F(0), boxes=1), _corner(F(0), boxes=127)
    assert sum(len(result.candidates) for result in refined.piece_results) > sum(len(result.candidates) for result in coarse.piece_results)
    module = import_module("omnibias.core.verified.gaussian_polyhedral_unique")
    entries = []
    original = RidgeBudget._entries

    def reserve(self, count):
        entries.append(count)
        return original(self, count)

    monkeypatch.setattr(RidgeBudget, "_entries", reserve)
    module._minimum_preflight(coarse, _Arithmetic(coarse.reduction.model.budget))
    small = entries[-1]
    module._minimum_preflight(refined, _Arithmetic(refined.reduction.model.budget))
    assert entries[-1] > small


def test_failed_exact_proof_arithmetic_is_inconclusive(monkeypatch):
    minimum = _minimum()

    def exhausted(*args):
        raise ValueError("test coefficient-bit exhaustion")

    monkeypatch.setattr(_Arithmetic, "mul", exhausted)
    assert certify_gaussian_polyhedral_unique(minimum) is None


def test_a_coarse_budget_exit_can_still_prove_exact_uniqueness():
    minimum = _minimum(boxes=1)
    assert minimum.status == "box_budget" and minimum.gap_upper > minimum.value_tolerance
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None and result.method == "strict_convexity"


@pytest.mark.parametrize("backend", ["libm_fallback", "unknown"])
def test_conditional_or_unknown_cached_provenance_cannot_be_upgraded(backend):
    minimum = _minimum()
    minimum.certificate["meta"]["transcend_backend"] = backend
    minimum.certificate["honesty"]["unconditional_transcendentals"] = False
    changed = seal_certificate(minimum.certificate)
    minimum.certificate.clear()
    minimum.certificate.update(changed)
    assert verify_certificate_digest(minimum.certificate)
    assert not certificate_is_unconditional(minimum.certificate)
    assert certify_gaussian_polyhedral_unique(minimum) is None
