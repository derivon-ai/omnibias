# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Constrained inverse, feasible-point certificates and forced evolution."""

from copy import deepcopy
from fractions import Fraction as F

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_evolution import (
    GaussianAffineEvolution,
    certify_gaussian_affine_evolution,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import (
    certify_gaussian_affine_inverse,
    inverse_gaussian_affine,
)


def _restriction(**kwargs):
    model = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)), **kwargs)
    restriction = restrict_gaussian_heat(model, ((0, 1),), (1,))
    assert restriction is not None
    return restriction


def test_constraints_remove_actual_ambient_nonconvexity_and_certify_global_minimum():
    restriction = _restriction()
    assert restriction.source_model.contraction_bound == F(5, 2)
    assert restriction.model.contraction_bound == F(1, 2)
    # At X=(0,1), the original posterior is uniform and its covariance is
    # [[1,2],[2,4]]. The Hessian's center-direction eigenvalue is -3/2.
    assert 1-F(5, 2) < 0
    result = inverse_gaussian_affine(restriction, (0, 0))
    assert result.status == "certified" and result.tolerance_met
    certificate = result.certificate
    assert certificate is not None
    assert certificate.feasible_point == (F(0), F(1))
    assert certificate.enclosure[0].lo <= 0 <= certificate.enclosure[0].hi
    assert certificate.enclosure[1] == Interval.point(1)
    assert certificate.objective_gap_bound == F(certificate.error_bound)**2/2
    assert verify_certificate_digest(certificate.certificate)


def test_untrusted_nonfeasible_proposal_becomes_exactly_feasible_with_gap_bound():
    restriction = _restriction()
    certificate = certify_gaussian_affine_inverse(restriction, (F(1, 8), 0), (3.0, -20.0))
    assert certificate is not None
    assert certificate.feasible_point == (F(3), F(1))
    assert certificate.root_certificate is not None
    assert certificate.error_bound == certificate.root_certificate.error_bound
    assert certificate.objective_gap_bound >= 0
    # The enclosing box describes the optimizer, not all feasible points.
    assert certificate.enclosure[0].lo <= certificate.enclosure[0].hi


def test_normal_target_component_does_not_change_optimizer_or_gap():
    restriction = _restriction()
    a = inverse_gaussian_affine(restriction, (F(1, 8), 0)).certificate
    b = inverse_gaussian_affine(restriction, (F(1, 8), 10**100)).certificate
    assert a is not None and b is not None
    assert a.target != b.target and a.projected_target == b.projected_target
    assert a.enclosure == b.enclosure and a.feasible_point == b.feasible_point
    assert a.objective_gap_bound == b.objective_gap_bound


def test_forced_time_derivative_and_tangent_jacobian_match_exact_counterexample():
    evolution = certify_gaussian_affine_evolution(_restriction(), (Interval.point(0),)*2)
    assert evolution is not None
    assert evolution.correction_norm_bound == 3
    for interval, exact in zip(evolution.time_derivative, (-2, 0), strict=True):
        assert interval.lo <= exact <= interval.hi
        assert interval.hi-interval.lo < 1e-10
    assert evolution.time_derivative[1] == Interval.point(0)
    assert evolution.normal_cost_covariance[0].lo <= -4 <= evolution.normal_cost_covariance[0].hi
    assert evolution.normal_forcing[0].lo <= -2 <= evolution.normal_forcing[0].hi
    assert evolution.diffusion[0].lo <= 0 <= evolution.diffusion[0].hi
    assert evolution.base_evolution.time_derivative[0].lo <= 0 <= evolution.base_evolution.time_derivative[0].hi
    assert evolution.time_derivative[0].hi < evolution.base_evolution.time_derivative[0].lo
    for row, expected in zip(evolution.target_jacobian, ((2, 0), (0, 0)), strict=True):
        for value, exact in zip(row, expected, strict=True):
            assert value.lo <= exact <= value.hi
    assert evolution.target_jacobian[1] == (Interval.point(0),)*2


def test_constant_normal_costs_remove_forcing_exactly():
    model = GaussianHeatMap(((-4, -1), (4, 1)), (0, F(1, 3)))
    restriction = restrict_gaussian_heat(model, ((1, 0),), (0,))
    assert restriction is not None
    assert restriction.normal_cost_differences == (0, 0)
    result = certify_gaussian_affine_evolution(restriction, (Interval.point(0), Interval.point(0.25)))
    assert result is not None
    assert result.normal_forcing == result.normal_cost_covariance == (Interval.point(0),)*2
    assert result.diffusion[0] == result.time_derivative[0] == Interval.point(0)
    assert result.correction_norm_bound == result.base_evolution.diffusion_norm_bound


def test_oblique_projector_uses_both_sides_of_target_jacobian():
    model = GaussianHeatMap(((0, 0), (1, 1)), (0, F(1, 4)), time=4)
    restriction = restrict_gaussian_heat(model, ((1, 2),), (3,))
    assert restriction is not None
    result = certify_gaussian_affine_evolution(restriction, (Interval.point(0.125), Interval.point(-0.25)))
    assert result is not None
    # Tangent rank is one; J annihilates the normal (1,2) and is symmetric.
    for row in result.target_jacobian:
        normal_action = row[0]+2*row[1]
        assert normal_action.lo <= 0 <= normal_action.hi
        assert normal_action.hi-normal_action.lo < 1e-8
    assert result.target_jacobian[0][1].intersect(result.target_jacobian[1][0])
    assert result.target_jacobian[0][1].hi < 0


def test_target_box_covers_point_derivatives_and_preserves_axis_constraints():
    restriction = _restriction()
    broad = certify_gaussian_affine_evolution(restriction, (Interval(-0.125, 0.125), Interval(-10, 10)))
    assert broad is not None
    assert broad.projected_target_box[1] == Interval.point(0)
    for target in (-0.125, 0, 0.125):
        point = certify_gaussian_affine_evolution(restriction, (Interval.point(target), Interval.point(100)))
        assert point is not None
        for full, narrow in zip(broad.time_derivative, point.time_derivative, strict=True):
            assert full.lo <= narrow.lo <= narrow.hi <= full.hi


@pytest.mark.parametrize("full_rank", (False, True))
def test_exact_affine_restriction_uses_no_backend(monkeypatch, full_rank):
    model = GaussianHeatMap(((2, -3), (2, -3)), (0, 10**400), time=F(3, 2))
    constraints, rhs = (((1, 0), (0, 1)), (1, -1)) if full_rank else (((1, 2),), (3,))
    restriction = restrict_gaussian_heat(model, constraints, rhs)
    assert restriction is not None
    target = (F(1, 4), F(-1, 8))
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = inverse_gaussian_affine(restriction, target)
    certificate = certify_gaussian_affine_inverse(restriction, target, (1e100, -1e100))
    assert result.certificate is not None and certificate is not None
    assert result.iterations == 0 and result.tolerance_met
    assert certificate.error_bound == certificate.objective_gap_bound == 0
    assert certificate.root_certificate is None
    assert certificate.feasible_point == result.certificate.feasible_point
    for row, expected in zip(constraints, rhs, strict=True):
        assert sum(c*x for c, x in zip(row, certificate.feasible_point, strict=True)) == expected
    evolution = certify_gaussian_affine_evolution(restriction, tuple(Interval.point(float(v)) for v in target))
    assert evolution is not None
    assert evolution.normal_forcing == evolution.diffusion == (Interval.point(0),)*2
    assert evolution.correction_norm_bound == 0
    for row, exact in zip(evolution.target_jacobian, restriction.projector, strict=True):
        for interval, value in zip(row, exact, strict=True):
            assert F(interval.lo) <= model.time*value <= F(interval.hi)
    if full_rank:
        assert certificate.feasible_point == (1, -1)
        assert evolution.time_derivative == (Interval.point(0),)*2


def test_empty_constraints_recover_original_inverse_and_evolution():
    model = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=2)
    restriction = restrict_gaussian_heat(model, (), ())
    assert restriction is not None
    result = inverse_gaussian_affine(restriction, (F(1, 4),))
    assert result.certificate is not None
    evolution = certify_gaussian_affine_evolution(restriction, (Interval.point(0.25),))
    assert evolution is not None
    assert evolution.normal_forcing == (Interval.point(0),)
    assert evolution.time_derivative == evolution.base_evolution.time_derivative


def test_coarse_valid_certificate_is_distinct_from_met_tolerance():
    result = inverse_gaussian_affine(_restriction(), (F(1, 8), 0), max_iterations=0)
    assert result.status == "certified" and not result.tolerance_met
    assert result.certificate is not None


def test_insufficient_projected_contraction_does_not_claim_nonexistence():
    restriction = _restriction(time=F(1, 2))
    assert restriction.model.contraction_bound == 1
    result = inverse_gaussian_affine(restriction, (0, 0))
    assert result.status == "outside_contraction_regime" and result.certificate is None
    assert certify_gaussian_affine_inverse(restriction, (0, 0), (0.0, 0.0)) is None
    assert certify_gaussian_affine_evolution(restriction, (Interval.point(0),)*2) is None


def test_certificate_snapshots_and_damaged_restriction_refusal():
    restriction = _restriction()
    result = inverse_gaussian_affine(restriction, (0, 0)).certificate
    evolution = certify_gaussian_affine_evolution(restriction, (Interval.point(0),)*2)
    assert result is not None and evolution is not None
    saved = deepcopy(result.certificate)
    restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert verify_certificate_digest(evolution.certificate)
    assert certify_gaussian_affine_inverse(restriction, (0, 0), (0.0, 0.0)) is None
    assert inverse_gaussian_affine(restriction, (0, 0)).status == "not_certified"
    assert certify_gaussian_affine_evolution(restriction, (Interval.point(0),)*2) is None


def test_missing_backend_raises_and_sticky_history_is_preserved(monkeypatch):
    restriction = _restriction()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = inverse_gaussian_affine(restriction, (0, 0)).certificate
    evolution = certify_gaussian_affine_evolution(restriction, (Interval.point(0),)*2)
    assert result is not None and evolution is not None and transcend.libm_fallback_used()
    assert result.certificate["meta"]["transcend_backend"] == evolution.certificate["meta"]["transcend_backend"] == "mpmath"
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    with pytest.raises(RuntimeError):
        inverse_gaussian_affine(restriction, (0, 0))
    with pytest.raises(RuntimeError):
        certify_gaussian_affine_evolution(restriction, (Interval.point(0),)*2)


@pytest.mark.parametrize("kwargs,error", [
    ({"max_iterations": True}, TypeError), ({"max_iterations": -1}, ValueError),
    ({"absolute_tolerance": 0}, ValueError), ({"absolute_tolerance": float("nan")}, ValueError),
])
def test_invalid_controls_raise(kwargs, error):
    with pytest.raises(error):
        inverse_gaussian_affine(_restriction(), (0, 0), **kwargs)
    with pytest.raises(error):
        certify_gaussian_affine_evolution(_restriction(), (Interval.point(0),)*2, **kwargs)


@pytest.mark.parametrize("target,error", [(None, TypeError), ((0.0, 0), TypeError), ((), ValueError)])
def test_invalid_exact_target_data_raise(target, error):
    with pytest.raises(error):
        inverse_gaussian_affine(_restriction(), target)
    with pytest.raises(error):
        certify_gaussian_affine_inverse(_restriction(), target, (0.0, 0.0))


def test_invalid_factories_and_nonfinite_proposals_raise():
    with pytest.raises(TypeError):
        GaussianAffineEvolution()
    for function, args in ((inverse_gaussian_affine, ((0, 0),)),
                           (certify_gaussian_affine_inverse, ((0, 0), (0.0, 0.0))),
                           (certify_gaussian_affine_evolution, ((Interval.point(0),)*2,))):
        with pytest.raises(TypeError):
            function(None, *args)
    with pytest.raises(ValueError):
        certify_gaussian_affine_inverse(_restriction(), (0, 0), (float("nan"), 0.0))


def test_aggregate_work_budget_refuses_composed_evolution():
    restriction = _restriction(budget=RidgeBudget(max_elimination_work=50000))
    assert certify_gaussian_affine_evolution(restriction, (Interval.point(0),)*2) is None
