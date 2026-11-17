# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent target derivatives, strict-face refusals and proof binding."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    seal_certificate,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_reduced_minimum import (
    certify_gaussian_polyhedral_reduced_minimum,
)
from omnibias.core.verified.gaussian_polyhedral_reduced_sensitivity import (
    GaussianPolyhedralReducedSensitivity,
    certify_gaussian_polyhedral_reduced_sensitivity,
)
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    compile_gaussian_polyhedral_reduction,
)
from omnibias.core.verified.gaussian_polyhedral_unique import certify_gaussian_polyhedral_unique


def _unique(model, inequalities, bounds, target, *, boxes=63):
    pytest.importorskip("mpmath")
    reduction = compile_gaussian_polyhedral_reduction(model, inequalities, bounds, target)
    assert reduction is not None
    minimum = certify_gaussian_polyhedral_reduced_minimum(reduction, max_boxes_per_piece=boxes)
    assert minimum is not None
    result = certify_gaussian_polyhedral_unique(minimum)
    assert result is not None
    return result


def _scalar(*, nu=1, target=0, inequalities=(), bounds=(), boxes=63):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=nu)
    return _unique(model, inequalities, bounds, (target,), boxes=boxes)


def _coupled(*, gauge=0, reverse=False, inactive=False):
    rows, bounds = ((-1, 1, 1), (-1, -1, -2)), (1, -3)
    if reverse:
        rows, bounds = rows[::-1], bounds[::-1]
    if inactive:
        rows, bounds = rows+((0, 0, 1),), bounds+(3,)
    model = GaussianHeatMap(((-1, 0, 0), (1, 0, 0)), (gauge, gauge), viscosity=F(1, 4))
    unique = _unique(model, rows, bounds, (-7, 0, 0))
    index = next(i for i, piece in enumerate(unique.minimum.reduction.pieces) if piece.active_indices == (0, 1))
    result = certify_gaussian_polyhedral_reduced_sensitivity(unique, index)
    assert result is not None
    return result


def _contains(interval, value):
    return F(interval.lo) <= value <= F(interval.hi)


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


@pytest.mark.parametrize("nu", [F(3, 4), F(1), F(3)])
def test_unconstrained_zero_target_derivative_matches_original_posterior_variance(nu):
    unique = _scalar(nu=nu)
    result = certify_gaussian_polyhedral_reduced_sensitivity(unique, 0)
    assert result is not None
    # At x=0 the original centers ±1 have posterior variance exactly one.
    derivative = 1/(1-F(1, 2)/nu)
    assert _contains(result.target_jacobian[0][0], derivative)
    assert result.base_jacobian == ((F(0),),)
    assert result.rank_one_numerator == ((2*nu,),)
    assert result.point_slope == (2*nu,)
    assert result.scalar_curvature == 2*nu
    assert result.multipliers == result.slacks == ()


@pytest.mark.parametrize("nu,target", [(F(1, 4), F(1, 10)), (F(1, 2), F(1, 10)), (F(1), F(-1, 7))])
def test_nonsymmetric_scalar_derivative_contains_independent_original_kkt_derivative(nu, target):
    mp = pytest.importorskip("mpmath")
    result = certify_gaussian_polyhedral_reduced_sensitivity(_scalar(nu=nu, target=target, boxes=255), 0)
    assert result is not None
    with mp.workdps(90):
        visc, wanted = _mp(mp, nu), _mp(mp, target)

        def velocity(x):
            left, right = mp.exp(-(x+1)**2/(4*visc)), mp.exp(-(x-1)**2/(4*visc))
            return x-(right-left)/(left+right)

        point = mp.findroot(lambda x: velocity(x)-wanted, (mp.mpf(".6"), mp.mpf("1.5")) if target > 0 else (-1, -.1))
        derivative = 1/mp.diff(velocity, point)
        interval = result.target_jacobian[0][0]
        assert _mp(mp, interval.lo) <= derivative <= _mp(mp, interval.hi)


def test_ambient_indefinite_stable_face_has_exact_positive_restricted_response():
    result = _coupled(inactive=True)
    vector = (F(1), F(3), F(-2))
    exact = tuple(tuple(left*right/12 for right in vector) for left in vector)
    assert result.unique.minimum.reduction.model.contraction_bound == 2
    # Original H=diag(-1,1,1); along v=(1,3,-2), v.H.v=12>0.
    assert -vector[0]**2+vector[1]**2+vector[2]**2 == 12
    assert result.scalar_curvature == 7
    assert result.point_slope == (F(1, 2), F(3, 2), F(-1))
    assert all(all(value == 0 for value in row) for row in result.tangent_projector)
    for row, expected in zip(result.target_jacobian, exact, strict=True):
        assert all(_contains(interval, value) for interval, value in zip(row, expected, strict=True))
    assert _contains(result.denominator, F(6)) and result.denominator.lo > 0
    assert _contains(result.multipliers[0], 4) and _contains(result.multipliers[1], 3)
    assert result.multipliers[2].lo == result.multipliers[2].hi == 0
    assert result.slacks[0].lo == result.slacks[0].hi == result.slacks[1].lo == result.slacks[1].hi == 0
    assert _contains(result.slacks[2], 1) and result.slacks[2].lo > 0


def test_exact_gram_projector_and_slope_identities():
    result = _coupled()
    b, inverse = result.projected_constraints, result.gram_inverse
    gram = tuple(tuple(sum(x*y for x, y in zip(left, right, strict=True)) for right in b) for left in b)
    assert tuple(tuple(sum(gram[i][k]*inverse[k][j] for k in range(2)) for j in range(2)) for i in range(2)) == ((1, 0), (0, 1))
    reduction = result.unique.minimum.reduction
    for index in result.active_indices:
        row = reduction.inequalities[index]
        assert sum(x*y for x, y in zip(row, result.point_slope, strict=True)) == 0
        assert all(sum(row[i]*result.base_jacobian[i][j] for i in range(3)) == 0 for j in range(3))
        assert all(sum(row[i]*result.rank_one_numerator[i][j] for i in range(3)) == 0 for j in range(3))


@pytest.mark.parametrize("cotangent", [(0, 0, 0), (-1, 1, 1), (-1, -1, -2), (-2*10**400, 0, -10**400)])
def test_factored_vjp_preserves_exact_zero_normal_directions(cotangent):
    result = _coupled()
    assert all(interval.lo == interval.hi == 0 for interval in result.vjp(cotangent))


def test_vjp_needs_no_new_solve_or_transcendental_backend(monkeypatch):
    result = _coupled()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduced_sensitivity")
    monkeypatch.setattr(module, "factor_positive_definite", lambda *_: pytest.fail("VJP refactorized geometry"))
    monkeypatch.setattr(module, "exp_iv", lambda *_: pytest.fail("VJP repeated transcendental evaluation"))
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    cotangent = (F(1, 3), F(2, 5), F(-1, 7))
    projection = sum(a*b for a, b in zip((1, 3, -2), cotangent, strict=True))/12
    assert all(_contains(interval, coordinate*projection) for interval, coordinate in zip(result.vjp(cotangent), (1, 3, -2), strict=True))


def test_factored_vjp_retains_nonzero_transverse_response_with_an_active_row():
    model = GaussianHeatMap(((-1, 0, 0), (1, 0, 0)), (0, 0), viscosity=1)
    unique = _unique(model, ((0, 1, 0),), (-1,), (0, 0, 0))
    index = next(i for i, piece in enumerate(unique.minimum.reduction.pieces) if piece.active_indices == (0,))
    result = certify_gaussian_polyhedral_reduced_sensitivity(unique, index)
    assert result is not None and result.base_jacobian == ((0, 0, 0), (0, 0, 0), (0, 0, 1))
    answer = result.vjp((F(1, 3), 10**400, F(2, 7)))
    assert all(_contains(interval, exact) for interval, exact in zip(answer, (F(2, 3), 0, F(2, 7)), strict=True))
    assert answer[1].lo == answer[1].hi == 0


def test_row_permutation_and_huge_common_bias_leave_primal_derivative_unchanged():
    first, second = _coupled(), _coupled(gauge=10**400, reverse=True)
    for name in ("target_jacobian", "base_jacobian", "rank_one_numerator", "point_slope", "scalar_curvature", "denominator"):
        assert getattr(first, name) == getattr(second, name)
    assert first.multipliers == second.multipliers[::-1]


def test_critical_unique_minimum_does_not_imply_finite_target_derivative():
    unique = _scalar(nu=F(1, 2), boxes=255)
    assert unique.method == "strict_convexity" and unique.curvature_lower_bound == 0
    assert certify_gaussian_polyhedral_reduced_sensitivity(unique, 0) is None


def test_singleton_unique_boundary_is_refused():
    unique = _scalar(inequalities=((1,), (-1,)), bounds=(F(1, 3), F(-1, 3)))
    assert unique.method == "singleton"
    assert certify_gaussian_polyhedral_reduced_sensitivity(unique, 0) is None


def test_coarse_hull_touching_validity_boundary_refuses_then_refinement_certifies():
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=1)
    coarse = _unique(model, ((1, -1),), (1,), (0, 0), boxes=1)
    assert coarse.parameter_hull[1] == F(1, 2)
    assert all(certify_gaussian_polyhedral_reduced_sensitivity(coarse, i) is None for i in range(2))
    refined = _unique(model, ((1, -1),), (1,), (0, 0), boxes=255)
    result = certify_gaussian_polyhedral_reduced_sensitivity(refined, 0)
    assert result is not None and result.active_indices == () and result.slacks[0].lo > 0


def test_zero_selected_multiplier_or_zero_inactive_slack_refuses():
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=1)
    unique = _unique(model, ((0, 1),), (0,), (0, 0))
    assert len(unique.minimum.reduction.pieces) == 2
    assert all(certify_gaussian_polyhedral_reduced_sensitivity(unique, i) is None for i in range(2))


def test_redundant_inactive_active_constraint_is_conservatively_refused():
    model = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), viscosity=1)
    unique = _unique(model, ((0, 1), (0, 2)), (-1, -2), (0, 0))
    assert all(certify_gaussian_polyhedral_reduced_sensitivity(unique, i) is None for i in range(len(unique.minimum.reduction.pieces)))


@pytest.mark.parametrize("strict", [False, True])
def test_success_and_backend_failure_preserve_strict_provenance(strict, monkeypatch):
    unique = _scalar()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    assert certify_gaussian_polyhedral_reduced_sensitivity(unique, 0) is not None
    assert transcend.strict_backend() is strict and not transcend.libm_fallback_used()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_polyhedral_reduced_sensitivity(unique, 0)
    assert transcend.strict_backend() is strict and not transcend.libm_fallback_used()


def test_sticky_history_refused_and_not_reset(monkeypatch):
    unique = _scalar()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_polyhedral_reduced_sensitivity(unique, 0)
    assert transcend.libm_fallback_used()


def test_arithmetic_failure_restores_strict_flag(monkeypatch):
    unique = _scalar()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduced_sensitivity")

    def fail(*_args):
        raise ArithmeticError("simulated overflow")

    monkeypatch.setattr(module, "exp_iv", fail)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    assert certify_gaussian_polyhedral_reduced_sensitivity(unique, 0) is None
    assert not transcend.strict_backend()


def test_nested_certificate_snapshots_and_frozen_record():
    result = _coupled()
    assert certificate_is_unconditional(result.certificate) and verify_certificate_digest(result.certificate)
    assert result.certificate["payload"]["unique_certificate"] == result.unique.certificate
    result.unique.certificate["payload"]["method"] = "damaged"
    assert verify_certificate_digest(result.certificate)
    with pytest.raises(ValueError, match="damaged"):
        result.vjp((1, 0, 0))
    with pytest.raises(FrozenInstanceError):
        result.piece_index = 0
    with pytest.raises(TypeError):
        GaussianPolyhedralReducedSensitivity()


@pytest.mark.parametrize("level", ["unique", "minimum", "reduction"])
def test_nested_digest_damage_refuses(level):
    unique = deepcopy(_scalar())
    obj = unique if level == "unique" else unique.minimum if level == "minimum" else unique.minimum.reduction
    obj.certificate["payload"]["damage"] = True
    assert certify_gaussian_polyhedral_reduced_sensitivity(unique, 0) is None


def test_resealed_malformed_payloads_and_broken_snapshots_refuse_cleanly():
    unique = _scalar()
    unique.certificate.update(seal_certificate({**unique.certificate, "payload": []}))
    assert certify_gaussian_polyhedral_reduced_sensitivity(unique, 0) is None
    result = _coupled()
    result.certificate.update(seal_certificate({**result.certificate, "payload": []}))
    with pytest.raises(ValueError, match="damaged"):
        result.vjp((1, 0, 0))
    unique = _scalar()
    payload = {**unique.certificate["payload"], "minimum_certificate": {}}
    unique.certificate.update(seal_certificate({**unique.certificate, "payload": payload}))
    assert certify_gaussian_polyhedral_reduced_sensitivity(unique, 0) is None


def test_sensitivity_preflight_reserves_nested_sources_and_matrices_together(monkeypatch):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=1, budget=RidgeBudget(max_matrix_entries=20_000))
    unique = _unique(model, (), (), (0,), boxes=1)
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduced_sensitivity")
    monkeypatch.setattr(module, "_unique_intact", lambda *_: pytest.fail("unreserved proof scan"))
    assert certify_gaussian_polyhedral_reduced_sensitivity(unique, 0) is None


@pytest.mark.parametrize("bad", [True, 1.5, "0"])
def test_bad_piece_index_type_raises(bad):
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_reduced_sensitivity(_scalar(), bad)


@pytest.mark.parametrize("bad", [-1, 1])
def test_piece_index_outside_cover_raises(bad):
    with pytest.raises(IndexError):
        certify_gaussian_polyhedral_reduced_sensitivity(_scalar(), bad)


@pytest.mark.parametrize("bad", [(1, 2), (True,), (1.0,)])
def test_invalid_vjp_cotangent_raises(bad):
    result = certify_gaussian_polyhedral_reduced_sensitivity(_scalar(), 0)
    assert result is not None
    with pytest.raises((TypeError, ValueError)):
        result.vjp(bad)


def test_wrong_input_record_type_raises():
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_reduced_sensitivity(object(), 0)
