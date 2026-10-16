# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact tangent derivatives, proposal rounding and matrix-free proof contracts."""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_sensitivity import (
    GaussianAffineTargetVJPCertificate,
    certify_gaussian_affine_target_vjp,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_sensitivity import certify_gaussian_target_vjp
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mean_map import MeanMap


def _axis(*, redundant=False, **kwargs):
    # At (0,1), the original Gaussian weights are equal. The ambient Hessian
    # has a negative eigenvalue, but the tangent inverse derivative is diag(2,0).
    model = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)), **kwargs)
    matrix, rhs = (((0, 1), (0, 2)), (1, 2)) if redundant else (((0, 1),), (1,))
    restriction = restrict_gaussian_heat(model, matrix, rhs)
    assert restriction is not None
    return restriction


def _oblique():
    # C=(1,2), rhs=0. Uniform posterior at zero gives q=5/32 and
    # S=(32/27)P, where P=[[4,-2],[-2,1]]/5, independently of the checker.
    model = GaussianHeatMap(((F(1, 2), F(-1, 4)), (F(-1, 2), F(1, 4))), (0, 0))
    restriction = restrict_gaussian_heat(model, ((1, 2),), (0,))
    assert restriction is not None
    return restriction


def _oblique_project(vector):
    x, y = map(F, vector)
    return (F(4, 5)*x-F(2, 5)*y, -F(2, 5)*x+F(1, 5)*y)


def _norm_squared(vector):
    return sum((F(value)**2 for value in vector), F(0))


def _assert_exact_bound(result, expected):
    assert result is not None
    error = tuple(center-value for center, value in zip(result.value, expected, strict=True))
    assert _norm_squared(error) <= result.error_bound**2
    for interval, value, distance, radius in zip(
        result.enclosure, expected, error, result.coordinate_error_bounds, strict=True,
    ):
        assert F(interval.lo) <= value <= F(interval.hi)
        assert abs(distance) <= radius
    assert result.error_bound == (result.base_error_bound+result.upstream_rounding_error_bound
                                  +result.proposal_rounding_error_bound)


@pytest.mark.parametrize("redundant", [False, True])
@pytest.mark.parametrize("proposal", [(0.0, 0.0), (2.0, 0.0), (-8.0, 1e100)])
def test_true_constrained_derivative_even_for_bad_proposals(redundant, proposal):
    restriction = _axis(redundant=redundant)
    assert restriction.source_model.contraction_bound == F(5, 2)
    assert restriction.model.contraction_bound == F(1, 2)
    result = certify_gaussian_affine_target_vjp(restriction, (0, 100), (0.0, 1.0), (1.0, 7.0), proposal)
    _assert_exact_bound(result, (F(2), F(0)))
    assert result.projected_target == (0, 0)
    assert result.projected_upstream == (1, 0)
    assert result.value == result.projected_proposal == (F(proposal[0]), 0)
    assert result.base_certificate is not None
    assert result.enclosure[1] == Interval.point(0)
    assert result.coordinate_error_bounds[1] == 0
    assert result.upstream_rounding_error_bound == result.proposal_rounding_error_bound == 0


def test_normal_parts_cancel_exactly_before_any_interval_arithmetic():
    restriction = _axis()
    results = [certify_gaussian_affine_target_vjp(
        restriction, (0, target), (0.0, 1.0), (1.0, normal), (2.0, -normal),
    ) for target, normal in ((0, 0.0), (10**300, 1e300))]
    first, second = results
    assert first is not None and second is not None
    for name in ("value", "projected_target", "projected_upstream", "projected_proposal",
                 "error_bound", "coordinate_error_bounds", "enclosure"):
        assert getattr(first, name) == getattr(second, name)
    assert first.upstream != second.upstream and first.proposal != second.proposal


@pytest.mark.parametrize("upstream,proposal", [((1.0, 0.0), (128/135, -64/135)),
    ((0.3, -0.2), (0.3, -0.2)), ((-0.75, 0.5), (0.0, 0.0))])
def test_oblique_projection_and_both_rounding_corrections(upstream, proposal):
    result = certify_gaussian_affine_target_vjp(_oblique(), (0, 0), (0.0, 0.0), upstream, proposal)
    pg = _oblique_project(upstream)
    expected = tuple(F(32, 27)*value for value in pg)
    _assert_exact_bound(result, expected)
    assert result.value == result.projected_proposal == _oblique_project(proposal)
    assert result.projected_upstream == pg
    assert result.value[0]+2*result.value[1] == 0
    rounded_g = tuple(F(float(value)) for value in pg)
    projected_rounded_g = _oblique_project(rounded_g)
    delta_g = tuple(a-b for a, b in zip(projected_rounded_g, pg, strict=True))
    completed = tuple(a+g-pg for a, g, pg in zip(result.value, rounded_g, projected_rounded_g, strict=True))
    rounded_z = tuple(F(float(value)) for value in completed)
    projected_z = _oblique_project(rounded_z)
    delta_a = tuple(a-b for a, b in zip(projected_z, result.value, strict=True))
    assert result.upstream_rounding_error_bound**2 >= F(32, 27)**2*_norm_squared(delta_g)
    assert result.proposal_rounding_error_bound**2 >= _norm_squared(delta_a)
    assert result.base_certificate.proposal == tuple(map(float, completed))
    for diagonal, radius in zip((F(4, 5), F(1, 5)), result.coordinate_error_bounds, strict=True):
        assert radius**2 >= diagonal*result.error_bound**2
        assert radius <= result.error_bound


def test_oblique_fixture_exercises_nonzero_rounding_terms():
    result = certify_gaussian_affine_target_vjp(_oblique(), (0, 0), (0.0, 0.0), (1.0, 0.0), (0.3, -0.2))
    assert result is not None
    assert result.upstream_rounding_error_bound > 0
    assert result.proposal_rounding_error_bound > 0


def test_exact_null_is_not_confused_with_one_ulp_near_null(monkeypatch):
    restriction = _oblique()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    normal = certify_gaussian_affine_target_vjp(restriction, (0, 0), (100.0, -50.0), (1.0, 2.0), (17.0, 23.0))
    assert normal is not None and normal.base_certificate is None
    assert normal.value == (0, 0) and normal.error_bound == 0
    assert normal.projected_proposal != (0, 0)
    assert normal.enclosure == (Interval.point(0),)*2
    # This small tangent component is exact binary64 data, not numerical noise
    # the checker may discard. It must take the rigorous posterior path.
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_affine_target_vjp(
            restriction, (0, 0), (0.0, 0.0), (1.0, math.nextafter(2.0, math.inf)), (0.0, 0.0),
        )


def test_affine_oblique_derivative_is_exact_nonbinary_without_backend(monkeypatch):
    source = GaussianHeatMap(((10**200, -10**200),), (10**400,), time=F(7, 3))
    restriction = restrict_gaussian_heat(source, ((1, 2),), (F(1, 3),))
    assert restriction is not None and restriction.model.contraction_bound == 0
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = certify_gaussian_affine_target_vjp(
        restriction, (10**300, F(2, 3)), (-1e300, 1e300), (1.0, 0.0), (-99.0, 12.0),
    )
    _assert_exact_bound(result, (F(28, 15), F(-14, 15)))
    assert result.value != result.projected_proposal
    assert result.error_bound == 0 and result.base_certificate is None
    assert result.enclosure[0].lo < result.enclosure[0].hi  # Binary64 enclosure of an exact rational.
    assert result.certificate["meta"]["transcend_backend"] == "not_used"
    assert transcend.libm_fallback_used() and not transcend.strict_backend()


def test_full_rank_constraint_has_zero_derivative_without_backend(monkeypatch):
    source = GaussianHeatMap(((-100, -100), (100, 100)), (0, 0))
    restriction = restrict_gaussian_heat(source, ((1, 0), (0, 1), (1, 1)), (F(1, 3), F(1, 7), F(10, 21)))
    assert restriction is not None and restriction.tangent_dimension == 0
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = certify_gaussian_affine_target_vjp(restriction, (7, -3), (0, 0), (1e300, -1e300), (1e300, 1e300))
    _assert_exact_bound(result, (F(0), F(0)))
    assert result.base_certificate is None and result.error_bound == 0


@pytest.mark.parametrize("time", [F(1, 2), F(1, 4)])
def test_noncontracting_regime_refuses_even_exact_normal_upstream(time, monkeypatch):
    restriction = _axis(time=time)
    assert restriction.model.contraction_bound >= 1
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    assert certify_gaussian_affine_target_vjp(restriction, (0, 0), (0, 1), (0, 1), (0, 0)) is None


def test_rank_zero_reproduces_unconstrained_checker_without_rounding_penalties():
    source = GaussianHeatMap(((-1,), (1,)), (0, 0), time=2)
    restriction = restrict_gaussian_heat(source, (), ())
    result = certify_gaussian_affine_target_vjp(restriction, (0,), (0.0,), (1.0,), (8/3,))
    base = certify_gaussian_target_vjp(source, (0,), (0.0,), (1.0,), (8/3,))
    _assert_exact_bound(result, (F(8, 3),))
    assert base is not None and result.base_certificate is not None
    assert result.error_bound == F(base.error_bound)
    assert result.upstream_rounding_error_bound == result.proposal_rounding_error_bound == 0
    assert result.value == (F(8/3),)


def test_root_proposal_is_untrusted_and_root_uncertainty_is_preserved():
    restriction = _axis()
    near = certify_gaussian_affine_target_vjp(restriction, (0, 0), (0, 1), (1, 0), (2, 0))
    coarse = certify_gaussian_affine_target_vjp(restriction, (0, 0), (1, -2), (1, 0), (2, 0))
    _assert_exact_bound(coarse, (F(2), F(0)))
    assert near is not None and coarse.base_certificate is not None
    assert coarse.base_certificate.root_certificate.error_bound > 1
    assert coarse.error_bound > near.error_bound*1000


def test_certification_uses_no_iteration_dense_covariance_or_factorization(monkeypatch):
    restriction = _oblique()

    def fail(*args, **kwargs):
        pytest.fail("derivative checker attempted an inverse solve, factorization or dense covariance")

    for module_name, names in (
        ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat",)),
        ("omnibias.core.verified.inverse_gaussian_affine", ("inverse_gaussian_affine", "inverse_gaussian_heat")),
        ("omnibias.core._rational_linear", ("factor_positive_definite", "solve_affine")),
        ("omnibias.core.verified.linalg", ("_float_inverse", "_symmetric_ldlt_solve", "interval_solve")),
    ):
        module = import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(MeanMap, "jacobian", fail)
    monkeypatch.setattr(GaussianHeatMap, "scaled_jacobian", fail)
    monkeypatch.setattr(GaussianHeatMap, "float_scaled_fixed_point", fail)
    result = certify_gaussian_affine_target_vjp(restriction, (0, 0), (0, 0), (1, 0), (128/135, -64/135))
    _assert_exact_bound(result, (F(128, 135), F(-64, 135)))


def test_factory_only_frozen_record_snapshots_inputs_and_dependencies():
    with pytest.raises(TypeError):
        GaussianAffineTargetVJPCertificate()
    restriction = _axis()
    target, point, upstream, proposal = [0, 0], [0.0, 1.0], [1.0, 0.0], [2.0, 0.0]
    result = certify_gaussian_affine_target_vjp(restriction, target, point, upstream, proposal)
    assert result is not None and result.base_certificate is not None
    assert verify_certificate_digest(result.certificate)
    assert certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    with pytest.raises(FrozenInstanceError):
        result.error_bound = F(0)
    for values in (target, point, upstream, proposal):
        values[0] = 99
    assert result.target == (0, 0) and result.scaled_point == (0.0, 1.0)
    assert result.upstream == (1.0, 0.0) and result.proposal == (2.0, 0.0)
    saved = deepcopy(result.certificate)
    result.base_certificate.root_certificate.certificate["payload"]["damaged"] = object()
    result.base_certificate.certificate["payload"]["damaged"] = object()
    restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert certify_gaussian_affine_target_vjp(restriction, (0, 0), (0, 1), (1, 0), (2, 0)) is None
    result.certificate["payload"]["projected_upstream"][0] = [99, 1]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("damage", ["adjoint", "root"])
def test_damaged_composed_certificate_refuses(damage, monkeypatch):
    restriction = _axis()
    base = certify_gaussian_target_vjp(restriction.model, (0, 0), (0, 1), (1, 0), (2, 0))
    assert base is not None
    changed = base.certificate if damage == "adjoint" else base.root_certificate.certificate
    changed["payload"]["damaged"] = object()
    module = import_module("omnibias.core.verified.gaussian_affine_sensitivity")
    monkeypatch.setattr(module, "certify_gaussian_target_vjp", lambda *args, **kwargs: base)
    assert certify_gaussian_affine_target_vjp(restriction, (0, 0), (0, 1), (1, 0), (2, 0)) is None


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_rigorous_proof_preserves_sticky_history_and_strict_state(monkeypatch, sticky, strict):
    pytest.importorskip("mpmath")
    restriction = _axis()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    result = certify_gaussian_affine_target_vjp(restriction, (0, 0), (0, 1), (1, 0), (2, 0))
    assert result is not None and certificate_is_unconditional(result.certificate)
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert transcend.libm_fallback_used() is sticky and transcend.strict_backend() is strict


def test_missing_backend_is_not_a_conditional_certificate(monkeypatch):
    restriction = _axis()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_affine_target_vjp(restriction, (0, 0), (0, 1), (1, 0), (2, 0))
    assert not transcend.strict_backend()
    # Strict sealing applies even to a structurally zero exact shortcut.
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_affine_target_vjp(restriction, (0, 0), (0, 1), (0, 1), (2, 0))
    assert transcend.strict_backend()


@pytest.mark.parametrize("index", [1, 2, 3, 4])
@pytest.mark.parametrize("bad,error", [(None, TypeError), ((), ValueError), ((0,), ValueError),
    ((0, 0, 0), ValueError), ((True, 0), TypeError), ((complex(1), 0), TypeError)])
def test_invalid_vector_shapes_and_types_raise(index, bad, error):
    arguments = [_axis(), (0, 0), (0, 1), (1, 0), (2, 0)]
    arguments[index] = bad
    with pytest.raises(error):
        certify_gaussian_affine_target_vjp(*arguments)


@pytest.mark.parametrize("index", [2, 3, 4])
@pytest.mark.parametrize("bad,error", [(F(1), TypeError), (math.inf, ValueError),
    (math.nan, ValueError), (2**53+1, ValueError), (10**400, ValueError)])
def test_binary64_contract_rejects_inexact_or_nonfinite_coordinates(index, bad, error):
    arguments = [_axis(), (0, 0), (0, 1), (1, 0), (2, 0)]
    arguments[index] = (bad, 0)
    with pytest.raises(error):
        certify_gaussian_affine_target_vjp(*arguments)


def test_invalid_exact_target_and_restriction_raise():
    with pytest.raises(TypeError):
        certify_gaussian_affine_target_vjp(None, (0,), (0,), (1,), (1,))
    with pytest.raises(TypeError):
        certify_gaussian_affine_target_vjp(_axis(), (0.0, 0), (0, 1), (1, 0), (2, 0))
    # Validation occurs before an otherwise applicable exact-null shortcut.
    with pytest.raises(ValueError):
        certify_gaussian_affine_target_vjp(_axis(), (0, 0), (math.inf, 1), (0, 1), (2, 0))


@pytest.mark.parametrize("limits,upstream", [
    (RidgeBudget(max_matrix_entries=1500), 1),
    (RidgeBudget(max_elimination_work=6000), 1),
    (RidgeBudget(max_coefficient_bits=64), 1e100),
])
def test_resource_failure_is_inconclusive_before_any_posterior(limits, upstream, monkeypatch):
    source = GaussianHeatMap(((0,), (0,)), (0, 1), budget=limits)
    restriction = restrict_gaussian_heat(source, (), ())
    assert restriction is not None
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    assert certify_gaussian_affine_target_vjp(restriction, (0,), (0,), (upstream,), (0,)) is None


def test_extreme_finite_arithmetic_refuses_without_false_certificate():
    restriction = _axis()
    assert certify_gaussian_affine_target_vjp(restriction, (0, 0), (0, 1), (1.7e308, 0), (0, 0)) is None
    source = GaussianHeatMap(((0,),), (0,), time=10**400)
    affine = restrict_gaussian_heat(source, (), ())
    assert affine is not None
    assert certify_gaussian_affine_target_vjp(affine, (0,), (0,), (1,), (0,)) is None
