# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form target Jacobians, tangent tails and cached-query contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from math import isfinite

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
from omnibias.core.verified.gaussian_entropy_face import certify_gaussian_entropy_face
from omnibias.core.verified.gaussian_entropy_residual import certify_gaussian_entropy_residual
from omnibias.core.verified.gaussian_entropy_sensitivity import (
    GaussianEntropySensitivity,
    certify_gaussian_entropy_sensitivity,
)
from omnibias.core.verified.gaussian_entropy_solver import solve_gaussian_entropy
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _face(*, centers=((-1,), (1,)), biases=(0, 0), viscosity=1, time=1, target=None,
          rows=(), bounds=(), reference=None, solve=False, budget=None):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time, budget=budget)
    projector = compile_polyhedral_projector(model.dimension, rows, bounds, budget=budget)
    assert projector is not None
    target = (0,)*model.dimension if target is None else target
    domain = certify_gaussian_posterior_domain(model, projector, target)
    assert domain is not None
    unique = certify_gaussian_entropy_unique(domain, reference=(0,)*model.dimension if reference is None else reference)
    assert unique is not None
    if solve:
        result = solve_gaussian_entropy(unique, tolerance=1e-12, max_iterations=32)
        assert result is not None
        residual = result.residual
    else:
        residual = certify_gaussian_entropy_residual(unique, (F(1, model.mode_count),)*model.mode_count)
        assert residual is not None
    face = certify_gaussian_entropy_face(residual)
    assert face is not None
    return face


def _sensitivity(face=None, **kwargs):
    result = certify_gaussian_entropy_sensitivity(_face() if face is None else face, **kwargs)
    assert result is not None
    return result


def _contains(interval, value):
    return F(interval.lo) <= value <= F(interval.hi)


def _matrix_contains(matrix, expected):
    assert len(matrix) == len(expected)
    assert all(_contains(interval, value) for row, exact in zip(matrix, expected, strict=True)
               for interval, value in zip(row, exact, strict=True))


@pytest.mark.parametrize("terms", [1, 2, 4, 16])
def test_scalar_uniform_two_center_jacobian_and_geometric_tail(terms):
    result = _sensitivity(terms=terms)
    assert result.terms == terms
    assert result.contraction_bound == F(1, 2)
    assert result.tail_norm_bound >= 2*F(1, 2)**terms > 0
    _matrix_contains(result.projected_covariance, ((F(1),),))
    _matrix_contains(result.polynomial, ((2*(1-F(1, 2)**terms),),))
    _matrix_contains(result.target_jacobian, ((F(2),),))
    assert _contains(result.vjp((F(3, 7),))[0], F(6, 7))


def test_larger_term_counts_reduce_the_geometric_tail_and_pointwise_width():
    face = _face()
    certificates = tuple(_sensitivity(face, terms=terms) for terms in (2, 8, 16))
    assert certificates[0].tail_norm_bound > certificates[1].tail_norm_bound > certificates[2].tail_norm_bound > 0
    widths = tuple(result.target_jacobian[0][0].hi-result.target_jacobian[0][0].lo for result in certificates)
    assert widths[0] > widths[1] > widths[2]


def test_nonunit_age_scales_both_jacobian_and_tail_correctly():
    result = _sensitivity(_face(time=2), terms=8)
    assert result.contraction_bound == F(1, 4)
    assert result.tail_norm_bound >= F(8, 3)*F(1, 4)**8
    _matrix_contains(result.target_jacobian, ((F(8, 3),),))


def test_coupled_symmetric_off_diagonal_response_matches_exact_inverse():
    face = _face(centers=((-1, -1), (1, 1)), viscosity=2)
    result = _sensitivity(face, terms=16)
    _matrix_contains(result.projected_covariance, ((F(1), F(1)), (F(1), F(1))))
    _matrix_contains(result.target_jacobian, ((F(3, 2), F(1, 2)), (F(1, 2), F(3, 2))))
    value = result.vjp((2, -1))
    assert _contains(value[0], F(5, 2)) and _contains(value[1], F(-1, 2))


def test_strict_active_face_retains_tangent_response_and_exact_normal_null():
    face = _face(centers=((-1, -1), (1, 1)), viscosity=2,
                 rows=((0, 1),), bounds=(0,), target=(0, 1))
    assert face.tangent_projector == ((F(1), F(0)), (F(0), F(0)))
    result = _sensitivity(face)
    _matrix_contains(result.projected_covariance, ((F(1), F(0)), (F(0), F(0))))
    _matrix_contains(result.target_jacobian, ((F(4, 3), F(0)), (F(0), F(0))))
    assert result.contraction_bound < F(1, 3)
    assert all(iv.lo == iv.hi == 0 for iv in result.vjp((0, 10**1000)))
    assert result.vjp((1, 10**1000)) == result.vjp((1, 0))


def test_opposite_and_redundant_equality_rows_preserve_oblique_normal_nulls():
    face = _face(centers=((-1, 0), (1, 0)), rows=((1, 1), (-1, -1), (2, 2)), bounds=(0, 0, 0))
    assert face.tangent_projector == ((F(1, 2), F(-1, 2)), (F(-1, 2), F(1, 2)))
    result = _sensitivity(face)
    exact = ((F(2, 3), F(-2, 3)), (F(-2, 3), F(2, 3)))
    _matrix_contains(result.target_jacobian, exact)
    assert all(iv.lo == iv.hi == 0 for iv in result.vjp((2**2500, 2**2500)))
    small = F(1, 2**100)
    near_normal = result.vjp((1, 1+small))
    assert _contains(near_normal[0], -F(2, 3)*small) and near_normal[0].hi < 0
    assert _contains(near_normal[1], F(2, 3)*small) and near_normal[1].lo > 0


def test_single_gaussian_affine_target_jacobian_is_time_times_projector():
    face = _face(centers=((F(1, 3), F(-1, 5)),), biases=(F(2, 7),), time=2,
                 rows=((1, 1), (-1, -1)), bounds=(F(1, 3), F(-1, 3)), target=(F(1, 7), F(2, 7)))
    result = _sensitivity(face, terms=1)
    assert result.contraction_bound == result.tail_norm_bound == 0
    _matrix_contains(result.target_jacobian, ((F(1), F(-1)), (F(-1), F(1))))
    assert all(iv.lo == iv.hi == 0 for iv in result.vjp((1, 1)))


def test_full_normal_singleton_has_zero_projected_covariance_and_jacobian():
    face = _face(centers=((-1, -1), (1, 1)), viscosity=2,
                 rows=((1, 0), (-1, 0), (0, 1), (0, -1)), bounds=(0, 0, 0, 0), target=(1, -2))
    result = _sensitivity(face, terms=1)
    assert result.contraction_bound == result.tail_norm_bound == 0
    assert all(iv.lo == iv.hi == 0 for row in result.projected_covariance for iv in row)
    assert all(iv.lo == iv.hi == 0 for row in result.target_jacobian for iv in row)
    assert all(iv.lo == iv.hi == 0 for iv in result.vjp((17, -23)))


def test_partial_face_annihilates_purely_normal_center_variation_before_moments():
    face = _face(centers=((0, -1), (0, 1)), rows=((0, 1),), bounds=(0,), target=(0, 1))
    result = _sensitivity(face, terms=1)
    assert face.residual.contraction_bound > 0
    assert result.contraction_bound == result.tail_norm_bound == 0
    assert all(iv.lo == iv.hi == 0 for row in result.projected_covariance for iv in row)
    _matrix_contains(result.target_jacobian, ((F(1), F(0)), (F(0), F(0))))


def test_nonconvex_q_four_original_model_has_certified_target_response():
    mp = pytest.importorskip("mpmath")
    face = _face(centers=((-2,), (2,)), biases=(0, -12), viscosity=F(1, 2), reference=(-2,), solve=True)
    assert face.residual.unique.domain.model.contraction_bound == 4
    result = _sensitivity(face, terms=4)
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-2*mp.tanh(2*x-6), (-2, mp.mpf("-1.9")))
        p = 1/(1+mp.exp(12-4*root))
        derivative = 1/(1-16*p*(1-p))
        bound = result.target_jacobian[0][0]
        assert mp.mpf(bound.lo) <= derivative <= mp.mpf(bound.hi)
    assert result.contraction_bound < F(1, 1000)


def test_underflowed_positive_covariance_does_not_become_a_false_zero_tail():
    face = _face(centers=((0,), (10,)), biases=(0, -1000), reference=(0,), solve=True)
    result = _sensitivity(face, terms=4)
    assert 0 < result.contraction_bound < F(1, 10**200)
    assert result.tail_norm_bound > 0
    true_geometric_bound = result.contraction_bound**4/(1-result.contraction_bound)
    assert result.tail_norm_bound >= true_geometric_bound
    assert all(isfinite(iv.lo) and isfinite(iv.hi) for row in result.target_jacobian for iv in row)


def test_proof_snapshot_and_factory_record_are_immutable():
    result = _sensitivity()
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    saved = deepcopy(result.certificate)
    with pytest.raises(TypeError):
        GaussianEntropySensitivity()
    with pytest.raises(FrozenInstanceError):
        result.terms = 0
    result.face.certificate["payload"]["type"] = "damaged"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    with pytest.raises(ValueError):
        result.vjp((0,))
    assert certify_gaussian_entropy_sensitivity(result.face) is None


@pytest.mark.parametrize("strict,sticky", [(False, False), (False, True), (True, False), (True, True)])
def test_cached_vjp_needs_no_backend_projection_or_solve(strict, sticky, monkeypatch):
    result = _sensitivity(_face(centers=((-1, -1), (1, 1)), viscosity=2))

    def fail(*args, **kwargs):
        pytest.fail("cached VJP performed a new solve or transcendental evaluation")

    for module_name in ("polyhedral_projection", "gaussian_entropy_residual", "gaussian_projection_minimum"):
        module = import_module("omnibias.core.verified."+module_name)
        monkeypatch.setattr(module, "_project_exact", fail)
    linear = import_module("omnibias.core._rational_linear")
    monkeypatch.setattr(linear, "solve_affine", fail)
    monkeypatch.setattr(linear, "factor_positive_definite", fail)
    for name in ("ln_iv", "exp_iv", "tanh_iv"):
        monkeypatch.setattr(transcend, name, fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    value = result.vjp((2, -1))
    assert _contains(value[0], F(5, 2)) and _contains(value[1], F(-1, 2))
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used() is sticky


@pytest.mark.parametrize("strict", [False, True])
def test_factory_strict_and_sticky_policy_is_preserved(strict, monkeypatch):
    face = _face()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    _sensitivity(face)
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_entropy_sensitivity(face)
    assert transcend.libm_fallback_used() and transcend.strict_backend() is strict


def test_factory_missing_backend_refuses_explicitly(monkeypatch):
    face = _face()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_sensitivity(face)


def test_conditional_or_damaged_cached_proof_refuses_even_zero_query():
    result = _sensitivity()
    result.certificate["meta"]["transcend_backend"] = "libm"
    changed = seal_certificate(result.certificate)
    result.certificate.clear()
    result.certificate.update(changed)
    assert not certificate_is_unconditional(result.certificate)
    with pytest.raises(ValueError):
        result.vjp((0,))


def test_conditional_face_cannot_be_promoted_to_rigorous_sensitivity():
    face = _face()
    face.certificate["meta"]["transcend_backend"] = "libm"
    changed = seal_certificate(face.certificate)
    face.certificate.clear()
    face.certificate.update(changed)
    assert certify_gaussian_entropy_sensitivity(face) is None


@pytest.mark.parametrize("level", ["residual", "unique", "domain"])
def test_deep_dependency_damage_refuses_factory_and_zero_query(level):
    result = _sensitivity()
    residual = result.face.residual
    dependency = residual if level == "residual" else residual.unique if level == "unique" else residual.unique.domain
    dependency.certificate["payload"]["type"] = "damaged"
    assert certify_gaussian_entropy_sensitivity(result.face) is None
    with pytest.raises(ValueError):
        result.vjp((0,))


@pytest.mark.parametrize("terms,error", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_invalid_term_count_raises_before_preflight(terms, error, monkeypatch):
    face = _face()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("invalid terms reached preflight"))
    with pytest.raises(error):
        certify_gaussian_entropy_sensitivity(face, terms=terms)


def test_invalid_face_type_raises():
    with pytest.raises(TypeError):
        certify_gaussian_entropy_sensitivity(object())


@pytest.mark.parametrize("cotangent,error", [((True,), TypeError), ((0.0,), TypeError), (("0",), TypeError),
                                            ((), ValueError), ((0, 0), ValueError)])
def test_invalid_exact_query_metadata_raises(cotangent, error):
    with pytest.raises(error):
        _sensitivity().vjp(cotangent)


def test_overlarge_terms_and_query_coefficients_refuse_with_resource_semantics():
    face = _face()
    assert certify_gaussian_entropy_sensitivity(face, terms=10**1000) is None
    with pytest.raises(ValueError):
        _sensitivity(face).vjp((10**1500,))


def test_more_terms_can_exhaust_exact_tail_bits_without_a_partial_certificate():
    face = _face(budget=RidgeBudget(max_coefficient_bits=128))
    assert _sensitivity(face, terms=8).tail_norm_bound > 0
    assert certify_gaussian_entropy_sensitivity(face, terms=16) is None


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_factory_preflight_precedes_proof_hashes(guard, monkeypatch):
    face = _face()
    module = import_module("omnibias.core.verified.gaussian_entropy_sensitivity")

    def exhausted(*args):
        raise ValueError("test exact sensitivity resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_face_intact", lambda *_a: pytest.fail("unreserved source scan"))
    assert certify_gaussian_entropy_sensitivity(face) is None


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_query_resource_failure_raises_before_cached_proof_hashes(guard, monkeypatch):
    result = _sensitivity()
    module = import_module("omnibias.core.verified.gaussian_entropy_sensitivity")

    def exhausted(*args):
        raise ValueError("test exact sensitivity query resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_sensitivity_intact", lambda *_a: pytest.fail("unreserved cached proof scan"))
    with pytest.raises(ValueError, match="resource"):
        result.vjp((0,))
