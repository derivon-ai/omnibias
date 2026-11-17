# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Composition, provenance and refusal contracts for original-parameter VJPs."""

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
from omnibias.core.verified.gaussian_affine_adjoint import compile_gaussian_affine_adjoint
from omnibias.core.verified.gaussian_affine_parameters import (
    GaussianAffineParameterVJP,
    certify_gaussian_affine_parameter_vjp,
)
from omnibias.core.verified.gaussian_affine_sensitivity import certify_gaussian_affine_target_vjp
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import certify_gaussian_inverse
from omnibias.core.verified.mean_map import MeanMap


def _geometry(*, time=1, budget=None):
    source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)), time=time, budget=budget)
    restriction = restrict_gaussian_heat(source, ((0, 1),), (1,))
    assert restriction is not None
    geometry = compile_gaussian_affine_adjoint(restriction)
    assert geometry is not None
    return geometry


def _query(geometry, **kwargs):
    arguments = {"target": (0, 0), "scaled_point": (0, 1), "upstream": (1, 0), "proposal": (2, 0)}
    arguments.update(kwargs)
    return certify_gaussian_affine_parameter_vjp(geometry, **arguments)


def test_factory_only_record_and_dependency_snapshots():
    with pytest.raises(TypeError):
        GaussianAffineParameterVJP()
    geometry = _geometry()
    target, point, upstream, proposal = [0, 0], [0.0, 1.0], [1.0, 0.0], [2.0, 0.0]
    result = certify_gaussian_affine_parameter_vjp(geometry, target, point, upstream, proposal)
    assert result is not None and result.root_certificate is not None
    assert result.geometry is geometry
    assert verify_certificate_digest(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert certificate_is_unconditional(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    with pytest.raises(FrozenInstanceError):
        result.rhs_gradient = ()
    for vector in (target, point, upstream, proposal):
        vector[0] = 99
    assert result.target_certificate.target == (0, 0)
    assert result.target_certificate.upstream == (1.0, 0.0)
    saved = deepcopy(result.certificate)
    result.root_certificate.certificate["payload"]["damaged"] = object()
    result.target_certificate.certificate["payload"]["damaged"] = object()
    geometry.certificate["payload"]["damaged"] = object()
    geometry.restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["time_gradient"]["lo"] = 123.0
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("dependency", ["geometry", "restriction"])
def test_damaged_geometry_refuses_before_any_composed_proof(dependency, monkeypatch):
    geometry = _geometry()
    certificate = geometry.certificate if dependency == "geometry" else geometry.restriction.certificate
    certificate["payload"]["damaged"] = object()
    module = import_module("omnibias.core.verified.gaussian_affine_parameters")

    def fail(*args, **kwargs):
        pytest.fail("damaged dependency must refuse before a composed proof")

    monkeypatch.setattr(module, "certify_gaussian_affine_target_vjp", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    assert _query(geometry) is None


def test_damaged_target_proof_is_not_used(monkeypatch):
    geometry = _geometry()
    sensitivity = certify_gaussian_affine_target_vjp(geometry.restriction, (0, 0), (0, 1), (1, 0), (2, 0))
    assert sensitivity is not None
    sensitivity.certificate["payload"]["damaged"] = object()
    module = import_module("omnibias.core.verified.gaussian_affine_parameters")
    monkeypatch.setattr(module, "certify_gaussian_affine_target_vjp", lambda *args, **kwargs: sensitivity)
    assert _query(geometry) is None


def test_damaged_separately_required_root_proof_is_not_used(monkeypatch):
    geometry = _geometry()
    root = certify_gaussian_inverse(geometry.restriction.model, (0, 0), (0, 1))
    assert root is not None
    root.certificate["payload"]["damaged"] = object()
    module = import_module("omnibias.core.verified.gaussian_affine_parameters")
    monkeypatch.setattr(module, "certify_gaussian_inverse", lambda *args, **kwargs: root)
    assert _query(geometry, upstream=(0, 1)) is None


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_complete_proof_preserves_prior_backend_state(monkeypatch, sticky, strict):
    geometry = _geometry()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    result = _query(geometry)
    assert result is not None and result.posterior
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert certificate_is_unconditional(result.certificate)
    assert transcend.libm_fallback_used() is sticky and transcend.strict_backend() is strict


@pytest.mark.parametrize("upstream", [(1, 0), (0, 1)])
def test_missing_backend_refuses_including_normal_upstream_needing_root(monkeypatch, upstream):
    geometry = _geometry()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    with pytest.raises(RuntimeError, match="mpmath"):
        _query(geometry, upstream=upstream)
    assert not transcend.strict_backend()


def test_affine_target_map_still_needs_posterior_for_center_gradients(monkeypatch):
    source = GaussianHeatMap(((0, 0), (0, 1)), (0, F(1, 3)))
    restriction = restrict_gaussian_heat(source, ((0, 1),), (0,))
    assert restriction is not None and restriction.model.contraction_bound == 0
    geometry = compile_gaussian_affine_adjoint(restriction)
    assert geometry is not None
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    with pytest.raises(RuntimeError, match="mpmath"):
        _query(geometry, scaled_point=(0, 0))
    # A normal upstream annihilates all model derivatives and needs no posterior.
    exact = _query(geometry, scaled_point=(0, 0), upstream=(0, 1))
    assert exact is not None and exact.posterior == () and exact.root_certificate is None
    assert exact.primal_multiplier is None
    assert exact.certificate["meta"]["transcend_backend"] == "not_used"


def test_exact_singleton_geometry_can_certify_without_backend(monkeypatch):
    source = GaussianHeatMap(((-1, -2), (1, 2)), (0, 0), time=10**300)
    restriction = restrict_gaussian_heat(source, ((1, 0), (0, 1)), (F(1, 3), F(-2, 7)))
    assert restriction is not None
    geometry = compile_gaussian_affine_adjoint(restriction)
    assert geometry is not None
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = _query(geometry, scaled_point=(-1e300, 1e300), upstream=(2, -3))
    assert result is not None and result.root_certificate is None
    assert result.posterior == () and result.primal_multiplier is None
    assert result.target_gradient == (Interval.point(0),)*2
    assert result.bias_gradient == (Interval.point(0),)*2
    for value, expected in zip(result.rhs_gradient, (2, -3), strict=True):
        assert value.contains(expected)
    assert transcend.libm_fallback_used() and not transcend.strict_backend()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        _query(geometry, upstream=(2, -3))


def test_no_query_iteration_factorization_or_dense_covariance(monkeypatch):
    geometry = _geometry()

    def fail(*args, **kwargs):
        pytest.fail("parameter proof attempted an iterative inverse or factorization")

    for module_name, names in (
        ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat",)),
        ("omnibias.core.verified.inverse_gaussian_affine", ("inverse_gaussian_affine", "inverse_gaussian_heat")),
        ("omnibias.core._rational_linear", ("factor_positive_definite", "solve_affine")),
        ("omnibias.core.verified.gaussian_affine_adjoint", ("factor_positive_definite",)),
        ("omnibias.core.verified.linalg", ("_float_inverse", "_symmetric_ldlt_solve", "interval_solve")),
    ):
        module = import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(MeanMap, "jacobian", fail)
    monkeypatch.setattr(GaussianHeatMap, "scaled_jacobian", fail)
    monkeypatch.setattr(GaussianHeatMap, "float_scaled_fixed_point", fail)
    assert _query(geometry) is not None
    assert _query(geometry, upstream=(0, 1)) is not None


@pytest.mark.parametrize("index", [1, 2, 3, 4])
@pytest.mark.parametrize("bad,error", [(None, TypeError), ((), ValueError), ((0,), ValueError),
    ((0, 0, 0), ValueError), ((True, 0), TypeError), ((complex(1), 0), TypeError)])
def test_vector_type_and_shape_validation(index, bad, error):
    arguments = [_geometry(), (0, 0), (0, 1), (1, 0), (2, 0)]
    arguments[index] = bad
    with pytest.raises(error):
        certify_gaussian_affine_parameter_vjp(*arguments)


@pytest.mark.parametrize("index", [2, 3, 4])
@pytest.mark.parametrize("bad,error", [(F(1), TypeError), (math.inf, ValueError),
    (math.nan, ValueError), (2**53+1, ValueError)])
def test_finite_binary64_vector_contract(index, bad, error):
    arguments = [_geometry(), (0, 0), (0, 1), (1, 0), (2, 0)]
    arguments[index] = (bad, 0)
    with pytest.raises(error):
        certify_gaussian_affine_parameter_vjp(*arguments)


def test_invalid_geometry_and_exact_target_raise():
    with pytest.raises(TypeError):
        _query(None)
    with pytest.raises(TypeError):
        _query(_geometry(), target=(0.0, 0))


@pytest.mark.parametrize("time", [F(1, 2), F(1, 4)])
def test_inapplicable_contraction_is_inconclusive_even_for_normal_upstream(time, monkeypatch):
    geometry = _geometry(time=time)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    assert _query(geometry) is None
    assert _query(geometry, upstream=(0, 1)) is None


@pytest.mark.parametrize("limits,target", [
    (RidgeBudget(max_matrix_entries=4000), 0),
    (RidgeBudget(max_elimination_work=10000), 0),
    (RidgeBudget(max_coefficient_bits=64), 10**100),
])
def test_aggregate_preflight_occurs_before_composed_certificate(limits, target, monkeypatch):
    source = GaussianHeatMap(((0,),), (0,), budget=limits)
    restriction = restrict_gaussian_heat(source, (), ())
    assert restriction is not None
    geometry = compile_gaussian_affine_adjoint(restriction)
    assert geometry is not None
    module = import_module("omnibias.core.verified.gaussian_affine_parameters")

    def fail(*args, **kwargs):
        pytest.fail("aggregate preflight must precede the composed certificate")

    monkeypatch.setattr(module, "certify_gaussian_affine_target_vjp", fail)
    assert certify_gaussian_affine_parameter_vjp(geometry, (target,), (0,), (1,), (0,)) is None


def test_nonfinite_multiplier_enclosure_is_inconclusive():
    source = GaussianHeatMap(((0,),), (0,))
    restriction = restrict_gaussian_heat(source, ((F(1, 10**200),),), (0,))
    assert restriction is not None
    geometry = compile_gaussian_affine_adjoint(restriction)
    assert geometry is not None
    assert certify_gaussian_affine_parameter_vjp(geometry, (0,), (0,), (1e200,), (0,)) is None
