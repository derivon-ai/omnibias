# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Polyhedral proof composition, strict-margin gates and resource contracts."""

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
from omnibias.core.verified.gaussian_affine import (
    InconsistentAffineConstraints,
    restrict_gaussian_heat,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import (
    GaussianPolyhedralFace,
    GaussianPolyhedralMinimum,
    GaussianPolyhedralParameterVJP,
    certify_gaussian_polyhedral_minimum,
    certify_gaussian_polyhedral_parameter_vjp,
    compile_gaussian_polyhedral_face,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mean_map import MeanMap


def _base(*, budget=None):
    result = restrict_gaussian_heat(GaussianHeatMap(((0,),), (0,), budget=budget), (), ())
    assert result is not None
    return result


def _face(*, budget=None):
    result = compile_gaussian_polyhedral_face(_base(budget=budget), ((1,), (-1,)), (1, 3), (0,))
    assert result is not None
    return result


def _minimum(face=None):
    result = certify_gaussian_polyhedral_minimum(_face() if face is None else face, (2,), (1,))
    assert result is not None and result.active_set_stable
    return result


def test_compilation_snapshots_sorts_and_retains_original_family():
    source = GaussianHeatMap(((0, 0, 0), (1, 2, 3)), (0, F(1, 3)), time=4)
    base = restrict_gaussian_heat(source, ((0, 0, 1),), (1,))
    assert base is not None
    inequalities = [[1, 0, 0], [0, 1, 0], [-1, 0, 0]]
    upper, active = [2, 3, 4], [1, 0]
    face = compile_gaussian_polyhedral_face(base, inequalities, upper, active)
    assert face is not None and face.active_indices == (0, 1)
    assert face.geometry.restriction.source_model is source
    assert face.geometry.restriction.constraints == ((0, 0, 1), (1, 0, 0), (0, 1, 0))
    assert face.geometry.restriction.rhs == (1, 2, 3)
    assert face.slack_directions == ((0, 0, 0),)*3
    assert face.slack_offsets == (0, 0, 6)
    inequalities[0][0] = upper[0] = active[0] = 99
    assert face.inequalities[0] == (1, 0, 0) and face.upper_bounds[0] == 2
    assert face.active_indices == (0, 1)


def test_independent_stacked_rows_are_required_for_parameter_domain():
    source = GaussianHeatMap(((0, 0),), (0,))
    redundant = restrict_gaussian_heat(source, ((1, 0), (2, 0)), (0, 0))
    assert redundant is not None
    with pytest.raises(ValueError, match="full row rank"):
        compile_gaussian_polyhedral_face(redundant, (), (), ())
    base = restrict_gaussian_heat(source, ((1, 0),), (0,))
    assert base is not None
    with pytest.raises(ValueError, match="full row rank"):
        compile_gaussian_polyhedral_face(base, ((2, 0),), (0,), (0,))
    with pytest.raises(InconsistentAffineConstraints):
        compile_gaussian_polyhedral_face(base, ((2, 0),), (1,), (0,))


def test_face_contraction_alone_cannot_certify_global_polyhedral_minimum(monkeypatch):
    source = GaussianHeatMap(((-2,), (2,)), (0, 0))
    base = restrict_gaussian_heat(source, (), ())
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((1,),), (1,), (0,))
    assert face is not None and face.geometry.restriction.model.contraction_bound == 0
    assert base.model.contraction_bound == 2

    def fail(*args, **kwargs):
        pytest.fail("global q_base gate must precede face stationarity")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "certify_gaussian_affine_stationarity", fail)
    assert certify_gaussian_polyhedral_minimum(face, (2,), (1,)) is None


@pytest.mark.parametrize("active", [(), (0,)])
def test_nonstrict_minimum_does_not_authorize_derivative_wrapper(active, monkeypatch):
    face = compile_gaussian_polyhedral_face(_base(), ((1,),), (0,), active)
    assert face is not None
    minimum = certify_gaussian_polyhedral_minimum(face, (0,), (0,))
    assert minimum is not None and not minimum.active_set_stable

    def fail(*args, **kwargs):
        pytest.fail("strict-margin gate must precede affine derivative certification")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "certify_gaussian_affine_parameter_vjp", fail)
    assert certify_gaussian_polyhedral_parameter_vjp(minimum, (1,), (0,)) is None
    assert certify_gaussian_polyhedral_parameter_vjp(minimum, (0,), (0,)) is None


def test_wrong_active_set_is_inconclusive_not_an_infeasibility_claim():
    active = compile_gaussian_polyhedral_face(_base(), ((1,),), (1,), (0,))
    inactive = compile_gaussian_polyhedral_face(_base(), ((1,),), (1,), ())
    assert active is not None and inactive is not None
    assert certify_gaussian_polyhedral_minimum(active, (0,), (1,)) is None  # Negative multiplier.
    assert certify_gaussian_polyhedral_minimum(inactive, (2,), (2,)) is None  # Negative slack.


def test_inactive_rows_have_exact_zero_local_derivatives_in_original_order():
    source = GaussianHeatMap(((0, 0),), (0,))
    base = restrict_gaussian_heat(source, (), ())
    face = compile_gaussian_polyhedral_face(base, ((-1, 0), (0, 1), (1, 0)), (3, 2, 1), (2, 1))
    assert face is not None
    minimum = certify_gaussian_polyhedral_minimum(face, (3, 5), (1, 2))
    assert minimum is not None and minimum.active_set_stable
    result = certify_gaussian_polyhedral_parameter_vjp(minimum, (2, -1), (0, 0))
    assert result is not None
    assert result.inequality_gradient[0] == (Interval.point(0),)*2
    assert result.inequality_rhs_gradient[0] == Interval.point(0)
    for interval, expected in zip(result.inequality_rhs_gradient, (0, -1, 2), strict=True):
        assert interval.contains(expected)
    assert result.equality_gradient == result.equality_rhs_gradient == ()


def test_all_records_are_factory_only_frozen_and_certificates_copy_dependencies():
    for record in (GaussianPolyhedralFace, GaussianPolyhedralMinimum, GaussianPolyhedralParameterVJP):
        with pytest.raises(TypeError):
            record()
    face = _face()
    minimum = _minimum(face)
    result = certify_gaussian_polyhedral_parameter_vjp(minimum, (1,), (0,))
    assert result is not None
    for record, field in ((face, "active_indices"), (minimum, "active_set_stable"), (result, "inequality_gradient")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)
        assert verify_certificate_digest(record.certificate)
        assert certificate_is_unconditional(record.certificate)
        assert schema_errors_v1(record.certificate) == []
        assert "theorem_prover_verified" not in record.certificate["honesty"]
    saved = deepcopy(result.certificate)
    result.base_vjp.certificate["payload"]["damaged"] = object()
    minimum.certificate["payload"]["damaged"] = object()
    face.certificate["payload"]["damaged"] = object()
    face.geometry.certificate["payload"]["damaged"] = object()
    face.base_restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["inequality_rhs_gradient"][0]["lo"] = 99
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("dependency", ["face", "base", "geometry", "restriction"])
def test_query_refuses_damaged_face_before_stationarity(dependency, monkeypatch):
    face = _face()
    record = {"face": face, "base": face.base_restriction, "geometry": face.geometry,
              "restriction": face.geometry.restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()

    def fail(*args, **kwargs):
        pytest.fail("damaged face must be refused before stationarity")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "certify_gaussian_affine_stationarity", fail)
    assert certify_gaussian_polyhedral_minimum(face, (2,), (1,)) is None


@pytest.mark.parametrize("dependency", ["minimum", "stationarity", "root"])
def test_vjp_refuses_damaged_minimum_before_derivative_evaluation(dependency, monkeypatch):
    minimum = _minimum()
    record = {"minimum": minimum, "stationarity": minimum.stationarity,
              "root": minimum.stationarity.root_certificate}[dependency]
    record.certificate["payload"]["damaged"] = object()

    def fail(*args, **kwargs):
        pytest.fail("damaged minimum must be refused before derivative evaluation")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "certify_gaussian_affine_parameter_vjp", fail)
    assert certify_gaussian_polyhedral_parameter_vjp(minimum, (1,), (0,)) is None


def test_damaged_base_refuses_compilation_before_restriction_rebuild(monkeypatch):
    base = _base()
    base.certificate["payload"]["damaged"] = object()

    def fail(*args, **kwargs):
        pytest.fail("damaged base must refuse before face restriction")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "restrict_gaussian_heat", fail)
    assert compile_gaussian_polyhedral_face(base, ((1,),), (1,), (0,)) is None


def test_queries_do_not_iterate_search_factor_or_construct_covariance(monkeypatch):
    source = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0), time=4)
    base = restrict_gaussian_heat(source, (), ())
    face = compile_gaussian_polyhedral_face(base, ((0, 1), (-1, 0)), (1, 3), (0,))
    assert face is not None

    def fail(*args, **kwargs):
        pytest.fail("query attempted a root solve, face search, factorization or dense covariance")

    for module_name, names in (
        ("omnibias.core.verified.gaussian_polyhedral", ("compile_gaussian_polyhedral_face", "restrict_gaussian_heat",
                                                     "compile_gaussian_affine_adjoint")),
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
    minimum = certify_gaussian_polyhedral_minimum(face, (0, 2), (0, 0.25))
    assert minimum is not None and minimum.active_set_stable
    assert certify_gaussian_polyhedral_parameter_vjp(minimum, (1, 1), (32/7, 0)) is not None


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_composed_proofs_preserve_sticky_and_strict_backend_state(sticky, strict, monkeypatch):
    face = _face()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    if sticky and strict:
        # The shared sealing policy refuses a not_used exact child while
        # strict mode retains prior fallback history, even before the later
        # posterior computation. Composition must not clear or bypass it.
        with pytest.raises(RuntimeError, match="libm_fallback"):
            _minimum(face)
        assert transcend.libm_fallback_used() and transcend.strict_backend()
        return
    minimum = _minimum(face)
    result = certify_gaussian_polyhedral_parameter_vjp(minimum, (1,), (0,))
    assert result is not None
    # The affine derivative has no posterior, but minimum stationarity did.
    assert result.base_vjp.certificate["meta"]["transcend_backend"] == "not_used"
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert transcend.libm_fallback_used() is sticky and transcend.strict_backend() is strict


def test_missing_backend_does_not_issue_a_conditional_minimum(monkeypatch):
    face = _face()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_polyhedral_minimum(face, (2,), (1,))
    assert not transcend.strict_backend()


@pytest.mark.parametrize("matrix,upper,active,error", [
    (None, (), (), TypeError), (((1.0,),), (1,), (), TypeError),
    (((True,),), (1,), (), TypeError), (((1, 2),), (1,), (), ValueError),
    (((1,),), (), (), ValueError), (((1,),), (1.0,), (), TypeError),
    (((1,),), (1,), None, TypeError), (((1,),), (1,), (True,), TypeError),
    (((1,),), (1,), (0.0,), TypeError), (((1,),), (1,), (-1,), ValueError),
    (((1,),), (1,), (1,), ValueError), (((1,), (1,)), (1, 1), (0, 0), ValueError),
])
def test_compile_invalid_domains_raise(matrix, upper, active, error):
    with pytest.raises(error):
        compile_gaussian_polyhedral_face(_base(), matrix, upper, active)


def test_invalid_record_types_raise():
    with pytest.raises(TypeError):
        compile_gaussian_polyhedral_face(None, (), ())
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_minimum(None, (0,), (0,))
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_parameter_vjp(None, (0,), (0,))


@pytest.mark.parametrize("target,point,error", [(None, (0,), TypeError), ((), (0,), ValueError),
    ((0.0,), (0,), TypeError), ((True,), (0,), TypeError), ((0,), (), ValueError),
    ((0,), (F(0),), TypeError), ((0,), (math.inf,), ValueError), ((0,), (2**53+1,), ValueError)])
def test_minimum_vector_contract(target, point, error):
    with pytest.raises(error):
        certify_gaussian_polyhedral_minimum(_face(), target, point)


@pytest.mark.parametrize("upstream,proposal,error", [(None, (0,), TypeError), ((), (0,), ValueError),
    ((True,), (0,), TypeError), ((F(1),), (0,), TypeError), ((1,), (), ValueError),
    ((1,), (math.inf,), ValueError), ((1,), (2**53+1,), ValueError)])
def test_vjp_vector_contract(upstream, proposal, error):
    with pytest.raises(error):
        certify_gaussian_polyhedral_parameter_vjp(_minimum(), upstream, proposal)


def test_active_duplicate_check_allocation_has_its_own_preflight(monkeypatch):
    base = _base(budget=RidgeBudget(max_matrix_entries=1100))

    def fail(*args, **kwargs):
        pytest.fail("duplicate-check set allocated before bounded entry preflight")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "set", fail, raising=False)
    assert compile_gaussian_polyhedral_face(base, ((1,),)*1200, (1,)*1200, tuple(range(1200))) is None


def test_compile_reserves_quadratic_mode_rebuild_before_nested_factory(monkeypatch):
    source = GaussianHeatMap(((0,),)*16, (0,)*16, budget=RidgeBudget(max_elimination_work=190000))
    base = restrict_gaussian_heat(source, (), ())
    assert base is not None

    def fail(*args, **kwargs):
        pytest.fail("aggregate reservation omitted the child quadratic-mode rebuild")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "restrict_gaussian_heat", fail)
    assert compile_gaussian_polyhedral_face(base, ((1,),), (1,), (0,)) is None


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=4000), RidgeBudget(max_elimination_work=30000)])
def test_compile_resource_refusal_precedes_nested_factory(limits, monkeypatch):
    base = _base(budget=limits)

    def fail(*args, **kwargs):
        pytest.fail("compile resource refusal must precede nested restriction")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "restrict_gaussian_heat", fail)
    assert compile_gaussian_polyhedral_face(base, ((1,),), (1,), (0,)) is None


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=9000), RidgeBudget(max_elimination_work=90000)])
def test_minimum_resource_refusal_precedes_stationarity(limits, monkeypatch):
    face = _face(budget=limits)

    def fail(*args, **kwargs):
        pytest.fail("minimum preflight must precede stationarity")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "certify_gaussian_affine_stationarity", fail)
    assert certify_gaussian_polyhedral_minimum(face, (2,), (1,)) is None


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=20000), RidgeBudget(max_elimination_work=200000)])
def test_vjp_resource_refusal_precedes_affine_derivative(limits, monkeypatch):
    minimum = _minimum(_face(budget=limits))

    def fail(*args, **kwargs):
        pytest.fail("VJP preflight must precede affine derivative")

    module = import_module("omnibias.core.verified.gaussian_polyhedral")
    monkeypatch.setattr(module, "certify_gaussian_affine_parameter_vjp", fail)
    assert certify_gaussian_polyhedral_parameter_vjp(minimum, (1,), (0,)) is None
