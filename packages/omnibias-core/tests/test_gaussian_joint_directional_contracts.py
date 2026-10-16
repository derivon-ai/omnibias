# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Joint Gaussian directions retain exact gauges and residual-query contracts."""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core._rational_linear import RationalLDLT
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_grouped_directional import (
    certify_gaussian_grouped_directional,
    compile_gaussian_grouped_critical_face,
)
from omnibias.core.verified.gaussian_grouped_equilibrium import certify_gaussian_grouped_equilibrium
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_joint_directional import (
    GaussianJointDirectional,
    certify_gaussian_joint_directional,
)
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_projected_equilibrium import (
    certify_gaussian_projected_equilibrium,
)
from omnibias.core.verified.interval import Interval


def _critical(*, uniform=False, budget=None, shift=(0, 0), projected_record=False):
    relative_centers = ((1, 0), (-1, 0), (0, 1), (0, -1))
    centers = tuple(tuple(value+offset for value, offset in zip(row, shift, strict=True)) for row in relative_centers)
    scores = (0, 0, 0 if uniform else 1, 0 if uniform else 1)
    biases = tuple(F(score)+sum(value*value for value in center)/F(16)
                   for score, center in zip(scores, relative_centers, strict=True))
    source = GaussianHeatMap(centers, biases, time=4, budget=budget)
    base = restrict_gaussian_heat(source, (), ())
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((1, 0), (0, 1)), shift, ())
    assert face is not None
    factory = certify_gaussian_projected_equilibrium if projected_record else certify_gaussian_grouped_equilibrium
    equilibrium = factory(face, (0, 0), shift)
    assert equilibrium is not None
    result = compile_gaussian_grouped_critical_face(equilibrium, (0,))
    assert result is not None
    return result


def _projected_critical():
    centers = ((1, 1), (1, -1), (3, 2), (3, -2), (3, 0))
    scores = (0, 0, 2, 2, 2)
    biases = tuple(F(score)+sum(value*value for value in center)/F(64)
                   for score, center in zip(scores, centers, strict=True))
    source = GaussianHeatMap(centers, biases, time=16)
    base = restrict_gaussian_heat(source, (), ())
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((1, 0), (0, 1)), (0, 0), (0,))
    assert face is not None
    equilibrium = certify_gaussian_projected_equilibrium(face, (0, 0), (0, 0))
    assert equilibrium is not None
    result = compile_gaussian_grouped_critical_face(equilibrium)
    assert result is not None
    return result


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("gauge", [0, F(2, 7), 2**2500])
def test_constant_bias_direction_has_exact_target_only_parity(uniform, gauge):
    face = _critical(uniform=uniform)
    target, proposal = (1, -1), (0, -4.4)
    previous = certify_gaussian_grouped_directional(face, target, proposal)
    result = certify_gaussian_joint_directional(face, target, (gauge,)*4, proposal)
    assert previous is not None and result is not None
    for name in ("value", "projected_proposal", "enclosure", "error_bound", "coordinate_error_bounds",
                 "residual_enclosure", "residual_norm_bound", "equality_multipliers", "weak_multipliers", "weak_slacks"):
        assert getattr(result, name) == getattr(previous, name)


@pytest.mark.parametrize("gauge", [0, 2**2500])
def test_exact_joint_cancellation_ignores_huge_proposal_and_common_bias_direction(gauge):
    face = _critical(uniform=True)
    beta = (gauge, gauge, gauge+1, gauge-1)
    result = certify_gaussian_joint_directional(face, (1, F(-1, 8)), beta, (1e200, 1e200))
    assert result is not None
    assert result.value == (0, 0) and result.projected_proposal == (0, F(1e200))
    assert result.error_bound == result.residual_norm_bound == 0
    assert result.enclosure == result.residual_enclosure == (Interval.point(0),)*2
    assert result.equality_multipliers == (Interval.point(1),)
    assert result.weak_multipliers[0] == Interval.point(1)
    assert result.weak_slacks == (Interval.point(0),)*2
    assert result.relative_bias_direction == (0, 0, 1, -1)
    assert result.group_bias_means == (0,)
    assert result.group_bias_covariances == ((0, F(1, 2)),)
    assert result.zero_projected_forcing
    assert result.forcing_enclosure == (Interval.point(1), Interval.point(0))
    assert result.projected_forcing_enclosure == (Interval.point(0),)*2
    assert certify_gaussian_joint_directional(face, (-1, F(-1, 8)), beta, (0, 0)) is None


def test_tiny_exact_joint_residual_is_not_classified_as_zero():
    face = _critical(uniform=True)
    epsilon = F(1, 2**100)
    proposal = float(-epsilon*F(64, 15))
    result = certify_gaussian_joint_directional(face, (1, F(-1, 8)-epsilon), (0, 0, 1, -1), (0, proposal))
    assert result is not None
    assert result.value == (0, F(proposal)) and not result.zero_projected_forcing
    assert result.error_bound > 0


def test_common_spatial_translation_is_cancelled_before_interval_covariance():
    ordinary, shifted = _critical(), _critical(shift=(2**1800, -2**1700))
    arguments = ((1, -1), (2**2500, 2**2500, 2**2500+1, 2**2500-1), (0, -3.6))
    left = certify_gaussian_joint_directional(ordinary, *arguments)
    right = certify_gaussian_joint_directional(shifted, *arguments)
    assert left is not None and right is not None
    for name in ("group_bias_means", "group_bias_covariances", "forcing_enclosure", "projected_forcing_enclosure",
                 "normal_forcing_enclosure", "enclosure", "error_bound", "equality_multipliers"):
        assert getattr(left, name) == getattr(right, name)


def test_projected_common_centroids_annihilate_unrepresentable_group_bias_means():
    face = _critical(projected_record=True)
    assert face.equilibrium.group_centroid_offsets == ((0, 0), (0, 0))
    target, proposal = (1, -1), (0, -4.4)
    baseline = certify_gaussian_grouped_directional(face, target, proposal)
    result = certify_gaussian_joint_directional(face, target, (0, 0, 2**2500, 2**2500), proposal)
    assert baseline is not None and result is not None
    assert result.group_bias_means == (0, 2**2500)
    assert result.group_bias_covariances == ((0, 0), (0, 0))
    assert result.forcing_enclosure == (Interval.point(1), Interval.point(-1))
    for name in ("value", "projected_proposal", "enclosure", "error_bound", "coordinate_error_bounds",
                 "residual_enclosure", "residual_norm_bound", "equality_multipliers", "weak_multipliers", "weak_slacks"):
        assert getattr(result, name) == getattr(baseline, name)


def test_between_group_normal_bias_covariance_is_retained_in_multiplier_response():
    mp = pytest.importorskip("mpmath")
    face = _projected_critical()
    result = certify_gaussian_joint_directional(face, (0, 0), (0, 0, 1, 1, 1), (100, 100))
    assert result is not None
    assert result.value == (0, 0) and result.error_bound == 0
    with mp.workdps(90):
        mass0, mass1 = 2/(2+3*mp.exp(2)), 3*mp.exp(2)/(2+3*mp.exp(2))
        expected = 2*mass0*mass1/16
        interval = result.equality_multipliers[0]
        assert mp.mpf(interval.lo) <= expected <= mp.mpf(interval.hi)
        assert interval.lo > 0
    # This row is primal-strong, so the negative derivative multiplier is valid.
    reversed_result = certify_gaussian_joint_directional(face, (0, 0), (0, 0, -1, -1, -1), (100, 100))
    assert reversed_result is not None and reversed_result.equality_multipliers[0].hi < 0


def test_query_uses_cached_curvature_and_weights_without_solve_or_exp(monkeypatch):
    face = _critical()

    def fail(*args, **kwargs):
        pytest.fail("joint query attempted a solve, posterior evaluation, or curvature compilation")

    for module_name, names in (
        ("omnibias.core.verified.gaussian_grouped_directional", ("compile_gaussian_grouped_critical_face", "solve_affine",
                                                               "factor_positive_definite", "_probabilities", "_moment")),
        ("omnibias.core.verified.mean_map", ("_probabilities",)),
        ("omnibias.core._rational_linear", ("solve_affine", "factor_positive_definite")),
        ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat", "certify_gaussian_inverse")),
        ("omnibias.core.verified.linalg", ("_float_inverse", "_symmetric_ldlt_solve", "interval_solve")),
    ):
        module = import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(RationalLDLT, "solve", fail)
    monkeypatch.setattr(transcend, "exp_iv", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = certify_gaussian_joint_directional(face, (1, -1), (0, 0, 1, -1), (0, -3.6))
    assert result is not None and certificate_is_unconditional(result.certificate)
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert transcend.libm_fallback_used() and not transcend.strict_backend()


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_nonuniform_query_preserves_backend_flags_and_compiled_provenance(sticky, strict, monkeypatch):
    face = _critical()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    result = certify_gaussian_joint_directional(face, (1, -1), (0, 0, 1, -1), (0, -3.6))
    assert result is not None
    assert result.certificate["meta"]["transcend_backend"] == face.certificate["meta"]["transcend_backend"]
    assert transcend.libm_fallback_used() is sticky and transcend.strict_backend() is strict


def test_single_group_query_needs_no_optional_backend(monkeypatch):
    face = _critical(uniform=True)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = certify_gaussian_joint_directional(face, (1, F(-1, 8)), (0, 0, 1, -1), (0, 100))
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "not_used"


def test_strict_sealing_still_requires_optional_backend(monkeypatch):
    face = _critical()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_joint_directional(face, (1, -1), (0, 0, 1, -1), (0, -3.6))
    assert transcend.strict_backend()


def test_factory_only_immutable_record_snapshots_inputs_and_certificate_dependencies():
    with pytest.raises(TypeError):
        GaussianJointDirectional()
    face = _critical()
    target, beta, proposal = [1, -1], [0, 0, 1, -1], [0, -3.6]
    result = certify_gaussian_joint_directional(face, target, beta, proposal)
    assert result is not None
    target[0] = beta[0] = proposal[0] = 99
    assert result.target_direction == (1, -1) and result.bias_direction == (0, 0, 1, -1)
    assert result.proposal == (0, -3.6)
    with pytest.raises(FrozenInstanceError):
        result.error_bound = F(0)
    assert verify_certificate_digest(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert certificate_is_unconditional(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    saved = deepcopy(result.certificate)
    face.certificate["payload"]["damaged"] = object()
    face.equilibrium.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["extra"] = [99]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("dependency", ["critical", "equilibrium", "face", "base", "geometry", "restriction"])
def test_dependency_tamper_refuses_before_exact_query_actions(dependency, monkeypatch):
    face = _critical()
    eq = face.equilibrium
    record = {"critical": face, "equilibrium": eq, "face": eq.face, "base": eq.face.base_restriction,
              "geometry": eq.face.geometry, "restriction": eq.face.geometry.restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()

    def fail(*args):
        pytest.fail("damaged evidence reached exact query actions")

    monkeypatch.setattr(_Arithmetic, "exact", fail)
    assert certify_gaussian_joint_directional(face, (1, -1), (0, 0, 1, -1), (0, -3.6)) is None


@pytest.mark.parametrize("argument,bad,error", [
    (1, None, TypeError), (1, (), ValueError), (1, (1,), ValueError), (1, (1.0, -1), TypeError),
    (1, (True, -1), TypeError), (1, (complex(1), -1), TypeError),
    (2, None, TypeError), (2, (), ValueError), (2, (0, 0, 0), ValueError), (2, (0, 0, 0, 0, 0), ValueError),
    (2, (0, 0, True, 0), TypeError), (2, (0, 0, 0.0, 0), TypeError), (2, (0, 0, complex(0), 0), TypeError),
    (3, (), ValueError), (3, (0,), ValueError), (3, (F(0), 0), TypeError),
    (3, (math.inf, 0), ValueError), (3, (math.nan, 0), ValueError), (3, (2**53+1, 0), ValueError),
])
def test_vector_validation_precedes_preflight_even_on_zero_shortcuts(argument, bad, error, monkeypatch):
    arguments = [_critical(uniform=True), (0, 0), (0,)*4, (0, 0)]
    arguments[argument] = bad

    def fail(*args):
        pytest.fail("malformed public data reached resource preflight")

    monkeypatch.setattr(_Arithmetic, "_entries", fail)
    with pytest.raises(error):
        certify_gaussian_joint_directional(*arguments)


def test_invalid_face_type_raises():
    with pytest.raises(TypeError):
        certify_gaussian_joint_directional(None, (0,), (0,), (0,))


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_preflight_precedes_exact_query_data_and_actions(guard, monkeypatch):
    face = _critical()

    def exhausted(*args):
        raise ValueError("test resource exhaustion")

    def fail(*args):
        pytest.fail("preflight missed exact query allocation")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(_Arithmetic, "exact", fail)
    assert certify_gaussian_joint_directional(face, (1, -1), (0, 0, 1, -1), (0, -3.6)) is None


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=70000), RidgeBudget(max_elimination_work=600000)])
def test_actual_aggregate_query_budget_can_refuse_after_successful_compilation(limits, monkeypatch):
    face = _critical(budget=limits)

    def fail(*args):
        pytest.fail("aggregate preflight missed exact query data")

    monkeypatch.setattr(_Arithmetic, "exact", fail)
    assert certify_gaussian_joint_directional(face, (1, -1), (0, 0, 1, -1), (0, -3.6)) is None


def test_bias_covariance_and_residual_work_share_a_cumulative_meter(monkeypatch):
    face = _critical()
    observed = []

    def bounded(self, count):
        observed.append(count)
        if count > observed[0]+25:
            raise ValueError("cumulative query budget exhausted")

    monkeypatch.setattr(RidgeBudget, "_work", bounded)
    assert certify_gaussian_joint_directional(face, (1, -1), (0, 0, 1, -1), (0, -3.6)) is None
    assert observed[-1] > observed[0]+25
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))


def test_successful_covariance_and_shared_response_never_reset_the_work_counter(monkeypatch):
    face = _projected_critical()
    module = import_module("omnibias.core.verified.gaussian_joint_directional")
    original, observed, response_counts = module._critical_response, [], []

    def record(self, count):
        observed.append(count)
        assert count <= self.max_elimination_work

    def response(face, projected, forcing, normal, arithmetic, **kwargs):
        response_counts.append(arithmetic.used)
        result = original(face, projected, forcing, normal, arithmetic, **kwargs)
        response_counts.append(arithmetic.used)
        return result

    monkeypatch.setattr(RidgeBudget, "_work", record)
    monkeypatch.setattr(module, "_critical_response", response)
    result = certify_gaussian_joint_directional(face, (0, -1), (0, 0, 1, 1, 1), (0, -17))
    assert result is not None
    assert observed[0] < response_counts[0] < response_counts[1] <= observed[-1]
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))


def test_exact_bit_growth_and_nonfinite_proposal_arithmetic_refuse():
    face = _critical(budget=RidgeBudget(max_coefficient_bits=64))
    assert certify_gaussian_joint_directional(face, (1, -10**100), (0,)*4, (0, 0)) is None
    assert certify_gaussian_joint_directional(face, (1, -1), (0, 0, 10**100, 0), (0, 0)) is None
    assert certify_gaussian_joint_directional(_critical(), (1, -1), (0, 0, 1, -1), (0, 1e308)) is None
