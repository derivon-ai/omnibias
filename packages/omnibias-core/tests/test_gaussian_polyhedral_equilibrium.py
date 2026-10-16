# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact original KKT data and complete weak-constraint bookkeeping."""

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
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    GaussianPolyhedralEquilibrium,
    certify_gaussian_polyhedral_equilibrium,
)


def _face(source=None, inequalities=((1,),), bounds=(0,), active=(0,), *, equalities=(), rhs=()):
    if source is None:
        source = GaussianHeatMap(((0,),), (0,))
    base = restrict_gaussian_heat(source, equalities, rhs)
    assert base is not None
    result = compile_gaussian_polyhedral_face(base, inequalities, bounds, active)
    assert result is not None
    return result


def test_active_constraint_balances_target_different_from_original_velocity():
    face = _face(bounds=(1,))
    result = certify_gaussian_polyhedral_equilibrium(face, (3,), (1,))
    assert result is not None
    assert result.posterior == (F(1),) and result.velocity == (F(1),)
    assert result.target != result.velocity
    assert result.hessian == ((F(1),),)
    assert result.equality_multipliers == () and result.inequality_multipliers == (2,)
    assert result.slacks == (0,)
    assert result.active_indices == result.strong_indices == (0,)
    assert result.weak_indices == () and result.strong_convexity_bound == 1


def test_every_tight_omitted_or_redundant_zero_multiplier_row_remains_weak():
    source = GaussianHeatMap(((0, 0),), (0,))
    face = _face(source, ((1, 0), (2, 0), (0, 1), (-1, 0), (0, 1)), (0, 0, 0, 2, 1), (0,))
    result = certify_gaussian_polyhedral_equilibrium(face, (1, 0), (0, 0))
    assert result is not None
    assert face.active_indices == (0,)
    assert result.active_indices == (0, 1, 2)
    assert result.strong_indices == (0,) and result.weak_indices == (1, 2)
    assert result.inequality_multipliers == (1, 0, 0, 0, 0)
    assert result.slacks == (0, 0, 0, 2, 1)


@pytest.mark.parametrize("selected", [(), (0,)])
def test_zero_selected_or_omitted_multiplier_is_a_weak_constraint(selected):
    result = certify_gaussian_polyhedral_equilibrium(_face(active=selected), (0,), (0,))
    assert result is not None and result.active_indices == result.weak_indices == (0,)
    assert result.strong_indices == () and result.inequality_multipliers == (0,)


def test_indefinite_ambient_hessian_and_signed_equality_multiplier_are_retained():
    source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)))
    face = _face(source, ((1, 0),), (0,), (0,), equalities=((0, 1),), rhs=(1,))
    result = certify_gaussian_polyhedral_equilibrium(face, (2, -3), (0, 1))
    assert result is not None
    assert source.contraction_bound == F(5, 2)
    assert face.base_restriction.model.contraction_bound == F(1, 2)
    assert result.posterior == (F(1, 2),)*2 and result.velocity == (0, 1)
    assert result.hessian == ((F(1, 2), F(-1)), (F(-1), F(-1)))
    assert result.equality_multipliers == (-4,) and result.inequality_multipliers == (2,)
    assert result.strong_convexity_bound == F(1, 2)
    # Full stationarity, including the unrestricted-sign equality multiplier.
    assert result.velocity[0]+result.inequality_multipliers[0] == result.target[0]
    assert result.velocity[1]+result.equality_multipliers[0] == result.target[1]


def test_oblique_equal_logits_give_independent_exact_covariance_formula():
    point = (F(1, 3), F(-1, 4))
    centers = ((0, 0), (1, 0), (0, 2))
    # Choose b_j=|x-a_j|²/(4 nu s), so each ORIGINAL logit equals zero.
    biases = tuple(sum((x-a)**2 for x, a in zip(point, center, strict=True))/12 for center in centers)
    source = GaussianHeatMap(centers, biases, viscosity=F(3, 2), time=2)
    face = _face(source, (), (), (), equalities=((1, 2),), rhs=(F(-1, 6),))
    target = (F(2, 7), F(19, 168))
    result = certify_gaussian_polyhedral_equilibrium(face, target, point)
    assert result is not None
    assert result.posterior == (F(1, 3),)*3
    assert result.velocity == (0, F(-11, 24))
    # Uniform covariance is [[2,-2],[-2,8]]/9. H=I/2-Cov/12.
    assert result.hessian == ((F(13, 27), F(1, 54)), (F(1, 54), F(23, 54)))
    assert result.equality_multipliers == (F(2, 7),)
    assert result.inequality_multipliers == result.slacks == ()
    assert result.active_indices == result.strong_indices == result.weak_indices == ()


def test_gauge_safe_exact_check_avoids_common_absolute_squares_and_floats():
    shift, common_bias = 2**2500, 2**3000
    source = GaussianHeatMap(((shift-1,), (shift+1,)), (common_bias, common_bias), time=2)
    face = _face(source, ((1,),), (shift,), (0,))
    result = certify_gaussian_polyhedral_equilibrium(face, (1,), (shift,))
    assert result is not None
    assert result.point == (F(shift),) and result.velocity == (0,)
    assert result.posterior == (F(1, 2),)*2 and result.hessian == ((F(3, 8),),)
    assert result.inequality_multipliers == (1,)
    # Squaring the common offset would violate the default 4096-bit budget.
    assert (shift*shift).bit_length() > source.budget.max_coefficient_bits


def test_unequal_original_logits_are_refused_even_at_a_selected_singleton():
    source = GaussianHeatMap(((-1,), (1,)), (0, 1), time=2)
    assert certify_gaussian_polyhedral_equilibrium(_face(source), (1,), (0,)) is None


def test_wrong_tangent_stationarity_cannot_be_hidden_by_selected_multipliers():
    source = GaussianHeatMap(((0, 0),), (0,))
    face = _face(source, ((1, 0),), (0,), (0,))
    assert certify_gaussian_polyhedral_equilibrium(face, (1, 1), (0, 0)) is None


def test_negative_multiplier_infeasible_point_and_unselected_violation_are_refused():
    assert certify_gaussian_polyhedral_equilibrium(_face(), (-1,), (0,)) is None
    assert certify_gaussian_polyhedral_equilibrium(_face(), (1,), (1,)) is None
    assert certify_gaussian_polyhedral_equilibrium(_face(active=()), (1,), (1,)) is None
    face = _face(inequalities=(), bounds=(), active=(), equalities=((1,),), rhs=(1,))
    assert certify_gaussian_polyhedral_equilibrium(face, (0,), (0,)) is None


def test_face_contraction_does_not_replace_global_base_contraction():
    source = GaussianHeatMap(((-2,), (2,)), (0, 0))
    face = _face(source)
    assert face.geometry.restriction.model.contraction_bound == 0
    assert face.base_restriction.model.contraction_bound == 2
    assert certify_gaussian_polyhedral_equilibrium(face, (1,), (0,)) is None


def test_factory_only_frozen_result_and_deep_dependency_snapshot():
    with pytest.raises(TypeError):
        GaussianPolyhedralEquilibrium()
    face = _face()
    target, point = [1], [0]
    result = certify_gaussian_polyhedral_equilibrium(face, target, point)
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.weak_indices = (0,)
    target[0] = point[0] = 99
    assert result.target == (1,) and result.point == (0,)
    assert verify_certificate_digest(result.certificate)
    assert certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    saved = deepcopy(result.certificate)
    face.certificate["payload"]["damaged"] = object()
    face.geometry.certificate["payload"]["damaged"] = object()
    face.base_restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["weak_indices"] = [0]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("dependency", ["face", "base", "geometry", "restriction"])
def test_damaged_dependencies_are_inconclusive(dependency):
    face = _face()
    record = {"face": face, "base": face.base_restriction, "geometry": face.geometry,
              "restriction": face.geometry.restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()
    assert certify_gaussian_polyhedral_equilibrium(face, (1,), (0,)) is None


@pytest.mark.parametrize("sticky", [False, True])
def test_no_transcendentals_iteration_or_factorization_and_preserved_history(sticky, monkeypatch):
    face = _face(GaussianHeatMap(((-1,), (1,)), (0, 0), time=2))

    def fail(*args, **kwargs):
        pytest.fail("exact equilibrium attempted numerical evaluation or a solve")

    for module_name, names in (
        ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat", "certify_gaussian_inverse")),
        ("omnibias.core.verified.gaussian_polyhedral", ("certify_gaussian_polyhedral_minimum",)),
        ("omnibias.core._rational_linear", ("factor_positive_definite", "solve_affine")),
        ("omnibias.core.verified.mean_map", ("_probabilities",)),
    ):
        module = import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = certify_gaussian_polyhedral_equilibrium(face, (1,), (0,))
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "not_used"
    assert transcend.libm_fallback_used() is sticky and not transcend.strict_backend()


def test_shared_strict_sealing_policy_is_preserved(monkeypatch):
    face = _face()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_polyhedral_equilibrium(face, (1,), (0,))
    assert transcend.strict_backend()


@pytest.mark.parametrize("index", [1, 2])
@pytest.mark.parametrize("bad,error", [(None, TypeError), ((), ValueError), ((0, 0), ValueError),
    ((True,), TypeError), ((0.0,), TypeError), ((complex(0),), TypeError)])
def test_exact_vector_domains(index, bad, error):
    arguments = [_face(), (1,), (0,)]
    arguments[index] = bad
    with pytest.raises(error):
        certify_gaussian_polyhedral_equilibrium(*arguments)


def test_invalid_face_type_raises():
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_equilibrium(None, (0,), (0,))


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_aggregate_resource_preflight_precedes_dependency_processing(guard, monkeypatch):
    face = _face()

    def exhausted(*args):
        raise ValueError("test exact budget exhausted")

    def fail(*args):
        pytest.fail("resource preflight must precede dependency processing")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    module = import_module("omnibias.core.verified.gaussian_polyhedral_equilibrium")
    monkeypatch.setattr(module, "_face_intact", fail)
    assert certify_gaussian_polyhedral_equilibrium(face, (1,), (0,)) is None


def test_exact_input_and_logit_intermediate_bit_growth_are_bounded():
    source = GaussianHeatMap(((-1,), (1,)), (0, 0), time=2, budget=RidgeBudget(max_coefficient_bits=64))
    face = _face(source, (), (), ())
    assert certify_gaussian_polyhedral_equilibrium(face, (10**100,), (0,)) is None
    assert certify_gaussian_polyhedral_equilibrium(face, (0,), (2**63,)) is None


def test_all_exact_arithmetic_uses_one_cumulative_work_meter(monkeypatch):
    face = _face()
    observed = []

    def bounded(self, count):
        observed.append(count)
        if count > observed[0]+15:
            raise ValueError("cumulative budget exhausted")

    monkeypatch.setattr(RidgeBudget, "_work", bounded)
    assert certify_gaussian_polyhedral_equilibrium(face, (1,), (0,)) is None
    assert len(observed) > 2 and observed[-1] > observed[0]+15
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))
