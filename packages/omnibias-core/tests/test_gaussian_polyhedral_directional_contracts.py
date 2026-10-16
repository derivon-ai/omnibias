# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact critical-face replay, bounded compilation and solve-free queries."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core._rational_linear import RationalLDLT
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
from omnibias.core.verified.gaussian_polyhedral_directional import (
    GaussianCriticalFace,
    GaussianPolyhedralDirectional,
    certify_gaussian_polyhedral_directional,
    compile_gaussian_critical_face,
)
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)


def _equilibrium(*, budget=None, strong=False):
    source = GaussianHeatMap(((0,),), (0,), budget=budget)
    base = restrict_gaussian_heat(source, (), ())
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((1,),), (0,), (0,) if strong else ())
    assert face is not None
    result = certify_gaussian_polyhedral_equilibrium(face, (1 if strong else 0,), (0,))
    assert result is not None
    return result


def _three_dimensional():
    source = GaussianHeatMap(((0, 0, 0),), (0,))
    base = restrict_gaussian_heat(source, ((0, 0, 1),), (0,))
    face = compile_gaussian_polyhedral_face(base, ((1, 0, 0), (0, 1, 0)), (0, 0), ())
    assert face is not None
    result = certify_gaussian_polyhedral_equilibrium(face, (0, 0, 0), (0, 0, 0))
    assert result is not None
    return result


def test_query_checks_exact_region_and_accepts_its_nonstrict_boundary():
    equilibrium = _equilibrium()
    free = compile_gaussian_critical_face(equilibrium)
    active = compile_gaussian_critical_face(equilibrium, (0,))
    assert free is not None and active is not None
    assert free.response == ((1,),) and free.region_inequalities == ((-1,),)
    assert active.response == ((0,),) and active.region_inequalities == ((1,),)
    assert certify_gaussian_polyhedral_directional(free, (1,)) is None
    assert certify_gaussian_polyhedral_directional(active, (-1,)) is None
    for face in (free, active):
        zero = certify_gaussian_polyhedral_directional(face, (0,))
        assert zero is not None and zero.value == (0,)
        assert zero.weak_multipliers == zero.weak_slacks == (0,)
        assert zero.quadratic_value == 0


def test_strong_directional_multiplier_has_no_nonnegativity_requirement():
    face = compile_gaussian_critical_face(_equilibrium(strong=True))
    assert face is not None and face.region_inequalities == ()
    result = certify_gaussian_polyhedral_directional(face, (-7,))
    assert result is not None and result.value == (0,)
    assert result.equality_multipliers == (-7,)
    assert result.weak_multipliers == result.weak_slacks == (0,)


def test_exact_precompiled_response_replays_every_coordinate_identity():
    face = compile_gaussian_critical_face(_three_dimensional(), (0,))
    assert face is not None
    assert face.constraints == ((0, 0, 1), (1, 0, 0))
    assert face.response == ((0, 0, 0), (0, 1, 0), (0, 0, 0))
    assert face.multiplier_response == ((0, 0, 1), (1, 0, 0))
    assert face.region_inequalities == ((1, 0, 0), (0, -1, 0))
    result = certify_gaussian_polyhedral_directional(face, (1, -2, -3))
    assert result is not None and result.value == (0, -2, 0)
    assert result.equality_multipliers == (-3, 1)
    assert result.weak_multipliers == (1, 0) and result.weak_slacks == (0, 2)
    assert result.quadratic_value == -2


def test_two_factors_are_each_reused_for_all_coordinate_solves(monkeypatch):
    equilibrium = _three_dimensional()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_directional")
    original_factor, original_solve = module.factor_positive_definite, RationalLDLT.solve
    factors, solves = [], []

    def factor(matrix, budget):
        result = original_factor(matrix, budget)
        factors.append(result)
        return result

    def solve(self, rhs, budget):
        solves.append(self)
        return original_solve(self, rhs, budget)

    monkeypatch.setattr(module, "factor_positive_definite", factor)
    monkeypatch.setattr(RationalLDLT, "solve", solve)
    assert compile_gaussian_critical_face(equilibrium, (0,)) is not None
    assert len(factors) == 2 and len(solves) == 6
    assert all(item is factors[0] for item in solves[:3])
    assert all(item is factors[1] for item in solves[3:])


def test_compiler_replay_refuses_a_corrupted_factor_solve(monkeypatch):
    equilibrium = _three_dimensional()
    monkeypatch.setattr(RationalLDLT, "solve", lambda self, rhs, budget: (F(0),)*len(rhs))
    assert compile_gaussian_critical_face(equilibrium, (0,)) is None


def test_query_replay_refuses_a_faulty_response_action(monkeypatch):
    face = compile_gaussian_critical_face(_equilibrium(), (0,))
    assert face is not None
    module = import_module("omnibias.core.verified.gaussian_polyhedral_directional")
    original = module._linear

    def corrupted(matrix, vector, arithmetic):
        return (F(1),) if matrix is face.response else original(matrix, vector, arithmetic)

    monkeypatch.setattr(module, "_linear", corrupted)
    assert certify_gaussian_polyhedral_directional(face, (1,)) is None


def test_query_performs_no_solves_search_or_numerical_inverse(monkeypatch):
    face = compile_gaussian_critical_face(_three_dimensional(), (0,))
    assert face is not None

    def fail(*args, **kwargs):
        pytest.fail("directional query attempted a solve, factorization or active-set search")

    for module_name, names in (
        ("omnibias.core.verified.gaussian_polyhedral_directional", ("compile_gaussian_critical_face", "solve_affine",
                                                                  "factor_positive_definite")),
        ("omnibias.core._rational_linear", ("solve_affine", "factor_positive_definite")),
        ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat", "certify_gaussian_inverse")),
        ("omnibias.core.verified.linalg", ("_float_inverse", "_symmetric_ldlt_solve", "interval_solve")),
    ):
        module = import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(RationalLDLT, "solve", fail)
    result = certify_gaussian_polyhedral_directional(face, (1, -2, -3))
    assert result is not None and result.value == (0, -2, 0)


def test_selected_redundant_weak_row_refuses_but_omitted_row_is_checked():
    source = GaussianHeatMap(((0, 0),), (0,))
    base = restrict_gaussian_heat(source, (), ())
    face = compile_gaussian_polyhedral_face(base, ((1, 0), (2, 0)), (0, 0), (0,))
    equilibrium = certify_gaussian_polyhedral_equilibrium(face, (1, 0), (0, 0))
    assert equilibrium is not None and equilibrium.strong_indices == (0,) and equilibrium.weak_indices == (1,)
    with pytest.raises(ValueError, match="full row rank"):
        compile_gaussian_critical_face(equilibrium, (1,))
    critical = compile_gaussian_critical_face(equilibrium)
    assert critical is not None and critical.region_inequalities == ((0, 0),)
    result = certify_gaussian_polyhedral_directional(critical, (-1, 2))
    assert result is not None and result.value == (0, 2)
    assert result.weak_slacks == result.weak_multipliers == (0, 0)


def test_overlarge_selection_refuses_before_quadratic_weak_membership_work(monkeypatch):
    source = GaussianHeatMap(((0,),), (0,))
    base = restrict_gaussian_heat(source, (), ())
    face = compile_gaussian_polyhedral_face(base, ((1,),)*64, (0,)*64, ())
    equilibrium = certify_gaussian_polyhedral_equilibrium(face, (0,), (0,))
    assert equilibrium is not None and len(equilibrium.weak_indices) == 64

    def fail(*args):
        pytest.fail("necessary row-count gate must precede dependent scans/processing")

    module = import_module("omnibias.core.verified.gaussian_polyhedral_directional")
    monkeypatch.setattr(module, "_equilibrium_intact", fail)
    with pytest.raises(ValueError, match="full row rank"):
        compile_gaussian_critical_face(equilibrium, tuple(range(64)))


def test_factory_only_frozen_artifacts_copy_dependencies_and_input_directions():
    with pytest.raises(TypeError):
        GaussianCriticalFace()
    with pytest.raises(TypeError):
        GaussianPolyhedralDirectional()
    equilibrium = _equilibrium()
    selected = [0]
    face = compile_gaussian_critical_face(equilibrium, selected)
    assert face is not None
    direction = [1]
    result = certify_gaussian_polyhedral_directional(face, direction)
    assert result is not None
    selected[0] = direction[0] = 99
    assert face.active_weak_indices == (0,) and result.direction == (1,)
    for record, field in ((face, "response"), (result, "value")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)
        assert verify_certificate_digest(record.certificate)
        assert certificate_is_unconditional(record.certificate)
        assert schema_errors_v1(record.certificate) == []
        assert "theorem_prover_verified" not in record.certificate["honesty"]
    saved = deepcopy(result.certificate)
    face.certificate["payload"]["damaged"] = object()
    equilibrium.certificate["payload"]["damaged"] = object()
    equilibrium.face.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["value"][0] = [99, 1]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("dependency", ["critical", "equilibrium", "original_face", "restriction"])
def test_damaged_dependencies_refuse_before_query_matrix_actions(dependency, monkeypatch):
    equilibrium = _equilibrium()
    critical = compile_gaussian_critical_face(equilibrium)
    assert critical is not None
    record = {"critical": critical, "equilibrium": equilibrium, "original_face": equilibrium.face,
              "restriction": equilibrium.face.base_restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()

    def fail(*args):
        pytest.fail("damaged dependency must refuse before matrix actions")

    module = import_module("omnibias.core.verified.gaussian_polyhedral_directional")
    monkeypatch.setattr(module, "_linear", fail)
    assert certify_gaussian_polyhedral_directional(critical, (-1,)) is None


def test_damaged_equilibrium_refuses_before_compile_elimination(monkeypatch):
    equilibrium = _equilibrium()
    equilibrium.certificate["payload"]["damaged"] = object()

    def fail(*args):
        pytest.fail("damaged equilibrium must refuse before elimination")

    module = import_module("omnibias.core.verified.gaussian_polyhedral_directional")
    monkeypatch.setattr(module, "solve_affine", fail)
    assert compile_gaussian_critical_face(equilibrium, (0,)) is None


@pytest.mark.parametrize("sticky", [False, True])
def test_exact_compile_and_query_need_no_backend_and_preserve_history(sticky, monkeypatch):
    equilibrium = _equilibrium()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    critical = compile_gaussian_critical_face(equilibrium)
    assert critical is not None
    result = certify_gaussian_polyhedral_directional(critical, (-1,))
    assert result is not None
    assert critical.certificate["meta"]["transcend_backend"] == result.certificate["meta"]["transcend_backend"] == "not_used"
    assert transcend.libm_fallback_used() is sticky and not transcend.strict_backend()


def test_shared_strict_sealing_policy_applies_to_both_factories(monkeypatch):
    equilibrium = _equilibrium()
    critical = compile_gaussian_critical_face(equilibrium)
    assert critical is not None
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        compile_gaussian_critical_face(equilibrium)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_polyhedral_directional(critical, (-1,))
    assert transcend.strict_backend()


@pytest.mark.parametrize("indices,error", [(None, TypeError), ("0", TypeError), ((True,), TypeError),
    ((0.0,), TypeError), ((-1,), ValueError), ((1,), ValueError), ((0, 0), ValueError)])
def test_invalid_selected_weak_indices_raise(indices, error):
    with pytest.raises(error):
        compile_gaussian_critical_face(_equilibrium(), indices)


@pytest.mark.parametrize("direction,error", [(None, TypeError), ((), ValueError), ((0, 0), ValueError),
    ((True,), TypeError), ((0.0,), TypeError), ((complex(0),), TypeError)])
def test_exact_direction_contract(direction, error):
    face = compile_gaussian_critical_face(_equilibrium())
    assert face is not None
    with pytest.raises(error):
        certify_gaussian_polyhedral_directional(face, direction)


def test_invalid_record_types_raise():
    with pytest.raises(TypeError):
        compile_gaussian_critical_face(None)
    with pytest.raises(TypeError):
        certify_gaussian_polyhedral_directional(None, (0,))


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=10000), RidgeBudget(max_elimination_work=55000)])
def test_compile_preflight_refuses_before_factorization(limits, monkeypatch):
    equilibrium = _equilibrium(budget=limits)

    def fail(*args):
        pytest.fail("compile preflight must precede factorization")

    module = import_module("omnibias.core.verified.gaussian_polyhedral_directional")
    monkeypatch.setattr(module, "factor_positive_definite", fail)
    assert compile_gaussian_critical_face(equilibrium) is None


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=20000), RidgeBudget(max_elimination_work=70000)])
def test_query_preflight_refuses_before_matrix_actions(limits, monkeypatch):
    critical = compile_gaussian_critical_face(_equilibrium(budget=limits))
    assert critical is not None

    def fail(*args):
        pytest.fail("query preflight must precede matrix actions")

    module = import_module("omnibias.core.verified.gaussian_polyhedral_directional")
    monkeypatch.setattr(module, "_linear", fail)
    assert certify_gaussian_polyhedral_directional(critical, (-1,)) is None


def test_direction_bit_growth_refuses_without_a_false_certificate():
    critical = compile_gaussian_critical_face(_equilibrium(budget=RidgeBudget(max_coefficient_bits=64)))
    assert critical is not None
    assert certify_gaussian_polyhedral_directional(critical, (-10**100,)) is None
