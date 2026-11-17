# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Dense interval curvature, exact geometry and residual-only query contracts."""

import math
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
from omnibias.core.verified.gaussian_grouped_directional import (
    GaussianGroupedCriticalFace,
    GaussianGroupedDirectional,
    certify_gaussian_grouped_directional,
    compile_gaussian_grouped_critical_face,
)
from omnibias.core.verified.gaussian_grouped_equilibrium import certify_gaussian_grouped_equilibrium
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.interval import Interval


def _equilibrium(*, budget=None, strong=False, uniform=False, unequal_sizes=False):
    centers = ((1, 0), (-1, 0), (0, 1), (0, -1))
    scores = (0, 0, 0 if uniform else 1, 0 if uniform else 1)
    if unequal_sizes:
        centers += ((0, 0),)
        scores += (1,)
    biases = tuple(F(score)+sum(value*value for value in center)/F(16)
                   for score, center in zip(scores, centers, strict=True))
    source = GaussianHeatMap(centers, biases, time=4, budget=budget)
    base = restrict_gaussian_heat(source, (), ())
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((1, 0), (0, 1)), (0, 0), (0,) if strong else ())
    assert face is not None
    equilibrium = certify_gaussian_grouped_equilibrium(face, (1 if strong else 0, 0), (0, 0))
    assert equilibrium is not None
    return equilibrium


def _critical(**kwargs):
    result = compile_gaussian_grouped_critical_face(_equilibrium(**kwargs), (0,))
    assert result is not None
    return result


def test_cardinality_normalization_and_exact_structural_moment_zeros():
    mp = pytest.importorskip("mpmath")
    face = _critical(unequal_sizes=True)
    with mp.workdps(90):
        masses = (2/(2+3*mp.e), 3*mp.e/(2+3*mp.e))
        for interval, expected in zip(face.group_probabilities, masses, strict=True):
            assert mp.mpf(interval.lo) <= expected <= mp.mpf(interval.hi)
        for index, interval in enumerate(face.posterior):
            expected = (1 if index < 2 else mp.e)/(2+3*mp.e)
            assert mp.mpf(interval.lo) <= expected <= mp.mpf(interval.hi)
    assert face.projector == ((0, 0), (0, 1))
    assert face.left_inverse == ((1, 0),)
    assert face.hessian[0][1] == face.hessian[1][0] == Interval.point(0)
    assert face.projected_hessian[0] == (Interval.point(0),)*2
    assert face.multiplier_hessian == ((Interval.point(0),)*2,)


def test_zero_tangent_direction_ignores_proposal_but_still_checks_weak_signs():
    face = _critical()
    result = certify_gaussian_grouped_directional(face, (1, 0), (1e200, 1e200))
    assert result is not None
    assert result.value == (0, 0) and result.projected_proposal == (0, F(1e200))
    assert result.error_bound == result.residual_norm_bound == 0
    assert result.enclosure == result.residual_enclosure == (Interval.point(0),)*2
    assert result.weak_multipliers[0].contains(1)
    assert result.weak_slacks == (Interval.point(0),)*2
    assert certify_gaussian_grouped_directional(face, (-1, 0), (0, 0)) is None


def test_nonzero_tangent_residual_is_not_replaced_by_a_tolerance_shortcut(monkeypatch):
    face = _critical()
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")

    def fail(*args):
        raise ArithmeticError("test nonzero projected forcing reached residual evaluation")

    monkeypatch.setattr(module, "_interval_action", fail)
    assert certify_gaussian_grouped_directional(face, (1, F(-1, 2**100)), (0, 0)) is None
    assert certify_gaussian_grouped_directional(face, (1, 0), (0, 99)) is not None


def test_conclusive_response_has_tangent_coordinate_bounds_and_untrusted_center():
    face = _critical()
    result = certify_gaussian_grouped_directional(face, (1, -1), (1e200, -4.4))
    assert result is not None
    assert result.value == result.projected_proposal == (0, F(-4.4))
    assert result.error_bound > 0 and result.coordinate_error_bounds == (0, result.error_bound)
    assert result.enclosure[0] == result.residual_enclosure[0] == Interval.point(0)
    assert result.weak_multipliers[0].lo > 0 and result.weak_slacks[1].lo > 0
    assert result.certificate["meta"]["transcend_backend"] == face.certificate["meta"]["transcend_backend"] == "mpmath"
    # A poor proposal cannot establish the same weak slack sign.
    assert certify_gaussian_grouped_directional(face, (1, -1), (0, 0)) is None
    assert certify_gaussian_grouped_directional(face, (-1, -1), (0, -4.4)) is None


def test_strong_derivative_multipliers_are_unrestricted():
    face = compile_gaussian_grouped_critical_face(_equilibrium(strong=True))
    assert face is not None
    result = certify_gaussian_grouped_directional(face, (-7, 0), (100, 100))
    assert result is not None and result.value == (0, 0)
    assert result.equality_multipliers[0].contains(-7)
    assert result.weak_multipliers == result.weak_slacks == (Interval.point(0),)*2


def test_one_row_gram_factor_is_reused_and_geometry_replay_precedes_posterior(monkeypatch):
    equilibrium = _equilibrium()
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")
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
    assert compile_gaussian_grouped_critical_face(equilibrium, (0,)) is not None
    assert len(factors) == 1 and len(solves) == 2 and all(item is factors[0] for item in solves)
    monkeypatch.setattr(RationalLDLT, "solve", lambda self, rhs, budget: (F(0),)*len(rhs))
    monkeypatch.setattr(module, "_probabilities", lambda *_: pytest.fail("posterior reached before geometry replay"))
    assert compile_gaussian_grouped_critical_face(equilibrium, (0,)) is None


def test_query_reuses_compiled_bounds_without_solves_or_transcendentals(monkeypatch):
    face = _critical()

    def fail(*args, **kwargs):
        pytest.fail("query attempted a solve, new curvature evaluation or transcendental")

    for module_name, names in (
        ("omnibias.core.verified.gaussian_grouped_directional", ("compile_gaussian_grouped_critical_face", "solve_affine",
                                                               "factor_positive_definite", "_probabilities", "_moment", "_mix")),
        ("omnibias.core._rational_linear", ("solve_affine", "factor_positive_definite")),
        ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat", "certify_gaussian_inverse")),
        ("omnibias.core.verified.linalg", ("_float_inverse", "_symmetric_ldlt_solve", "interval_solve")),
    ):
        module = import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(RationalLDLT, "solve", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = certify_gaussian_grouped_directional(face, (1, -1), (0, -4.4))
    assert result is not None and certificate_is_unconditional(result.certificate)
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert transcend.libm_fallback_used() and not transcend.strict_backend()


def test_single_group_needs_no_backend_but_nonuniform_groups_do(monkeypatch):
    uniform, nonuniform = _equilibrium(uniform=True), _equilibrium()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    face = compile_gaussian_grouped_critical_face(uniform, (0,))
    assert face is not None and face.group_probabilities == (Interval.point(1),)
    assert face.certificate["meta"]["transcend_backend"] == "not_used"
    result = certify_gaussian_grouped_directional(face, (1, -1), (0, -64/15))
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "not_used"
    with pytest.raises(RuntimeError, match="mpmath"):
        compile_gaussian_grouped_critical_face(nonuniform, (0,))
    assert not transcend.strict_backend()


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_nonuniform_compile_and_query_preserve_backend_state(sticky, strict, monkeypatch):
    equilibrium = _equilibrium()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    face = compile_gaussian_grouped_critical_face(equilibrium, (0,))
    assert face is not None
    result = certify_gaussian_grouped_directional(face, (1, -1), (0, -4.4))
    assert result is not None
    assert transcend.libm_fallback_used() is sticky and transcend.strict_backend() is strict


def test_strict_sealing_still_requires_backend_for_a_query(monkeypatch):
    face = _critical()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_grouped_directional(face, (1, -1), (0, -4.4))
    assert transcend.strict_backend()


def test_factory_only_frozen_records_snapshot_dependencies_and_inputs():
    for record in (GaussianGroupedCriticalFace, GaussianGroupedDirectional):
        with pytest.raises(TypeError):
            record()
    equilibrium = _equilibrium()
    active = [0]
    face = compile_gaussian_grouped_critical_face(equilibrium, active)
    assert face is not None
    direction, proposal = [1, -1], [0, -4.4]
    result = certify_gaussian_grouped_directional(face, direction, proposal)
    assert result is not None
    active[0] = direction[0] = proposal[0] = 99
    assert face.active_weak_indices == (0,) and result.direction == (1, -1) and result.proposal == (0, -4.4)
    for record, field in ((face, "hessian"), (result, "value")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)
        assert verify_certificate_digest(record.certificate)
        assert schema_errors_v1(record.certificate) == []
        assert certificate_is_unconditional(record.certificate)
        assert "theorem_prover_verified" not in record.certificate["honesty"]
    saved = deepcopy(result.certificate)
    face.certificate["payload"]["damaged"] = object()
    equilibrium.certificate["payload"]["damaged"] = object()
    equilibrium.face.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["error_bound"] = [0, 1]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("dependency", ["critical", "equilibrium", "face", "base"])
def test_damaged_dependencies_refuse_before_query_actions(dependency, monkeypatch):
    face = _critical()
    record = {"critical": face, "equilibrium": face.equilibrium, "face": face.equilibrium.face,
              "base": face.equilibrium.face.base_restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")
    monkeypatch.setattr(module, "_linear", lambda *_: pytest.fail("damaged proof reached query actions"))
    assert certify_gaussian_grouped_directional(face, (1, -1), (0, -4.4)) is None


def test_damaged_equilibrium_refuses_before_compile_elimination(monkeypatch):
    equilibrium = _equilibrium()
    equilibrium.certificate["payload"]["damaged"] = object()
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")
    monkeypatch.setattr(module, "solve_affine", lambda *_: pytest.fail("damaged proof reached elimination"))
    assert compile_gaussian_grouped_critical_face(equilibrium, (0,)) is None


@pytest.mark.parametrize("indices,error", [(None, TypeError), ("0", TypeError), ((True,), TypeError),
    ((0.0,), TypeError), ((-1,), ValueError), ((2,), ValueError), ((0, 0), ValueError)])
def test_selected_weak_index_contract(indices, error):
    with pytest.raises(error):
        compile_gaussian_grouped_critical_face(_equilibrium(), indices)


@pytest.mark.parametrize("direction,proposal,error", [(None, (0, 0), TypeError), ((), (0, 0), ValueError),
    ((1,), (0, 0), ValueError), ((1.0, -1), (0, 0), TypeError), ((True, -1), (0, 0), TypeError),
    ((1, -1), (), ValueError), ((1, -1), (F(0), 0), TypeError), ((1, -1), (math.inf, 0), ValueError),
    ((1, -1), (2**53+1, 0), ValueError)])
def test_query_input_contract_even_on_potential_shortcuts(direction, proposal, error):
    with pytest.raises(error):
        certify_gaussian_grouped_directional(_critical(), direction, proposal)


def test_invalid_record_types_raise():
    with pytest.raises(TypeError):
        compile_gaussian_grouped_critical_face(None)
    with pytest.raises(TypeError):
        certify_gaussian_grouped_directional(None, (0,), (0,))


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=20000), RidgeBudget(max_elimination_work=227000)])
def test_compile_preflight_refuses_before_geometry_or_posterior(limits, monkeypatch):
    equilibrium = _equilibrium(budget=limits)
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")
    monkeypatch.setattr(module, "solve_affine", lambda *_: pytest.fail("preflight missed compile geometry"))
    monkeypatch.setattr(module, "_probabilities", lambda *_: pytest.fail("preflight missed posterior"))
    assert compile_gaussian_grouped_critical_face(equilibrium, (0,)) is None


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=30000), RidgeBudget(max_elimination_work=300000)])
def test_query_preflight_refuses_before_matrix_actions(limits, monkeypatch):
    face = _critical(budget=limits)
    module = import_module("omnibias.core.verified.gaussian_grouped_directional")
    monkeypatch.setattr(module, "_linear", lambda *_: pytest.fail("preflight missed query actions"))
    assert certify_gaussian_grouped_directional(face, (1, -1), (0, -4.4)) is None


def test_exact_bits_and_finite_arithmetic_fail_inconclusively():
    face = _critical(budget=RidgeBudget(max_coefficient_bits=64))
    assert certify_gaussian_grouped_directional(face, (1, -10**100), (0, 0)) is None
    assert certify_gaussian_grouped_directional(_critical(), (1, -1), (0, 1e308)) is None
