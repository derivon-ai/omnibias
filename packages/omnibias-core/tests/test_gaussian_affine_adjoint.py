# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact normal lifts, row-coordinate dependence and bounded compilation."""

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
from omnibias.core.verified.gaussian_affine_adjoint import (
    GaussianAffineAdjointGeometry,
    compile_gaussian_affine_adjoint,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _restriction(matrix, rhs=None, *, dimension=None, budget=None):
    if dimension is None:
        dimension = len(matrix[0])
    if rhs is None:
        rhs = (0,)*len(matrix)
    model = GaussianHeatMap(((0,)*dimension,), (0,), budget=budget)
    result = restrict_gaussian_heat(model, matrix, rhs)
    assert result is not None
    return result


def _dot(left, right):
    return sum((a*b for a, b in zip(left, right, strict=True)), F(0))


def _assert_identities(result):
    assert result is not None
    restriction, left = result.restriction, result.left_inverse
    rows, dimension = result.constraint_count, result.dimension
    assert len(left) == rows and all(len(row) == dimension for row in left)
    assert all(isinstance(value, F) for row in left for value in row)
    for i in range(rows):
        for j in range(rows):
            assert _dot(left[i], restriction.constraints[j]) == int(i == j)
    for i in range(dimension):
        for j in range(dimension):
            assert sum((restriction.constraints[k][i]*left[k][j] for k in range(rows)), F(0)) == (
                int(i == j)-restriction.projector[i][j])


@pytest.mark.parametrize("row", [(1, 2), (3, -4, 7), (F(1, 3), F(-2, 7), 0)])
def test_one_row_lift_matches_closed_rational_formula(row):
    restriction = _restriction((row,), (F(2, 3),))
    result = compile_gaussian_affine_adjoint(restriction)
    _assert_identities(result)
    norm_squared = _dot(row, row)
    assert result.left_inverse == (tuple(F(value)/norm_squared for value in row),)
    assert result.tangent_dimension == len(row)-1
    assert result.restriction is restriction
    # L^T d is the minimum-norm anchor; RHS changes do not enter L itself.
    assert tuple(value*F(2, 3) for value in result.left_inverse[0]) == restriction.anchor


def test_two_row_lift_uses_independent_two_by_two_gram_inverse():
    matrix = ((1, 2, -1, 3), (2, -1, 4, 1))
    result = compile_gaussian_affine_adjoint(_restriction(matrix, (F(1, 3), F(-2, 7))))
    _assert_identities(result)
    aa, ab, bb = _dot(matrix[0], matrix[0]), _dot(matrix[0], matrix[1]), _dot(matrix[1], matrix[1])
    determinant = aa*bb-ab*ab
    expected = (tuple((bb*x-ab*y)/determinant for x, y in zip(*matrix, strict=True)),
                tuple((aa*y-ab*x)/determinant for x, y in zip(*matrix, strict=True)))
    assert result.left_inverse == expected
    # Inhomogeneous minimum-norm displacement is L^T delta_rhs and is normal.
    delta = (F(3, 11), F(-5, 13))
    displacement = tuple(sum(result.left_inverse[j][i]*delta[j] for j in range(2)) for i in range(4))
    assert tuple(_dot(row, displacement) for row in matrix) == delta
    assert all(_dot(row, displacement) == 0 for row in result.restriction.projector)


def test_square_constraints_give_inverse_transpose():
    result = compile_gaussian_affine_adjoint(_restriction(((2, 1), (1, 1)), (3, -2)))
    _assert_identities(result)
    assert result.left_inverse == ((1, -1), (-1, 2))
    assert result.tangent_dimension == 0


def test_exact_near_dependence_is_not_a_floating_rank_decision():
    epsilon = F(1, 2**100)
    result = compile_gaussian_affine_adjoint(_restriction(((1, 1), (1, 1+epsilon))))
    _assert_identities(result)
    assert result.left_inverse == (((1+epsilon)/epsilon, -1/epsilon), (-1/epsilon, 1/epsilon))


def test_row_basis_changes_lift_coordinates_but_not_the_normal_projector():
    original = ((1, 2, -1), (0, 1, 2))
    # A=[[2,1],[-1,1]], so L_new=A^(-T)L_old.
    changed = tuple(tuple(a*x+b*y for x, y in zip(*original, strict=True)) for a, b in ((2, 1), (-1, 1)))
    first = compile_gaussian_affine_adjoint(_restriction(original))
    second = compile_gaussian_affine_adjoint(_restriction(changed))
    _assert_identities(first)
    _assert_identities(second)
    expected = tuple(tuple((a*x+b*y)/3 for x, y in zip(*first.left_inverse, strict=True))
                     for a, b in ((1, 1), (-1, 2)))
    assert second.left_inverse == expected
    assert first.restriction.projector == second.restriction.projector


def test_no_constraints_need_no_factorization(monkeypatch):
    restriction = _restriction((), (), dimension=3)

    def fail(*args, **kwargs):
        pytest.fail("empty constraint geometry must not factor a matrix")

    module = import_module("omnibias.core.verified.gaussian_affine_adjoint")
    monkeypatch.setattr(module, "factor_positive_definite", fail)
    result = compile_gaussian_affine_adjoint(restriction)
    _assert_identities(result)
    assert result.left_inverse == () and result.constraint_count == 0
    assert result.dimension == result.tangent_dimension == 3


@pytest.mark.parametrize("matrix", [((0, 0),), ((1, 2), (2, 4)), ((1, 0), (0, 1), (1, 1))])
def test_redundant_rows_refuse_arbitrary_parameter_derivatives(matrix, monkeypatch):
    restriction = _restriction(matrix)

    def fail(*args, **kwargs):
        pytest.fail("redundant rows must be rejected before factorization")

    module = import_module("omnibias.core.verified.gaussian_affine_adjoint")
    monkeypatch.setattr(module, "factor_positive_definite", fail)
    with pytest.raises(ValueError, match="full row rank"):
        compile_gaussian_affine_adjoint(restriction)


def test_geometry_does_not_require_gaussian_contraction():
    source = GaussianHeatMap(((-4, -1), (4, 1)), (0, 0))
    restriction = restrict_gaussian_heat(source, ((0, 1),), (2,))
    assert restriction is not None and restriction.model.contraction_bound == 8
    result = compile_gaussian_affine_adjoint(restriction)
    _assert_identities(result)
    assert result.left_inverse == ((0, 1),)


def test_one_existing_factorization_is_reused_for_every_coordinate(monkeypatch):
    restriction = _restriction(((1, 2, 3, 4), (0, 1, -1, 2)))
    module = import_module("omnibias.core.verified.gaussian_affine_adjoint")
    actual_factor, actual_solve = module.factor_positive_definite, RationalLDLT.solve
    factors, solves = [], []

    def factor(matrix, budget):
        result = actual_factor(matrix, budget)
        factors.append(result)
        return result

    def solve(self, rhs, budget):
        solves.append(self)
        return actual_solve(self, rhs, budget)

    monkeypatch.setattr(module, "factor_positive_definite", factor)
    monkeypatch.setattr(RationalLDLT, "solve", solve)
    _assert_identities(compile_gaussian_affine_adjoint(restriction))
    assert len(factors) == 1 and len(solves) == restriction.dimension
    assert all(value is factors[0] for value in solves)


def test_exact_identity_replay_detects_a_wrong_solve(monkeypatch):
    restriction = _restriction(((1, 2),))
    monkeypatch.setattr(RationalLDLT, "solve", lambda self, rhs, budget: (F(0),)*len(rhs))
    assert compile_gaussian_affine_adjoint(restriction) is None


def test_factory_only_frozen_record_and_independent_certificate_snapshot():
    with pytest.raises(TypeError):
        GaussianAffineAdjointGeometry()
    restriction = _restriction(((1, 2),), (3,))
    result = compile_gaussian_affine_adjoint(restriction)
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.left_inverse = ()
    assert verify_certificate_digest(result.certificate)
    assert certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    saved = deepcopy(result.certificate)
    restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert compile_gaussian_affine_adjoint(restriction) is None
    result.certificate["payload"]["left_inverse"][0][0] = [99, 1]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("bad", [None, 1, (), {}, GaussianHeatMap(((0,),), (0,))])
def test_invalid_restriction_type_raises(bad):
    with pytest.raises(TypeError, match="GaussianAffineRestriction"):
        compile_gaussian_affine_adjoint(bad)


@pytest.mark.parametrize("sticky", [False, True])
def test_exact_compilation_needs_no_backend_and_preserves_history(monkeypatch, sticky):
    restriction = _restriction(((1, 2),))
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = compile_gaussian_affine_adjoint(restriction)
    _assert_identities(result)
    assert result.certificate["meta"]["transcend_backend"] == "not_used"
    assert transcend.libm_fallback_used() is sticky and not transcend.strict_backend()


def test_exact_compilation_respects_shared_strict_sealing_policy(monkeypatch):
    restriction = _restriction(((1, 2),))
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        compile_gaussian_affine_adjoint(restriction)
    assert transcend.strict_backend()


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=1100), RidgeBudget(max_elimination_work=1000)])
def test_metadata_storage_or_work_failure_is_inconclusive(limits):
    restriction = _restriction((), (), dimension=1, budget=limits)
    assert compile_gaussian_affine_adjoint(restriction) is None


def test_gram_coefficient_growth_is_bounded_even_when_the_result_would_fit():
    restriction = _restriction(((2**40,),), budget=RidgeBudget(max_coefficient_bits=64))
    # L=2^-40 fits, but the required Gram intermediate 2^80 does not.
    assert compile_gaussian_affine_adjoint(restriction) is None


def test_factorization_and_solves_share_the_compilation_work_meter(monkeypatch):
    restriction = _restriction(((1, 2), (2, -1)))
    observed = []

    def bounded_work(self, count):
        observed.append(count)
        if count > 3900:
            raise ValueError("test compilation budget exhausted")

    monkeypatch.setattr(RidgeBudget, "_work", bounded_work)
    assert compile_gaussian_affine_adjoint(restriction) is None
    assert observed[0] < 3900 < observed[-1]
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))
