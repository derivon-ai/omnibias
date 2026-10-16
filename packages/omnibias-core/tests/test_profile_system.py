# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Profile quotient identities checked with independent rational polynomials."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from random import Random

import pytest
from omnibias.core._rational_linear import solve_affine
from omnibias.core.operator_spec import OperatorPlanningBudget, OperatorSpec
from omnibias.core.profile_system import ProfilePack, ProfileSystem
from omnibias.core.realization.algebraic import add, divide, multiply, trim
from omnibias.core.ridge_system import RidgeBudget, RidgeSystem


def _oracle_symbol(operator, direction, carrier):
    if operator is None:
        return ()
    result = ()
    for alpha, coefficient in operator.resolve_terms(len(direction)):
        term = (coefficient,)
        for a, w, exponent in zip(carrier, direction, alpha, strict=True):
            for _ in range(exponent):
                term = multiply(term, (Q(a), Q(w)))
        result = add(result, term)
    return result


def _oracle_residual(operators, pack):
    columns = []
    for equation in operators:
        polynomial = ()
        for j, operator in enumerate(equation):
            symbol = _oracle_symbol(operator, pack.direction, pack.carrier)
            q = trim(tuple(row[j] for row in pack.coefficients))
            polynomial = add(polynomial, multiply(symbol, q))
        reduced = divide(polynomial, pack.annihilator)[1]
        columns.append(reduced + (Q(0),) * (pack.profile_order - len(reduced)))
    return tuple(tuple(column[k] for column in columns) for k in range(pack.profile_order))


def test_heat_decay_sine_has_quotient_solutions_despite_empty_arbitrary_profile_kernel():
    heat = OperatorSpec.polynomial({(1, 0): 1, (0, 2): -1})
    assert RidgeSystem.from_operators([[heat]], (0, 1)).kernel_basis(3) == ()
    system = ProfileSystem.from_operators([[heat]], (0, 1), carrier=(-1, 0), annihilator=(1, 0, 1))
    assert system.matrices == (((0,),), ((0,),))
    result = system.solve()
    assert result.consistent and result.rank == 0
    assert result.particular is not None and result.particular.coefficients == ((0,), (0,))
    assert [pack.coefficients for pack in result.homogeneous_basis] == [((1,), (0,)), ((0,), (1,))]
    for pack in result.homogeneous_basis:
        assert system.residual_coefficients(pack) == _oracle_residual([[heat]], pack) == ((0,), (0,))


def test_laplace_exponential_sine_and_compact_high_dimensional_laplacian():
    laplacian = OperatorSpec.laplacian()
    system = ProfileSystem.from_operators([[laplacian]], (0, 1), carrier=(1, 0), annihilator=(1, 0, 1))
    assert len(system.solve().homogeneous_basis) == 2
    for pack in system.solve().homogeneous_basis:
        assert _oracle_residual([[laplacian]], pack) == ((0,), (0,))
    # An ambient polynomial expansion violates the operator's one-term budget.
    compact = OperatorSpec.laplacian(3, planning_budget=OperatorPlanningBudget(max_symbol_terms=1))
    wide = ProfileSystem.from_operators([[compact]], (1,) * 100, annihilator=(1, 0, 1))
    assert wide.matrices == (((-1_000_000,),), ((0,),))
    assert wide.solve().rank == 2


def test_repeated_root_resonance_retains_nilpotent_information():
    derivative_minus_one = OperatorSpec.polynomial({(1,): 1, (0,): -1})
    system = ProfileSystem.from_operators([[derivative_minus_one]], (1,), annihilator=(1, -2, 1))
    # (D-1) has a nontrivial kernel and image modulo (D-1)^2.
    result = system.solve(((-1,), (1,)))
    assert result.consistent and result.rank == 1
    assert result.particular is not None
    assert system.residual_coefficients(result.particular) == ((-1,), (1,))
    assert len(result.homogeneous_basis) == 1
    for pack in (result.particular, *result.homogeneous_basis):
        assert system.residual_coefficients(pack) == _oracle_residual([[derivative_minus_one]], pack)
    assert not system.solve(((1,),)).consistent


def test_repeated_sine_root_supports_forced_elliptic_identity():
    # phi=(sin(z)-z*cos(z))/2 satisfies (D²+1)phi=sin(z), (D²+1)²phi=0.
    laplacian = OperatorSpec.laplacian()
    system = ProfileSystem.from_operators([[laplacian]], (0, 1), carrier=(1, 0),
                                          annihilator=(1, 0, 2, 0, 1))
    result = system.solve(((1,), (0,), (1,), (0,)))
    assert result.consistent and result.rank == 2
    assert result.particular is not None
    assert system.residual_coefficients(result.particular) == ((1,), (0,), (1,), (0,))
    assert _oracle_residual([[laplacian]], result.particular) == ((1,), (0,), (1,), (0,))
    assert len(result.homogeneous_basis) == 2


def test_invertible_forced_screened_poisson_and_high_degree_forcing_reduction():
    screened = OperatorSpec.polynomial({(2,): 1, (0,): -2})
    system = ProfileSystem.from_operators([[screened]], (1,), annihilator=(1, 0, 1))
    # s^4 + 2s^3 = 1-2s in this quotient. Operator multiplier is -3.
    result = system.solve(((0,), (0,), (0,), (2,), (1,)))
    assert result.consistent and result.rank == 2 and not result.homogeneous_basis
    assert result.particular is not None
    assert result.particular.coefficients == ((Q(-1, 3),), (Q(2, 3),))
    assert system.residual_coefficients(result.particular) == ((1,), (-2,))


def test_inconsistent_forcing_is_restricted_and_still_returns_homogeneous_kernel():
    system = ProfileSystem.from_operators([[OperatorSpec.mixed((1,))]], (1,), annihilator=(0, 1))
    result = system.solve(((1,),))
    assert not result.consistent and result.particular is None and result.rank == 0
    assert len(result.homogeneous_basis) == 1
    assert result.homogeneous_basis[0].coefficients == ((1,),)
    assert system.residual_coefficients(result.homogeneous_basis[0]) == ((0,),)


@pytest.mark.parametrize("annihilator", [(1, 0, 1), (0, 0, 1), (1, -2, 1), (2, 0, 2)])
def test_random_coupled_systems_match_independent_polynomial_division(annihilator):
    rng = Random(726)
    operators = [[OperatorSpec.polynomial({(2, 0): rng.randrange(-3, 4),
                                          (0, 1): Q(rng.randrange(1, 4), 3),
                                          (1, 1): rng.randrange(-3, 4),
                                          (0, 0): rng.randrange(-3, 4)}) for _ in range(2)] for _ in range(2)]
    system = ProfileSystem.from_operators(operators, (Q(1, 2), -2), carrier=(Q(2, 3), 1),
                                          annihilator=annihilator)
    for i, row in enumerate(operators):
        for c, operator in enumerate(row):
            expected = divide(_oracle_symbol(operator, system.direction, system.carrier), system.annihilator)[1]
            assert tuple(matrix[i][c] for matrix in system.matrices) == expected + (0,) * (2 - len(expected))
    planted = ProfilePack(system.direction, system.carrier, system.annihilator, ((Q(2, 5), -1), (3, Q(-2, 7))))
    forcing = _oracle_residual(operators, planted)
    assert system.residual_coefficients(planted) == forcing
    result = system.solve(forcing)
    assert result.consistent and result.particular is not None
    assert _oracle_residual(operators, result.particular) == forcing
    assert result.rank + len(result.homogeneous_basis) == 4
    for pack in result.homogeneous_basis:
        assert _oracle_residual(operators, pack) == ((0, 0), (0, 0))


def test_zero_rectangular_system_has_full_kernel_and_canonical_zero_particular():
    system = ProfileSystem.from_operators([[None, None, None], [None, None, None]], (1,), annihilator=(1, 0, 1))
    result = system.solve()
    assert result.consistent and result.rank == 0 and len(result.homogeneous_basis) == 6
    assert result.particular is not None
    assert result.particular.coefficients == ((0, 0, 0), (0, 0, 0))
    assert system.solve(()).consistent


def test_monic_canonicalization_and_frozen_pack_metadata():
    pack = ProfilePack((1,), (0,), (2, 0, 2, 0), ((1,), (0,)))
    assert pack.annihilator == (1, 0, 1)
    assert pack.profile_order == 2 and pack.outputs == 1
    with pytest.raises(FrozenInstanceError):
        pack.carrier = (Q(1),)


@pytest.mark.parametrize("annihilator", [(), (0,), (1,), (0, 0), (1.0, 1), (False, 1)])
def test_invalid_annihilators_are_rejected(annihilator):
    with pytest.raises((TypeError, ValueError)):
        ProfileSystem.from_operators([[None]], (1,), annihilator=annihilator)


@pytest.mark.parametrize("direction,carrier", [((), ()), ((0,), (0,)), ((1.0,), (0,)),
                                             ((True,), (0,)), ((1,), (0, 1)), ((1,), (False,))])
def test_invalid_direction_and_carrier_are_rejected(direction, carrier):
    with pytest.raises((TypeError, ValueError)):
        ProfileSystem.from_operators([[None]], direction, carrier=carrier, annihilator=(1, 0, 1))


@pytest.mark.parametrize("operators", [[], [[]], [[None], [None, None]], [[1]]])
def test_invalid_operator_arrays_are_rejected(operators):
    with pytest.raises((TypeError, ValueError)):
        ProfileSystem.from_operators(operators, (1,), annihilator=(1, 0, 1))


def test_operator_dimensions_and_ambiguous_directions_are_checked():
    with pytest.raises(ValueError, match="dimension"):
        ProfileSystem.from_operators([[OperatorSpec.mixed((1, 0))]], (1,), annihilator=(1, 0, 1))
    with pytest.raises(ValueError, match="directional"):
        ProfileSystem.from_operators([[OperatorSpec.directional(1)]], (1,), annihilator=(1, 0, 1))


@pytest.mark.parametrize("coefficients", [(), ((1,),), ((1,), (1, 2)), ((True,), (0,)), ((1.0,), (0,))])
def test_invalid_pack_shapes_and_coefficients_are_rejected(coefficients):
    with pytest.raises((TypeError, ValueError)):
        ProfilePack((1,), (0,), (1, 0, 1), coefficients)


def test_forcing_metadata_and_pack_compatibility_are_checked():
    system = ProfileSystem.from_operators([[None]], (1,), annihilator=(1, 0, 1))
    for forcing in (((1, 2),), ((True,),), ((1.0,),)):
        with pytest.raises((TypeError, ValueError)):
            system.solve(forcing)
    for pack in (ProfilePack((2,), (0,), (1, 0, 1), ((1,), (0,))),
                 ProfilePack((1,), (1,), (1, 0, 1), ((1,), (0,))),
                 ProfilePack((1,), (0,), (0, 0, 1), ((1,), (0,))),
                 ProfilePack((1,), (0,), (1, 0, 1), ((1, 0), (0, 1)))):
        with pytest.raises(ValueError, match="match"):
            system.residual_coefficients(pack)


def test_degree_order_metadata_and_normalization_growth_budgets():
    with pytest.raises(ValueError, match="degree"):
        ProfileSystem.from_operators([[None]], (1,), annihilator=(1, 0, 1), budget=RidgeBudget(max_degree=0))
    with pytest.raises(ValueError, match="annihilator order"):
        ProfileSystem.from_operators([[None]], (1,), annihilator=(1, 0, 1), budget=RidgeBudget(max_order=1))
    with pytest.raises(ValueError, match="operator order"):
        ProfileSystem.from_operators([[OperatorSpec.mixed((3,))]], (1,), annihilator=(1, 0, 1),
                                     budget=RidgeBudget(max_order=2))
    with pytest.raises(ValueError, match="bit budget"):
        ProfileSystem.from_operators([[None]], (8,), annihilator=(1, 0, 1),
                                     budget=RidgeBudget(max_coefficient_bits=3))
    # Each source fraction fits three bits, but monic normalization creates 49/6.
    with pytest.raises(ValueError, match="bit budget"):
        ProfileSystem.from_operators([[None]], (1,), annihilator=(Q(7, 3), Q(2, 7)),
                                     budget=RidgeBudget(max_coefficient_bits=3))


def test_matrix_output_metadata_and_work_budgets():
    # System metadata fits; augmented elimination storage does not.
    system = ProfileSystem.from_operators([[None, None]], (1,), annihilator=(1, 0, 1),
                                          budget=RidgeBudget(max_matrix_entries=8))
    with pytest.raises(ValueError, match="matrix entries"):
        system.solve(((1,),))
    # Even the homogeneous basis metadata (normal, carrier, modulus) counts.
    zero = ProfileSystem.from_operators([[None]], (1,), annihilator=(1, 0, 1),
                                        budget=RidgeBudget(max_matrix_entries=12))
    with pytest.raises(ValueError, match="matrix entries"):
        zero.solve()
    with pytest.raises(ValueError, match="work"):
        ProfileSystem.from_operators([[None]], (1,) * 100, annihilator=(1, 0, 1),
                                     budget=RidgeBudget(max_elimination_work=100))
    limited = ProfileSystem.from_operators([[None]], (1,), annihilator=(1, 0, 1),
                                           budget=RidgeBudget(max_elimination_work=40))
    with pytest.raises(ValueError, match="work"):
        limited.solve()


def test_homogeneous_affine_solver_counts_its_zero_particular_storage():
    matrix = ((Q(1), Q(0), Q(0), Q(0)), (Q(0), Q(1), Q(0), Q(0)))
    with pytest.raises(ValueError, match="matrix entries"):
        solve_affine(matrix, None, RidgeBudget(max_matrix_entries=8))


def test_modular_reduction_and_affine_elimination_growth_are_checked():
    # s² becomes 3s-1; s³ then produces a coefficient exceeding two bits.
    with pytest.raises(ValueError, match="bit budget"):
        ProfileSystem.from_operators([[OperatorSpec.mixed((3,))]], (1,), annihilator=(1, -3, 1),
                                     budget=RidgeBudget(max_coefficient_bits=2))
    system = ProfileSystem((1,), (0,), (0, 1), (((2, 3), (3, 2)),),
                           RidgeBudget(max_coefficient_bits=2))
    with pytest.raises(ValueError, match="bit budget"):
        system.solve()


def test_forcing_and_residual_resource_limits():
    system = ProfileSystem.from_operators([[None]], (1,), annihilator=(0, 1),
                                          budget=RidgeBudget(max_order=1, max_coefficient_bits=2))
    with pytest.raises(ValueError, match="forcing order"):
        system.solve(((0,), (0,), (1,)))
    with pytest.raises(ValueError, match="bit budget"):
        system.solve(((8,),))
    with pytest.raises(ValueError, match="bit budget"):
        system.residual_coefficients(ProfilePack((1,), (0,), (0, 1), ((8,),)))
