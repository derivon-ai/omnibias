# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact PDE identities checked independently in multivariate polynomial space."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from math import factorial

import pytest
from omnibias.core.operator_spec import OperatorPlanningBudget, OperatorSpec
from omnibias.core.ridge_system import RidgeBudget, RidgeSystem, RidgeSyzygy

Polynomial = dict[tuple[int, ...], Fraction]


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for alpha, value in left.items():
        for beta, other in right.items():
            index = tuple(a + b for a, b in zip(alpha, beta, strict=True))
            result[index] = result.get(index, Fraction(0)) + value * other
    return result


def _differentiate(polynomial: Polynomial, alpha: tuple[int, ...]) -> Polynomial:
    result: Polynomial = {}
    for powers, value in polynomial.items():
        if any(p < a for p, a in zip(powers, alpha, strict=True)):
            continue
        index = tuple(p - a for p, a in zip(powers, alpha, strict=True))
        for p, a in zip(powers, alpha, strict=True):
            value *= factorial(p) // factorial(p - a)
        result[index] = value
    return result


def _profile_fields(pack: RidgeSyzygy, bias: Fraction) -> list[Polynomial]:
    """Expand a degree-seven scalar polynomial in actual ambient variables."""
    zero = (0,) * len(pack.direction)
    z = {zero: bias}
    for i, weight in enumerate(pack.direction):
        z[tuple(int(i == j) for j in range(len(zero)))] = weight
    powers = [{zero: Fraction(1)}]
    for _ in range(7):
        powers.append(_multiply(powers[-1], z))
    profile = [Fraction((-1) ** k, k + 1) for k in range(8)]
    fields: list[Polynomial] = [{} for _ in range(pack.outputs)]
    for derivative, row in enumerate(pack.coefficients):
        for degree in range(derivative, len(profile)):
            factor = profile[degree] * factorial(degree) / factorial(degree - derivative)
            for output, coefficient in enumerate(row):
                for alpha, value in powers[degree - derivative].items():
                    fields[output][alpha] = fields[output].get(alpha, Fraction(0)) + coefficient * factor * value
    return fields


def _assert_multivariate_pde(
    operators: list[list[OperatorSpec | None]], pack: RidgeSyzygy,
) -> None:
    fields = _profile_fields(pack, Fraction(2, 3))
    for equation in operators:
        residual: Polynomial = {}
        for operator, field in zip(equation, fields, strict=True):
            if operator is None:
                continue
            for alpha, coefficient in operator.resolve_terms(len(pack.direction)):
                for index, value in _differentiate(field, alpha).items():
                    residual[index] = residual.get(index, Fraction(0)) + coefficient * value
        assert not any(residual.values())


def _wave_operators() -> list[list[OperatorSpec | None]]:
    identity = OperatorSpec.mixed((0, 0))
    dt = OperatorSpec.mixed((1, 0))
    minus_dt = OperatorSpec.polynomial({(1, 0): -1})
    minus_dx = OperatorSpec.polynomial({(0, 1): -1})
    return [[minus_dt, identity, None],
            [minus_dx, None, identity],
            [None, dt, OperatorSpec.polynomial({(0, 1): -4})]]


def test_mixed_order_pack_requires_derivative_coupling() -> None:
    operators = [[OperatorSpec.mixed((2, 0)), OperatorSpec.mixed((0, 1))]]
    system = RidgeSystem.from_operators(operators, (1, 1))
    assert system.order == 2
    assert system.equations == 1
    assert system.outputs == 2
    assert system.kernel_basis() == ()
    basis = system.kernel_basis(1)
    assert len(basis) == 1
    assert basis[0].coefficients == ((-1, 0), (0, 1))
    assert system.residual_coefficients(basis[0]) == ((0,),) * 4
    _assert_multivariate_pde(operators, basis[0])


@pytest.mark.parametrize("direction", [(2, 1), (-2, 1), (Fraction(2, 3), Fraction(1, 3))])
def test_coupled_wave_characteristic_and_polynomial_shifts(direction: tuple[Fraction | int, ...]) -> None:
    operators = _wave_operators()
    system = RidgeSystem.from_operators(operators, direction)
    assert system.kernel_basis() == ()
    basis = system.kernel_basis(2)
    assert len(basis) == 2
    assert [pack.degree for pack in basis] == [1, 2]
    for pack in basis:
        assert not any(value for row in system.residual_coefficients(pack) for value in row)
        _assert_multivariate_pde(operators, pack)
    if direction == (2, 1):
        assert basis[0].coefficients == ((1, 0, 0), (0, 2, 1))
        assert basis[1].coefficients == ((0, 0, 0), (1, 0, 0), (0, 2, 1))


def test_noncharacteristic_wave_and_heat_have_no_universal_pack() -> None:
    assert RidgeSystem.from_operators(_wave_operators(), (3, 1)).kernel_basis(3) == ()
    # Heat allows some exponential profiles, but arbitrary phi is not allowed.
    heat = OperatorSpec.polynomial({(1, 0): 1, (0, 2): -1})
    system = RidgeSystem.from_operators([[heat]], (1, 1))
    assert system.matrices == (((0,),), ((1,),), ((-1,),))
    assert system.kernel_basis(4) == ()


def test_degree_zero_divergence_kernel_is_activation_independent() -> None:
    operators = [[OperatorSpec.mixed((1, 0)), OperatorSpec.mixed((0, 1))]]
    system = RidgeSystem.from_operators(operators, (2, 3))
    (pack,) = system.kernel_basis()
    assert pack.coefficients == ((Fraction(-3, 2), 1),)
    _assert_multivariate_pde(operators, pack)


def test_zero_system_returns_all_bounded_polynomial_vectors() -> None:
    system = RidgeSystem.from_operators([[None, None], [None, None]], (1,))
    basis = system.kernel_basis(2)
    assert system.order == 0
    assert len(basis) == 6
    assert [pack.degree for pack in basis] == [0, 0, 1, 1, 2, 2]
    for pack in basis:
        assert system.residual_coefficients(pack) == ((0, 0),) * (pack.degree + 1)


def test_symbol_cancellation_retains_zero_matrix_shape() -> None:
    operator = OperatorSpec.polynomial({(2, 0): 1, (0, 2): -1})
    system = RidgeSystem.from_operators([[operator]], (1, 1))
    assert system.matrices == (((0,),),)
    assert system.kernel_basis()[0].coefficients == ((1,),)


def test_laplacian_power_is_compact_and_order_zero_is_identity() -> None:
    tiny_symbol_budget = OperatorPlanningBudget(max_symbol_terms=1)
    laplacian = OperatorSpec.laplacian(3, planning_budget=tiny_symbol_budget)
    system = RidgeSystem.from_operators([[laplacian]], (1,) * 100)
    assert system.order == 6
    assert system.matrices[-1] == ((1_000_000,),)
    assert not any(v for matrix in system.matrices[:-1] for row in matrix for v in row)
    identity = RidgeSystem.from_operators([[OperatorSpec.laplacian(0)]], (2, 3))
    assert identity.matrices == (((1,),),)


def test_residual_coefficients_reports_nonzero_identity() -> None:
    system = RidgeSystem.from_operators([[OperatorSpec.mixed((2,)), OperatorSpec.mixed((0,))]], (2,))
    pack = RidgeSyzygy((2,), ((1, 3), (2, 0)))
    assert system.residual_coefficients(pack) == ((3,), (0,), (4,), (8,))


def test_exact_metadata_is_immutable_and_canonical() -> None:
    pack = RidgeSyzygy((1, 2), ((1, 0), (0, 0)))
    assert pack.direction == (Fraction(1), Fraction(2))
    assert pack.degree == 0
    assert pack.outputs == 2
    assert hash(pack)
    with pytest.raises(FrozenInstanceError):
        pack.direction = (Fraction(3),)  # type: ignore[misc]


@pytest.mark.parametrize("direction", [(), (0, 0), (1.0,), (True,)])
def test_invalid_directions_are_rejected(direction: tuple[object, ...]) -> None:
    with pytest.raises((ValueError, TypeError)):
        RidgeSystem.from_operators([[None]], direction)  # type: ignore[arg-type]


@pytest.mark.parametrize("operators", [[], [[]], [[None], [None, None]], [[1]]])
def test_invalid_operator_matrices_are_rejected(operators: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        RidgeSystem.from_operators(operators, (1,))  # type: ignore[arg-type]


def test_dimension_and_directional_operators_are_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        RidgeSystem.from_operators([[OperatorSpec.mixed((1, 0))]], (1,))
    with pytest.raises(ValueError, match="directional"):
        RidgeSystem.from_operators([[OperatorSpec.directional(1)]], (1,))


@pytest.mark.parametrize("coefficients", [(), ((),), ((1,), (1, 2)), ((1.0,),), ((True,),)])
def test_invalid_pack_coefficients_are_rejected(coefficients: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        RidgeSyzygy((1,), coefficients)  # type: ignore[arg-type]


def test_residual_compatibility_is_checked() -> None:
    system = RidgeSystem.from_operators([[None]], (1,))
    for pack in (RidgeSyzygy((2,), ((1,),)), RidgeSyzygy((1,), ((1, 2),))):
        with pytest.raises(ValueError, match="match"):
            system.residual_coefficients(pack)


@pytest.mark.parametrize("degree", [-1, True, 1.5])
def test_invalid_kernel_degree_is_rejected(degree: object) -> None:
    with pytest.raises(ValueError):
        RidgeSystem.from_operators([[None]], (1,)).kernel_basis(degree)  # type: ignore[arg-type]


def test_resource_limits_cover_matrix_work_degree_order_and_bits() -> None:
    operators = [[OperatorSpec.mixed((2,)), OperatorSpec.mixed((1,))]]
    entries = RidgeSystem.from_operators(operators, (1,), budget=RidgeBudget(max_matrix_entries=8))
    with pytest.raises(ValueError, match="matrix entries"):
        entries.kernel_basis(1)
    work = RidgeSystem.from_operators(operators, (1,), budget=RidgeBudget(max_elimination_work=100))
    with pytest.raises(ValueError, match="work"):
        work.kernel_basis(1)
    degree = RidgeSystem.from_operators([[None]], (1,), budget=RidgeBudget(max_degree=0))
    with pytest.raises(ValueError, match="degree"):
        degree.kernel_basis(1)
    with pytest.raises(ValueError, match="order"):
        RidgeSystem.from_operators(operators, (1,), budget=RidgeBudget(max_order=1))
    with pytest.raises(ValueError, match="bit budget"):
        RidgeSystem.from_operators([[None]], (8,), budget=RidgeBudget(max_coefficient_bits=3))
    # Input values fit two bits; rational elimination creates a larger product.
    growth = RidgeSystem((1,), (((2, 3), (3, 2)),), RidgeBudget(max_coefficient_bits=2))
    with pytest.raises(ValueError, match="bit budget"):
        growth.kernel_basis()


def test_basis_output_storage_has_an_independent_limit() -> None:
    system = RidgeSystem.from_operators([[None] * 4], (1,), budget=RidgeBudget(max_matrix_entries=8))
    with pytest.raises(ValueError, match="matrix entries"):
        system.kernel_basis()


def test_restriction_work_accounts_for_every_laplacian_and_zero_system_direction() -> None:
    laplacian = OperatorSpec.laplacian()
    with pytest.raises(ValueError, match="work"):
        RidgeSystem.from_operators([[laplacian, laplacian]], (1,) * 100,
                                   budget=RidgeBudget(max_elimination_work=400))
    with pytest.raises(ValueError, match="work"):
        RidgeSystem.from_operators([[None]], (1,) * 100,
                                   budget=RidgeBudget(max_elimination_work=99))


@pytest.mark.parametrize("budget", [RidgeBudget(max_degree=0), RidgeBudget(max_coefficient_bits=2)])
def test_residual_resource_limits_are_checked(budget: RidgeBudget) -> None:
    system = RidgeSystem.from_operators([[None]], (1,), budget=budget)
    pack = (RidgeSyzygy((1,), ((0,), (1,))) if budget.max_degree == 0
            else RidgeSyzygy((1,), ((8,),)))
    with pytest.raises(ValueError, match="budget"):
        system.residual_coefficients(pack)


@pytest.mark.parametrize("kwargs", [{"max_degree": -1}, {"max_order": True}, {"max_matrix_entries": 0}])
def test_invalid_budgets_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        RidgeBudget(**kwargs)
