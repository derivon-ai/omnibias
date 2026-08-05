# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for Buckingham-Pi dimensional analysis (exact integer null-space)."""

from __future__ import annotations

import pytest
from omnibias.symbolic.dimensional import (
    DimensionalSystem,
    PiGroup,
    buckingham_pi_groups,
    dimension_matrix,
    dimensionless_residual,
    filter_dimensionless_monomials,
    integer_null_space,
    is_dimensionless,
    n_dimensionless_groups,
)


def _reynolds_system() -> DimensionalSystem:
    return DimensionalSystem.from_dimensions(
        {
            "rho": {"M": 1, "L": -3},
            "U": {"L": 1, "T": -1},
            "L": {"L": 1},
            "mu": {"M": 1, "L": -1, "T": -1},
        },
        base_dimensions=["M", "L", "T"],
    )


def _pendulum_system() -> DimensionalSystem:
    return DimensionalSystem.from_dimensions(
        {
            "t": {"T": 1},
            "L": {"L": 1},
            "g": {"L": 1, "T": -2},
            "m": {"M": 1},
        },
        base_dimensions=["M", "L", "T"],
    )


def _drag_system() -> DimensionalSystem:
    return DimensionalSystem.from_dimensions(
        {
            "F": {"M": 1, "L": 1, "T": -2},
            "rho": {"M": 1, "L": -3},
            "U": {"L": 1, "T": -1},
            "D": {"L": 1},
            "mu": {"M": 1, "L": -1, "T": -1},
        },
        base_dimensions=["M", "L", "T"],
    )


def _sign_canonical(vec: dict[str, int]) -> dict[str, int]:
    """Flip the global sign so the comparison is sign-agnostic."""
    for value in vec.values():
        if value != 0:
            if value < 0:
                return {k: -v for k, v in vec.items()}
            return vec
    return vec


# --------------------------------------------------------------------------- #
# Reynolds number                                                             #
# --------------------------------------------------------------------------- #
def test_reynolds_group_is_recovered() -> None:
    system = _reynolds_system()
    assert n_dimensionless_groups(system) == 1
    groups = buckingham_pi_groups(system)
    assert len(groups) == 1
    recovered = _sign_canonical(groups[0].as_dict())
    assert recovered == {"rho": 1, "U": 1, "L": 1, "mu": -1}
    assert is_dimensionless(system, groups[0].as_dict())


def test_reynolds_dimension_matrix_shape_and_values() -> None:
    system = _reynolds_system()
    mat = dimension_matrix(system)
    assert mat == [[1, 0, 0, 1], [-3, 1, 1, -1], [0, -1, 0, -1]]


# --------------------------------------------------------------------------- #
# Pendulum period                                                            #
# --------------------------------------------------------------------------- #
def test_pendulum_group_excludes_mass() -> None:
    system = _pendulum_system()
    assert n_dimensionless_groups(system) == 1
    group = buckingham_pi_groups(system)[0]
    recovered = _sign_canonical(group.as_dict())
    # t^2 g / L is dimensionless; the period is independent of mass m.
    assert recovered == {"t": 2, "L": -1, "g": 1, "m": 0}
    assert recovered["m"] == 0
    assert is_dimensionless(system, group.as_dict())


# --------------------------------------------------------------------------- #
# multiple groups                                                            #
# --------------------------------------------------------------------------- #
def test_drag_has_two_groups_containing_reynolds_and_drag_coefficient() -> None:
    system = _drag_system()
    assert n_dimensionless_groups(system) == 2
    groups = buckingham_pi_groups(system)
    assert len(groups) == 2
    for group in groups:
        assert is_dimensionless(system, group.as_dict())
    # the classic groups are dimensionless in this system
    assert is_dimensionless(system, {"rho": 1, "U": 1, "D": 1, "mu": -1})  # Reynolds
    assert is_dimensionless(system, {"F": 1, "rho": -1, "U": -2, "D": -2})  # drag coeff


# --------------------------------------------------------------------------- #
# null-space algebra                                                          #
# --------------------------------------------------------------------------- #
def test_integer_null_space_vectors_annihilate_matrix_and_are_primitive() -> None:
    import math

    system = _drag_system()
    mat = dimension_matrix(system)
    basis = integer_null_space(mat)
    assert len(basis) == 2
    for vec in basis:
        # D @ vec == 0
        for row in mat:
            assert sum(a * b for a, b in zip(row, vec, strict=True)) == 0
        # primitive: gcd of entries is 1
        g = 0
        for value in vec:
            g = math.gcd(g, abs(value))
        assert g == 1
        # canonical sign: first nonzero entry positive
        first = next(v for v in vec if v != 0)
        assert first > 0


def test_integer_null_space_handles_full_rank_and_empty() -> None:
    assert integer_null_space([[1, 0], [0, 1]]) == []  # trivial null space
    assert integer_null_space([]) == []


def test_dimensionless_residual_reports_net_dimension() -> None:
    system = _pendulum_system()
    # t alone has net dimension T^1 -> (M,L,T) = (0,0,1)
    assert dimensionless_residual(system, {"t": 1}) == (0, 0, 1)
    assert not is_dimensionless(system, {"t": 1})
    # accepts a positional sequence aligned to variable order
    assert dimensionless_residual(system, [2, -1, 1, 0]) == (0, 0, 0)


# --------------------------------------------------------------------------- #
# library filter + formatting + validation                                   #
# --------------------------------------------------------------------------- #
def test_filter_dimensionless_monomials_keeps_only_pi_products() -> None:
    system = _pendulum_system()
    candidates = [
        {"t": 1},
        {"t": 2, "g": 1, "L": -1},
        {"t": 1, "g": 1},
        {"L": 1, "g": -1, "t": 2},
    ]
    kept = filter_dimensionless_monomials(system, candidates)
    assert kept == [{"t": 2, "g": 1, "L": -1}]


def test_pi_group_formula_renders_fraction() -> None:
    group = PiGroup(powers=(("rho", 1), ("U", 1), ("L", 1), ("mu", -1)))
    assert group.formula() == "rho * U * L / mu"
    assert PiGroup(powers=(("a", 2), ("b", -1), ("c", -1))).formula() == "a^2 / (b * c)"
    assert PiGroup(powers=(("x", 0),)).formula() == "1"


def test_from_dimensions_infers_sorted_base_dimensions() -> None:
    system = DimensionalSystem.from_dimensions({"v": {"L": 1, "T": -1}, "m": {"M": 1}})
    assert system.base_dimensions == ("L", "M", "T")


def test_invalid_exponent_rows_raise() -> None:
    with pytest.raises(ValueError):
        DimensionalSystem(
            variable_names=("a", "b"),
            base_dimensions=("M", "L"),
            exponents=((1, 0),),  # only one row for two variables
        )
    with pytest.raises(ValueError):
        DimensionalSystem(
            variable_names=("a",),
            base_dimensions=("M", "L"),
            exponents=((1, 0, 0),),  # row longer than base_dimensions
        )
