# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent exact geometric oracles and failure contracts."""

from __future__ import annotations

import ast
import random
from fractions import Fraction as F
from pathlib import Path

import pytest
from omnibias.core.affine_geometry import (
    AffineBudget,
    AffineBudgetExceeded,
    ExactAffineGeometry,
    InconsistentAffineConstraints,
    exact_affine_geometry,
)


def test_single_plane_analytic_projection_and_distance() -> None:
    geometry = exact_affine_geometry(((2, -1, 2),), (3,))
    assert geometry.consistent and geometry.rank == 1
    assert geometry.anchor == (F(2, 3), F(-1, 3), F(2, 3))
    point = (F(1, 2), F(-3, 2), F(7, 4))
    residual = sum(a * x for a, x in zip((2, -1, 2), point, strict=True)) - 3
    expected = tuple(x - residual * a / 9 for a, x in zip((2, -1, 2), point, strict=True))
    assert geometry.project(point) == expected
    assert geometry.squared_distance(point) == residual * residual / 9
    assert geometry.half_squared_distance(point) == residual * residual / 18
    assert geometry.contains(expected)
    assert not geometry.contains(point)
    assert geometry.project(expected) == expected


def test_redundant_rows_and_nonorthogonal_affine_line() -> None:
    # (x,y,z)=(t,1-2t,3+t). Analytic nearest parameter follows a 1-D quadratic.
    geometry = exact_affine_geometry(((2, 1, 0), (-1, 0, 1), (4, 2, 0)), (1, 3, 2))
    point = (F(2), F(-5), F(1))
    direction, origin = (1, -2, 1), (0, 1, 3)
    t = sum(v * (x - a) for x, a, v in zip(point, origin, direction, strict=True)) / F(6)
    expected = tuple(F(a) + t * v for a, v in zip(origin, direction, strict=True))
    assert geometry.rank == 2 and geometry.tangent_dimension == 1
    assert geometry.project(point) == expected
    assert geometry.squared_distance(point) == sum((x - y) ** 2 for x, y in zip(point, expected, strict=True))


def test_empty_full_rank_and_zero_rows() -> None:
    full = exact_affine_geometry((), (), dimension=3)
    assert full.rank == 0 and full.anchor == (0, 0, 0)
    assert full.contains((2, 4, -7)) and full.project((2, 4, -7)) == (2, 4, -7)
    assert full.squared_distance((2, 4, -7)) == 0
    point = exact_affine_geometry(((1, 2), (3, 5)), (3, 8))
    assert point.project((10, -20)) == (1, 1)
    assert point.tangent_projector == ((0, 0), (0, 0))
    assert point.squared_distance((10, -20)) == 9**2 + 21**2
    zero = exact_affine_geometry(((0, 0), (0, 0)), (0, 0))
    assert zero.contains((12, 3)) and zero.project((12, 3)) == (12, 3)


@pytest.mark.parametrize("matrix,rhs", [(((1, 2), (2, 4)), (1, 3)), (((0, 0),), (1,))])
def test_inconsistency_is_a_proven_verdict_not_a_resource_refusal(matrix: tuple, rhs: tuple) -> None:
    geometry = exact_affine_geometry(matrix, rhs)
    assert not geometry.consistent
    assert geometry.anchor is None and geometry.tangent_projector is None
    assert not geometry.contains((0, 0))
    with pytest.raises(InconsistentAffineConstraints):
        geometry.project((0, 0))
    with pytest.raises(InconsistentAffineConstraints):
        geometry.squared_distance((0, 0))


def test_equation_recombination_and_scaling_invariance() -> None:
    original = exact_affine_geometry(((2, 1, 0), (-1, 0, 1)), (1, 3))
    # S=[[2,3],[-1,4]], det(S)=11. Include a redundant extra equation.
    mixed = exact_affine_geometry(((1, 2, 3), (-6, -1, 4), (F(2, 7), F(1, 7), 0)),
                                 (11, 11, F(1, 7)))
    assert mixed.anchor == original.anchor
    assert mixed.tangent_projector == original.tangent_projector
    assert mixed.rank == original.rank
    for point in ((1, 2, 3), (F(2, 5), F(-3, 7), 11)):
        assert mixed.project(point) == original.project(point)
        assert mixed.squared_distance(point) == original.squared_distance(point)


def test_random_pythagorean_oracle() -> None:
    rng = random.Random(101)
    for _ in range(20):
        normal = tuple(rng.randrange(1, 8) for _ in range(3))
        anchor = tuple(F(rng.randrange(-5, 6), rng.randrange(1, 5)) for _ in range(3))
        rhs = sum(a * x for a, x in zip(normal, anchor, strict=True))
        geometry = exact_affine_geometry((normal,), (rhs,))
        point = tuple(F(rng.randrange(-8, 9), 3) for _ in range(3))
        projection = geometry.project(point)
        # An independent feasible competitor formed using n-orthogonal v.
        tangent = (normal[1], -normal[0], 0)
        other = tuple(a + 2 * v for a, v in zip(anchor, tangent, strict=True))
        assert geometry.contains(other)
        d2 = geometry.squared_distance(point)
        assert sum((x - y)**2 for x, y in zip(point, other, strict=True)) == (
            d2 + sum((x - y)**2 for x, y in zip(projection, other, strict=True)))


@pytest.mark.parametrize("matrix,rhs,kwargs,error", [
    ((), (), {}, ValueError),
    (((1, 2),), (1,), {"dimension": 3}, ValueError),
    (((1, 2), (3,)), (1, 2), {}, ValueError),
    (((1, 2),), (), {}, ValueError),
    (((1.0, 2),), (1,), {}, TypeError),
    (((True, 2),), (1,), {}, TypeError),
    (("12",), (1,), {}, TypeError),
    (((1, 2),), (1,), {"budget": 5}, TypeError),
])
def test_invalid_inputs(matrix: tuple, rhs: tuple, kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        exact_affine_geometry(matrix, rhs, **kwargs)


def test_query_input_and_construction_are_sealed() -> None:
    geometry = exact_affine_geometry(((1, 2),), (3,))
    for point in ((1,), (1.0, 1), (True, 1)):
        with pytest.raises((ValueError, TypeError)):
            geometry.project(point)
    with pytest.raises(TypeError, match="use exact_affine_geometry"):
        ExactAffineGeometry()
    with pytest.raises((AttributeError, TypeError)):
        geometry.rank = 100  # type: ignore[misc]


@pytest.mark.parametrize("budget", [
    AffineBudget(max_dimension=1),
    AffineBudget(max_rows=1),
    AffineBudget(max_entries=1),
    AffineBudget(max_work=1),
    AffineBudget(max_coefficient_bits=1),
])
def test_explicit_resource_refusals(budget: AffineBudget) -> None:
    with pytest.raises(AffineBudgetExceeded):
        exact_affine_geometry(((2, 1), (1, 3)), (1, 2), budget=budget)


def test_intermediate_coefficient_growth_and_query_limits() -> None:
    # Inputs fit three bits; projection's 1/50 denominator does not.
    with pytest.raises(AffineBudgetExceeded, match="coefficient"):
        exact_affine_geometry(((5, 5),), (1,), budget=AffineBudget(max_coefficient_bits=3))
    geometry = exact_affine_geometry(((1, 0),), (0,), budget=AffineBudget(max_coefficient_bits=4))
    with pytest.raises(AffineBudgetExceeded, match="coefficient"):
        geometry.contains((32, 0))
    with pytest.raises(AffineBudgetExceeded, match="coefficient"):
        geometry.squared_distance((4, 0))  # squared intermediate 16 exceeds four bits


def test_core_module_has_no_backend_or_gaussian_runtime_dependencies() -> None:
    import omnibias.core.affine_geometry as module

    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and ("gaussian" in name or name.startswith(("torch", "jax", "numpy", "sympy")))
                   for name in imports)
