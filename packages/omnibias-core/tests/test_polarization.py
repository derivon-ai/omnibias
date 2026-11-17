# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Polarization plans reproduce mixed partial derivatives of polynomials exactly (rational arithmetic)."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import factorial

import pytest
from omnibias.core.polarization import MAX_POLARIZED_ORDER, polarization_plan


def _monomial_dir_derivative(coeffs, direction, n):
    """``D^n_w`` of ``p(x) = sum c_beta x^beta`` at ``x = 0``: ``n!`` times the degree-``n`` part evaluated at ``w`` / ``n!``..."""
    # p's n-th directional derivative at 0 is n! * (homogeneous degree-n part of p)(w).
    total = Fraction(0)
    for beta, c in coeffs.items():
        if sum(beta) == n:
            term = Fraction(c)
            for wi, bi in zip(direction, beta, strict=True):
                term *= Fraction(wi) ** bi
            total += term
    return total * factorial(n)


def _true_partial(coeffs, alpha):
    c = coeffs.get(tuple(alpha), 0)
    out = Fraction(c)
    for a in alpha:
        out *= factorial(a)
    return out


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_plans_reproduce_every_partial_of_a_generic_polynomial(dim):
    alphas = [a for a in product(range(MAX_POLARIZED_ORDER + 1), repeat=dim) if 1 <= sum(a) <= MAX_POLARIZED_ORDER]
    coeffs = {a: Fraction(7 + 3 * i, 5) for i, a in enumerate(product(range(MAX_POLARIZED_ORDER + 1), repeat=dim))}
    plan = polarization_plan(alphas, dim)
    dirs = [[_monomial_dir_derivative(coeffs, d, n) for n in range(1, plan.order + 1)] for d in plan.directions]
    for r, alpha in enumerate(alphas):
        got = sum(
            plan.coefficients[r][j][p] * dirs[j][p] for j in range(len(plan.directions)) for p in range(plan.order)
        )
        assert got == _true_partial(coeffs, alpha), alpha


def test_shared_directions_are_carried_once_and_pure_derivatives_use_one():
    assert len(polarization_plan([(4, 0)], 2).directions) == 1
    assert len(polarization_plan([(2, 2)], 2).directions) == 4  # (1,1), (1,0), (1,-1), (0,1)
    shared = polarization_plan([(4, 0), (0, 4), (2, 2)], 2)
    assert len(shared.directions) == 4  # axes plus the two diagonals


def test_refusals():
    with pytest.raises(ValueError, match="at least one"):
        polarization_plan([])
    with pytest.raises(ValueError, match="total derivative order"):
        polarization_plan([(5,)], 1)
    with pytest.raises(ValueError, match="total derivative order"):
        polarization_plan([(0, 0)], 2)
    with pytest.raises(ValueError, match="nonnegative"):
        polarization_plan([(1, -1)], 2)
    with pytest.raises(ValueError, match="length"):
        polarization_plan([(1, 1), (1,)], 2)


# ---------------------------------------------------------------------------------------------------- sparse plans


def _entries(plan):
    return sorted((r, plan.directions[j], p, c) for r, j, p, c in plan.nonzeros())


def _dense_bilaplacian(dim, coordinates=None):
    idx = list(range(dim)) if coordinates is None else list(coordinates)
    alphas = [tuple(4 * (k == i) for k in range(dim)) for i in idx]
    for p, i in enumerate(idx):
        for j in idx[p + 1 :]:
            alphas.append(tuple(2 * (k == i) + 2 * (k == j) for k in range(dim)))
    return alphas


@pytest.mark.parametrize("dim", [3, 7])
def test_structured_plans_equal_the_dense_request_lists(dim):
    from omnibias.core.polarization import bilaplacian_plan, laplacian_gradient_plan

    for structured, alphas in (
        (bilaplacian_plan(dim), _dense_bilaplacian(dim)),
        (bilaplacian_plan(dim, [2, 0, 1]), _dense_bilaplacian(dim, [2, 0, 1])),
        (
            laplacian_gradient_plan(dim),
            [tuple(2 * (k == j) + (k == i) for k in range(dim)) for i in range(dim) for j in range(dim)],
        ),
    ):
        dense = polarization_plan(alphas, dim)
        assert structured.directions == dense.directions
        assert structured.orders == dense.orders
        assert _entries(structured) == _entries(dense)


def test_sparse_arrays_are_the_nonzeros_of_the_dense_table():
    import numpy as np

    plan = polarization_plan([(2, 2, 0), (1, 1, 1), (3, 0, 1), (0, 2, 0)], 3)
    table = plan.coefficients
    nonzero = {(r, j, p - 1): c for r, req in enumerate(table) for j, per in enumerate(req) for p, c in enumerate(per, 1) if c}
    assert {(r, j, p - 1): c for r, j, p, c in plan.nonzeros()} == nonzero
    assert plan.nnz == len(nonzero)
    assert np.array_equal(plan.dense_floats(), np.asarray(plan.float_coefficients()))
    assert np.array_equal(plan.direction_matrix(), np.asarray(plan.directions))
    assert not plan.rows.flags.writeable and not plan.values.flags.writeable


def test_plans_are_memoized_and_the_dense_constructor_still_works():
    from omnibias.core.polarization import PolarizationPlan, polarization_plan_sparse

    plan = polarization_plan([(2, 2), (1, 1)], 2)
    assert polarization_plan([(2, 2), (1, 1)], 2) is plan
    assert polarization_plan_sparse([((0, 2), (1, 2)), ((0, 1), (1, 1))], 2) is plan
    rebuilt = PolarizationPlan(plan.directions, plan.coefficients, plan.orders)
    assert _entries(rebuilt) == _entries(plan) and rebuilt.directions == plan.directions


def test_the_bilaplacian_plan_at_256_inputs_is_built_without_its_dense_table():
    from omnibias.core.polarization import bilaplacian_plan

    m = 256
    plan = bilaplacian_plan(m)
    assert plan.num_requests == m + m * (m - 1) // 2
    assert plan.num_directions == m * m  # e_i, e_i + e_j, e_i - e_j
    assert plan.nnz == m + 4 * (m * (m - 1) // 2)  # one entry per pure request, four per mixed one
    assert plan.dense_entries == 8_623_489_024 and not plan.dense_is_cheap
    with pytest.raises(ValueError, match="nonzeros"):
        _ = plan.coefficients
    with pytest.raises(ValueError, match="nonzeros"):
        plan.dense_floats()


def test_sparse_requests_refuse_what_the_dense_ones_refuse():
    from omnibias.core.polarization import polarization_plan_sparse

    with pytest.raises(ValueError, match="at least one"):
        polarization_plan_sparse([], 3)
    with pytest.raises(ValueError, match="total derivative order"):
        polarization_plan_sparse([((0, 5),)], 3)
    with pytest.raises(ValueError, match="nonnegative"):
        polarization_plan_sparse([((0, -1), (1, 2))], 3)
    with pytest.raises(ValueError, match="outside"):
        polarization_plan_sparse([((3, 1),)], 3)
