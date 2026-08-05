# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Polynomial algebra and the monomial / Gram machinery."""

from __future__ import annotations

import pytest
from omnibias.sos.monomials import (
    MonomialBasis,
    SOSProblem,
    gram_products,
    gram_to_poly,
    monomial_basis,
)
from omnibias.sos.problem import Polynomial


def test_polynomial_algebra_and_eval() -> None:
    x = Polynomial.variable(0, 2)
    y = Polynomial.variable(1, 2)
    p = (x + y) * (x + y)
    assert p.coefficient((2, 0)) == 1.0
    assert p.coefficient((1, 1)) == 2.0
    assert p.coefficient((0, 2)) == 1.0
    assert p.evaluate([1.0, 2.0]) == pytest.approx(9.0)
    assert (x - y + 3.0).evaluate([2.0, 1.0]) == pytest.approx(4.0)
    assert p.degree() == 2


def test_zero_and_cancellation() -> None:
    x = Polynomial.variable(0, 1)
    assert (x - x).coeffs == {}
    assert (x - x).degree() == -1
    assert Polynomial.zero(3).evaluate([1.0, 2.0, 3.0]) == 0.0


def test_partial_and_gradient() -> None:
    x = Polynomial.variable(0, 2)
    y = Polynomial.variable(1, 2)
    p = x * x * y + y  # x^2 y + y
    dpx = p.partial(0)  # 2 x y
    dpy = p.partial(1)  # x^2 + 1
    assert dpx.coefficient((1, 1)) == 2.0
    assert dpy.coefficient((2, 0)) == 1.0
    assert dpy.coefficient((0, 0)) == 1.0
    grad = p.gradient()
    assert grad[0].coeffs == dpx.coeffs
    assert grad[1].coeffs == dpy.coeffs


def test_arity_mismatch_raises() -> None:
    x2 = Polynomial.variable(0, 2)
    x3 = Polynomial.variable(0, 3)
    with pytest.raises(ValueError, match="arity"):
        _ = x2 + x3
    with pytest.raises(ValueError, match="length"):
        _ = Polynomial(2, {(1,): 1.0})


def test_monomial_basis_shape_and_order() -> None:
    basis = monomial_basis(2, 2)
    # 1, x, y, x^2, xy, y^2  (graded, constant first)
    assert basis[0] == (0, 0)
    assert len(basis) == 6
    assert all(sum(basis[i]) <= sum(basis[i + 1]) for i in range(len(basis) - 1))
    assert MonomialBasis.up_to_degree(3, 2).size == 10  # C(2+3,3)


def test_gram_roundtrip() -> None:
    basis = monomial_basis(2, 1)  # (0,0),(1,0),(0,1)
    gram = [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    poly = gram_to_poly(gram, basis, 2)  # (x + y)^2
    assert poly.coefficient((2, 0)) == 1.0
    assert poly.coefficient((1, 1)) == 2.0
    assert poly.coefficient((0, 2)) == 1.0


def test_sosproblem_default_half_degree_and_representability() -> None:
    x = Polynomial.variable(0, 2)
    y = Polynomial.variable(1, 2)
    quartic = x * x * x * x + y * y * y * y + 1.0
    problem = SOSProblem.for_polynomial(quartic)
    assert problem.basis.size == monomial_basis(2, 2).__len__()
    assert problem.representable()


def test_products_are_disjoint_per_entry() -> None:
    # Each Gram entry (i, j) feeds exactly one product monomial -- the property the
    # exact rational projection relies on.
    basis = monomial_basis(3, 2)
    seen: set[tuple[int, int]] = set()
    for pairs in gram_products(basis).values():
        for i, j, _mult in pairs:
            assert (i, j) not in seen
            seen.add((i, j))
