# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Scoped exact factorisation: rational roots, Yun square-free, linear divisors."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.holonomic._core.factor import (
    linear_factorization,
    monic_linear_divisors,
    rational_roots,
    roots_with_multiplicity,
    square_free,
)
from omnibias.holonomic._core.rational_poly import Poly, pmonic, pmul, to_poly


def _xminus(root: Fraction | int) -> Poly:
    """The monic linear polynomial ``x - root``."""
    return to_poly([-Fraction(root), Fraction(1)])


def _prod(*polys: Poly) -> Poly:
    out: Poly = (Fraction(1),)
    for p in polys:
        out = pmul(out, p)
    return out


def test_rational_roots_integer_and_fractional() -> None:
    # (x - 1)(x - 2)(x + 3)
    p = _prod(_xminus(1), _xminus(2), _xminus(-3))
    assert rational_roots(p) == [Fraction(-3), Fraction(1), Fraction(2)]
    # 2x - 1 has the fractional root 1/2.
    assert rational_roots(to_poly([-1, 2])) == [Fraction(1, 2)]


def test_rational_roots_include_zero() -> None:
    # x^2 (x - 3) has roots {0, 3}.
    p = _prod((Fraction(0), Fraction(0), Fraction(1)), _xminus(3))
    assert rational_roots(p) == [Fraction(0), Fraction(3)]


def test_no_rational_roots() -> None:
    assert rational_roots(to_poly([1, 0, 1])) == []  # x^2 + 1


def test_roots_with_multiplicity() -> None:
    # (x - 1)^2 (x - 2)
    p = _prod(_xminus(1), _xminus(1), _xminus(2))
    assert roots_with_multiplicity(p) == [(Fraction(1), 2), (Fraction(2), 1)]


def test_square_free_reconstructs() -> None:
    # f = (x - 1)^2 (x - 2)^3 (x + 4)
    f = _prod(_xminus(1), _xminus(1), _xminus(2), _xminus(2), _xminus(2), _xminus(-4))
    decomp = square_free(f)
    recon: Poly = (Fraction(1),)
    for g, i in decomp:
        assert pmonic(g) == g  # each factor is monic
        for _ in range(i):
            recon = pmul(recon, g)
    assert recon == pmonic(f)
    # multiplicities present: 1 (for x+4), 2 (for x-1), 3 (for x-2)
    assert sorted(i for _g, i in decomp) == [1, 2, 3]


def test_linear_factorization_leaves_irreducible_cofactor() -> None:
    # (x - 2) (x^2 + 1): one rational root, quadratic cofactor untouched.
    p = _prod(_xminus(2), to_poly([1, 0, 1]))
    roots, cofactor = linear_factorization(p)
    assert roots == [(Fraction(2), 1)]
    assert cofactor == to_poly([1, 0, 1])


def test_monic_linear_divisors_enumerates_products() -> None:
    # (x + 1)(x + 2): divisors are 1, (x+1), (x+2), (x+1)(x+2).
    p = _prod(_xminus(-1), _xminus(-2))
    divisors = monic_linear_divisors(p)
    assert (Fraction(1),) in divisors
    assert _xminus(-1) in divisors
    assert _xminus(-2) in divisors
    assert _prod(_xminus(-1), _xminus(-2)) in divisors
    assert len(divisors) == 4


def test_monic_linear_divisors_with_multiplicity() -> None:
    # (x - 1)^2: divisors 1, (x-1), (x-1)^2.
    p = _prod(_xminus(1), _xminus(1))
    divisors = monic_linear_divisors(p)
    assert len(divisors) == 3
    assert _prod(_xminus(1), _xminus(1)) in divisors


def test_irreducible_has_only_trivial_divisor() -> None:
    divisors = monic_linear_divisors(to_poly([1, 0, 1]))  # x^2 + 1
    assert divisors == [(Fraction(1),)]


def test_square_free_zero_raises() -> None:
    with pytest.raises(ValueError, match="zero polynomial"):
        square_free(())
