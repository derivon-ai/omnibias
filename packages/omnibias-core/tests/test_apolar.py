# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
from fractions import Fraction
from math import comb

import pytest
from omnibias.core.apolar import apolar_algebra


def test_laplacian_algebra_relations_and_readout():
    a = apolar_algebra({(2, 0): 1, (0, 2): 1})
    x, y = a.generators
    assert a.size == 4
    assert a.multiply(x, x) == a.multiply(y, y)
    assert not any(a.multiply(x, y))
    assert not any(a.multiply(a.multiply(x, x), x))
    assert a.apply(a.multiply(x, x)) == 2
    assert a.multiply(a.unit, x) == x


def test_mixed_derivative_normalization_and_associativity():
    a = apolar_algebra({(2, 1): 1})
    assert a.size == 6
    x, y = a.generators
    assert a.apply(a.multiply(a.multiply(x, x), y)) == 2
    u = tuple(Fraction(i + 1, 3) for i in range(a.size))
    v = tuple(Fraction((-1)**i, i + 1) for i in range(a.size))
    assert a.multiply(a.multiply(u, v), x) == a.multiply(u, a.multiply(v, x))


@pytest.mark.parametrize("d,k", [(1, 3), (2, 2), (3, 2), (3, 3)])
def test_laplacian_power_dimension(d, k):
    from itertools import product
    from math import factorial
    terms = {}
    for alpha in product(range(k + 1), repeat=d):
        if sum(alpha) == k:
            coeff = factorial(k)
            for power in alpha:
                coeff //= factorial(power)
            terms[tuple(2 * v for v in alpha)] = coeff
    a = apolar_algebra(terms)
    assert a.size == 2 * comb(d + k - 1, k - 1) + comb(d + k - 1, k)


def test_biharmonic_polynomial_readout():
    a = apolar_algebra({(4, 0): 1, (2, 2): 2, (0, 4): 1})
    x, y = a.generators
    z = tuple(u + 2*v for u, v in zip(x, y, strict=True))
    z2 = a.multiply(z, z)
    assert a.apply(a.multiply(z2, z2)) == 600  # Δ²(x+2y)^4 = 4!*(1²+2²)².


def test_refusals_and_bounds():
    with pytest.raises(ValueError, match="homogeneous"):
        apolar_algebra({(1,): 1, (2,): 1})
    with pytest.raises(TypeError, match="Fraction"):
        apolar_algebra({(2,): 1.0})
    with pytest.raises(ValueError, match="monomial budget"):
        apolar_algebra({(4, 4): 1}, max_monomials=10)
    with pytest.raises(ValueError, match="basis budget"):
        apolar_algebra({(2, 2): 1}, max_basis=3)
    with pytest.raises(ValueError, match="work budget"):
        apolar_algebra({(2,): 1}, max_products=2)
    with pytest.raises(ValueError, match="bit budget"):
        apolar_algebra({(2,): 2**100}, max_coefficient_bits=64)
