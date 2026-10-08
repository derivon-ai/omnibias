# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded exact apolar algebras for experimental derivative programs.

For a nonzero homogeneous operator symbol P, the quotient by its differential
annihilator is the smallest scalar algebra realizing p -> P(d)p(0) through a
linear readout. This statement concerns state dimension, not runtime optimality.
Construction uses rational column elimination and never imports a tensor backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import factorial, prod

Index = tuple[int, ...]
Sparse = dict[Index, Fraction]


@dataclass(frozen=True)
class ApolarAlgebra:
    """Monomial basis, exact multiplication, generators, and operator readout.

``products`` contains (left, right, output, weight), including both orders for
off-diagonal pairs. ``readout`` returns the raw differential operator value;
smooth functions must be lifted with Taylor-normalized coefficients.
    """

    dim: int
    order: int
    indices: tuple[Index, ...]
    products: tuple[tuple[int, int, int, Fraction], ...]
    generators: tuple[tuple[Fraction, ...], ...]
    readout: tuple[Fraction, ...]

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def unit(self) -> tuple[Fraction, ...]:
        return (Fraction(1),) + (Fraction(0),) * (self.size - 1)

    def multiply(self, a: Sequence[Fraction | int], b: Sequence[Fraction | int]) -> tuple[Fraction, ...]:
        """Exact reference multiplication, for construction checks and tests."""
        if len(a) != self.size or len(b) != self.size:
            raise ValueError(f"expected {self.size} coordinates per operand")
        out = [Fraction(0)] * self.size
        for i, j, k, c in self.products:
            out[k] += c * a[i] * b[j]
        return tuple(out)

    def apply(self, value: Sequence[Fraction | int]) -> Fraction:
        if len(value) != self.size:
            raise ValueError(f"expected {self.size} coordinates")
        return sum((c * v for c, v in zip(self.readout, value, strict=True)), Fraction())

    def explain(self) -> dict[str, object]:
        return {"algebra": "apolar", "variables": self.dim, "order": self.order,
                "coefficients": self.size, "multiplication_terms": len(self.products),
                "arithmetic": "exact rational", "minimality": "scalar algebra representations"}


def _subtract(v: Sparse, q: Sparse, scale: Fraction) -> None:
    for key, value in q.items():
        new = v.get(key, Fraction()) - scale * value
        if new:
            v[key] = new
        else:
            v.pop(key, None)


def apolar_algebra(
    symbol: Mapping[Index, int | Fraction], *, max_monomials: int = 512,
    max_basis: int = 128, max_products: int = 1_000_000,
    max_order: int = 32, max_coefficient_bits: int = 4096,
) -> ApolarAlgebra:
    """Construct a small homogeneous constant-coefficient operator algebra.

Keys are exponent tuples; values must be integers or Fractions. All nonzero
terms must have the same degree. Bounds cover candidate monomials, elimination
state, coefficient growth, and stored multiplication terms; rejection occurs
before allocating a dense tensor. This is a host-side experimental planner.
    """
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in
           (max_monomials, max_basis, max_products, max_order, max_coefficient_bits)):
        raise ValueError("apolar construction budgets must be positive integers")
    if not symbol:
        raise ValueError("operator symbol must be nonzero")
    dimension = len(next(iter(symbol)))
    if dimension < 1:
        raise ValueError("operator needs at least one variable")
    terms: Sparse = {}
    for alpha, value in symbol.items():
        if len(alpha) != dimension or any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in alpha):
            raise ValueError("operator indices must be equally sized nonnegative integer tuples")
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise TypeError("operator coefficients must be int or Fraction")
        if value:
            terms[tuple(alpha)] = Fraction(value)
    if not terms:
        raise ValueError("operator symbol must be nonzero")
    orders = {sum(a) for a in terms}
    if len(orders) != 1:
        raise ValueError("only homogeneous operator symbols are supported")
    order = orders.pop()
    if order > max_order:
        raise ValueError(f"operator order {order} exceeds construction budget {max_order}")

    def check_fraction(value: Fraction) -> None:
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > max_coefficient_bits:
            raise ValueError("apolar rational coefficient exceeds bit budget")

    for value in terms.values():
        check_fraction(value)
    candidates: set[Index] = set()
    for alpha in terms:
        if prod(a + 1 for a in alpha) > max_monomials:
            raise ValueError("operator divisor closure exceeds monomial budget; request fewer directions")
        for beta in product(*(range(a + 1) for a in alpha)):
            candidates.add(beta)
            if len(candidates) > max_monomials:
                raise ValueError("operator divisor closure exceeds monomial budget; request fewer terms")
    ordered = sorted(candidates, key=lambda a: (sum(a), a))

    def image(alpha: Index) -> Sparse:
        out: Sparse = {}
        for gamma, coefficient in terms.items():
            if all(a <= g for a, g in zip(alpha, gamma, strict=True)):
                beta = tuple(g - a for a, g in zip(alpha, gamma, strict=True))
                out[beta] = coefficient * prod(factorial(g) // factorial(b) for g, b in zip(gamma, beta, strict=True))
        return out

    pivots: list[Index] = []
    reduced: list[Sparse] = []
    transforms: list[list[Fraction]] = []
    basis: list[Index] = []
    for alpha in ordered:
        v = image(alpha)
        weights: list[Fraction] = []
        for pivot, column in zip(pivots, reduced, strict=True):
            scale = v.get(pivot, Fraction())
            weights.append(scale)
            if scale:
                _subtract(v, column, scale)
        if not v:
            continue
        if len(basis) >= max_basis:
            raise ValueError(f"apolar state exceeds basis budget {max_basis}; request a smaller operator")
        pivot = min(v, key=lambda a: (sum(a), a))
        scale = v[pivot]
        column = {key: val / scale for key, val in v.items()}
        transform = [Fraction(0)] * len(basis) + [1 / scale]
        for weight, previous in zip(weights, transforms, strict=True):
            for i, val in enumerate(previous):
                transform[i] -= weight * val / scale
        for val in (*column.values(), *transform):
            check_fraction(val)
        pivots.append(pivot)
        reduced.append(column)
        transforms.append(transform)
        basis.append(alpha)

    def coordinates(alpha: Index) -> tuple[Fraction, ...]:
        v = image(alpha)
        out = [Fraction(0)] * len(basis)
        for pivot, column, transform in zip(pivots, reduced, transforms, strict=True):
            scale = v.get(pivot, Fraction())
            if scale:
                _subtract(v, column, scale)
                for i, val in enumerate(transform):
                    out[i] += scale * val
        if v:
            raise ArithmeticError("apolar basis failed to span a product")
        for val in out:
            check_fraction(val)
        return tuple(out)

    # Also bound pair enumeration when most resulting products happen to vanish.
    if len(basis) ** 2 > max_products:
        raise ValueError("apolar multiplication pair count exceeds work budget")
    multiplication: list[tuple[int, int, int, Fraction]] = []
    for i, alpha in enumerate(basis):
        for j in range(i, len(basis)):
            beta = basis[j]
            if sum(alpha) + sum(beta) > order:
                continue
            result = coordinates(tuple(a + b for a, b in zip(alpha, beta, strict=True)))
            for k, val in enumerate(result):
                if val:
                    multiplication.append((i, j, k, val))
                    if i != j:
                        multiplication.append((j, i, k, val))
                    if len(multiplication) > max_products:
                        raise ValueError("apolar multiplication terms exceed work budget")
    generators = tuple(coordinates(tuple(int(i == j) for j in range(dimension))) for i in range(dimension))
    readout = tuple(terms.get(alpha, Fraction()) * prod(factorial(a) for a in alpha) for alpha in basis)
    return ApolarAlgebra(dimension, order, tuple(basis), tuple(multiplication), generators, readout)


__all__ = ["ApolarAlgebra", "apolar_algebra"]
