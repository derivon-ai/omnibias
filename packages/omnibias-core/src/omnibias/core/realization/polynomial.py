# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact, budgeted coefficient maps of finite polynomial neural networks.

These are polynomials in the network parameters, not the Riccati polynomials
in an activation value.  No floating-point reconstruction is implicit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

Rational: TypeAlias = int | Fraction
MultiIndex: TypeAlias = tuple[int, ...]


def rational(value: Rational) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("exact coefficients require int or Fraction, not floats")
    return Fraction(value)


@dataclass(frozen=True)
class AlgebraBudget:
    max_terms: int = 100_000
    max_degree: int = 64
    max_bits: int = 8192
    max_products: int = 2_000_000

    def __post_init__(self) -> None:
        values = (self.max_terms, self.max_degree, self.max_bits, self.max_products)
        if any(type(v) is not int or v < 1 for v in values):
            raise ValueError("algebra budgets must be positive")


DEFAULT_ALGEBRA_BUDGET = AlgebraBudget()


class AlgebraBudgetExceeded(RuntimeError):
    """An exact computation stopped; this is not mathematical exclusion."""


@dataclass(frozen=True, init=False)
class SparsePolynomial:
    nvars: int
    terms: tuple[tuple[MultiIndex, Fraction], ...]
    budget: AlgebraBudget

    def __init__(
        self,
        nvars: int,
        terms: Mapping[MultiIndex, Rational] | None = None,
        *,
        budget: AlgebraBudget = DEFAULT_ALGEBRA_BUDGET,
    ) -> None:
        if type(nvars) is not int or nvars < 0:
            raise ValueError("nvars must be nonnegative")
        clean: list[tuple[MultiIndex, Fraction]] = []
        for idx, coeff in (terms or {}).items():
            if len(idx) != nvars or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in idx
            ):
                raise ValueError("invalid polynomial multi-index")
            c = rational(coeff)
            if not c:
                continue
            if sum(idx) > budget.max_degree:
                raise AlgebraBudgetExceeded("polynomial degree budget exceeded")
            if max(abs(c.numerator).bit_length(), c.denominator.bit_length()) > budget.max_bits:
                raise AlgebraBudgetExceeded("rational coefficient bit budget exceeded")
            clean.append((tuple(idx), c))
        if len(clean) > budget.max_terms:
            raise AlgebraBudgetExceeded("polynomial term budget exceeded")
        object.__setattr__(self, "nvars", nvars)
        object.__setattr__(
            self, "terms", tuple(sorted(clean, key=lambda item: (sum(item[0]), item[0])))
        )
        object.__setattr__(self, "budget", budget)

    @classmethod
    def constant(
        cls, nvars: int, value: Rational, *, budget: AlgebraBudget = DEFAULT_ALGEBRA_BUDGET
    ) -> SparsePolynomial:
        return cls(nvars, {(0,) * nvars: value}, budget=budget)

    @classmethod
    def variable(
        cls, nvars: int, axis: int, *, budget: AlgebraBudget = DEFAULT_ALGEBRA_BUDGET
    ) -> SparsePolynomial:
        if not 0 <= axis < nvars:
            raise ValueError("variable axis out of range")
        index = tuple(int(i == axis) for i in range(nvars))
        return cls(nvars, {index: 1}, budget=budget)

    def _coerce(self, value: SparsePolynomial | Rational) -> SparsePolynomial:
        p = (
            value
            if isinstance(value, SparsePolynomial)
            else self.constant(self.nvars, value, budget=self.budget)
        )
        if p.nvars != self.nvars:
            raise ValueError("polynomial variable counts differ")
        return p

    def __add__(self, other: SparsePolynomial | Rational) -> SparsePolynomial:
        p = self._coerce(other)
        terms = dict(self.terms)
        for i, c in p.terms:
            terms[i] = terms.get(i, Fraction(0)) + c
        return SparsePolynomial(self.nvars, terms, budget=self.budget)

    def __radd__(self, other: Rational) -> SparsePolynomial:
        return self + other

    def __neg__(self) -> SparsePolynomial:
        return SparsePolynomial(self.nvars, {i: -c for i, c in self.terms}, budget=self.budget)

    def __sub__(self, other: SparsePolynomial | Rational) -> SparsePolynomial:
        return self + -self._coerce(other)

    def __mul__(self, other: SparsePolynomial | Rational) -> SparsePolynomial:
        p = self._coerce(other)
        if len(self.terms) * len(p.terms) > self.budget.max_products:
            raise AlgebraBudgetExceeded("polynomial product budget exceeded")
        terms: dict[MultiIndex, Fraction] = {}
        for i, a in self.terms:
            for j, b in p.terms:
                k = tuple(x + y for x, y in zip(i, j, strict=True))
                terms[k] = terms.get(k, Fraction(0)) + a * b
                if len(terms) > self.budget.max_terms:
                    raise AlgebraBudgetExceeded("intermediate polynomial term budget exceeded")
        return SparsePolynomial(self.nvars, terms, budget=self.budget)

    def __rmul__(self, other: Rational) -> SparsePolynomial:
        return self * other

    def __pow__(self, exponent: int) -> SparsePolynomial:
        if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent < 0:
            raise ValueError("polynomial powers must be nonnegative integers")
        result = self.constant(self.nvars, 1, budget=self.budget)
        base = self
        while exponent:
            if exponent & 1:
                result = result * base
            exponent //= 2
            if exponent:
                base = base * base
        return result

    def derivative(self, axis: int) -> SparsePolynomial:
        if not 0 <= axis < self.nvars:
            raise ValueError("derivative axis out of range")
        terms = {}
        for idx, c in self.terms:
            if idx[axis]:
                j = list(idx)
                j[axis] -= 1
                terms[tuple(j)] = c * idx[axis]
        return SparsePolynomial(self.nvars, terms, budget=self.budget)

    def evaluate(self, point: Sequence[Rational]) -> Fraction:
        if len(point) != self.nvars:
            raise ValueError("evaluation point has wrong dimension")
        values = tuple(rational(x) for x in point)
        out = Fraction(0)
        for idx, c in self.terms:
            for x, power in zip(values, idx, strict=True):
                c *= x**power
            out += c
        return out

    def to_payload(self) -> dict[str, object]:
        return {
            "nvars": self.nvars,
            "terms": [[list(i), [c.numerator, c.denominator]] for i, c in self.terms],
        }


@dataclass(frozen=True)
class PolynomialNetworkSpec:
    """Row-major weights, followed by biases, for each layer in order.

    ``activations[l][k]`` is the coefficient of ``z**k`` in hidden layer l.
    The last layer is affine. For homogeneous networks set ``biases=False``.
    """

    dims: tuple[int, ...]
    activations: tuple[tuple[Rational, ...], ...]
    biases: bool = True

    def __post_init__(self) -> None:
        if type(self.biases) is not bool:
            raise TypeError("biases must be a boolean")
        if len(self.dims) < 2 or any(
            isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in self.dims
        ):
            raise ValueError("network dimensions must be positive integers")
        if len(self.activations) != len(self.dims) - 2 or any(not a for a in self.activations):
            raise ValueError("one nonempty coefficient tuple per hidden layer is required")
        for a in self.activations:
            for c in a:
                rational(c)

    @property
    def n_params(self) -> int:
        return sum(
            m * n + (m if self.biases else 0)
            for n, m in zip(self.dims[:-1], self.dims[1:], strict=True)
        )

    @classmethod
    def monomial(
        cls, dims: Sequence[int], degree: int = 2, *, biases: bool = False
    ) -> PolynomialNetworkSpec:
        if type(degree) is not int or degree < 1:
            raise ValueError("activation degree must be positive")
        return cls(tuple(dims), ((0,) * degree + (1,),) * (len(dims) - 2), biases)

    def to_payload(self) -> dict[str, object]:
        return {
            "dims": list(self.dims),
            "biases": self.biases,
            "activations": [
                [[rational(c).numerator, rational(c).denominator] for c in a]
                for a in self.activations
            ],
        }


@dataclass(frozen=True)
class PolynomialRealizationMap:
    spec: PolynomialNetworkSpec
    input_indices: tuple[MultiIndex, ...]
    outputs: tuple[tuple[SparsePolynomial, ...], ...]

    @property
    def polynomials(self) -> tuple[SparsePolynomial, ...]:
        return tuple(p for row in self.outputs for p in row)

    @property
    def fingerprint(self) -> str:
        body = {
            "spec": self.spec.to_payload(),
            "indices": self.input_indices,
            "map": [p.to_payload() for p in self.polynomials],
        }
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def evaluate(self, parameters: Sequence[Rational]) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(tuple(p.evaluate(parameters) for p in row) for row in self.outputs)

    def jacobian(self, parameters: Sequence[Rational]) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(p.derivative(i).evaluate(parameters) for i in range(self.spec.n_params))
            for p in self.polynomials
        )


def compile_coefficient_map(
    spec: PolynomialNetworkSpec, *, budget: AlgebraBudget = DEFAULT_ALGEBRA_BUDGET
) -> PolynomialRealizationMap:
    """Compile a polynomial map in all parameters, collecting all input terms.

    The returned index ordering is explicit and includes only input monomials
    occurring symbolically. Reaching a budget raises ``AlgebraBudgetExceeded``.
    """
    p = spec.n_params
    nvars = p + spec.dims[0]
    variables = [SparsePolynomial.variable(nvars, i, budget=budget) for i in range(p)]
    values = [SparsePolynomial.variable(nvars, p + i, budget=budget) for i in range(spec.dims[0])]
    offset = 0
    for layer, (n_in, n_out) in enumerate(zip(spec.dims[:-1], spec.dims[1:], strict=True)):
        weights = [variables[offset + i * n_in : offset + (i + 1) * n_in] for i in range(n_out)]
        offset += n_in * n_out
        bias = (
            variables[offset : offset + n_out]
            if spec.biases
            else [SparsePolynomial.constant(nvars, 0, budget=budget)] * n_out
        )
        offset += n_out if spec.biases else 0
        result = []
        for row, b in zip(weights, bias, strict=True):
            z = b
            for w, x in zip(row, values, strict=True):
                z = z + w * x
            if layer < len(spec.activations):
                a = SparsePolynomial.constant(nvars, 0, budget=budget)
                for coeff in reversed(spec.activations[layer]):
                    a = a * z + coeff
                z = a
            result.append(z)
        values = result
    indices = tuple(
        sorted({idx[p:] for value in values for idx, _ in value.terms}, key=lambda i: (sum(i), i))
    )
    # A zero activation can yield a zero map with no terms. Keep the constant
    # slot so even this map has an explicit, nonempty coefficient target.
    if not indices:
        indices = ((0,) * spec.dims[0],)
    rows = []
    for value in values:
        collected: dict[MultiIndex, dict[MultiIndex, Fraction]] = {i: {} for i in indices}
        for idx, c in value.terms:
            collected[idx[p:]][idx[:p]] = c
        rows.append(tuple(SparsePolynomial(p, collected[i], budget=budget) for i in indices))
    return PolynomialRealizationMap(spec, indices, tuple(rows))


__all__ = [
    "AlgebraBudget",
    "AlgebraBudgetExceeded",
    "MultiIndex",
    "PolynomialNetworkSpec",
    "PolynomialRealizationMap",
    "Rational",
    "SparsePolynomial",
    "compile_coefficient_map",
    "rational",
]
