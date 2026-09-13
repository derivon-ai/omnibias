# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact real realization and Laurent closure witnesses tied to a source map."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.realization.algebraic import AlgebraicNumber, RealAlgebraicField
from omnibias.core.realization.polynomial import (
    PolynomialRealizationMap,
    Rational,
    SparsePolynomial,
    rational,
)


@dataclass(frozen=True)
class RationalRealizationWitness:
    source_fingerprint: str
    parameters: tuple[Fraction, ...]
    target: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(map(rational, self.parameters)))
        object.__setattr__(self, "target", tuple(map(rational, self.target)))

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": "rational_realization",
            "source_fingerprint": self.source_fingerprint,
            "parameters": [_q(x) for x in self.parameters],
            "target": [_q(x) for x in self.target],
        }

    def verify(self, source: PolynomialRealizationMap) -> bool:
        return (
            self.source_fingerprint == source.fingerprint
            and len(self.target) == len(source.polynomials)
            and len(self.parameters) == source.spec.n_params
            and tuple(p.evaluate(self.parameters) for p in source.polynomials) == self.target
        )


def rational_realization_witness(
    source: PolynomialRealizationMap, parameters: Sequence[Rational], target: Sequence[Rational]
) -> RationalRealizationWitness:
    out = RationalRealizationWitness(
        source.fingerprint, tuple(map(rational, parameters)), tuple(map(rational, target))
    )
    if not out.verify(source):
        raise ValueError("parameters do not realize this source map's target")
    return out


def algebraic_evaluate(
    polynomial: SparsePolynomial, parameters: Sequence[AlgebraicNumber], field: RealAlgebraicField
) -> AlgebraicNumber:
    if len(parameters) != polynomial.nvars or any(p.field != field for p in parameters):
        raise ValueError("incompatible algebraic parameters")
    result = field.constant(0)
    for idx, coeff in polynomial.terms:
        term = field.constant(coeff)
        for p, exponent in zip(parameters, idx, strict=True):
            term = term * p**exponent
        result = result + term
    return result


@dataclass(frozen=True)
class AlgebraicRealizationWitness:
    source_fingerprint: str
    field: RealAlgebraicField
    parameters: tuple[AlgebraicNumber, ...]
    target: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", tuple(map(rational, self.target)))

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": "algebraic_realization",
            "source_fingerprint": self.source_fingerprint,
            "primitive": [_q(rational(x)) for x in self.field.polynomial],
            "isolating_interval": [_q(rational(self.field.lo)), _q(rational(self.field.hi))],
            "parameters": [[_q(x) for x in p.coefficients] for p in self.parameters],
            "target": [_q(x) for x in self.target],
        }

    def verify(self, source: PolynomialRealizationMap) -> bool:
        if (
            self.source_fingerprint != source.fingerprint
            or len(self.target) != len(source.polynomials)
            or len(self.parameters) != source.spec.n_params
            or any(p.field != self.field for p in self.parameters)
        ):
            return False
        return all(
            (algebraic_evaluate(p, self.parameters, self.field) - target).is_zero()
            for p, target in zip(source.polynomials, self.target, strict=True)
        )


@dataclass(frozen=True, init=False)
class LaurentPolynomial:
    terms: tuple[tuple[int, Fraction], ...]

    def __init__(self, terms: Mapping[int, Rational]) -> None:
        if any(isinstance(k, bool) or not isinstance(k, int) for k in terms):
            raise ValueError("Laurent powers must be integers")
        object.__setattr__(
            self, "terms", tuple(sorted((k, rational(v)) for k, v in terms.items() if rational(v)))
        )

    def __add__(self, other: LaurentPolynomial) -> LaurentPolynomial:
        result = dict(self.terms)
        for k, c in other.terms:
            result[k] = result.get(k, Fraction(0)) + c
        return LaurentPolynomial(result)

    def __mul__(self, other: LaurentPolynomial) -> LaurentPolynomial:
        result: dict[int, Fraction] = {}
        for i, c in self.terms:
            for j, d in other.terms:
                result[i + j] = result.get(i + j, Fraction(0)) + c * d
        return LaurentPolynomial(result)

    def __pow__(self, exponent: int) -> LaurentPolynomial:
        if type(exponent) is not int or exponent < 0:
            raise ValueError("only nonnegative powers of a Laurent polynomial are supported")
        result, base = LaurentPolynomial({0: 1}), self
        while exponent:
            if exponent & 1:
                result = result * base
            exponent //= 2
            if exponent:
                base = base * base
        return result

    def evaluate(self, epsilon: Rational) -> Fraction:
        e = rational(epsilon)
        if not e and any(k < 0 for k, _ in self.terms):
            raise ZeroDivisionError("Laurent path is defined away from zero")
        return sum((v * e**k for k, v in self.terms), Fraction(0))


@dataclass(frozen=True)
class LaurentClosureWitness:
    source_fingerprint: str
    parameters: tuple[LaurentPolynomial, ...]
    target: tuple[Fraction, ...]
    radius: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", tuple(map(rational, self.target)))
        object.__setattr__(self, "radius", rational(self.radius))

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": "laurent_closure",
            "source_fingerprint": self.source_fingerprint,
            "parameters": [[[k, _q(c)] for k, c in p.terms] for p in self.parameters],
            "target": [_q(x) for x in self.target],
            "radius": _q(self.radius),
        }

    def realized_path(self, source: PolynomialRealizationMap) -> tuple[LaurentPolynomial, ...]:
        if (
            source.fingerprint != self.source_fingerprint
            or len(self.parameters) != source.spec.n_params
        ):
            raise ValueError("closure path is bound to a different source")
        outputs = []
        for p in source.polynomials:
            value = LaurentPolynomial({})
            for idx, c in p.terms:
                term = LaurentPolynomial({0: c})
                for parameter, exponent in zip(self.parameters, idx, strict=True):
                    term = term * parameter**exponent
                value = value + term
            outputs.append(value)
        return tuple(outputs)

    def verify(self, source: PolynomialRealizationMap) -> bool:
        if self.radius <= 0 or len(self.target) != len(source.polynomials):
            return False
        try:
            path = self.realized_path(source)
        except ValueError:
            return False
        return all(
            not any(k < 0 for k, _ in p.terms) and dict(p.terms).get(0, Fraction(0)) == t
            for p, t in zip(path, self.target, strict=True)
        )

    def coefficient_error_bounds(
        self, source: PolynomialRealizationMap, radius: Rational | None = None
    ) -> tuple[Fraction, ...]:
        rho = self.radius if radius is None else rational(radius)
        if not self.verify(source) or not 0 < rho <= self.radius:
            raise ValueError("verified path and radius within the certified interval required")
        return tuple(
            sum((abs(c) * rho**k for k, c in p.terms if k > 0), Fraction(0))
            for p in self.realized_path(source)
        )


def _q(q: Fraction) -> list[int]:
    return [q.numerator, q.denominator]


def _decode_q(value: Any) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(x) is not int for x in value)
        or value[1] <= 0
    ):
        raise ValueError("malformed exact rational witness value")
    return Fraction(value[0], value[1])


def witness_from_payload(
    payload: Mapping[str, Any], source: PolynomialRealizationMap
) -> RationalRealizationWitness | AlgebraicRealizationWitness | LaurentClosureWitness:
    """Decode a portable exact witness and replay it against its original map."""
    if payload.get("source_fingerprint") != source.fingerprint:
        raise ValueError("witness source fingerprint differs")
    try:
        target = tuple(_decode_q(x) for x in payload["target"])
        result: RationalRealizationWitness | AlgebraicRealizationWitness | LaurentClosureWitness
        if payload["kind"] == "rational_realization":
            result = RationalRealizationWitness(
                source.fingerprint, tuple(_decode_q(x) for x in payload["parameters"]), target
            )
        elif payload["kind"] == "algebraic_realization":
            endpoints = payload["isolating_interval"]
            if len(endpoints) != 2:
                raise ValueError("primitive root needs two endpoints")
            field = RealAlgebraicField(
                tuple(_decode_q(x) for x in payload["primitive"]),
                _decode_q(endpoints[0]),
                _decode_q(endpoints[1]),
            )
            parameters = tuple(
                field.number(tuple(_decode_q(x) for x in row)) for row in payload["parameters"]
            )
            result = AlgebraicRealizationWitness(source.fingerprint, field, parameters, target)
        elif payload["kind"] == "laurent_closure":
            paths = []
            for row in payload["parameters"]:
                terms = {}
                for term in row:
                    if len(term) != 2 or type(term[0]) is not int or term[0] in terms:
                        raise ValueError("malformed Laurent coefficient")
                    terms[term[0]] = _decode_q(term[1])
                paths.append(LaurentPolynomial(terms))
            result = LaurentClosureWitness(
                source.fingerprint, tuple(paths), target, _decode_q(payload["radius"])
            )
        else:
            raise ValueError("unsupported exact witness kind")
        if not result.verify(source):
            raise ValueError("decoded witness failed exact source replay")
        return result
    except (TypeError, KeyError, IndexError) as exc:
        raise ValueError("malformed exact realization witness") from exc


__all__ = [
    "AlgebraicRealizationWitness",
    "LaurentClosureWitness",
    "LaurentPolynomial",
    "RationalRealizationWitness",
    "algebraic_evaluate",
    "rational_realization_witness",
    "witness_from_payload",
]
