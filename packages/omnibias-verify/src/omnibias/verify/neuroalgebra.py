# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact rational, bounded polynomial realization search.

Exclusion covers the specified parameter box only. An exhausted budget returns
Inconclusive, even when no floating-point optimizer found a realization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, TypeAlias

from omnibias.core.proof.certificate import Cert
from omnibias.core.realization.polynomial import (
    PolynomialRealizationMap,
    Rational,
    SparsePolynomial,
    rational,
)
from omnibias.core.realization.witness import (
    RationalRealizationWitness,
    rational_realization_witness,
)

RationalBox: TypeAlias = tuple[tuple[Fraction, Fraction], ...]


def _box(box: Sequence[Sequence[Rational]]) -> RationalBox:
    if any(len(pair) != 2 for pair in box):
        raise ValueError("each parameter interval needs two endpoints")
    result = tuple((rational(pair[0]), rational(pair[1])) for pair in box)
    if any(lo > hi for lo, hi in result):
        raise ValueError("reversed parameter interval")
    return result


def _product(
    a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    products = [x * y for x in a for y in b]
    return min(products), max(products)


def polynomial_interval(
    poly: SparsePolynomial, box: Sequence[Sequence[Rational]]
) -> tuple[Fraction, Fraction]:
    """Natural enclosure with exact rational endpoints and even-power tightening."""
    bounds = _box(box)
    if len(bounds) != poly.nvars:
        raise ValueError("parameter box dimension differs")
    lower = upper = Fraction(0)
    for powers, coeff in poly.terms:
        term = (coeff, coeff)
        for (lo, hi), n in zip(bounds, powers, strict=True):
            if n % 2 == 0:
                power = (
                    Fraction(0) if lo <= 0 <= hi and n else min(lo**n, hi**n),
                    max(lo**n, hi**n),
                )
            else:
                power = (lo**n, hi**n)
            term = _product(term, power)
        lower += term[0]
        upper += term[1]
    return lower, upper


@dataclass(frozen=True)
class ExclusionLeaf:
    equation: int
    enclosure: tuple[Fraction, Fraction]


@dataclass(frozen=True)
class BoxSplit:
    axis: int
    midpoint: Fraction
    left: ExclusionLeaf | BoxSplit
    right: ExclusionLeaf | BoxSplit


@dataclass(frozen=True)
class PolynomialBoxCertificate:
    source_fingerprint: str
    target: tuple[Fraction, ...]
    box: RationalBox
    tree: ExclusionLeaf | BoxSplit

    def to_replay_certificate(self, source: PolynomialRealizationMap) -> Cert:
        """Export a source-bound, finite subdivision/interval Lean obligation."""
        from omnibias.core.proof.realization_algebra_replay import polynomial_box_replay_certificate

        if not self.verify(source):
            raise ValueError("invalid source-bound box exclusion")

        def q(x: Fraction) -> list[int]:
            return [x.numerator, x.denominator]

        def encode(tree: ExclusionLeaf | BoxSplit) -> dict[str, object]:
            if isinstance(tree, ExclusionLeaf):
                return {"equation": tree.equation, "enclosure": [q(x) for x in tree.enclosure]}
            return {
                "axis": tree.axis,
                "midpoint": q(tree.midpoint),
                "left": encode(tree.left),
                "right": encode(tree.right),
            }

        equations = tuple(p - t for p, t in zip(source.polynomials, self.target, strict=True))
        return polynomial_box_replay_certificate(equations, self.box, encode(self.tree))

    def verify(self, source: PolynomialRealizationMap) -> bool:
        """Replay exact source equations and binary coverage, including every leaf."""
        if source.fingerprint != self.source_fingerprint or len(self.target) != len(
            source.polynomials
        ):
            return False
        if len(self.box) != source.spec.n_params or any(lo > hi for lo, hi in self.box):
            return False
        equations = tuple(p - t for p, t in zip(source.polynomials, self.target, strict=True))

        def visit(tree: ExclusionLeaf | BoxSplit, box: RationalBox) -> bool:
            if isinstance(tree, ExclusionLeaf):
                if not 0 <= tree.equation < len(equations):
                    return False
                enclosure = polynomial_interval(equations[tree.equation], box)
                return enclosure == tree.enclosure and (enclosure[1] < 0 or enclosure[0] > 0)
            if not 0 <= tree.axis < len(box):
                return False
            lo, hi = box[tree.axis]
            if not lo < tree.midpoint < hi:
                return False
            left = box[: tree.axis] + ((lo, tree.midpoint),) + box[tree.axis + 1 :]
            right = box[: tree.axis] + ((tree.midpoint, hi),) + box[tree.axis + 1 :]
            return visit(tree.left, left) and visit(tree.right, right)

        return visit(self.tree, self.box)


@dataclass(frozen=True)
class BoundedRealizationReport:
    status: Literal["excluded_on_box", "realizable_in_box", "inconclusive"]
    source_fingerprint: str
    target: tuple[Fraction, ...]
    box: RationalBox
    visited_boxes: int
    unresolved_boxes: tuple[RationalBox, ...]
    certificate: PolynomialBoxCertificate | None
    witness: RationalRealizationWitness | None
    detail: str


def bounded_realization_search(
    source: PolynomialRealizationMap,
    target: Sequence[Rational],
    box: Sequence[Sequence[Rational]],
    *,
    max_nodes: int = 4095,
    max_depth: int = 32,
) -> BoundedRealizationReport:
    """Search arbitrary compiled polynomial networks on an explicit rational box.

    Exact midpoint roots produce replayable witnesses. Other roots are not
    guessed; algebraic isolated roots require the separate algebraic witness API.
    """
    if type(max_nodes) is not int or max_nodes < 1 or type(max_depth) is not int or max_depth < 0:
        raise ValueError("search budgets must be positive nodes and nonnegative depth")
    bounds = _box(box)
    values = tuple(rational(t) for t in target)
    if len(bounds) != source.spec.n_params or len(values) != len(source.polynomials):
        raise ValueError("realization search dimensions differ")
    equations = tuple(p - t for p, t in zip(source.polynomials, values, strict=True))
    visited = 0
    unresolved: list[RationalBox] = []
    found: RationalRealizationWitness | None = None

    def visit(current: RationalBox, depth: int) -> ExclusionLeaf | BoxSplit | None:
        nonlocal visited, found
        if visited >= max_nodes:
            unresolved.append(current)
            return None
        visited += 1
        for index, equation in enumerate(equations):
            enclosure = polynomial_interval(equation, current)
            if enclosure[1] < 0 or enclosure[0] > 0:
                return ExclusionLeaf(index, enclosure)
        point = tuple((lo + hi) / 2 for lo, hi in current)
        if all(equation.evaluate(point) == 0 for equation in equations):
            found = rational_realization_witness(source, point, values)
            return None
        axis = max(range(len(current)), key=lambda i: current[i][1] - current[i][0])
        lo, hi = current[axis]
        if depth >= max_depth or lo == hi:
            unresolved.append(current)
            return None
        midpoint = (lo + hi) / 2
        left_box = current[:axis] + ((lo, midpoint),) + current[axis + 1 :]
        right_box = current[:axis] + ((midpoint, hi),) + current[axis + 1 :]
        left = visit(left_box, depth + 1)
        if found is not None:
            return None
        right = visit(right_box, depth + 1)
        return (
            BoxSplit(axis, midpoint, left, right)
            if left is not None and right is not None
            else None
        )

    tree = visit(bounds, 0)
    if found is not None:
        return BoundedRealizationReport(
            "realizable_in_box",
            source.fingerprint,
            values,
            bounds,
            visited,
            (),
            None,
            found,
            "exact rational midpoint realizes every compiled coefficient",
        )
    if tree is not None:
        certificate = PolynomialBoxCertificate(source.fingerprint, values, bounds, tree)
        if not certificate.verify(source):
            raise ArithmeticError("internal exclusion coverage failed replay")
        return BoundedRealizationReport(
            "excluded_on_box",
            source.fingerprint,
            values,
            bounds,
            visited,
            (),
            certificate,
            None,
            "every leaf excludes one exact coefficient equation; scope is this box only",
        )
    return BoundedRealizationReport(
        "inconclusive",
        source.fingerprint,
        values,
        bounds,
        visited,
        tuple(unresolved),
        None,
        None,
        "search budget or interval dependency left unresolved parameter boxes",
    )


__all__ = [
    "BoundedRealizationReport",
    "BoxSplit",
    "ExclusionLeaf",
    "PolynomialBoxCertificate",
    "RationalBox",
    "bounded_realization_search",
    "polynomial_interval",
]
