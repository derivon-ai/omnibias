# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Holonomic guess families: prefix-verified annihilators in a degree box.

All-``n`` continuation stays the Zeilberger / :class:`HolonomicProof` obligation.
A hit here is prefix-exact only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, factorial

from omnibias.core.proof.discovery import Candidate, DiscoveredEquation, ExactCheck, Statement
from omnibias.holonomic._core.guess import guess_algebraic, guess_dfinite, guess_recurrence


def _honesty(*, discovered: bool) -> dict[str, bool]:
    return {
        "discovered_by_omnibias": discovered,
        "jacobian_conjecture_proof_claim": False,
        "jacobian_n2_claim": False,
        "navier_stokes_proof_claim": False,
        "prefix_verified_only": True,
    }


def fibonacci_samples(n: int = 16) -> tuple[int, ...]:
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return tuple(seq[:n])


def exp_series(n: int = 14) -> tuple[Fraction, ...]:
    return tuple(Fraction(1, factorial(m)) for m in range(n))


def linear_series(n: int = 10) -> tuple[Fraction, ...]:
    """Taylor coefficients of ``f(x) = x``."""

    out = [Fraction(0)] * n
    if n > 1:
        out[1] = Fraction(1)
    return tuple(out)


def catalan_samples(n: int = 14) -> tuple[int, ...]:
    return tuple(comb(2 * k, k) // (k + 1) for k in range(n))


def _box_neighbors(
    candidate: tuple[int, int],
    *,
    max_a: int,
    max_b: int,
    min_a: int = 1,
) -> list[tuple[int, int]]:
    a, b = candidate
    out: list[tuple[int, int]] = []
    if a > min_a:
        out.append((a - 1, b))
    if a < max_a:
        out.append((a + 1, b))
    if b > 0:
        out.append((a, b - 1))
    if b < max_b:
        out.append((a, b + 1))
    return out


@dataclass
class HolonomicRecurrenceGuessFamily:
    samples: tuple[int | Fraction, ...]
    max_order: int = 2
    max_index_degree: int = 1
    name: str = "holonomic_recurrence_guess"
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="holonomic_recurrence_guess",
            obligation="a prefix-verified P-recurrence in the (order, degree) box",
            parent="P-recursive sequences",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return self.max_order * (self.max_index_degree + 1)

    def origin(self) -> tuple[int, int]:
        return (1, 0)

    def neighbors(self, candidate: Candidate) -> Sequence[tuple[int, int]]:
        return _box_neighbors(
            candidate,  # type: ignore[arg-type]
            max_a=self.max_order,
            max_b=self.max_index_degree,
        )

    def score(self, candidate: Candidate) -> int:
        order, degree = candidate  # type: ignore[misc]
        return -(10 * int(order) + int(degree))

    def check(self, candidate: Candidate) -> ExactCheck | None:
        order, degree = candidate  # type: ignore[misc]
        if not (1 <= int(order) <= self.max_order and 0 <= int(degree) <= self.max_index_degree):
            return None
        op = guess_recurrence(
            self.samples,
            max_order=int(order),
            max_index_degree=int(degree),
        )
        ok = op is not None
        pretty = None if op is None else str(op)
        equation = (
            None
            if pretty is None
            else DiscoveredEquation(kind="ore", pretty=pretty, coefficients=())
        )
        return ExactCheck(
            ok=ok,
            payload={
                "order": int(order),
                "index_degree": int(degree),
                "equation": None if equation is None else equation.as_dict(),
                "annihilator": pretty,
                "honesty": _honesty(discovered=ok),
            },
        )


@dataclass
class HolonomicDFiniteGuessFamily:
    series: tuple[int | Fraction, ...]
    max_order: int = 2
    max_degree: int = 2
    name: str = "holonomic_dfinite_guess"
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="holonomic_dfinite_guess",
            obligation="a prefix-verified differential annihilator in the degree box",
            parent="D-finite functions",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return self.max_order * (self.max_degree + 1)

    def origin(self) -> tuple[int, int]:
        return (1, 0)

    def neighbors(self, candidate: Candidate) -> Sequence[tuple[int, int]]:
        return _box_neighbors(candidate, max_a=self.max_order, max_b=self.max_degree)  # type: ignore[arg-type]

    def score(self, candidate: Candidate) -> int:
        order, degree = candidate  # type: ignore[misc]
        return -(10 * int(order) + int(degree))

    def check(self, candidate: Candidate) -> ExactCheck | None:
        order, degree = candidate  # type: ignore[misc]
        if not (1 <= int(order) <= self.max_order and 0 <= int(degree) <= self.max_degree):
            return None
        op = guess_dfinite(self.series, max_order=int(order), max_degree=int(degree))
        ok = op is not None
        pretty = None if op is None else str(op)
        equation = (
            None
            if pretty is None
            else DiscoveredEquation(kind="ore", pretty=pretty, coefficients=())
        )
        return ExactCheck(
            ok=ok,
            payload={
                "order": int(order),
                "degree": int(degree),
                "equation": None if equation is None else equation.as_dict(),
                "annihilator": pretty,
                "honesty": _honesty(discovered=ok),
            },
        )


@dataclass
class HolonomicAlgebraicGuessFamily:
    series: tuple[int | Fraction, ...]
    max_x_degree: int = 2
    max_y_degree: int = 2
    name: str = "holonomic_algebraic_guess"
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="holonomic_algebraic_guess",
            obligation="a prefix-verified algebraic equation P(x, y)=0 in the degree box",
            parent="algebraic functions",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return (self.max_x_degree + 1) * self.max_y_degree

    def origin(self) -> tuple[int, int]:
        return (0, 1)

    def neighbors(self, candidate: Candidate) -> Sequence[tuple[int, int]]:
        dx, dy = candidate  # type: ignore[misc]
        return _box_neighbors(
            (int(dx), int(dy)),
            max_a=self.max_x_degree,
            max_b=self.max_y_degree,
            min_a=0,
        )

    def score(self, candidate: Candidate) -> int:
        dx, dy = candidate  # type: ignore[misc]
        return -(10 * int(dy) + int(dx))

    def check(self, candidate: Candidate) -> ExactCheck | None:
        dx, dy = candidate  # type: ignore[misc]
        if not (0 <= int(dx) <= self.max_x_degree and 1 <= int(dy) <= self.max_y_degree):
            return None
        poly = guess_algebraic(
            self.series,
            max_x_degree=int(dx),
            max_y_degree=int(dy),
        )
        ok = poly is not None
        pretty = None if poly is None else str(poly)
        equation = (
            None
            if pretty is None
            else DiscoveredEquation(kind="polynomial_identity", pretty=pretty, coefficients=())
        )
        return ExactCheck(
            ok=ok,
            payload={
                "max_x_degree": int(dx),
                "max_y_degree": int(dy),
                "equation": None if equation is None else equation.as_dict(),
                "annihilator": pretty,
                "honesty": _honesty(discovered=ok),
            },
        )


__all__ = [
    "HolonomicAlgebraicGuessFamily",
    "HolonomicDFiniteGuessFamily",
    "HolonomicRecurrenceGuessFamily",
    "catalan_samples",
    "exp_series",
    "fibonacci_samples",
    "linear_series",
]
