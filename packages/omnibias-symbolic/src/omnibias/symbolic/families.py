# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact operator-span families for the discovery engine.

Each family has an exact ``Q`` checker. Score is a heuristic. Uniqueness is
span/box-scoped — never a parent theorem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb

from omnibias.core.proof.discovery import (
    Candidate,
    DiscoveredEquation,
    ExactCheck,
    Statement,
)
from omnibias.core.verified.coeffs import tanh_poly_coeffs_exact
from omnibias.symbolic.dimensional import integer_null_space
from omnibias.symbolic.discrete import try_recurrence
from omnibias.symbolic.lift import (
    planted_heat_rational,
    snap_sparse_equation,
    sparse_from_coeffs,
)

Number = int | Fraction


def _honesty(*, discovered: bool) -> dict[str, bool]:
    return {
        "discovered_by_omnibias": discovered,
        "erdos_146_claim": False,
        "erdos_180_claim": False,
        "erdos_183_claim": False,
        "jacobian_conjecture_proof_claim": False,
        "jacobian_n2_claim": False,
        "navier_stokes_proof_claim": False,
        "ten_proofs_formalization_claim": False,
    }


def _eval_int_poly(coeffs: Sequence[int], value: Fraction) -> Fraction:
    acc = Fraction(0)
    for power, coef in enumerate(coeffs):
        acc += coef * value**power
    return acc


def catalan_samples(n: int = 16) -> tuple[int, ...]:
    return tuple(comb(2 * k, k) // (k + 1) for k in range(n))


def fibonacci_samples(n: int = 16) -> tuple[int, ...]:
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return tuple(seq[:n])


def bell_samples(n: int = 17) -> tuple[int, ...]:
    row = [1]
    out = [1]
    for _ in range(n - 1):
        nxt = [row[-1]]
        for value in row:
            nxt.append(nxt[-1] + value)
        row = nxt
        out.append(row[0])
    return tuple(out)


@dataclass
class RecurrenceSpanFamily:
    """Bounded ``(order, index_degree)`` box with an exact nullity-one checker."""

    samples: tuple[Number, ...]
    max_order: int = 2
    max_index_degree: int = 1
    name: str = "recurrence_span"
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="recurrence_span",
            obligation="a unique P-recurrence in the (order, index_degree) box",
            parent="P-recursive sequences",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return self.max_order * (self.max_index_degree + 1)

    def origin(self) -> tuple[int, int]:
        return (1, 0)

    def neighbors(self, candidate: Candidate) -> Sequence[tuple[int, int]]:
        order, degree = candidate  # type: ignore[misc]
        out: list[tuple[int, int]] = []
        if order > 1:
            out.append((order - 1, degree))
        if order < self.max_order:
            out.append((order + 1, degree))
        if degree > 0:
            out.append((order, degree - 1))
        if degree < self.max_index_degree:
            out.append((order, degree + 1))
        return out

    def score(self, candidate: Candidate) -> int:
        order, degree = candidate  # type: ignore[misc]
        return -(10 * int(order) + int(degree))

    def check(self, candidate: Candidate) -> ExactCheck | None:
        order, degree = candidate  # type: ignore[misc]
        if not (1 <= int(order) <= self.max_order and 0 <= int(degree) <= self.max_index_degree):
            return None
        relation = try_recurrence(self.samples, int(order), int(degree))
        ok = relation is not None
        equation = None
        if relation is not None:
            flat = tuple(str(coef) for poly in relation.coefficients for coef in poly)
            equation = DiscoveredEquation(
                kind="recurrence",
                pretty=relation.pretty(),
                coefficients=flat,
            )
        return ExactCheck(
            ok=ok,
            payload={
                "order": int(order),
                "index_degree": int(degree),
                "equation": None if equation is None else equation.as_dict(),
                "annihilator": None if equation is None else equation.pretty,
                "honesty": _honesty(discovered=ok),
            },
        )


_ACT_TERMS = ("one", "y", "y2", "yp", "ypp")


@dataclass
class ActivationIdentityFamily:
    """Exact jet-monomial span for tanh, evaluated at rational ``t = tanh z``."""

    name: str = "activation_identity_exact"
    complete: bool = True
    activation: str = "tanh"
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="activation_identity_exact",
            obligation="a unique linear relation in {1, y, y^2, y', y''} for tanh",
            parent="Riccati identities",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return (1 << len(_ACT_TERMS)) - 1

    def origin(self) -> int:
        return 1

    def neighbors(self, candidate: Candidate) -> Sequence[int]:
        value = int(candidate)  # type: ignore[arg-type]
        out: list[int] = []
        for bit in range(len(_ACT_TERMS)):
            nxt = value ^ (1 << bit)
            if 0 < nxt <= self.cardinality():
                out.append(nxt)
        return out

    def score(self, candidate: Candidate) -> int:
        value = int(candidate)  # type: ignore[arg-type]
        has_yp = 5 if value & (1 << 3) else 0
        return has_yp - value.bit_count()

    def check(self, candidate: Candidate) -> ExactCheck | None:
        mask = int(candidate)  # type: ignore[arg-type]
        if not 0 < mask <= self.cardinality():
            return None
        samples = (
            Fraction(-3, 4),
            Fraction(-1, 2),
            Fraction(-1, 4),
            Fraction(1, 4),
            Fraction(1, 2),
            Fraction(3, 4),
        )
        t1 = tanh_poly_coeffs_exact(1)
        t2 = tanh_poly_coeffs_exact(2)
        columns: list[int] = [bit for bit in range(len(_ACT_TERMS)) if mask & (1 << bit)]
        frac_rows: list[list[Fraction]] = []
        for t in samples:
            values = {
                "one": Fraction(1),
                "y": t,
                "y2": t * t,
                "yp": _eval_int_poly(t1, t),
                "ypp": _eval_int_poly(t2, t),
            }
            frac_rows.append([values[_ACT_TERMS[bit]] for bit in columns])
        int_rows: list[list[int]] = []
        for row in frac_rows:
            lcm = 1
            for value in row:
                lcm = lcm * value.denominator // _gcd(lcm, value.denominator)
            int_rows.append([int(value * lcm) for value in row])
        null = integer_null_space(int_rows)
        ok = len(null) == 1
        pretty = None
        coeffs: tuple[str, ...] = ()
        if ok:
            vector = null[0]
            pieces: list[str] = []
            coeff_list: list[str] = []
            for bit, coef in zip(columns, vector, strict=True):
                coeff_list.append(str(coef))
                if coef == 0:
                    continue
                pieces.append(f"({coef})*{_ACT_TERMS[bit]}")
            pretty = " + ".join(pieces) + " = 0"
            coeffs = tuple(coeff_list)
        equation = (
            None
            if pretty is None
            else DiscoveredEquation(kind="polynomial_identity", pretty=pretty, coefficients=coeffs)
        )
        return ExactCheck(
            ok=ok,
            payload={
                "mask": mask,
                "activation": self.activation,
                "equation": None if equation is None else equation.as_dict(),
                "annihilator": pretty,
                "honesty": _honesty(discovered=ok),
            },
        )


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


@dataclass
class PdeOperatorSpanFamily:
    """Planted rational heat: ``u_t = D u_xx`` on an integer grid."""

    diffusivity: Fraction = Fraction(1, 8)
    name: str = "pde_operator_span"
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="pde_operator_span",
            obligation="snapped heat-operator coefficients with identically zero residual",
            parent="heat equation",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return 1

    def origin(self) -> str:
        return "heat"

    def neighbors(self, candidate: Candidate) -> Sequence[str]:
        return ()

    def score(self, candidate: Candidate) -> int:
        return 0

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if candidate != "heat":
            return None
        design, target, names = planted_heat_rational(diffusivity=self.diffusivity)
        soft = sparse_from_coeffs(
            (0.0, 0.0, float(self.diffusivity)),
            names,
        )
        snapped = snap_sparse_equation(soft, design, target, denom_bound=16)
        ok = snapped is not None
        return ExactCheck(
            ok=ok,
            payload={
                "diffusivity": str(self.diffusivity),
                "equation": None if snapped is None else snapped.as_dict(),
                "annihilator": None if snapped is None else snapped.pretty,
                "honesty": _honesty(discovered=ok),
            },
        )


__all__ = [
    "ActivationIdentityFamily",
    "PdeOperatorSpanFamily",
    "RecurrenceSpanFamily",
    "bell_samples",
    "catalan_samples",
    "fibonacci_samples",
]
