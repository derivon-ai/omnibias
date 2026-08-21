# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""SOS degree box: positivity in a Gram span, not “only equations.”"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from omnibias.core.proof.discovery import Candidate, ExactCheck, Statement
from omnibias.sos.certify import certify_sos
from omnibias.sos.problem import Polynomial


def planted_sum_of_squares() -> Polynomial:
    x = Polynomial.variable(0, 2)
    y = Polynomial.variable(1, 2)
    one = Polynomial.constant(1.0, 2)
    return x * x + y * y + one


@dataclass
class SosDegreeFamily:
    """Candidate = half-degree; check = existing ``certify_sos`` on a planted poly."""

    polynomial: Polynomial = field(default_factory=planted_sum_of_squares)
    max_half_degree: int = 2
    name: str = "sos_degree_search"
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="sos_degree_search",
            obligation="the planted polynomial is SOS at some half-degree in the box",
            parent="global nonnegativity",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return self.max_half_degree

    def origin(self) -> int:
        return 1

    def neighbors(self, candidate: Candidate) -> Sequence[int]:
        value = int(candidate)  # type: ignore[arg-type]
        out: list[int] = []
        if value > 1:
            out.append(value - 1)
        if value < self.max_half_degree:
            out.append(value + 1)
        return out

    def score(self, candidate: Candidate) -> int:
        return -int(candidate)  # type: ignore[arg-type]

    def check(self, candidate: Candidate) -> ExactCheck | None:
        half = int(candidate)  # type: ignore[arg-type]
        if not 1 <= half <= self.max_half_degree:
            return None
        cert = certify_sos(self.polynomial, half_degree=half)
        ok = cert.status == "proved"
        return ExactCheck(
            ok=ok,
            payload={
                "half_degree": half,
                "status": cert.status,
                "honesty": {
                    "discovered_by_omnibias": ok,
                    "unproven_claim": False,
                },
            },
        )


__all__ = [
    "SosDegreeFamily",
    "planted_sum_of_squares",
]
