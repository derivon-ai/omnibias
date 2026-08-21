# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Blind 2-edge-colouring search on ``K_5`` (not Erdős 183).

The origin is the monochrome colouring. Neighbors are 1-flips. This module
does not seed a named cycle colouring. A hit is ``R_2(3) > 5`` as a finite
obligation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

from omnibias.combinatorics.ramsey import is_triangle_free_edge_colouring
from omnibias.core.proof.discovery import Candidate, ExactCheck, Statement, run_discovery

_K5_EDGES = tuple(combinations(range(5), 2))
_K3_EDGES = tuple(combinations(range(3), 2))


def _honesty(*, discovered: bool) -> dict[str, bool]:
    return {
        "discovered_by_omnibias": discovered,
        "erdos_183_claim": False,
        "ramsey_colouring_replay": False,
        "ten_proofs_formalization_claim": False,
    }


def _colours(edges: Sequence[tuple[int, int]], bits: Sequence[int]) -> dict[tuple[int, int], int]:
    return {edge: int(bit) for edge, bit in zip(edges, bits, strict=True)}


def _triangle_count(n: int, bits: Sequence[int], edges: Sequence[tuple[int, int]]) -> int:
    colours = _colours(edges, bits)
    count = 0
    for a, b, c in combinations(range(n), 3):
        if colours[(a, b)] == colours[(b, c)] == colours[(a, c)]:
            count += 1
    return count


@dataclass
class RamseyColouringFamily:
    """2-edge-colourings of ``K_5`` as 10-bit tuples."""

    name: str = "ramsey_colouring_search"
    complete: bool = True
    bit_length: int = 10
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="ramsey_colouring_search",
            obligation="a triangle-free 2-edge-colouring of K_5 (R_2(3) > 5)",
            parent="Erdős 183 / R_k(3)=k^{Θ(k)}",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return 1 << self.bit_length

    def origin(self) -> tuple[int, ...]:
        return (0,) * self.bit_length

    def neighbors(self, candidate: Candidate) -> Sequence[tuple[int, ...]]:
        bits = tuple(int(v) for v in candidate)  # type: ignore[arg-type]
        out: list[tuple[int, ...]] = []
        for i in range(len(bits)):
            nxt = list(bits)
            nxt[i] = 1 - nxt[i]
            out.append(tuple(nxt))
        return out

    def score(self, candidate: Candidate) -> int:
        bits = tuple(int(v) for v in candidate)  # type: ignore[arg-type]
        return -_triangle_count(5, bits, _K5_EDGES)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        bits = tuple(int(v) for v in candidate)  # type: ignore[arg-type]
        if len(bits) != self.bit_length or any(b not in (0, 1) for b in bits):
            return None
        colours = _colours(_K5_EDGES, bits)
        ok = is_triangle_free_edge_colouring(5, 2, colours)
        return ExactCheck(
            ok=ok,
            payload={
                "n": 5,
                "k": 2,
                "colours": [list(edge) + [bit] for edge, bit in zip(_K5_EDGES, bits, strict=True)],
                "finite_statement": "R_2(3) > 5",
                "honesty": _honesty(discovered=ok),
            },
        )


@dataclass
class RamseyK3UniversalFamily:
    """Cheap universal smoke: not every 2-colouring of ``K_3`` is monochrome."""

    name: str = "ramsey_k3_universal"
    complete: bool = True
    bit_length: int = 3
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="ramsey_k3_universal",
            obligation="every 2-edge-colouring of K_3 has a monochrome triangle",
            parent="finite Ramsey smoke",
            parent_status="already_false",
            existential=False,
        )
    )

    def cardinality(self) -> int:
        return 8

    def origin(self) -> tuple[int, ...]:
        return (0, 0, 0)

    def neighbors(self, candidate: Candidate) -> Sequence[tuple[int, ...]]:
        bits = tuple(int(v) for v in candidate)  # type: ignore[arg-type]
        out: list[tuple[int, ...]] = []
        for i in range(len(bits)):
            nxt = list(bits)
            nxt[i] = 1 - nxt[i]
            out.append(tuple(nxt))
        return out

    def score(self, candidate: Candidate) -> int:
        bits = tuple(int(v) for v in candidate)  # type: ignore[arg-type]
        return -_triangle_count(3, bits, _K3_EDGES)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        bits = tuple(int(v) for v in candidate)  # type: ignore[arg-type]
        if len(bits) != 3:
            return None
        # A counterexample to the universal is a triangle-free colouring.
        colours = _colours(_K3_EDGES, bits)
        ok = is_triangle_free_edge_colouring(3, 2, colours)
        return ExactCheck(
            ok=ok,
            payload={
                "n": 3,
                "k": 2,
                "honesty": _honesty(discovered=ok),
            },
        )


def search_ramsey_colouring(*, proposer: str = "score_guided", budget: int = 256) -> object:
    family = RamseyColouringFamily()
    return run_discovery(family.statement, family, proposer, budget=budget)


__all__ = [
    "RamseyColouringFamily",
    "RamseyK3UniversalFamily",
    "search_ramsey_colouring",
]
