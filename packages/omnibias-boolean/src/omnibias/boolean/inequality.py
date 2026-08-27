# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Boolean inequality adapter (theory 09-30).

Exact ``solve_system`` on a finite cube. Propose is skipped.
``budget == 0`` is ``search_incomplete``. ``p_equals_np_claim`` stays false.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from omnibias.boolean._core.equations import solve_system
from omnibias.boolean._core.truth_table import TruthTable, check_truth_table
from omnibias.core.proof.discovery import ExactCheck
from omnibias.core.proof.inequality import (
    InequalitySort,
    InequalitySystem,
    Proposal,
    RationalWitness,
    register_inequality_backend,
)


def _tables(raw: object) -> tuple[TruthTable, ...]:
    tables: list[TruthTable] = []
    for item in raw if isinstance(raw, Sequence) else ():
        table = tuple(int(bit) for bit in item)
        check_truth_table(table)
        tables.append(table)
    return tuple(tables)


class BooleanInequalityBackend:
    sort: ClassVar[InequalitySort] = "boolean"

    def propose(self, system: InequalitySystem) -> Proposal:
        return Proposal(method="exact_skip", skipped=True)

    def rationalize(
        self, system: InequalitySystem, proposal: Proposal
    ) -> RationalWitness:
        return RationalWitness(method="none")

    def check(
        self, system: InequalitySystem, witness: RationalWitness
    ) -> ExactCheck:
        if int(system.data.get("budget", 1)) == 0:
            return ExactCheck(
                ok=False,
                payload={
                    "role": "inconclusive",
                    "method": "search_incomplete",
                    "honesty": {
                        "complete_solver": False,
                        "p_equals_np_claim": False,
                    },
                    "detail": "search_incomplete",
                },
            )
        tables = _tables(system.data.get("tables", ()))
        if not tables:
            return ExactCheck(
                ok=False,
                payload={
                    "role": "inconclusive",
                    "method": "empty_system",
                    "honesty": {"complete_solver": False},
                    "detail": "no_tables",
                },
            )
        solution = solve_system(tables)
        honesty = {
            "complete_solver": True,
            "p_equals_np_claim": False,
            "unsat_proof": not solution.consistent,
        }
        if solution.consistent:
            sols = solution.enumerate_solutions()
            return ExactCheck(
                ok=True,
                payload={
                    "role": "witness",
                    "method": "solve_system",
                    "honesty": honesty,
                    "detail": "boolean_sat",
                    "n_solutions": len(sols),
                },
            )
        return ExactCheck(
            ok=False,
            payload={
                "role": "empty",
                "method": "solve_system",
                "honesty": honesty,
                "detail": "boolean_unsat",
                "complete": True,
            },
        )


def _register() -> None:
    register_inequality_backend(BooleanInequalityBackend())


_register()


__all__ = ["BooleanInequalityBackend"]
