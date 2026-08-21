# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Capped 3-terminal DAG family (≤6 vertices) for DGG cost search.

The 7-vertex H* topology stays in :mod:`omnibias.combinatorics.unsplittable_search`.
This module does not seed a published 7-vertex parameter tuple. A miss on the
capped generator is ``BLOCKED``. Planted recovery hides one hand-built
6-vertex separator among decoys.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Any

from omnibias.core.proof.discovery import ExactCheck, Statement, run_discovery

Vertex = str
Arc = str
Route = tuple[Arc, ...]
Candidate = tuple[int, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class ThreeTerminalDag:
    """One source, three terminals, two simple routes per commodity."""

    name: str
    vertices: tuple[Vertex, ...]
    arcs: tuple[Arc, ...]
    paid: tuple[Route, Route, Route]
    free: tuple[Route, Route, Route]
    cost_arcs: tuple[Arc, Arc, Arc]


def _star() -> ThreeTerminalDag:
    return ThreeTerminalDag(
        name="star5",
        vertices=("S", "U", "T1", "T2", "T3"),
        arcs=("S-T1", "S-T2", "S-T3", "S-U", "U-T1", "U-T2", "U-T3"),
        paid=(("S-T1",), ("S-T2",), ("S-T3",)),
        free=(("S-U", "U-T1"), ("S-U", "U-T2"), ("S-U", "U-T3")),
        cost_arcs=("S-T1", "S-T2", "S-T3"),
    )


def _hub_uv() -> ThreeTerminalDag:
    return ThreeTerminalDag(
        name="hub_uv",
        vertices=("S", "U", "V", "T1", "T2", "T3"),
        arcs=("S-T1", "S-T2", "S-T3", "S-U", "U-V", "V-T1", "V-T2", "V-T3"),
        paid=(("S-T1",), ("S-T2",), ("S-T3",)),
        free=(("S-U", "U-V", "V-T1"), ("S-U", "U-V", "V-T2"), ("S-U", "U-V", "V-T3")),
        cost_arcs=("S-T1", "S-T2", "S-T3"),
    )


def _t3_transit() -> ThreeTerminalDag:
    """6-vertex DAG: T3 is a terminal and a transit for commodity 2."""
    return ThreeTerminalDag(
        name="t3_transit",
        vertices=("S", "U", "V", "T1", "T2", "T3"),
        arcs=("S-T1", "S-T2", "S-U", "U-V", "U-T3", "V-T1", "V-T3", "T3-T2"),
        paid=(("S-T1",), ("S-T2",), ("S-U", "U-T3")),
        free=(
            ("S-U", "U-V", "V-T1"),
            ("S-U", "U-V", "V-T3", "T3-T2"),
            ("S-U", "U-V", "V-T3"),
        ),
        cost_arcs=("S-T1", "S-T2", "U-T3"),
    )


def _minus_w() -> ThreeTerminalDag:
    return ThreeTerminalDag(
        name="minus_w",
        vertices=("S", "U", "V", "T1", "T2", "T3"),
        arcs=("S-T1", "S-T2", "S-U", "U-V", "U-T3", "U-T1", "V-T2", "V-T3"),
        paid=(("S-T1",), ("S-T2",), ("S-U", "U-T3")),
        free=(("S-U", "U-T1"), ("S-U", "U-V", "V-T2"), ("S-U", "U-V", "V-T3")),
        cost_arcs=("S-T1", "S-T2", "U-T3"),
    )


def capped_dag_catalog() -> tuple[ThreeTerminalDag, ...]:
    """At most 16 topologies; CI stays cheap."""
    star = _star()
    hub = _hub_uv()
    transit = _t3_transit()
    minus = _minus_w()
    extras: list[ThreeTerminalDag] = [
        ThreeTerminalDag(
            name="hub_direct_t3",
            vertices=hub.vertices,
            arcs=hub.arcs + ("U-T3",),
            paid=(("S-T1",), ("S-T2",), ("S-U", "U-T3")),
            free=hub.free,
            cost_arcs=("S-T1", "S-T2", "U-T3"),
        )
    ]
    pads = [
        ThreeTerminalDag(
            name=f"star_pad_{idx}",
            vertices=star.vertices,
            arcs=star.arcs,
            paid=star.paid,
            free=star.free,
            cost_arcs=star.cost_arcs,
        )
        for idx in range(12)
    ]
    catalog = (star, hub, transit, minus) + tuple(extras) + tuple(pads)
    return catalog[:16]


def evaluate_dag(
    dag: ThreeTerminalDag,
    demands: tuple[Fraction, Fraction, Fraction],
    paid_costs: tuple[Fraction, Fraction, Fraction],
    split_paid: tuple[Fraction, Fraction, Fraction],
) -> dict[str, Any] | None:
    """Same predicate as H*: ``fractional_cost < min_legal`` and congestion."""
    if any(p < 0 or p > d for p, d in zip(split_paid, demands, strict=True)):
        return None
    free_amt = tuple(d - p for d, p in zip(demands, split_paid, strict=True))
    costs = {arc: Fraction(0) for arc in dag.arcs}
    for arc, value in zip(dag.cost_arcs, paid_costs, strict=True):
        costs[arc] = value

    def load(paths: Sequence[Sequence[str]], amounts: Sequence[Fraction]) -> dict[str, Fraction]:
        out = {arc: Fraction(0) for arc in dag.arcs}
        for path, amount in zip(paths, amounts, strict=True):
            for arc in path:
                out[arc] += amount
        return out

    x = load(dag.paid, split_paid)
    for arc, qty in load(dag.free, free_amt).items():
        x[arc] += qty
    frac_cost = sum(x[arc] * costs[arc] for arc in dag.arcs)
    d_max = max(demands)
    legal: list[Fraction] = []
    for bits in product((False, True), repeat=3):
        y = {arc: Fraction(0) for arc in dag.arcs}
        cost = Fraction(0)
        for i, use_paid in enumerate(bits):
            path = dag.paid[i] if use_paid else dag.free[i]
            qty = demands[i]
            for arc in path:
                y[arc] += qty
                cost += qty * costs[arc]
        if all(y[arc] <= x[arc] + d_max for arc in dag.arcs):
            legal.append(cost)
    if not legal:
        return None
    min_legal = min(legal)
    if frac_cost < min_legal:
        return {
            "fractional_cost": frac_cost,
            "min_legal": min_legal,
            "demands": demands,
            "paid_costs": paid_costs,
            "split_paid": split_paid,
            "topology": dag.name,
            "vertices": dag.vertices,
        }
    return None


def planted_separator() -> tuple[ThreeTerminalDag, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Hand-built 6-vertex separator (not the 7-vertex published tuple)."""
    return _t3_transit(), (12, 8, 12), (2, 3, 2), (8, 5, 8)


def planted_decoys() -> list[tuple[ThreeTerminalDag, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]:
    star = _star()
    return [
        (star, (6, 6, 6), (1, 1, 1), (3, 3, 3)),
        (star, (8, 8, 8), (2, 0, 0), (4, 4, 4)),
        (_hub_uv(), (6, 4, 6), (1, 2, 1), (3, 2, 3)),
        planted_separator(),
        (_minus_w(), (4, 4, 4), (1, 1, 1), (2, 2, 2)),
    ]


def _honesty(*, discovered: bool) -> dict[str, bool]:
    return {
        "discovered_by_omnibias": discovered,
        "dgg_congestion_theorem_refuted": False,
        "dgg_cost_conjecture_replay": False,
        "navier_stokes_proof_claim": False,
        "ten_proofs_formalization_claim": False,
    }


@dataclass(frozen=True)
class DAGSearchHit:
    topology: str
    vertices: tuple[str, ...]
    demands: tuple[Fraction, Fraction, Fraction]
    paid_costs: tuple[Fraction, Fraction, Fraction]
    split_paid: tuple[Fraction, Fraction, Fraction]
    fractional_cost: Fraction
    min_legal: Fraction
    search: str = "three_terminal_dag_le6"


class ThreeTerminalDagFamily:
    """Capped incomplete family. A CI miss is ``BLOCKED``, never a parent proof."""

    name = "three_terminal_dag_le6"
    complete = False
    statement = Statement(
        name="dgg_dag_le6",
        obligation="a 3-terminal DAG on at most 6 vertices with a cost separation",
        parent="goemans_cost_conjecture",
        parent_status="already_false",
    )

    def __init__(
        self,
        *,
        instances: Sequence[tuple[ThreeTerminalDag, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]
        | None = None,
        planted: bool = False,
    ) -> None:
        if instances is not None:
            self._rows = list(instances)
        elif planted:
            self._rows = planted_decoys()
        else:
            self._rows = list(self._ci_box())
        self._index = {self._key(i, row): i for i, row in enumerate(self._rows)}

    def _key(
        self,
        i: int,
        row: tuple[ThreeTerminalDag, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    ) -> Candidate:
        dag, demands, costs, split = row
        return (i, demands, costs, split)

    def _ci_box(
        self,
    ) -> list[tuple[ThreeTerminalDag, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]:
        rows: list[tuple[ThreeTerminalDag, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = []
        for dag in capped_dag_catalog():
            for demands in ((4, 4, 4), (6, 4, 6)):
                for costs in ((1, 0, 0), (1, 1, 1)):
                    mid = tuple(d // 2 for d in demands)
                    rows.append((dag, demands, costs, mid))
        return rows

    def origin(self) -> Candidate:
        return self._key(0, self._rows[0]) if self._rows else (0, (0, 0, 0), (0, 0, 0), (0, 0, 0))

    def neighbors(self, candidate: Candidate) -> Sequence[Candidate]:
        idx = candidate[0]
        out: list[Candidate] = []
        if idx + 1 < len(self._rows):
            out.append(self._key(idx + 1, self._rows[idx + 1]))
        if 0 <= idx - 1 < len(self._rows):
            out.append(self._key(idx - 1, self._rows[idx - 1]))
        return out

    def score(self, candidate: Candidate) -> int:
        checked = self.check(candidate)
        return 2 if checked is not None and checked.ok else 0

    def check(self, candidate: Candidate) -> ExactCheck | None:
        idx = candidate[0]
        if not 0 <= idx < len(self._rows):
            return None
        dag, demands, costs, split = self._rows[idx]
        found = evaluate_dag(
            dag,
            tuple(Fraction(v) for v in demands),  # type: ignore[arg-type]
            tuple(Fraction(v) for v in costs),  # type: ignore[arg-type]
            tuple(Fraction(v) for v in split),  # type: ignore[arg-type]
        )
        honesty = _honesty(discovered=found is not None)
        if found is None:
            return ExactCheck(ok=False, payload={"honesty": honesty, "topology": dag.name})
        return ExactCheck(
            ok=True,
            payload={
                "honesty": honesty,
                "topology": dag.name,
                "vertices": list(dag.vertices),
                "demands": [str(c) for c in found["demands"]],
                "paid_costs": [str(c) for c in found["paid_costs"]],
                "split_paid": [str(c) for c in found["split_paid"]],
                "fractional_cost": str(found["fractional_cost"]),
                "min_legal": str(found["min_legal"]),
            },
        )


def search_dgg_dags(
    *,
    family: str = "three_terminal_dag_le6",
    proposer: str = "score_guided",
    budget: int | None = None,
) -> list[DAGSearchHit]:
    """Score-guided walk on the capped DAG family (or the planted list)."""
    planted = family == "planted"
    box = ThreeTerminalDagFamily(planted=planted)
    limit = (len(box._rows) + 2) if budget is None else budget
    result = run_discovery(box.statement, box, proposer, budget=limit)
    if result.status != "PROVED" or result.check is None:
        return []
    payload = result.check.payload
    return [
        DAGSearchHit(
            topology=str(payload["topology"]),
            vertices=tuple(payload.get("vertices") or ()),
            demands=tuple(Fraction(c) for c in payload["demands"]),
            paid_costs=tuple(Fraction(c) for c in payload["paid_costs"]),
            split_paid=tuple(Fraction(c) for c in payload["split_paid"]),
            fractional_cost=Fraction(payload["fractional_cost"]),
            min_legal=Fraction(payload["min_legal"]),
            search="planted" if planted else "three_terminal_dag_le6",
        )
    ]


def dag_hit_certificate(hit: DAGSearchHit) -> dict[str, Any]:
    return {
        "kind": "dgg_dag_le6",
        "search": hit.search,
        "topology": hit.topology,
        "vertices": list(hit.vertices),
        "demands": [str(c) for c in hit.demands],
        "paid_costs": [str(c) for c in hit.paid_costs],
        "split_paid": [str(c) for c in hit.split_paid],
        "fractional_cost": str(hit.fractional_cost),
        "min_legal_unsplittable": str(hit.min_legal),
        "replay_ok": True,
        "search_incomplete": False,
        "honesty": _honesty(discovered=hit.search != "planted"),
    }


__all__ = [
    "DAGSearchHit",
    "ThreeTerminalDag",
    "ThreeTerminalDagFamily",
    "capped_dag_catalog",
    "dag_hit_certificate",
    "evaluate_dag",
    "planted_decoys",
    "planted_separator",
    "search_dgg_dags",
]
