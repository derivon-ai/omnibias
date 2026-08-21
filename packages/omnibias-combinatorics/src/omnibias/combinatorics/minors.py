# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Named forbidden-minor hypotheses on hosts with ``n≤5``.

This is a finite branch-set test, not Robertson–Seymour and not Erdős 146 / 180.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product

from omnibias.combinatorics.extremal import Graph, c4
from omnibias.core.proof.condition import (
    ConditionHypothesis,
    ConditionToken,
    GrammarSpec,
    condition_honesty,
    emit_condition,
    register_condition_sort,
)
from omnibias.core.proof.discovery import Candidate, ExactCheck
from omnibias.core.proof.observe import Observation


def _undirected(edges: Iterable[tuple[int, int]], n: int) -> Graph:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return {i: frozenset(nbrs) for i, nbrs in adj.items()}


def complete_graph(n: int) -> Graph:
    return _undirected(((i, j) for i in range(n) for j in range(i + 1, n)), n)


def path_graph(n: int) -> Graph:
    return _undirected(((i, i + 1) for i in range(n - 1)), n)


def k2() -> Graph:
    return complete_graph(2)


def k3() -> Graph:
    return complete_graph(3)


def k4() -> Graph:
    return complete_graph(4)


def _connected(graph: Graph, verts: frozenset[int]) -> bool:
    if not verts:
        return False
    start = next(iter(verts))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in graph[u]:
            if v in verts and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen == set(verts)


def contains_minor(host: Graph, pattern: Graph) -> bool:
    """``True`` iff ``pattern`` is a minor of ``host`` (branch-set witness, ``n≤5``)."""

    h_verts = list(host)
    p_verts = list(pattern)
    k = len(p_verts)
    n = len(h_verts)
    if k == 0 or k > n or n > 5:
        return False
    for coloring in product(range(k + 1), repeat=n):
        sets: list[set[int]] = [set() for _ in range(k)]
        for i, color in enumerate(coloring):
            if color > 0:
                sets[color - 1].add(h_verts[i])
        if any(len(branch) == 0 for branch in sets):
            continue
        if not all(_connected(host, frozenset(branch)) for branch in sets):
            continue
        ok_edges = True
        for i, u in enumerate(p_verts):
            for v in pattern[u]:
                j = p_verts.index(v)
                if i >= j:
                    continue
                if not any(b in host[a] for a in sets[i] for b in sets[j]):
                    ok_edges = False
                    break
            if not ok_edges:
                break
        if ok_edges:
            return True
    return False


_PATTERNS = {
    "K2": k2,
    "K3": k3,
    "C4": c4,
}


def _minor_hypothesis(name: str) -> ConditionHypothesis:
    return ConditionHypothesis(
        sort="forbidden_minor",
        tokens=(ConditionToken("forbidden_minor", name),),
    )


@dataclass
class ForbiddenMinorFamily:
    """Search named minors of a tiny host. Not a graph-minor theorem."""

    host: Graph = field(default_factory=k4)
    patterns: tuple[str, ...] = ("K2", "K3", "C4")
    name: str = "condition_forbidden_minor"
    grammar: GrammarSpec = field(
        default_factory=lambda: GrammarSpec(
            sorts=("forbidden_minor",),
            tokens_by_sort={"forbidden_minor": ("K2", "K3", "C4")},
            constructors=("add_minor_edge",),
            complete=False,
        )
    )

    def __post_init__(self) -> None:
        self.statement = emit_condition(
            _minor_hypothesis(self.patterns[0]),
            parent="graph minors",
            parent_status="already_true",
            obligation="the host contains a named minor in this grammar",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return len(self.patterns)

    def origin(self) -> ConditionHypothesis:
        return _minor_hypothesis(self.patterns[0])

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis) or not candidate.tokens:
            return ()
        current = candidate.tokens[0].name
        return tuple(
            _minor_hypothesis(name) for name in self.patterns if name != current
        )

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis) or not candidate.tokens:
            return 0
        return 1 if candidate.tokens[0].name == "K3" else 0

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "forbidden_minor":
            return None
        if not candidate.tokens:
            return None
        name = candidate.tokens[0].name
        builder = _PATTERNS.get(name)
        if builder is None:
            return None
        ok = contains_minor(self.host, builder())
        return ExactCheck(
            ok=ok,
            payload={
                "hypothesis": candidate.as_dict(),
                "pattern": name,
                "honesty": condition_honesty(discovered=ok),
            },
        )


def observation_k4() -> Observation:
    edges = tuple((i, j) for i in range(4) for j in range(i + 1, 4))
    return Observation(tag="k4", graph_n=4, graph_edges=edges)


def observation_path(n: int = 4) -> Observation:
    edges = tuple((i, i + 1) for i in range(n - 1))
    return Observation(tag="path", graph_n=n, graph_edges=edges)


def bind_forbidden_minor(observation: Observation | None = None) -> ForbiddenMinorFamily | None:
    if observation is None:
        return ForbiddenMinorFamily()
    if not observation.graph_edges and observation.graph_n <= 0:
        return None
    n = observation.graph_n
    if n <= 0:
        n = max((max(edge) for edge in observation.graph_edges), default=-1) + 1
    if n > 5:
        return None
    return ForbiddenMinorFamily(host=_undirected(observation.graph_edges, n))


def _register() -> None:
    register_condition_sort("forbidden_minor", bind_forbidden_minor)


_register()


__all__ = [
    "ForbiddenMinorFamily",
    "bind_forbidden_minor",
    "complete_graph",
    "contains_minor",
    "k2",
    "k3",
    "k4",
    "observation_k4",
    "observation_path",
    "path_graph",
]
