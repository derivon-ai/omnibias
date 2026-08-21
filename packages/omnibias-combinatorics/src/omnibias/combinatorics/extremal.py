# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Structural replay of the compactness / 2-degenerate templates (not Erdős 146 / 180).

The graphs ``C4``, ``C6``, ``jTemplate``, ``kTemplate``, and ``pairGraph(4,2)``
are encoded as adjacency lists. Predicates are finite. This module does **not**
discharge ``¬IsCompactFamily`` or an ``atTop`` extremal inequality.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from omnibias.core.proof.discovery import Candidate, ExactCheck, Statement

Graph = dict[int, frozenset[int]]


def _undirected(edges: Iterable[tuple[int, int]], n: int) -> Graph:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return {i: frozenset(nbrs) for i, nbrs in adj.items()}


def cycle_graph(n: int) -> Graph:
    return _undirected(((i, (i + 1) % n) for i in range(n)), n)


def c4() -> Graph:
    return cycle_graph(4)


def c6() -> Graph:
    return cycle_graph(6)


def _j_base(copy: int, base: int) -> int:
    if base == 0:
        return 0 if copy == 0 else 1
    if base == 1:
        return 2
    return 3


def j_template() -> Graph:
    """21-vertex ``jTemplate`` (Lean ``JVertex`` / ``jTemplateRelation``)."""
    # 0..3 : inl inl Fin 4
    # 4 + 2*copy + center : inl inr
    # 8 + copy*6 + i*2 + j : inr inl
    # 20 : unit
    edges: list[tuple[int, int]] = []
    for copy in range(2):
        for i in range(3):
            for j in range(2):
                c_idx = 8 + copy * 6 + i * 2 + j
                edges.append((_j_base(copy, i), c_idx))
                b_idx = 4 + 2 * copy + j
                edges.append((b_idx, c_idx))
    edges.append((0, 20))
    edges.append((1, 20))
    return _undirected(edges, 21)


def _subdivision_index(kind: str, *coords: int) -> int:
    if kind == "base":
        return coords[0]
    if kind == "center":
        return 3 + coords[0]
    base, center = coords
    return 6 + 3 * base + center


def subdivision_graph(k: int) -> Graph:
    """``SubdivisionGraph k`` on ``(Fin 3 ⊕ Fin k) ⊕ (Fin 3 × Fin k)``."""
    n = 3 + k + 3 * k
    edges: list[tuple[int, int]] = []
    for base in range(3):
        for center in range(k):
            pair = _subdivision_index("pair", base, center)
            edges.append((_subdivision_index("base", base), pair))
            edges.append((_subdivision_index("center", center), pair))
    return _undirected(edges, n)


def k_template() -> Graph:
    """30-vertex ``kTemplate``: two copies of ``SubdivisionGraph 3`` plus a bridge."""
    one = subdivision_graph(3)
    n = 15
    edges: list[tuple[int, int]] = []
    for copy in range(2):
        offset = copy * n
        for u, nbrs in one.items():
            for v in nbrs:
                if u < v:
                    edges.append((u + offset, v + offset))
    center0 = _subdivision_index("center", 0)
    edges.append((center0, n + center0))
    return _undirected(edges, 2 * n)


def pair_graph(base_size: int = 4, depth: int = 2) -> Graph:
    """Layered pair graph: level ``i+1`` is the 2-subsets of level ``i``."""
    if base_size < 2 or depth < 1:
        raise ValueError("base_size >= 2 and depth >= 1 required")
    layers: list[list[tuple[int, ...]]] = [[(i,) for i in range(base_size)]]
    for _ in range(depth):
        prev = layers[-1]
        nxt = [tuple(sorted((a, b))) for a in range(len(prev)) for b in range(a + 1, len(prev))]
        layers.append(nxt)
    offsets = []
    total = 0
    for layer in layers:
        offsets.append(total)
        total += len(layer)
    edges: list[tuple[int, int]] = []
    for level in range(1, len(layers)):
        for idx, pair in enumerate(layers[level]):
            upper = offsets[level] + idx
            for parent in pair:
                edges.append((upper, offsets[level - 1] + parent))
    return _undirected(edges, total)


def is_connected(graph: Graph) -> bool:
    if not graph:
        return True
    start = next(iter(graph))
    seen = {start}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        for nbr in graph[v]:
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return seen == set(graph)


def is_bipartite(graph: Graph) -> bool:
    colour: dict[int, int] = {}
    for start in graph:
        if start in colour:
            continue
        colour[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for nbr in graph[v]:
                if nbr not in colour:
                    colour[nbr] = 1 - colour[v]
                    queue.append(nbr)
                elif colour[nbr] == colour[v]:
                    return False
    return True


def has_cycle(graph: Graph) -> bool:
    seen: set[int] = set()

    def dfs(v: int, parent: int | None) -> bool:
        seen.add(v)
        for nbr in graph[v]:
            if nbr not in seen:
                if dfs(nbr, v):
                    return True
            elif nbr != parent:
                return True
        return False

    return any(dfs(start, None) for start in graph if start not in seen)


def is_two_degenerate(graph: Graph) -> bool:
    remaining = {v: set(nbrs) for v, nbrs in graph.items()}
    while remaining:
        peel = [v for v, nbrs in remaining.items() if len(nbrs) <= 2]
        if not peel:
            return False
        for v in peel:
            for nbr in remaining[v]:
                remaining[nbr].discard(v)
            del remaining[v]
    return True


def max_degree(graph: Graph) -> int:
    return max((len(nbrs) for nbrs in graph.values()), default=0)


def _honesty(*, replay: bool) -> dict[str, bool]:
    return {
        "discovered_by_omnibias": False,
        "erdos_146_claim": False,
        "erdos_180_claim": False,
        "extremal_graph_replay": replay,
        "ten_proofs_formalization_claim": False,
    }


def verify_forbidden_family() -> dict[str, Any]:
    graphs = {"C4": c4(), "C6": c6(), "j_template": j_template(), "k_template": k_template()}
    checks = {}
    ok = True
    for name, graph in graphs.items():
        row = {
            "n": len(graph),
            "connected": is_connected(graph),
            "bipartite": is_bipartite(graph),
            "has_cycle": has_cycle(graph),
        }
        checks[name] = row
        if not (row["connected"] and row["bipartite"] and row["has_cycle"]):
            ok = False
    return {
        "kind": "extremal_forbidden_family",
        "checks": checks,
        "replay_ok": ok,
        "honesty": _honesty(replay=ok),
    }


def verify_pair_graph() -> dict[str, Any]:
    graph = pair_graph(4, 2)
    two_deg = is_two_degenerate(graph)
    deg = max_degree(graph)
    ok = is_connected(graph) and is_bipartite(graph) and two_deg and deg > 2
    return {
        "kind": "extremal_pair_graph",
        "n": len(graph),
        "connected": is_connected(graph),
        "bipartite": is_bipartite(graph),
        "two_degenerate": two_deg,
        "max_degree": deg,
        "replay_ok": ok,
        "honesty": _honesty(replay=ok),
    }


_TEMPLATE_NAMES = ("C4", "C6", "j_template", "k_template")


def _template_graph(name: str) -> Graph:
    if name == "C4":
        return c4()
    if name == "C6":
        return c6()
    if name == "j_template":
        return j_template()
    if name == "k_template":
        return k_template()
    raise ValueError(f"unknown template {name!r}")


@dataclass
class ExtremalSearchFamily:
    """Which named template satisfies the connected-bipartite-cycle predicate."""

    name: str = "extremal_template_search"
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="extremal_template_search",
            obligation="a named template that is connected, bipartite, and has a cycle",
            parent="Erdős 146 / 180",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return len(_TEMPLATE_NAMES)

    def origin(self) -> str:
        return _TEMPLATE_NAMES[0]

    def neighbors(self, candidate: Candidate) -> Sequence[str]:
        current = str(candidate)
        return [name for name in _TEMPLATE_NAMES if name != current]

    def score(self, candidate: Candidate) -> int:
        graph = _template_graph(str(candidate))
        return int(is_connected(graph)) + int(is_bipartite(graph)) + int(has_cycle(graph))

    def check(self, candidate: Candidate) -> ExactCheck | None:
        name = str(candidate)
        if name not in _TEMPLATE_NAMES:
            return None
        graph = _template_graph(name)
        ok = is_connected(graph) and is_bipartite(graph) and has_cycle(graph)
        return ExactCheck(
            ok=ok,
            payload={
                "template": name,
                "n": len(graph),
                "honesty": {
                    "discovered_by_omnibias": ok,
                    "erdos_146_claim": False,
                    "erdos_180_claim": False,
                    "extremal_graph_replay": False,
                    "ten_proofs_formalization_claim": False,
                },
            },
        )


__all__ = [
    "ExtremalSearchFamily",
    "c4",
    "c6",
    "has_cycle",
    "is_bipartite",
    "is_connected",
    "is_two_degenerate",
    "j_template",
    "k_template",
    "max_degree",
    "pair_graph",
    "subdivision_graph",
    "verify_forbidden_family",
    "verify_pair_graph",
]
