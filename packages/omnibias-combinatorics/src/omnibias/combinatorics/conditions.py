# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Pattern-language sorts: finite colouring and extremal-template predicates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

from omnibias.combinatorics.extremal import (
    ExtremalSearchFamily,
    has_cycle,
    is_bipartite,
    is_connected,
)
from omnibias.combinatorics.minors import _undirected
from omnibias.combinatorics.ramsey_search import RamseyColouringFamily
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


def _is_complete(n: int, edges: Sequence[tuple[int, int]]) -> bool:
    wanted = {(min(a, b), max(a, b)) for a, b in combinations(range(n), 2)}
    have = {(min(a, b), max(a, b)) for a, b in edges if a != b}
    return have == wanted


def _colouring_bits(observation: Observation) -> tuple[int, ...] | None:
    raw = observation.extra_map().get("colouring")
    if not raw:
        return None
    bits = tuple(int(part) for part in raw.replace(";", ",").split(",") if part)
    if len(bits) == 10 and all(bit in (0, 1) for bit in bits):
        return bits
    return None


@dataclass
class EdgeColouringFamily:
    """K5 2-edge-colourings. ``erdos_183_claim`` stays False."""

    planted: tuple[int, ...] | None = None
    name: str = "condition_edge_colouring"
    complete: bool = False
    empty_miss_detail: str = "no witness in enumerated grammar"

    def __post_init__(self) -> None:
        self._inner = RamseyColouringFamily()
        self.statement = emit_condition(
            ConditionHypothesis(
                sort="edge_colouring",
                tokens=(ConditionToken("edge_colouring", "K5"),),
            ),
            parent="finite Ramsey colourings",
            parent_status="already_true",
            obligation="a triangle-free 2-edge-colouring of K_5 (R_2(3) > 5)",
        )

    def cardinality(self) -> int:
        return 1 if self.planted is not None else self._inner.cardinality()

    def origin(self) -> tuple[int, ...]:
        return self.planted if self.planted is not None else self._inner.origin()

    def neighbors(self, candidate: Candidate) -> Sequence[tuple[int, ...]]:
        if self.planted is not None:
            return ()
        return self._inner.neighbors(candidate)

    def score(self, candidate: Candidate) -> int:
        return self._inner.score(candidate)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        inner = self._inner.check(candidate)
        if inner is None:
            return None
        payload = dict(inner.payload)
        honesty = dict(payload.get("honesty", {})) if isinstance(payload.get("honesty"), dict) else {}
        honesty.update(condition_honesty(discovered=inner.ok))
        honesty["erdos_183_claim"] = False
        payload["honesty"] = honesty
        return ExactCheck(ok=inner.ok, payload=payload)


def observation_k5() -> Observation:
    edges = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
    return Observation(graph_n=5, graph_edges=edges)


def bind_edge_colouring(observation: Observation | None = None) -> EdgeColouringFamily | None:
    if observation is None:
        return EdgeColouringFamily()
    bits = _colouring_bits(observation)
    if bits is not None:
        return EdgeColouringFamily(planted=bits)
    n = observation.graph_n
    if n <= 0:
        n = max((max(edge) for edge in observation.graph_edges), default=-1) + 1
    if n == 5 and _is_complete(5, observation.graph_edges):
        return EdgeColouringFamily()
    return None


@dataclass
class ExtremalTemplateConditionFamily:
    """Connected-bipartite-cycle predicate on a packed host or a named template."""

    host_ok: bool | None = None
    extra_name: str | None = None
    name: str = "condition_extremal_template"
    complete: bool = False
    empty_miss_detail: str = "no witness in enumerated grammar"
    grammar: GrammarSpec = field(
        default_factory=lambda: GrammarSpec(
            sorts=("extremal_template",),
            tokens_by_sort={"extremal_template": ("C4", "C6", "j_template", "k_template")},
            complete=False,
        )
    )

    def __post_init__(self) -> None:
        self._inner = ExtremalSearchFamily()
        self.statement = emit_condition(
            ConditionHypothesis(
                sort="extremal_template",
                tokens=(ConditionToken("extremal_template", "C4"),),
            ),
            parent="extremal graph templates",
            parent_status="already_true",
            obligation="a named template that is connected, bipartite, and has a cycle",
        )

    @property
    def complete_flag(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return 1 if self.host_ok is not None else self._inner.cardinality()

    def origin(self) -> str:
        if self.extra_name is not None:
            return self.extra_name
        return "obs_graph" if self.host_ok is not None else self._inner.origin()

    def neighbors(self, candidate: Candidate) -> Sequence[str]:
        if self.host_ok is not None:
            return ()
        return self._inner.neighbors(candidate)

    def score(self, candidate: Candidate) -> int:
        return 1 if str(candidate) in {"C4", "obs_graph"} else 0

    def check(self, candidate: Candidate) -> ExactCheck | None:
        name = str(candidate)
        if self.host_ok is not None and name == "obs_graph":
            ok = self.host_ok
            return ExactCheck(
                ok=ok,
                payload={
                    "template": "obs_graph",
                    "honesty": {
                        **condition_honesty(discovered=ok),
                        "erdos_146_claim": False,
                        "erdos_180_claim": False,
                    },
                },
            )
        inner = self._inner.check(name)
        if inner is None:
            return None
        payload = dict(inner.payload)
        honesty = dict(payload.get("honesty", {})) if isinstance(payload.get("honesty"), dict) else {}
        honesty.update(condition_honesty(discovered=inner.ok))
        honesty["erdos_146_claim"] = False
        honesty["erdos_180_claim"] = False
        payload["honesty"] = honesty
        return ExactCheck(ok=inner.ok, payload=payload)


def observation_c4() -> Observation:
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    return Observation(graph_n=4, graph_edges=edges, extra=(("template", "C4"),))


def bind_extremal_template(
    observation: Observation | None = None,
) -> ExtremalTemplateConditionFamily | None:
    if observation is None:
        return ExtremalTemplateConditionFamily()
    extra_name = observation.extra_map().get("template")
    n = observation.graph_n
    if n <= 0 and observation.graph_edges:
        n = max((max(edge) for edge in observation.graph_edges), default=-1) + 1
    if extra_name:
        return ExtremalTemplateConditionFamily(extra_name=extra_name)
    if observation.graph_edges and n in {4, 6} and len(observation.graph_edges) == n:
        host = _undirected(observation.graph_edges, n)
        ok = is_connected(host) and is_bipartite(host) and has_cycle(host)
        return ExtremalTemplateConditionFamily(host_ok=ok)
    return None


def _register() -> None:
    register_condition_sort("edge_colouring", bind_edge_colouring)
    register_condition_sort("extremal_template", bind_extremal_template)


_register()


__all__ = [
    "EdgeColouringFamily",
    "ExtremalTemplateConditionFamily",
    "bind_edge_colouring",
    "bind_extremal_template",
    "observation_c4",
    "observation_k5",
]
