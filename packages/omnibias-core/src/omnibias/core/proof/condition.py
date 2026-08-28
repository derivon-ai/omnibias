# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hypothesis language whose points are conditions, not colourings or maps.

A :class:`ConditionHypothesis` is a typed expression in a finite
:class:`GrammarSpec`. Proposers emit hypotheses. Typed constructors grow the
grammar. The accept gate is an exact ``Q`` / finite checker in the owning
package — this module never imports holonomic / symbolic / combinatorics / pinn.

An incomplete grammar miss is ``search_incomplete``. It is never “no condition
exists” and never a parent theorem. ``GrammarSpec.complete`` defaults ``False``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from omnibias.core.proof.discovery import (
    Candidate,
    ExactCheck,
    FiniteFamily,
    Statement,
    run_discovery,
)
from omnibias.core.proof.observe import (
    ClassHit,
    ClassMemory,
    ClassSelection,
    Observation,
    class_hit_from_check,
    rank_class_hits,
)

ConditionSort = Literal[
    "conservation",
    "dfinite",
    "edge_colouring",
    "extremal_template",
    "forbidden_minor",
    "fractional_order",
    "jet_monomial",
    "ore",
    "pde_operator",
    "piecewise_hybrid",
    "residual_sign",
    "sos_onset",
    "sos_template",
]

ALL_CONDITION_SORTS: tuple[ConditionSort, ...] = (
    "conservation",
    "dfinite",
    "edge_colouring",
    "extremal_template",
    "forbidden_minor",
    "fractional_order",
    "jet_monomial",
    "ore",
    "pde_operator",
    "piecewise_hybrid",
    "residual_sign",
    "sos_onset",
    "sos_template",
)

TYPED_CONSTRUCTORS: frozenset[str] = frozenset(
    {
        "add_minor_edge",
        "compose_jets",
        "multiply_columns",
        "raise_ore_degree",
        "split_partition",
    }
)

ConditionSortFactory = Callable[..., FiniteFamily | None]

_SORT_FACTORIES: dict[ConditionSort, ConditionSortFactory] = {}


def condition_honesty(*, discovered: bool) -> dict[str, bool]:
    """Honesty keys for a condition-language check. Forbidden claims stay False."""

    return {
        "discovered_by_omnibias": discovered,
        "dgg_congestion_theorem_refuted": False,
        "erdos_146_claim": False,
        "erdos_180_claim": False,
        "erdos_183_claim": False,
        "jacobian_conjecture_proof_claim": False,
        "jacobian_n2_claim": False,
        "navier_stokes_proof_claim": False,
        "no_condition_exists_claim": False,
        "ten_proofs_formalization_claim": False,
        "unnamed_condition_complete_claim": False,
    }


@dataclass(frozen=True)
class ConditionToken:
    """One typed atom of a condition (a jet monomial, Ore generator, …)."""

    sort: ConditionSort
    name: str

    def as_dict(self) -> dict[str, str]:
        return {"sort": self.sort, "name": self.name}


@dataclass(frozen=True)
class ConditionHypothesis:
    """A proposed condition. Hashable :class:`~omnibias.core.proof.discovery.Candidate`."""

    sort: ConditionSort
    tokens: tuple[ConditionToken, ...]
    constructors: tuple[str, ...] = ()
    coefficients: tuple[str, ...] = ()
    grammar_id: str = "v1"

    def pretty(self) -> str:
        names = ", ".join(token.name for token in self.tokens)
        return f"{self.sort}[{names}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sort": self.sort,
            "tokens": [token.as_dict() for token in self.tokens],
            "constructors": list(self.constructors),
            "coefficients": list(self.coefficients),
            "grammar_id": self.grammar_id,
            "pretty": self.pretty(),
        }

    def token_names(self) -> tuple[str, ...]:
        return tuple(token.name for token in self.tokens)


@dataclass(frozen=True)
class GrammarSpec:
    """Finite hypothesis class. ``complete`` defaults False (incomplete grammar)."""

    sorts: tuple[ConditionSort, ...]
    tokens_by_sort: Mapping[ConditionSort, tuple[str, ...]]
    constructors: tuple[str, ...] = ()
    max_growth_depth: int = 1
    complete: bool = False
    grammar_id: str = "v1"

    def tokens_for(self, sort: ConditionSort) -> tuple[str, ...]:
        return tuple(self.tokens_by_sort.get(sort, ()))


def emit_condition(
    hypothesis: ConditionHypothesis,
    *,
    parent: str,
    parent_status: Literal["already_false", "already_true", "open"] = "open",
    existential: bool = True,
    obligation: str | None = None,
) -> Statement:
    """Turn a proposed condition into a :class:`Statement` the engine can adjudicate."""

    return Statement(
        name=f"condition_{hypothesis.sort}",
        obligation=obligation or f"the hypothesis {hypothesis.pretty()} holds in its grammar",
        parent=parent,
        parent_status=parent_status,
        existential=existential,
    )


def apply_constructor(
    hypothesis: ConditionHypothesis,
    constructor: str,
    grammar: GrammarSpec,
    *,
    left: str | None = None,
    right: str | None = None,
) -> ConditionHypothesis | None:
    """Rewrite ``hypothesis``. Returns ``None`` if the constructor is illegal."""

    if constructor not in TYPED_CONSTRUCTORS:
        return None
    if constructor not in grammar.constructors:
        return None
    if len(hypothesis.constructors) >= grammar.max_growth_depth:
        return None
    names = hypothesis.token_names()
    if constructor in {"compose_jets", "multiply_columns"}:
        if left is None or right is None:
            return None
        if left not in names or right not in names:
            return None
        product = f"{left}*{right}"
        if product in names:
            return None
        token = ConditionToken(hypothesis.sort, product)
        return ConditionHypothesis(
            sort=hypothesis.sort,
            tokens=hypothesis.tokens + (token,),
            constructors=hypothesis.constructors + (constructor,),
            coefficients=hypothesis.coefficients,
            grammar_id=hypothesis.grammar_id,
        )
    if constructor == "raise_ore_degree":
        if hypothesis.sort != "ore" or not hypothesis.coefficients:
            return None
        order = int(hypothesis.coefficients[0]) + 1
        rest = hypothesis.coefficients[1:]
        return ConditionHypothesis(
            sort="ore",
            tokens=hypothesis.tokens,
            constructors=hypothesis.constructors + ("raise_ore_degree",),
            coefficients=(str(order),) + rest,
            grammar_id=hypothesis.grammar_id,
        )
    if constructor == "split_partition":
        if any("@" in token.name for token in hypothesis.tokens):
            return None
        grown: list[ConditionToken] = []
        for token in hypothesis.tokens:
            grown.append(ConditionToken(token.sort, f"{token.name}@0"))
            grown.append(ConditionToken(token.sort, f"{token.name}@1"))
        return ConditionHypothesis(
            sort=hypothesis.sort,
            tokens=tuple(grown),
            constructors=hypothesis.constructors + ("split_partition",),
            coefficients=hypothesis.coefficients,
            grammar_id=hypothesis.grammar_id,
        )
    if constructor == "add_minor_edge":
        if left is None or right is None or left == right:
            return None
        lo, hi = (left, right) if left < right else (right, left)
        if not _vertex_ok(lo) or not _vertex_ok(hi):
            return None
        edge = f"e:{lo}-{hi}"
        if edge in names:
            return None
        token = ConditionToken("forbidden_minor", edge)
        return ConditionHypothesis(
            sort="forbidden_minor",
            tokens=hypothesis.tokens + (token,),
            constructors=hypothesis.constructors + ("add_minor_edge",),
            coefficients=hypothesis.coefficients,
            grammar_id=hypothesis.grammar_id,
        )
    return None


def _vertex_ok(label: str) -> bool:
    if not label.isdigit():
        return False
    return 0 <= int(label) <= 4


def grow_neighbors(
    hypothesis: ConditionHypothesis,
    grammar: GrammarSpec,
) -> tuple[ConditionHypothesis, ...]:
    """All one-step typed growths of ``hypothesis`` inside ``grammar``."""

    seen: set[ConditionHypothesis] = set()
    out: list[ConditionHypothesis] = []
    names = hypothesis.token_names()
    product_ctor = (
        "compose_jets"
        if "compose_jets" in grammar.constructors
        else "multiply_columns"
        if "multiply_columns" in grammar.constructors
        else None
    )
    if product_ctor is not None:
        for i, left in enumerate(names):
            for right in names[i:]:
                grown = apply_constructor(
                    hypothesis, product_ctor, grammar, left=left, right=right
                )
                if grown is not None and grown not in seen:
                    seen.add(grown)
                    out.append(grown)
    if "raise_ore_degree" in grammar.constructors:
        grown = apply_constructor(hypothesis, "raise_ore_degree", grammar)
        if grown is not None and grown not in seen:
            seen.add(grown)
            out.append(grown)
    if "split_partition" in grammar.constructors:
        grown = apply_constructor(hypothesis, "split_partition", grammar)
        if grown is not None and grown not in seen:
            seen.add(grown)
            out.append(grown)
    if "add_minor_edge" in grammar.constructors:
        verts = [str(i) for i in range(5)]
        for i, left in enumerate(verts):
            for right in verts[i + 1 :]:
                grown = apply_constructor(
                    hypothesis, "add_minor_edge", grammar, left=left, right=right
                )
                if grown is not None and grown not in seen:
                    seen.add(grown)
                    out.append(grown)
    return tuple(out)


def register_condition_sort(sort: ConditionSort, factory: ConditionSortFactory) -> None:
    """Register a package factory that builds a :class:`FiniteFamily` for ``sort``."""

    existing = _SORT_FACTORIES.get(sort)
    if existing is not None and existing is not factory:
        raise ValueError(f"condition sort collision: {sort!r} already registered")
    _SORT_FACTORIES[sort] = factory


def list_condition_sorts() -> tuple[ConditionSort, ...]:
    return tuple(sorted(_SORT_FACTORIES, key=str))


def condition_sort_factory(sort: ConditionSort) -> ConditionSortFactory | None:
    return _SORT_FACTORIES.get(sort)


def _invoke_factory(
    factory: ConditionSortFactory,
    observation: Observation | None,
) -> FiniteFamily | None:
    """Call ``factory(obs)``; on ``TypeError`` fall back to a zero-arg class."""

    try:
        result = factory(observation)
    except TypeError:
        try:
            result = factory()
        except TypeError:
            return None
    return result


def bind_sorts(
    observation: Observation | None,
    sorts: Sequence[ConditionSort] | None = None,
) -> dict[ConditionSort, FiniteFamily]:
    """Apply registered factories. ``None`` means the sort does not apply."""

    wanted: tuple[ConditionSort, ...]
    if sorts is not None:
        wanted = tuple(sorts)
    else:
        registered = list_condition_sorts()
        wanted = registered if registered else ALL_CONDITION_SORTS
    out: dict[ConditionSort, FiniteFamily] = {}
    for sort in wanted:
        factory = _SORT_FACTORIES.get(sort)
        if factory is None:
            continue
        family = _invoke_factory(factory, observation)
        if family is not None:
            out[sort] = family
    return out


def _reset_condition_sorts_for_tests() -> None:
    """Test helper. Not part of the public engine."""

    _SORT_FACTORIES.clear()


@dataclass
class KindMetaFamily:
    """Candidates are :data:`ConditionSort` tags. Missing sorts force incomplete grammar."""

    sorts: tuple[ConditionSort, ...]
    families: Mapping[ConditionSort, FiniteFamily] | None = None
    observation: Observation | None = None
    sort_scores: Mapping[str, int] | None = None
    grammar_complete: bool = False
    budget: int = 8
    name: str = "condition_kind_meta"
    empty_miss_detail: str = "no witness in enumerated grammar"
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="condition_kind_meta",
            obligation="some registered condition sort has an exact witness in its box",
            parent="condition language",
            parent_status="open",
        )
    )
    complete: bool = field(init=False)
    _resolved: dict[ConditionSort, FiniteFamily] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        resolved: dict[ConditionSort, FiniteFamily] = {}
        if self.families is not None:
            for sort, family in self.families.items():
                resolved[sort] = family
        else:
            resolved.update(bind_sorts(self.observation, self.sorts))
        missing = tuple(sort for sort in self.sorts if sort not in resolved)
        grammar_complete = bool(self.grammar_complete) and not missing and bool(self.sorts)
        object.__setattr__(self, "grammar_complete", grammar_complete)
        object.__setattr__(self, "complete", grammar_complete)
        object.__setattr__(self, "_resolved", resolved)

    def cardinality(self) -> int:
        return len(self.sorts)

    def origin(self) -> ConditionSort:
        if not self.sorts:
            raise ValueError("KindMetaFamily needs at least one sort")
        return self.sorts[0]

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionSort]:
        if not isinstance(candidate, str) or candidate not in self.sorts:
            return ()
        index = self.sorts.index(candidate)
        out: list[ConditionSort] = []
        if index > 0:
            out.append(self.sorts[index - 1])
        if index + 1 < len(self.sorts):
            out.append(self.sorts[index + 1])
        return out

    def score(self, candidate: Candidate) -> int:
        if self.sort_scores is None:
            return 0
        return int(self.sort_scores.get(str(candidate), 0))

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, str) or candidate not in self.sorts:
            return None
        sort = cast(ConditionSort, candidate)
        family = self._resolved.get(sort)
        if family is None:
            return ExactCheck(
                ok=False,
                payload={
                    "sort": candidate,
                    "missing": True,
                    "honesty": condition_honesty(discovered=False),
                    "grammar_complete": self.grammar_complete,
                },
            )
        result = run_discovery(
            family.statement,
            family,
            "score_guided",
            budget=self.budget,
        )
        ok = result.status == "PROVED"
        return ExactCheck(
            ok=ok,
            payload={
                "sort": candidate,
                "sub": result.as_dict(),
                "grammar_complete": self.grammar_complete,
                "honesty": condition_honesty(discovered=ok),
                "no_condition_exists_claim": False,
            },
        )


@dataclass
class GrammarGrowthFamily:
    """Walk constructor-grown hypotheses. ``checker`` is the exact accept gate."""

    grammar: GrammarSpec
    seed: ConditionHypothesis
    checker: Callable[[ConditionHypothesis], ExactCheck | None]
    name: str = "condition_grammar_growth"
    empty_miss_detail: str = "no witness in enumerated grammar"
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="condition_grammar_growth",
            obligation="a grown hypothesis in this grammar has an exact witness",
            parent="condition language",
            parent_status="open",
        )
    )

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int | None:
        return None

    def origin(self) -> ConditionHypothesis:
        return self.seed

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis):
            return ()
        return grow_neighbors(candidate, self.grammar)

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis):
            return 0
        return -len(candidate.tokens)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis):
            return None
        return self.checker(candidate)


def _grow_miss_hits(
    resolved: Mapping[ConditionSort, FiniteFamily],
    *,
    max_growth_depth: int,
) -> list[ClassHit]:
    """One constructor step (or product columns) after an exact miss."""

    hits: list[ClassHit] = []
    if max_growth_depth < 1:
        return hits
    for sort, family in resolved.items():
        grown_fn = getattr(family, "grown_product_check", None)
        if callable(grown_fn):
            checked = grown_fn()
            if isinstance(checked, ExactCheck) and checked.ok:
                hits.append(class_hit_from_check(str(sort), checked))
                continue
        grammar = getattr(family, "grammar", None)
        if grammar is None or not getattr(grammar, "constructors", ()):
            continue
        origin = family.origin()
        if not isinstance(origin, ConditionHypothesis):
            continue
        for grown in grow_neighbors(origin, grammar):
            checked = family.check(grown)
            if isinstance(checked, ExactCheck) and checked.ok:
                hits.append(class_hit_from_check(str(sort), checked))
                break
    return hits


def select_class(
    observation: Observation,
    *,
    collect: bool = True,
    memory: ClassMemory | None = None,
    gate: Any | None = None,
    grammar_complete: bool = False,
    sorts: Sequence[ConditionSort] | None = None,
    budget: int | None = None,
    inner_budget: int | None = None,
    grow: bool = True,
    max_growth_depth: int = 1,
) -> ClassSelection:
    """Bind, optionally reorder, collect certified hits, and rank.

    ``grammar_complete`` is accepted and ignored — this driver never marks a
    grammar complete. Soft RMSE never writes :class:`ExactCheck`. On an exact
    miss, ``grow=True`` applies one legal constructor step (still incomplete).
    """

    _ = grammar_complete
    if sorts is not None:
        requested = tuple(sorts)
    else:
        registered = list_condition_sorts()
        requested = registered if registered else ALL_CONDITION_SORTS
    if gate is not None:
        proposed = tuple(str(item) for item in gate.propose(observation))
        order = [sort for sort in proposed if sort in requested]
        for sort in requested:
            if sort not in order:
                order.append(sort)
        ordered = cast(tuple[ConditionSort, ...], tuple(order))
        sort_scores = {str(sort): len(ordered) - index for index, sort in enumerate(ordered)}
    else:
        ordered = requested
        sort_scores = None
    n_sorts = len(ordered)
    if budget is None:
        outer = n_sorts
    else:
        outer = budget
    inner = 32 if inner_budget is None else inner_budget
    meta = KindMetaFamily(
        sorts=ordered,
        observation=observation,
        sort_scores=sort_scores,
        grammar_complete=False,
        budget=inner,
    )
    result = run_discovery(
        meta.statement,
        meta,
        "score_guided",
        budget=outer,
        collect=collect,
    )
    hits: list[ClassHit] = []
    for candidate in result.solutions:
        checked: ExactCheck | None
        if candidate == result.candidate and result.check is not None:
            checked = result.check
        else:
            checked = meta.check(candidate)
        if checked is None or not checked.ok:
            continue
        unique = False
        if result.characterization is not None and len(result.solutions) == 1:
            unique = result.characterization.unique_in_family
        hits.append(class_hit_from_check(str(candidate), checked, unique_in_family=unique))
    if grow and outer > 0 and not any(hit.tier == "exact" for hit in hits):
        hits.extend(
            _grow_miss_hits(meta._resolved, max_growth_depth=max_growth_depth)
        )
    ranked = rank_class_hits(hits)
    best = ranked[0] if ranked else None
    if memory is not None and best is not None and best.tier == "exact":
        memory.record(observation, best.sort)
    discovered = best is not None and best.tier in {"exact", "enclosure"}
    return ClassSelection(
        observation=observation,
        hits=tuple(ranked),
        best=best,
        honesty=condition_honesty(discovered=discovered),
        search_incomplete=result.search_incomplete or not meta.complete,
        grammar_complete=False,
        result=result,
    )


__all__ = [
    "ALL_CONDITION_SORTS",
    "ConditionHypothesis",
    "ConditionSort",
    "ConditionSortFactory",
    "ConditionToken",
    "GrammarGrowthFamily",
    "GrammarSpec",
    "KindMetaFamily",
    "TYPED_CONSTRUCTORS",
    "apply_constructor",
    "bind_sorts",
    "condition_honesty",
    "condition_sort_factory",
    "emit_condition",
    "grow_neighbors",
    "list_condition_sorts",
    "register_condition_sort",
    "select_class",
]
