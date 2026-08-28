# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite discovery loop: statement → family → proposer → exact checker → verdict.

This is **not** a second prove/disprove engine. :class:`~omnibias.core.proof.ProofMachine`
still adjudicates. This module is the missing *proposer* layer over a named finite
family. The checker is exact; a proposer only emits candidates.

A hit certifies the **finite obligation** in that family. A miss on an incomplete
family is ``BLOCKED`` (``search_incomplete``), never a proof of the parent.
Exhaustive miss is claimed only when the proposer iterator ended *and* (if the
family reports ``cardinality``) every member was evaluated. ``budget == 0`` is
always ``search_incomplete``.

``Statement.existential`` selects the verdict on a hit or an exhausted miss:

* existential + hit → ``PROVED`` (witness)
* existential + exhausted complete miss → ``BLOCKED`` (empty solution set in the box)
* universal + hit → ``DISPROVED`` (counterexample)
* universal + exhausted complete miss → ``PROVED`` of that finite universal

``coordinate_newton`` is a discrete finite-difference walk, not
``CubicNewton``. ``onehot_anneal`` is a 1-flip anneal on a discrete box, not
``omnibias-qubo``.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal, Protocol, cast

ParentStatus = Literal["already_false", "already_true", "open"]
DiscoveryStatus = Literal["PROVED", "DISPROVED", "BLOCKED"]
ProposerName = Literal["score_guided", "coordinate_newton", "onehot_anneal"]
EquationKind = Literal[
    "recurrence",
    "ore",
    "polynomial_identity",
    "pde_span",
    "map_witness",
    "graph_witness",
    "conservation",
    "sos_template",
    "forbidden_minor",
]

Candidate = Hashable

_BOX_SCOPED_NOTE = "uniqueness is span/box-scoped; not a parent theorem"


@dataclass(frozen=True)
class Statement:
    """A finite obligation, never an implicit parent claim."""

    name: str
    obligation: str
    parent: str
    parent_status: ParentStatus
    existential: bool = True


@dataclass(frozen=True)
class ExactCheck:
    """Result of an exact family checker. ``payload`` must carry ``honesty``."""

    ok: bool
    payload: dict[str, Any]


@dataclass(frozen=True)
class DiscoveredEquation:
    """An equation-valued witness. Coefficients are JSON-able fraction strings."""

    kind: EquationKind
    pretty: str
    coefficients: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pretty": self.pretty,
            "coefficients": list(self.coefficients),
        }


@dataclass(frozen=True)
class Characterization:
    """Box-scoped uniqueness. Never a parent-theorem claim."""

    unique_in_family: bool
    family_complete: bool
    exhausted: bool
    solution_count: int | None = None
    annihilator: str | None = None
    note: str = _BOX_SCOPED_NOTE

    def as_dict(self) -> dict[str, Any]:
        return {
            "unique_in_family": self.unique_in_family,
            "family_complete": self.family_complete,
            "exhausted": self.exhausted,
            "solution_count": self.solution_count,
            "annihilator": self.annihilator,
            "note": self.note,
        }


class FiniteFamily(Protocol):
    """A named, bounded search space with an exact witness predicate."""

    name: str
    complete: bool
    statement: Statement

    def origin(self) -> Candidate:
        """A seed candidate (need not be a witness)."""
        ...

    def neighbors(self, candidate: Candidate) -> Sequence[Candidate]:
        """Adjacent candidates in the discrete family."""
        ...

    def score(self, candidate: Candidate) -> Fraction | int:
        """Heuristic only. The accept gate is :meth:`check`."""
        ...

    def check(self, candidate: Candidate) -> ExactCheck | None:
        """Exact witness test, or ``None`` if the candidate is ill-formed."""
        ...

    def cardinality(self) -> int | None:
        """Known finite size, or ``None`` if unknown. Optional on implementations."""
        ...


class Proposer(Protocol):
    """Emits at most ``budget`` candidates from a family."""

    name: str

    def propose(self, family: FiniteFamily, budget: int) -> Iterator[Candidate]: ...


@dataclass(frozen=True)
class DiscoveryResult:
    """Outcome of :func:`run_discovery` (finite obligation only)."""

    status: DiscoveryStatus
    statement: Statement
    family: str
    proposer: str
    budget: int
    evaluated: int
    candidate: Candidate | None
    check: ExactCheck | None
    detail: str
    search_incomplete: bool = False
    characterization: Characterization | None = None
    equation: DiscoveredEquation | None = None
    solutions: tuple[Candidate, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.check.payload) if self.check is not None else {}
        honesty = payload.get("honesty", {})
        equation = self.equation.as_dict() if self.equation is not None else None
        characterization = (
            self.characterization.as_dict() if self.characterization is not None else None
        )
        return {
            "kind": "finite_discovery",
            "status": self.status,
            "statement": self.statement.name,
            "obligation": self.statement.obligation,
            "parent": self.statement.parent,
            "parent_status": self.statement.parent_status,
            "existential": self.statement.existential,
            "family": self.family,
            "proposer": self.proposer,
            "budget": self.budget,
            "evaluated": self.evaluated,
            "candidate": _candidate_json(self.candidate),
            "solutions": [_candidate_json(item) for item in self.solutions],
            "search_incomplete": self.search_incomplete,
            "replay_ok": self.status in ("PROVED", "DISPROVED"),
            "detail": self.detail,
            "honesty": dict(honesty) if isinstance(honesty, Mapping) else {},
            "payload": payload,
            "equation": equation,
            "characterization": characterization,
        }


def _candidate_json(candidate: Candidate | None) -> Any:
    if candidate is None:
        return None
    if isinstance(candidate, tuple):
        return [_candidate_json(part) for part in candidate]
    if isinstance(candidate, int | str | bool):
        return candidate
    if isinstance(candidate, Fraction):
        return str(candidate)
    as_dict = getattr(candidate, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    return str(candidate)


def _as_int_tuple(candidate: Candidate) -> tuple[int, ...] | None:
    if isinstance(candidate, int):
        return (candidate,)
    if isinstance(candidate, tuple) and all(isinstance(x, int) for x in candidate):
        return cast(tuple[int, ...], candidate)
    return None


def _score(family: FiniteFamily, candidate: Candidate) -> Fraction:
    value = family.score(candidate)
    return value if isinstance(value, Fraction) else Fraction(value)


class ScoreGuidedWalk:
    """Walk neighbors in nonincreasing score order (CI default proposer)."""

    name = "score_guided"

    def propose(self, family: FiniteFamily, budget: int) -> Iterator[Candidate]:
        if budget <= 0:
            return
        start = family.origin()
        seen: set[Candidate] = {start}
        queue: list[tuple[Fraction, int, Candidate]] = [(-_score(family, start), 0, start)]
        tie = 1
        yielded = 0
        while queue and yielded < budget:
            queue.sort(key=lambda item: (item[0], item[1]))
            _neg, _t, current = queue.pop(0)
            yield current
            yielded += 1
            if yielded >= budget:
                return
            ranked = sorted(
                family.neighbors(current),
                key=lambda cand: _score(family, cand),
                reverse=True,
            )
            for nbr in ranked:
                if nbr in seen:
                    continue
                seen.add(nbr)
                queue.append((-_score(family, nbr), tie, nbr))
                tie += 1


class CoordinateNewton:
    """Finite-difference step on integer coordinates, then snap.

    Not :class:`omnibias.torch.optim.CubicNewton` and not a closed-form Hessian.
    """

    name = "coordinate_newton"

    def propose(self, family: FiniteFamily, budget: int) -> Iterator[Candidate]:
        if budget <= 0:
            return
        current = family.origin()
        seen: set[Candidate] = set()
        yielded = 0
        while yielded < budget:
            if current not in seen:
                seen.add(current)
                yield current
                yielded += 1
                if yielded >= budget:
                    return
            nxt = _newton_step(family, current)
            if nxt is None or nxt == current or nxt in seen:
                nxt = _best_unseen_neighbor(family, current, seen)
            if nxt is None:
                return
            current = nxt


def _newton_step(family: FiniteFamily, current: Candidate) -> Candidate | None:
    coords = _as_int_tuple(current)
    if coords is None:
        return _best_unseen_neighbor(family, current, set())
    stepped: list[int] = []
    for i, value in enumerate(coords):
        plus = list(coords)
        minus = list(coords)
        plus[i] = value + 1
        minus[i] = value - 1
        plus_t, minus_t = tuple(plus), tuple(minus)
        s_plus = _score(family, plus_t if len(coords) > 1 or not isinstance(current, int) else plus[0])
        s_minus = _score(family, minus_t if len(coords) > 1 or not isinstance(current, int) else minus[0])
        # Families use int tuples. Rebuild as the same shape as ``current``.
        if isinstance(current, int):
            s_plus = _score(family, plus[0])
            s_minus = _score(family, minus[0])
        grad = (s_plus - s_minus) / 2
        if grad > 0:
            stepped.append(value + 1)
        elif grad < 0:
            stepped.append(value - 1)
        else:
            stepped.append(value)
    if isinstance(current, int):
        return stepped[0]
    return tuple(stepped)


def _best_unseen_neighbor(
    family: FiniteFamily,
    current: Candidate,
    seen: set[Candidate],
) -> Candidate | None:
    best: Candidate | None = None
    best_score: Fraction | None = None
    for nbr in family.neighbors(current):
        if nbr in seen:
            continue
        value = _score(family, nbr)
        if best_score is None or value > best_score:
            best, best_score = nbr, value
    return best


class OneHotAnneal:
    """1-flip Metropolis walk on the discrete box (QUBO-shaped; no ``omnibias-qubo``)."""

    name = "onehot_anneal"

    def __init__(self, *, seed: int = 0, temperature: Fraction | int = 1) -> None:
        self.seed = seed
        self.temperature = (
            temperature if isinstance(temperature, Fraction) else Fraction(temperature)
        )

    def propose(self, family: FiniteFamily, budget: int) -> Iterator[Candidate]:
        if budget <= 0:
            return
        rng = _LCG(self.seed)
        current = family.origin()
        seen: set[Candidate] = set()
        yielded = 0
        temp = self.temperature if self.temperature > 0 else Fraction(1)
        while yielded < budget:
            if current not in seen:
                seen.add(current)
                yield current
                yielded += 1
                if yielded >= budget:
                    return
            nbrs = [n for n in family.neighbors(current) if n not in seen] or list(
                family.neighbors(current)
            )
            if not nbrs:
                return
            nxt = nbrs[rng.next() % len(nbrs)]
            delta = _score(family, nxt) - _score(family, current)
            # Metropolis on energy = -score: accept if score improves or by chance.
            if delta >= 0 or _accept(rng, delta, temp):
                current = nxt
            elif nxt not in seen:
                current = nxt


class _LCG:
    """Tiny integer LCG so the core stays numpy-free."""

    def __init__(self, seed: int) -> None:
        self.state = seed % 2147483647

    def next(self) -> int:
        self.state = (1103515245 * self.state + 12345) % 2147483647
        return self.state


def _accept(rng: _LCG, delta: Fraction, temp: Fraction) -> bool:
    # Accept with probability ~ 1 / (1 + 2 * |delta| / temp), no math.exp.
    weight = temp / (temp + 2 * abs(delta))
    draw = Fraction(rng.next() % 1000, 1000)
    return draw < weight


def get_proposer(name: ProposerName | str, **kwargs: Any) -> Proposer:
    if name == "score_guided":
        return ScoreGuidedWalk()
    if name == "coordinate_newton":
        return CoordinateNewton()
    if name == "onehot_anneal":
        return OneHotAnneal(
            seed=int(kwargs.get("seed", 0)),
            temperature=kwargs.get("temperature", 1),
        )
    raise ValueError(f"unknown proposer {name!r}")


def _family_cardinality(family: FiniteFamily) -> int | None:
    getter = getattr(family, "cardinality", None)
    if getter is None:
        return None
    value = getter() if callable(getter) else getter
    return None if value is None else int(value)


def _proposer_exhausted(
    *,
    budget: int,
    evaluated: int,
    cardinality: int | None,
    stopped_for_hit: bool,
) -> bool:
    if budget <= 0 or stopped_for_hit:
        return False
    if cardinality is not None:
        return evaluated >= cardinality
    return 0 < evaluated < budget


def _equation_from_check(checked: ExactCheck | None) -> DiscoveredEquation | None:
    if checked is None:
        return None
    raw = checked.payload.get("equation")
    if isinstance(raw, DiscoveredEquation):
        return raw
    if not isinstance(raw, Mapping):
        return None
    kind = raw.get("kind")
    pretty = raw.get("pretty")
    if not isinstance(kind, str) or not isinstance(pretty, str):
        return None
    coeffs = raw.get("coefficients", ())
    if isinstance(coeffs, list | tuple):
        coeff_tuple = tuple(str(item) for item in coeffs)
    else:
        coeff_tuple = ()
    if kind not in (
        "recurrence",
        "ore",
        "polynomial_identity",
        "pde_span",
        "map_witness",
        "graph_witness",
        "conservation",
        "sos_template",
        "forbidden_minor",
    ):
        return None
    return DiscoveredEquation(
        kind=cast(EquationKind, kind),
        pretty=pretty,
        coefficients=coeff_tuple,
    )


def _annihilator_from_check(checked: ExactCheck | None) -> str | None:
    if checked is None:
        return None
    raw = checked.payload.get("annihilator")
    if isinstance(raw, str):
        return raw
    equation = _equation_from_check(checked)
    return None if equation is None else equation.pretty


def run_discovery(
    statement: Statement,
    family: FiniteFamily,
    proposer: Proposer | ProposerName | str = "score_guided",
    *,
    budget: int = 64,
    collect: bool = False,
) -> DiscoveryResult:
    """Propose up to ``budget`` candidates; accept the first exact hit unless ``collect``."""
    walker = get_proposer(proposer) if isinstance(proposer, str) else proposer
    evaluated = 0
    last: Candidate | None = None
    last_check: ExactCheck | None = None
    hits: list[tuple[Candidate, ExactCheck]] = []
    if budget < 0:
        raise ValueError("budget must be >= 0")
    for candidate in walker.propose(family, budget):
        evaluated += 1
        last = candidate
        checked = family.check(candidate)
        last_check = checked
        if checked is not None and checked.ok:
            hits.append((candidate, checked))
            if not collect:
                break
    stopped_for_hit = bool(hits) and not collect
    cardinality = _family_cardinality(family)
    exhausted = _proposer_exhausted(
        budget=budget,
        evaluated=evaluated,
        cardinality=cardinality,
        stopped_for_hit=stopped_for_hit,
    )
    truly_exhausted = exhausted and family.complete
    first_check = hits[0][1] if hits else last_check
    equation = _equation_from_check(hits[0][1] if hits else None)
    annihilator = _annihilator_from_check(hits[0][1] if hits else None)
    solutions = tuple(item[0] for item in hits)
    if hits:
        unique = truly_exhausted and len(hits) == 1
        characterization = Characterization(
            unique_in_family=unique,
            family_complete=family.complete,
            exhausted=truly_exhausted,
            solution_count=len(hits) if collect or truly_exhausted else None,
            annihilator=annihilator,
        )
        if statement.existential:
            return DiscoveryResult(
                status="PROVED",
                statement=statement,
                family=family.name,
                proposer=walker.name,
                budget=budget,
                evaluated=evaluated,
                candidate=hits[0][0],
                check=hits[0][1],
                detail="exact witness in family",
                search_incomplete=False,
                characterization=characterization,
                equation=equation,
                solutions=solutions,
            )
        return DiscoveryResult(
            status="DISPROVED",
            statement=statement,
            family=family.name,
            proposer=walker.name,
            budget=budget,
            evaluated=evaluated,
            candidate=hits[0][0],
            check=hits[0][1],
            detail="counterexample to universal obligation",
            search_incomplete=False,
            characterization=characterization,
            equation=equation,
            solutions=solutions,
        )
    if not truly_exhausted:
        characterization = Characterization(
            unique_in_family=False,
            family_complete=family.complete,
            exhausted=False,
            solution_count=None,
            annihilator=None,
        )
        return DiscoveryResult(
            status="BLOCKED",
            statement=statement,
            family=family.name,
            proposer=walker.name,
            budget=budget,
            evaluated=evaluated,
            candidate=last,
            check=first_check if first_check is not None and not first_check.ok else None,
            detail="search_incomplete",
            search_incomplete=True,
            characterization=characterization,
            equation=None,
            solutions=(),
        )
    characterization = Characterization(
        unique_in_family=False,
        family_complete=True,
        exhausted=True,
        solution_count=0,
        annihilator=None,
    )
    if statement.existential:
        miss_detail = getattr(family, "empty_miss_detail", None)
        if not isinstance(miss_detail, str) or not miss_detail:
            miss_detail = "no witness in enumerated family"
        return DiscoveryResult(
            status="BLOCKED",
            statement=statement,
            family=family.name,
            proposer=walker.name,
            budget=budget,
            evaluated=evaluated,
            candidate=last,
            check=None,
            detail=miss_detail,
            search_incomplete=False,
            characterization=characterization,
            equation=None,
            solutions=(),
        )
    return DiscoveryResult(
        status="PROVED",
        statement=statement,
        family=family.name,
        proposer=walker.name,
        budget=budget,
        evaluated=evaluated,
        candidate=last,
        check=None,
        detail="no counterexample in complete family",
        search_incomplete=False,
        characterization=characterization,
        equation=None,
        solutions=(),
    )


@dataclass
class IntegerIntervalFamily:
    """Toy 1-D family used by core tests and the discovery cookbook."""

    lo: int
    hi: int
    target_square: int = 4
    name: str = "integer_square"
    complete: bool = True
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="exists_square",
            obligation="some integer x in the interval with x^2 equal to the target",
            parent="toy",
            parent_status="already_true",
        )
    )

    def cardinality(self) -> int:
        return max(0, self.hi - self.lo + 1)

    def origin(self) -> int:
        return 0 if self.lo <= 0 <= self.hi else self.lo

    def neighbors(self, candidate: Candidate) -> Sequence[int]:
        if not isinstance(candidate, int):
            return ()
        out: list[int] = []
        if candidate - 1 >= self.lo:
            out.append(candidate - 1)
        if candidate + 1 <= self.hi:
            out.append(candidate + 1)
        return out

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, int):
            return 0
        return -abs(candidate * candidate - self.target_square)

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, int):
            return None
        value = candidate
        if not self.lo <= value <= self.hi:
            return None
        ok = value * value == self.target_square
        return ExactCheck(
            ok=ok,
            payload={
                "x": value,
                "honesty": {
                    "discovered_by_omnibias": ok,
                    "jacobian_conjecture_proof_claim": False,
                    "navier_stokes_proof_claim": False,
                },
            },
        )


__all__ = [
    "Candidate",
    "Characterization",
    "CoordinateNewton",
    "DiscoveredEquation",
    "DiscoveryResult",
    "DiscoveryStatus",
    "EquationKind",
    "ExactCheck",
    "FiniteFamily",
    "IntegerIntervalFamily",
    "OneHotAnneal",
    "ParentStatus",
    "Proposer",
    "ProposerName",
    "ScoreGuidedWalk",
    "Statement",
    "get_proposer",
    "run_discovery",
]
