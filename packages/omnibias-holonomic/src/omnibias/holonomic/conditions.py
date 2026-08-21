# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Ore-sort condition family: prefix-verified annihilators as hypotheses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

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
from omnibias.holonomic.families import (
    HolonomicDFiniteGuessFamily,
    HolonomicRecurrenceGuessFamily,
    exp_series,
    fibonacci_samples,
)


def _ore_hypothesis(order: int, degree: int) -> ConditionHypothesis:
    return ConditionHypothesis(
        sort="ore",
        tokens=(ConditionToken("ore", "S"), ConditionToken("ore", "n")),
        coefficients=(str(order), str(degree)),
    )


@dataclass
class OreConditionFamily:
    """Wrap :class:`HolonomicRecurrenceGuessFamily` as an Ore hypothesis box."""

    samples: tuple[int | Fraction, ...] = field(default_factory=fibonacci_samples)
    max_order: int = 2
    max_index_degree: int = 1
    name: str = "condition_ore"
    grammar: GrammarSpec = field(
        default_factory=lambda: GrammarSpec(
            sorts=("ore",),
            tokens_by_sort={"ore": ("S", "n")},
            constructors=("raise_ore_degree",),
            complete=False,
        )
    )

    def __post_init__(self) -> None:
        self._inner = HolonomicRecurrenceGuessFamily(
            samples=self.samples,
            max_order=self.max_order,
            max_index_degree=self.max_index_degree,
        )
        self.statement = emit_condition(
            _ore_hypothesis(1, 0),
            parent="P-recursive sequences",
            parent_status="already_true",
            obligation="a prefix-verified P-recurrence in the Ore-degree box",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return self.max_order * (self.max_index_degree + 1)

    def origin(self) -> ConditionHypothesis:
        return _ore_hypothesis(1, 0)

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis) or len(candidate.coefficients) < 2:
            return ()
        order, degree = int(candidate.coefficients[0]), int(candidate.coefficients[1])
        out: list[ConditionHypothesis] = []
        if order > 1:
            out.append(_ore_hypothesis(order - 1, degree))
        if order < self.max_order:
            out.append(_ore_hypothesis(order + 1, degree))
        if degree > 0:
            out.append(_ore_hypothesis(order, degree - 1))
        if degree < self.max_index_degree:
            out.append(_ore_hypothesis(order, degree + 1))
        return out

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis) or len(candidate.coefficients) < 2:
            return 0
        return -(10 * int(candidate.coefficients[0]) + int(candidate.coefficients[1]))

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "ore":
            return None
        if len(candidate.coefficients) < 2:
            return None
        order, degree = int(candidate.coefficients[0]), int(candidate.coefficients[1])
        inner = self._inner.check((order, degree))
        if inner is None:
            return None
        payload = dict(inner.payload)
        payload["hypothesis"] = candidate.as_dict()
        honesty = dict(payload.get("honesty", {})) if isinstance(payload.get("honesty"), dict) else {}
        honesty.update(condition_honesty(discovered=inner.ok))
        payload["honesty"] = honesty
        return ExactCheck(ok=inner.ok, payload=payload)


def observation_fibonacci(n: int = 16) -> Observation:
    return Observation(
        tag="fibonacci",
        sequence=tuple(str(value) for value in fibonacci_samples(n)),
    )


def bind_ore(observation: Observation | None = None) -> OreConditionFamily | None:
    if observation is None:
        return OreConditionFamily()
    if not observation.sequence:
        return None
    samples = tuple(Fraction(item) for item in observation.sequence)
    typed = tuple(int(item) if item.denominator == 1 else item for item in samples)
    return OreConditionFamily(samples=typed)


def _dfinite_hypothesis(order: int, degree: int) -> ConditionHypothesis:
    return ConditionHypothesis(
        sort="dfinite",
        tokens=(ConditionToken("dfinite", "D"), ConditionToken("dfinite", "x")),
        coefficients=(str(order), str(degree)),
    )


@dataclass
class DfiniteConditionFamily:
    """Wrap :class:`HolonomicDFiniteGuessFamily` as a D-finite hypothesis box."""

    series: tuple[int | Fraction, ...] = field(default_factory=exp_series)
    max_order: int = 2
    max_degree: int = 2
    name: str = "condition_dfinite"
    grammar: GrammarSpec = field(
        default_factory=lambda: GrammarSpec(
            sorts=("dfinite",),
            tokens_by_sort={"dfinite": ("D", "x")},
            complete=False,
        )
    )

    def __post_init__(self) -> None:
        self._inner = HolonomicDFiniteGuessFamily(
            series=self.series,
            max_order=self.max_order,
            max_degree=self.max_degree,
        )
        self.statement = emit_condition(
            _dfinite_hypothesis(1, 0),
            parent="D-finite functions",
            parent_status="already_true",
            obligation="a prefix-verified differential annihilator in the degree box",
        )
        self.empty_miss_detail = "no witness in enumerated grammar"

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def cardinality(self) -> int:
        return self.max_order * (self.max_degree + 1)

    def origin(self) -> ConditionHypothesis:
        return _dfinite_hypothesis(1, 0)

    def neighbors(self, candidate: Candidate) -> Sequence[ConditionHypothesis]:
        if not isinstance(candidate, ConditionHypothesis) or len(candidate.coefficients) < 2:
            return ()
        order, degree = int(candidate.coefficients[0]), int(candidate.coefficients[1])
        out: list[ConditionHypothesis] = []
        if order > 1:
            out.append(_dfinite_hypothesis(order - 1, degree))
        if order < self.max_order:
            out.append(_dfinite_hypothesis(order + 1, degree))
        if degree > 0:
            out.append(_dfinite_hypothesis(order, degree - 1))
        if degree < self.max_degree:
            out.append(_dfinite_hypothesis(order, degree + 1))
        return out

    def score(self, candidate: Candidate) -> int:
        if not isinstance(candidate, ConditionHypothesis) or len(candidate.coefficients) < 2:
            return 0
        return -(10 * int(candidate.coefficients[0]) + int(candidate.coefficients[1]))

    def check(self, candidate: Candidate) -> ExactCheck | None:
        if not isinstance(candidate, ConditionHypothesis) or candidate.sort != "dfinite":
            return None
        if len(candidate.coefficients) < 2:
            return None
        order, degree = int(candidate.coefficients[0]), int(candidate.coefficients[1])
        inner = self._inner.check((order, degree))
        if inner is None:
            return None
        payload = dict(inner.payload)
        payload["hypothesis"] = candidate.as_dict()
        honesty = dict(payload.get("honesty", {})) if isinstance(payload.get("honesty"), dict) else {}
        honesty.update(condition_honesty(discovered=inner.ok))
        payload["honesty"] = honesty
        return ExactCheck(ok=inner.ok, payload=payload)


def observation_exp_series(n: int = 14) -> Observation:
    return Observation(
        sequence=tuple(str(value) for value in exp_series(n)),
        extra=(("series", "1"),),
    )


def bind_dfinite(observation: Observation | None = None) -> DfiniteConditionFamily | None:
    if observation is None:
        return DfiniteConditionFamily()
    if not observation.sequence or observation.extra_map().get("series") != "1":
        return None
    samples = tuple(Fraction(item) for item in observation.sequence)
    typed = tuple(int(item) if item.denominator == 1 else item for item in samples)
    return DfiniteConditionFamily(series=typed)


def _register() -> None:
    register_condition_sort("ore", bind_ore)
    register_condition_sort("dfinite", bind_dfinite)


_register()


__all__ = [
    "DfiniteConditionFamily",
    "OreConditionFamily",
    "bind_dfinite",
    "bind_ore",
    "observation_exp_series",
    "observation_fibonacci",
]
