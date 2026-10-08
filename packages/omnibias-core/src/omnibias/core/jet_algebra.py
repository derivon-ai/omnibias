# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite Taylor coefficient layouts with explicit resource budgets.

Mixed layouts contain the downward closure of the requested multi-indices,
not a surrounding total-degree simplex. All stored coefficients are divided
by their multi-index factorial. This module imports no numerical backend.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product
from math import factorial, prod
from operator import index as integer_index
from typing import Any

from omnibias.core.multi_index import get_multi_index_budget

MultiIndex = tuple[int, ...]


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = integer_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


@dataclass(frozen=True)
class JetBudget:
    """Limits checked before coefficient layouts, schedules or tensors grow."""

    max_coefficients: int = field(default_factory=get_multi_index_budget)
    max_multiplication_terms: int = 1_000_000
    max_tensor_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_coefficients", "max_multiplication_terms", "max_tensor_bytes"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))

    def check_layout(self, coefficients: int, terms: int) -> None:
        if coefficients > self.max_coefficients:
            raise ValueError(f"jet needs {coefficients} coefficients; budget is {self.max_coefficients}; "
                             "reduce derivative order or request fewer mixed targets")
        if terms > self.max_multiplication_terms:
            raise ValueError(f"jet needs {terms} multiplication terms; budget is {self.max_multiplication_terms}; "
                             "reduce derivative order or request fewer mixed targets")

    def to_config(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in (
            "max_coefficients", "max_multiplication_terms", "max_tensor_bytes")}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> JetBudget:
        return cls(**config)


def _closure(targets: tuple[MultiIndex, ...], max_work: int) -> Iterable[MultiIndex]:
    # Assign an overlapping divisor to its first target, avoiding a potentially
    # oversized set during the count-only budget pass.
    work = 0
    for position, alpha in enumerate(targets):
        for beta in product(*(range(a + 1) for a in alpha)):
            work += 1
            included = False
            for earlier in targets[:position]:
                work += 1
                if work > max_work:
                    raise ValueError("mixed divisor enumeration exceeds the metadata traversal budget")
                if all(b <= a for b, a in zip(beta, earlier, strict=True)):
                    included = True
                    break
            if work > max_work:
                raise ValueError("mixed divisor enumeration exceeds the metadata traversal budget")
            if not included:
                yield beta


def _frontier(targets: tuple[MultiIndex, ...], budget: JetBudget) -> tuple[MultiIndex, ...]:
    """Remove dominated target boxes with a bounded metadata traversal."""
    maximal: list[MultiIndex] = []
    comparisons = 0
    for alpha in reversed(targets):
        dominated = False
        for beta in maximal:
            comparisons += 1
            if comparisons > budget.max_multiplication_terms:
                raise ValueError("mixed target normalization exceeds the metadata traversal budget")
            if all(a <= b for a, b in zip(alpha, beta, strict=True)):
                dominated = True
                break
        if not dominated:
            maximal.append(alpha)
    return tuple(maximal)


@dataclass(frozen=True, init=False)
class JetSpec:
    """Immutable directional or selected mixed Taylor algebra.

    Construct with :meth:`directional` or :meth:`mixed`. ``pairs[k]`` contains
    every ordered Cauchy-product pair contributing to coefficient ``k``.
    """

    mode: str
    targets: tuple[MultiIndex, ...]
    indices: tuple[MultiIndex, ...]
    budget: JetBudget
    degrees: tuple[int, ...]
    factorials: tuple[int, ...]
    pairs: tuple[tuple[tuple[int, int], ...], ...]

    @classmethod
    def _build(cls, mode: str, targets: tuple[MultiIndex, ...],
               budget: JetBudget) -> JetSpec:
        indices: tuple[MultiIndex, ...]
        if mode == "directional":
            n = targets[0][0]
            budget.check_layout(n + 1, (n + 1) * (n + 2) // 2)
            indices = tuple((k,) for k in range(n + 1))
        else:
            frontier = _frontier(targets, budget)
            candidates = 0
            for alpha in frontier:
                count = prod(a + 1 for a in alpha)
                terms = prod((a + 1) * (a + 2) // 2 for a in alpha)
                budget.check_layout(count, terms)
                candidates += count
                if candidates > budget.max_multiplication_terms:
                    raise ValueError("mixed divisor enumeration exceeds the metadata traversal budget")
            count = terms = 0
            for alpha in _closure(frontier, budget.max_multiplication_terms):
                count += 1
                terms += prod(a + 1 for a in alpha)
                budget.check_layout(count, terms)
            indices = tuple(sorted(_closure(frontier, budget.max_multiplication_terms),
                                   key=lambda a: (sum(a), a)))
        positions = {alpha: i for i, alpha in enumerate(indices)}
        pairs = tuple(tuple((positions[beta], positions[tuple(a - b for a, b in zip(alpha, beta, strict=True))])
                            for beta in product(*(range(a + 1) for a in alpha)))
                      for alpha in indices)
        result = object.__new__(cls)
        for name, value in (
            ("mode", mode), ("targets", targets), ("indices", indices), ("budget", budget),
            ("degrees", tuple(map(sum, indices))),
            ("factorials", tuple(prod(factorial(a) for a in alpha) for alpha in indices)),
            ("pairs", pairs),
        ):
            object.__setattr__(result, name, value)
        return result

    @classmethod
    def directional(cls, order: int, budget: JetBudget | None = None) -> JetSpec:
        return cls._build("directional", ((_integer(order, "order"),),), budget or JetBudget())

    @classmethod
    def mixed(cls, targets: Iterable[Iterable[int]], budget: JetBudget | None = None) -> JetSpec:
        budget = budget or JetBudget()
        rows = []
        for target in targets:
            rows.append(tuple(_integer(a, "multi-index component") for a in target))
            if len(rows) > budget.max_coefficients:
                raise ValueError("requested targets exceed the coefficient budget")
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("targets must be nonempty multi-indices of the same positive dimension")
        canonical = tuple(sorted(set(rows), key=lambda a: (sum(a), a)))
        return cls._build("mixed", canonical, budget)

    @property
    def dim(self) -> int:
        return len(self.indices[0])

    @property
    def order(self) -> int:
        return self.degrees[-1]

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def multiplication_terms(self) -> int:
        return sum(map(len, self.pairs))

    def index(self, alpha: Iterable[int]) -> int:
        key = tuple(_integer(a, "multi-index component") for a in alpha)
        try:
            return self.indices.index(key)
        except ValueError as exc:
            raise ValueError(f"multi-index {key} is outside this jet layout") from exc

    def check_tensor(self, shape: Iterable[int], itemsize: int) -> int:
        """Check one packed jet, given the shape of a *single* coefficient.

        Returns its estimated bytes. This excludes activation auxiliaries,
        backend workspaces and backward tapes; callers must budget those too.
        """
        dimensions = tuple(_integer(d, "tensor dimension") for d in shape)
        needed = self.size * prod(dimensions) * _integer(itemsize, "itemsize", 1)
        if needed > self.budget.max_tensor_bytes:
            raise ValueError(f"jet tensor needs {needed} bytes; budget is {self.budget.max_tensor_bytes}; "
                             "reduce batch size or feature width")
        return needed

    def explain(self) -> dict[str, Any]:
        return {"mode": self.mode, "dim": self.dim, "order": self.order,
                "coefficients": self.size, "multiplication_terms": self.multiplication_terms,
                "coefficient_convention": "D^alpha f / alpha!", "budget": self.budget.to_config()}

    def to_config(self) -> dict[str, Any]:
        return {"version": 1, "mode": self.mode, "targets": [list(a) for a in self.targets],
                "budget": self.budget.to_config()}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> JetSpec:
        if config.get("version", 1) != 1:
            raise ValueError("unsupported jet layout configuration version")
        budget = JetBudget.from_config(config.get("budget", {}))
        targets = config["targets"]
        if config["mode"] == "directional":
            if len(targets) != 1 or len(targets[0]) != 1:
                raise ValueError("directional configuration needs one univariate target")
            return cls.directional(targets[0][0], budget)
        if config["mode"] == "mixed":
            return cls.mixed(targets, budget)
        raise ValueError(f"unknown jet mode {config['mode']!r}")


__all__ = ["JetBudget", "JetSpec"]
