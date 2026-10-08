# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Derivative requests that do not eagerly allocate a surrounding Taylor jet.

An operator is a raw derivative readout, ``sum(c_alpha * D**alpha)``.  Its
coefficients are exact host metadata.  In particular constructing a Laplacian
request does not enumerate its coordinate terms or their divisor closure.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from math import comb, factorial, prod
from operator import index as integer_index
from typing import Any

from omnibias.core.jet_algebra import JetBudget, JetSpec

Index = tuple[int, ...]
Terms = tuple[tuple[Index, Fraction], ...]


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = integer_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


@dataclass(frozen=True)
class OperatorPlanningBudget:
    """Bounds on graph inspection and exact symbolic metadata, not tensor memory."""

    max_graph_nodes: int = 10_000
    max_planning_work: int = 1_000_000
    max_symbol_terms: int = 10_000
    max_order: int = 128
    max_coefficient_bits: int = 4096

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))

    def check_graph(self, nodes: int, work: int = 0) -> None:
        nodes = _integer(nodes, "graph nodes")
        work = _integer(work, "planning work")
        if nodes > self.max_graph_nodes:
            raise ValueError(f"operator graph has {nodes} nodes; budget is {self.max_graph_nodes}; "
                             "compile a smaller region")
        if work > self.max_planning_work:
            raise ValueError(f"operator planning needs {work} steps; budget is {self.max_planning_work}; "
                             "request a narrower operator or compile a smaller region")

    def to_config(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OperatorPlanningBudget:
        return cls(**dict(config))


@dataclass(frozen=True, init=False)
class OperatorSpec:
    """Immutable derivative query, independent of a chosen evaluation strategy.

    ``mixed`` requests one multi-index.  ``polynomial`` requests one exact
    linear combination of mixed derivatives.  ``directional`` uses a runtime
    direction and resolves to a univariate coordinate operator; its ``order``
    is unrelated to the number of ambient input coordinates.
    """

    mode: str
    order: int
    alpha: Index | None
    power: int | None
    terms: Terms
    dim: int | None
    budget: JetBudget
    planning_budget: OperatorPlanningBudget

    @classmethod
    def _build(cls, mode: str, order: int, alpha: Index | None, power: int | None,
               terms: Terms, dimension: int | None, budget: JetBudget | None,
               planning_budget: OperatorPlanningBudget | None) -> OperatorSpec:
        budget = budget or JetBudget()
        planning_budget = planning_budget or OperatorPlanningBudget()
        if order > planning_budget.max_order:
            raise ValueError(f"operator order {order} exceeds planning budget {planning_budget.max_order}")
        # A compact request may be feasible even when its complete mixed jet
        # is not. Only the smallest univariate state is checked at this point.
        budget.check_layout(order + 1, 0)
        result = object.__new__(cls)
        for name, value in (("mode", mode), ("order", order), ("alpha", alpha),
                            ("power", power), ("terms", terms), ("dim", dimension),
                            ("budget", budget), ("planning_budget", planning_budget)):
            object.__setattr__(result, name, value)
        return result

    @classmethod
    def directional(cls, order: int, budget: JetBudget | None = None, *,
                    planning_budget: OperatorPlanningBudget | None = None) -> OperatorSpec:
        return cls._build("directional", _integer(order, "order"), None, None, (), None,
                          budget, planning_budget)

    @classmethod
    def mixed(cls, alpha: Iterable[int], budget: JetBudget | None = None, *,
              planning_budget: OperatorPlanningBudget | None = None) -> OperatorSpec:
        limits = planning_budget or OperatorPlanningBudget()
        values: list[int] = []
        for value in alpha:
            values.append(_integer(value, "multi-index component"))
            if len(values) > limits.max_planning_work:
                raise ValueError("multi-index dimension exceeds the operator planning budget")
        if not values:
            raise ValueError("a mixed derivative needs a nonempty multi-index")
        index = tuple(values)
        return cls._build("mixed", sum(index), index, None, (), len(index), budget, limits)

    @classmethod
    def laplacian(cls, power: int = 1, budget: JetBudget | None = None, *,
                  planning_budget: OperatorPlanningBudget | None = None) -> OperatorSpec:
        power = _integer(power, "Laplacian power")
        return cls._build("laplacian", 2 * power, None, power, (), None, budget, planning_budget)

    @classmethod
    def laplacian_power(cls, power: int = 1, budget: JetBudget | None = None, *,
                        planning_budget: OperatorPlanningBudget | None = None) -> OperatorSpec:
        """Alias for :meth:`laplacian`. Power zero requests the primal."""
        return cls.laplacian(power, budget, planning_budget=planning_budget)

    @classmethod
    def polynomial(cls, terms: Mapping[Index, int | Fraction], budget: JetBudget | None = None, *,
                   planning_budget: OperatorPlanningBudget | None = None) -> OperatorSpec:
        limits = planning_budget or OperatorPlanningBudget()
        if not terms:
            raise ValueError("an operator polynomial needs at least one nonzero term")
        if len(terms) > limits.max_symbol_terms:
            raise ValueError("operator polynomial exceeds the symbol term budget")
        dimension: int | None = None
        canonical: list[tuple[Index, Fraction]] = []
        work = 0
        for raw_index, coefficient in terms.items():
            index = tuple(_integer(a, "multi-index component") for a in raw_index)
            if not index or (dimension is not None and len(index) != dimension):
                raise ValueError("operator multi-indices must have the same positive dimension")
            dimension = len(index)
            work += dimension
            limits.check_graph(0, work)
            if isinstance(coefficient, bool) or not isinstance(coefficient, (int, Fraction)):
                raise TypeError("operator coefficients must be int or Fraction")
            exact = Fraction(coefficient)
            if max(exact.numerator.bit_length(), exact.denominator.bit_length()) > limits.max_coefficient_bits:
                raise ValueError("operator coefficient exceeds the exact bit budget")
            if exact:
                canonical.append((index, exact))
        if not canonical:
            raise ValueError("an operator polynomial needs at least one nonzero term")
        canonical.sort(key=lambda item: (sum(item[0]), item[0]))
        return cls._build("polynomial", max(sum(a) for a, _ in canonical), None, None,
                          tuple(canonical), dimension, budget, limits)

    def check_dimension(self, dimension: int) -> int:
        """Validate an ambient coordinate dimension without enumerating coefficients."""
        dimension = _integer(dimension, "coordinate dimension", 1)
        if self.dim is not None and self.dim != dimension:
            raise ValueError(f"operator dimension {self.dim} does not match input dimension {dimension}")
        return dimension

    def resolve_terms(self, dimension: int) -> Terms:
        """Expand a raw coordinate readout only when a strategy requires it.

        Directional requests always return a univariate term. Laplacian powers
        check both term count and dimension-times-term work before enumeration.
        """
        dimension = self.check_dimension(dimension)
        if self.mode == "directional":
            return (((self.order,), Fraction(1)),)
        if self.mode == "mixed":
            assert self.alpha is not None
            return ((self.alpha, Fraction(1)),)
        if self.mode == "polynomial":
            return self.terms
        assert self.power is not None
        count = comb(dimension + self.power - 1, self.power)
        if count > self.planning_budget.max_symbol_terms:
            raise ValueError(f"Laplacian expansion needs {count} terms; symbol budget is "
                             f"{self.planning_budget.max_symbol_terms}; use a contraction strategy")
        self.planning_budget.check_graph(0, count * dimension)
        self.budget.check_layout(count, count)
        resolved: list[tuple[Index, Fraction]] = []
        for choices in combinations_with_replacement(range(dimension), self.power):
            counts = Counter(choices)
            index = tuple(2 * counts.get(i, 0) for i in range(dimension))
            coefficient = factorial(self.power) // prod(factorial(n) for n in counts.values())
            resolved.append((index, Fraction(coefficient)))
        return tuple(sorted(resolved, key=lambda item: item[0]))

    def jet_spec(self, dimension: int) -> JetSpec:
        """Allocate the bounded general Taylor fallback, never during construction."""
        if self.mode == "directional":
            self.check_dimension(dimension)
            return JetSpec.directional(self.order, self.budget)
        return JetSpec.mixed((a for a, _ in self.resolve_terms(dimension)), self.budget)

    def jet_readout(self, layout: JetSpec) -> tuple[tuple[int, Fraction], ...]:
        """Weights converting a normalized jet to this raw derivative readout."""
        return tuple((layout.index(alpha), coefficient * prod(factorial(a) for a in alpha))
                     for alpha, coefficient in self.resolve_terms(layout.dim))

    def explain(self, dimension: int | None = None) -> dict[str, Any]:
        if dimension is not None:
            self.check_dimension(dimension)
        return {"mode": self.mode, "order": self.order, "alpha": self.alpha,
                "power": self.power, "dimension": self.dim if dimension is None else dimension,
                "stored_symbol_terms": len(self.terms), "readout": "sum c_alpha D^alpha",
                "jet_allocated": False, "budget": self.budget.to_config(),
                "planning_budget": self.planning_budget.to_config()}

    def to_config(self) -> dict[str, Any]:
        return {"version": 1, "mode": self.mode, "order": self.order,
                "alpha": list(self.alpha) if self.alpha is not None else None, "power": self.power,
                "terms": [[list(a), c.numerator, c.denominator] for a, c in self.terms],
                "budget": self.budget.to_config(), "planning_budget": self.planning_budget.to_config()}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OperatorSpec:
        if config.get("version", 1) != 1:
            raise ValueError("unsupported operator configuration version")
        budget = JetBudget.from_config(config.get("budget", {}))
        planning = OperatorPlanningBudget.from_config(config.get("planning_budget", {}))
        mode = config["mode"]
        if mode == "directional":
            return cls.directional(config["order"], budget, planning_budget=planning)
        if mode == "mixed":
            return cls.mixed(config["alpha"], budget, planning_budget=planning)
        if mode == "laplacian":
            return cls.laplacian(config["power"], budget, planning_budget=planning)
        if mode == "polynomial":
            rows = config["terms"]
            if len(rows) > planning.max_symbol_terms:
                raise ValueError("operator configuration exceeds the symbol term budget")
            terms: dict[Index, Fraction] = {}
            for alpha, numerator, denominator in rows:
                key = tuple(alpha)
                if key in terms:
                    raise ValueError("operator configuration contains duplicate terms")
                if isinstance(numerator, bool) or not isinstance(numerator, int):
                    raise ValueError("operator numerator must be an integer")
                denominator = _integer(denominator, "operator denominator", 1)
                if max(numerator.bit_length(), denominator.bit_length()) > planning.max_coefficient_bits:
                    raise ValueError("operator coefficient exceeds the exact bit budget")
                terms[key] = Fraction(numerator, denominator)
            return cls.polynomial(terms, budget, planning_budget=planning)
        raise ValueError(f"unknown operator mode {mode!r}")


__all__ = ["OperatorPlanningBudget", "OperatorSpec"]
