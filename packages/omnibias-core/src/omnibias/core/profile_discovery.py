# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact finite-library PDE relations of independently selectable profile packs.

This infers rational linear identities from structured coefficient metadata,
not from samples of an unknown function. Every returned relation holds for
every admissible profile in every supplied pack, hence for their independent
linear superpositions. It neither chooses a sparse law nor proves that a
candidate library or the supplied families determine a unique physical PDE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core._rational_linear import solve_affine
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.profile_system import ProfilePack, ProfileSystem
from omnibias.core.ridge_system import RidgeBudget, Vector, _integer


@dataclass(frozen=True)
class ProfileRelations:
    """A rational vector-space basis in the supplied candidate ordering.

    ``relations[i][j]`` multiplies candidate j. Rank is the exact rank of the
    stacked coefficient matrix; its row count includes zero/redundant rows.
    An empty basis means no nonzero relation in this finite candidate library.
    """

    relations: tuple[Vector, ...]
    rank: int
    candidate_count: int
    constraint_count: int


class _DiscoveryBudget(RidgeBudget):
    """Accumulate maxima of per-call work meters and reserve resident storage."""

    _spent_work: int
    _phase_work: int
    _reserved_entries: int

    def __init__(self, limits: RidgeBudget):
        super().__init__(**{name: getattr(limits, name) for name in RidgeBudget.__dataclass_fields__})
        object.__setattr__(self, "_spent_work", 0)
        object.__setattr__(self, "_phase_work", 0)
        object.__setattr__(self, "_reserved_entries", 0)

    def phase(self, reserved_entries: int) -> None:
        object.__setattr__(self, "_spent_work", self._spent_work + self._phase_work)
        object.__setattr__(self, "_phase_work", 0)
        object.__setattr__(self, "_reserved_entries", reserved_entries)
        self._entries(0)

    def _entries(self, count: int) -> None:
        super()._entries(self._reserved_entries + count)

    def _work(self, count: int) -> None:
        object.__setattr__(self, "_phase_work", max(self._phase_work, count))
        super()._work(self._spent_work + self._phase_work)


def discover_profile_relations(
    packs: Sequence[ProfilePack], candidates: Sequence[tuple[int, OperatorSpec]], *,
    budget: RidgeBudget | None = None,
) -> ProfileRelations:
    """Find all exact scalar PDE rows in a caller-supplied operator library.

    A candidate ``(component, operator)`` means applying that scalar operator
    to the selected output component. Packs must share input dimension and
    output count; carriers, directions and annihilators may differ. Each pack
    is tested for EVERY profile in its declared annihilator kernel. Profiles
    and mixture weights are independently selectable, so cross-pack
    cancellation for a specially tied mixture is deliberately not inferred.

    The j-th matrix column stacks the canonical remainder coefficients of
    candidate j acting on each pack. Its rational nullspace is the full answer
    within this library, including any identities already present between
    duplicate candidates. No samples, fitted ranks, or numerical tolerances
    enter the calculation. Refusal by ``budget`` is distinct from no relation.

    Work is bounded across all restrictions, residuals and elimination, using
    each public call's cumulative work estimate. Storage conservatively covers
    the resident feature matrix, temporary coefficient arrays, elimination
    copy and returned basis; it counts coefficient slots, not bytes.
    """
    limits = RidgeBudget() if budget is None else budget
    if not isinstance(limits, RidgeBudget):
        raise TypeError("budget must be a RidgeBudget")
    if not packs or not candidates:
        raise ValueError("profile discovery needs nonempty packs and candidates")
    limits._entries(len(packs) + len(candidates))
    if not all(isinstance(pack, ProfilePack) for pack in packs):
        raise TypeError("packs must contain only ProfilePack instances")
    dimension, outputs = len(packs[0].direction), packs[0].outputs
    rows = 0
    for pack in packs:
        if len(pack.direction) != dimension or pack.outputs != outputs:
            raise ValueError("profile packs must share input dimension and output count")
        if pack.profile_order > limits.max_order or pack.profile_order - 1 > limits.max_degree:
            raise ValueError("profile quotient order exceeds the discovery resource budget")
        rows += pack.profile_order
        limits._entries(rows * len(candidates))
    selected: list[tuple[int, OperatorSpec]] = []
    for candidate in candidates:
        if not isinstance(candidate, (tuple, list)) or len(candidate) != 2:
            raise ValueError("each candidate must be a (component, OperatorSpec) pair")
        component = _integer(candidate[0], "candidate component")
        if component >= outputs:
            raise ValueError("candidate component is outside the profile output count")
        operator = candidate[1]
        if not isinstance(operator, OperatorSpec):
            raise TypeError("candidate operators must be OperatorSpec instances")
        if operator.mode == "directional":
            raise ValueError("directional operators are ambiguous in profile discovery")
        operator.check_dimension(dimension)
        if operator.order > limits.max_order:
            raise ValueError("candidate operator order exceeds the discovery resource budget")
        selected.append((component, operator))

    columns = len(selected)
    entries = rows * columns
    meter = _DiscoveryBudget(limits)
    meter._entries(entries)
    meter._work(len(packs) + columns + entries)
    features = [[Fraction(0)] * columns for _ in range(rows)]
    offset = 0
    for pack in packs:
        # Restriction temporarily holds mutable and immutable coefficient
        # matrices; residual convolution also holds several quotient vectors.
        temporary = 4 * pack.profile_order * outputs + 6 * pack.profile_order + 6 * dimension + 4
        for column, (component, operator) in enumerate(selected):
            meter.phase(entries + temporary)
            operators: list[OperatorSpec | None] = [None] * outputs
            operators[component] = operator
            system = ProfileSystem.from_operators(
                (operators,), pack.direction, carrier=pack.carrier,
                annihilator=pack.annihilator, budget=meter,
            )
            meter.phase(entries + temporary)
            residual = system.residual_coefficients(pack)
            for order, values in enumerate(residual):
                features[offset + order][column] = values[0]
            del system, residual
        offset += pack.profile_order
    # Reserve the original and elimination matrices when the helper checks
    # returned basis storage. Its initial matrix check is conservative too.
    meter.phase(2 * entries)
    solution = solve_affine(features, None, meter)
    return ProfileRelations(solution.homogeneous_basis, solution.rank, columns, rows)


__all__ = ["ProfileRelations", "discover_profile_relations"]
