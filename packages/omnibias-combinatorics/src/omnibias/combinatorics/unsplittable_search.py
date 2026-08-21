# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Blind H* parameter-box search for a cost-separating unsplittable instance.

The 7-vertex topology and the two Paid/Free routes per terminal are public.
Winning numeric parameters are what search must find. This module does not
import a published cost pair or a named replay constructor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.proof.discovery import ExactCheck, Statement, run_discovery

HSTAR_ARCS: tuple[str, ...] = (
    "S-T1",
    "S-T2",
    "S-U",
    "U-V",
    "U-T3",
    "V-T1",
    "V-W",
    "W-T2",
    "W-T3",
)
HSTAR_PAID: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] = (
    ("S-T1",),
    ("S-T2",),
    ("S-U", "U-T3"),
)
HSTAR_FREE: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] = (
    ("S-U", "U-V", "V-T1"),
    ("S-U", "U-V", "V-W", "W-T2"),
    ("S-U", "U-V", "V-W", "W-T3"),
)


def _frac(value: Fraction | int) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _load(paths: Sequence[Sequence[str]], amounts: Sequence[Fraction]) -> dict[str, Fraction]:
    out: dict[str, Fraction] = {arc: Fraction(0) for arc in HSTAR_ARCS}
    for path, amount in zip(paths, amounts, strict=True):
        for arc in path:
            out[arc] += amount
    return out


def evaluate_hstar(
    demands: tuple[Fraction, Fraction, Fraction],
    paid_costs: tuple[Fraction, Fraction, Fraction],
    split_paid: tuple[Fraction, Fraction, Fraction],
) -> dict[str, Any] | None:
    """Return a separator dict if fractional cost is strictly below every legal unsplittable."""
    if any(p < 0 or p > d for p, d in zip(split_paid, demands, strict=True)):
        return None
    free = tuple(d - p for d, p in zip(demands, split_paid, strict=True))
    costs = {arc: Fraction(0) for arc in HSTAR_ARCS}
    costs["S-T1"], costs["S-T2"], costs["U-T3"] = paid_costs
    x = _load(HSTAR_PAID, split_paid)
    for arc, qty in _load(HSTAR_FREE, free).items():
        x[arc] += qty
    frac_cost = sum(x[arc] * costs[arc] for arc in HSTAR_ARCS)
    d_max = max(demands)
    legal: list[Fraction] = []
    for bits in ((a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)):
        y = {arc: Fraction(0) for arc in HSTAR_ARCS}
        cost = Fraction(0)
        for i, use_paid in enumerate(bits):
            path = HSTAR_PAID[i] if use_paid else HSTAR_FREE[i]
            qty = demands[i]
            for arc in path:
                y[arc] += qty
                cost += qty * costs[arc]
        if all(y[arc] <= x[arc] + d_max for arc in HSTAR_ARCS):
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
        }
    return None


@dataclass(frozen=True)
class DGGSearchHit:
    demands: tuple[Fraction, Fraction, Fraction]
    paid_costs: tuple[Fraction, Fraction, Fraction]
    split_paid: tuple[Fraction, Fraction, Fraction]
    fractional_cost: Fraction
    min_legal: Fraction
    search: str = "hstar_parameter_box"


def matches_known_replay(hit: DGGSearchHit) -> bool:
    """Classify a hit against the published AFP parameter tuple (after the fact)."""
    return (
        hit.demands == (Fraction(15), Fraction(10), Fraction(15))
        and hit.paid_costs == (Fraction(2), Fraction(3), Fraction(2))
        and hit.split_paid == (Fraction(10), Fraction(6), Fraction(10))
    )


class HstarParameterFamily:
    """H* topology, integer parameter box. Incomplete as a parent proof."""

    name = "hstar_parameter_box"
    complete = False
    statement = Statement(
        name="dgg_hstar_box",
        obligation="some H* parameter-box instance with a cost separation",
        parent="goemans_cost_conjecture",
        parent_status="already_false",
    )

    def __init__(
        self,
        instances: Sequence[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]
        | None = None,
    ) -> None:
        if instances is not None:
            self._items = list(instances)
            self.name = "planted"
        else:
            self._items = list(_hstar_box_candidates())
        self._index = {item: i for i, item in enumerate(self._items)}

    def origin(self) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        return self._items[0]

    def neighbors(
        self,
        candidate: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    ) -> Sequence[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]:
        idx = self._index.get(candidate)
        out: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = []
        if idx is not None and idx + 1 < len(self._items):
            out.append(self._items[idx + 1])
        if idx is not None and idx > 0:
            out.append(self._items[idx - 1])
        return out

    def score(
        self,
        candidate: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    ) -> int:
        checked = self.check(candidate)
        return 2 if checked is not None and checked.ok else 0

    def check(
        self,
        candidate: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    ) -> ExactCheck | None:
        demands, costs, split = candidate
        found = evaluate_hstar(
            tuple(_frac(v) for v in demands),  # type: ignore[arg-type]
            tuple(_frac(v) for v in costs),  # type: ignore[arg-type]
            tuple(_frac(v) for v in split),  # type: ignore[arg-type]
        )
        if found is None:
            return ExactCheck(
                ok=False,
                payload={"honesty": {"dgg_congestion_theorem_refuted": False}},
            )
        return ExactCheck(
            ok=True,
            payload={
                "demands": found["demands"],
                "paid_costs": found["paid_costs"],
                "split_paid": found["split_paid"],
                "fractional_cost": found["fractional_cost"],
                "min_legal": found["min_legal"],
                "honesty": {"dgg_congestion_theorem_refuted": False},
            },
        )


def search_dgg(
    *,
    family: str = "hstar_parameter_box",
    instances: Sequence[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] | None = None,
    max_hits: int = 1,
    proposer: str = "score_guided",
    budget: int | None = None,
) -> list[DGGSearchHit]:
    """Propose H* parameter tuples and accept the first exact separator."""
    if family not in {"hstar_parameter_box", "planted"}:
        raise ValueError(f"unknown DGG family {family!r}")
    box = HstarParameterFamily(instances)
    if instances is not None:
        label = "planted"
    else:
        label = "hstar_parameter_box"
    limit = len(box._items) + 2 if budget is None else budget
    result = run_discovery(box.statement, box, proposer, budget=limit)
    if result.status != "PROVED" or result.check is None:
        return []
    payload = result.check.payload
    hits = [
        DGGSearchHit(
            demands=payload["demands"],
            paid_costs=payload["paid_costs"],
            split_paid=payload["split_paid"],
            fractional_cost=payload["fractional_cost"],
            min_legal=payload["min_legal"],
            search=label,
        )
    ]
    return hits[:max_hits]


def _hstar_box_candidates():
    """Small inner box first, then the full CI grid."""
    seen: set[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = set()
    phases = (
        ((8, 10), (1, 2, 3)),
        ((8, 10, 12, 15), (0, 1, 2, 3)),
    )
    for demands_box, cost_box in phases:
        for d1 in demands_box:
            for d2 in demands_box:
                for d3 in demands_box:
                    for c1 in cost_box:
                        for c2 in cost_box:
                            for c3 in cost_box:
                                if c1 == c2 == c3 == 0:
                                    continue
                                for p1 in _split_choices(d1):
                                    for p2 in _split_choices(d2):
                                        for p3 in _split_choices(d3):
                                            item = ((d1, d2, d3), (c1, c2, c3), (p1, p2, p3))
                                            if item in seen:
                                                continue
                                            seen.add(item)
                                            yield item


def _split_choices(demand: int) -> tuple[int, ...]:
    mid = demand // 2
    hi = (2 * demand) // 3
    values = {mid, hi, demand - mid}
    return tuple(sorted(v for v in values if 0 <= v <= demand))


__all__ = [
    "DGGSearchHit",
    "HSTAR_ARCS",
    "HSTAR_FREE",
    "HSTAR_PAID",
    "HstarParameterFamily",
    "evaluate_hstar",
    "matches_known_replay",
    "search_dgg",
]
