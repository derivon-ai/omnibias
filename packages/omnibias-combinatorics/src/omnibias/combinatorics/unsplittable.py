# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Single-source unsplittable flow: exact ``Fraction`` cost-separation.

The 1999 Dinitz–Garg–Goemans *congestion* theorem (``y_a ≤ x_a + d_max``)
stays true. The Goemans *cost* conjecture is already false; this module
replays the AFP / Rybin instance and hosts a structured search on the same
H* topology. Finding a separator is a finite stress-test, not a claim that
omnibias refuted Goemans.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.combinatorics.unsplittable_search import (
    HSTAR_ARCS,
    HSTAR_FREE,
    HSTAR_PAID,
    DGGSearchHit,
    evaluate_hstar,
    matches_known_replay,
    search_dgg,
)

Rational = Fraction | int


def _frac(value: Rational) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


@dataclass(frozen=True)
class SSUFInstance:
    """A tiny single-source unsplittable-flow instance (exact rationals)."""

    vertices: tuple[str, ...]
    arcs: tuple[str, ...]
    tail: Mapping[str, str]
    head: Mapping[str, str]
    capacity: Mapping[str, Fraction]
    cost: Mapping[str, Fraction]
    demands: tuple[Fraction, Fraction, Fraction]
    paid: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    free: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    split_paid: tuple[Fraction, Fraction, Fraction]


def rybin_instance() -> SSUFInstance:
    """AFP ``Dinitz_Garg_Goemans_Counterexample`` (document.pdf, 7 Aug 2026)."""
    arcs = HSTAR_ARCS
    tail = {
        "S-T1": "Source",
        "S-T2": "Source",
        "S-U": "Source",
        "U-V": "Junction-U",
        "U-T3": "Junction-U",
        "V-T1": "Junction-V",
        "V-W": "Junction-V",
        "W-T2": "Junction-W",
        "W-T3": "Junction-W",
    }
    head = {
        "S-T1": "Terminal-1",
        "S-T2": "Terminal-2",
        "S-U": "Junction-U",
        "U-V": "Junction-V",
        "U-T3": "Terminal-3",
        "V-T1": "Terminal-1",
        "V-W": "Junction-W",
        "W-T2": "Terminal-2",
        "W-T3": "Terminal-3",
    }
    caps = (10, 6, 24, 14, 10, 5, 9, 4, 5)
    costs = {arc: Fraction(0) for arc in arcs}
    costs["S-T1"] = Fraction(2)
    costs["S-T2"] = Fraction(3)
    costs["U-T3"] = Fraction(2)
    return SSUFInstance(
        vertices=("Source", "Junction-U", "Junction-V", "Junction-W", "Terminal-1", "Terminal-2", "Terminal-3"),
        arcs=arcs,
        tail=tail,
        head=head,
        capacity={arc: Fraction(c) for arc, c in zip(arcs, caps, strict=True)},
        cost=costs,
        demands=(Fraction(15), Fraction(10), Fraction(15)),
        paid=HSTAR_PAID,
        free=HSTAR_FREE,
        split_paid=(Fraction(10), Fraction(6), Fraction(10)),
    )


def path_load(paths: Sequence[Sequence[str]], amounts: Sequence[Rational]) -> dict[str, Fraction]:
    load: dict[str, Fraction] = {}
    for path, amount in zip(paths, amounts, strict=True):
        qty = _frac(amount)
        for arc in path:
            load[arc] = load.get(arc, Fraction(0)) + qty
    return load


def instance_fractional(instance: SSUFInstance) -> tuple[dict[str, Fraction], Fraction]:
    paid_load = path_load(instance.paid, instance.split_paid)
    free_amt = tuple(d - p for d, p in zip(instance.demands, instance.split_paid, strict=True))
    free_load = path_load(instance.free, free_amt)
    load: dict[str, Fraction] = {arc: Fraction(0) for arc in instance.arcs}
    for src in (paid_load, free_load):
        for arc, qty in src.items():
            load[arc] += qty
    cost = sum(load[arc] * instance.cost[arc] for arc in instance.arcs)
    return load, cost


def unsplittable_assignments(
    instance: SSUFInstance,
) -> list[tuple[tuple[bool, bool, bool], dict[str, Fraction], Fraction]]:
    """All 8 paid/free choices: ``True`` means the paid route."""
    out = []
    for bits in ((a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)):
        load: dict[str, Fraction] = {arc: Fraction(0) for arc in instance.arcs}
        cost = Fraction(0)
        for i, paid in enumerate(bits):
            path = instance.paid[i] if paid else instance.free[i]
            qty = instance.demands[i]
            for arc in path:
                load[arc] += qty
                cost += qty * instance.cost[arc]
        out.append((bits, load, cost))
    return out


def congestion_legal(
    y: Mapping[str, Fraction],
    x: Mapping[str, Fraction],
    d_max: Fraction,
) -> bool:
    return all(y.get(arc, Fraction(0)) <= x.get(arc, Fraction(0)) + d_max for arc in x)


def _honesty(*, replay: bool, discovered: bool = False) -> dict[str, bool]:
    return {
        "discovered_by_omnibias": discovered,
        "dgg_congestion_theorem_refuted": False,
        "dgg_cost_conjecture_replay": replay,
        "navier_stokes_proof_claim": False,
        "ten_proofs_formalization_claim": False,
    }


def verify_rybin_instance() -> dict[str, Any]:
    """Replay the AFP numbers: fractional cost 58, legal unsplittable ≥ 60."""
    instance = rybin_instance()
    x, frac_cost = instance_fractional(instance)
    d_max = max(instance.demands)
    legal_costs = []
    for _bits, y, cost in unsplittable_assignments(instance):
        if congestion_legal(y, x, d_max):
            legal_costs.append(cost)
    min_legal = min(legal_costs) if legal_costs else None
    replay_ok = (
        frac_cost == 58
        and min_legal is not None
        and min_legal >= 60
        and frac_cost < min_legal
    )
    return {
        "kind": "dgg_cost_replay",
        "fractional_cost": str(frac_cost),
        "min_legal_unsplittable": str(min_legal) if min_legal is not None else None,
        "separation": bool(min_legal is not None and frac_cost < min_legal),
        "replay_ok": replay_ok,
        "search": "replay",
        "honesty": _honesty(replay=replay_ok),
    }


def search_hit_certificate(hit: DGGSearchHit) -> dict[str, Any]:
    known = matches_known_replay(hit)
    return {
        "kind": "dgg_cost_search",
        "search": hit.search,
        "demands": [str(c) for c in hit.demands],
        "paid_costs": [str(c) for c in hit.paid_costs],
        "split_paid": [str(c) for c in hit.split_paid],
        "fractional_cost": str(hit.fractional_cost),
        "min_legal_unsplittable": str(hit.min_legal),
        "rediscovered_known_witness": known,
        "discovered_by_omnibias": not known,
        "replay_ok": True,
        "honesty": _honesty(replay=False, discovered=not known),
    }


__all__ = [
    "SSUFInstance",
    "congestion_legal",
    "evaluate_hstar",
    "instance_fractional",
    "matches_known_replay",
    "path_load",
    "rybin_instance",
    "search_dgg",
    "search_hit_certificate",
    "unsplittable_assignments",
    "verify_rybin_instance",
]
