# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Finite Hilbert XVI chart-cell ledger and one Maletto combinatorial type.

A complete list of labels is not G1, G4, or Hilbert XVI. Verdicts apply to
exact identities only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

CellStatus = str


@dataclass(frozen=True)
class ChartCell:
    name: str
    regime: str
    first_hit: str
    remainder_order: int
    overlaps: tuple[str, ...]
    identity_verdict: str
    g1_item_closed: bool


def _honesty() -> dict[str, bool]:
    return {
        "g1_passed": False,
        "g4_passed": False,
        "full_graphic_cyclicity_proved": False,
        "full_hilbert16_solved": False,
    }


def chart_cells() -> tuple[ChartCell, ...]:
    """Named cells of the current atlas, including recorded kill sequences."""

    return (
        ChartCell("N", "Delta >= chi_N > 0", "selected tube", 2, ("F",), "PROVED", False),
        ChartCell("F", "sep >= sep0, L >= Lmin", "selected tube", 2, ("N", "D"), "PROVED", False),
        ChartCell(
            "D",
            "epsilon |log sep| <= 1, L >= Lmin",
            "selected tube",
            1,
            ("F", "WF"),
            "PROVED",
            False,
        ),
        ChartCell("C", "sep = 0, L >= Lmin", "incomplete", 0, ("WF",), "PROVED", False),
        ChartCell("O", "L -> 0, lambda1 <= -lmin", "incoming only", 0, (), "PROVED", False),
        ChartCell("WF", "W-fold, sigma = sqrt(epsilon)", "selected tube", 0, ("D", "WS", "LI"), "PROVED", False),
        ChartCell("WS", "W-separation, chi = O(1)", "selected tube", 0, ("WF", "D", "WL"), "PROVED", False),
        ChartCell(
            "LI",
            "tau = epsilon log(1/sep)",
            "selected tube",
            0,
            ("WF", "WS", "D"),
            "PROVED",
            False,
        ),
        ChartCell(
            "WL",
            "Lambda = epsilon log(1/W)",
            "selected tube",
            0,
            ("WS", "WF"),
            "PROVED",
            False,
        ),
        ChartCell(
            "kill_super_small_sep",
            "sep = exp(-1/epsilon^2), kappa = 1/sep, L >= Lmin",
            "admitted incoming and chi tube",
            0,
            (),
            "PROVED",
            False,
        ),
        ChartCell(
            "kill_shrinking_root",
            "L = 1/n, lambda1 = -2, r1 -> 0",
            "incoming only",
            0,
            (),
            "PROVED",
            False,
        ),
    )


def g1_from_cells(cells: Sequence[ChartCell] | None = None) -> bool:
    rows = tuple(cells) if cells is not None else chart_cells()
    return all(cell.g1_item_closed for cell in rows)


def ledger_payload() -> dict[str, object]:
    rows = chart_cells()
    return {
        "cells": [
            {
                "name": cell.name,
                "regime": cell.regime,
                "first_hit": cell.first_hit,
                "remainder_order": cell.remainder_order,
                "overlaps": list(cell.overlaps),
                "identity_verdict": cell.identity_verdict,
                "g1_item_closed": cell.g1_item_closed,
            }
            for cell in rows
        ],
        "g1_passed": g1_from_cells(rows),
        "honesty": _honesty(),
    }


def is_dyck_bits(word: Sequence[int]) -> bool:
    height = 0
    for bit in word:
        if bit not in (0, 1):
            return False
        height += 1 if bit == 1 else -1
        if height < 0:
            return False
    return height == 0


def bezout_edge_bound(counts: Sequence[int], degree: int) -> bool:
    return all(type(n) is int and 0 <= n <= degree for n in counts)


def replay_maletto_type(
    counts: Sequence[int],
    words: Sequence[Sequence[int]],
    trees: Sequence[tuple[int, Sequence[int]]],
    *,
    degree: int,
) -> dict[str, object]:
    """Replay one combinatorial curve type. Not a classification or G5 pass."""

    dyck = all(is_dyck_bits(word) for word in words)
    trees_ok = all(is_dyck_bits(shape) for _, shape in trees)
    bezout = bezout_edge_bound(counts, degree)
    ok = dyck and trees_ok and bezout and len(counts) == 6 and len(words) == 4
    return {
        "combinatorial_type_ok": ok,
        "dyck_words": dyck,
        "floating_trees": trees_ok,
        "bezout_edge_bound": bezout,
        "algebraic_smoothness": "BLOCKED",
        "honesty": _honesty(),
        "scope": "One (n, W, T) type. Not 119 cubics, not the octic target, not G1.",
    }


# Published §1.1 example of Maletto, arXiv:2606.21449v1: the quartic
# f = x^4 + x^2 y^2 + 2 x y^3 - y^4 - 2 x^3 z + x y^2 z - y^3 z
#     - 3 x^2 z^2 - 2 x y z^2 + 2 y^2 z^2 + y z^3 + z^4
# together with the NWT triple (n, W, T).
MALETTO_QUARTIC_EXAMPLE: Mapping[str, object] = {
    "degree": 4,
    "source": "Maletto arXiv:2606.21449v1 §1.1",
    "counts": (1, 2, 1, 1, 0, 1),
    "words": ((1, 1, 0, 0), (1, 0), (1, 0, 1, 0), (1, 0)),
    "trees": ((10, (1, 0)),),
    "terms": (
        ((4, 0, 0), 1),
        ((2, 2, 0), 1),
        ((1, 3, 0), 2),
        ((0, 4, 0), -1),
        ((3, 0, 1), -2),
        ((1, 2, 1), 1),
        ((0, 3, 1), -1),
        ((2, 0, 2), -3),
        ((1, 1, 2), -2),
        ((0, 2, 2), 2),
        ((0, 1, 3), 1),
        ((0, 0, 4), 1),
    ),
}


def rematch_shrinking_root(L: Fraction, lambda1: Fraction, *, Lmin: Fraction = Fraction(1, 2)) -> dict[str, object]:
    """Coefficient membership of the shrinking-root sequence in existing charts."""

    lambda0 = -L
    first_root = L >= Lmin and lambda1 < 0
    height_nonneg = lambda1 >= 0
    exponential = L >= Lmin
    grazing_coeff = lambda0 <= 0 and abs(lambda1) <= 2 and L <= 2
    return {
        "L": [L.numerator, L.denominator],
        "lambda1": [lambda1.numerator, lambda1.denominator],
        "lambda0": [lambda0.numerator, lambda0.denominator],
        "incoming_first_hit_retained": True,
        "first_root_Lmin": first_root,
        "height_nonnegative_lambda1": height_nonneg,
        "exponential_Lmin": exponential,
        "grazing_coefficient_box": grazing_coeff,
        "selected_small_label_first_root": first_root,
        "sr2_rematch": False,
        "honesty": _honesty(),
    }
