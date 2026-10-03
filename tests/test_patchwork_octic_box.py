# SPDX-License-Identifier: Apache-2.0
"""The next octic patchwork box stays off the two recorded searches."""

from __future__ import annotations

from benchmarks.patchwork_octic import search_box_identity

_EIGHT_SEED = {
    "seed_ids": list(range(8)),
    "budget_per_seed": 2048,
    "flip_depth": 1,
    "triangulation_limit": 64,
}


def _signature(box: dict[str, object]) -> tuple[object, ...]:
    return (
        tuple(box["seed_ids"]),  # type: ignore[arg-type]
        box["budget_per_seed"],
        box["flip_depth"],
        box["triangulation_limit"],
    )


def test_smoke_box_matches_the_recorded_one_seed_search() -> None:
    smoke = search_box_identity(full=False, budget=None, seeds=None, seed_start=0)
    assert smoke["seed_ids"] == [0]
    assert smoke["budget_per_seed"] == 8
    assert smoke["flip_depth"] == 0
    assert smoke["triangulation_limit"] == 1
    assert smoke["searched_scheme"] == "14 + 1<2 + 1<4>>"


def test_seed8_full_box_is_disjoint_from_both_recorded_searches() -> None:
    box = search_box_identity(full=True, budget=2048, seeds=8, seed_start=8)
    smoke = search_box_identity(full=False, budget=None, seeds=None, seed_start=0)
    assert box["seed_ids"] == list(range(8, 16))
    assert box["budget_per_seed"] == 2048
    assert box["flip_depth"] == 3
    assert box["triangulation_limit"] == 128
    assert box["searched_scheme"] == "14 + 1<2 + 1<4>>"
    assert "1<14>" not in str(box["searched_scheme"])
    assert set(box["seed_ids"]).isdisjoint(smoke["seed_ids"])
    assert set(box["seed_ids"]).isdisjoint(_EIGHT_SEED["seed_ids"])
    assert _signature(box) != _signature(smoke)
    assert _signature(box) != _signature(_EIGHT_SEED)
    assert box["budget_per_seed"] * len(box["seed_ids"]) == 16384
