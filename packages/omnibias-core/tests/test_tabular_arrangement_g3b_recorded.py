# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""05-02 G3b capacity leftover is recorded, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3b_capacity_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "tabular_arrangement_capacity_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3b = payload["g3b"]
    assert g3b["name"] == "g3b_capacity_boost_h2"
    assert g3b["earned"] is False
    assert g3b["passed"] is False
    assert g3b["reported"] is True
    assert g3b["leftover_recorded"] is True
    assert int(g3b["leftover_id"]) == 49
    assert int(g3b["leftover_tick"]) == 92
    assert g3b["in_ci_all_passed"] is False
    assert g3b["primary_arm"] == "boost_h2"
    assert g3b["g3_frozen"] is True
    assert int(g3b["n_scored"]) == 8
    assert int(g3b["not_worse"]) < int(g3b["need"])
    assert int(g3b["not_worse"]) == 4
    assert int(g3b["need"]) == 6
    assert g3b["win_loss_tie"] == [4, 4, 0]
    assert int(g3b["tab_boost_not_worse"]) == 4
    assert g3b["tab_boost_would_earn_g3b"] is False
    assert g3b["source"] == "docs/benchmarks/tabular_arrangement_capacity.json"
    assert payload["honesty"]["g3b_earned"] is False
    assert payload["honesty"]["g3b_in_ci_all_passed"] is False
    assert payload["honesty"]["g3b_reported"] is True
    assert payload["honesty"]["g3b_leftover_recorded"] is True
    assert int(payload["honesty"]["g3b_leftover_id"]) == 49
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3b_capacity_boost_h2" not in names
    assert payload["gates"]["all_passed"] is True
