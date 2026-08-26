# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-07 G3 complexity is earned: O(p) multipole crosses dense."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_complexity_is_earned_and_in_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "pack_tree_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_complexity"
    assert g3["earned"] is True
    assert g3["passed"] is True
    assert g3["reported"] is True
    assert g3["leftover_recorded"] is False
    assert int(g3["leftover_id"]) == 13
    assert int(g3["leftover_tick"]) == 74
    assert g3["in_ci_all_passed"] is True
    assert g3["crossover_m"] is not None
    assert int(g3["crossover_m"]) <= 3200
    assert float(g3["hier_over_dense_at_m_hi"]) <= 1.0
    assert payload["honesty"]["g3_earned"] is True
    assert payload["honesty"]["g3_leftover_recorded"] is False
    assert int(payload["honesty"]["g3_leftover_id"]) == 13
    assert payload["honesty"]["far_eval_is_per_source_taylor"] is False
    assert payload["config"]["g3_in_all_passed"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3_complexity" in names
    assert payload["gates"]["all_passed"] is True
