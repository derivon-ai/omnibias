# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-06 G2 is earned; G3 leftover-record stays out of all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g2_earned_and_g3_cost_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "bem_net_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cost = payload["cost"]
    assert cost["name"] == "cost_single_layer_vs_n"
    assert cost["earned"] is False
    assert cost["passed"] is False
    assert cost["reported"] is True
    assert cost["in_ci_all_passed"] is False
    ns = {int(row["n_quad"]) for row in cost["rows"]}
    assert {12, 24, 48} <= ns
    assert "g2_disc_accuracy" not in cost
    g2 = payload["g2"]
    assert g2["name"] == "g2_disc_accuracy"
    assert g2["earned"] is True
    assert g2["passed"] is True
    assert g2["reported"] is True
    assert g2["leftover_recorded"] is False
    assert int(g2["leftover_id"]) == 24
    assert int(g2["leftover_tick"]) == 75
    assert g2["in_ci_all_passed"] is True
    assert float(g2["rel_l2"]) <= 1e-8
    assert float(g2["skill_vs_zero"]) > 0.0
    assert payload["honesty"]["g2_disc_accuracy_earned"] is True
    assert payload["honesty"]["g2_leftover_recorded"] is False
    assert int(payload["honesty"]["g2_leftover_id"]) == 24
    assert int(payload["honesty"]["g2_leftover_tick"]) == 75
    assert payload["honesty"]["cost_in_ci_all_passed"] is False
    assert payload["config"]["g2_in_all_passed"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g2_disc_accuracy" in names
    assert "g3_exterior_win" not in names
    assert "cost_single_layer_vs_n" not in names
    assert payload["gates"]["all_passed"] is True
    g3 = payload["g3"]
    assert g3["name"] == "g3_exterior_win"
    assert g3["earned"] is False
    assert g3["passed"] is False
    assert g3["reported"] is True
    assert g3["leftover_recorded"] is True
    assert int(g3["leftover_id"]) == 30
    assert int(g3["leftover_tick"]) == 62
    assert g3["in_ci_all_passed"] is False
    assert g3["volume_pinn"] is False
    assert g3["stays_full"] is True
    assert g3["pack_tree_crossover_m"] is not None
    assert float(g3["pack_tree_hier_over_dense_at_m_hi"]) <= 1.0
    assert g3["pack_tree_far_eval_is_per_source_taylor"] is False
    assert payload["honesty"]["g3_exterior_win_earned"] is False
    assert payload["honesty"]["g3_leftover_recorded"] is True
    assert int(payload["honesty"]["g3_leftover_id"]) == 30
    assert payload["honesty"]["g3_in_ci_all_passed"] is False
