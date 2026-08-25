# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-06 single-layer cost is reported; G2/G3 stay out of all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_cost_is_reported_and_g2_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "bem_net_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cost = payload["cost"]
    assert cost["name"] == "cost_single_layer_vs_n"
    assert cost["earned"] is False
    assert cost["passed"] is False
    assert cost["reported"] is True
    assert cost["in_ci_all_passed"] is False
    assert cost["g2_disc_accuracy"]["earned"] is False
    assert cost["g2_disc_accuracy"]["stays_full"] is True
    ns = {int(row["n_quad"]) for row in cost["rows"]}
    assert {12, 24, 48} <= ns
    assert payload["honesty"]["g2_disc_accuracy_earned"] is False
    assert payload["honesty"]["cost_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g2_disc_accuracy" not in names
    assert "g3_exterior_win" not in names
    assert "cost_single_layer_vs_n" not in names
    assert payload["gates"]["all_passed"] is True
    g3 = payload["g3"]
    assert g3["name"] == "g3_exterior_win"
    assert g3["earned"] is False
    assert g3["passed"] is False
    assert g3["reported"] is True
    assert g3["in_ci_all_passed"] is False
    assert g3["volume_pinn"] is False
    assert g3["stays_full"] is True
    assert g3["pack_tree_crossover_m"] is None
    assert float(g3["pack_tree_hier_over_dense_at_m_hi"]) > 1.0
    assert payload["honesty"]["g3_exterior_win_earned"] is False
    assert payload["honesty"]["g3_in_ci_all_passed"] is False
