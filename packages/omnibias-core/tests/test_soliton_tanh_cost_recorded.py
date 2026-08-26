# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-09 G4 init-win is leftover-recorded, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_cost_is_reported_and_g4_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "soliton_tanh_method_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cost = payload["cost"]
    assert cost["name"] == "cost_algebraic_vs_init_win"
    assert cost["earned"] is False
    assert cost["passed"] is False
    assert cost["reported"] is True
    assert cost["leftover_recorded"] is True
    assert int(cost["leftover_id"]) == 44
    assert int(cost["leftover_tick"]) == 88
    assert cost["in_ci_all_passed"] is False
    assert cost["g4_init_win"]["earned"] is False
    assert cost["g4_init_win"]["reported"] is True
    assert cost["g4_init_win"]["leftover_recorded"] is True
    assert int(cost["g4_init_win"]["leftover_id"]) == 22
    assert int(cost["g4_init_win"]["leftover_tick"]) == 64
    assert cost["g4_init_win"]["stays_full"] is True
    names = {row["name"] for row in cost["rows"]}
    assert {"burgers", "kdv", "mkdv"} <= names
    assert all(float(row["published_residual_l1"]) == 0.0 for row in cost["rows"])
    assert payload["honesty"]["g4_init_win_earned"] is False
    assert payload["honesty"]["g4_leftover_recorded"] is True
    assert int(payload["honesty"]["g4_leftover_id"]) == 22
    assert payload["honesty"]["cost_leftover_recorded"] is True
    assert int(payload["honesty"]["cost_leftover_id"]) == 44
    assert payload["honesty"]["cost_in_ci_all_passed"] is False
    gate_names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_init_win" not in gate_names
    assert "cost_algebraic_vs_init_win" not in gate_names
    assert payload["gates"]["all_passed"] is True
    g2 = payload["g2"]
    assert g2["name"] == "g2_balance_degree"
    assert g2["passed"] is True
    assert g2["in_ci_all_passed"] is True
    assert int(g2["mismatches"]) == 0
    g3 = payload["g3"]
    assert g3["name"] == "g3_numerical_residual"
    assert g3["passed"] is True
    assert g3["in_ci_all_passed"] is True
    assert int(g3["violations"]) == 0
    assert float(g3["worst_rel"]) <= 1e-14
    assert payload["honesty"]["g2_earned"] is True
    assert payload["honesty"]["g3_earned"] is True
    assert payload["config"]["gates_in_scope"] == ["g1", "g2", "g3", "g5"]
