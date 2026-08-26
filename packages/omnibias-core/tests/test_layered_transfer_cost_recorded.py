# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-11 G4/G5 leftover-recorded; both stay out of all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_cost_is_reported_and_g4_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "layered_transfer_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cost = payload["cost"]
    assert cost["name"] == "cost_stack_vs_periods"
    assert cost["earned"] is False
    assert cost["passed"] is False
    assert cost["reported"] is True
    assert cost["leftover_recorded"] is True
    assert int(cost["leftover_id"]) == 46
    assert int(cost["leftover_tick"]) == 90
    assert cost["in_ci_all_passed"] is False
    assert cost["g4_inverse_design"]["earned"] is False
    assert cost["g4_inverse_design"]["reported"] is True
    assert cost["g4_inverse_design"]["leftover_recorded"] is True
    assert int(cost["g4_inverse_design"]["leftover_id"]) == 23
    assert int(cost["g4_inverse_design"]["leftover_tick"]) == 67
    assert cost["g4_inverse_design"]["stays_full"] is True
    periods = {int(row["n_periods"]) for row in cost["rows"]}
    assert {1, 2, 4, 8} <= periods
    assert payload["honesty"]["g4_inverse_design_earned"] is False
    assert payload["honesty"]["g4_leftover_recorded"] is True
    assert int(payload["honesty"]["g4_leftover_id"]) == 23
    assert payload["honesty"]["cost_leftover_recorded"] is True
    assert int(payload["honesty"]["cost_leftover_id"]) == 46
    assert payload["honesty"]["cost_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_inverse_design" not in names
    assert "g5_mlp_conservation" not in names
    assert "cost_stack_vs_periods" not in names
    assert payload["gates"]["all_passed"] is True
    g5 = payload["g5"]
    assert g5["name"] == "g5_mlp_conservation"
    assert g5["earned"] is False
    assert g5["passed"] is False
    assert g5["reported"] is True
    assert g5["leftover_recorded"] is True
    assert int(g5["leftover_id"]) == 27
    assert int(g5["leftover_tick"]) == 68
    assert g5["in_ci_all_passed"] is False
    assert g5["unitarity_refuses_lossy"] is True
    assert float(g5["unstructured_energy_violation"]) > float(g5["structural_energy_violation"])
    assert payload["honesty"]["g5_mlp_conservation_reported"] is True
    assert payload["honesty"]["g5_leftover_recorded"] is True
    assert int(payload["honesty"]["g5_leftover_id"]) == 27
    assert payload["honesty"]["g5_in_ci_all_passed"] is False
    g2 = payload["g2"]
    assert g2["name"] == "g2_band_edges"
    assert g2["passed"] is True
    assert g2["in_ci_all_passed"] is True
    assert float(g2["rel_lo"]) <= 1e-10
    assert float(g2["rel_hi"]) <= 1e-10
    g6 = payload["g6"]
    assert g6["name"] == "g6_parity"
    assert g6["passed"] is True
    assert g6["in_ci_all_passed"] is True
    assert float(g6["max_abs"]) == 0.0
    assert payload["honesty"]["g2_earned"] is True
    assert payload["honesty"]["g6_earned"] is True
    assert payload["config"]["gates_in_scope"] == ["g1", "g2", "g3", "g6"]
