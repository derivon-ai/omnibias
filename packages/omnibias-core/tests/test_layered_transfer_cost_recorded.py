# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-11 stack cost is reported; G4 inverse-design stays out of all_passed."""

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
    assert cost["in_ci_all_passed"] is False
    assert cost["g4_inverse_design"]["earned"] is False
    assert cost["g4_inverse_design"]["stays_full"] is True
    periods = {int(row["n_periods"]) for row in cost["rows"]}
    assert {1, 2, 4, 8} <= periods
    assert payload["honesty"]["g4_inverse_design_earned"] is False
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
    assert g5["in_ci_all_passed"] is False
    assert g5["unitarity_refuses_lossy"] is True
    assert float(g5["unstructured_energy_violation"]) > float(g5["structural_energy_violation"])
    assert payload["honesty"]["g5_mlp_conservation_reported"] is True
    assert payload["honesty"]["g5_in_ci_all_passed"] is False
