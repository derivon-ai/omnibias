# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-09 algebraic cost is reported; G4 init-win stays out of all_passed."""

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
    assert cost["in_ci_all_passed"] is False
    assert cost["g4_init_win"]["earned"] is False
    assert cost["g4_init_win"]["stays_full"] is True
    names = {row["name"] for row in cost["rows"]}
    assert {"burgers", "kdv", "mkdv"} <= names
    assert all(float(row["published_residual_l1"]) == 0.0 for row in cost["rows"])
    assert payload["honesty"]["g4_init_win_earned"] is False
    assert payload["honesty"]["cost_in_ci_all_passed"] is False
    gate_names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_init_win" not in gate_names
    assert "cost_algebraic_vs_init_win" not in gate_names
    assert payload["gates"]["all_passed"] is True
