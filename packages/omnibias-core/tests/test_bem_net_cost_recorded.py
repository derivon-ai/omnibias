# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-06 single-layer cost is reported; G2 disc-accuracy stays out of all_passed."""

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
    assert "cost_single_layer_vs_n" not in names
    assert payload["gates"]["all_passed"] is True
