# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-03 G2 is recorded unearned: order-6 jet is not 5x faster than autodiff."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g2_jet_cost_is_unearned_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "jetkan_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g2 = payload["g2"]
    assert g2["name"] == "g2_jet_cost"
    assert g2["earned"] is False
    assert g2["passed"] is False
    assert g2["reported"] is True
    assert g2["leftover_recorded"] is True
    assert int(g2["leftover_id"]) == 41
    assert int(g2["leftover_tick"]) == 85
    assert g2["in_ci_all_passed"] is False
    assert int(g2["order"]) == 6
    assert int(g2["depth"]) == 3
    assert float(g2["autodiff_over_jet"]) < float(g2["expected"])
    assert payload["honesty"]["g2_earned"] is False
    assert payload["honesty"]["g2_reported"] is True
    assert payload["honesty"]["g2_leftover_recorded"] is True
    assert int(payload["honesty"]["g2_leftover_id"]) == 41
    assert payload["honesty"]["g2_in_ci_all_passed"] is False
    assert payload["honesty"]["g2_compares_order6_to_order6"] is True
    assert payload["config"]["g2_in_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g2_jet_cost" not in names
    assert payload["gates"]["all_passed"] is True
