# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""03-13 G4 is in CI all_passed: indicator birth vs matched-count fixed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_efficiency_is_in_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "adaptive_refinement_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_efficiency_win"
    assert g4["earned"] is True
    assert g4["passed"] is True
    assert g4["in_ci_all_passed"] is True
    assert g4["family"] == "boundary_layer_ode"
    assert int(g4["n_seeds"]) == 5
    assert int(g4["n_hits"]) == 5
    assert float(g4["fixed_over_adaptive"]) >= float(g4["expected"])
    assert payload["honesty"]["g4_earned"] is True
    assert payload["honesty"]["g4_in_ci_all_passed"] is True
    assert payload["honesty"]["g4_uses_propose_refinement"] is True
    assert payload["honesty"]["g4_hand_placed_oracle"] is False
    assert payload["config"]["g4_in_all_passed"] is True
    assert "g4" in payload["config"]["gates_in_scope"]
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_efficiency_win" in names
    assert payload["gates"]["all_passed"] is True
