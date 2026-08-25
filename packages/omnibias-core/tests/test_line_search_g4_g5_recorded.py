# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""03-12 G4/G5 are recorded: G4 unearned vs strong Wolfe; G5 reports crossover."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_g5_are_recorded_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "jet_line_search_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    g5 = payload["g5"]
    assert g4["name"] == "g4_step_count_win"
    assert g4["earned"] is False
    assert g4["passed"] is False
    assert g4["in_ci_all_passed"] is False
    assert g4["baseline"] == "strong_wolfe_cubic_or_quadratic"
    assert float(g4["wolfe_over_jet"]) < float(g4["expected"])
    assert int(g4["n_seeds"]) == 5
    assert g5["name"] == "g5_cost_crossover_table"
    assert g5["earned"] is False
    assert g5["passed"] is False
    assert g5["reported"] is True
    assert g5["in_ci_all_passed"] is False
    assert g5["crossover_order"] is not None
    assert g5["crossover_depth"] is not None
    assert g5["favourable_regime"] is True
    assert g5["unfavourable_regime"] is True
    assert payload["honesty"]["g4_earned"] is False
    assert payload["honesty"]["g5_earned"] is False
    assert payload["honesty"]["g4_baseline_is_strong_wolfe"] is True
    assert payload["honesty"]["g5_compares_jet_to_trial"] is True
    assert payload["config"]["g4_in_all_passed"] is False
    assert payload["config"]["g5_in_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_step_count_win" not in names
    assert "g5_cost_crossover_table" not in names
    assert payload["gates"]["all_passed"] is True
