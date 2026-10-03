# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""03-12 G4 is earned on the named suite; G5 stays leftover-recorded.

G4 stays out of ``gates`` / CI ``all_passed``. The smoke JSON is the
measurement, not a hard-coded pass.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_earned_and_g5_recorded_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "jet_line_search_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    g5 = payload["g5"]
    assert g4["name"] == "g4_step_count_win"
    assert g4["reported"] is True
    assert int(g4["leftover_id"]) == 47
    assert int(g4["leftover_tick"]) == 91
    assert g4["in_ci_all_passed"] is False
    assert g4["baseline"] == "strong_wolfe_cubic_or_quadratic"
    assert float(g4["expected"]) == 2.0
    assert int(g4["n_seeds"]) == 5
    assert int(g4["jet_hits"]) == 5
    assert int(g4["wolfe_hits"]) == 5
    assert float(g4["wolfe_over_jet"]) >= float(g4["expected"])
    assert g4["earned"] is True
    assert g4["passed"] is True
    assert g4["leftover_recorded"] is False
    assert len(g4["rows"]) == 5
    for row in g4["rows"]:
        assert row["jet_hit"] is True
        assert row["wolfe_hit"] is True
    assert "Leftover #47 closed" in g4["note"]
    assert g5["name"] == "g5_cost_crossover_table"
    assert g5["earned"] is False
    assert g5["passed"] is False
    assert g5["reported"] is True
    assert g5["leftover_recorded"] is True
    assert int(g5["leftover_id"]) == 48
    assert int(g5["leftover_tick"]) == 91
    assert g5["in_ci_all_passed"] is False
    assert g5["crossover_order"] is not None
    assert g5["crossover_depth"] is not None
    assert g5["favourable_regime"] is True
    assert g5["unfavourable_regime"] is True
    assert payload["honesty"]["g4_earned"] is True
    assert payload["honesty"]["g4_reported"] is True
    assert payload["honesty"]["g4_leftover_recorded"] is False
    assert int(payload["honesty"]["g4_leftover_id"]) == 47
    assert payload["honesty"]["g4_in_ci_all_passed"] is False
    assert payload["honesty"]["g5_earned"] is False
    assert payload["honesty"]["g5_leftover_recorded"] is True
    assert int(payload["honesty"]["g5_leftover_id"]) == 48
    assert payload["honesty"]["g4_baseline_is_strong_wolfe"] is True
    assert payload["honesty"]["g5_compares_jet_to_trial"] is True
    assert payload["config"]["g4_in_all_passed"] is False
    assert payload["config"]["g5_in_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_step_count_win" not in names
    assert "g5_cost_crossover_table" not in names
    assert payload["gates"]["all_passed"] is True
