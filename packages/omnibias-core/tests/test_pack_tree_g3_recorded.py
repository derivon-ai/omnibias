# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-07 G1–G5 are earned in pack-tree smoke (G3 leftover #13 closed)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_complexity_is_earned_and_in_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "pack_tree_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_complexity"
    assert g3["earned"] is True
    assert g3["passed"] is True
    assert g3["reported"] is True
    assert g3["leftover_recorded"] is False
    assert int(g3["leftover_id"]) == 13
    assert int(g3["leftover_tick"]) == 74
    assert g3["in_ci_all_passed"] is True
    assert g3["crossover_m"] is not None
    assert int(g3["crossover_m"]) <= 3200
    assert float(g3["hier_over_dense_at_m_hi"]) <= 1.0
    assert payload["honesty"]["g3_earned"] is True
    assert payload["honesty"]["g3_leftover_recorded"] is False
    assert int(payload["honesty"]["g3_leftover_id"]) == 13
    assert payload["honesty"]["far_eval_is_per_source_taylor"] is False
    assert payload["config"]["g3_in_all_passed"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3_complexity" in names
    assert payload["gates"]["all_passed"] is True


def test_g2_g4_g5_are_earned_and_in_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "pack_tree_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g2 = payload["g2"]
    assert g2["name"] == "g2_bound_soundness"
    assert g2["passed"] is True
    assert g2["in_ci_all_passed"] is True
    assert int(g2["violations"]) == 0
    assert int(g2["n_checks"]) >= 100
    g4 = payload["g4"]
    assert g4["name"] == "g4_target_accuracy"
    assert g4["passed"] is True
    assert g4["in_ci_all_passed"] is True
    assert int(g4["violations"]) == 0
    assert all(float(row["separation"]) > float(row["radius"]) for row in g4["instances"])
    g5 = payload["g5"]
    assert g5["name"] == "g5_parity"
    assert g5["passed"] is True
    assert g5["in_ci_all_passed"] is True
    assert float(g5["max_abs"]) == 0.0
    assert payload["honesty"]["g2_earned"] is True
    assert payload["honesty"]["g4_earned"] is True
    assert payload["honesty"]["g5_earned"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert names == [
        "g1_eta0_bit_identical",
        "g2_bound_soundness",
        "g3_complexity",
        "g4_target_accuracy",
        "g5_parity",
    ]
    assert payload["config"]["gates_in_scope"] == ["g1", "g2", "g3", "g4", "g5"]
