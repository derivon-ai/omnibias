# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-07 G3 is recorded unearned: far_eval is not an O(p) multipole."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_complexity_is_unearned_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "pack_tree_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_complexity"
    assert g3["earned"] is False
    assert g3["passed"] is False
    assert g3["in_ci_all_passed"] is False
    assert g3["crossover_m"] is None
    assert float(g3["hier_over_dense_at_m_hi"]) > 1.0
    assert payload["honesty"]["g3_earned"] is False
    assert payload["honesty"]["far_eval_is_per_source_taylor"] is True
    assert payload["config"]["g3_in_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3_complexity" not in names
    assert payload["gates"]["all_passed"] is True
