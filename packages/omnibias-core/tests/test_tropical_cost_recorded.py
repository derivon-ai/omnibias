# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""01-08 cost vs n/D is reported, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_cost_vs_n_d_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "tropical_homotopy_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cost = payload["cost"]
    assert cost["name"] == "cost_vs_n_d"
    assert cost["earned"] is False
    assert cost["passed"] is False
    assert cost["reported"] is True
    assert cost["in_ci_all_passed"] is False
    assert int(cost["cutoff_n"]) == 10
    assert int(cost["cutoff_d"]) == 3
    assert cost["refuses_over_cutoff"] is True
    ns = {(int(row["n"]), int(row["dim"])) for row in cost["rows"]}
    assert (4, 2) in ns and (10, 3) in ns
    assert payload["honesty"]["cost_earned"] is False
    assert payload["honesty"]["cost_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "cost_vs_n_d" not in names
    assert payload["gates"]["all_passed"] is True
