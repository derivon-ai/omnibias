# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-05 G4 hard vs penalized is reported, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_hard_vs_penalized_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "multi_interface_pinn_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_hard_vs_penalized"
    assert g4["earned"] is False
    assert g4["passed"] is False
    assert g4["reported"] is True
    assert g4["in_ci_all_passed"] is False
    assert g4["training_loop"] is False
    assert g4["equal_budget"] is False
    assert g4["stays_full"] is True
    assert payload["honesty"]["g4_earned"] is False
    assert payload["honesty"]["g4_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_hard_vs_penalized" not in names
    assert payload["gates"]["all_passed"] is True
    # G3 leftover stays reported; do not reopen it as a CI gate.
    assert "g3_mixed_vs_baselines" not in names
    assert payload["g3"]["earned"] is False
