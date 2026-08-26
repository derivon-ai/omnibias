# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-13 G4 permutability leftover-record stays out of all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_permutability_is_leftover_recorded_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "linearizing_transforms_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_permutability"
    assert g4["earned"] is False
    assert g4["passed"] is False
    assert g4["reported"] is True
    assert g4["leftover_recorded"] is True
    assert int(g4["leftover_id"]) == 52
    assert int(g4["leftover_tick"]) == 94
    assert g4["in_ci_all_passed"] is False
    assert g4["sequential_backlund_api"] is False
    assert g4["stays_full"] is True
    assert payload["honesty"]["g4_permutability_earned"] is False
    assert payload["honesty"]["g4_leftover_recorded"] is True
    assert int(payload["honesty"]["g4_leftover_id"]) == 52
    assert payload["honesty"]["g4_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_permutability" not in names
    assert payload["gates"]["all_passed"] is True
