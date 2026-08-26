# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-13 G3 Burgers leftover-record stays out of all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_burgers_is_leftover_recorded_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "linearizing_transforms_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_burgers_win"
    assert g3["earned"] is False
    assert g3["passed"] is False
    assert g3["reported"] is True
    assert g3["leftover_recorded"] is True
    assert int(g3["leftover_id"]) == 39
    assert int(g3["leftover_tick"]) == 81
    assert g3["in_ci_all_passed"] is False
    assert g3["training_loop"] is False
    assert g3["stays_full"] is True
    assert payload["honesty"]["g3_burgers_win_earned"] is False
    assert payload["honesty"]["g3_leftover_recorded"] is True
    assert int(payload["honesty"]["g3_leftover_id"]) == 39
    assert payload["honesty"]["g3_in_ci_all_passed"] is False
    assert payload["honesty"]["navier_stokes_proof_claim"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3_burgers_win" not in names
    assert "g3_burgers_init" not in names
    assert "g1_cole_hopf" in names
    assert "g5_negative_control" in names
    assert "g6_parity" in names
    assert payload["gates"]["all_passed"] is True
    assert payload["g1"]["passed"] is True
    assert payload["g5"]["passed"] is True
