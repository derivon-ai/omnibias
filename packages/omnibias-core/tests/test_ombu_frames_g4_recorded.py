# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""01-06 G4 OMBU denoising is recorded, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_denoising_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "ombu_frames_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_denoising"
    assert g4["earned"] is False
    assert g4["passed"] is False
    assert g4["reported"] is True
    assert g4["leftover_recorded"] is True
    assert int(g4["leftover_id"]) == 10
    assert int(g4["leftover_tick"]) == 82
    assert g4["in_ci_all_passed"] is False
    assert int(g4["n_seeds"]) == 5
    assert int(g4["beats_n1_wins"]) == 5
    assert int(g4["skill_positive_wins"]) == 0
    assert float(g4["skill_vs_identity"]) < 0.0
    assert g4["both_gates"] is False
    assert payload["honesty"]["g4_earned"] is False
    assert payload["honesty"]["g4_reported"] is True
    assert payload["honesty"]["g4_leftover_recorded"] is True
    assert int(payload["honesty"]["g4_leftover_id"]) == 10
    assert payload["honesty"]["g4_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3_dilation" in names
    assert "g4_denoising" not in names
    assert payload["gates"]["all_passed"] is True
