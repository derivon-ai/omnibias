# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-10 G4 many-body is leftover-recorded; G5 stays out of all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_cost_is_reported_and_g4_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "hermite_ladder_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cost = payload["cost"]
    assert cost["name"] == "cost_exact_vs_fd_orbitals"
    assert cost["earned"] is False
    assert cost["passed"] is False
    assert cost["reported"] is True
    assert cost["in_ci_all_passed"] is False
    assert cost["g4_many_body"]["earned"] is False
    assert cost["g4_many_body"]["reported"] is True
    assert cost["g4_many_body"]["leftover_recorded"] is True
    assert int(cost["g4_many_body"]["leftover_id"]) == 21
    assert int(cost["g4_many_body"]["leftover_tick"]) == 65
    assert cost["g4_many_body"]["stays_full"] is True
    assert cost["g4_many_body"]["one_d_no_improvement"] is True
    ns = {int(row["n"]) for row in cost["rows"]}
    assert {2, 4, 8} <= ns
    assert payload["honesty"]["g4_many_body_earned"] is False
    assert payload["honesty"]["g4_leftover_recorded"] is True
    assert int(payload["honesty"]["g4_leftover_id"]) == 21
    assert payload["honesty"]["cost_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_ferminet" not in names
    assert "g5_anharmonic" not in names
    assert "cost_exact_vs_fd_orbitals" not in names
    assert payload["gates"]["all_passed"] is True
    g5 = payload["g5"]
    assert g5["name"] == "g5_anharmonic"
    assert g5["earned"] is False
    assert g5["passed"] is False
    assert g5["reported"] is True
    assert g5["in_ci_all_passed"] is False
    assert g5["lost_to_grid"] is True
    assert float(g5["fd_grid_ground"]) < float(g5["oscillator_rayleigh"])
    assert payload["honesty"]["g5_anharmonic_reported"] is True
    assert payload["honesty"]["g5_in_ci_all_passed"] is False
