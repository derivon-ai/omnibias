# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-08 G5 anisotropic-interface is leftover-recorded, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_cost_is_reported_and_g5_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "equivariant_scan_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cost = payload["cost"]
    assert cost["name"] == "cost_orbit_vs_L"
    assert cost["earned"] is False
    assert cost["passed"] is False
    assert cost["reported"] is True
    assert cost["leftover_recorded"] is True
    assert int(cost["leftover_id"]) == 43
    assert int(cost["leftover_tick"]) == 87
    assert cost["in_ci_all_passed"] is False
    assert cost["g5_anisotropic_interface"]["earned"] is False
    assert cost["g5_anisotropic_interface"]["reported"] is True
    assert cost["g5_anisotropic_interface"]["leftover_recorded"] is True
    assert int(cost["g5_anisotropic_interface"]["leftover_id"]) == 25
    assert int(cost["g5_anisotropic_interface"]["leftover_tick"]) == 63
    assert cost["g5_anisotropic_interface"]["stays_full"] is True
    ls = {int(row["L"]) for row in cost["rows"]}
    assert {4, 8, 16} <= ls
    assert payload["honesty"]["g5_anisotropic_interface_earned"] is False
    assert payload["honesty"]["g5_leftover_recorded"] is True
    assert int(payload["honesty"]["g5_leftover_id"]) == 25
    assert payload["honesty"]["cost_leftover_recorded"] is True
    assert int(payload["honesty"]["cost_leftover_id"]) == 43
    assert payload["honesty"]["cost_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g5_anisotropic_interface" not in names
    assert "cost_orbit_vs_L" not in names
    assert payload["gates"]["all_passed"] is True
