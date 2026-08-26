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
    g3 = payload["g3"]
    assert g3["name"] == "g3_discrete_equivariance"
    assert g3["passed"] is True
    assert g3["in_ci_all_passed"] is True
    assert float(g3["cyclic_ulp"]) <= 4.0
    assert all(float(r) >= 1.6 for r in g3["off_orbit_rates"])
    g4 = payload["g4"]
    assert g4["name"] == "g4_metric_correction"
    assert g4["passed"] is True
    assert g4["in_ci_all_passed"] is True
    assert float(g4["corrected_rel_err"]) <= 0.01
    assert float(g4["uncorrected_rel_err"]) >= 0.4
    assert payload["honesty"]["g3_earned"] is True
    assert payload["honesty"]["g4_earned"] is True
    assert payload["config"]["gates_in_scope"] == ["g1", "g2", "g3", "g4"]
