# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-12 G4 Burgers RH is leftover-recorded, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_burgers_rh_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "equality_intersection_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_burgers_rh"
    assert g4["earned"] is False
    assert g4["passed"] is False
    assert g4["reported"] is True
    assert g4["leftover_recorded"] is True
    assert int(g4["leftover_id"]) == 29
    assert int(g4["leftover_tick"]) == 69
    assert g4["in_ci_all_passed"] is False
    assert g4["rh_match"] is True
    assert float(g4["rh_abs_err"]) <= float(g4["rh_tol"])
    assert int(g4["n_seeds"]) == 5
    assert g4["units_fit_from_data"] is False
    assert g4["stays_full"] is True
    assert payload["honesty"]["g4_earned"] is False
    assert payload["honesty"]["g4_leftover_recorded"] is True
    assert int(payload["honesty"]["g4_leftover_id"]) == 29
    assert payload["honesty"]["g4_in_ci_all_passed"] is False
    assert payload["honesty"]["g4_units_fit_from_data"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_burgers_rh" not in names
    assert payload["gates"]["all_passed"] is True
    g2 = payload["g2"]
    assert g2["name"] == "g2_ift_vs_unrolled"
    assert g2["passed"] is True
    assert g2["in_ci_all_passed"] is True
    assert float(g2["rel"]) <= 1e-8
    assert float(g2["iter_rel"]) <= 1e-12
    assert g2["memory_independent_of_max_iter"] is True
    g3 = payload["g3"]
    assert g3["name"] == "g3_degeneracy_refusal"
    assert g3["passed"] is True
    assert g3["in_ci_all_passed"] is True
    assert g3["converged"] is False
    assert float(g3["condition"]) > 1e6
    g5 = payload["g5"]
    assert g5["name"] == "g5_ansatz_reject"
    assert g5["passed"] is True
    assert g5["in_ci_all_passed"] is True
    assert g5["wrong_raised"] is True
    assert g5["level3_general_solver"] is False
    g6 = payload["g6"]
    assert g6["name"] == "g6_parity"
    assert g6["passed"] is True
    assert g6["in_ci_all_passed"] is True
    assert float(g6["max_abs"]) == 0.0
    assert payload["honesty"]["g2_earned"] is True
    assert payload["honesty"]["g3_earned"] is True
    assert payload["honesty"]["g5_earned"] is True
    assert payload["honesty"]["g6_earned"] is True
    assert payload["config"]["gates_in_scope"] == ["g1", "g2", "g3", "g5", "g6"]
