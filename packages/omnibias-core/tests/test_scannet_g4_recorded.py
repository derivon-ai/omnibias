# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-01 G4 is leftover-recorded: k-NN may win on density; this protocol Scan-Net wins."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_knn_boundary_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "scannet_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_knn_density_boundary"
    assert g4["earned"] is False
    assert g4["passed"] is False
    assert g4["reported"] is True
    assert g4["leftover_recorded"] is True
    assert int(g4["leftover_id"]) == 17
    assert int(g4["leftover_tick"]) == 52
    assert g4["in_ci_all_passed"] is False
    assert int(g4["n_seeds"]) == 5
    assert int(g4["knn_wins"]) < int(g4["n_seeds"])
    assert payload["honesty"]["g4_earned"] is False
    assert payload["honesty"]["g4_leftover_recorded"] is True
    assert int(payload["honesty"]["g4_leftover_id"]) == 17
    assert payload["honesty"]["g4_in_ci_all_passed"] is False
    assert payload["honesty"]["g4_truth_is_knn_oracle"] is False
    assert payload["honesty"]["g4_scan_is_constant_mean"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_knn_density_boundary" not in names
    assert payload["gates"]["all_passed"] is True
