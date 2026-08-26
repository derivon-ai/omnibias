# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-14 G4 gauge covariance is leftover-recorded, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_gauge_covariance_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "holonomy_band_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_gauge_covariance"
    assert g4["earned"] is False
    assert g4["passed"] is False
    assert g4["reported"] is True
    assert g4["leftover_recorded"] is True
    assert int(g4["leftover_id"]) == 35
    assert int(g4["leftover_tick"]) == 71
    assert g4["in_ci_all_passed"] is False
    assert g4["open_line_flagged"] is True
    assert g4["open_holonomy_not_invariant"] is True
    assert g4["random_gauge_api"] is False
    assert g4["stays_full"] is True
    assert payload["honesty"]["g4_earned"] is False
    assert payload["honesty"]["g4_leftover_recorded"] is True
    assert int(payload["honesty"]["g4_leftover_id"]) == 35
    assert payload["honesty"]["g4_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_open_line_flagged" not in names
    assert "g4_gauge_covariance" not in names
    assert payload["gates"]["all_passed"] is True
