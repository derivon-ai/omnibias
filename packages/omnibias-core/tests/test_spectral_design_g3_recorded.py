# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""01-07 G3 Mscale leftover is recorded, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_spectral_bias_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "spectral_design_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_spectral_bias_steps"
    assert g3["earned"] is False
    assert g3["passed"] is False
    assert g3["reported"] is True
    assert g3["leftover_recorded"] is True
    assert int(g3["leftover_id"]) == 40
    assert int(g3["leftover_tick"]) == 83
    assert g3["in_ci_all_passed"] is False
    assert int(g3["n_seeds"]) == 5
    assert int(g3["geometric_hits"]) == 0
    assert int(g3["planned_hits"]) == 0
    assert g3["both_reached_gate"] is False
    assert payload["honesty"]["g3_earned"] is False
    assert payload["honesty"]["g3_reported"] is True
    assert payload["honesty"]["g3_leftover_recorded"] is True
    assert int(payload["honesty"]["g3_leftover_id"]) == 40
    assert payload["honesty"]["g3_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3_spectral_bias_steps" not in names
    assert payload["gates"]["all_passed"] is True
