# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-14 G3 Magnus bound is reported, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_magnus_bound_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "holonomy_band_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_magnus_bound"
    assert g3["earned"] is False
    assert g3["passed"] is False
    assert g3["reported"] is True
    assert g3["in_ci_all_passed"] is False
    assert g3["magnus_holonomy_api"] is False
    assert g3["stays_full"] is True
    assert g3["zero_in_every_bound"] is True
    assert g3["refuses_outside_radius"] is True
    assert int(g3["violations"]) == 0
    assert int(g3["n_grid"]) >= 6
    assert int(g3["n_sample"]) == 8
    assert payload["honesty"]["g3_earned"] is False
    assert payload["honesty"]["g3_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3_magnus_bound" not in names
    assert payload["gates"]["all_passed"] is True
