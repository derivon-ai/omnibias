# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-05 G3 mixed-condition skill is reported, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_mixed_vs_baselines_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "multi_interface_pinn_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_mixed_vs_baselines"
    assert g3["earned"] is False
    assert g3["passed"] is False
    assert g3["reported"] is True
    assert g3["in_ci_all_passed"] is False
    assert g3["training_loop"] is False
    assert g3["stays_full"] is True
    assert g3["partitioned_field_exported"] is True
    assert g3["fbpinn_field_exported"] is True
    assert payload["honesty"]["g3_earned"] is False
    assert payload["honesty"]["g3_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3_mixed_vs_baselines" not in names
    assert payload["gates"]["all_passed"] is True
