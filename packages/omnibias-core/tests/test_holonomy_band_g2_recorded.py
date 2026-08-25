# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-14 G2 closed-form exactness is earned and in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g2_closed_form_is_earned_and_in_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "holonomy_band_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g2 = payload["g2"]
    assert g2["name"] == "g2_closed_form"
    assert g2["earned"] is True
    assert g2["passed"] is True
    assert g2["reported"] is True
    assert g2["in_ci_all_passed"] is True
    assert int(g2["substeps"]) == 4096
    assert float(g2["abelian_abs_err"]) <= float(g2["abs_tol"])
    assert float(g2["su2_abs_err"]) <= float(g2["abs_tol"])
    assert float(g2["abelian_cost_ratio"]) <= float(g2["cost_ratio_max"])
    assert float(g2["su2_cost_ratio"]) <= float(g2["cost_ratio_max"])
    assert payload["honesty"]["g2_earned"] is True
    assert payload["honesty"]["g2_in_ci_all_passed"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g2_closed_form" in names
    gate = next(row for row in payload["gates"]["entries"] if row["name"] == "g2_closed_form")
    assert gate["passed"] is True
    assert gate["in_ci_all_passed"] is True
    assert payload["gates"]["all_passed"] is True
