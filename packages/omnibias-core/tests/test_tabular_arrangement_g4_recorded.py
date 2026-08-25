# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""05-02 G4 obliqueness diagnostic is recorded, not in CI all_passed."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_diagnostic_is_reported_and_out_of_all_passed() -> None:
    path = REPO / "docs" / "benchmarks" / "tabular_arrangement_public_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_diagnostic_predictiveness"
    assert g4["earned"] is False
    assert g4["passed"] is False
    assert g4["reported"] is True
    assert g4["in_ci_all_passed"] is False
    assert g4["retuned"] is False
    assert int(g4["n_scored"]) == 8
    assert float(g4["predictiveness"]) < float(g4["need"])
    assert float(g4["predictiveness"]) == 0.25
    assert g4["source"] == "docs/benchmarks/tabular_arrangement_public.json"
    assert payload["honesty"]["g4_earned"] is False
    assert payload["honesty"]["g4_in_ci_all_passed"] is False
    assert payload["honesty"]["g4_reported"] is True
    assert payload["honesty"]["obliqueness_diagnostic_retuned"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g4_diagnostic_predictiveness" not in names
    assert payload["gates"]["all_passed"] is True
