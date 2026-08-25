# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-04 G4 conditioning is recorded on the smoke artifact."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_conditioning_is_recorded() -> None:
    path = REPO / "docs" / "benchmarks" / "weak_form_vpinn_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_conditioning"
    assert g4["reported"] is True
    assert float(g4["need"]) == 10.0
    assert payload["honesty"]["g4_is_unit_test"] is False
    assert payload["honesty"]["g4_reported"] is True
    assert payload["config"]["g4_is_unit_test"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    if g4["earned"] is True:
        assert g4["passed"] is True
        assert g4["in_ci_all_passed"] is True
        assert float(g4["ratio_strong_over_weak"]) >= float(g4["need"])
        assert "g4_conditioning" in names
        assert payload["honesty"]["g4_in_ci_all_passed"] is True
    else:
        assert g4["passed"] is False
        assert g4["in_ci_all_passed"] is False
        assert "g4_conditioning" not in names
        assert payload["honesty"]["g4_in_ci_all_passed"] is False
    assert payload["gates"]["all_passed"] is True
