# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""01-10 G3 vocabulary coverage is recorded on the smoke artifact."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_vocabulary_is_recorded() -> None:
    path = REPO / "docs" / "benchmarks" / "jet_bundle_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_vocabulary"
    assert g3["reported"] is True
    assert int(g3["n_terms"]) == 8
    names = [row["name"] for row in payload["gates"]["entries"]]
    if g3["earned"] is True:
        assert g3["passed"] is True
        assert g3["in_ci_all_passed"] is True
        assert int(g3["n_covered"]) == int(g3["n_terms"])
        assert g3["missing"] == []
        assert g3["competing"] == []
        assert "g3_vocabulary" in names
        assert payload["honesty"]["g3_in_ci_all_passed"] is True
    else:
        assert g3["passed"] is False
        assert g3["in_ci_all_passed"] is False
        assert "g3_vocabulary" not in names
        assert payload["honesty"]["g3_in_ci_all_passed"] is False
    assert payload["gates"]["all_passed"] is True
    assert payload["honesty"]["reformulation_not_discovery"] is True
    assert payload["g1"]["earned"] is True
    assert payload["g2"]["earned"] is True
