# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""06-03 G3 falsifiers-first is recorded on the smoke artifact."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_falsifiers_first_is_recorded() -> None:
    path = REPO / "docs" / "benchmarks" / "theory_homes_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_falsifiers_first"
    assert g3["reported"] is True
    assert int(g3["n_units"]) == 4
    names = [row["name"] for row in payload["gates"]["entries"]]
    if g3["earned"] is True:
        assert g3["passed"] is True
        assert g3["in_ci_all_passed"] is True
        assert int(g3["n_recorded"]) == 4
        assert g3["missing_units"] == []
        assert g3["unrecorded"] == []
        assert g3["wave1_section"] is True
        units = [row["unit"] for row in g3["rows"]]
        assert units == ["A4", "A5", "A6", "A7"]
        assert "g3_falsifiers_first" in names
        assert payload["honesty"]["g3_in_ci_all_passed"] is True
    else:
        assert g3["passed"] is False
        assert g3["in_ci_all_passed"] is False
        assert "g3_falsifiers_first" not in names
        assert payload["honesty"]["g3_in_ci_all_passed"] is False
    assert payload["gates"]["all_passed"] is True
    assert payload["honesty"]["g3_is_git_order_proof"] is False
    assert payload["g1"]["earned"] is True
    assert payload["g2"]["earned"] is True
