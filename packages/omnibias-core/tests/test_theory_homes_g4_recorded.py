# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""06-03 G4 deletion discipline is recorded on the smoke artifact."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g4_deletion_discipline_is_recorded() -> None:
    path = REPO / "docs" / "benchmarks" / "theory_homes_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g4 = payload["g4"]
    assert g4["name"] == "g4_deletion_discipline"
    assert g4["reported"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    if g4["earned"] is True:
        assert g4["passed"] is True
        assert g4["in_ci_all_passed"] is True
        assert int(g4["n_failed"]) == 0
        assert int(g4["n_retired"]) == 0
        assert g4["failed_units"] == []
        assert g4["retired"] == []
        assert g4["unmatched"] == []
        assert g4["empty_reason"] == []
        assert g4["vacuous_no_failed_falsifier"] is True
        assert "g4_deletion_discipline" in names
        assert payload["honesty"]["g4_in_ci_all_passed"] is True
        assert payload["honesty"]["g4_vacuous_no_failed_falsifier"] is True
    else:
        assert g4["passed"] is False
        assert g4["in_ci_all_passed"] is False
        assert "g4_deletion_discipline" not in names
        assert payload["honesty"]["g4_in_ci_all_passed"] is False
    assert payload["gates"]["all_passed"] is True
    assert payload["honesty"]["g4_is_same_commit_proof"] is False
    assert payload["g1"]["earned"] is True
    assert payload["g2"]["earned"] is True
    assert payload["g3"]["earned"] is True
