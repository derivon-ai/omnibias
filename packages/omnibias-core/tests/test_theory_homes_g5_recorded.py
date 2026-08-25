# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""06-03 G5 promotion criterion is recorded on the smoke artifact."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g5_promotion_criterion_is_recorded() -> None:
    path = REPO / "docs" / "benchmarks" / "theory_homes_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g5 = payload["g5"]
    assert g5["name"] == "g5_promotion_criterion"
    assert g5["reported"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    if g5["earned"] is True:
        assert g5["passed"] is True
        assert g5["in_ci_all_passed"] is True
        assert g5["criterion_written"] is True
        assert g5["arrangement_minted"] is False
        assert int(g5["n_lines"]) < int(g5["line_need"])
        assert g5["size_met"] is False
        assert int(g5["n_external_consumers"]) >= int(g5["consumer_need"])
        assert g5["consumers_met"] is True
        assert "omnibias-graph" in g5["external_consumers"]
        assert "omnibias-convex" in g5["external_consumers"]
        assert g5["promotion_licensed"] is False
        assert g5["vacuous_not_promoted"] is True
        assert "g5_promotion_criterion" in names
        assert payload["honesty"]["g5_in_ci_all_passed"] is True
        assert payload["honesty"]["g5_vacuous_not_promoted"] is True
        assert payload["honesty"]["g5_promotion_licensed"] is False
    else:
        assert g5["passed"] is False
        assert g5["in_ci_all_passed"] is False
        assert "g5_promotion_criterion" not in names
        assert payload["honesty"]["g5_in_ci_all_passed"] is False
    assert payload["gates"]["all_passed"] is True
    assert payload["honesty"]["omnibias_arrangement_package"] is False
    assert payload["honesty"]["book_tree"] is False
    assert payload["g1"]["earned"] is True
    assert payload["g2"]["earned"] is True
    assert payload["g3"]["earned"] is True
    assert payload["g4"]["earned"] is True
