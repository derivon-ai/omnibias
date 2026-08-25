# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""06-03 G1/G2 packaging homes are recorded on the smoke artifact."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g2_homes_declared_is_recorded() -> None:
    path = REPO / "docs" / "benchmarks" / "theory_homes_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g2 = payload["g2"]
    assert g2["name"] == "g2_homes_declared"
    assert g2["reported"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    if g2["earned"] is True:
        assert g2["passed"] is True
        assert g2["in_ci_all_passed"] is True
        assert int(g2["n_specs"]) > 0
        assert int(g2["n_missing_home"]) == 0
        assert g2["missing_home"] == []
        assert g2["unexpected_packages"] == {}
        assert "g2_homes_declared" in names
        assert payload["honesty"]["g2_in_ci_all_passed"] is True
    else:
        assert g2["passed"] is False
        assert g2["in_ci_all_passed"] is False
        assert "g2_homes_declared" not in names
        assert payload["honesty"]["g2_in_ci_all_passed"] is False
    assert payload["gates"]["all_passed"] is True
    assert payload["honesty"]["new_packages_allowed"] is False
    assert payload["honesty"]["omnibias_arrangement_package"] is False
    assert payload["honesty"]["book_tree"] is False


def test_g1_zero_new_packages_is_recorded() -> None:
    path = REPO / "docs" / "benchmarks" / "theory_homes_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g1 = payload["g1"]
    assert g1["name"] == "g1_zero_new_packages"
    assert g1["reported"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    if g1["earned"] is True:
        assert g1["passed"] is True
        assert g1["in_ci_all_passed"] is True
        assert g1["allowlist"] == []
        assert g1["arrangement_minted"] is False
        assert g1["folded_resurrected"] == []
        assert "g1_zero_new_packages" in names
        assert payload["honesty"]["g1_in_ci_all_passed"] is True
    else:
        assert g1["passed"] is False
        assert g1["in_ci_all_passed"] is False
        assert "g1_zero_new_packages" not in names
        assert payload["honesty"]["g1_in_ci_all_passed"] is False
    assert payload["honesty"]["g5_earned"] is False
