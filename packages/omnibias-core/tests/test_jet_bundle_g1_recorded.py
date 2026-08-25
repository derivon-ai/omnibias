# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""01-10 G1/G2 contact tests are recorded on the smoke artifact."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g1_contact_is_recorded() -> None:
    path = REPO / "docs" / "benchmarks" / "jet_bundle_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g1 = payload["g1"]
    assert g1["name"] == "g1_contact"
    assert g1["reported"] is True
    assert int(g1["need"]) == 200
    names = [row["name"] for row in payload["gates"]["entries"]]
    if g1["earned"] is True:
        assert g1["passed"] is True
        assert g1["in_ci_all_passed"] is True
        assert int(g1["n_holonomic_true"]) >= int(g1["need"])
        assert int(g1["n_corrupted_false"]) >= int(g1["need"])
        assert int(g1["n_misclass"]) == 0
        assert "g1_contact" in names
        assert payload["honesty"]["g1_in_ci_all_passed"] is True
    else:
        assert g1["passed"] is False
        assert g1["in_ci_all_passed"] is False
        assert "g1_contact" not in names
        assert payload["honesty"]["g1_in_ci_all_passed"] is False
    assert payload["gates"]["all_passed"] is True
    assert payload["honesty"]["reformulation_not_discovery"] is True
    assert payload["honesty"]["omnibias_jetbundle_package"] is False
    assert payload["honesty"]["g3_earned"] is False


def test_g2_rate_is_recorded() -> None:
    path = REPO / "docs" / "benchmarks" / "jet_bundle_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g2 = payload["g2"]
    assert g2["name"] == "g2_rate"
    assert g2["reported"] is True
    names = [row["name"] for row in payload["gates"]["entries"]]
    if g2["earned"] is True:
        assert g2["passed"] is True
        assert g2["in_ci_all_passed"] is True
        assert "g2_rate" in names
        assert payload["honesty"]["g2_in_ci_all_passed"] is True
        for ratio in g2["holonomic_ratios"]:
            assert 0.15 <= float(ratio) <= 0.35
        for ratio in g2["corrupted_ratios"]:
            assert 0.40 <= float(ratio) <= 0.60
    else:
        assert g2["passed"] is False
        assert g2["in_ci_all_passed"] is False
        assert "g2_rate" not in names
        assert payload["honesty"]["g2_in_ci_all_passed"] is False
