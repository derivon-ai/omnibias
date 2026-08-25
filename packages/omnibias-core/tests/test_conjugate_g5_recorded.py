# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""01-12 G5 is recorded unearned on the CCF campaign smoke (not CI)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g5_capacity_is_ccf_and_unearned() -> None:
    path = REPO / "docs" / "benchmarks" / "conjugate_hilbert_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g5 = payload["g5_capacity"]
    assert g5["name"] == "g5_dictionary_capacity"
    assert g5["earned"] is False
    assert g5["passed"] is False
    assert g5["reported"] is True
    assert g5["in_ci_all_passed"] is False
    assert g5["ten_x"] is False
    assert float(g5["matched_width_ratio_vs_n0"]) < 10.0
    assert "ccf_conjugate_sweep_smoke.json" in str(g5["source"])
    assert payload["honesty"]["g5_earned"] is False
    assert payload["honesty"]["g5_reported"] is True
    assert payload["honesty"]["g5_in_all_passed"] is False
    assert payload["honesty"]["g5_in_ci_all_passed"] is False
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g5_dictionary_capacity" not in names
    assert payload["gates"]["all_passed"] is True


def test_sweep_artifact_agrees_with_g5_record() -> None:
    sweep = json.loads(
        (REPO / "docs" / "benchmarks" / "ccf_conjugate_sweep_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    assert sweep["g2_measurement"]["ten_x"] is False
    assert float(sweep["g2_measurement"]["matched_width_ratio_vs_n0"]) < 10.0
    assert int(sweep["best"]["max_order"]) == 0
