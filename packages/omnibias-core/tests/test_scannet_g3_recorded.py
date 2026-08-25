# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""02-01 G3 is in CI all_passed: Scan-Net wall/point vs named k-NN."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_g3_cost_is_in_all_passed_against_knn() -> None:
    path = REPO / "docs" / "benchmarks" / "scannet_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g3 = payload["g3"]
    assert g3["name"] == "g3_cost_vs_n"
    assert g3["passed"] is True
    assert g3["in_ci_all_passed"] is True
    assert payload["config"]["g3_in_all_passed"] is True
    assert payload["honesty"]["g3_in_ci_all_passed"] is True
    assert "g3" in payload["config"]["gates_in_scope"]
    names = [row["name"] for row in payload["gates"]["entries"]]
    assert "g3_cost_vs_n" in names
    assert float(g3["scan_ratio_hi_over_lo"]) <= float(g3["scan_ratio_max"])
    assert float(g3["knn_ratio_hi_over_lo"]) >= float(g3["knn_ratio_min"])
    assert g3["n"] == [32, 320, 3200]
