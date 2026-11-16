# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The hardware record quoted by docs/hardware.md follows its registered shipping rule and the page lists what it records."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "benchmarks" / "hardware" / "hardware_matrix.json"
DOC = ROOT / "docs" / "hardware.md"


def _record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_a_calibration_ships_only_for_a_device_that_meets_the_registered_rule() -> None:
    rec = _record()
    rule = rec["ship_rule"]
    for tag, dev in rec["devices"].items():
        cv, generic = dev["own_cv"], dev["generic"]
        finite_worst = dev["own_cv_unscored"]["worst_finite"]
        meets = cv["median"] <= rule["median"] and cv["worst"] <= rule["worst"] and cv["median"] <= generic["median"]
        assert dev["ship"] is meets, tag
        # the outcome does not depend on how an unmeasured program is scored
        assert dev["ship"] is False and finite_worst > rule["worst"], tag
    assert rec["calibration_files_shipped"] == [t for t, d in rec["devices"].items() if d["ship"]]


def test_unmeasured_devices_are_not_in_the_measured_set_and_are_named_in_the_page() -> None:
    rec, text = _record(), DOC.read_text(encoding="utf-8")
    assert not set(rec["not_measured"]) & set(rec["devices"])
    for name in ("H100 PCIe", "H200", "L40", "V100 SXM2", "V100 PCIe", "TPU"):
        assert name in text
    assert "Unmeasured" in text or "unmeasured" in text


def test_page_quotes_the_probe_numbers_of_the_record() -> None:
    rec, text = _record(), DOC.read_text(encoding="utf-8")
    for dev in rec["devices"].values():
        assert dev["n_problems"] == rec["problems_per_device"]
        assert math.isfinite(dev["probe"]["gemm_f64_flops"])
    a100 = rec["devices"]["a100_80"]["probe"]
    assert f"{a100['gemm_f64_flops'] / 1e12:.1f}" in text and f"{a100['bandwidth_bytes_per_s'] / 1e9:,.0f}" in text
