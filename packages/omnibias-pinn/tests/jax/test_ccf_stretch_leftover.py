# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""07-03 leftover #55: Hardy N>0 dictionary did not clear stretch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "benchmarks"))
from _gates import CCF_STRETCH_LEFTOVER, CCF_STRETCH_RESIDUAL_GATE  # noqa: E402

SMOKE = REPO / "docs" / "benchmarks" / "reproduce_deepmind_ccf_smoke.json"


def test_stretch_gate_is_still_1e_13() -> None:
    assert CCF_STRETCH_RESIDUAL_GATE == 1e-13
    assert CCF_STRETCH_LEFTOVER["stretch_gate"] == 1e-13
    assert CCF_STRETCH_LEFTOVER["leftover_id"] == 55
    assert CCF_STRETCH_LEFTOVER["leftover_recorded"] is True
    assert CCF_STRETCH_LEFTOVER["stretch_1e-13_cleared"] is False
    assert CCF_STRETCH_LEFTOVER["navier_stokes_proof_claim"] is False


def test_smoke_records_leftover_without_forging_ns() -> None:
    payload = json.loads(SMOKE.read_text(encoding="utf-8"))
    assert payload["gates"]["stretch_1e-13_cleared"] is False
    leftover = payload["honesty"]["leftover"]
    assert leftover["leftover_id"] == 55
    assert leftover["leftover_recorded"] is True
    assert leftover["stretch_gate"] == 1e-13
    assert payload["honesty"]["navier_stokes_proof_claim"] is False
