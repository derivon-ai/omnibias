# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The dispatch-gate record that the planner documentation quotes is internally consistent with its own verdict."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "benchmarks" / "planner" / "dispatch_gate_a100.json"
DOC = ROOT / "docs" / "operator-planner.md"


def _record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _rounds(rec: dict) -> list[tuple[str, dict]]:
    """Round one, and round two when the record has it (PREREGISTRATION.md, Addendum B); each is judged by the same registered gate."""
    out = [("round one", rec)]
    if "round2" in rec:
        out.append(("round two", rec["round2"]))
    return out


def _check_round(label: str, rnd: dict, mp_passed: bool) -> None:
    held, gate = rnd["heldout"], rnd["gate"]
    regret = {k: float(v) for k, v in held["regret_frozen_model"].items()}  # an infinite regret is stored as the string "inf"
    assert all(math.isfinite(regret[k]) for k in ("median", "p90")), label
    assert rnd["verdict"]["median_regret"] is (regret["median"] <= gate["median_regret"]), label
    assert rnd["verdict"]["worst_regret"] is (regret["worst"] <= gate["worst_regret"]), label
    assert rnd["verdict"]["fallback_lower95"] is (held["fallback"]["lower95_ratio"] >= gate["fallback_lower95"]), label
    assert rnd["verdict"]["break_even"] is (held["break_even"]["cached_fraction_within_100"] >= gate["break_even_fraction"]), label
    assert rnd["passed"] is (all(rnd["verdict"].values()) and mp_passed), label


def test_verdict_follows_from_the_numbers_and_the_registered_gate() -> None:
    rec = _record()
    for label, rnd in _rounds(rec):
        _check_round(label, rnd, rec["mp_passed"])


def test_a_calibration_file_ships_only_if_a_round_passed_the_whole_gate() -> None:
    rec = _record()
    shipped = (ROOT / "packages" / "omnibias-core" / "src" / "omnibias" / "core" / "calibration" / "a100.json").exists()
    assert shipped is any(rnd["passed"] for _, rnd in _rounds(rec))


def test_every_group_above_the_worst_gate_is_listed() -> None:
    rec = _record()
    for label, rnd in _rounds(rec):
        over = rnd["heldout"]["regret_over_gate_worst"]
        gate = rnd["gate"]["worst_regret"]
        assert over and all(float(row["regret"]) > gate for row in over), label
        worst = float(rnd["heldout"]["regret_frozen_model"]["worst"])
        assert math.isclose(max(float(row["regret"]) for row in over), worst, abs_tol=1e-3) or math.isinf(worst), label  # rows are rounded


def test_round_two_worst_regret_does_not_depend_on_the_unmeasured_first_choice() -> None:
    """Round two's infinite worst regret is a first choice that was never measured; leaving it out must still fail the gate."""
    rec = _record()
    if "round2" not in rec:
        return
    held = rec["round2"]["heldout"]
    rows = held["regret_over_gate_worst"]
    finite = [float(r["regret"]) for r in rows if math.isfinite(float(r["regret"]))]
    assert math.isclose(max(finite), held["worst_regret_finite_groups_only"], abs_tol=1e-3)
    assert max(finite) > rec["round2"]["gate"]["worst_regret"]


def test_documentation_states_the_outcome_of_the_record() -> None:
    rec, text = _record(), DOC.read_text(encoding="utf-8")
    assert "dispatch_gate_a100.json" in text
    if not rec["passed"]:
        assert "The gate failed" in text and "no A100 calibration file is shipped" in text
    assert f"{float(rec['heldout']['regret_frozen_model']['worst']):.1f}" in text
    if "round2" in rec:
        r2 = rec["round2"]
        assert "round two" in text.lower() and r2["frozen_model_sha256"][:8] in text
        assert f"{r2['heldout']['worst_regret_finite_groups_only']:.1f}" in text
