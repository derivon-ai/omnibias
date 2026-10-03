# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Uniform-in-r1 comparison first-hit on one eps slab; not every eps or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_compare import (
    identity_verdicts,
    report,
    residual_cmp_emax,
    residual_cmp_phases,
    residual_cmp_xfar,
)


def test_stage_c_compare_identities_are_exact() -> None:
    assert residual_cmp_emax() == 0
    assert residual_cmp_xfar() == 0
    assert residual_cmp_phases() == 0
    assert Fraction(2) - Fraction(1) ** 2 == Fraction(1)
    assert Fraction(1, 4) / Fraction(1, 32) == Fraction(8)
    assert Fraction(310) * Fraction(1, 40) == Fraction(8) - Fraction(1, 4)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_compare_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-compare-v1"
    assert payload["stage_c_compare"] is True
    assert payload["outgoing_first_hit"] is False
    enclosure = payload["enclosure"]
    assert enclosure["reached"] is True
    assert enclosure["phases"] == 310
    assert float(enclosure["gap_lo"]) > 0.125
    assert float(enclosure["time_hi"]) < 250.0
    stall = payload["stall"]
    assert stall["reached"] is False
    honesty = payload["honesty"]
    assert honesty["stage_c_compare"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_compare" not in reasons


def test_local_stage_c_compare_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_compare")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
