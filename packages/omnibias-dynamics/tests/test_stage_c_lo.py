# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Kill-line Stage-C C=2 lower T(h) envelope; not first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_lo import (
    identity_verdicts,
    report,
    residual_half_minus,
    residual_one_eps,
    residual_wrap_room,
)


def test_stage_c_lo_identities_are_exact() -> None:
    assert residual_half_minus() == 0
    assert residual_one_eps() == 0
    assert residual_wrap_room() == 0
    assert Fraction(1, 2) - Fraction(1, 32) == Fraction(15, 32)
    assert Fraction(1, 16) - Fraction(17, 512) == Fraction(15, 512)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_lo_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-lo-v1"
    assert payload["stage_c_lo"] is True
    assert payload["sample_inside"] is True
    enclosure = payload["enclosure"]
    assert enclosure["below_declared"] is True
    assert enclosure["start_above"] is True
    assert enclosure["half_above"] is True
    assert enclosure["th_above"] is True
    assert enclosure["excludes_zero"] is True
    assert enclosure["start_lo"] > 0.0625
    assert enclosure["half_lo"] > 0.25
    honesty = payload["honesty"]
    assert honesty["stage_c_lo"] is True
    assert honesty["stage_c_int"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_lo" not in reasons


def test_local_stage_c_lo_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_lo")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
