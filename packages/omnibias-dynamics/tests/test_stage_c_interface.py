# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Comparison first-hit from every start in (0, 1/2]; not the height flag or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.stage_c_interface import (
    identity_verdicts,
    report,
    residual_iface_edge,
    residual_iface_phases,
    residual_iface_time,
)


def test_stage_c_interface_identities_are_exact() -> None:
    assert residual_iface_edge() == 0
    assert residual_iface_phases() == 0
    assert residual_iface_time() == 0
    assert Fraction(1) + Fraction(1, 2) * (Fraction(1, 2) - 2) == Fraction(1, 4)
    assert Fraction(60) * Fraction(1, 40) == Fraction(3, 2)
    assert Fraction(5) * Fraction(1, 4) / Fraction(1, 16) ** 2 == Fraction(320)
    assert Fraction(1, 4) > Fraction(1, 5)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_stage_c_interface_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-stage-c-interface-v1"
    assert payload["stage_c_interface"] is True
    assert payload["outgoing_first_hit"] is False
    enclosure = payload["enclosure"]
    assert enclosure["reached"] is True
    assert enclosure["phases"] == 60
    assert float(enclosure["gap_lo"]) > 0.2
    assert float(enclosure["y_neck"]) > 16.0
    stall = payload["stall"]
    assert stall["reached"] is False
    honesty = payload["honesty"]
    assert honesty["stage_c_interface"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "stage_c_interface" not in reasons


def test_local_stage_c_interface_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_stage_c_interface")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
