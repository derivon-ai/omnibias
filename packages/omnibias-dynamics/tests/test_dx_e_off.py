# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""dx_e factors on lambda1 in [-4, -2]; not lambda1 < -4 or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.dx_e_off import (
    identity_verdicts,
    report,
    residual_off_chi,
    residual_off_u,
    residual_off_wall,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_dx_e_off_identities_are_exact() -> None:
    assert residual_off_wall() == 0
    assert residual_off_u() == 0
    assert residual_off_chi() == 0
    assert Fraction(1) - Fraction(5, 8) == Fraction(3, 8)
    assert Fraction(1) / Fraction(1, 2) == 2
    assert Fraction(11, 5) + 2 == Fraction(21, 5)
    assert Fraction(9, 64) / Fraction(3, 8) == Fraction(3, 8)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_dx_e_off_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-dx-e-off-v1"
    assert payload["dx_e_off"] is True
    assert payload["outgoing_first_hit"] is False
    enclosure = payload["enclosure"]
    assert enclosure["below_cap"] is True
    assert float(enclosure["h_hi"]) < float(Fraction(11, 5))
    assert float(enclosure["after_lo"]) > 0.125
    assert float(enclosure["c_hi"]) < 2.0
    assert enclosure["extra_above_floor"] is True
    stall = payload["stall"]
    assert stall["finite"] is False
    honesty = payload["honesty"]
    assert honesty["dx_e_off"] is True
    assert honesty["dx_e_leading"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "dx_e_off" not in reasons


def test_local_dx_e_off_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_dx_e_off")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
