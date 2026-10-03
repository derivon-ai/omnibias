# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""dx_e factors on lambda1 in [-3/2, -2); not lambda1 in (-3/2, 0) or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.dx_e_near import (
    identity_verdicts,
    report,
    residual_near_chi,
    residual_near_u,
    residual_near_wall,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_dx_e_near_identities_are_exact() -> None:
    assert residual_near_wall() == 0
    assert residual_near_u() == 0
    assert residual_near_chi() == 0
    assert Fraction(3, 4) - Fraction(5, 8) == Fraction(1, 8)
    assert Fraction(1) / Fraction(1, 4) == 4
    assert Fraction(11, 5) + 3 == Fraction(26, 5)
    assert Fraction(3, 16) * Fraction(4, 3) * Fraction(1, 16) == Fraction(1, 64)
    assert Fraction(9, 64) / Fraction(1, 8) == Fraction(9, 8)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_dx_e_near_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-dx-e-near-v1"
    assert payload["dx_e_near"] is True
    assert payload["outgoing_first_hit"] is False
    enclosure = payload["enclosure"]
    assert float(enclosure["after_lo"]) > 0.125
    assert float(enclosure["h_hi"]) < float(Fraction(11, 5))
    assert float(enclosure["c_hi"]) < 2.0
    assert float(enclosure["rem_lo"]) < 0.0
    assert float(enclosure["extra_lo"]) > 0.05
    stall = payload["stall"]
    assert stall["finite"] is False
    assert float(stall["after_lo"]) < 0.0
    honesty = payload["honesty"]
    assert honesty["dx_e_near"] is True
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "dx_e_near" not in reasons


def test_local_dx_e_near_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_dx_e_near")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
