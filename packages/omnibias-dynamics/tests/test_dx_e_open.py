# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""dx_e factors on lambda1 in (-3/2, 0); not a uniform C<2 or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.dx_e_open import (
    identity_verdicts,
    report,
    residual_open_chi,
    residual_open_gap,
    residual_open_wall,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_dx_e_open_identities_are_exact() -> None:
    assert residual_open_wall() == 0
    assert residual_open_chi() == 0
    assert residual_open_gap() == 0
    assert Fraction(1) - Fraction(5, 8) * Fraction(8, 5) == 0
    assert Fraction(11, 5) + 5 == Fraction(36, 5)
    assert Fraction(3, 16) - Fraction(1, 20) == Fraction(11, 80)
    assert Fraction(11, 80) > Fraction(1, 8)
    assert Fraction(3, 80) - Fraction(1, 32) == Fraction(1, 160)
    assert Fraction(1, 32) * Fraction(36, 5) == Fraction(9, 40)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_dx_e_open_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-dx-e-open-v1"
    assert payload["dx_e_open"] is True
    assert payload["outgoing_first_hit"] is False
    enclosure = payload["enclosure"]
    assert float(enclosure["after_lo"]) > 0.125
    assert float(enclosure["h_hi"]) < float(Fraction(11, 5))
    assert float(enclosure["factor_hi"]) < 0.2
    assert float(enclosure["eps_hi"]) > 0.0
    assert float(enclosure["rem_lo"]) < 0.0
    stall = payload["stall"]
    assert stall["finite"] is False
    assert float(stall["after_lo"]) < 0.0
    honesty = payload["honesty"]
    assert honesty["dx_e_open"] is True
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "dx_e_open" not in reasons


def test_local_dx_e_open_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_dx_e_open")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
