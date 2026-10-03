# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""dx_e factors for every lambda1 <= -2; not lambda1 in (-2, 0) or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.dx_e_ray import (
    identity_verdicts,
    report,
    residual_ray_chi,
    residual_ray_coeff,
    residual_ray_x,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_dx_e_ray_identities_are_exact() -> None:
    assert residual_ray_x() == 0
    assert residual_ray_coeff() == 0
    assert residual_ray_chi() == 0
    assert Fraction(2) * Fraction(1) == 2
    assert Fraction(3, 8) / 2 == Fraction(3, 16)
    assert Fraction(11, 5) + 2 == Fraction(21, 5)
    assert Fraction(3) / (Fraction(3) - Fraction(1, 2)) == Fraction(6, 5)
    assert Fraction(6, 5) < 2
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_dx_e_ray_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-dx-e-ray-v1"
    assert payload["dx_e_ray"] is True
    assert payload["outgoing_first_hit"] is False
    enclosure = payload["enclosure"]
    assert float(enclosure["after_lo"]) > 0.125
    assert float(enclosure["h_hi"]) < float(Fraction(11, 5))
    assert float(enclosure["c_hi"]) < 2.0
    assert float(enclosure["rem_lo"]) < 0.0
    stall = payload["stall"]
    assert stall["finite"] is False
    assert float(stall["after_lo"]) < 0.0
    honesty = payload["honesty"]
    assert honesty["dx_e_ray"] is True
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "dx_e_ray" not in reasons


def test_local_dx_e_ray_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_dx_e_ray")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
