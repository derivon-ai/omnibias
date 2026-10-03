# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Incoming V=1/4 first-hit on the reverse cubic; not E_sigma or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_in import (
    KILL_L_PACK,
    identity_verdicts,
    report,
    residual_hdot_rev,
    residual_v0_grazing,
    residual_vdot_rev_init,
    residual_vin_wall,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.vh_orbit import field_f, field_g


def test_e_sigma_in_identities_are_exact() -> None:
    eps = Fraction(1, 16)
    v0 = Fraction(0)
    h0 = 4 * eps**3
    vdot_rev = -(field_f(Fraction(0), Fraction(-2), eps, v0) + h0 * field_g(eps, v0))
    assert residual_v0_grazing(v0) == 0
    assert residual_hdot_rev(Fraction(1, 8), Fraction(1, 8), Fraction(1)) == 0
    assert residual_vdot_rev_init(vdot_rev, eps) == 0
    assert residual_vin_wall(Fraction(1, 4), Fraction(1, 4)) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert KILL_L_PACK == (Fraction(9, 25), Fraction(1, 16), Fraction(0))


def test_incoming_vwall_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-in-v1"
    assert payload["incoming_vwall_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"9/25", "1/16", "0"}
    assert all(row["status"] == "certified" and row["replayed"] for row in payload["pack"].values())
    assert payload["short"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["incoming_vwall_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "incoming_vwall_hit" not in reasons


def test_local_e_sigma_in_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_in")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
