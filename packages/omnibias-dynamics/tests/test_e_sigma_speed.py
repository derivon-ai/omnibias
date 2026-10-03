# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Incoming GRAZING comparison speed; not E_sigma first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_out_speed import cmp_F, phi
from omnibias.dynamics.e_sigma_speed import (
    enclose_phi_zero,
    identity_verdicts,
    report,
    residual_F_zero,
    residual_phi_zero,
    residual_phi_zero_factor,
    residual_T_in,
    t_in_majorant,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_speed_identities_are_exact() -> None:
    eps = Fraction(1, 16)
    assert residual_F_zero(eps) == 0
    assert residual_phi_zero(eps) == 0
    assert residual_phi_zero_factor(eps) == 0
    assert residual_T_in(eps, t_in_majorant(eps)) == 0
    assert t_in_majorant(eps) == Fraction(16384, 17)
    assert cmp_F(Fraction(0), eps) == -4 * eps**3 * (1 + eps)
    assert phi(Fraction(0), eps) == -2 * eps + 4 * eps**3
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert enclose_phi_zero()["strictly_negative"] is True
    assert enclose_phi_zero(eps=Fraction(0))["strictly_negative"] is False
    assert enclose_phi_zero(eps=Fraction(1))["strictly_negative"] is False


def test_e_sigma_speed_bound_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-speed-v1"
    assert payload["e_sigma_speed_bound"] is True
    assert payload["outgoing_first_hit"] is False
    honesty = payload["honesty"]
    assert honesty["e_sigma_speed_bound"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_speed_bound" not in reasons


def test_local_e_sigma_speed_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_speed")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
