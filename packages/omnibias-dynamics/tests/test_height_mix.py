# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""C!=0 height mixing; |g_h|=O(nu^2). Not first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.height_mix import (
    enclose_g_h,
    identity_verdicts,
    report,
    residual_ell_mix,
    residual_g_lead,
    residual_V_h_c,
    residual_V_mix,
    residual_V_v_ell,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_height_mix_identities_are_exact() -> None:
    nu, v, c, h = Fraction(1, 16), Fraction(1, 2), Fraction(2), Fraction(1, 4)
    assert residual_ell_mix(nu, v, c, h) == 0
    assert residual_V_mix(nu, v, c, h) == 0
    assert residual_V_v_ell(nu, v, c, h) == 0
    assert residual_V_h_c(nu, v, c) == 0
    assert residual_g_lead(nu, Fraction(1, 3), h) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_height_mix_gh_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-height-mix-v1"
    assert payload["height_mix_gh"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["enclosure"]["ell_positive"] is True
    assert payload["enclosure"]["g_h_o_nu2"] is True
    honesty = payload["honesty"]
    assert honesty["height_mix_gh"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "height_mix_gh" not in reasons


def test_larger_height_mix_box_refuses_ell() -> None:
    wide = enclose_g_h(nu_hi=Fraction(1))
    assert wide["ell_positive"] is False
    assert wide["g_h_o_nu2"] is False


def test_local_height_mix_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_height_mix")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
