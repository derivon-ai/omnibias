# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""k=1+O(eps) C=0 jet and cubic remainder prefactor; not first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.k_zeta_remainder import (
    identity_verdicts,
    k_normal,
    report,
    residual_cubic_vs_tworoot,
    residual_k_box_gap,
    residual_k_lead_embed,
    residual_k_normal,
    residual_k_normal_vs_lead,
    residual_k_turn,
    residual_relative_prefactor,
)


def test_k_zeta_remainder_identities_are_exact() -> None:
    nu, v, a, c = Fraction(1, 16), Fraction(1, 2), Fraction(1), Fraction(0)
    r1, r2, x = Fraction(1, 5), Fraction(9, 5), Fraction(1)
    L, lam1 = r1 * r2, -(r1 + r2)
    kay = 1 - a * nu * v + c * nu**2 * v * v
    assert residual_k_turn(kay, a, nu, v, c) == 0
    assert residual_k_normal(a, nu, v) == 0
    assert residual_k_lead_embed(nu, v) == 0
    assert residual_k_normal_vs_lead(nu, v) == 0
    assert residual_k_box_gap(Fraction(2)) == 0
    assert residual_cubic_vs_tworoot(L, lam1, nu, x, r1, r2) == 0
    assert residual_relative_prefactor(nu, nu * x) == 0
    assert abs(k_normal(a, nu, v) - 1) <= 3 * nu
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_k_zeta_remainder_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-k-zeta-remainder-v1"
    assert payload["k_zeta_compact"] is True
    assert payload["outgoing_first_hit"] is False
    honesty = payload["honesty"]
    assert honesty["k_zeta_compact"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["outgoing_first_hit"]["reason"] == "unimplemented"
    assert "k_zeta_compact" not in reasons


def test_local_k_zeta_remainder_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_k_zeta_remainder")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
