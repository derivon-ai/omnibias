# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Uniform C=2 leading |q| ratio on lambda1=-2; not first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.q_ratio_c2 import (
    identity_verdicts,
    q_ratio_leading,
    report,
    residual_kill_square,
    residual_ratio_disc,
    residual_ratio_gap,
    residual_ratio_outer,
)


def test_q_ratio_c2_identities_are_exact() -> None:
    r1, r2, lam1 = Fraction(1, 5), Fraction(9, 5), Fraction(-2)
    L = r1 * r2
    x_in, x_out = Fraction(1), Fraction(3)
    assert residual_ratio_gap(x_in, r1, r2, lam1, L) == 0
    assert residual_ratio_disc(r1, r2, lam1, L) == 0
    assert residual_kill_square(x_in, L) == 0
    assert residual_ratio_outer(x_out, r1, r2, L) == 0
    assert lam1**2 - 12 * L - 16 < 0
    assert 2 * (x_in - Fraction(1, 2)) ** 2 + Fraction(3, 2) + L > 0
    assert q_ratio_leading(x_in, r1, r2) < 2
    assert q_ratio_leading(x_out, r1, r2) < 2
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_q_ratio_c2_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-q-ratio-c2-v1"
    assert payload["q_ratio_c2"] is True
    assert payload["outgoing_first_hit"] is False
    honesty = payload["honesty"]
    assert honesty["q_ratio_c2"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert reasons["outgoing_first_hit"]["reason"] == "unimplemented"
    assert "q_ratio_c2" not in reasons


def test_local_q_ratio_c2_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_q_ratio_c2")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
