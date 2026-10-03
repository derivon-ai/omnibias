# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact shrinking-root I-map; not outgoing first-hit or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.shrinking_root_leading import (
    identity_verdicts,
    report,
    residual_c_limit,
    residual_r1_limit_dx,
    residual_rescaled_b,
    residual_two_root_dx,
    residual_xi_limit,
    two_root_dx_dkappa,
)


def test_shrinking_root_imap_identities_are_exact() -> None:
    x, r1, r2 = Fraction(2), Fraction(1, 5), Fraction(4)
    assert residual_two_root_dx(x, r1, r2) == 0
    assert residual_r1_limit_dx(x, r1, r2) == 0
    assert residual_c_limit(r1, r2) == 0
    assert residual_rescaled_b(Fraction(2), r1, r2) == 0
    assert residual_xi_limit(Fraction(2), r1, r2) == 0
    assert two_root_dx_dkappa(x, r1, r2) * x == (x - r1) * (x - r2)
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_report_refuses_outgoing_first_hit_and_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["outgoing_first_hit"] is False
    honesty = payload["honesty"]
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"


def test_local_shrinking_root_imap_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_shrinking_root_imap")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
