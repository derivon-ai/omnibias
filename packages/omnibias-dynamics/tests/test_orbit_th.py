# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""C=0 actual-versus-comparison T_h gap; not integrated T-h or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags
from omnibias.dynamics.orbit_th import (
    identity_verdicts,
    report,
    residual_th_split,
    residual_th_touching,
)


def test_orbit_th_identities_are_exact() -> None:
    q, h, kay = Fraction(3, 16), Fraction(1, 4), Fraction(17, 16)
    c, eps, kinetic = Fraction(2), Fraction(1, 16), Fraction(1, 8)
    assert residual_th_split(q, h, kay, c, eps, kinetic) == 0
    assert residual_th_touching(c, eps, kinetic, h, kay) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())


def test_orbit_th_c0_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-orbit-th-v1"
    assert payload["orbit_th_c0"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["k_bound"]["within_three"] is True
    honesty = payload["honesty"]
    assert honesty["orbit_th_c0"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "orbit_th_c0" not in reasons


def test_local_orbit_th_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_orbit_th")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
