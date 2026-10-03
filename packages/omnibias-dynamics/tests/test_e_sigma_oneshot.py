# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""One-shot Lohner E_sigma from V=0; not uniform, Z_x, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_oneshot import (
    KILL_L_PACK,
    identity_verdicts,
    report,
    residual_oneshot_h0,
    residual_oneshot_pack_L,
    residual_oneshot_short,
    residual_oneshot_T,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_oneshot_identities_are_exact() -> None:
    step = Fraction(1, 4)
    assert residual_oneshot_h0(Fraction(1, 1024)) == 0
    assert residual_oneshot_T(Fraction(280), step) == 0
    assert residual_oneshot_short(Fraction(200), step) == 0
    assert residual_oneshot_pack_L(Fraction(9, 25), Fraction(1, 16)) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert KILL_L_PACK == (Fraction(9, 25), Fraction(1, 16), Fraction(0))


def test_e_sigma_oneshot_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-oneshot-v1"
    assert payload["e_sigma_oneshot_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"9/25", "1/16", "0"}
    assert all(
        row["status"] == "certified" and row["replayed"] and row["transverse_positive"]
        for row in payload["pack"].values()
    )
    assert payload["short"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_oneshot_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_oneshot_hit" not in reasons


def test_local_e_sigma_oneshot_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_oneshot")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
