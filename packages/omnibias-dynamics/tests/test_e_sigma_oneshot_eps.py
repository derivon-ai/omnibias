# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Finite shrinking-eps one-shot E_sigma pack; not uniform, Z_x, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_oneshot_eps import (
    EPS_PACK,
    horizon_steps,
    identity_verdicts,
    report,
    residual_oeps_h0_25,
    residual_oeps_pack_sum,
    residual_oeps_T20,
    residual_oeps_T25,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_oneshot_eps_identities_are_exact() -> None:
    step = Fraction(1, 4)
    assert residual_oeps_pack_sum(Fraction(1, 16), Fraction(1, 20), Fraction(1, 25)) == 0
    assert residual_oeps_T20(Fraction(400), step) == 0
    assert residual_oeps_T25(Fraction(1000), step) == 0
    assert residual_oeps_h0_25(Fraction(4, 15625)) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert EPS_PACK == (16, 20, 25)
    assert horizon_steps(16) == 280
    assert horizon_steps(20) == 400
    assert horizon_steps(25) == 1000


def test_e_sigma_oneshot_eps_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-oneshot-eps-v1"
    assert payload["e_sigma_oneshot_eps"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"16", "20", "25"}
    assert all(
        row["status"] == "certified" and row["replayed"] and row["transverse_positive"]
        for row in payload["pack"].values()
    )
    assert payload["short"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_oneshot_eps"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_oneshot_eps" not in reasons


def test_local_e_sigma_oneshot_eps_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_oneshot_eps")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
