# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Finite shrinking-eps aligned E_sigma pack; not uniform, Lohner-from-V=0, or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_eps import (
    EPS_PACK,
    identity_verdicts,
    incoming_steps,
    report,
    residual_sigma_h0_16,
    residual_sigma_h0_20,
    residual_sigma_horizon,
    residual_sigma_pack_sum,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_eps_identities_are_exact() -> None:
    assert residual_sigma_pack_sum(Fraction(1, 16), Fraction(1, 20), Fraction(1, 25)) == 0
    assert residual_sigma_h0_16(Fraction(1, 1024)) == 0
    assert residual_sigma_h0_20(Fraction(1, 2000)) == 0
    assert residual_sigma_horizon(Fraction(80), Fraction(1, 8)) == 0
    assert all(status == "PROVED" for status in identity_verdicts().values())
    assert EPS_PACK == (16, 20, 25)
    assert incoming_steps(16) == 97
    assert incoming_steps(20) == 176
    assert incoming_steps(25) == 328


def test_e_sigma_eps_pack_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-eps-v1"
    assert payload["e_sigma_eps_pack"] is True
    assert payload["outgoing_first_hit"] is False
    assert set(payload["pack"]) == {"16", "20", "25"}
    assert all(row["status"] == "certified" and row["replayed"] for row in payload["pack"].values())
    assert all(row["contains_align"] for row in payload["walls"].values())
    assert payload["short"]["status"] != "certified"
    assert payload["from_grazing_start"]["status"] != "certified"
    honesty = payload["honesty"]
    assert honesty["e_sigma_eps_pack"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_eps_pack" not in reasons


def test_local_e_sigma_eps_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_eps")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
